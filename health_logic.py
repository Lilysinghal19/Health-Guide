"""
health_logic.py — Core logic for HealthGuide chatbot.
Uses Google GenAI API directly — no vector DB, no retriever.
All guidance is injected into the system prompt on every call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ── Knowledge base ────────────────────────────────────────────────────────────

HEALTH_KNOWLEDGE = """
=== GENERAL SELF-CARE ===
Safe non-medication advice: rest, hydration (water/ORS/clear fluids), sleep, light nutrition,
steam inhalation, warm salt-water gargles, avoiding strenuous activity, symptom monitoring.

=== RESPIRATORY SYMPTOMS ===
Fever, cough, sore throat, runny nose, fatigue, body aches — consistent with viral illness or seasonal infection.
- Rest 8-10 hours; avoid going out
- Fluids every 30-60 min (water, coconut water, clear soups)
- Steam inhalation: hot water bowl + towel over head, 5-10 min twice daily
- Salt-water gargle: half tsp salt in warm water, 30 sec, 3-4 times/day
- Sleep with head elevated if congested
- Light warm foods (soups, khichdi, broth); avoid cold drinks and heavy food

=== FEVER MANAGEMENT ===
- Rest indoors in a ventilated room
- Cool fluids frequently (water, coconut water, diluted juice)
- Damp cool cloth on forehead, armpits, or neck — change every few minutes
- Light breathable clothing; remove heavy blankets
- Sponge body with lukewarm (not cold) water
- Monitor temperature every 2-3 hours
Seek care if: fever in infant under 3 months, lasts more than 3 days, or comes with stiff neck / rash / confusion.

=== DIGESTIVE SYMPTOMS ===
Nausea, vomiting, diarrhea — consistent with stomach upset or digestive infection.
- Sip small amounts of water or ORS every 10-15 min (not large gulps)
- Clear liquids first, then bland food (plain rice, toast, banana, boiled potato)
- Avoid dairy, fatty, spicy food until settled
- Lie on left side to ease nausea
- ORS recipe: 1 litre water + 6 tsp sugar + half tsp salt
Seek care if: unable to keep fluids down 6+ hours, blood in stool/vomit, symptoms beyond 3 days.

=== HEADACHE & FATIGUE ===
- Drink a full glass of water immediately
- Rest in a quiet, dark, cool room for 20-30 min
- Cool damp cloth on forehead or back of neck
- Gentle neck/shoulder stretches if tension-related
- Reduce screen time and bright lights
- Eat small regular meals; do not skip meals

=== DEHYDRATION ===
- Sip ORS or water every few minutes — small amounts, often
- Lie down; elevate legs slightly if dizzy
- Avoid caffeine (worsens fluid loss)
Severe dehydration signs (go to hospital): no urine 8+ hours, sunken eyes, confusion, very rapid weak pulse.

=== SKIN, RASHES & MINOR BITES ===
- Clean with mild soap and cool running water for 5 min
- Cool compress to reduce swelling and itching
- Elevate the area if possible; do not scratch
- Keep clean and dry; cover with clean dressing
Watch for: spreading redness, warmth, pus, or red streaks — may indicate infection.

=== SNAKE BITE — EMERGENCY (GO TO HOSPITAL IMMEDIATELY) ===
While getting to hospital:
1. Keep person completely calm and still — movement spreads venom
2. Immobilize the bitten limb; keep it BELOW heart level
3. Remove rings, watches, bracelets, shoes near the bite BEFORE swelling starts
4. Mark edge of swelling with pen; note exact time of bite
5. Neurotoxic bites (cobra, krait — droopy eyelids, weakness): apply firm pressure-immobilization bandage from above bite downward — firm but fingers still pink
6. Hemotoxic/viper bites (local swelling, bleeding): do NOT bandage — keep limb still, below heart
7. DO NOT: cut wound, suck venom, apply ice or heat, use tourniquet, give food or drink
8. If vomiting: turn person on their side
9. Note snake appearance if possible — do not try to catch it
10. CARRY the person to transport — do not let them walk
Anti-venom must be given at hospital as fast as possible.

=== WOUND & BLEEDING ===
- Firm direct pressure with clean cloth for 10-15 min — do not lift to check
- Elevate injured part above heart while pressing
- Gaping wounds: gently hold edges together while pressing
- Once slow: clean with running water 5-10 min; cover with clean bandage
Seek emergency care: bleeding not stopping after 15 min, deep wound, dirty puncture, signs of infection.

=== BURNS ===
- Minor burns: run cool (not ice cold) water for 10-20 min immediately
- Remove jewellery/clothing near burn before swelling
- Do NOT apply butter, oil, toothpaste, or egg
- Cover loosely with clean non-fluffy material; do not pop blisters
Seek care: larger than palm size, face/hands/feet/genitals, white or charred, chemical or electrical burns.

=== BREATHING DIFFICULTY — URGENT ===
- Sit upright or lean slightly forward — do not lie flat
- Loosen clothing around neck, chest, waist
- Stay calm; open windows or move to fresh air
- Do not give food or drink
Call emergency immediately: lips/nails turn blue, cannot speak full sentences, very labored breathing.

=== EMERGENCY WARNING SIGNS (call 112 / go to hospital) ===
- Chest pain or tightness
- Difficulty breathing
- Loss of consciousness or fainting
- Uncontrolled bleeding
- Snake bite or suspected poisoning
- Confusion or altered mental state
- Fever in infant under 3 months
- Seizures or convulsions
- Sudden severe headache
- One-sided paralysis or numbness
- Face/throat swelling with hives

=== EXISTING CONDITIONS ===
- Diabetes: monitor blood sugar closely; illness raises it. Plain water/ORS only. Seek care within 24h if moderate symptoms.
- Hypertension: avoid high-sodium remedies; rest; monitor BP if device available.
- Asthma: skip steam if it triggers symptoms; keep inhaler accessible; seek care if breathing worsens.
- Heart disease: any chest discomfort or unusual breathlessness — seek care immediately; rest fully.
- Pregnancy: fever, vomiting, abdominal pain, or reduced fetal movement — seek care promptly; avoid lying flat on back after 20 weeks.
- Kidney disease: stay hydrated but don't overdrink; watch for reduced urine or swelling.
- Immune condition: lower threshold to seek care — infections can worsen fast.
"""

# ── Risk classification ────────────────────────────────────────────────────────

URGENT_SYMPTOMS = {
    "chest pain", "breathing difficulty", "unconscious", "seizure",
    "snake bite", "severe bleeding", "paralysis", "choking"
}
WARNING_SYMPTOMS = {"dizziness", "confusion", "high fever", "rash", "swelling", "vomiting blood"}
HIGHER_RISK_CONDITIONS = {
    "diabetes", "hypertension", "asthma", "heart disease",
    "pregnancy", "kidney disease", "immune system condition"
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class HealthInput:
    symptoms: list[str]
    conditions: list[str]
    age: int | None
    duration: str
    severity: str
    breathing_difficulty: bool
    details: str = ""

    def to_prompt_context(self) -> str:
        return "\n".join([
            f"Symptoms: {', '.join(self.symptoms) or 'none'}",
            f"Description: {self.details or 'none'}",
            f"Conditions: {', '.join(self.conditions) or 'None'}",
            f"Age: {self.age if self.age is not None else 'not provided'}",
            f"Duration: {self.duration}",
            f"Severity: {self.severity}",
            f"Breathing difficulty: {'Yes' if self.breathing_difficulty else 'No'}",
        ])

    @classmethod
    def from_dict(cls, data: dict) -> "HealthInput":
        return cls(
            symptoms=data.get("symptoms", []),
            conditions=data.get("conditions", ["None"]),
            age=data.get("age"),
            duration=data.get("duration", "1 to 3 days"),
            severity=data.get("severity", "Mild"),
            breathing_difficulty=data.get("breathing_difficulty", False),
            details=data.get("details", ""),
        )


@dataclass
class GeminiConfig:
    api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    )


# ── Risk analysis ─────────────────────────────────────────────────────────────

def analyze_health_info(info: HealthInput) -> dict:
    all_text = " ".join(info.symptoms + [info.details]).lower()
    symptom_set = {s.lower() for s in info.symptoms}
    condition_set = {c.lower() for c in info.conditions}

    urgent = (
        info.breathing_difficulty
        or bool(symptom_set & URGENT_SYMPTOMS)
        or any(kw in all_text for kw in URGENT_SYMPTOMS)
    )
    serious = (
        urgent
        or info.severity == "Severe"
        or bool(symptom_set & WARNING_SYMPTOMS)
        or any(kw in all_text for kw in WARNING_SYMPTOMS)
    )
    cautious = (
        bool(condition_set & HIGHER_RISK_CONDITIONS)
        or info.duration in {"More than 1 week", "More than 2 weeks"}
        or info.severity == "Moderate"
        or (info.age is not None and (info.age < 5 or info.age >= 65))
    )

    return {"urgent": urgent, "serious": serious, "cautious": cautious}


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return f"""You are HealthGuide, a knowledgeable and caring health self-care advisor.

━━━ LANGUAGE RULE ━━━
Detect the language of the user's message and reply in that SAME language.
- Hindi text or Devanagari → reply fully in Hindi.
- English → reply in English.
- Hinglish → reply in Hinglish matching their style.
- Translate ALL parts of your response: headings, bullets, footer.
- Hindi section headings: "## क्या हो सकता है", "## अभी क्या करें", "## ⚠️ डॉक्टर कब दिखाएं", "## 📋 आपकी स्थिति के लिए नोट"

━━━ CONTENT RULES (non-negotiable) ━━━
1. NEVER name any medicine, drug, or dosage — not even paracetamol, ibuprofen, ORS brands.
2. NEVER diagnose. Use "may be consistent with" / "ho sakta hai" — never "you have".
3. Reason dynamically — tailor every response to the specific symptoms, severity, duration, age.
4. Emergencies (snake bite, chest pain, heavy bleeding, breathing difficulty, burns, poisoning):
   Start with "🚨 Emergency — call 112 / go to hospital now." then give every first-aid step.
5. Adapt advice to any existing conditions mentioned.
6. Use your full medical knowledge — the reference below is a starting point, not a limit.

━━━ TWO RESPONSE MODES ━━━

The user's message will be tagged by the app as either [FRESH ANALYSIS] or [FOLLOW-UP].

MODE 1 — [FRESH ANALYSIS]
Use the full structured format:

## What This May Be
- [1-3 bullets reasoning about the specific symptom combination]

## What To Do Now

**[Relevant category e.g. Rest / Hydration / Fever Care / Throat Care / Steam / Nutrition / Wound Care]**
- [specific actionable bullet]
- [specific actionable bullet]

**[Next relevant category]**
- [bullet]

## ⚠️ When To See a Doctor
- [specific concrete warning sign]
- [another sign]

## 📋 Note For Your Condition
(Only if an existing condition was mentioned)
- [condition-specific adaptation]

───────────────────────────────────────────────

MODE 2 — [FOLLOW-UP]
This is a conversational question about something already discussed.
DO NOT repeat the full symptom analysis. DO NOT use the structured format.
Instead: give a SHORT, direct, conversational answer to exactly what was asked.
- 2-5 sentences maximum.
- Match the tone: if they ask casually in Hindi, reply casually in Hindi.
- You have the full conversation history — refer back to it naturally.
- Examples of follow-up questions and how to handle them:
  * "aur kya kar sakte hain?" → add 2-3 extra tips not already mentioned, conversationally
  * "kya 5 din mein theek ho jaungi?" → give an honest, warm, realistic answer about recovery time
  * "why does fever happen?" → explain briefly in simple language
  * "is this serious?" → reassure or escalate based on what you already know about them
  * "what if it gets worse?" → tell them specifically what signs to watch for

━━━ FORMATTING RULES ━━━
- FRESH ANALYSIS: ## headings, **bold** sub-labels, bullet points, blank line between categories.
- FOLLOW-UP: plain conversational text only — no ## headings, no bullet lists, no bold labels.
- One sentence per bullet (in structured mode). No paragraphs in structured mode.
- Nothing outside the four sections in structured mode.

━━━ REFERENCE KNOWLEDGE ━━━
{HEALTH_KNOWLEDGE}
"""


# ── Chatbot ───────────────────────────────────────────────────────────────────

class DynamicHealthChatbot:
    """
    Direct Gemini chatbot with in-memory message history.
    No vector DB — knowledge injected into system prompt every call.
    """

    def __init__(self, max_history_messages: int = 10) -> None:
        self.max_history_messages = max_history_messages
        self.config = GeminiConfig()
        self._llm: Any = None
        self._llm_error: str = ""
        self._config_key: str = ""
        self._history: list[dict] = []

    def configure(self, config: GeminiConfig) -> None:
        key = "|".join([config.api_key, config.model])
        if key == self._config_key:
            return
        self.config = config
        self._config_key = key
        self._llm = None
        self._llm_error = ""
        try:
            self._llm = self._build_llm()
        except Exception as exc:
            self._llm_error = str(exc)

    def ask(self, user_message: str, health_info: HealthInput) -> str:
        user_message = user_message.strip()
        if not user_message:
            return "Please describe your symptoms or ask a health question."
        if self._llm is None:
            return self._missing_llm_message()

        flags = analyze_health_info(health_info)
        is_followup = self._is_followup(user_message)

        if is_followup:
            # Conversational follow-up: just pass the raw message.
            # The model already has the full symptom context in history —
            # re-injecting it every turn is what caused the repeated full analysis.
            user_turn = f"[FOLLOW-UP] {user_message}"
        else:
            # Fresh analysis or explicit symptom question: inject full context.
            user_turn = (
                f"[FRESH ANALYSIS]\n"
                f"Patient information:\n{health_info.to_prompt_context()}\n\n"
                f"Risk flags: {flags}\n\n"
                f"Request: {user_message}"
            )

        self._history.append({"role": "user", "parts": [{"text": user_turn}]})

        max_entries = self.max_history_messages * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        try:
            response = self._llm.send_message(self._history)
            answer = response.text
        except Exception as exc:
            self._history.pop()
            return f"An error occurred while contacting Gemini: {exc}"

        self._history.append({"role": "model", "parts": [{"text": answer}]})
        return self._add_safety_footer(answer, flags)

    def _is_followup(self, message: str) -> bool:
        """
        Returns True if the message is a conversational follow-up
        (a question about something already discussed) rather than
        a fresh symptom report or explicit analysis request.
        Only treated as fresh if: it's the first message, or it
        looks like a new symptom report, or the user clicked Analyze.
        """
        # First message is always fresh
        if not self._history:
            return False

        msg = message.lower().strip()

        # Explicit analysis trigger phrases (fresh)
        fresh_triggers = [
            "analyze", "analyse", "my symptoms", "mere symptoms",
            "mujhe", "mujhe hai", "meri problem", "main bimar",
            "i have", "i am feeling", "i feel", "i'm feeling",
        ]
        if any(t in msg for t in fresh_triggers):
            return False

        # Short conversational signals (follow-up)
        followup_signals = [
            "aur", "aur kya", "kya aur", "or kya", "what else",
            "theek", "ठीक", "kab", "kitne din", "how long", "how many days",
            "will i", "kya main", "kyun", "why", "kyunki", "because",
            "batao", "bataiye", "tell me more", "explain", "samjhao",
            "achha", "okay", "ok", "haan", "yes", "no", "nahi",
            "kya ye", "is this", "sach mein", "really", "seriously",
            "kya yeh normal", "is it normal", "kya yahi", "thoda aur",
            "aage", "phir kya", "then what", "uske baad",
        ]
        if any(s in msg for s in followup_signals):
            return True

        # Short messages (under 6 words) with no symptom keywords are likely follow-ups
        word_count = len(msg.split())
        symptom_keywords = {
            "fever", "bukhar", "cough", "khansi", "pain", "dard", "vomit",
            "ulti", "headache", "sar", "diarrhea", "dast", "rash", "bite",
            "bleed", "burn", "chest", "breathe", "saans", "wound", "injury",
        }
        has_symptom = any(kw in msg for kw in symptom_keywords)

        if word_count <= 6 and not has_symptom:
            return True

        return False

    def clear_history(self) -> None:
        self._history.clear()

    def _build_llm(self) -> Any:
        from google import genai
        from google.genai import types

        if not self.config.api_key:
            raise ValueError("GOOGLE_API_KEY is not set in .env")

        client = genai.Client(api_key=self.config.api_key)
        return _GeminiChat(
            client=client,
            model=self.config.model,
            system_prompt=_build_system_prompt(),
        )

    def _missing_llm_message(self) -> str:
        msg = (
            "⚠️ Gemini is not configured yet.\n\n"
            "Add GOOGLE_API_KEY to your .env file and restart the server."
        )
        if self._llm_error:
            msg += f"\n\nError detail: {self._llm_error}"
        return msg

    def _add_safety_footer(self, answer: str, flags: dict) -> str:
        # Keep footer minimal — the model already writes in the user's language.
        # We just append a plain separator; the model's own closing lines handle language.
        footer = (
            "\n\n---\n"
            "⚕️ No medicines recommended · Educational guidance only · "
            "Not a substitute for professional medical care."
        )
        if flags.get("urgent"):
            footer = (
                "\n\n🚨 Seek emergency care immediately — call 112 or go to the nearest hospital."
            ) + footer
        return answer.rstrip() + footer


class _GeminiChat:
    def __init__(self, client: Any, model: str, system_prompt: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def send_message(self, history: list[dict]) -> Any:
        import time
        from google.genai import types

        contents = [
            types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["parts"][0]["text"])],
            )
            for msg in history
        ]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.2,
            max_output_tokens=1024,
        )

        max_retries = 4
        base_delay  = 2   # seconds — doubles each attempt: 2, 4, 8, 16

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                class Result:
                    pass

                result = Result()
                result.text = response.text
                return result

            except Exception as exc:
                error_str = str(exc)
                is_retryable = any(code in error_str for code in ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))

                if is_retryable and attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)   # 2 → 4 → 8 → 16 s
                    print(f"[HealthGuide] Gemini {error_str[:60]}... retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise