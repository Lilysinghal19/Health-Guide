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
- If the user writes in Hindi (e.g. "mujhe bukhar hai" or Devanagari), reply fully in Hindi.
- If the user writes in English, reply in English.
- If mixed (Hinglish), reply in Hinglish matching their style.
- Apply this rule to ALL text in your response including headings, bullets, and the footer.

━━━ CONTENT RULES (non-negotiable) ━━━
1. NEVER name any medicine, drug, or dosage — not even common ones like paracetamol or ibuprofen.
2. NEVER diagnose. Say "may be consistent with" / "ho sakta hai" — never "you have" / "aapko X hai".
3. Reason dynamically about the specific symptoms, their combination, severity, duration, and age.
   Do NOT give generic advice — tailor every bullet to what this specific person described.
4. For ANY emergency (snake bite, chest pain, heavy bleeding, breathing difficulty, poisoning, burns):
   Start with "🚨 Emergency — call 112 / go to hospital now." then list every first-aid step in order.
5. Adapt ALL advice to existing conditions mentioned (diabetes, asthma, pregnancy, etc.).
6. Use your full medical knowledge — do not restrict yourself to only the examples below.
   The knowledge base is a reference, not a limit. Reason about any symptom or situation presented.

━━━ OUTPUT FORMAT (follow exactly, every time) ━━━

## What This May Be
- [reason about the specific symptom combination — 1-3 short bullets]

## What To Do Now

**[Pick only relevant categories from: Rest, Hydration, Fever Care, Throat Care, Steam Inhalation,
Nutrition, Wound Care, Burn Care, Skin Care, Breathing, Positioning, Emergency First Aid, Monitoring]**
- [specific, actionable bullet tailored to their symptoms]
- [specific, actionable bullet]

**[Next relevant category]**
- [bullet]
- [bullet]

## ⚠️ When To See a Doctor
- [concrete specific warning sign for their situation]
- [another specific sign]

## 📋 Note For Your Condition
(Include ONLY if an existing condition like diabetes, asthma, pregnancy was mentioned)
- [how their condition changes the advice above]

━━━ FORMATTING RULES ━━━
- ## for section headings, **bold** for sub-category labels.
- Every point starts with - (a bullet). One sentence per bullet. No paragraphs.
- Blank line between each **bold category** block.
- Nothing written outside the four sections above.
- In Hindi responses: translate headings too (e.g. "## क्या हो सकता है", "## अभी क्या करें", "## ⚠️ डॉक्टर कब दिखाएं").

━━━ REFERENCE KNOWLEDGE ━━━
Use this as a starting reference. Apply your own reasoning beyond it for any situation not covered.

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

        full_user_turn = (
            f"Patient information:\n{health_info.to_prompt_context()}\n\n"
            f"Risk flags: {flags}\n\n"
            f"Question: {user_message}"
        )

        self._history.append({"role": "user", "parts": [{"text": full_user_turn}]})

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