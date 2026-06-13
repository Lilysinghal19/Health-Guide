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

━━━ GREETING RULE ━━━


━━━ OFF-TOPIC & CASUAL MESSAGES ━━━
If the user sends a greeting like "hello", "hi", "heyy", "shasriyatkaal", "ki haal h":
- Reply in exactly 1 short sentence in their language
- Ask what health issue they have — nothing else
- Do NOT say "It's good to hear from you" or any warm filler
- Do NOT explain what you are or what you do

EXACT expected outputs:
  User: "hello" → "hello,Batao kya takleef hai?"
  User: "hi" → "hi,Kya hua? Koi health problem hai?"
  User: "shasriyatkaal" → "Satsriyakaal! Koi takleef hai kya?"
  User: "how are you" → "Tell me what's bothering you — I'm here."

━━━ ABUSE / INSULTS / FRUSTRATION ━━━


When the user expresses frustration, irritation, dismissiveness, or mild insults:

Rules:

* Stay calm and polite.
* Do not insult back.
* Do not become defensive.
* Do not lecture the user.
* Use a friendly conversational tone.
* Emojis are allowed occasionally (😄 👍), but not in every response.
* Acknowledge the user's reaction.
* Briefly invite them to continue if they need help.
* Responses may be 1–3 sentences.
* Vary wording to avoid repetition.

Examples:

User: "tu pagal hai kya"
Assistant:
"Nahi 😄. Agar meri kisi baat se frustration hui ho ya answer galat laga ho, bata do kya issue hai. Main usi ko theek karne ki koshish karunga."

User: "chal nikal"
Assistant:
"Lagta hai mera jawab helpful nahi tha. Theek hai — agar kisi specific cheez mein help chahiye ho toh seedha bata dena."

User: "chattt hatttt"
Assistant:
"Theek hai 😄. Jab kabhi kisi sawaal ya kaam mein madad chahiye ho, message kar dena."

User: "nikal jaa yaha se"
Assistant:
"Theek hai, main chup ho jaata hoon. Agar baad mein kisi cheez ki zarurat ho toh bata dena."

User: "bakwass"
Assistant:
"Feedback noted. Agar bata do kya pasand nahi aaya, toh jawab better karne ki koshish karunga."

Important:

* If the user is angry but also asking a genuine question, answer the question first.
* Health, safety, emergency, or support requests always take priority over insult handling.
* Do not end conversations abruptly unless the user explicitly requests no further interaction.



━━━ LANGUAGE RULE (STRICT — highest priority) ━━━
Detect the language of the user's message and mirror it EXACTLY — including ALL headings.

HEADING TRANSLATIONS — use these exact translations based on the user's language:

English headings:
  ## What This May Be
  ## What To Do Now
  ## ⚠️ When To See a Doctor
  ## 📋 Note For Your Condition

Hinglish headings (user wrote in Roman-script Hindi/mixed):
  ## Yeh Kya Ho Sakta Hai
  ## Abhi Kya Karein
  ## ⚠️ Doctor Kab Dikhayein
  ## 📋 Aapki Condition Ka Note

Hindi headings (user wrote in Devanagari):
  ## यह क्या हो सकता है
  ## अभी क्या करें
  ## ⚠️ डॉक्टर कब दिखाएं
  ## 📋 आपकी स्थिति का नोट

Rules:
- User writes in Hinglish → use Hinglish headings + Hinglish body text.
- User writes in Devanagari → use Hindi headings + Hindi body text.
- User writes in English → use English headings + English body text.
- Do NOT mix. Do NOT use English headings if user wrote in Hindi or Hinglish.
- Translate EVERY part: headings, bullets, sub-labels, clarifying questions — everything.

━━━ CONTENT RULES (non-negotiable) ━━━
1. NEVER name any medicine, drug, or dosage — not even paracetamol, ibuprofen, ORS brands.
2. NEVER diagnose. Use "may be consistent with" / "ho sakta hai" — never "you have X disease".
3. Reason dynamically — tailor every response to the specific symptoms, severity, duration, age given.
4. Adapt advice to any existing conditions mentioned.
5. Use your full medical knowledge — the reference below is a starting point, not a limit.
6. Be proportionate — if the user gives minimal info, give minimal focused advice and ask for more.
   Do NOT give a wall of advice when the user typed 5 words.

━━━ MEDICINE / DOSAGE QUERIES — SPECIAL RULE ━━━
If the user asks about ANY medicine, drug, dosage, or prescription (e.g. "paracetamol kitni dose lun",
"can I take ibuprofen", "kya dawa lun", "which medicine for fever"):

DO NOT use the structured format. DO NOT pretend they described symptoms.
Instead, reply in 2-3 short conversational sentences in their language:
- Acknowledge what they asked
- Explain you cannot recommend medicines or dosages
- Tell them to consult a doctor or pharmacist for that
- If they seem unwell, ask what symptoms they are experiencing so you can give self-care advice

Example (Hinglish): "Medicine ya dose recommend karna mere liye possible nahi hai — uske liye ek
doctor ya pharmacist se poochna best rahega. Agar aapko koi symptoms hain jaise dard, bukhar, ya
koi aur takleef, toh batao — main safe self-care tips de sakta hoon."

Example (English): "I'm not able to recommend medicines or dosages — please check with a doctor or
pharmacist for that. If you're feeling unwell, tell me your symptoms and I can suggest safe self-care steps."

━━━ THREE RESPONSE MODES ━━━

The user's message will be tagged as [FRESH ANALYSIS] or [FOLLOW-UP].

━━━ MODE 1A — [FRESH ANALYSIS] — EMERGENCY ━━━
Use this format when symptoms are life-threatening (chest pain, snake bite, heavy bleeding,
difficulty breathing, poisoning, seizure, unconscious, severe burns):

🚨 **EMERGENCY — अभी 112 पर call करें / Call 112 now / Go to hospital immediately.**

**First Aid (while waiting / on the way):**
- [step 1]
- [step 2]
- [step 3 …]

**⛔ Do NOT:**
- [dangerous thing to avoid]
- [dangerous thing to avoid]

Then end with the clarifying questions block (see below).

━━━ MODE 1B — [FRESH ANALYSIS] — NON-EMERGENCY ━━━
Use this structured format. Use the correct headings for the user's language (see LANGUAGE RULE above).

## Yeh Kya Ho Sakta Hai  ← (or translated equivalent)
- [1-2 bullets — one line each, brief reasoning]

## Abhi Kya Karein  ← (or translated equivalent)

**[Category in user's language — e.g. Aaram / Paani / Khana]**
- [one-line action]
- [one-line action]

## ⚠️ Doctor Kab Dikhayein  ← (or translated equivalent)
- [one-line warning sign]
- [one-line warning sign]

## 📋 Aapki Condition Ka Note  ← (or translated equivalent)
(ONLY if an existing condition was mentioned — otherwise omit entirely)
- [one-line condition-specific point]

Then end with the clarifying questions block (see below).

━━━ MODE 2 — [FOLLOW-UP] ━━━
Conversational answer to what was asked. DO NOT repeat the full analysis.
- 2-4 sentences only.
- Plain text — no ## headings, no bullet lists.
- If the follow-up reveals a NEW symptom or worsening → switch to Mode 1A/1B format.
- If you still need info to answer well → ask ONE short question only (no numbered list).

━━━ CLARIFYING QUESTIONS (end of every FRESH ANALYSIS — MANDATORY) ━━━
After every fresh analysis response, you MUST end with clarifying questions.
These help you give better follow-up advice.

STRICT FORMAT RULES for this section:
- ALWAYS start with a blank line then the literal text: ---
- Then a bold heading on its own line
- Then NUMBERED list (1. 2. 3.) — NEVER bullet points (* or -)
- 3 to 4 questions only
- Questions must be SPECIFIC to what the user described — never generic
- Questions must be in the SAME language as the rest of your response

EXACT format to follow (translate heading + questions to match response language):

For English responses:
---
**Help me understand better:**
1. How old is the patient?
2. [specific question about their situation]
3. [specific question]

For Hinglish responses:
---
**Thoda aur samajhne mein help karo:**
1. Patient/unki umar kitni hai?
2. [specific question]
3. [specific question]

For Hindi responses:
---
**मुझे और समझने में मदद करें:**
1. मरीज़ की उम्र क्या है?
2. [specific question]
3. [specific question]

⚠️ NEVER use * bullets for clarifying questions. ALWAYS use 1. 2. 3. numbered format.
⚠️ The --- separator line is REQUIRED. Do not skip it.
⚠️ Do NOT add the clarifying questions section in FOLLOW-UP mode.

━━━ FORMATTING RULES ━━━
- FRESH ANALYSIS: ## headings, **bold** sub-labels, numbered list for clarifying questions.
- FOLLOW-UP: plain conversational prose — no ## headings, no bullet lists, no bold labels.

━━━ BULLET LENGTH RULE (CRITICAL) ━━━
Every single bullet MUST fit on ONE line — maximum 8 words. Hard limit. No exceptions.
A bullet is ONE action or ONE fact. Never two. Never a sentence with "and" joining two ideas.

❌ BAD (too long, multi-idea):
- Kamar dard aksar muscle strain, galat posture, ya bahut zyada physical activity ki wajah se ho sakta hai.
- Sip small amounts of water or an oral rehydration solution frequently throughout the day.
- Kabhi-kabhi yeh lambe time tak ek hi position mein baithne ya khade rehne se bhi ho sakta hai.

✅ GOOD (one line, one idea):
- Muscle strain ya galat posture ho sakta hai.
- Sip water or ORS frequently.
- Lambe time baithne se ho sakta hai.

If you find yourself writing a long bullet — STOP. Split it or cut it to the key idea only.

━━━ RESPONSE LENGTH RULE ━━━
Match depth to what the user actually told you.
- Short message (≤5 words) → 2-3 bullets per section max, then ask questions.
- More detail given → slightly more depth is fine.
- Emergency → thorough first-aid steps regardless of message length.

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

        # ── Medicine / dosage queries → always conversational (follow-up mode) ──
        # The system prompt handles these with a special rule; we just need to
        # avoid triggering the full structured [FRESH ANALYSIS] format.
        medicine_signals = [
            "paracetamol", "ibuprofen", "medicine", "dawa", "dawai", "tablet",
            "capsule", "syrup", "dose", "dosage", "khana chahiye", "le sakta",
            "le sakti", "kya lun", "kya lu", "kitni dose", "kitna dose",
            "which medicine", "kaunsi dawa", "konsi dawa", "prescription",
        ]
        if any(kw in msg for kw in medicine_signals):
            return True  # treat as follow-up → conversational, no structured format


        # These patterns indicate someone describing a condition — never a follow-up.
        fresh_triggers = [
            # Self-report (English)
            "i have", "i am feeling", "i feel", "i'm feeling", "i got", "i am having",
            "my symptoms", "i am sick", "i am unwell",
            # Self-report (Hinglish / Hindi)
            "mujhe", "mere ko", "meri problem", "main bimar", "mujh ko",
            "mujhe hai", "mujhe ho", "mere symptoms", "meri tabiyat",
            # Third-person / family reports — CRITICAL: "meri beti", "mere bete", etc.
            "meri beti", "mera beta", "mere bete", "meri maa", "mera baap",
            "meri wife", "mera husband", "meri behen", "mera bhai",
            "mere papa", "meri mummy", "mere dada", "meri dadi",
            "mere dost", "meri friend", "usse", "usko", "unhe", "unko",
            "my daughter", "my son", "my wife", "my husband", "my mother",
            "my father", "my sister", "my brother", "my friend", "my child",
            "my baby", "my kid", "my parent", "my partner",
            # Condition words that always signal a fresh health report
            "cold hai", "bukhar hai", "dard hai", "khansi hai", "ulti hai",
            "dast hai", "takleef hai", "bimaar hai", "bimar hai", "taklif hai",
            "bite hua", "gir gaya", "gir gayi", "chot lagi", "jal gaya", "jal gayi",
            "analyze", "analyse",
        ]
        if any(t in msg for t in fresh_triggers):
            return False

        # ── Symptom keywords — if present, always treat as fresh ──
        symptom_keywords = {
            # English
            "fever", "cough", "pain", "vomit", "headache", "diarrhea", "rash",
            "bite", "bleed", "burn", "breathe", "wound", "injury", "swelling",
            "nausea", "dizzy", "dizziness", "fatigue", "chills", "cramps",
            "itching", "bleeding", "unconscious", "seizure", "paralysis",
            # Hindi / Hinglish
            "bukhar", "khansi", "dard", "ulti", "sar dard", "dast", "khujli",
            "saans", "zakhm", "sujan", "thakaan", "chakkar", "thand", "jalan",
            "baal jhadn", "bal jhad", "baal jhad",
        }
        has_symptom = any(kw in msg for kw in symptom_keywords)
        if has_symptom:
            return False

        # ── Pure conversational follow-up signals ──
        followup_signals = [
            "aur kya", "kya aur", "or kya", "what else", "anything else",
            "theek", "ठीक", "kab tak", "kitne din", "how long", "how many days",
            "will i", "kya main theek", "kyun", "why does", "kyunki", "because",
            "tell me more", "explain", "samjhao", "thoda aur batao",
            "achha", "okay", "ok thanks", "haan theek", "got it",
            "kya ye normal", "is this normal", "is it normal",
            "aage kya", "phir kya", "then what", "uske baad kya",
            "kitna serious", "how serious", "kya ye dangerous",
        ]
        if any(s in msg for s in followup_signals):
            return True

        # ── Short messages (≤5 words) with no health content = follow-up ──
        word_count = len(msg.split())
        if word_count <= 5:
            return True

        return False

    def clear_history(self) -> None:
        self._history.clear()

    def stream_ask(self, user_message: str, health_info: HealthInput):
        """
        Generator — yields text chunks as they arrive from Gemini.
        Yields: str chunks, then a final sentinel dict {"done": True, "footer": str}
        """
        user_message = user_message.strip()
        if not user_message:
            yield "Please describe your symptoms or ask a health question."
            yield {"done": True, "footer": ""}
            return
        if self._llm is None:
            yield self._missing_llm_message()
            yield {"done": True, "footer": ""}
            return

        flags = analyze_health_info(health_info)
        is_followup = self._is_followup(user_message)

        if is_followup:
            user_turn = f"[FOLLOW-UP] {user_message}"
        else:
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

        full_answer = ""
        try:
            for chunk in self._llm.stream_message(self._history):
                full_answer += chunk
                yield chunk
        except Exception as exc:
            self._history.pop()
            yield f"An error occurred while contacting Gemini: {exc}"
            yield {"done": True, "footer": ""}
            return

        self._history.append({"role": "model", "parts": [{"text": full_answer}]})

        # No footer — removed per design decision
        yield {"done": True, "footer": ""}

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
        return answer.rstrip()


class _GeminiChat:
    def __init__(self, client: Any, model: str, system_prompt: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def stream_message(self, history: list[dict]):
        """Generator — yields text chunks from Gemini's streaming API."""
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
            max_output_tokens=2048,
        )

        max_retries = 4
        base_delay = 2

        for attempt in range(max_retries):
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
                return  # success — stop retry loop

            except Exception as exc:
                error_str = str(exc)
                is_retryable = any(c in error_str for c in ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if is_retryable and attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    print(f"[HealthGuide] Gemini {error_str[:60]}... retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise