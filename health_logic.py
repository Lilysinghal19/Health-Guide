"""
health_logic.py — Core logic for HealthGuide chatbot.
Uses Google Vertex AI (Gemini) directly — no vector DB, no retriever.
All guidance is injected into the system prompt on every call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# ── Knowledge base (injected into system prompt directly) ─────────────────────
# Since we have no vector DB, we simply pass ALL guidance to Gemini every time.
# Gemini 2.5 Flash has a 1M token context window — this is fine.

HEALTH_KNOWLEDGE = """
=== GENERAL SELF-CARE ===
Provide educational guidance only. Never diagnose, prescribe, name medicines, suggest dosages,
or replace a healthcare professional. Safe non-medication advice includes: rest, hydration
(water/clear fluids/ORS), sleep, light nutrition, steam inhalation if comfortable, warm
salt-water gargles, avoiding strenuous activity, and symptom monitoring.

=== RESPIRATORY SYMPTOMS ===
Fever, cough, sore throat, runny nose, fatigue, body aches may be consistent with viral
illness, throat irritation, or seasonal infection.
Self-care steps:
- Rest 8-10 hours; avoid going out unnecessarily
- Drink fluids every 30-60 minutes (water, coconut water, clear soups)
- Steam inhalation: bowl of hot water, lean over with towel over head, 5-10 min twice daily
- Salt-water gargle: half tsp salt in warm water, gargle 30 sec, 3-4 times/day
- Sleep with head elevated if congested
- Eat light warm foods (khichdi, soups, broth)
- Avoid cold drinks, fried/heavy food, dairy if congested

=== FEVER MANAGEMENT (non-medication) ===
- Rest and stay indoors in a ventilated room
- Drink cool fluids frequently (water, coconut water, diluted juice, ORS)
- Apply damp cool cloth on forehead, armpits, or back of neck; change every few minutes
- Wear light, breathable clothing; remove heavy blankets
- Sponge body with lukewarm (not cold) water
- Monitor temperature every 2-3 hours
Seek medical care if: very high fever, fever in infant under 3 months, fever lasting
more than 3 days, or fever with stiff neck, rash, confusion, or breathing difficulty.

=== DIGESTIVE SYMPTOMS ===
Nausea, vomiting, diarrhea may be consistent with stomach upset, food irritation, or
digestive infection.
Self-care steps:
- Sip small amounts of water or ORS every 10-15 minutes rather than large gulps
- Rest the stomach: start with clear liquids (water, diluted juice, clear broth), then
  gradually introduce bland food (plain rice, toast, banana, boiled potato)
- Avoid dairy, fatty, spicy, or heavy foods until symptoms settle
- Lie on left side to reduce nausea
- Keep track of how many times vomiting/diarrhea occurs
- For diarrhea: ORS (1 litre water + 6 tsp sugar + half tsp salt) sip continuously
Watch for dehydration: dark/no urine, dry mouth, dizziness, sunken eyes, extreme weakness.
Seek care if: unable to keep any fluids down for 6+ hours, blood in stool/vomit,
symptoms lasting more than 3 days.

=== HEADACHE & FATIGUE ===
Often linked to dehydration, poor sleep, stress, eye strain, or viral illness.
Self-care steps:
- Drink a full glass of water immediately
- Rest in a quiet, dark, cool room for 20-30 minutes
- Apply a cool damp cloth to forehead or back of neck
- Gentle neck and shoulder stretches if tension-related
- Reduce screen time and bright lights
- Ensure 7-9 hours of sleep
- Eat small regular meals; do not skip meals

=== DEHYDRATION ===
Self-care steps:
- Sip ORS, water, or diluted coconut water every few minutes (small amounts, often)
- Lie down and rest; elevate legs slightly if dizzy
- Clear broth and diluted fruit juices also help
- Avoid caffeinated drinks (worsen fluid loss)
Signs of severe dehydration needing emergency care: no urination for 8+ hours, sunken
eyes, confusion, inability to keep fluids down, very rapid weak pulse.

=== SKIN SYMPTOMS, RASHES & MINOR INSECT BITES ===
- Clean area with mild soap and cool running water for 5 minutes
- Apply cool compress (cloth soaked in cold water) to reduce swelling and itching
- Keep area elevated if possible
- Do not scratch; trim nails short to prevent infection
- Keep the area clean and dry; cover with clean dressing
Watch for: spreading redness, increased warmth, pus, red streaks from the wound —
these may indicate infection requiring medical care.

=== SNAKE BITE FIRST AID (EMERGENCY — GO TO HOSPITAL IMMEDIATELY) ===
While getting to hospital, do the following:
1. Keep the person completely calm and still — any movement pumps venom faster through lymph
2. Immobilize the bitten limb using a splint or firm bandage; keep it BELOW heart level
3. Remove all tight items near bite BEFORE swelling starts: rings, watches, bracelets, shoes
4. Mark the outer edge of any swelling with a pen and write the time of the bite
5. For neurotoxic snake bites (cobra, krait — cause droopy eyelids, weakness, trouble breathing):
   Apply a firm pressure-immobilization bandage starting 5-10 cm above the bite, wrapping DOWN
   toward fingers/toes with even pressure (like a sprain bandage — firm but fingers still pink)
6. For hemotoxic/viper bites (cause local swelling, tissue damage, bleeding): do NOT apply
   pressure bandage — keep limb still, below heart, and get to hospital
7. DO NOT: cut the wound, suck out venom, apply ice or heat, use a tourniquet, give food or drink
8. Lay the person flat; if they vomit, turn them on their side
9. Note the snake's appearance if possible (colour, pattern, head shape) — do not try to catch it
10. Carry the person to transport; do not let them walk
This is a medical emergency. Anti-venom must be given at a hospital as fast as possible.

=== WOUND & BLEEDING FIRST AID ===
- Apply direct firm pressure with a clean cloth or bandage for 10-15 minutes continuously
  (do not lift to check — this breaks the clot)
- Elevate the injured part above heart level while maintaining pressure
- For deep or gaping wounds: gently hold edges together while pressing
- Once bleeding slows: clean with clean running water for 5-10 minutes
- Cover with clean bandage; change bandage if it soaks through (add more on top, do not remove)
- Do not use dirty cloth; do not probe the wound
Seek emergency care if: bleeding does not stop after 15 minutes of firm pressure, wound
is deep or gaping, puncture from dirty object, signs of infection (heat, pus, red streaks),
wound on face, hands, or genitals.

=== BURNS FIRST AID ===
- For minor burns (small, redness, no blisters): immediately run cool (not cold, not ice) water
  over the burn for at least 10-20 minutes
- Remove jewellery and clothing near the burn before swelling starts
- Do NOT apply: butter, oil, toothpaste, raw egg — these trap heat and cause infection
- Cover loosely with clean non-fluffy material (cling film or clean plastic bag is ideal)
- Do not pop blisters
- Keep burn elevated if possible
Seek immediate care for: burns larger than palm size, burns on face/hands/feet/genitals/joints,
any burns in children or elderly, burns that are white/charred/no pain (deep burn),
chemical or electrical burns — go to hospital regardless of size.

=== BREATHING DIFFICULTY (URGENT) ===
This is an emergency if severe. While getting help:
- Sit upright or lean slightly forward (do not lie flat — this worsens breathing)
- Loosen all tight clothing around neck, chest, and waist
- Stay as calm as possible — panic increases oxygen demand
- Open windows or move to fresh air if indoors
- If known asthma: use the prescribed inhaler technique (spacer if available)
- Do not give food or drink
Call emergency services immediately if: lips or fingernails turn blue, cannot speak full
sentences, breathing is visibly very labored, child's ribs show with every breath.

=== WARNING SIGNS REQUIRING IMMEDIATE MEDICAL CARE ===
Go to hospital or call emergency services (112 in India) for any of these:
- Chest pain or tightness
- Difficulty breathing or very rapid breathing
- Loss of consciousness or fainting
- Uncontrolled or heavy bleeding
- Suspected poisoning, snake bite, or envenomation
- Confusion, altered mental state, or strange behaviour
- High fever in infants under 3 months (any fever is emergency)
- Seizures or convulsions
- Sudden severe headache (worst headache of life)
- Paralysis, numbness, or sudden weakness on one side of body
- Severe allergic reaction: swelling of face, lips, or throat; hives with breathing difficulty
- Vomiting blood or blood in stool

=== EXISTING CONDITIONS — ADAPTED ADVICE ===
- Diabetes: Monitor blood sugar closely during illness; illness causes blood sugar to rise.
  Stay well hydrated with plain water or ORS. Watch for confusion, excessive thirst, nausea.
  Seek medical care earlier than healthy adults — within 24 hours if moderate symptoms.
- Hypertension: Avoid high-sodium home remedies. Do not eat salty broths excessively.
  Rest; avoid strenuous activity. Monitor blood pressure if device available. Seek care early.
- Asthma: Avoid steam inhalation if it triggers symptoms. Stay away from dust, smoke, cold air.
  Keep inhaler accessible. Seek care immediately if breathing worsens.
- Heart disease: Any chest discomfort, palpitations, or unusual breathlessness — seek care
  immediately. Rest completely. Avoid strenuous activity, even mild exertion.
- Pregnancy: Any fever, vomiting, abdominal pain, bleeding, reduced fetal movement — seek
  medical care promptly. Stay well hydrated. Avoid lying flat on back after 20 weeks.
- Kidney disease: Stay hydrated but do not overdrink. Avoid high-potassium foods during illness.
  Watch for reduced urine output or swelling. Seek care if symptoms are moderate or severe.
- Immune system condition: Lower threshold to seek care — infections can worsen quickly.
  Any fever, wound infection, or breathing difficulty warrants prompt medical attention.
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
            f"Symptoms selected: {', '.join(self.symptoms) or 'none'}",
            f"Free-text description: {self.details or 'none'}",
            f"Existing conditions: {', '.join(self.conditions) or 'None'}",
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
    credentials_path: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    )
    project: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    )
    location: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    )


# ── Risk analysis ─────────────────────────────────────────────────────────────

def analyze_health_info(info: HealthInput) -> dict:
    """Return risk flags used to shape the prompt and response urgency."""
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


def _build_system_prompt() -> str:
    return f"""You are HealthGuide, a careful and compassionate health self-care advisor.

STRICT RULES — follow these without exception:
1. NEVER name any medicine, drug, supplement, or dosage — not even paracetamol, ibuprofen, ORS brands, etc.
2. NEVER diagnose. Use language like "may be consistent with" instead of "you have".
3. Always give ALL relevant non-medication self-care steps: rest, hydration, sleep, nutrition,
   steam, salt gargles, wound care, positioning, cooling, monitoring, etc. Be thorough.
4. For emergencies (snake bite, bleeding, chest pain, breathing difficulty, burns):
   Give EVERY possible first-aid step the person should do WHILE getting to hospital.
   Lead with "🚨 This is an emergency — call 112 or go to hospital immediately." then list steps.
5. Adapt advice to existing conditions (e.g., diabetes → mention blood sugar monitoring;
   asthma → caution with steam; pregnancy → safe positioning; heart disease → rest completely).
6. If the situation is urgent, say so clearly at the very start.
7. End EVERY response with a short "⚠️ Watch for these warning signs:" section listing
   specific signs that mean the person should seek medical care immediately.
8. Keep responses clear, structured, and easy to follow under stress.

You have the following health guidance knowledge available:

{HEALTH_KNOWLEDGE}

Use this knowledge to inform your responses. If the user describes a situation not covered
above, use your best judgment while staying within the non-medication, self-care scope.
"""


# ── Chatbot ───────────────────────────────────────────────────────────────────

class DynamicHealthChatbot:
    """
    Direct Gemini chatbot with in-memory message history.
    No vector DB, no retriever — knowledge is injected into the system prompt.
    """

    def __init__(self, max_history_messages: int = 10) -> None:
        self.max_history_messages = max_history_messages
        self.config = GeminiConfig()
        self._llm: Any = None
        self._llm_error: str = ""
        self._config_key: str = ""
        # Chat history: list of {"role": "user"|"model", "parts": [{"text": "..."}]}
        self._history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def configure(self, config: GeminiConfig) -> None:
        """Rebuild the LLM client only when config actually changes."""
        key = "|".join([config.credentials_path, config.project, config.location, config.model])
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

        # Build the full user turn: patient context + message
        full_user_turn = (
            f"Patient information:\n{health_info.to_prompt_context()}\n\n"
            f"Risk flags: {flags}\n\n"
            f"Question: {user_message}"
        )

        # Append to history
        self._history.append({"role": "user", "parts": [{"text": full_user_turn}]})

        # Trim to max_history_messages pairs (user+model = 2 entries per turn)
        max_entries = self.max_history_messages * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        try:
            response = self._llm.send_message(self._history)
            answer = response.text
        except Exception as exc:
            # Remove the failed user turn so history stays clean
            self._history.pop()
            answer = f"An error occurred while contacting Gemini: {exc}"
            return answer

        # Append model response to history
        self._history.append({"role": "model", "parts": [{"text": answer}]})

        return self._add_safety_footer(answer, flags)

    def clear_history(self) -> None:
        self._history.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_llm(self) -> Any:
        """Build a Vertex AI GenerativeModel client."""
        self._prepare_gcp_credentials()

        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        vertexai.init(
            project=self.config.project or None,
            location=self.config.location,
        )

        model = GenerativeModel(
            model_name=self.config.model,
            system_instruction=_build_system_prompt(),
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )

        # Return a chat session starter — we manage history ourselves
        # so we use the model's start_chat with empty history each time
        # and pass our accumulated history on each call.
        return _StatelessChat(model)

    def _prepare_gcp_credentials(self) -> None:
        cred_path = self.config.credentials_path.strip()

        if not cred_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
                "Add it to your .env file:\n"
                r"GOOGLE_APPLICATION_CREDENTIALS=C:\Users\abhis\OneDrive\Documents\Desktop\HealthGuide\keys\gcp_key.json"
            )

        if not os.path.exists(cred_path):
            raise FileNotFoundError(
                f"GCP key file not found at:\n  {cred_path}\n\n"
                "Check that the path in your .env file is correct and the file exists."
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

        # Always read project_id from the JSON key file.
        # This overrides any placeholder value like "your-gcp-project-id".
        with open(cred_path, encoding="utf-8") as f:
            key_data = json.load(f)

        project_from_key = key_data.get("project_id", "").strip()

        # Treat placeholder / empty project as "not set" — use key file value
        _PLACEHOLDERS = {"", "your-gcp-project-id", "your-project-id", "YOUR_PROJECT_ID"}
        if not self.config.project or self.config.project.strip() in _PLACEHOLDERS:
            if not project_from_key:
                raise ValueError(
                    "Could not find 'project_id' inside your GCP JSON key file.\n"
                    "Set GOOGLE_CLOUD_PROJECT explicitly in your .env file."
                )
            self.config.project = project_from_key

    def _missing_llm_message(self) -> str:
        msg = (
            "⚠️ Gemini is not configured yet.\n\n"
            "To fix this:\n"
            "1. Copy .env.example to .env\n"
            "2. Set GOOGLE_APPLICATION_CREDENTIALS = full path to your GCP JSON key\n"
            "3. Set GOOGLE_CLOUD_PROJECT = your GCP project ID\n"
            "4. Restart the server with: python app.py\n\n"
            "Only the file PATH goes in .env — never paste the JSON contents."
        )
        if self._llm_error:
            msg += f"\n\nError detail: {self._llm_error}"
        return msg

    def _add_safety_footer(self, answer: str, flags: dict) -> str:
        footer = (
            "\n\n---\n"
            "⚕️ *No medicines recommended. Educational guidance only — "
            "not a substitute for professional medical care.*"
        )
        if flags.get("urgent"):
            footer = (
                "\n\n🚨 *This situation may require emergency care. "
                "Call 112 or go to the nearest hospital immediately.*"
            ) + footer
        return answer.rstrip() + footer


class _StatelessChat:
    """
    Thin wrapper so we can pass explicit history on every call
    instead of relying on a stateful ChatSession object.
    This gives us full control over history trimming.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def send_message(self, history: list[dict]) -> Any:
        """Start a fresh chat with the full history injected, then get the last response."""
        from vertexai.generative_models import Content, Part

        # Convert our dict history to Vertex AI Content objects
        # Skip the last entry (current user turn) — send it separately
        past = history[:-1]
        current = history[-1]

        contents = [
            Content(role=entry["role"], parts=[Part.from_text(p["text"]) for p in entry["parts"]])
            for entry in past
        ]
        current_text = current["parts"][0]["text"]

        chat = self._model.start_chat(history=contents)
        return chat.send_message(current_text)