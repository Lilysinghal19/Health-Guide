from dataclasses import dataclass


WARNING_SYMPTOMS = {"chest pain", "dizziness"}

HIGHER_RISK_CONDITIONS = {
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "pregnancy",
    "kidney disease",
    "immune system condition",
}


@dataclass
class HealthInput:
    symptoms: list[str]
    conditions: list[str]
    age: int | None
    duration: str
    severity: str
    breathing_difficulty: bool
    details: str = ""


def analyze_health_info(info: HealthInput) -> dict:
    symptom_set = {item.lower() for item in info.symptoms}
    condition_set = {item.lower() for item in info.conditions}

    has_cold_pattern = bool(
        symptom_set
        & {"fever", "cough", "sore throat", "runny nose", "fatigue", "body aches"}
    )
    has_digestive_pattern = bool(symptom_set & {"nausea", "vomiting", "diarrhea"})
    has_headache_pattern = bool(symptom_set & {"headache", "dizziness"})
    has_warning_symptom = bool(symptom_set & WARNING_SYMPTOMS)
    has_higher_risk_condition = bool(condition_set & HIGHER_RISK_CONDITIONS)
    has_age_risk = info.age is not None and (info.age < 5 or info.age >= 65)
    has_long_duration = info.duration in {"More than 1 week", "More than 2 weeks"}
    has_emergency_sign = info.breathing_difficulty or "chest pain" in symptom_set

    possible_causes = []
    if has_cold_pattern:
        possible_causes.append(
            "a mild viral illness, throat irritation, seasonal infection, or respiratory irritation"
        )
    if has_digestive_pattern:
        possible_causes.append(
            "a short-term stomach upset, food-related irritation, or digestive infection"
        )
    if has_headache_pattern:
        possible_causes.append(
            "tiredness, dehydration, stress, poor sleep, or an infection-related headache"
        )
    if not possible_causes:
        possible_causes.append(
            "a mild short-term health issue, though the pattern is not specific enough to identify one cause"
        )

    self_care = [
        "Drink plenty of water and fluids.",
        "Get adequate rest and sleep.",
        "Eat light, nutritious foods that are easy to tolerate.",
        "Avoid strenuous activity until you feel better.",
        "Monitor symptoms, temperature, breathing, and energy level.",
    ]

    if "sore throat" in symptom_set:
        self_care.append("Gargle with warm salt water.")
    if symptom_set & {"cough", "runny nose", "sore throat"}:
        self_care.append("Use steam inhalation if it feels comfortable.")
    if has_digestive_pattern:
        self_care.append(
            "Take small, frequent sips of fluid and choose bland, soft foods while symptoms settle."
        )

    seek_care = [
        "Symptoms worsen or do not improve after several days.",
        "Fever becomes high or persists.",
        "You feel unusually weak, confused, faint, or dehydrated.",
    ]

    if has_higher_risk_condition:
        seek_care.append(
            "Because of your existing condition, contact a healthcare professional earlier if symptoms continue or worsen."
        )
    if has_long_duration:
        seek_care.append(
            "Symptoms lasting more than a week should be reviewed by a healthcare professional."
        )
    if info.breathing_difficulty:
        seek_care.append("Breathing difficulty needs urgent medical attention.")
    if "chest pain" in symptom_set:
        seek_care.append("Chest pain needs urgent medical attention.")
    if has_age_risk:
        seek_care.append(
            "Very young children and older adults should be assessed sooner when symptoms are moderate, severe, or persistent."
        )

    return {
        "possible_causes": _unique(possible_causes),
        "self_care": _unique(self_care),
        "seek_care": _unique(seek_care),
        "urgent": has_emergency_sign,
        "serious": has_emergency_sign
        or info.severity == "Severe"
        or has_warning_symptom
        or has_age_risk,
        "cautious": has_higher_risk_condition
        or has_long_duration
        or info.severity == "Moderate",
    }


def build_response(analysis: dict) -> str:
    if analysis["urgent"]:
        opening = (
            "Please seek urgent medical help.\n\n"
            "Some information you shared can be serious. Contact a healthcare "
            "professional or local emergency services now, especially if symptoms are worsening."
        )
    elif analysis["serious"]:
        opening = (
            "Medical guidance is recommended.\n\n"
            "Your symptoms may be consistent with "
            + "; ".join(analysis["possible_causes"])
            + "."
        )
    else:
        opening = (
            "Your symptoms may be consistent with "
            + "; ".join(analysis["possible_causes"])
            + "."
        )

    caution = ""
    if analysis["cautious"] and not analysis["urgent"]:
        caution = (
            "\n\nExisting conditions, symptom duration, or symptom severity can "
            "change how quickly you should get checked."
        )

    return (
        f"{opening}{caution}\n\n"
        f"Suggested self-care:\n{_bullet_list(analysis['self_care'])}\n\n"
        f"Seek medical care if:\n{_bullet_list(analysis['seek_care'])}\n\n"
        "No medicine advice: This chatbot does not recommend drug names, dosages, or prescriptions."
    )


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
