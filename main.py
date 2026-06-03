from pathlib import Path

from flask import Flask, jsonify, request, send_file

from health_logic import DynamicHealthChatbot, GeminiConfig, HealthInput


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_KEY_PATH = BASE_DIR / "keys" / "gcp_key.json"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"

app = Flask(__name__)
chatbot = DynamicHealthChatbot(max_history_messages=10)


@app.get("/")
def index():
    return send_file(BASE_DIR / "ui.html")


@app.get("/status")
def status():
    return jsonify(
        {
            "key_exists": DEFAULT_KEY_PATH.exists(),
            "key_path": str(DEFAULT_KEY_PATH),
            "model": DEFAULT_MODEL,
            "location": DEFAULT_LOCATION,
        }
    )


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    message = str(data.get("message", "")).strip()
    patient = data.get("health_info", {})

    health_info = HealthInput(
        symptoms=patient.get("symptoms", []),
        conditions=patient.get("conditions", ["None"]),
        age=parse_age(patient.get("age")),
        duration=patient.get("duration", "1 to 3 days"),
        severity=patient.get("severity", "Mild"),
        breathing_difficulty=patient.get("breathing") == "Yes",
        details=patient.get("details", ""),
    )

    chatbot.configure(
        GeminiConfig(
            credentials_path=str(DEFAULT_KEY_PATH),
            location=DEFAULT_LOCATION,
            model=DEFAULT_MODEL,
        )
    )
    answer = chatbot.ask(message, health_info)
    return jsonify({"answer": answer})


@app.post("/clear")
def clear():
    chatbot.clear_history()
    return jsonify({"ok": True})


def parse_age(value):
    if value in (None, ""):
        return None
    try:
        age = int(value)
    except ValueError:
        return None
    return age if 0 <= age <= 120 else None


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
