"""
app.py — Flask backend for HealthGuide chatbot.
Serves the frontend and exposes a /chat API endpoint.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from health_logic import DynamicHealthChatbot, GeminiConfig, HealthInput

# Load .env before anything else
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow browser requests from same origin

# Single shared chatbot instance (stateful session per user in production
# would use session IDs, but this is fine for a local / single-user app)
chatbot = DynamicHealthChatbot(max_history_messages=10)

# Configure Gemini from environment on startup
_config = GeminiConfig(
    api_key=os.getenv("GOOGLE_API_KEY", ""),
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
)
chatbot.configure(_config)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body (JSON):
      {
        "message": "What should I do?",
        "health_info": {
          "symptoms": ["Fever", "Cough"],
          "conditions": ["Diabetes"],
          "age": 34,
          "duration": "1 to 3 days",
          "severity": "Mild",
          "breathing_difficulty": false,
          "details": "Started yesterday evening"
        }
      }
    Returns: { "reply": "..." }
    """
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    health_data = data.get("health_info") or {}

    if not message:
        return jsonify({"error": "message is required"}), 400

    health_info = HealthInput.from_dict(health_data)
    reply = chatbot.ask(message, health_info)
    return jsonify({"reply": reply})


@app.route("/clear", methods=["POST"])
def clear():
    """Clear chat history."""
    chatbot.clear_history()
    return jsonify({"status": "cleared"})


@app.route("/health")
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n✅ HealthGuide running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
