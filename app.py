"""
app.py — Flask backend for HealthGuide chatbot.
Serves the frontend and exposes a streaming /chat SSE endpoint.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

from health_logic import DynamicHealthChatbot, GeminiConfig, HealthInput

load_dotenv()

app = Flask(__name__)
CORS(app)

chatbot = DynamicHealthChatbot(max_history_messages=10)

_config = GeminiConfig(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
)
chatbot.configure(_config)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat  →  text/event-stream (SSE)

    Body (JSON):
      { "message": "...", "health_info": { ... } }

    Streams events:
      data: {"chunk": "...text..."}   — one per Gemini token batch
      data: {"done": true, "footer": "..."}  — signals end
      data: {"error": "..."}          — on failure
    """
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    health_data = data.get("health_info") or {}

    if not message:
        return jsonify({"error": "message is required"}), 400

    health_info = HealthInput.from_dict(health_data)

    def generate():
        try:
            for item in chatbot.stream_ask(message, health_info):
                if isinstance(item, dict):
                    # Final sentinel with footer
                    yield f"data: {json.dumps(item)}\n\n"
                else:
                    # Text chunk
                    yield f"data: {json.dumps({'chunk': item})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


@app.route("/clear", methods=["POST"])
def clear():
    chatbot.clear_history()
    return jsonify({"status": "cleared"})


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n✅ HealthGuide running at http://localhost:{port}\n")
    # threaded=True is required for SSE to work properly with Flask dev server
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)