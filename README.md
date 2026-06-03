# HealthGuide Chatbot

A non-medication health self-care advisor powered by Google Gemini (Vertex AI).

---

## Project Structure

```
healthguide/
├── app.py              ← Flask backend (serves UI + /chat API)
├── health_logic.py     ← Gemini chain, analysis logic
├── requirements.txt    ← Python dependencies
├── .env                ← YOUR secrets (never commit this)
├── .env.example        ← Template — copy to .env and fill in
├── .gitignore          ← Protects .env and JSON key from git
└── templates/
    └── index.html      ← Frontend UI
```

---

## Step 1 — Place your GCP JSON key

**Do NOT put the JSON key inside the project folder.**
Keep it in a safe location outside the repo, for example:

```
Windows:  C:\Users\YourName\gcp-keys\healthguide-key.json
Mac/Linux: /home/yourname/gcp-keys/healthguide-key.json
```

Why outside the project? So there's zero chance of accidentally committing it.
The `.gitignore` already blocks `*.json` inside the project, but keeping it outside
is the safest practice.

---

## Step 2 — Create your .env file

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then open `.env` and fill in your values:

```env
# Full path to your JSON key (use forward slashes, even on Windows)
GOOGLE_APPLICATION_CREDENTIALS=C:/Users/YourName/gcp-keys/healthguide-key.json

# Your GCP project ID — find it in Google Cloud Console top bar
GOOGLE_CLOUD_PROJECT=your-actual-project-id

GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash

PORT=5000
FLASK_DEBUG=false
```

**Never** paste the JSON key contents into `.env`. Only the file PATH goes there.

---

## Step 3 — Enable Vertex AI in GCP

In Google Cloud Console:
1. Go to **APIs & Services → Library**
2. Search for **Vertex AI API** and click **Enable**
3. Make sure your service account has the **Vertex AI User** role

---

## Step 4 — Install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Step 5 — Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Security checklist

| What | Where | In git? |
|------|-------|---------|
| GCP JSON key | Outside project folder | ❌ Never |
| `.env` file | Inside project folder | ❌ Never (in .gitignore) |
| `.env.example` | Inside project folder | ✅ Yes (no real values) |
| `health_logic.py` | Inside project folder | ✅ Yes |
| `app.py` | Inside project folder | ✅ Yes |

---

## API endpoint

`POST /chat`
```json
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
```

Returns: `{ "reply": "..." }`

`POST /clear` — Clears chat history
