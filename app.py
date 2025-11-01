# app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

# -----------------------------
# 1) GEMINI API KEY MANAGEMENT (Safe for GitHub)
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️  Gemini API key not found!")
    print("Please set it before running the app:")
    print("  ▶ On Windows PowerShell:  setx GEMINI_API_KEY \"YOUR_API_KEY_HERE\"")
    print("  ▶ On macOS/Linux:          export GEMINI_API_KEY=\"YOUR_API_KEY_HERE\"")
    raise RuntimeError("Missing Gemini API key. Please set GEMINI_API_KEY environment variable.")

# Configure Gemini SDK
MODEL_NAME = "gemini-2.5-flash-lite"
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# 2) CUSTOMIZED CHAT PROFILES (for Aerospace AI)
# -----------------------------
PROFILES = {
    "default": {
        "system": (
            "You are 'AstroMind', an expert AI specializing in aerospace engineering, deep space exploration, "
            "astrophysics, astronomy, and rocket science. Provide clear, accurate, and technically rich answers. "
            "If asked about general topics, connect them to space or physics when possible. "
            "Be concise, factual, and cite NASA/ESA/ISRO references when relevant."
        ),
        "generation_config": {"temperature": 0.55, "top_p": 0.9, "top_k": 40, "max_output_tokens": 512},
    },
    "teacher": {
        "system": (
            "You are 'AstroMentor', a friendly aerospace instructor. Explain step-by-step, "
            "use analogies from space missions (like Chandrayaan, Mars Rover, or Apollo), "
            "and make complex physics easy to understand for students."
        ),
        "generation_config": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": 700},
    },
    "support": {
        "system": (
            "You are 'MissionControl', a technical support AI for aerospace systems. "
            "Help users troubleshoot rocket propulsion issues, satellite systems, and simulation software. "
            "Ask 1–2 clarifying questions first, then list solutions in bullet points."
        ),
        "generation_config": {"temperature": 0.5, "top_p": 0.85, "top_k": 40, "max_output_tokens": 600},
    },
}

# -----------------------------
# 3) FLASK APP SETUP
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.get("/")
def index():
    return render_template("index.html")

# -----------------------------
# 4) CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history", [])             # [{role: "user"/"model", text: "..."}]
    mode = (data.get("mode") or "default").lower()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Pick profile (fallback to default)
    profile = PROFILES.get(mode, PROFILES["default"])
    system_instruction = profile["system"]
    generation_config = profile["generation_config"]

    # Build chat history for Gemini (keep last few turns)
    contents = []
    for turn in history[-8:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [turn.get("text", "")]})
    contents.append({"role": "user", "parts": [user_message]})

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )
        resp = model.generate_content(contents)
        text = resp.text or "(No response text)"
        return jsonify({"reply": text, "mode": mode})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 5) APP RUNNER
# -----------------------------
if __name__ == "__main__":
    print("🚀 Launching Aerospace AI Chatbot on http://localhost:8000")
    print("💫 Model:", MODEL_NAME)
    print("🌌 Mode options: default | teacher | support")
    app.run(host="0.0.0.0", port=8000, debug=True)
