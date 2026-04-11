# backend/app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from groq_client import ask_groq
from rag import retrieve, build_prompt  # <-- ADD THIS IMPORT

load_dotenv()

app = Flask(__name__)

# Allow requests from the React dev server (port 5173)
# In production, replace "*" with your actual domain
CORS(app, origins=["http://localhost:5173", os.getenv("FRONTEND_URL", "*")])

# Which Ollama model to use (override with MODEL_NAME env var)
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

@app.route("/health", methods=["GET"])
def health():
    """Simple health check — useful for testing that the server is up."""
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives: { "message": "What is your experience with Python?" }
    Returns:  { "answer": "..." }
    """
    data = request.get_json(silent=True)

    # Validate input
    if not data or not data.get("message"):
        return jsonify({"error": "Missing 'message' field"}), 400

    question = data["message"].strip()

    if len(question) > 1000:
        return jsonify({"error": "Message too long (max 1000 chars)"}), 400

    try:
        # 1. Retrieve relevant chunks from ChromaDB
        relevant_chunks = retrieve(question, n_results=4)
        
        # 2. Build the prompt using the retrieved chunks
        prompt = build_prompt(question, relevant_chunks)
        
        # 3. Get answer from Groq
        answer = ask_groq(prompt).strip()
        return jsonify({"answer": answer})

    except Exception as e:
        # Log to console; return a safe error to the client
        print(f"[ERROR] {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    # debug=True gives auto-reload during development
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True)