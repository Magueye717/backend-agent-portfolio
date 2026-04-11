import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def ask_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama-3.1-8b-instant",  # ⚡ fast + good
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # reduce hallucination
        "max_tokens": 400,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]