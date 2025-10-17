# test_groq.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL,
    "messages": [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
}

resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
print(resp.status_code)
try:
    print(resp.json())
except Exception:
    print(resp.text)
