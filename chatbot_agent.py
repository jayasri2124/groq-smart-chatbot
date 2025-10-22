import os
import sqlite3
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from docx import Document  # ✅ for Word file support

# Load .env
load_dotenv()

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("⚠️ Missing GROQ_API_KEY in .env")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- EMBEDDINGS ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- FILE RAG (Retrieval) ---
class FileRAG:
    def __init__(self):
        self.docs = []  # list of (filename, text)
        self.index = None

    def add_file(self, text_or_path, filename: str):
        """Add a file to the in-memory RAG index."""
        text = ""
        # ✅ handle DOCX files
        if filename.endswith(".docx"):
            try:
                doc = Document(text_or_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                text = f"⚠️ Error reading DOCX: {e}"
        elif filename.endswith(".txt"):
            # handle text files
            if isinstance(text_or_path, bytes):
                text = text_or_path.decode("utf-8", errors="ignore")
            else:
                text = text_or_path
        else:
            text = str(text_or_path)

        self.docs.append((filename, text))
        self._rebuild_index()

    def _rebuild_index(self):
        if not self.docs:
            self.index = None
            return
        texts = [t[1] for t in self.docs]
        embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 2) -> str:
        """Return relevant text context from uploaded files."""
        if not self.index or not self.docs:
            return ""
        q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
        _, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx][1])
        return "\n\n".join(results)


# --- SQLITE CACHE ---
class SQLDatabase:
    def __init__(self, db_path="data.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT
            )
            """
        )
        self.conn.commit()

    def insert(self, question: str, answer: str):
        self.cursor.execute("INSERT INTO info (question, answer) VALUES (?, ?)", (question, answer))
        self.conn.commit()

    def query(self, question: str):
        q = "SELECT answer FROM info WHERE question LIKE ? LIMIT 1"
        row = self.cursor.execute(q, (f"%{question}%",)).fetchone()
        return row[0] if row else None


# --- SMART CHAT AGENT ---
class SmartChatAgent:
    def __init__(self):
        self.rag = FileRAG()
        self.db = SQLDatabase()

    def _groq_chat(self, prompt: str) -> str:
        """Call Groq API for chat completion."""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        try:
            res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            try:
                data = res.json()
            except Exception:
                res.raise_for_status()
                return "⚠️ Groq API returned non-JSON response"

            if res.status_code != 200:
                err = data.get("error") or data
                return f"⚠️ Groq API error: {res.status_code} {err}"

            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or choice.get("text") or ""
            return content.strip()
        except Exception as e:
            return f"⚠️ Groq API exception: {e}"

    def handle_query(self, query: str) -> str:
        """Handles a user question using DB → RAG → Groq fallback logic."""
        if not query or not query.strip():
            return "⚠️ Please type a question."

        # 1️⃣ Check DB cache
        db_answer = self.db.query(query)
        if db_answer:
            return f"(From Database)\n{db_answer}"

        # 2️⃣ Search in uploaded docs (RAG)
        rag_ctx = self.rag.search(query)
        if rag_ctx:
            prompt = f"Answer using only this context:\n{rag_ctx}\n\nQuestion: {query}"
            answer = self._groq_chat(prompt)
            try:
                self.db.insert(query, answer)
            except Exception:
                pass
            return answer

        # 3️⃣ Fallback to Groq directly
        answer = self._groq_chat(query)
        try:
            self.db.insert(query, answer)
        except Exception:
            pass
        return answer
