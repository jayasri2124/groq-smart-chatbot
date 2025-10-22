import os
from flask import Flask, request, render_template, jsonify
from chatbot_agent import SmartChatAgent
from docx import Document  # ✅ import for Word files

app = Flask(__name__, template_folder="templates")
agent = SmartChatAgent()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("question", "")
    answer = agent.handle_query(user_input)
    return jsonify({"answer": answer})


@app.route("/upload", methods=["POST"])
def upload():
    """Accepts .txt or .docx and adds to RAG."""
    file = request.files.get("file")
    if not file:
        return jsonify({"message": "⚠️ No file uploaded."})

    filename = file.filename
    try:
        if filename.endswith(".docx"):
            doc = Document(file)
            content = "\n".join([p.text for p in doc.paragraphs])
        else:
            content = file.read().decode("utf-8", errors="ignore")

        agent.rag.add_file(content, filename)
        return jsonify({"message": f"✅ File '{filename}' added to RAG."})
    except Exception as e:
        return jsonify({"message": f"⚠️ Upload failed: {e}"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
