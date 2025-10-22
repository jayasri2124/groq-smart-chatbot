import streamlit as st
from chatbot_agent import SmartChatAgent
from docx import Document        # for .docx files
import fitz                      # for .pdf files (PyMuPDF)
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="💬 Shine Smart Chatbot", layout="centered")

# --- TITLE ---
st.title("💬 Shine Smart Chatbot")
st.markdown("Ask me anything or upload a text/Word/PDF file for RAG-based answers!")

# --- INIT CHATBOT AGENT ---
agent = SmartChatAgent()

# --- FILE UPLOAD SECTION ---
st.subheader("📁 Upload a .txt, .docx, or .pdf file for RAG")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=["txt", "docx", "pdf"], 
    help="Supports text, Word, and PDF files up to 200MB."
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    text = ""

    try:
        if file_name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        elif file_name.endswith(".docx"):
            # Read Word file from memory
            doc = Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([p.text for p in doc.paragraphs])

        elif file_name.endswith(".pdf"):
            # Read PDF pages
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in pdf_doc:
                text += page.get_text()

        if text.strip():
            agent.rag.add_file(text, file_name)
            st.success(f"✅ File '{file_name}' added to RAG successfully!")
        else:
            st.warning("⚠️ The file appears to be empty or unreadable.")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

st.divider()

# --- QUESTION SECTION ---
st.subheader("💬 Ask your question:")
question = st.text_area("Type your question here:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("🤔 Thinking..."):
            answer = agent.handle_query(question)
        st.subheader("Response:")
        st.write(answer)
    else:
        st.warning("Please enter a question before clicking Ask.")

st.markdown("---")
st.caption("Built with ❤️ using Groq, Streamlit, and Sentence Transformers.")
