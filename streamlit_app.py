from chatbot_agent import SmartChatAgent
from dotenv import load_dotenv
import os
import streamlit as st

# --- Load environment variables first ---
load_dotenv()

# --- Get API Key (from .env or Streamlit secrets) ---
groq_api_key = os.getenv("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY. Please add it to your .env file or Streamlit secrets.")
else:
    st.success("✅ API key loaded successfully.")

# --- Optional check ---
st.write("API Key loaded:", bool(groq_api_key))

# --- Streamlit setup ---
st.set_page_config(page_title="🤖 Shine Smart Chatbot", page_icon="💬")
st.title("💬 Shine Smart Chatbot")
st.markdown("Ask me anything or upload a text file for RAG-based answers!")

# --- Initialize chatbot agent ---
agent = SmartChatAgent()

# --- Upload text file for RAG ---
uploaded_file = st.file_uploader("📁 Upload a .txt file,doc file", type=["txt"])
if uploaded_file is not None:
    text_data = uploaded_file.read().decode("utf-8")
    agent.rag.add_file(text_data, uploaded_file.name)
    st.success(f"✅ '{uploaded_file.name}' added to RAG knowledge base!")

# --- Chat input area ---
user_query = st.text_area("💬 Ask your question:")
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            answer = agent.handle_query(user_query)
            st.success("Response:")
            st.write("Response:", answer)
