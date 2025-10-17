import streamlit as st
import os
from chatbot_agent import SmartChatAgent
from dotenv import load_dotenv

<<<<<<< HEAD

# Load from secrets.toml
groq_api_key = st.secrets["GROQ_API_KEY"]

# Optional: check if key loaded
st.write("API Key loaded successfully:", bool(groq_api_key))
# Load environment variables
load_dotenv()

st.set_page_config(page_title="🤖 shine Smart Chatbot", page_icon="💬")
st.title("💬 shine Smart Chatbot")
st.markdown("Ask me anything or upload a text file for RAG-based answers!")

# Initialize chatbot
agent = SmartChatAgent()

# --- Upload text file for RAG ---
uploaded_file = st.file_uploader("📁 Upload a .txt file", type=["txt"])
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
            st.write(answer)
    else:
        st.warning("Please type a question first.")

st.markdown("---")
st.markdown("Built with ❤️ using Groq, Streamlit, and Sentence Transformers.")
