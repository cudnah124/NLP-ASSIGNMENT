import streamlit as st
import os
import sys

# Add src to path so we can import rag_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RAGEngine

st.set_page_config(page_title="Contract QA Bot", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Contract Assistant (RAG)")
st.markdown("Hỏi đáp thông tin về hợp đồng dựa trên các điều khoản đã được phân tích.")

# Initialize RAG Engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine. Make sure you added your Gemini API Key to .env and ran data_ingestion.py first.\nError: {e}")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the contract..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu hợp đồng..."):
            response = engine.ask(prompt)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
