"""
chat_app.py
A ChatGPT-style chat application built with Streamlit,
now wired to a real FastAPI backend instead of mock replies.

Run it with:
    streamlit run chat_app.py
"""

import os
import requests
import streamlit as st

# 🔌 BACKEND — Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ASK_ENDPOINT = f"{BACKEND_URL}/ask"
REQUEST_TIMEOUT = 60  # seconds — LLM calls can take a while

st.set_page_config(page_title="SuzanGPT", page_icon="icon2.png")
col1, col2 = st.columns([1, 5])
with col1:
    st.image("icon2.png", width=100)
with col2:
    st.title("SuzanGPT")
st.caption("Hi, I've read all your candidates' CVs. Ask me anything.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"|"assistant", "content": "..."}


# 🔌 BACKEND — Real API call replacing the old mock generator
def ask_backend(question: str) -> str:
    """
    Send the question to the FastAPI backend and return the answer string.
    Raises requests.RequestException on network/HTTP errors.
    """
    response = requests.post(
        ASK_ENDPOINT,
        json={"question": question},          # ← adjust key if your API expects something else
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["answer"]          # ← adjust key if your API returns something else


# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="icon2.png" if message["role"] == "assistant" else None):
        st.markdown(message["content"])


if prompt := st.chat_input("Type your message . . ."):
    # Save and display the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 🔌 BACKEND — Call the real API instead of generating a mock reply
    with st.chat_message("assistant", avatar="icon2.png"):
        with st.spinner("Thinking..."):
            try:
                response = ask_backend(prompt)
            except requests.Timeout:
                response = "⏱️ The backend took too long to respond. Try again in a moment."
            except requests.ConnectionError:
                response = f"🔌 Cannot reach the backend at `{BACKEND_URL}`. Is it running?"
            except requests.HTTPError as e:
                response = f"⚠️ Backend returned an error: `{e.response.status_code}`"
            except Exception as e:
                response = f"❌ Something went wrong: `{type(e).__name__}: {e}`"
        st.markdown(response)

    # Save the assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})


# Sidebar
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Backend**")
    st.code(BACKEND_URL, language=None)  # show which backend the app is talking to
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "SuzanGPT — a chat UI on top of my CV-RAG FastAPI backend, "
        "with Pinecone retrieval and a local semantic cache."
    )