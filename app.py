import streamlit as st
import time
import os
import json

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from langchain_ibm.chat_models import convert_to_openai_tool

st.set_page_config(page_title="QA Chatbot", layout="wide")


st.title("QA Chatbot with ACS LLMs 🤖")


# Sidebar for credentials and settings
with st.sidebar:
    st.header("🔧 Settings")
    
    api_key = st.text_input("API Key *", type="password")
    project_id = st.text_input("Project ID *")
    model_options = {
        "Google": [
            "google/flan-t5-xl"
        ],
        "IBM Granite": [
            "ibm/granite-13b-instruct-v2",
            "ibm/granite-3-2-8b-instruct",
            "ibm/granite-3-2b-instruct",
            "ibm/granite-3-3-8b-instruct",
            "ibm/granite-3-8b-instruct",
            "ibm/granite-8b-code-instruct",
            "ibm/granite-guardian-3-2b",
            "ibm/granite-guardian-3-8b",
            "ibm/granite-vision-3-2-2b"
        ],
        "Meta (LLaMA)": [
            "meta-llama/llama-2-13b-chat",
            "meta-llama/llama-3-2-11b-vision-instruct",
            "meta-llama/llama-3-2-1b-instruct",
            "meta-llama/llama-3-2-3b-instruct",
            "meta-llama/llama-3-2-90b-vision-instruct",
            "meta-llama/llama-3-3-70b-instruct",
            "meta-llama/llama-3-405b-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "meta-llama/llama-guard-3-11b-vision"
        ],
        "Mistral": [
            "mistralai/mistral-large",
            "mistralai/mistral-medium-2505",
            "mistralai/mistral-small-3-1-24b-instruct-2503",
            "mistralai/pixtral-12b"
        ]
    }
    
    # Create a two-level selectbox for model selection
    model_provider = st.selectbox("Model Provider", options=list(model_options.keys()))
    model_id = st.selectbox("Model ID *", options=model_options[model_provider])

    url = st.text_input("Endpoint URL *")

    proxy_ip = st.text_input("Proxy IP (optional)")
    proxy_port = st.text_input("Proxy Port (optional)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new user message
user_input = st.chat_input("Ask your question here...")

if user_input:
    if not all([api_key, project_id, model_id]):
        st.error("❗ Please fill in all required fields in the sidebar (marked with *)")
        st.stop()

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Proxy setup if provided
    if proxy_ip and proxy_port:
        proxy = f"http://{proxy_ip}:{proxy_port}"
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    try:
        credentials = Credentials(url=url, api_key=api_key)
        params = TextChatParameters(temperature=1)
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params
        )

        # Build message history for model
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history if m["role"] in ["user", "assistant"]]

        start_time = time.time()

        response = model.chat(messages=messages)

        end_time = time.time()
        
        content = response["choices"][0]["message"]["content"]
        elapsed = end_time - start_time
        tokens = len(content.split())
        speed = tokens / elapsed if elapsed > 0 else 0

        # Add assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": content})
        with st.chat_message("assistant"):
            st.markdown(content)

        st.info(f"⏱ Time: {elapsed:.2f}s | 🧮 Tokens: {tokens} | ⚡ Speed: {speed:.2f} tokens/s")

    except Exception as e:
        st.error(f"Error: {str(e)}")
