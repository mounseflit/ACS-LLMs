import base64
import time
from typing import Any, Dict, List, Optional, Union

import streamlit as st

try:
    from ibm_watsonx_ai import Credentials  # type: ignore
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
    # Import parameter classes when available. Some versions of the SDK only
    # support TextChatParameters, so we import conditionally.
    try:
        from ibm_watsonx_ai.foundation_models.schema import TextChatParameters  # type: ignore
    except Exception:
        TextChatParameters = None  # type: ignore
except Exception:
    # Provide a clear error message if the IBM SDK is missing. Running the app
    # without the SDK installed will result in this placeholder being used.
    raise ImportError(
        "The ibm_watsonx_ai Python package is required to run this app. "
        "Install it with `pip install ibm-watsonx-ai` and try again."
    )


def encode_image_to_base64(uploaded_file: Any) -> Optional[Dict[str, str]]:
    """Encode an uploaded image file to a base64 data URI.

    Returns a dictionary with two keys: ``data_uri`` and ``format``. If the
    uploaded file is None or cannot be read, returns None.

    Parameters
    ----------
    uploaded_file: Any
        File-like object returned by st.file_uploader.

    Returns
    -------
    Optional[Dict[str, str]]
        Dictionary containing the base64 data URI and the image format, or
        None if encoding fails.
    """
    if not uploaded_file:
        return None
    try:
        image_bytes = uploaded_file.getvalue()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        # Derive the MIME subtype from the uploaded file's content type
        content_type: str = uploaded_file.type or "image/png"
        subtype = content_type.split("/")[-1]
        data_uri = f"data:{content_type};base64,{encoded}"
        return {"data_uri": data_uri, "format": subtype}
    except Exception:
        return None


def build_message_history(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert session chat history into the message format expected by the API.

    Each entry in ``chat_history`` contains a ``role`` and a ``content`` key.
    ``content`` may be a string (for purely textual messages) or a list of
    dictionaries (for multimodal messages). This helper simply forwards the
    stored structure without modification to preserve the context. Only user and
    assistant roles are included in the output.

    Parameters
    ----------
    chat_history: List[Dict[str, Any]]
        The Streamlit session chat history.

    Returns
    -------
    List[Dict[str, Any]]
        Messages ready for use with the watsonx.ai API.
    """
    messages: List[Dict[str, Any]] = []
    for entry in chat_history:
        if entry.get("role") in {"user", "assistant"}:
            content = entry["content"]
            if isinstance(content, list):
                # For multimodal messages, remove image entries (handled separately)
                new_content = [part for part in content if part.get("type") != "image_url"]
                messages.append({"role": entry["role"], "content": new_content})
            else:
                messages.append({"role": entry["role"], "content": content})
    return messages


def main() -> None:
    """Entry point for the Streamlit application."""

    st.set_page_config(page_title="QA Chatbot", layout="wide")
    st.title("QA Chatbot with ACS LLMs 🤖")

    # Sidebar for credentials and settings
    with st.sidebar:
        st.header("🔧 Settings")

        # Fetch credentials from secrets or user input
        api_key = st.text_input("API Key *", type="password") or st.secrets.get("api_key", "")
        instance_id = st.text_input("instance_id")  or st.secrets.get("instance_id", "")
        username = st.text_input("username")  or st.secrets.get("username", "")
    
        project_id = (
            st.secrets.get("project_id", "")
            or st.text_input("Project ID *")
        )
        project_type = st.selectbox("Project Type", options=["Text", "Code", "Vision", "Other"])

        
        # Define available model options per project type
        model_vision_options: Dict[str, List[str]] = {
            
            "Mistral": [
                "mistralai/pixtral-12b",
                "mistralai/mistral-small-3-1-24b-instruct-2503", 
                "mistralai/mistral-medium-2505"
            ],
            "Meta (LLaMA)": [
                "meta-llama/llama-3-2-11b-vision-instruct",
                "meta-llama/llama-3-2-90b-vision-instruct",
            ],
            "IBM Granite": ["ibm/granite-vision-3-2-2b"]
        }
        model_code_options: Dict[str, List[str]] = {
            "IBM Granite": ["ibm/granite-8b-code-instruct"]
        }
        model_text_options: Dict[str, List[str]] = {
            "IBM Granite": [
                "ibm/granite-3-2-8b-instruct",
                "ibm/granite-3-8b-instruct",
                "ibm/granite-3-2b-instruct",
                "ibm/granite-3-3-8b-instruct",
                "ibm/granite-3-8b-instruct",
            ],
            "Meta (LLaMA)": [
                "meta-llama/llama-3-2-1b-instruct",
                "meta-llama/llama-3-2-3b-instruct",
                "meta-llama/llama-3-3-70b-instruct",
                "meta-llama/llama-3-405b-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            ],
            "Mistral": [
                "mistralai/mistral-large",
                "mistralai/mistral-medium-2505",
                "mistralai/mistral-small-3-1-24b-instruct-2503",
            ],
        }
        model_other_options: Dict[str, List[str]] = {
            "Google": ["google/flan-t5-xl"],
            "IBM Granite": [
                "ibm/granite-13b-instruct-v2",
                "ibm/granite-guardian-3-2b",
                "ibm/granite-guardian-3-8b",
            ],
            "Meta (LLaMA)": [
                "meta-llama/llama-2-13b-chat",
                "meta-llama/llama-guard-3-11b-vision"
            ]
        }

        # Choose the appropriate option set based on project type
        if project_type == "Vision":
            current_options = model_vision_options
        elif project_type == "Code":
            current_options = model_code_options
        elif project_type == "Other":
            current_options = model_other_options
        else:
            current_options = model_text_options

        model_provider = st.selectbox("Model Provider", options=list(current_options.keys()))
        model_id = st.selectbox("Model ID *", options=current_options.get(model_provider, []))

        url = st.secrets.get("url", "") or st.text_input("Endpoint URL *")

        # Additional input fields for vision models
        uploaded_image = None  # type: Optional[Any]
        remote_image_url: str = ""
        if project_type == "Vision":
            st.markdown("### 🖼️ Image Input (for Vision models)")
            # Let the user choose to upload a file or provide a URL
            input_method = st.radio(
                "Select image source",
                options=["None", "Upload"],
                horizontal=True,
            )
            if input_method == "Upload":
                uploaded_image = st.file_uploader(
                    "Upload an image (PNG/JPEG)",
                    type=["png", "jpg", "jpeg"],
                    key="vision_upload",
                )
          

    # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    # Display past messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # If the content is a list, it may contain text and/or image parts
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "text":
                        st.markdown(part.get("text", ""))
            else:
                # Simple text content
                st.markdown(message.get("content", ""))

    # Input field for new user message
    user_input = st.chat_input("Ask your question here...")

    if user_input:
        # Enforce required fields
        if not all([api_key, project_id, model_id]):
            st.error("❗ Please fill in all required fields in the sidebar (marked with *)")
            st.stop()

        # Build the current user message payload depending on project type
        current_content: Union[str, List[Dict[str, Any]]]
        vision_image_entry: Optional[Dict[str, Any]] = None
        # Only one image is sent: uploaded image takes priority, else remote URL
        if project_type == "Vision":
            if uploaded_image:
                image_info = encode_image_to_base64(uploaded_image)
                if image_info:
                    vision_image_entry = {
                        "type": "image_url",
                        "image_url": {"url": image_info["data_uri"]},
                    }
            elif remote_image_url:
                if remote_image_url.strip():
                    vision_image_entry = {
                        "type": "image_url",
                        "image_url": {"url": remote_image_url},
                    }

        # Construct content for the current user message
        if project_type == "Vision" and vision_image_entry:
            # Only one image entry allowed: include text and one image
            current_content = [
                {"type": "text", "text": user_input},
                vision_image_entry
            ]
        else:
            # Purely textual message
            current_content = user_input

        # Append user message to history for display
        st.session_state.chat_history.append({"role": "user", "content": current_content})
        with st.chat_message("user"):
            if isinstance(current_content, list):
                for part in current_content:
                    if part["type"] == "text":
                        st.markdown(part["text"])
                    elif part["type"] == "image_url":
                        # Display the image back to the user
                        img_url = part["image_url"]["url"]
                        if img_url.startswith("data:image"):
                            _, b64data = img_url.split(",", 1)
                            st.image(base64.b64decode(b64data), caption="Your uploaded image")
                        else:
                            st.image(img_url, caption="Image from URL")
            else:
                st.markdown(current_content)

        # Prepare credentials and model
        try:
            if !instance_id or !username :
                credentials = Credentials(url=url, api_key=api_key)
                # Use TextChatParameters when available for temperature control on text models
                params_obj = TextChatParameters(temperature=1) if TextChatParameters else None
                model = ModelInference(
                    model_id=model_id,
                    credentials=credentials,
                    project_id=project_id,
                    params=params_obj if project_type != "Vision" else None,
                )
            else :
                credentials = Credentials(url=url, api_key=api_key, instance_id=instance_id, username=username, version = "5.1", verify=False)
                # Use TextChatParameters when available for temperature control on text models
                params_obj = TextChatParameters(temperature=1) if TextChatParameters else None
                model = ModelInference(
                    model_id=model_id,
                    credentials=credentials,
                    project_id=project_id,
                    params=params_obj if project_type != "Vision" else None,
                )

            # Build message history for the API call
            if project_type == "Vision":
                # Remove all previous images from history, only send latest image
                messages = build_message_history(st.session_state.chat_history)
                # If the current user input has an image, add it to the last user message
                if vision_image_entry:
                    # Find the last user message and add the image
                    for i in range(len(messages)-1, -1, -1):
                        if messages[i]["role"] == "user":
                            # If the content is a list, append image; else, make it a list
                            if isinstance(messages[i]["content"], list):
                                messages[i]["content"].append(vision_image_entry)
                            else:
                                messages[i]["content"] = [
                                    {"type": "text", "text": messages[i]["content"]},
                                    vision_image_entry
                                ]
                            break
            else:
                messages = build_message_history(st.session_state.chat_history)

            start_time = time.time()

            # Invoke the model with the constructed messages. For vision models we pass
            # no additional params, but the API may require specifying input_type
            # as "chat"; leaving params=None uses defaults.
            response = model.chat(messages=messages)

            end_time = time.time()

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                content = response

            elapsed = end_time - start_time
            # Estimate tokens as whitespace‑separated words
            tokens = len(str(content).split())
            speed = tokens / elapsed if elapsed > 0 else 0

            # Append assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "content": content})
            with st.chat_message("assistant"):
                st.markdown(content)

            st.info(f"⏱ Time: {elapsed:.2f}s | 🧮 Tokens: {tokens} | ⚡ Speed: {speed:.2f} tokens/s")

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
