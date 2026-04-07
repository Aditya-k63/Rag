import streamlit as st
import requests

API_URL = "http://localhost:8000"

# --- Page Config ---
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="",
    layout="wide"
)

st.title("RAG Assistant")
st.caption("Upload any PDF and ask questions about it")

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Upload to Knowledge Base", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Uploaded successfully!")
                        st.info(f" Chunks inserted: {data['chunks_inserted']}")
                    else:
                        st.error(f"Upload failed: {response.json()['detail']}")
                except Exception as e:
                    st.error(f" Error: {str(e)}")

    st.divider()

    # Health check
    st.header("🔌 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("API is running ")
        else:
            st.error("API error ")
    except:
        st.error("API not reachable ")

    st.divider()
    st.caption("Built with FastAPI + pgvector + Groq")

# --- Main: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": 5}
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    chunks = data["chunks_used"]

                    st.markdown(answer)
                    st.caption(f"Based on {chunks} relevant chunks")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    st.error("Failed to get answer from API")

            except Exception as e:
                st.error(f"Error: {str(e)}")