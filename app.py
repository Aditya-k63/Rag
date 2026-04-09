import streamlit as st
import requests

API_URL = "http://localhost:8000"
API_KEY = "rag-secret-2026"  # must match your .env
HEADERS = {"X-API-Key": API_KEY}

# --- Page Config ---
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title(" RAG Assistant")
st.caption("Upload any PDF and ask questions about it")

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# --- Sidebar ---
with st.sidebar:
    st.header(" Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Auto-upload as soon as a new file is selected
    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded:
        with st.spinner(f"Auto-uploading '{uploaded_file.name}'..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    headers=HEADERS
                )

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.last_uploaded = uploaded_file.name
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "chunks": data["chunks_inserted"]
                    })
                    st.success(f"'{uploaded_file.name}' uploaded!")
                    st.info(f"{data['chunks_inserted']} chunks added to knowledge base")

                elif response.status_code == 409:
                    # Already ingested
                    st.session_state.last_uploaded = uploaded_file.name
                    st.warning(f" '{uploaded_file.name}' was already in the knowledge base.")

                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")

            except Exception as e:
                st.error(f" Error: {str(e)}")

    st.divider()

 

    # --- API Status ---
    st.header("🔌 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            data = health.json()
            st.success("API is running ")
            st.caption(f"Cache size: {data.get('cache_size', 0)} entries")
        else:
            st.error("API error ")
    except:
        st.error("API not reachable ")

    st.divider()

    # --- Chat History Controls ---
    st.header(" Chat History")
    st.caption(f"{len(st.session_state.messages)} messages in this session")

    if st.button(" Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with FastAPI + pgvector + Groq")

# --- Main: Chat Interface ---

# Display full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show chunk count for assistant messages if available
        if message["role"] == "assistant" and "chunks_used" in message:
            st.caption(f"📚 Based on {message['chunks_used']} relevant chunks")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": 5},
                    headers=HEADERS
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    chunks_used = data["chunks_used"]

                    st.markdown(answer)
                    st.caption(f"Based on {chunks_used} relevant chunks")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "chunks_used": chunks_used
                    })

                elif response.status_code == 401:
                    st.error(" API key invalid. Check your API_KEY in app.py")

                else:
                    st.error(f"Failed: {response.json().get('detail', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error: {str(e)}")