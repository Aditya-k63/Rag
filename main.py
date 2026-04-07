import os
import io
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI(title="RAG API", description="Retrieval Augmented Generation with pgvector + Groq")

# Initialize models once at startup
embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

# --- Request/Response Schemas ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int

class UploadResponse(BaseModel):
    filename: str
    chunks_inserted: int
    message: str

# --- DB Connection ---
def get_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    register_vector(conn)
    return conn

# --- Retrieve Chunks ---
def retrieve_chunks(query: str, top_k: int = 5):
    query_embedding = embedder.encode(query).tolist()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity
        FROM document_sections
        ORDER BY similarity DESC
        LIMIT %s;
    """, (query_embedding, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results

# --- Generate Answer ---
def generate_answer(query: str, chunks: list):
    context = "\n\n".join([f"[Score: {score:.3f}]\n{content}"
                           for content, score in chunks])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )

    return response.choices[0].message.content

# --- Extract text from PDF ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# --- Chunk text ---
def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

# --- Insert chunks into pgvector ---
def insert_chunks(chunks: list, filename: str):
    conn = get_connection()
    cur = conn.cursor()

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        metadata = {"source": filename, "chunk_index": i}

        try:
            cur.execute(
                "INSERT INTO document_sections (content, meta, embedding) VALUES (%s, %s, %s)",
                (chunk, json.dumps(metadata), embedding)
            )
            print(f"Inserted chunk {i}")
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
            conn.rollback()
            raise

    conn.commit()
    cur.close()
    conn.close()
# --- Routes ---
@app.get("/")
def root():
    return {"status": "RAG API is running ✅"}

@app.get("/health")
def health():
    try:
        conn = get_connection()
        conn.close()
        return {"status": "healthy", "db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve_chunks(request.question, request.top_k)
    answer = generate_answer(request.question, chunks)

    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=len(chunks)
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10MB")

    try:
        # Extract text
        text = extract_text_from_pdf(contents)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Chunk text
        chunks = chunk_text(text)
        if len(chunks) > 500:
            raise HTTPException(status_code=400, detail=f"PDF too large — generated {len(chunks)} chunks, max is 500")

        # Insert into pgvector
        insert_chunks(chunks, file.filename)

        return UploadResponse(
            filename=file.filename,
            chunks_inserted=len(chunks),
            message=f"Successfully ingested '{file.filename}' into the knowledge base"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")