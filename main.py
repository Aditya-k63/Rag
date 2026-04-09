import os
import io
import json
import logging
import psycopg2
import numpy as np
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Connection Pool ---
connection_pool = pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

def get_connection():
    conn = connection_pool.getconn()
    register_vector(conn)
    return conn

def release_connection(conn):
    connection_pool.putconn(conn)

# --- Query Cache ---
query_cache = {}
CACHE_MAX_SIZE = 100

def get_cached_answer(question: str, top_k: int):
    key = f"{question.strip().lower()}::{top_k}"
    return query_cache.get(key)

def set_cached_answer(question: str, top_k: int, result: dict):
    key = f"{question.strip().lower()}::{top_k}"
    if len(query_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(query_cache))
        del query_cache[oldest_key]
    query_cache[key] = result

app = FastAPI(title="RAG API", description="Hybrid RAG with pgvector + Groq")

# --- Auth ---
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it as header: X-API-Key"
        )
    return key

# --- Models ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Constants ---
MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

# --- Schemas ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int

class UploadResponse(BaseModel):
    filename: str
    chunks_inserted: int
    message: str

class EvaluatedQueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    faithfulness: float
    answer_relevance: float
    context_precision: float
    overall_score: float

# --- DB Helpers ---
def is_already_ingested(filename: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_sections WHERE meta->>'source' = %s", (filename,))
        count = cur.fetchone()[0]
        cur.close()
        return count > 0
    finally:
        release_connection(conn)

def fetch_all_chunks():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, content FROM document_sections;")
        results = cur.fetchall()
        cur.close()
        return results
    finally:
        release_connection(conn)

def insert_chunks(chunks: list, filename: str):
    conn = get_connection()
    inserted = 0
    try:
        cur = conn.cursor()
        for i, chunk in enumerate(chunks):
            try:
                embedding = embedder.encode(chunk).tolist()
                metadata = {"source": filename, "chunk_index": i}
                cur.execute(
                    "INSERT INTO document_sections (content, meta, embedding) VALUES (%s, %s, %s)",
                    (chunk, json.dumps(metadata), embedding)
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Failed to insert chunk {i} from '{filename}': {e}")
                conn.rollback()
                raise RuntimeError(f"Insert failed at chunk {i}: {e}")
        conn.commit()
        cur.close()
        logger.info(f"Successfully inserted {inserted} chunks from '{filename}'")
    finally:
        release_connection(conn)

def log_evaluation(question: str, answer: str, metrics: dict):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                id SERIAL PRIMARY KEY,
                question TEXT,
                answer TEXT,
                faithfulness FLOAT,
                answer_relevance FLOAT,
                context_precision FLOAT,
                overall_score FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            INSERT INTO rag_evaluations
            (question, answer, faithfulness, answer_relevance, context_precision, overall_score)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            question, answer,
            metrics["faithfulness"],
            metrics["answer_relevance"],
            metrics["context_precision"],
            metrics["overall_score"]
        ))
        conn.commit()
        cur.close()
    finally:
        release_connection(conn)

# --- Search ---
def bm25_search(query: str, all_chunks: list, top_k: int = 10):
    contents = [chunk[1] for chunk in all_chunks]
    tokenized_corpus = [doc.lower().split() for doc in contents]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(all_chunks[i][0], all_chunks[i][1], scores[i]) for i in top_indices]

def vector_search(query: str, top_k: int = 10):
    query_embedding = embedder.encode(query).tolist()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_sections
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embedding, top_k))
        results = cur.fetchall()
        cur.close()
        return results
    finally:
        release_connection(conn)

def reciprocal_rank_fusion(bm25_results: list, vector_results: list, k: int = 60):
    fused_scores = {}
    contents_map = {}
    for rank, (doc_id, content, score) in enumerate(bm25_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        contents_map[doc_id] = content
    for rank, (doc_id, content, score) in enumerate(vector_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        contents_map[doc_id] = content
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(doc_id, contents_map[doc_id], fused_scores[doc_id]) for doc_id in sorted_ids]

def rerank(query: str, candidates: list, top_k: int = 3):
    pairs = [(query, content) for _, content, _ in candidates]
    scores = reranker.predict(pairs)
    scored = [(candidates[i][0], candidates[i][1], float(scores[i])) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

def retrieve_chunks(query: str, top_k: int = 5):
    all_chunks = fetch_all_chunks()
    bm25_results = bm25_search(query, all_chunks, top_k=10)
    vector_results = vector_search(query, top_k=10)
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    reranked = rerank(query, fused[:20], top_k=top_k)
    return [(content, score) for _, content, score in reranked]

# --- Generation ---
def generate_answer(query: str, chunks: list):
    context = "\n\n".join([f"[Score: {score:.3f}]\n{content}" for content, score in chunks])
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
- Be specific and detailed in your answer.
- If the answer is partially in the context, share what you know and flag what's missing.
- If the answer is completely absent from the context, say exactly: "I don't have enough information to answer that."
- Never make up facts not present in the context.

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

# --- PDF Processing ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

# --- Evaluation ---
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_faithfulness(answer: str, chunks: list) -> float:
    no_info_phrases = ["i don't have enough information", "i cannot answer", "not in the context"]
    if any(phrase in answer.lower() for phrase in no_info_phrases):
        return 1.0
    context = " ".join(chunks)
    return round(cosine_similarity(embedder.encode(answer), embedder.encode(context)), 4)

def compute_answer_relevance(question: str, answer: str) -> float:
    return round(cosine_similarity(embedder.encode(question), embedder.encode(answer)), 4)

def compute_context_precision(question: str, chunks: list, threshold: float = 0.4) -> float:
    q_emb = embedder.encode(question)
    relevant = sum(1 for chunk in chunks if cosine_similarity(q_emb, embedder.encode(chunk)) >= threshold)
    return round(relevant / len(chunks), 4) if chunks else 0.0

def evaluate(question: str, answer: str, chunks: list) -> dict:
    faithfulness = compute_faithfulness(answer, chunks)
    relevance = compute_answer_relevance(question, answer)
    precision = compute_context_precision(question, chunks)
    overall = round((faithfulness * 0.4) + (relevance * 0.4) + (precision * 0.2), 4)
    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": precision,
        "overall_score": overall
    }

# --- Routes ---
@app.get("/")
def root():
    return {"status": "Hybrid RAG API is running ✅"}

@app.get("/health")
async def health():
    conn = get_connection()
    try:
        conn.cursor().execute("SELECT 1")
        return {"status": "healthy", "db": "connected", "cache_size": len(query_cache)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")
    finally:
        release_connection(conn)

@app.get("/documents", dependencies=[Depends(verify_api_key)])
async def list_documents():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT meta->>'source' AS source, COUNT(*) AS chunks
            FROM document_sections
            WHERE meta IS NOT NULL
            GROUP BY meta->>'source'
            ORDER BY source;
        """)
        rows = cur.fetchall()
        cur.close()
        return {"documents": [{"filename": r[0], "chunks": r[1]} for r in rows]}
    finally:
        release_connection(conn)

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cached = get_cached_answer(request.question, request.top_k)
    if cached:
        logger.info(f"Cache hit: {request.question}")
        return QueryResponse(**cached)

    logger.info(f"Query received: {request.question}")
    chunks = retrieve_chunks(request.question, request.top_k)
    answer = generate_answer(request.question, chunks)
    logger.info(f"Query answered with {len(chunks)} chunks")

    result = {"question": request.question, "answer": answer, "chunks_used": len(chunks)}
    set_cached_answer(request.question, request.top_k, result)

    return QueryResponse(**result)

@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_pdf(file: UploadFile = File(...), force: bool = False):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10MB")

    if is_already_ingested(file.filename) and not force:
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' already ingested. Use ?force=true to re-upload."
        )

    if force:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM document_sections WHERE meta->>'source' = %s", (file.filename,))
            conn.commit()
            cur.close()
            logger.info(f"Force re-upload: deleted old chunks for '{file.filename}'")
        finally:
            release_connection(conn)

    try:
        text = extract_text_from_pdf(contents)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        chunks = chunk_text(text)
        if len(chunks) > 500:
            raise HTTPException(status_code=400, detail=f"PDF too large — {len(chunks)} chunks, max is 500")

        insert_chunks(chunks, file.filename)
        query_cache.clear()  # clear cache after new PDF added
        logger.info(f"Upload complete: {file.filename}. Cache cleared.")

        return UploadResponse(
            filename=file.filename,
            chunks_inserted=len(chunks),
            message=f"Successfully ingested '{file.filename}' into the knowledge base"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/evaluate-query", response_model=EvaluatedQueryResponse, dependencies=[Depends(verify_api_key)])
async def evaluate_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve_chunks(request.question, request.top_k)
    answer = generate_answer(request.question, chunks)

    chunks_text = [content for content, _ in chunks]
    metrics = evaluate(request.question, answer, chunks_text)
    log_evaluation(request.question, answer, metrics)

    return EvaluatedQueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=len(chunks),
        **metrics
    )

@app.post("/cache/clear", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    count = len(query_cache)
    query_cache.clear()
    logger.info(f"Cache cleared — {count} entries removed")
    return {"message": f"Cache cleared. {count} entries removed."}