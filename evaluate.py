import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
import json

load_dotenv()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- Cosine Similarity Helper ----
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---- Metric 1: Faithfulness ----
# Checks if the answer is grounded in the context (hallucination detection)
def compute_faithfulness(answer: str, chunks: list[str]) -> float:
    context = " ".join(chunks)
    answer_emb = embedder.encode(answer)
    context_emb = embedder.encode(context)
    score = cosine_similarity(answer_emb, context_emb)
    return round(score, 4)

# ---- Metric 2: Answer Relevance ----
# Checks if the answer actually addresses the question
def compute_answer_relevance(question: str, answer: str) -> float:
    q_emb = embedder.encode(question)
    a_emb = embedder.encode(answer)
    score = cosine_similarity(q_emb, a_emb)
    return round(score, 4)

# ---- Metric 3: Context Precision ----
# Checks what fraction of retrieved chunks are actually relevant to the question
def compute_context_precision(question: str, chunks: list[str], threshold: float = 0.4) -> float:
    q_emb = embedder.encode(question)
    relevant = 0
    for chunk in chunks:
        c_emb = embedder.encode(chunk)
        sim = cosine_similarity(q_emb, c_emb)
        if sim >= threshold:
            relevant += 1
    return round(relevant / len(chunks), 4) if chunks else 0.0

# ---- Full Evaluation ----
def evaluate(question: str, answer: str, chunks: list[str]) -> dict:
    faithfulness = compute_faithfulness(answer, chunks)
    relevance = compute_answer_relevance(question, answer)
    precision = compute_context_precision(question, chunks)

    # Overall score — weighted average
    overall = round((faithfulness * 0.4) + (relevance * 0.4) + (precision * 0.2), 4)

    return {
        "faithfulness": faithfulness,       # Is answer grounded in context?
        "answer_relevance": relevance,      # Does answer address the question?
        "context_precision": precision,     # Are retrieved chunks useful?
        "overall_score": overall
    }

# ---- Log to DB ----
def log_evaluation(question: str, answer: str, metrics: dict):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
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
    conn.close()

# ---- Test Suite ----
TEST_QUESTIONS = [
    "What is a Python list comprehension?",
    "What were the three main reasons for the founding of the Georgia colony?",
    "How does a for loop work in Python?",
    "What is the purpose of the continue statement?",
]

if __name__ == "__main__":
    # Import your RAG pipeline
    from main import retrieve_chunks, generate_answer

    print("=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)

    all_scores = []

    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")

        # Run RAG pipeline
        raw_chunks = retrieve_chunks(question, top_k=5)
        chunks_text = [content for content, _ in raw_chunks]
        answer = generate_answer(question, raw_chunks)

        # Evaluate
        metrics = evaluate(question, answer, chunks_text)

        # Log to DB
        log_evaluation(question, answer, metrics)

        # Print results
        print(f"Answer: {answer[:120]}...")
        print(f"  Faithfulness:      {metrics['faithfulness']:.4f}  (hallucination check)")
        print(f"  Answer Relevance:  {metrics['answer_relevance']:.4f}  (addresses question?)")
        print(f"  Context Precision: {metrics['context_precision']:.4f}  (chunks useful?)")
        print(f"  Overall Score:     {metrics['overall_score']:.4f}")

        all_scores.append(metrics["overall_score"])

    print("\n" + "=" * 60)
    print(f"AVERAGE OVERALL SCORE: {np.mean(all_scores):.4f}")
    print("=" * 60)