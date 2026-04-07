import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def retrieve_chunks(query, top_k=3):
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

def generate_answer(query, chunks):
    # Build context from retrieved chunks
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

def rag_query(query):
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    # Step 1 — Retrieve
    chunks = retrieve_chunks(query, top_k=3)
    print(f"Retrieved {len(chunks)} chunks")
    
    # Step 2 — Generate
    answer = generate_answer(query, chunks)
    print(f"\nAnswer:\n{answer}")
    
    return answer

if __name__ == "__main__":
    # Test it
    rag_query("What is a Python list comprehension?")
    rag_query("How does a for loop work in Python?")