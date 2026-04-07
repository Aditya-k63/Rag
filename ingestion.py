import os
import pickle
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

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

def ingest_data():
    try:
        with open("processed_data.pkl", "rb") as f:
            data = pickle.load(f)
        print("Pickle file loaded successfully.")
    except FileNotFoundError:
        print("Error: processed_data.pkl not found!")
        return

    try:
        conn = get_connection()
        cur = conn.cursor()
        print("Connected to PostgreSQL.")

        print(f"Inserting {len(data['chunks'])} chunks...")
        for i in range(len(data['chunks'])):
            content = data['chunks'][i]
            metadata = json.dumps(data['metadata'][i])
            embedding = data['embeddings'][i].tolist()  #  convert numpy to list

            cur.execute(
                "INSERT INTO document_sections (content, metapip install groq, embedding) VALUES (%s, %s, %s)",
                (content, metadata, embedding)
            )

        conn.commit()
        cur.close()
        conn.close()
        print("Database Hydration Complete!")

    except Exception as e:
        print(f"Database Error: {e}")

def sanity_check():
    query = "What is a Python list comprehension?"
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()  #convert to list

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_sections
            ORDER BY similarity DESC
            LIMIT 3;
        """, (query_embedding,))

        rows = cur.fetchall()
        if not rows:
            print("No results found — is the table empty?")
        for row in rows:
            print(f"\nScore: {row[1]:.4f}\nContent: {row[0][:200]}...")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Sanity Check Error: {e}")

if __name__ == "__main__":
    ingest_data()
    sanity_check()