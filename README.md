# 🧠 RAG Assistant

I built this because I wanted to understand how vector search and LLMs actually work together — not just read about it. You upload any PDF, ask questions about it, and get answers that are grounded in the actual document. No hallucinations, no guessing.

Tested on history documents, research papers, and textbooks.

---

## The problem it solves

Most LLMs hallucinate when you ask about specific documents. This system doesn't — it finds the most relevant chunks from your PDF first, then uses the LLM only to form a clean answer from those chunks. If the answer isn't in the document, it says so.

---

## How it works

```
You upload a PDF
      ↓
Text is extracted, split into chunks, and embedded as 384-dim vectors
      ↓
Stored in PostgreSQL with pgvector
      ↓
You ask a question
      ↓
BM25 (keyword) + vector search run in parallel
      ↓
Results merged with Reciprocal Rank Fusion
      ↓
Cross-encoder reranker picks the best chunks
      ↓
Groq LLM generates a grounded answer
```

---

## Tech stack

| Layer | Tool |
|-------|------|
| Vector DB | pgvector (PostgreSQL 18) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Retrieval | BM25 + vector search + RRF |
| Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Groq API (llama-3.1-8b-instant) |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## Why hybrid search?

Most RAG tutorials just do a single vector search. That works okay, but it misses a lot — especially in history or technical documents full of proper nouns, dates, and exact terms.

This system combines three things:

- **BM25** catches exact keyword matches — names, dates, specific terms
- **Vector search** catches semantic meaning — even when the wording is different
- **Cross-encoder reranking** scores each (question, chunk) pair together, which is significantly more accurate than embedding similarity alone

The result is noticeably better answers on domain-specific documents.

---

## Getting started

### What you need

- Python 3.11+
- PostgreSQL 18
- pgvector installed
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Install pgvector on Windows

Download the zip for your PostgreSQL version from [pgvector releases](https://github.com/pgvector/pgvector/releases) and copy:

```
vector.dll          →  C:\Program Files\PostgreSQL\18\lib\
vector.control
vector--*.sql       →  C:\Program Files\PostgreSQL\18\share\extension\
```

Then in psql or pgAdmin:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Clone and install

```bash
git clone https://github.com/Aditya-k63/Rag.git
cd Rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Configure your environment

Create a `.env` file in the root folder:

```env
DB_NAME=your_database_name
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=your_groq_api_key
API_KEY=your_chosen_secret_key
```

### Create the database table

```sql
CREATE TABLE document_sections (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    meta JSONB,
    embedding VECTOR(384)
);

CREATE INDEX ON document_sections
USING hnsw (embedding vector_cosine_ops);
```

### Run

Open two terminals:

```bash
# Terminal 1 — backend
uvicorn main:app --reload

# Terminal 2 — frontend
streamlit run app.py
```

- Swagger docs → `http://localhost:8000/docs`
- Chat UI → `http://localhost:8501`

---

## API reference

All routes except `/health` require the header `X-API-Key: your_key`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if API and DB are up |
| GET | `/documents` | List all ingested PDFs |
| POST | `/upload` | Upload a PDF (max 10MB) |
| POST | `/query` | Ask a question |
| POST | `/evaluate-query` | Ask + get quality scores |
| POST | `/cache/clear` | Clear the query cache |

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"question": "What caused World War I?", "top_k": 5}'
```

---

## Evaluation

```bash
python evaluate.py
```

Measures three things per answer:

- **Faithfulness** — is the answer grounded in the retrieved chunks?
- **Relevance** — does it actually answer the question?
- **Context precision** — were the retrieved chunks useful?

Results are logged to the `rag_evaluations` table in PostgreSQL.

Average overall score after tuning: **0.75 / 1.0**

---

## What's next

- [ ] Docker setup for one-command deployment
- [ ] Filter queries by specific PDF source
- [ ] Delete endpoint to remove documents from the knowledge base
- [ ] Support for DOCX and plain text files

---

## Requirements

```
fastapi
uvicorn
psycopg2-binary
pgvector
python-dotenv
sentence-transformers
groq
pypdf
langchain-text-splitters
python-multipart
streamlit
requests
rank-bm25
numpy
```

---

> Built by [Aditya Kumar](https://github.com/Aditya-k63) as part of an ML portfolio project.
