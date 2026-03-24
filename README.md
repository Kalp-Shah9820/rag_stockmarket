# RAG Stock Market Intelligence

> Production-grade Hybrid RAG system for financial news analysis with citation grounding.

## Architecture

```
User Query → FastAPI → LangGraph Agent → Query Rewriter
    → Hybrid Retrieval (pgvector + FTS) → Relevance Check
    → Cross-Encoder Re-ranking → LLM Generation → Answer + Citations
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 2. Setup

```bash
# Clone & enter project
cd RAG

# Copy env file and set your OpenAI key
cp .env.example .env
# Edit .env → set OPENAI_API_KEY

# Start PostgreSQL + pgvector
docker-compose up -d

# Install dependencies
pip install -r requirements.txt
```

### 3. Ingest Data

```bash
python -m scripts.ingest
```

This will:
- Download the XA7 Stock Market News dataset from HuggingFace
- Chunk documents (700 chars, 100 overlap)
- Generate embeddings (all-MiniLM-L6-v2)
- Store in PostgreSQL with pgvector

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

API available at http://localhost:8000
- `POST /ask` — Submit a question
- `GET /health` — Health check
- `GET /stats` — Database stats

### 5. Run the UI

```bash
streamlit run ui/app.py
```

UI available at http://localhost:8501

### 6. Evaluate (RAGAS)

```bash
python -m src.evaluation
```

## Configuration

All settings are in `config/settings.yaml`:
- Database, embedding model, chunking params
- Retrieval weights, re-ranker model
- LLM model, system prompt, guardrails
- Evaluation metrics

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Embedding | all-MiniLM-L6-v2 (384d) |
| Vector DB | PostgreSQL + pgvector |
| FTS | PostgreSQL tsvector |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | OpenAI GPT-4o-mini |
| Agent | LangGraph state machine |
| API | FastAPI |
| UI | Streamlit |
| Eval | RAGAS |

## Project Structure

```
RAG/
├── config/settings.yaml    — Master config
├── src/
│   ├── config.py           — Config loader (Pydantic)
│   ├── models.py           — Data models
│   ├── database.py         — PostgreSQL + pgvector
│   ├── ingestion.py        — Data pipeline
│   ├── embeddings.py       — Embedding model
│   ├── retriever.py        — Hybrid retrieval + RRF
│   ├── reranker.py         — Cross-encoder
│   ├── generator.py        — LLM answer gen
│   ├── guardrails.py       — Safety checks
│   ├── graph.py            — LangGraph agent
│   └── evaluation.py       — RAGAS evaluation
├── api/main.py             — FastAPI app
├── ui/app.py               — Streamlit UI
├── scripts/ingest.py       — CLI ingestion
├── tests/                  — Unit tests
├── docker-compose.yml      — PostgreSQL container
└── Dockerfile              — App container
```

