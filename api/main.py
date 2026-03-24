"""
FastAPI Application — REST API for the RAG Stock Market system.

Endpoints:
  POST /ask         — Submit a query and get a grounded answer
  GET  /health      — System health check
  GET  /stats       — Database statistics
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import settings
from src.models import QueryInput, GeneratedAnswer, HealthResponse
from src.database import check_connection, get_chunk_count, init_database
from src.graph import run_agent


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("🚀 Starting RAG Stock Market API")
    try:
        init_database()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.warning(f"⚠️  Database init skipped: {e}")
    yield
    logger.info("👋 Shutting down")


# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Production-grade Hybrid RAG for Stock Market News",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────

@app.post("/ask", response_model=GeneratedAnswer)
async def ask(payload: QueryInput):
    """
    Submit a stock-market question and receive a grounded answer
    with citations.
    """
    try:
        t0 = time.perf_counter()
        answer = run_agent(payload.query)
        answer.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return answer
    except Exception as e:
        logger.error(f"❌ /ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app.version,
        db_connected=check_connection(),
    )


@app.get("/stats")
async def stats():
    """Return database statistics."""
    try:
        count = get_chunk_count()
        return {
            "total_chunks": count,
            "database": settings.database.name,
            "embedding_model": settings.embedding.model_name,
            "reranker_model": settings.reranker.model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run with: uvicorn api.main:app --reload ──────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
