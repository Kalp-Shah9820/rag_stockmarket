"""
Hybrid Retriever — combines pgvector cosine similarity with PostgreSQL
full-text search, merges results via weighted Reciprocal Rank Fusion,
and deduplicates.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from src.config import settings
from src.models import DocumentChunk, RetrievedChunk
from src.embeddings import embed_query
from src.database import vector_search, fts_search


def _row_to_chunk(row: dict) -> DocumentChunk:
    """Convert a DB row dict into a DocumentChunk."""
    return DocumentChunk(
        id=row["id"],
        doc_id=row["doc_id"],
        title=row["title"],
        content=row["content"],
        chunk_index=row["chunk_index"],
        metadata=row.get("metadata", {}),
    )


def _reciprocal_rank(rank: int, k: int = 60) -> float:
    """RRF score: 1 / (k + rank)."""
    return 1.0 / (k + rank)


def hybrid_retrieve(query: str) -> List[RetrievedChunk]:
    """
    Perform hybrid retrieval:
    1. Vector search (cosine similarity via pgvector)
    2. Full-text search (BM25-style via PostgreSQL tsvector)
    3. Merge with weighted Reciprocal Rank Fusion
    4. Deduplicate and return top-k
    """
    cfg = settings.retrieval

    # ── Step 1: Vector search ────────────────────────────────
    query_emb = embed_query(query)
    vec_results = vector_search(query_emb, top_k=cfg.vector_top_k)
    logger.debug(f"Vector search returned {len(vec_results)} results")

    # ── Step 2: Full-text search ─────────────────────────────
    fts_results = fts_search(query, top_k=cfg.fts_top_k)
    logger.debug(f"FTS returned {len(fts_results)} results")

    # ── Step 3: Weighted RRF merge ───────────────────────────
    scored: dict[str, RetrievedChunk] = {}

    # Score vector results
    for rank, row in enumerate(vec_results, start=1):
        chunk_id = row["id"]
        rrf = _reciprocal_rank(rank) * cfg.vector_weight
        if chunk_id not in scored:
            scored[chunk_id] = RetrievedChunk(
                chunk=_row_to_chunk(row),
                vector_score=float(row.get("score", 0)),
            )
        scored[chunk_id].combined_score += rrf

    # Score FTS results
    for rank, row in enumerate(fts_results, start=1):
        chunk_id = row["id"]
        rrf = _reciprocal_rank(rank) * cfg.fts_weight
        if chunk_id not in scored:
            scored[chunk_id] = RetrievedChunk(
                chunk=_row_to_chunk(row),
            )
        scored[chunk_id].fts_score = float(row.get("score", 0))
        scored[chunk_id].combined_score += rrf

    # ── Step 4: Sort & return top-k ──────────────────────────
    merged = sorted(scored.values(), key=lambda x: x.combined_score, reverse=True)
    results = merged[: cfg.final_top_k]

    logger.info(
        f"🔍 Hybrid retrieval: {len(vec_results)} vec + {len(fts_results)} fts "
        f"→ {len(scored)} unique → top {len(results)}"
    )
    return results
