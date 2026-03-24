"""
Unit tests for the retriever module.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models import DocumentChunk, RetrievedChunk


# ── Test RRF scoring ────────────────────────────────────────
def test_reciprocal_rank():
    from src.retriever import _reciprocal_rank

    assert _reciprocal_rank(1) == pytest.approx(1 / 61)
    assert _reciprocal_rank(2) == pytest.approx(1 / 62)
    assert _reciprocal_rank(1, k=0) == pytest.approx(1.0)


# ── Test row-to-chunk conversion ────────────────────────────
def test_row_to_chunk():
    from src.retriever import _row_to_chunk

    row = {
        "id": "test-id",
        "doc_id": "doc-1",
        "title": "Test Title",
        "content": "Test content",
        "chunk_index": 0,
        "metadata": {"source": "test"},
    }
    chunk = _row_to_chunk(row)
    assert isinstance(chunk, DocumentChunk)
    assert chunk.id == "test-id"
    assert chunk.title == "Test Title"


# ── Test hybrid retrieval (mocked DB) ───────────────────────
@patch("src.retriever.vector_search")
@patch("src.retriever.fts_search")
@patch("src.retriever.embed_query")
def test_hybrid_retrieve_merges_results(
    mock_embed, mock_fts, mock_vector
):
    from src.retriever import hybrid_retrieve

    mock_embed.return_value = [0.1] * 384

    # Vector returns 2 results
    mock_vector.return_value = [
        {"id": "a", "doc_id": "d1", "title": "T1", "content": "C1",
         "chunk_index": 0, "metadata": {}, "score": 0.9},
        {"id": "b", "doc_id": "d2", "title": "T2", "content": "C2",
         "chunk_index": 0, "metadata": {}, "score": 0.7},
    ]

    # FTS returns 2 results (one overlapping)
    mock_fts.return_value = [
        {"id": "b", "doc_id": "d2", "title": "T2", "content": "C2",
         "chunk_index": 0, "metadata": {}, "score": 5.0},
        {"id": "c", "doc_id": "d3", "title": "T3", "content": "C3",
         "chunk_index": 0, "metadata": {}, "score": 3.0},
    ]

    results = hybrid_retrieve("test query")

    # Should have 3 unique chunks (a, b, c)
    assert len(results) <= 5
    assert all(isinstance(r, RetrievedChunk) for r in results)

    # Chunk "b" should have the highest combined score (appears in both)
    ids = [r.chunk.id for r in results]
    assert "b" in ids


# ── Test empty results ──────────────────────────────────────
@patch("src.retriever.vector_search")
@patch("src.retriever.fts_search")
@patch("src.retriever.embed_query")
def test_hybrid_retrieve_empty(mock_embed, mock_fts, mock_vector):
    from src.retriever import hybrid_retrieve

    mock_embed.return_value = [0.1] * 384
    mock_vector.return_value = []
    mock_fts.return_value = []

    results = hybrid_retrieve("nonexistent query")
    assert results == []
