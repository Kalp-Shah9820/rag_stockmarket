"""
Data Ingestion Pipeline
- Loads XA7 Stock Market News dataset from HuggingFace
- Chunks documents with RecursiveCharacterTextSplitter
- Generates embeddings
- Stores in PostgreSQL / pgvector
"""

from __future__ import annotations

import hashlib
from typing import List

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import settings
from src.models import DocumentChunk
from src.embeddings import get_embeddings
from src.database import init_database, insert_chunks, get_chunk_count


def load_xa7_dataset() -> list[dict]:
    """Download and return the XA7 stock-market news dataset."""
    cfg = settings.dataset
    logger.info(f"📥 Loading dataset: {cfg.hf_repo} (split={cfg.split})")
    ds = load_dataset(cfg.hf_repo, split=cfg.split)
    records = [dict(row) for row in ds]
    logger.info(f"   Loaded {len(records):,} records")
    return records


def _doc_id(text: str) -> str:
    """Deterministic document id from content hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def chunk_documents(records: list[dict]) -> List[DocumentChunk]:
    """Split raw records into overlapping chunks."""
    cfg = settings.chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )

    chunks: List[DocumentChunk] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Chunking documents...", total=len(records))
        for record in records:
            # Adapt to dataset schema — typical fields: title, text/content
            title = record.get("title", "")
            text = record.get("text", record.get("content", ""))
            if not text:
                progress.advance(task)
                continue

            doc_id = _doc_id(text)
            splits = splitter.split_text(text)

            for idx, split_text in enumerate(splits):
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    title=title,
                    content=split_text,
                    chunk_index=idx,
                    metadata={
                        "source": record.get("source", ""),
                        "date": record.get("date", ""),
                        "url": record.get("url", ""),
                        "category": record.get("category", ""),
                    },
                )
                chunks.append(chunk)
            progress.advance(task)

    logger.info(f"✂️  Created {len(chunks):,} chunks from {len(records):,} documents")
    return chunks


def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Generate embeddings for all chunks (batched)."""
    embedder = get_embeddings()
    texts = [c.content for c in chunks]
    batch_size = settings.embedding.batch_size

    logger.info(f"🧠 Generating embeddings (batch_size={batch_size})...")
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = embedder.embed_documents(batch)
        all_embeddings.extend(embs)
        logger.info(
            f"   Embedded batch {i // batch_size + 1}"
            f" / {(len(texts) - 1) // batch_size + 1}"
        )

    for chunk, emb in zip(chunks, all_embeddings):
        chunk.embedding = emb

    logger.info(f"✅ Embedded {len(chunks):,} chunks")
    return chunks


def run_ingestion():
    """End-to-end ingestion: load → chunk → embed → store."""
    logger.info("🚀 Starting ingestion pipeline")

    # 1. Init DB
    init_database()

    # 2. Check if data already ingested
    existing = get_chunk_count()
    if existing > 0:
        logger.warning(
            f"⚠️  Database already contains {existing:,} chunks. "
            "Skipping ingestion. Drop table to re-ingest."
        )
        return

    # 3. Load dataset
    records = load_xa7_dataset()

    # 4. Chunk
    chunks = chunk_documents(records)

    # 5. Embed
    chunks = embed_chunks(chunks)

    # 6. Store
    insert_chunks(chunks)

    total = get_chunk_count()
    logger.info(f"🎉 Ingestion complete — {total:,} chunks in database")


if __name__ == "__main__":
    run_ingestion()
