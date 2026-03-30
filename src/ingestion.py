"""
Data Ingestion Pipeline
- Loads XA7 Stock Market News dataset from HuggingFace
- Chunks documents with RecursiveCharacterTextSplitter
- Generates embeddings
- Stores in ChromaDB
"""

from __future__ import annotations

import hashlib
from typing import List

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import settings
from src.models import DocumentChunk
from src.embeddings import get_embeddings
from src.database import init_database, insert_chunks, get_chunk_count
from src.keyword_search import update_bm25_index


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
    """Split raw records into overlapping chunks with robust field mapping."""
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
        
        # Verbose debug for first record
        if records:
            logger.debug(f"First record keys: {list(records[0].keys())}")
            
        for record in records:
            # Helper to get field case-insensitively
            def get_field(keys_to_try):
                for k in record.keys():
                    if k.lower() in keys_to_try:
                        return record[k]
                return None

            title = get_field(["headline", "title", "name", "description"]) or ""
            text = get_field(["text", "content", "body"]) or title
            
            if not text:
                progress.advance(task)
                continue
            
            # Default source for this dataset is 'news'
            source = get_field(["source"]) or "news"
            date = get_field(["date", "time"]) or ""
            label = get_field(["label", "target"]) or ""
            
            doc_id = _doc_id(str(text))
            splits = splitter.split_text(str(text))

            for idx, split_text in enumerate(splits):
                category = get_field(["category", "genre"]) or ""
                
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    title=str(title),
                    content=str(split_text),
                    chunk_index=idx,
                    metadata={
                        "source": str(source),
                        "date": str(date),
                        "url": str(get_field(["url", "link"]) or ""),
                        "category": str(category),
                        "label": str(label),
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
    if chunks:
        chunks = embed_chunks(chunks)
        
        # 6. Store
        insert_chunks(chunks)

        # 7. Update BM25 Index
        update_bm25_index(chunks)
        
        total = get_chunk_count()
        logger.info(f"🎉 Ingestion complete — {total:,} chunks in database")
    else:
        logger.warning("⚠️ No chunks created. Ingestion stopped.")


if __name__ == "__main__":
    run_ingestion()
