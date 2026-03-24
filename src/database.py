"""
Database layer — PostgreSQL + pgvector.
Handles table creation, vector extension, FTS index, and CRUD.
"""

from __future__ import annotations

import json
from typing import List, Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from loguru import logger

from src.config import settings
from src.models import DocumentChunk


# ── Connection helpers ───────────────────────────────────────
def get_connection():
    """Return a new psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(settings.database.url)
    register_vector(conn)
    return conn


def init_database():
    """Create extensions, tables, and indexes if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Enable extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

        dim = settings.embedding.dimension

        # Main chunks table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT '',
                content     TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                metadata    JSONB DEFAULT '{{}}'::jsonb,
                embedding   vector({dim}),
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Vector similarity index (IVFFlat for scale)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        # Full-text search index
        cur.execute("""
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS fts_vector tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
            ) STORED;
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_fts
            ON document_chunks USING gin(fts_vector);
        """)

        # Trigram index for fuzzy matching
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm
            ON document_chunks USING gin(content gin_trgm_ops);
        """)

        conn.commit()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Database init failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ── CRUD ─────────────────────────────────────────────────────
def insert_chunks(chunks: List[DocumentChunk], batch_size: int = 100):
    """Batch-insert document chunks with embeddings."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            values = []
            for c in batch:
                values.append((
                    c.id,
                    c.doc_id,
                    c.title,
                    c.content,
                    c.chunk_index,
                    json.dumps(c.metadata),
                    c.embedding,
                ))
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO document_chunks
                    (id, doc_id, title, content, chunk_index, metadata, embedding)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                values,
                template="(%s, %s, %s, %s, %s, %s::jsonb, %s::vector)",
            )
            conn.commit()
            logger.info(f"  Inserted batch {i // batch_size + 1} ({len(batch)} chunks)")
        logger.info(f"✅ Inserted {len(chunks)} chunks total")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Insert failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def vector_search(
    query_embedding: List[float], top_k: int = 20
) -> List[dict]:
    """Cosine-similarity search via pgvector."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, doc_id, title, content, chunk_index, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def fts_search(query: str, top_k: int = 20) -> List[dict]:
    """Full-text search with ts_rank scoring."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, doc_id, title, content, chunk_index, metadata,
                   ts_rank_cd(fts_vector, plainto_tsquery('english', %s)) AS score
            FROM document_chunks
            WHERE fts_vector @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def get_chunk_count() -> int:
    """Return total number of stored chunks."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        cur.close()
        conn.close()


def check_connection() -> bool:
    """Quick health-check for DB connectivity."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        return True
    except Exception:
        return False
