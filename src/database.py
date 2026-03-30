"""
Vector Database Layer — ChromaDB implementation.
"""

from __future__ import annotations

from typing import List, Optional
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config import settings
from src.models import DocumentChunk


class VectorDB:
    """Wrapper for ChromaDB operations."""
    
    _client: Optional[chromadb.ClientAPI] = None
    _collection: Optional[chromadb.Collection] = None

    @classmethod
    def get_client(cls) -> chromadb.ClientAPI:
        if cls._client is None:
            persist_dir = settings.vector_db.persist_directory
            os.makedirs(persist_dir, exist_ok=True)
            cls._client = chromadb.PersistentClient(path=persist_dir)
        return cls._client

    @classmethod
    def get_collection(cls) -> chromadb.Collection:
        if cls._collection is None:
            client = cls.get_client()
            cls._collection = client.get_or_create_collection(
                name=settings.vector_db.collection_name,
                metadata={"hnsw:space": settings.vector_db.distance_metric}
            )
        return cls._collection

    @classmethod
    def insert_chunks(cls, chunks: List[DocumentChunk]):
        """Insert chunks into ChromaDB."""
        collection = cls.get_collection()
        
        ids = [c.id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = []
        for c in chunks:
            m = c.metadata.copy()
            m["doc_id"] = c.doc_id
            m["title"] = c.title
            m["chunk_index"] = c.chunk_index
            metadatas.append(m)

        # Batch insert
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"✅ Inserted {len(chunks)} chunks into ChromaDB")

    @classmethod
    def vector_search(cls, query_embedding: List[float], top_k: int = 20, 
                      filter_dict: Optional[dict] = None) -> List[dict]:
        """Search ChromaDB by embedding with optional metadata filtering."""
        collection = cls.get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=filter_dict
        )

        
        formatted = []
        if not results["ids"]:
            return []
            
        for i in range(len(results["ids"][0])):
            cid = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            dist = results["distances"][0][i]
            
            # Convert distance to similarity score if needed (Chroma distances vary by metric)
            # For cosine, distance is 1 - similarity. So similarity = 1 - distance.
            score = 1 - dist if settings.vector_db.distance_metric == "cosine" else dist
            
            formatted.append({
                "id": cid,
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title", ""),
                "content": doc,
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": meta,
                "score": score
            })
        return formatted

    @classmethod
    def get_chunk_count(cls) -> int:
        try:
            return cls.get_collection().count()
        except Exception:
            return 0


# Legacy function names for compatibility with other modules
def init_database():
    VectorDB.get_collection()

def insert_chunks(chunks: List[DocumentChunk]):
    VectorDB.insert_chunks(chunks)

def vector_search(
    query_embedding: List[float],
    top_k: int = 20,
    filter_dict: Optional[dict] = None,
) -> List[dict]:
    return VectorDB.vector_search(query_embedding, top_k, filter_dict)

def get_chunk_count() -> int:
    return VectorDB.get_chunk_count()

def check_connection() -> bool:
    try:
        VectorDB.get_client().heartbeat()
        return True
    except Exception:
        return False
