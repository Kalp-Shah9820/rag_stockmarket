"""
Configuration loader — reads settings.yaml + .env and exposes
a validated Pydantic settings object used everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"


# ── Sub-models ───────────────────────────────────────────────
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "rag_stockmarket"
    user: str = "rag_user"
    password: str = "rag_password"
    pool_size: int = 5

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    device: str = "cpu"


class ChunkingConfig(BaseModel):
    chunk_size: int = 700
    chunk_overlap: int = 100
    separators: List[str] = ["\n\n", "\n", ". ", " "]


class RetrievalConfig(BaseModel):
    vector_top_k: int = 20
    fts_top_k: int = 20
    final_top_k: int = 5
    vector_weight: float = 0.6
    fts_weight: float = 0.4
    similarity_threshold: float = 0.3


class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 32
    device: str = "cpu"


class GeneratorConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a financial news analyst. Answer questions using ONLY the "
        "provided context. Cite every claim with [Source N]. If the context "
        'is insufficient, say "I don\'t have enough information."'
    )


class GuardrailsConfig(BaseModel):
    allowed_topics: List[str] = [
        "stock market", "finance", "investing",
        "trading", "economy", "earnings", "market analysis",
    ]
    max_query_length: int = 500
    block_pii: bool = True


class AgentConfig(BaseModel):
    max_retries: int = 2
    relevance_threshold: float = 0.5


class DatasetConfig(BaseModel):
    name: str = "XA7-stock-market-news"
    source: str = "huggingface"
    hf_repo: str = "Lettria/XA7-stock-market-news"
    split: str = "train"


class EvaluationConfig(BaseModel):
    metrics: List[str] = [
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall",
    ]
    sample_size: int = 50


class AppConfig(BaseModel):
    name: str = "RAG Stock Market Intelligence"
    version: str = "1.0.0"
    log_level: str = "INFO"


# ── Master settings ──────────────────────────────────────────
class Settings(BaseModel):
    app: AppConfig = AppConfig()
    database: DatabaseConfig = DatabaseConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    generator: GeneratorConfig = GeneratorConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    agent: AgentConfig = AgentConfig()
    dataset: DatasetConfig = DatasetConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """Load settings from YAML, overlay with env-vars where applicable."""
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Override database fields from env if present
    db_overrides = {}
    for key in ("host", "port", "name", "user", "password"):
        env_key = f"DATABASE_{key.upper()}"
        val = os.getenv(env_key)
        if val is not None:
            db_overrides[key] = int(val) if key == "port" else val
    if db_overrides:
        raw.setdefault("database", {}).update(db_overrides)

    # Override OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    return Settings(**raw)


# Singleton
settings = load_settings()
