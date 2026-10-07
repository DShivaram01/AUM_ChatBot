"""Single owner for loaded models, embedders, and retrieval stores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeManager:
    """Owns the process-local runtime; UI modules receive no model objects."""
    embedder: object | None = None
    reranker: object | None = None
    llm_tok: object | None = None
    llm_model: object | None = None
    cos_index: object | None = None
    cos_embeddings: object | None = None
    cos_metadata: object | None = None
    cos_texts: object | None = None
    cos_bm25: object | None = None
    housing_index: object | None = None
    housing_embeddings: object | None = None
    housing_chunks: object | None = None
    housing_ok: bool = False

    @property
    def ready(self) -> bool:
        return self.llm_tok is not None and self.llm_model is not None
