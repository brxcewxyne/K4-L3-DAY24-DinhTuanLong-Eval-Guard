"""Cross-encoder reranking for Day 24's hybrid retrieval results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_MODEL_CACHE: dict[str, object] = {}


class CrossEncoderReranker:
    """Score each (question, retrieved text) pair, then retain the best rows."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model=None):
        self.model_name = model_name
        self._model = model

    def _get_model(self):
        if self._model is None:
            if self.model_name not in _MODEL_CACHE:
                from sentence_transformers import CrossEncoder

                _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def rerank(
        self,
        question: str,
        docs: list[Any],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not docs:
            return []
        if not question.strip():
            raise ValueError("question must not be empty")

        def field(doc, name, default=None):
            return doc.get(name, default) if isinstance(doc, dict) else getattr(doc, name, default)

        pairs = [(question, field(doc, "text")) for doc in docs]
        scores = self._get_model().predict(pairs)
        if len(scores) != len(docs):
            raise ValueError("Reranker returned a different number of scores")

        scored = []
        for index, (doc, raw_score) in enumerate(zip(docs, scores)):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"Invalid reranker score at position {index}")
            scored.append((score, index, doc))

        # Stable on ties: preserve the hybrid search order.
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            RerankResult(
                text=field(doc, "text"),
                original_score=float(field(doc, "score", 0.0)),
                rerank_score=score,
                metadata=dict(field(doc, "metadata", {}) or {}),
                rank=rank,
            )
            for rank, (score, _, doc) in enumerate(scored[:top_k])
        ]