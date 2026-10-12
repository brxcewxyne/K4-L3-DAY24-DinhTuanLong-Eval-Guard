"""Hybrid Vietnamese BM25 and BGE-M3/Qdrant retrieval for Day 24."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> list[str]:
    """Use underthesea compounds as one BM25 token (keep underscores)."""
    if not text or not text.strip():
        return []
    from underthesea import word_tokenize

    return word_tokenize(text.lower(), format="text").split()


class BM25Search:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.bm25: Any = None

    def index(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        tokens = [segment_vietnamese(c["text"]) for c in self.documents]
        self.bm25 = BM25Okapi(tokens) if tokens and any(tokens) else None

    def search(self, question: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or top_k <= 0:
            return []
        tokens = segment_vietnamese(question)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=dict(self.documents[i].get("metadata", {})),
                method="bm25",
            )
            for i in ranked[:top_k] if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self, client=None, encoder=None) -> None:
        from qdrant_client import QdrantClient

        self.client = client if client is not None else QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=30,
        )
        self._encoder = encoder
        self.collection = COLLECTION_NAME

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.collection = collection
        # Load and validate the model before replacing the collection.
        if chunks:
            vectors = self._get_encoder().encode(
                [c["text"] for c in chunks], batch_size=32,
                show_progress_bar=True, normalize_embeddings=True,
            )
            if len(vectors) != len(chunks):
                raise ValueError("Embedding count differs from chunk count")
            if any(len(vector) != EMBEDDING_DIM for vector in vectors):
                raise ValueError(f"Expected {EMBEDDING_DIM} embedding dimensions")
        else:
            vectors = []
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        for start in range(0, len(chunks), 64):
            points = []
            for i in range(start, min(start + 64, len(chunks))):
                vector = vectors[i].tolist()
                points.append(PointStruct(
                    id=i,
                    vector=vector,
                    payload={"text": chunks[i]["text"], "metadata": dict(chunks[i].get("metadata", {}))},
                ))
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self, question: str, top_k: int = DENSE_TOP_K,
        collection: str | None = None,
    ) -> list[SearchResult]:
        if not question.strip() or top_k <= 0:
            return []
        vector = self._get_encoder().encode(question, normalize_embeddings=True).tolist()
        points = self.client.search(
            collection_name=collection or self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(
                text=(point.payload or {}).get("text", ""),
                score=float(point.score),
                metadata=dict((point.payload or {}).get("metadata", {})),
                method="dense",
            )
            for point in points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Combine rankings using sum(1 / (k + one-based rank))."""
    if k <= 0 or top_k <= 0:
        raise ValueError("k and top_k must be positive")
    merged: dict[tuple[str, str], tuple[SearchResult, float]] = {}
    for results in results_list:
        seen: set[tuple[str, str]] = set()
        for rank, result in enumerate(results, start=1):
            key = (str(result.metadata.get("source", "")), result.text)
            if key in seen:
                continue
            seen.add(key)
            previous = merged.get(key)
            merged[key] = (result if previous is None else previous[0],
                           (0.0 if previous is None else previous[1]) + 1.0 / (k + rank))
    ranked = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)
    return [
        SearchResult(item.text, float(score), dict(item.metadata), "hybrid")
        for item, score in ranked[:top_k]
    ]


class HybridSearch:
    def __init__(self, bm25=None, dense=None) -> None:
        self.bm25 = bm25 if bm25 is not None else BM25Search()
        self.dense = dense if dense is not None else DenseSearch()

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks, collection=collection)

    def search(self, question: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        return reciprocal_rank_fusion([
            self.bm25.search(question, BM25_TOP_K),
            self.dense.search(question, DENSE_TOP_K),
        ], top_k=top_k)