"""Document loading and chunking for the Day 24 RAG evaluation corpus.

The Day 24 assembler calls load_documents() and chunk_hierarchical() directly.
Semantic and structure-aware strategies are retained for comparison with Day 18.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Read Markdown and text-layer PDF files without discarding old policies."""
    from pypdf import PdfReader

    root = Path(data_dir)
    documents: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".pdf"}:
            continue
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
        else:
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        if text.strip():
            relative = path.relative_to(root).as_posix()
            documents.append({
                "text": text,
                "metadata": {"source": relative, "title": path.stem},
            })
    return documents


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Paragraph baseline retained for Day 18 comparisons."""
    metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    result: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > chunk_size:
            result.append(Chunk(current, {**metadata, "chunk_index": len(result)}))
            current = paragraph
        else:
            current = candidate
    if current:
        result.append(Chunk(current, {**metadata, "chunk_index": len(result)}))
    return result


_SEMANTIC_MODEL = None


def _semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    """Group adjacent sentences whose embedding cosine similarity meets threshold."""
    import numpy as np

    metadata = dict(metadata or {})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text or "") if s.strip()]
    if not sentences:
        return []
    groups: list[list[str]] = [[sentences[0]]]
    if len(sentences) > 1:
        embeddings = _semantic_model().encode(sentences, normalize_embeddings=True)
        for i in range(1, len(sentences)):
            similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
            if similarity >= threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])
    return [
        Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
    ]


def _units(text: str, limit: int) -> list[str]:
    """Prefer sentence boundaries, then word boundaries for oversized text."""
    if limit <= 0:
        raise ValueError("Chunk sizes must be positive")
    # A Markdown table row is one unit. Never flatten columns across lines.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])[^\S\n]+|\n+", text or "") if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        current = ""
        for word in sentence.split():
            if len(word) > limit:
                if current:
                    units.append(current)
                    current = ""
                units.extend(word[i:i + limit] for i in range(0, len(word), limit))
                continue
            candidate = f"{current} {word}" if current else word
            if current and len(candidate) > limit:
                units.append(current)
                current = word
            else:
                current = candidate
        if current:
            units.append(current)
    return units


def _group(units: list[str], limit: int, separator: str = " ") -> list[str]:
    groups: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{separator}{unit}" if current else unit
        if current and len(candidate) > limit:
            groups.append(current)
            current = unit
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Return parents and searchable children linked by document-scoped IDs."""
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("Chunk sizes must be positive")
    metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    parent_units = [unit for paragraph in paragraphs for unit in _units(paragraph, parent_size)]
    parent_texts = _group(parent_units, parent_size, "\n")
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "unknown"))
    prefix = f"Nguá»n: {source}\n" if source != "unknown" else ""
    content_limit = child_size - len(prefix)
    if content_limit <= 0:
        raise ValueError("Source name leaves no room for child content")
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{source}:parent:{parent_index}"
        parents.append(Chunk(
            parent_text,
            {**metadata, "strategy": "hierarchical", "chunk_type": "parent",
             "chunk_index": parent_index, "parent_id": parent_id},
            parent_id,
        ))
        child_texts = _group(_units(parent_text, content_limit), content_limit, "\n")
        for child_index, child_text in enumerate(child_texts):
            children.append(Chunk(
                f"{prefix}{child_text}",
                {**metadata, "strategy": "hierarchical", "chunk_type": "child",
                 "chunk_index": child_index, "parent_id": parent_id},
                parent_id,
            ))
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Keep Markdown sections, including their lists and tables, intact."""
    metadata = dict(metadata or {})
    parts = re.split(r"(^#{1,3}\s+.+$)", text or "", flags=re.MULTILINE)
    result: list[Chunk] = []
    if parts[0].strip():
        result.append(Chunk(parts[0].strip(), {
            **metadata, "strategy": "structure", "section": "Introduction", "chunk_index": 0,
        }))
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result.append(Chunk(f"{header}\n\n{body}".strip(), {
            **metadata, "strategy": "structure", "section": header.lstrip("# ").strip(),
            "chunk_index": len(result),
        }))
    return result