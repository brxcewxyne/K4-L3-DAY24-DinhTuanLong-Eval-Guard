"""Module 5: chunk enrichment for the Day 24 RAG pipeline.

Contract consumed by scripts/setup_answers.py (see doc/PIPELINE_INTERFACE.md):
    enrich_chunks(chunks) -> list[EnrichedChunk]
- one item per input chunk, in the same order (count/order preserved);
- every item exposes .enriched_text and .auto_metadata as consumed by setup;
- original chunk metadata (source provenance) is copied into .auto_metadata.

Live calls go through src.llm_provider (OpenCode Go / OpenAI). Each chunk gets
one combined LLM call (summary + questions + context + metadata in a single
request), using response_format JSON mode when the endpoint supports it and a
validated plain response otherwise. Every item records the actual provider,
model and fallback status (attributes + enrichment_* keys in auto_metadata).

If no provider is configured, the documented deterministic fallback applies:
the original text is kept unchanged, metadata is preserved, and
method="deterministic_fallback" with provider/model = None marks the result so
fallback output is never presented as LLM output. Once a provider (or client)
is configured, call failures raise instead of silently falling back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from config import GENERATION_MODEL
from src.llm_provider import LLMProvider, create_client, resolve_llm_provider

METHOD_LLM = "llm_combined"
METHOD_FALLBACK = "deterministic_fallback"

_SYSTEM_PROMPT = """Phân tích đoạn văn và CHỈ trả về JSON hợp lệ, không thêm chữ nào ngoài JSON:
{
  "summary": "tóm tắt 2-3 câu ngắn gọn, chỉ dùng dữ kiện có trong đoạn văn",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {"topic": "...", "category": "policy|hr|it|finance"}
}
Không bịa thêm dữ kiện ngoài đoạn văn."""


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    auto_metadata: dict
    method: str
    provider: str | None = None
    model: str | None = None


def _enrichment_metadata(method: str, provider_name: str | None, model: str | None) -> dict:
    return {
        "enrichment_method": method,
        "enrichment_provider": provider_name,
        "enrichment_model": model,
    }


def _fallback(text: str, metadata: dict) -> EnrichedChunk:
    auto_metadata = {**metadata, **_enrichment_metadata(METHOD_FALLBACK, None, None)}
    return EnrichedChunk(
        original_text=text,
        enriched_text=text,
        auto_metadata=auto_metadata,
        method=METHOD_FALLBACK,
        provider=None,
        model=None,
    )


def _json_mode_unsupported(error: Exception) -> bool:
    text = str(error).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text


def _create_completion(client, model: str, messages: list[dict], json_mode: bool):
    kwargs = {"model": model, "temperature": 0, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def _parse_json_object(content: str) -> dict | None:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(content[start:end + 1])
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    return parsed


def _enrich_single_call(text: str, source: str, client, model: str) -> dict:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    ]
    try:
        response = _create_completion(client, model, messages, json_mode=True)
    except Exception as error:
        if not _json_mode_unsupported(error):
            raise RuntimeError(f"Enrichment LLM call failed ({model}): {error}") from error
        print(f"  ⚠️  {model} rejected response_format; retrying without JSON mode.")
        try:
            response = _create_completion(client, model, messages, json_mode=False)
        except Exception as retry_error:
            raise RuntimeError(
                f"Enrichment LLM call failed without JSON mode ({model}): {retry_error}"
            ) from retry_error
    content = response.choices[0].message.content or ""
    parsed = _parse_json_object(content)
    if parsed is None:
        raise RuntimeError(
            f"Enrichment LLM returned invalid JSON ({model}): {content[:200]!r}")
    return parsed


def enrich_chunks(chunks: list[dict], client=None,
                  provider: LLMProvider | None = None) -> list[EnrichedChunk]:
    if not chunks:
        return []
    if provider is None:
        provider = resolve_llm_provider()
    if client is None and provider is not None:
        client = create_client(provider)
    if client is None:
        return [_fallback(chunk["text"], dict(chunk.get("metadata") or {})) for chunk in chunks]

    model = provider.model if provider is not None else GENERATION_MODEL
    provider_name = provider.name if provider is not None else None
    enriched: list[EnrichedChunk] = []
    for chunk in chunks:
        text = chunk["text"]
        metadata = dict(chunk.get("metadata") or {})
        parsed = _enrich_single_call(text, str(metadata.get("source", "")), client, model)
        context_line = parsed.get("context", "")
        enriched_text = (f"{context_line}\n\n{text}"
                         if isinstance(context_line, str) and context_line.strip() else text)
        auto_metadata: dict = {}
        llm_metadata = parsed.get("metadata")
        if isinstance(llm_metadata, dict):
            auto_metadata.update(llm_metadata)
        auto_metadata.update(metadata)
        auto_metadata.update(_enrichment_metadata(METHOD_LLM, provider_name, model))
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            auto_metadata=auto_metadata,
            method=METHOD_LLM,
            provider=provider_name,
            model=model,
        ))
    return enriched
