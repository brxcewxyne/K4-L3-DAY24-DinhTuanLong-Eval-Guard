"""Offline interface tests for src/m5_enrichment.py; LLM calls are mocked."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
import src.m5_enrichment as m5
from src.llm_provider import LLMProvider

CHUNKS = [
    {"text": "Đoạn một.", "metadata": {"source": "a.md", "parent_id": "a:parent:0"}},
    {"text": "Đoạn hai.", "metadata": {"source": "b.md", "parent_id": "b:parent:0"}},
]

PROVIDER = LLMProvider(
    name="opencode-go",
    base_url="https://opencode.ai/zen/go/v1",
    model="deepseek-v4.1-flash",
    api_key="test-key",
)


@pytest.fixture
def no_provider(monkeypatch):
    monkeypatch.setattr(config, "OPENCODE_GO_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")


def _llm_client(payload):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])
    return client


def test_without_provider_uses_deterministic_fallback(no_provider):
    result = m5.enrich_chunks(CHUNKS)

    assert len(result) == len(CHUNKS)
    assert [item.original_text for item in result] == ["Đoạn một.", "Đoạn hai."]
    for item in result:
        assert item.enriched_text == item.original_text
        assert item.method == m5.METHOD_FALLBACK
        assert item.provider is None and item.model is None
        assert item.auto_metadata["enrichment_method"] == m5.METHOD_FALLBACK
        assert item.auto_metadata["enrichment_provider"] is None
        assert item.auto_metadata["enrichment_model"] is None
    assert result[0].auto_metadata["source"] == "a.md"
    assert result[0].auto_metadata["parent_id"] == "a:parent:0"
    assert result[1].auto_metadata["source"] == "b.md"


def test_empty_input_returns_empty_list(no_provider):
    assert m5.enrich_chunks([]) == []


def test_llm_enrichment_records_provider_model_and_method():
    client = _llm_client({
        "summary": "Tóm tắt.", "questions": ["Câu hỏi?"],
        "context": "Đoạn văn thuộc tài liệu a.md.",
        "metadata": {"topic": "HR", "source": "fake-source"},
    })

    result = m5.enrich_chunks(CHUNKS, client=client, provider=PROVIDER)

    assert len(result) == len(CHUNKS)
    first = result[0]
    assert first.method == m5.METHOD_LLM
    assert first.provider == "opencode-go"
    assert first.model == "deepseek-v4.1-flash"
    assert first.enriched_text.startswith("Đoạn văn thuộc tài liệu a.md.")
    assert first.enriched_text.endswith("Đoạn một.")
    assert first.auto_metadata["topic"] == "HR"
    assert first.auto_metadata["source"] == "a.md"
    assert first.auto_metadata["parent_id"] == "a:parent:0"
    assert first.auto_metadata["enrichment_method"] == m5.METHOD_LLM
    assert first.auto_metadata["enrichment_provider"] == "opencode-go"
    assert first.auto_metadata["enrichment_model"] == "deepseek-v4.1-flash"
    assert client.chat.completions.create.call_count == 2
    call = client.chat.completions.create.call_args_list[0].kwargs
    assert call["model"] == "deepseek-v4.1-flash"
    assert call["response_format"] == {"type": "json_object"}


def test_json_mode_unsupported_is_retried_and_validated():
    client = Mock()
    unsupported = RuntimeError(
        "Error code: 400 - {'error': {'message': 'response_format is not supported'}}")
    wrapped = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='Đây là kết quả: {"context": "Mô tả.", "metadata": {"topic": "HR"}} hết.'))])
    client.chat.completions.create.side_effect = [unsupported, wrapped]

    result = m5.enrich_chunks([CHUNKS[0]], client=client, provider=PROVIDER)

    assert result[0].method == m5.METHOD_LLM
    assert result[0].enriched_text.startswith("Mô tả.")
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 2
    assert "response_format" in calls[0].kwargs
    assert "response_format" not in calls[1].kwargs


def test_api_failure_raises_instead_of_falling_back():
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("HTTP 500")

    with pytest.raises(RuntimeError, match="Enrichment LLM call failed"):
        m5.enrich_chunks([CHUNKS[0]], client=client, provider=PROVIDER)


def test_invalid_json_raises_instead_of_falling_back():
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))])

    with pytest.raises(RuntimeError, match="invalid JSON"):
        m5.enrich_chunks([CHUNKS[0]], client=client, provider=PROVIDER)


def test_setup_answers_consumes_real_m5_items(no_provider, monkeypatch):
    from scripts import setup_answers

    children = [SimpleNamespace(text="raw", metadata={"source": "policy"}, parent_id="parent")]
    search = SimpleNamespace(index=Mock(), search=Mock())
    monkeypatch.setitem(sys.modules, "src.m1_chunking", SimpleNamespace(
        load_documents=lambda: [{"text": "document", "metadata": {"source": "policy"}}],
        chunk_hierarchical=lambda text, metadata: ([], children)))
    monkeypatch.setitem(sys.modules, "src.m2_search",
                        SimpleNamespace(HybridSearch=lambda: search))
    monkeypatch.setitem(sys.modules, "src.m3_rerank",
                        SimpleNamespace(CrossEncoderReranker=object))

    returned = setup_answers.build_pipeline()

    assert returned[0] is search
    indexed = search.index.call_args.args[0]
    assert indexed[0]["text"] == "raw"
    metadata = indexed[0]["metadata"]
    assert metadata["source"] == "policy"
    assert metadata["parent_id"] == "parent"
    assert metadata["enrichment_method"] == m5.METHOD_FALLBACK
    assert metadata["enrichment_provider"] is None
