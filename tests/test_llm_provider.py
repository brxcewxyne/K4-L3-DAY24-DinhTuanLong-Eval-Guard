"""Offline tests for provider configuration and live-mode generation; no API calls."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
import scripts.setup_answers as setup
from src import llm_provider
from src.llm_provider import LLMProvider

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


@pytest.fixture
def search_reranker():
    search = Mock()
    search.search.return_value = [
        SimpleNamespace(text="ngữ cảnh", score=0.9, metadata={"source": "a.md"})]
    reranker = Mock()
    reranker.rerank.return_value = [SimpleNamespace(text="ngữ cảnh")]
    return search, reranker


def _client(content="câu trả lời", model=None):
    client = Mock()
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    if model is not None:
        response.model = model
    client.chat.completions.create.return_value = response
    return client


def test_documented_opencode_go_defaults():
    if "OPENCODE_GO_BASE_URL" not in os.environ:
        assert config.OPENCODE_GO_BASE_URL == "https://opencode.ai/zen/go/v1"
    if "OPENCODE_GO_MODEL" not in os.environ:
        assert config.OPENCODE_GO_MODEL == "deepseek-v4.1-flash"


def test_resolve_prefers_opencode_go(monkeypatch):
    monkeypatch.setattr(config, "OPENCODE_GO_API_KEY", "test-opencode-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-openai-key")

    provider = llm_provider.resolve_llm_provider()

    assert provider == LLMProvider(
        name="opencode-go",
        base_url="https://opencode.ai/zen/go/v1",
        model="deepseek-v4.1-flash",
        api_key="test-opencode-key",
    )


def test_resolve_falls_back_to_openai(monkeypatch):
    monkeypatch.setattr(config, "OPENCODE_GO_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-openai-key")

    provider = llm_provider.resolve_llm_provider()

    assert provider.name == "openai"
    assert provider.base_url is None
    assert provider.model == config.GENERATION_MODEL


def test_resolve_none_without_keys(no_provider):
    assert llm_provider.resolve_llm_provider() is None


def test_create_client_passes_base_url_and_session_header(monkeypatch):
    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(config, "OPENCODE_GO_SESSION", "session-fixed")

    llm_provider.create_client(PROVIDER)

    assert created["api_key"] == "test-key"
    assert created["base_url"] == "https://opencode.ai/zen/go/v1"
    assert created["default_headers"] == {"x-opencode-session": "session-fixed"}


def test_create_client_generates_session_when_unset(monkeypatch):
    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(config, "OPENCODE_GO_SESSION", "")

    llm_provider.create_client(PROVIDER)

    session = created["default_headers"]["x-opencode-session"]
    assert isinstance(session, str) and session


def test_create_client_without_base_url(monkeypatch):
    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    openai_provider = LLMProvider(name="openai", base_url=None, model="gpt-4o-mini", api_key="k")

    llm_provider.create_client(openai_provider)

    assert created == {"api_key": "k"}


def test_run_query_strict_uses_provider_model(search_reranker):
    search, reranker = search_reranker
    client = _client()

    answer, contexts, mode, model = setup.run_query(
        "Câu hỏi?", search, reranker, 3, client, strict=True, provider=PROVIDER)

    assert (answer, contexts, mode, model) == ("câu trả lời", ["ngữ cảnh"], "llm",
                                               "deepseek-v4.1-flash")
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-v4.1-flash"
    assert "ngữ cảnh" in kwargs["messages"][1]["content"]


def test_run_query_strict_creates_client_from_provider(monkeypatch, search_reranker):
    search, reranker = search_reranker
    client = _client()
    mocked_create = Mock(return_value=client)
    monkeypatch.setattr(setup, "create_client", mocked_create)

    _, _, mode, model = setup.run_query(
        "Câu hỏi?", search, reranker, 3, None, strict=True, provider=PROVIDER)

    mocked_create.assert_called_once_with(PROVIDER)
    assert (mode, model) == ("llm", "deepseek-v4.1-flash")


def test_run_query_records_reported_model(search_reranker):
    search, reranker = search_reranker
    client = _client(model="deepseek-v4.1-flash-2026-06-01")

    _, _, mode, model = setup.run_query(
        "Câu hỏi?", search, reranker, 3, client, strict=True, provider=PROVIDER)

    assert mode == "llm"
    assert model == "deepseek-v4.1-flash-2026-06-01"


def test_run_query_strict_refuses_api_failure(search_reranker):
    search, reranker = search_reranker
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("HTTP 401")

    with pytest.raises(RuntimeError, match="generation failed"):
        setup.run_query("Câu hỏi?", search, reranker, 3, client,
                        strict=True, provider=PROVIDER)


def test_run_query_strict_refuses_without_provider(no_provider, search_reranker):
    search, reranker = search_reranker

    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        setup.run_query("Câu hỏi?", search, reranker, 3, None, strict=True)


def test_run_query_strict_refuses_without_contexts(no_provider):
    search = Mock()
    search.search.return_value = []
    reranker = Mock()
    reranker.rerank.return_value = []

    with pytest.raises(RuntimeError, match="No contexts"):
        setup.run_query("Câu hỏi?", search, reranker, 3, None, strict=True)


def test_run_query_non_strict_keeps_documented_fallback(search_reranker):
    search, reranker = search_reranker
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("boom")

    answer, contexts, mode, model = setup.run_query(
        "Câu hỏi?", search, reranker, 3, client)

    assert (answer, contexts, mode, model) == ("ngữ cảnh", ["ngữ cảnh"],
                                               "extractive_fallback", None)


def test_run_query_non_strict_without_provider_uses_extractive(no_provider, search_reranker):
    search, reranker = search_reranker

    answer, contexts, mode, model = setup.run_query("Câu hỏi?", search, reranker, 3)

    assert (mode, model) == ("extractive_fallback", None)


def test_build_answer_record_includes_provider():
    item = {"id": 7, "distribution": "factual", "question": "Q?", "ground_truth": "GT"}

    record = setup.build_answer_record(item, "A", ["C"], "llm", "deepseek-v4.1-flash",
                                       "opencode-go")

    assert record == {
        "id": 7, "distribution": "factual", "question": "Q?", "ground_truth": "GT",
        "answer": "A", "contexts": ["C"],
        "generation_mode": "llm", "generation_model": "deepseek-v4.1-flash",
        "generation_provider": "opencode-go"}


def test_answers_schema_accepts_generation_provider():
    from src.experiment_support import read_json
    from src.report_contracts import validate_answers

    questions = read_json(config.TEST_SET_PATH)
    rows = [setup.build_answer_record(q, "câu trả lời", ["ngữ cảnh"], "llm",
                                      "deepseek-v4.1-flash", "opencode-go")
            for q in questions]

    validate_answers(rows, questions)
