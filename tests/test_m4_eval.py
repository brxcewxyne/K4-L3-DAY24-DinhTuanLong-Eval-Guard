"""Offline interface tests for src/m4_eval.py; RAGAS/LLM/embedding calls are mocked."""
import math
import sys
from types import SimpleNamespace

import pytest

import src.m4_eval as m4
from src.llm_provider import LLMProvider

PROVIDER = LLMProvider(
    name="opencode-go",
    base_url="https://opencode.ai/zen/go/v1",
    model="deepseek-v4.1-flash",
    api_key="test-key",
)


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    monkeypatch.setattr(m4, "resolve_llm_provider", lambda: PROVIDER)
    return PROVIDER


def _row(question="Q", answer="A", contexts=None, ground_truth="GT",
         faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7, context_recall=0.6):
    return {"question": question, "answer": answer,
            "contexts": list(contexts if contexts is not None else ["C"]),
            "ground_truth": ground_truth,
            "faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
            "context_precision": context_precision, "context_recall": context_recall}


def test_evaluate_ragas_returns_rows_and_truthful_evaluator_models(monkeypatch, provider):
    rows = [_row("Q1", "A1", ["C1"], "G1"), _row("Q2", "A2", ["C2"], "G2")]
    monkeypatch.setattr(m4, "_run_ragas", lambda *args, **kwargs: rows)

    raw = m4.evaluate_ragas(["Q1", "Q2"], ["A1", "A2"], [["C1"], ["C2"]], ["G1", "G2"])

    per_question = raw["per_question"]
    assert len(per_question) == 2
    first = per_question[0]
    assert isinstance(first, m4.EvalResult)
    assert (first.question, first.answer, first.contexts, first.ground_truth) == (
        "Q1", "A1", ["C1"], "G1")
    for metric in m4.METRIC_NAMES:
        value = getattr(first, metric)
        assert type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0
    models = raw["metadata"]["evaluator_models"]
    assert models == [provider.model, m4.EVALUATOR_EMBEDDING_MODEL]
    assert all(isinstance(model, str) and model.strip() for model in models)


def test_evaluate_ragas_rejects_nan_scores(monkeypatch):
    rows = [_row(faithfulness=float("nan"))]
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[0])
        return rows

    monkeypatch.setattr(m4, "_run_ragas", fake)

    with pytest.raises(ValueError, match="faithfulness") as excinfo:
        m4.evaluate_ragas(["Q"], ["A"], [["C"]], ["G"])
    assert "NaN" in str(excinfo.value)
    assert len(calls) == 1 + m4._MAX_METRIC_RETRIES


def test_evaluate_ragas_retries_nan_rows_and_merges(monkeypatch):
    good0, bad1, good2 = _row("Q0"), _row("Q1", faithfulness=float("nan")), _row("Q2")
    retried1 = _row("Q1")
    calls = []

    def fake(questions, answers, contexts, ground_truths):
        calls.append(list(questions))
        if len(questions) == 3:
            return [good0, bad1, good2]
        return [retried1]

    monkeypatch.setattr(m4, "_run_ragas", fake)

    raw = m4.evaluate_ragas(["Q0", "Q1", "Q2"], ["A0", "A1", "A2"],
                            [["C0"], ["C1"], ["C2"]], ["G0", "G1", "G2"])

    assert calls == [["Q0", "Q1", "Q2"], ["Q1"]]
    assert raw["per_question"][1].faithfulness == 0.9
    assert len(raw["per_question"]) == 3


def test_evaluate_ragas_rejects_missing_metric(monkeypatch):
    row = _row()
    del row["context_recall"]
    monkeypatch.setattr(m4, "_run_ragas", lambda *args, **kwargs: [row])

    with pytest.raises(ValueError, match="context_recall"):
        m4.evaluate_ragas(["Q"], ["A"], [["C"]], ["G"])


def test_evaluate_ragas_rejects_out_of_range_score(monkeypatch):
    monkeypatch.setattr(m4, "_run_ragas", lambda *args, **kwargs: [_row(faithfulness=1.5)])

    with pytest.raises(ValueError, match="outside"):
        m4.evaluate_ragas(["Q"], ["A"], [["C"]], ["G"])


def test_evaluate_ragas_clamps_float_epsilon(monkeypatch):
    rows = [_row(answer_relevancy=1.0000000000000002, context_recall=-1e-16)]
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[0])
        return rows

    monkeypatch.setattr(m4, "_run_ragas", fake)

    raw = m4.evaluate_ragas(["Q"], ["A"], [["C"]], ["G"])

    first = raw["per_question"][0]
    assert first.answer_relevancy == 1.0
    assert first.context_recall == 0.0
    assert len(calls) == 1


def test_evaluate_ragas_propagates_backend_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("RAGAS exploded")

    monkeypatch.setattr(m4, "_run_ragas", boom)

    with pytest.raises(RuntimeError, match="RAGAS exploded"):
        m4.evaluate_ragas(["Q"], ["A"], [["C"]], ["G"])


def test_evaluate_ragas_rejects_partial_results(monkeypatch):
    monkeypatch.setattr(m4, "_run_ragas", lambda *args, **kwargs: [_row()])

    with pytest.raises(ValueError, match="1 rows for 2 questions"):
        m4.evaluate_ragas(["Q1", "Q2"], ["A1", "A2"], [["C1"], ["C2"]], ["G1", "G2"])


def test_evaluate_ragas_validates_input_shapes():
    with pytest.raises(ValueError, match="equal length"):
        m4.evaluate_ragas(["Q"], ["A"], [], ["G"])
    with pytest.raises(ValueError, match="at least one question"):
        m4.evaluate_ragas([], [], [], [])
    with pytest.raises(ValueError, match=r"contexts\[0\]"):
        m4.evaluate_ragas(["Q"], ["A"], ["not-a-list"], ["G"])


def test_run_ragas_refuses_without_provider(monkeypatch):
    monkeypatch.setattr(m4, "resolve_llm_provider", lambda: None)

    with pytest.raises(RuntimeError, match="no LLM provider"):
        m4._run_ragas(["Q"], ["A"], [["C"]], ["G"])
    with pytest.raises(RuntimeError, match="no LLM provider"):
        m4.evaluator_models()


def test_run_ragas_wires_dataset_metrics_and_explicit_evaluators(monkeypatch, provider):
    captured = {}

    class FakeDataset:
        @staticmethod
        def from_dict(payload):
            captured["dataset"] = payload
            return payload

    class FakeFrame:
        def to_dict(self, orient=None):
            captured["orient"] = orient
            return [_row("q1", "a1", ["c1"], "g1")]

    class FakeResult:
        def to_pandas(self):
            return FakeFrame()

    def fake_evaluate(dataset, metrics=None, llm=None, embeddings=None, **kwargs):
        captured["metrics"] = metrics
        captured["llm"] = llm
        captured["embeddings"] = embeddings
        return FakeResult()

    class FakeLLMWrapper:
        def __init__(self, llm):
            self.llm = llm

    class FakeEmbeddingsWrapper:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=fake_evaluate))
    monkeypatch.setitem(sys.modules, "ragas.metrics", SimpleNamespace(
        faithfulness="F", answer_relevancy="AR",
        context_precision="CP", context_recall="CR"))
    monkeypatch.setitem(sys.modules, "ragas.llms",
                        SimpleNamespace(LangchainLLMWrapper=FakeLLMWrapper))
    monkeypatch.setitem(sys.modules, "ragas.embeddings",
                        SimpleNamespace(LangchainEmbeddingsWrapper=FakeEmbeddingsWrapper))
    monkeypatch.setitem(sys.modules, "langchain_openai",
                        SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings",
                        SimpleNamespace(HuggingFaceEmbeddings=FakeHuggingFaceEmbeddings))

    rows = m4._run_ragas(["q1"], ["a1"], [["c1"]], ["g1"])

    assert captured["dataset"] == {"question": ["q1"], "answer": ["a1"],
                                   "contexts": [["c1"]], "ground_truth": ["g1"]}
    assert captured["metrics"] == ["F", "AR", "CP", "CR"]
    llm_kwargs = captured["llm"].llm.kwargs
    assert llm_kwargs["model"] == provider.model
    assert llm_kwargs["base_url"] == provider.base_url
    assert llm_kwargs["default_headers"]["x-opencode-session"]
    embed_kwargs = captured["embeddings"].embeddings.kwargs
    assert embed_kwargs["model_name"] == m4.EVALUATOR_EMBEDDING_MODEL
    assert rows == [_row("q1", "a1", ["c1"], "g1")]
