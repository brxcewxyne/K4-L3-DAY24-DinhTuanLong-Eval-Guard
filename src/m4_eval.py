"""Module 4: RAGAS evaluator adapter for Day 24 Phase A.

Contract consumed by src/phase_a_ragas.run_ragas_50q() (see
doc/PIPELINE_INTERFACE.md):
    evaluate_ragas(questions, answers, contexts, ground_truths) -> dict
- "per_question": one EvalResult object per input question, each with the four
  metric attributes as finite Python floats in [0, 1];
- "metadata": {"evaluator_models": [...]} with the evaluator chat/embedding
  model IDs actually used for this evaluation run (never the judge model).

Evaluator configuration (truthful identifiers, no invented models):
- chat LLM: the resolved provider model (OpenCode Go deepseek-v4.1-flash when
  OPENCODE_GO_API_KEY is set; otherwise the OpenAI fallback model);
- embeddings: local sentence-transformers model EVALUATOR_EMBEDDING_MODEL
  (OpenCode Go has no embeddings endpoint; RAGAS answer_relevancy needs one).

Failure policy: a backend error, a row-count mismatch, a missing metric or a
NaN/Inf/out-of-range score raises immediately. Failed scores are never
replaced with 0 or silently dropped, so an invalid evaluation cannot appear
successful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.llm_provider import OPENCODE_GO_PROVIDER, opencode_session_id, resolve_llm_provider

EVALUATOR_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
# RAGAS có thể trả 1.0 + vài ulp do sai số float; chỉ clamp trong epsilon này,
# ngoài khoảng thật vẫn raise.
_SCORE_EPSILON = 1e-9
_MAX_METRIC_RETRIES = 2


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def evaluator_models() -> list[str]:
    provider = resolve_llm_provider()
    if provider is None:
        raise RuntimeError(
            "evaluate_ragas: no LLM provider configured (set OPENCODE_GO_API_KEY or "
            "OPENAI_API_KEY); refusing to fabricate evaluation results")
    return [provider.model, EVALUATOR_EMBEDDING_MODEL]


def _validate_inputs(questions, answers, contexts, ground_truths) -> None:
    groups = (("questions", questions), ("answers", answers),
              ("contexts", contexts), ("ground_truths", ground_truths))
    for name, value in groups:
        if not isinstance(value, list):
            raise ValueError(f"evaluate_ragas: {name} must be a list")
    if not questions:
        raise ValueError("evaluate_ragas: at least one question is required")
    if len({len(value) for _, value in groups}) != 1:
        raise ValueError(
            "evaluate_ragas: questions/answers/contexts/ground_truths must have equal length")
    for index, context_list in enumerate(contexts):
        if not isinstance(context_list, list):
            raise ValueError(f"evaluate_ragas: contexts[{index}] must be a list of strings")


def _finite_score(value, metric: str, index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"RAGAS metric '{metric}' is not numeric for question index {index}: {value!r}")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"RAGAS metric '{metric}' is missing or not numeric for question index {index}: {value!r}"
        ) from error
    if not math.isfinite(score):
        raise ValueError(
            f"RAGAS metric '{metric}' is NaN/infinite for question index {index}; "
            "refusing to fabricate a score")
    if score < -_SCORE_EPSILON or score > 1.0 + _SCORE_EPSILON:
        raise ValueError(
            f"RAGAS metric '{metric}' is outside [0, 1] for question index {index}: {score}")
    return min(1.0, max(0.0, score))


def _run_ragas(questions: list[str], answers: list[str], contexts: list[list[str]],
               ground_truths: list[str]) -> list[dict]:
    """Run the real RAGAS evaluation with explicitly configured evaluators."""
    provider = resolve_llm_provider()
    if provider is None:
        raise RuntimeError(
            "evaluate_ragas: no LLM provider configured (set OPENCODE_GO_API_KEY or "
            "OPENAI_API_KEY); refusing to fabricate evaluation results")

    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                               faithfulness)

    llm_kwargs = {"model": provider.model, "temperature": 0, "api_key": provider.api_key}
    if provider.base_url:
        llm_kwargs["base_url"] = provider.base_url
    if provider.name == OPENCODE_GO_PROVIDER:
        llm_kwargs["default_headers"] = {"x-opencode-session": opencode_session_id()}
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(**llm_kwargs))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EVALUATOR_EMBEDDING_MODEL))

    dataset = Dataset.from_dict({
        "question": [str(question) for question in questions],
        "answer": [str(answer) for answer in answers],
        "contexts": [list(context_list) for context_list in contexts],
        "ground_truth": [str(ground_truth) for ground_truth in ground_truths],
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    return result.to_pandas().to_dict(orient="records")


def _row_problem(row) -> str | None:
    """Return a problem description for a RAGAS row, or None when metrics are usable."""
    if not isinstance(row, dict):
        return f"not a mapping: {type(row).__name__}"
    for metric in METRIC_NAMES:
        if metric not in row:
            return f"missing {metric}"
        value = row[metric]
        if isinstance(value, bool):
            return f"{metric} is not numeric"
        try:
            score = float(value)
        except (TypeError, ValueError):
            return f"{metric} is not numeric"
        if not math.isfinite(score):
            return f"{metric} is NaN/infinite"
        if score < -_SCORE_EPSILON or score > 1.0 + _SCORE_EPSILON:
            return f"{metric} outside [0, 1]"
    return None


def evaluate_ragas(questions: list[str], answers: list[str], contexts: list[list[str]],
                   ground_truths: list[str]) -> dict:
    _validate_inputs(questions, answers, contexts, ground_truths)
    models = evaluator_models()
    rows = _run_ragas(questions, answers, contexts, ground_truths)
    if not isinstance(rows, list) or len(rows) != len(questions):
        returned = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            f"RAGAS returned {returned} rows for {len(questions)} questions; "
            "refusing partial results")

    problems = [index for index, row in enumerate(rows) if _row_problem(row)]
    attempt = 0
    while problems and attempt < _MAX_METRIC_RETRIES:
        attempt += 1
        print(f"  ⚠️  RAGAS metrics invalid for {len(problems)} question(s) "
              f"(indexes {problems}); retry {attempt}/{_MAX_METRIC_RETRIES}.")
        retried = _run_ragas([questions[i] for i in problems],
                             [answers[i] for i in problems],
                             [contexts[i] for i in problems],
                             [ground_truths[i] for i in problems])
        if not isinstance(retried, list) or len(retried) != len(problems):
            returned = len(retried) if isinstance(retried, list) else type(retried).__name__
            raise ValueError(
                f"RAGAS retry returned {returned} rows for {len(problems)} questions")
        for position, index in enumerate(problems):
            rows[index] = retried[position]
        problems = [index for index, row in enumerate(rows) if _row_problem(row)]

    per_question: list[EvalResult] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"RAGAS row {index} is not a mapping: {type(row).__name__}")
        missing = [metric for metric in METRIC_NAMES if metric not in row]
        if missing:
            raise ValueError(f"RAGAS row {index} is missing metrics: {', '.join(missing)}")
        scores = {metric: _finite_score(row[metric], metric, index) for metric in METRIC_NAMES}
        row_contexts = row.get("contexts", contexts[index])
        if row_contexts is None:
            row_contexts = []
        per_question.append(EvalResult(
            question=str(row.get("question", questions[index])),
            answer=str(row.get("answer", answers[index])),
            contexts=[str(context) for context in row_contexts],
            ground_truth=str(row.get("ground_truth", ground_truths[index])),
            **scores,
        ))
    return {
        "per_question": per_question,
        "metadata": {"evaluator_models": models},
    }
