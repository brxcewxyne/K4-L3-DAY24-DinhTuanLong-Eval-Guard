"""One-request live smoke test for the configured LLM provider.

Sends exactly one chat completion (through src.m5_enrichment on one chunk) and
prints the provider, model, method and a text snippet. Requires
OPENCODE_GO_API_KEY (or OPENAI_API_KEY); the key is never printed.

Usage (Windows PowerShell):
    $env:OPENCODE_GO_API_KEY = "<key>"
    python scripts/smoke_llm.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_provider import resolve_llm_provider
from src.m5_enrichment import enrich_chunks

SAMPLE = {
    "text": "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.",
    "metadata": {"source": "smoke_test"},
}


def main() -> int:
    provider = resolve_llm_provider()
    if provider is None:
        print("No LLM provider configured. Set OPENCODE_GO_API_KEY (or OPENAI_API_KEY) first.")
        return 1
    print(f"provider={provider.name} model={provider.model} "
          f"base_url={provider.base_url or '(provider default)'}")
    result = enrich_chunks([SAMPLE])[0]
    print(f"method={result.method} provider={result.provider} model={result.model}")
    print(f"enriched_text[:120]={result.enriched_text[:120]!r}")
    if result.method != "llm_combined":
        print("WARNING: the call did not go through the LLM path.")
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
