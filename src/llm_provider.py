"""Resolve the OpenAI-compatible LLM provider used by the repository's live calls.

Priority: OpenCode Go (OPENCODE_GO_API_KEY) -> OpenAI (OPENAI_API_KEY) -> None.
Settings are read from the config module at call time so tests can swap them.

The API key is never logged or embedded in any output; only the provider name,
base URL and model ID are exposed for metadata.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import config

OPENCODE_GO_PROVIDER = "opencode-go"
OPENAI_PROVIDER = "openai"


@dataclass(frozen=True)
class LLMProvider:
    name: str
    base_url: str | None
    model: str
    api_key: str


def resolve_llm_provider() -> LLMProvider | None:
    if config.OPENCODE_GO_API_KEY:
        return LLMProvider(
            name=OPENCODE_GO_PROVIDER,
            base_url=config.OPENCODE_GO_BASE_URL or None,
            model=config.OPENCODE_GO_MODEL,
            api_key=config.OPENCODE_GO_API_KEY,
        )
    if config.OPENAI_API_KEY:
        return LLMProvider(
            name=OPENAI_PROVIDER,
            base_url=None,
            model=config.GENERATION_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
    return None


def opencode_session_id() -> str:
    """Stable session id for OpenCode Go routing (required header since 2026-09)."""
    return config.OPENCODE_GO_SESSION or str(uuid.uuid4())


def create_client(provider: LLMProvider, client=None):
    if client is not None:
        return client
    from openai import OpenAI

    kwargs = {"api_key": provider.api_key}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    if provider.name == OPENCODE_GO_PROVIDER:
        kwargs["default_headers"] = {"x-opencode-session": opencode_session_id()}
    return OpenAI(**kwargs)
