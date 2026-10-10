"""The single boundary through which agents talk to the LLM."""

from __future__ import annotations

from typing import Any

import anthropic

from .config import ConfigurationError, get_settings
from .offline import OfflineLLMClient


class LLMError(RuntimeError):
    """Raised when an LLM call fails for any reason (API, network, bad payload)."""


def get_llm_client() -> Any:
    """Return the Anthropic client, or the deterministic fake in offline mode."""
    settings = get_settings()
    if settings.offline_mode:
        return OfflineLLMClient()
    if not settings.anthropic_api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. Add it to the environment or .env, "
            "or set OFFLINE_MODE=1 to run with canned responses."
        )
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def complete(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Send one user prompt and return the text of the first content block.

    Every failure — provider error, timeout, dropped connection, or an
    unexpected response shape — is re-raised as ``LLMError`` so each agent
    can degrade with one narrow ``except`` instead of crashing the run.
    """
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as exc:
        raise LLMError(str(exc) or type(exc).__name__) from exc
