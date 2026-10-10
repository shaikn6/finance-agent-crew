"""Tests for src/llm.py — the single LLM call boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm import LLMError, complete


def _client_returning(text: str) -> AsyncMock:
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text=text)])
    )
    return client


class TestComplete:
    async def test_returns_first_text_block(self):
        client = _client_returning('{"ok": true}')
        result = await complete(client, model="m", prompt="hello")
        assert result == '{"ok": true}'
        client.messages.create.assert_awaited_once_with(
            model="m",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
        )

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("boom"), TimeoutError("slow"), ConnectionError("net down")],
    )
    async def test_any_call_failure_becomes_llm_error(self, error):
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=error)
        with pytest.raises(LLMError, match=str(error)):
            await complete(client, model="m", prompt="hello")

    async def test_empty_content_becomes_llm_error(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=MagicMock(content=[]))
        with pytest.raises(LLMError):
            await complete(client, model="m", prompt="hello")
