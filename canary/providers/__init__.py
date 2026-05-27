from __future__ import annotations

from .anthropic import AnthropicProvider
from .openai import OpenAIProvider


def pick_provider(model: str):
    if model.startswith("claude-"):
        return AnthropicProvider()
    return OpenAIProvider()


__all__ = ["pick_provider", "AnthropicProvider", "OpenAIProvider"]
