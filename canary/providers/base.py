from __future__ import annotations

from typing import Any, AsyncIterator, Protocol

from ..events import Event


class Provider(Protocol):
    """A provider streams normalized events for one assistant turn."""

    kind: str  # "anthropic" or "openai"

    def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[Event]: ...
