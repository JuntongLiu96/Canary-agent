"""Talks to copilot-api's Anthropic-compatible /v1/messages endpoint.

Forwards an Anthropic-shaped payload and translates the SSE event stream
into Canary's normalized events. Used for `claude-*` models.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..config import COPILOT_BASE_URL
from ..events import Event


class AnthropicProvider:
    kind = "anthropic"

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[Event]:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 8192,
            "system": system,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t["input_schema"],
                }
                for t in tools
            ]

        url = f"{COPILOT_BASE_URL}/v1/messages"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield {"type": "error", "message": f"HTTP {resp.status_code}: {body.decode(errors='replace')}"}
                    return
                async for ev in _parse_anthropic_sse(resp):
                    yield ev


async def _parse_anthropic_sse(resp: httpx.Response) -> AsyncIterator[Event]:
    """Convert Anthropic SSE events into normalized Canary events."""
    block_kinds: dict[int, str] = {}    # index -> "text" | "tool_use"
    block_ids: dict[int, str] = {}      # index -> tool_use id
    stop_reason: str | None = None

    async for raw_line in resp.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if not data:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = obj.get("type")

        if etype == "content_block_start":
            idx = obj["index"]
            block = obj.get("content_block", {})
            btype = block.get("type")
            block_kinds[idx] = btype
            if btype == "tool_use":
                bid = block.get("id", f"toolu_{idx}")
                block_ids[idx] = bid
                yield {"type": "tool_use_start", "id": bid, "name": block.get("name", "")}

        elif etype == "content_block_delta":
            idx = obj["index"]
            delta = obj.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                yield {"type": "text", "text": delta.get("text", "")}
            elif dtype == "input_json_delta":
                bid = block_ids.get(idx, "")
                yield {"type": "tool_use_delta", "id": bid, "partial_json": delta.get("partial_json", "")}

        elif etype == "content_block_stop":
            idx = obj["index"]
            if block_kinds.get(idx) == "tool_use":
                yield {"type": "tool_use_stop", "id": block_ids.get(idx, "")}

        elif etype == "message_delta":
            sr = obj.get("delta", {}).get("stop_reason")
            if sr:
                stop_reason = sr

        elif etype == "message_stop":
            yield {"type": "message_stop", "stop_reason": stop_reason}
            return
