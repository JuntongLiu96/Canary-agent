"""Talks to copilot-api's OpenAI-compatible /v1/chat/completions endpoint.

Converts to/from Anthropic-shaped messages so the agent loop only has to
deal with one canonical message format internally (Anthropic blocks).
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..config import COPILOT_BASE_URL
from ..events import Event


class OpenAIProvider:
    kind = "openai"

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[Event]:
        oa_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        oa_messages.extend(_to_openai_messages(messages))

        payload: dict[str, Any] = {
            "model": model,
            "messages": oa_messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        url = f"{COPILOT_BASE_URL}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield {"type": "error", "message": f"HTTP {resp.status_code}: {body.decode(errors='replace')}"}
                    return
                async for ev in _parse_openai_sse(resp):
                    yield ev


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped messages -> OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "user":
            # May contain text blocks and/or tool_result blocks.
            text_parts: list[str] = []
            for block in content:
                if block["type"] == "text":
                    text_parts.append(block["text"])
                elif block["type"] == "tool_result":
                    tc_content = block.get("content", "")
                    if isinstance(tc_content, list):
                        tc_content = "".join(
                            b.get("text", "") for b in tc_content if b.get("type") == "text"
                        )
                    out.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": tc_content,
                    })
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if block["type"] == "text":
                    text_parts.append(block["text"])
                elif block["type"] == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            msg: dict[str, Any] = {"role": "assistant"}
            msg["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
    return out


async def _parse_openai_sse(resp: httpx.Response) -> AsyncIterator[Event]:
    # OpenAI streams tool_calls in fragments: a per-index dict with id+function.name on first
    # chunk, then function.arguments deltas. We track per-index state and emit Canary events.
    tc_state: dict[int, dict[str, Any]] = {}  # index -> {id, name, started, args_buf}
    finish_reason: str | None = None

    async for raw_line in resp.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = obj.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        delta = ch.get("delta") or {}

        text = delta.get("content")
        if text:
            yield {"type": "text", "text": text}

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            st = tc_state.setdefault(idx, {"id": None, "name": None, "started": False, "args_buf": ""})
            if tc.get("id"):
                st["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                st["name"] = fn["name"]
            if not st["started"] and st["id"] and st["name"]:
                st["started"] = True
                yield {"type": "tool_use_start", "id": st["id"], "name": st["name"]}
            args_frag = fn.get("arguments")
            if args_frag:
                st["args_buf"] += args_frag
                if st["started"]:
                    yield {"type": "tool_use_delta", "id": st["id"], "partial_json": args_frag}

        if ch.get("finish_reason"):
            finish_reason = ch["finish_reason"]

    for st in tc_state.values():
        if st["started"]:
            yield {"type": "tool_use_stop", "id": st["id"]}

    yield {"type": "message_stop", "stop_reason": finish_reason}
