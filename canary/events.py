"""Normalized event stream used by both providers and the SSE channel.

Events are plain dicts so they JSON-serialize trivially. Schema:

  {"type": "text",            "text": str}
  {"type": "tool_use_start",  "id": str, "name": str}
  {"type": "tool_use_delta",  "id": str, "partial_json": str}
  {"type": "tool_use_stop",   "id": str}
  {"type": "tool_result",     "id": str, "content": str, "is_error": bool}
  {"type": "message_stop",    "stop_reason": str | None}
  {"type": "error",           "message": str}
"""
from __future__ import annotations

from typing import Any

Event = dict[str, Any]
