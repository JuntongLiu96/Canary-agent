"""Subagent loader + `Agent` tool.

Each subagents/*.md has frontmatter (name, description, tools?, model?)
and a body that becomes the subagent's extra system prompt.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import yaml

from ..config import SUBAGENTS_DIR
from . import Tool, ToolRegistry

SpawnFn = Callable[[dict[str, Any], str, str], Awaitable[str]]


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm) or {}, body.lstrip("\n")


def load_subagents() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not SUBAGENTS_DIR.exists():
        return out
    for md in SUBAGENTS_DIR.glob("*.md"):
        try:
            meta, body = _parse_frontmatter(md.read_text(encoding="utf-8"))
            out.append({
                "name": meta.get("name", md.stem),
                "description": meta.get("description", ""),
                "tools": meta.get("tools"),
                "model": meta.get("model"),
                "body": body,
            })
        except Exception:
            continue
    return out


def register_agent_tool(
    registry: ToolRegistry,
    subagents: list[dict[str, Any]],
    spawn: SpawnFn,
) -> None:
    by_name = {s["name"]: s for s in subagents}

    async def handler(args: dict[str, Any]) -> str:
        sub_type = args.get("subagent_type", "")
        prompt = args.get("prompt", "")
        description = args.get("description", "")
        s = by_name.get(sub_type)
        if s is None:
            return f"Unknown subagent '{sub_type}'. Available: {', '.join(by_name) or '(none)'}"
        return await spawn(s, description, prompt)

    desc = (
        "Delegate work to a subagent. Runs a nested loop with a fresh context; returns the "
        "subagent's final text. Available: "
        + (", ".join(f"{s['name']} ({s['description']})" for s in subagents) or "(none configured)")
    )
    registry.register(Tool(
        name="Agent",
        description=desc,
        input_schema={
            "type": "object",
            "properties": {
                "subagent_type": {"type": "string"},
                "description": {"type": "string", "description": "Short label for what this run is doing."},
                "prompt": {"type": "string", "description": "The task for the subagent."},
            },
            "required": ["subagent_type", "prompt"],
        },
        handler=handler,
    ))
