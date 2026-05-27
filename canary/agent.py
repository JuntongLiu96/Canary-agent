"""The agent's while-loop. Claude-Code-shaped, intentionally tiny."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .config import DEFAULT_MODEL
from .events import Event
from .providers import pick_provider
from .tools import ToolRegistry


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    model: str = DEFAULT_MODEL
    messages: list[dict[str, Any]] = field(default_factory=list)
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    skills_index: list[dict[str, str]] = field(default_factory=list)
    subagents_index: list[dict[str, str]] = field(default_factory=list)
    mcp_tools_index: list[dict[str, str]] = field(default_factory=list)    # {server, name, description}
    mcp_prompts_index: list[dict[str, str]] = field(default_factory=list)  # {server, name, description}
    allow_tools: list[str] | None = None  # name patterns; None = all
    extra_system: str = ""


def build_system_prompt(session: Session) -> str:
    cwd = os.getcwd()
    parts = [
        "You are Canary, a small test agent. Use tools to accomplish the user's task. "
        "You can call MCP tools (named mcp__server__tool), invoke skills via the Skill tool, "
        "and delegate to subagents via the Agent tool. Be concise.",
        f"Working directory: {cwd}",
    ]
    if session.mcp_tools_index:
        by_server: dict[str, list[dict[str, str]]] = {}
        for t in session.mcp_tools_index:
            by_server.setdefault(t["server"], []).append(t)
        lines = ["# Available MCP servers"]
        for server, tools in by_server.items():
            lines.append(f"## {server}")
            for t in tools:
                desc = ((t.get("description") or "").splitlines() or [""])[0][:120]
                lines.append(f"- mcp__{server}__{t['name']}: {desc}")
            prompts = [p for p in session.mcp_prompts_index if p["server"] == server]
            if prompts:
                lines.append(f"  prompts (user-invokable as /{server}:<name>):")
                for p in prompts:
                    desc = ((p.get("description") or "").splitlines() or [""])[0][:120]
                    lines.append(f"  - {p['name']}: {desc}")
        parts.append("\n".join(lines))
    if session.skills_index:
        lines = ["# Available skills (invoke with the `Skill` tool):"]
        lines += [f"- {s['name']}: {s['description']}" for s in session.skills_index]
        parts.append("\n".join(lines))
    if session.subagents_index:
        lines = ["# Available subagents (invoke with the `Agent` tool):"]
        lines += [f"- {s['name']}: {s['description']}" for s in session.subagents_index]
        parts.append("\n".join(lines))
    if session.extra_system:
        parts.append(session.extra_system)
    return "\n\n".join(parts)


async def run_agent(session: Session, user_input: str | list[dict[str, Any]]) -> AsyncIterator[Event]:
    if isinstance(user_input, str):
        session.messages.append({"role": "user", "content": user_input})
    else:
        session.messages.append({"role": "user", "content": user_input})

    while True:
        provider = pick_provider(session.model)
        try:
            tools = session.registry.schemas(session.allow_tools)
            system = build_system_prompt(session)
        except Exception as e:
            yield {"type": "error", "message": f"prompt build failed: {e!r}"}
            return

        assistant_blocks: list[dict[str, Any]] = []
        cur_text: str | None = None
        partial_args: dict[str, str] = {}
        tool_names: dict[str, str] = {}

        async for ev in provider.stream(session.model, system, session.messages, tools):
            yield ev
            t = ev["type"]
            if t == "text":
                if cur_text is None:
                    cur_text = ""
                cur_text += ev["text"]
            elif t == "tool_use_start":
                if cur_text is not None:
                    assistant_blocks.append({"type": "text", "text": cur_text})
                    cur_text = None
                tool_names[ev["id"]] = ev["name"]
                partial_args[ev["id"]] = ""
            elif t == "tool_use_delta":
                partial_args[ev["id"]] = partial_args.get(ev["id"], "") + ev["partial_json"]
            elif t == "tool_use_stop":
                raw = partial_args.get(ev["id"], "") or "{}"
                try:
                    inp = json.loads(raw)
                except json.JSONDecodeError:
                    inp = {"_raw": raw}
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": ev["id"],
                    "name": tool_names.get(ev["id"], ""),
                    "input": inp,
                })
            elif t == "error":
                if cur_text is not None:
                    assistant_blocks.append({"type": "text", "text": cur_text})
                    cur_text = None
                session.messages.append({"role": "assistant", "content": assistant_blocks or [{"type": "text", "text": ""}]})
                return

        if cur_text is not None:
            assistant_blocks.append({"type": "text", "text": cur_text})

        if not assistant_blocks:
            assistant_blocks = [{"type": "text", "text": ""}]
        session.messages.append({"role": "assistant", "content": assistant_blocks})

        tool_uses = [b for b in assistant_blocks if b["type"] == "tool_use"]
        if not tool_uses:
            return

        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            tool = session.registry.get(tu["name"])
            if tool is None:
                content = f"Unknown tool: {tu['name']}"
                is_err = True
            else:
                try:
                    content = await tool.handler(tu["input"])
                    is_err = False
                except Exception as e:
                    content = f"Tool error: {e!r}"
                    is_err = True
            yield {"type": "tool_result", "id": tu["id"], "content": content, "is_error": is_err}
            results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": content,
                **({"is_error": True} if is_err else {}),
            })

        session.messages.append({"role": "user", "content": results})
