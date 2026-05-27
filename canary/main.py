"""FastAPI entry point.

Run:  uv run uvicorn canary.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Session, run_agent
from .config import (
    COPILOT_BASE_URL,
    CONFIG_DIR,
    DEFAULT_MODEL,
    SKILLS_DIR,
    SUBAGENTS_DIR,
    WEB_DIR,
    load_settings,
)
from .mcp_client import MCPHub
from .skills import load_skills, register_skill_tool
from .tools import ToolRegistry
from .tools.mcp_tools import register_mcp_tools
from .tools.native import NATIVE_TOOLS
from .tools.subagent import load_subagents, register_agent_tool

log = logging.getLogger("canary")
logging.basicConfig(level=logging.INFO)

sessions: dict[str, Session] = {}
queues: dict[str, asyncio.Queue] = {}
mcp_hub: MCPHub | None = None
skills_index: list[dict[str, str]] = []
subagents_index: list[dict[str, str]] = []
shared_registry: ToolRegistry = ToolRegistry()


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in NATIVE_TOOLS:
        reg.register(t)
    register_skill_tool(reg, skills_index)
    if mcp_hub is not None:
        register_mcp_tools(reg, mcp_hub)
    register_agent_tool(reg, subagents_index, _spawn_subagent)
    return reg


async def _spawn_subagent(subagent_def: dict[str, Any], description: str, prompt: str) -> str:
    """Run a nested agent loop. Returns the final assistant text."""
    sub = Session(
        model=subagent_def.get("model") or DEFAULT_MODEL,
        registry=_build_registry(),
        skills_index=skills_index,
        subagents_index=[],  # subagents can't recurse by default
        allow_tools=subagent_def.get("tools"),
        extra_system=subagent_def.get("body", ""),
    )
    final_text: list[str] = []
    async for ev in run_agent(sub, prompt):
        if ev["type"] == "text":
            final_text.append(ev["text"])
    return "".join(final_text) or "(subagent returned no text)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_hub, skills_index, subagents_index
    skills_index = load_skills()
    subagents_index = load_subagents()
    mcp_hub = MCPHub()
    try:
        await mcp_hub.start()
    except Exception as e:
        log.warning("MCP startup failed: %s", e)
    log.info("Canary ready. Skills=%d Subagents=%d MCP-tools=%d MCP-prompts=%d",
             len(skills_index), len(subagents_index),
             len(mcp_hub.tools) if mcp_hub else 0,
             len(mcp_hub.prompts) if mcp_hub else 0)
    yield
    if mcp_hub is not None:
        await mcp_hub.stop()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str | None = None
    blocks: list[dict[str, Any]] | None = None  # for prompts/get injection
    model: str | None = None


def _get_or_create_session(sid: str, model: str | None) -> Session:
    sess = sessions.get(sid)
    if sess is None:
        sess = Session(id=sid, model=model or DEFAULT_MODEL)
        sessions[sid] = sess
    if sid not in queues:
        queues[sid] = asyncio.Queue()
    if model:
        sess.model = model
    sess.registry = _build_registry()
    sess.skills_index = skills_index
    sess.subagents_index = subagents_index
    return sess


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sess = _get_or_create_session(req.session_id, req.model)
    q = queues[req.session_id]
    user_input = req.blocks if req.blocks else (req.message or "")

    async def run():
        try:
            async for ev in run_agent(sess, user_input):
                await q.put(ev)
        except Exception as e:
            await q.put({"type": "error", "message": repr(e)})
        finally:
            await q.put({"type": "done"})

    asyncio.create_task(run())
    return {"ok": True}


@app.get("/api/stream/{session_id}")
async def stream(session_id: str):
    if session_id not in queues:
        queues[session_id] = asyncio.Queue()

    async def gen():
        q = queues[session_id]
        while True:
            ev = await q.get()
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get("type") == "done":
                # don't close — UI may send next message; but flush a heartbeat
                continue

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/models")
async def models():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{COPILOT_BASE_URL}/v1/models")
        r.raise_for_status()
        return r.json()


@app.get("/api/state/{session_id}")
async def state(session_id: str):
    sess = sessions.get(session_id)
    if sess is None:
        raise HTTPException(404)
    return {"id": sess.id, "model": sess.model, "messages": sess.messages}


@app.post("/api/reset/{session_id}")
async def reset(session_id: str):
    if session_id in sessions:
        sessions[session_id].messages.clear()
    return {"ok": True}


@app.get("/api/prompts")
async def prompts():
    if mcp_hub is None:
        return {"prompts": []}
    return {"prompts": mcp_hub.prompts}


class PromptGetRequest(BaseModel):
    server: str
    name: str
    arguments: dict[str, str] = {}


@app.post("/api/prompts/get")
async def prompts_get(req: PromptGetRequest):
    if mcp_hub is None:
        raise HTTPException(503, "MCP not initialized")
    messages = await mcp_hub.get_prompt(req.server, req.name, req.arguments)
    # Convert MCP prompt messages -> Anthropic-shaped content blocks for one user turn.
    blocks: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text", "")
        prefix = "" if role == "user" else f"[{role}] "
        if text:
            blocks.append({"type": "text", "text": prefix + text})
    return {"blocks": blocks}


@app.post("/api/mcp/refresh")
async def mcp_refresh():
    if mcp_hub is None:
        raise HTTPException(503)
    await mcp_hub.refresh()
    return {"tools": len(mcp_hub.tools), "prompts": len(mcp_hub.prompts)}


# ───────── Management: edit MCP config, skills, subagents ─────────

@app.get("/api/manage/mcp")
async def manage_mcp_get():
    path = CONFIG_DIR / "mcp.json"
    return {"text": path.read_text(encoding="utf-8") if path.exists() else '{\n  "mcpServers": {}\n}\n'}


class TextPayload(BaseModel):
    text: str


@app.put("/api/manage/mcp")
async def manage_mcp_put(req: TextPayload):
    try:
        json.loads(req.text)  # validate
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    (CONFIG_DIR / "mcp.json").write_text(req.text, encoding="utf-8")
    return {"ok": True, "note": "Restart server to apply MCP changes (stdio servers can't be hot-reloaded safely)."}


def _list_md_dir(base, glob: str) -> list[dict[str, str]]:
    if not base.exists():
        return []
    return [{"name": p.stem if glob == "*.md" else p.parent.name, "path": str(p.relative_to(base))} for p in base.glob(glob)]


@app.get("/api/manage/skills")
async def manage_skills_list():
    return {"items": _list_md_dir(SKILLS_DIR, "*/SKILL.md")}


@app.get("/api/manage/skills/{name}")
async def manage_skills_get(name: str):
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.exists():
        return {"text": "---\nname: " + name + "\ndescription: \n---\n"}
    return {"text": path.read_text(encoding="utf-8")}


@app.put("/api/manage/skills/{name}")
async def manage_skills_put(name: str, req: TextPayload):
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    if not safe:
        raise HTTPException(400, "Invalid name")
    path = SKILLS_DIR / safe / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(req.text, encoding="utf-8")
    _reload_skills_subagents()
    return {"ok": True}


@app.delete("/api/manage/skills/{name}")
async def manage_skills_delete(name: str):
    path = SKILLS_DIR / name / "SKILL.md"
    if path.exists():
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
    _reload_skills_subagents()
    return {"ok": True}


@app.get("/api/manage/subagents")
async def manage_sub_list():
    return {"items": _list_md_dir(SUBAGENTS_DIR, "*.md")}


@app.get("/api/manage/subagents/{name}")
async def manage_sub_get(name: str):
    path = SUBAGENTS_DIR / f"{name}.md"
    if not path.exists():
        return {"text": "---\nname: " + name + "\ndescription: \ntools: []\n---\n"}
    return {"text": path.read_text(encoding="utf-8")}


@app.put("/api/manage/subagents/{name}")
async def manage_sub_put(name: str, req: TextPayload):
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    if not safe:
        raise HTTPException(400, "Invalid name")
    path = SUBAGENTS_DIR / f"{safe}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(req.text, encoding="utf-8")
    _reload_skills_subagents()
    return {"ok": True}


@app.delete("/api/manage/subagents/{name}")
async def manage_sub_delete(name: str):
    path = SUBAGENTS_DIR / f"{name}.md"
    if path.exists():
        path.unlink()
    _reload_skills_subagents()
    return {"ok": True}


def _reload_skills_subagents() -> None:
    global skills_index, subagents_index
    skills_index = load_skills()
    subagents_index = load_subagents()


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
