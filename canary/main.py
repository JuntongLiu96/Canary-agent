"""FastAPI entry point.

Run:  uv run uvicorn canary.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
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
from .tools.memory import (
    MEMORY_POLICY_PROMPT, MEMORY_TOOLS, memory_enabled,
    set_request_scope, reset_request_scope,
)
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

# Prometheus metrics
cart_aborts_total = Counter('cart_aborts_total', 'Total number of cart aborts')


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in NATIVE_TOOLS:
        reg.register(t)
    if memory_enabled():
        for t in MEMORY_TOOLS:
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
        mcp_tools_index=mcp_hub.tools if mcp_hub else [],
        mcp_prompts_index=mcp_hub.prompts if mcp_hub else [],
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
    if mcp_hub is not None:
        sess.mcp_tools_index = mcp_hub.tools
        sess.mcp_prompts_index = mcp_hub.prompts
    if memory_enabled() and MEMORY_POLICY_PROMPT not in sess.extra_system:
        sess.extra_system = (sess.extra_system + "\n\n" + MEMORY_POLICY_PROMPT).strip()
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


# ───────── Cart endpoints ─────────

@app.post("/cart/abort")
async def cart_abort():
    """Abort a cart and increment the Prometheus counter."""
    cart_aborts_total.inc()
    return {"ok": True, "message": "Cart aborted"}


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


# ───────── AgenticEval HTTP adapter (Option A) ─────────
# https://github.com/.../agentic-eval — expects /eval/health, /eval/run, /eval/judge

class EvalRunRequest(BaseModel):
    prompt: str
    metadata: dict[str, Any] = {}


class EvalJudgeRequest(BaseModel):
    messages: list[dict[str, Any]]


@app.get("/eval/health")
async def eval_health():
    return {"ok": True}


def _flatten_for_eval(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped (role + content blocks) -> flat [{role, content}] for eval."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        text_parts: list[str] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                args_json = json.dumps(block.get("input", {}), ensure_ascii=False)
                text_parts.append(f"[tool_use {block.get('name','')}({args_json})]")
            elif btype == "tool_result":
                tc = block.get("content", "")
                if isinstance(tc, list):
                    tc = "".join(b.get("text", "") for b in tc if b.get("type") == "text")
                tool_role = "tool"
                out.append({"role": tool_role, "content": str(tc)})
                continue
        if text_parts:
            out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def _compute_eval_metadata(
    messages: list[dict[str, Any]],
    must_retrieve_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Self-report agent-side metrics derivable from session.messages.

    Reports only what the agent itself knows. Metrics that require an oracle
    (n_reduction, files_recall vs expected, fama, etc.) are the harness's job —
    see docs/11-agent-integration-guide.md.

    When the case declares ``must_retrieve_memory_ids`` (the cards that SHOULD
    surface), we also compute rank-aware retrieval quality from the passage
    ids the agent actually saw: ``retrieval_recall_at_k`` (fraction of expected
    ids present anywhere in the agent's retrievals) and ``retrieval_mrr`` (mean
    reciprocal rank of the first expected id across retrieval calls). These are
    strictly better than the binary ``retrieval_hit_rate``, which a case can
    pass while retrieving the wrong card.
    """
    trajectory_steps = 0
    tool_calls = 0
    files_touched: set[str] = set()
    memory_retrievals: list[dict[str, Any]] = []
    memory_distills: list[dict[str, Any]] = []
    last_tool_use: dict[str, Any] | None = None

    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            if m["role"] == "assistant":
                trajectory_steps += 1
            continue
        for block in content:
            btype = block.get("type")
            if btype == "tool_use":
                tool_calls += 1
                trajectory_steps += 1
                name = block.get("name", "")
                args = block.get("input", {}) or {}
                last_tool_use = {"name": name, "args": args}
                if name in ("write_file", "edit_file", "str_replace") and args.get("path"):
                    files_touched.add(args["path"])
                elif name == "memory_distill":
                    memory_distills.append({"task": args.get("task"), "outcome": args.get("outcome")})
            elif btype == "tool_result" and last_tool_use:
                if last_tool_use["name"] == "memory_retrieve":
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        raw = "".join(b.get("text", "") for b in raw if b.get("type") == "text")
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        memory_retrievals.append({
                            "query": last_tool_use["args"].get("query", ""),
                            "passages": parsed.get("passages", []) if isinstance(parsed, dict) else [],
                        })
                    except (json.JSONDecodeError, TypeError):
                        memory_retrievals.append({"query": last_tool_use["args"].get("query", ""), "passages": []})
                last_tool_use = None

    retrieval_hit_rate = (
        sum(1 for r in memory_retrievals if r["passages"]) / len(memory_retrievals)
        if memory_retrievals else 0.0
    )
    out_meta: dict[str, Any] = {
        "trajectory_steps": trajectory_steps,
        "tool_calls_count": tool_calls,
        "files_touched": sorted(files_touched),
        "memory_retrievals_count": len(memory_retrievals),
        "memory_distills_count": len(memory_distills),
        "retrieval_hit_rate": retrieval_hit_rate,
        "memory_retrievals": memory_retrievals,
        "memory_distills": memory_distills,
    }

    # Rank-aware retrieval quality vs the expected card ids (oracle supplied by
    # the case via metadata.must_retrieve_memory_ids). Only emitted when the
    # case declares an expectation — otherwise these keys stay absent so a
    # programmatic scorer doesn't read a vacuous 0.
    want = [str(i) for i in (must_retrieve_memory_ids or []) if i]
    if want:
        want_set = set(want)
        # Union of all retrieved ids across the agent's retrieval calls.
        retrieved_ids: set[str] = set()
        for r in memory_retrievals:
            for p in r.get("passages", []):
                mid = p.get("memory_id") if isinstance(p, dict) else None
                if mid:
                    retrieved_ids.add(str(mid))
        recall = len(want_set & retrieved_ids) / len(want_set)
        # MRR: best (lowest) rank at which ANY expected id appears, per call;
        # average the reciprocal ranks over calls that returned passages.
        rr_values: list[float] = []
        for r in memory_retrievals:
            passages = r.get("passages", []) or []
            if not passages:
                continue
            best_rank = None
            for rank, p in enumerate(passages, start=1):
                mid = str(p.get("memory_id")) if isinstance(p, dict) else None
                if mid in want_set:
                    best_rank = rank
                    break
            rr_values.append(1.0 / best_rank if best_rank else 0.0)
        out_meta["retrieval_recall_at_k"] = recall
        out_meta["retrieval_mrr"] = (sum(rr_values) / len(rr_values)) if rr_values else 0.0
    return out_meta


async def _run_scheduled_erasures(
    metadata: dict[str, Any], turn_index: int, scope: dict[str, str] | None
) -> None:
    """Execute GDPR DELETEs scheduled to fire after ``turn_index``.

    The exporter emits ``metadata.erasure_schedule = {"<turn_index>": [{memory_id,
    reason}, ...]}`` from the case's ``post_session_N`` blocks. Erasure is a
    system/compliance action — the agent never deletes. We hit
    ``DELETE /memory/{id}`` with the case scope so the NEXT turn's retrieval can't
    surface the erased facts and the tombstone probe sees erased=true + audit_hash.
    """
    schedule = metadata.get("erasure_schedule") or {}
    deletes = schedule.get(str(turn_index)) if isinstance(schedule, dict) else None
    if not deletes:
        return
    s = scope or {}
    headers = {"X-Org-Id": str(s.get("org_id") or "")}
    if s.get("repo_id"):
        headers["X-Repo-Id"] = str(s["repo_id"])
    if s.get("agent_id"):
        headers["X-Agent-Id"] = str(s["agent_id"])
    tok = os.environ.get("MEMSVC_TOKEN", "")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    memsvc = os.environ.get("MEMSVC_BASE_URL", "http://localhost:9200")
    async with httpx.AsyncClient(base_url=memsvc, timeout=30, headers=headers) as c:
        for d in deletes:
            mid = d.get("memory_id") if isinstance(d, dict) else None
            if not mid:
                continue
            try:
                await c.delete(f"/memory/{mid}", params={"reason": d.get("reason", "user-request")})
            except Exception as e:
                log.warning("scheduled erasure of %s failed: %s", mid, e)


@app.post("/eval/run")
async def eval_run(req: EvalRunRequest):
    """Fresh agent session for each eval case."""
    sess = Session(model=DEFAULT_MODEL)
    sess.registry = _build_registry()
    sess.skills_index = skills_index
    sess.subagents_index = subagents_index
    if mcp_hub is not None:
        sess.mcp_tools_index = mcp_hub.tools
        sess.mcp_prompts_index = mcp_hub.prompts
    if memory_enabled():
        sess.extra_system = MEMORY_POLICY_PROMPT

    # Materialize env_state.files (from testcase metadata) into a tempdir
    # and chdir there so write_file / read_file / edit_file tools see them
    # as the working tree. See docs/11-agent-integration-guide.md
    # (env_state.files contract).
    env_state = (req.metadata or {}).get("env_state") or {}
    files = env_state.get("files") if isinstance(env_state, dict) else None
    tmpdir: str | None = None
    prev_cwd: str | None = None
    if isinstance(files, dict) and files:
        tmpdir = tempfile.mkdtemp(prefix="evomem-case-")
        for rel, content in files.items():
            dest = Path(tmpdir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content if isinstance(content, str) else str(content), encoding="utf-8")
        prev_cwd = os.getcwd()
        os.chdir(tmpdir)

    # AE-3: per-request memory scope from testcase_metadata.scope; falls back to
    # MEMSVC_* env. Without this every distill lands under the env-var scope and
    # cross-case retrieval is empty.
    scope = (req.metadata or {}).get("scope") or None
    scope_token = set_request_scope(scope) if scope else None

    # Which turn of a multi-session case is this? The orchestrator sends
    # turn_index per turn (0-based). Ingestion seeding and between-session
    # system actions (GDPR erasure) are turn-sensitive: seed only once, and
    # run a post_session_N erasure after the turn it follows.
    turn_index = int((req.metadata or {}).get("turn_index") or 0)

    # Expected card ids for rank-aware retrieval quality (recall@k / MRR). The
    # exporter lifts these from the case's expected_retrieval.must_retrieve_
    # memory_ids into testcase metadata; _compute_eval_metadata joins them
    # against the passages the agent actually retrieved.
    must_retrieve = (req.metadata or {}).get("must_retrieve_memory_ids") or []

    # NOTE: case.ingestion[] (prior-run memory the case wants preset) is NOT
    # seeded here. Seeding through the agent bridge would hand the case oracle
    # to the very system under test — a cheating path. The eval harness seeds
    # memsvc out-of-band before the turn loop (agentic-eval
    # orchestrator -> harness_probes.seed_memory_store), and the standalone
    # evomem-seed CLI does the same for ad-hoc runs. The agent only ever writes
    # memory via its own memory_distill tool (self_distillation). See
    # docs/11-agent-integration-guide.md "Seeding is out-of-band".
    try:
        try:
            async for _ev in run_agent(sess, req.prompt):
                pass
        except Exception as e:
            flat = _flatten_for_eval(sess.messages)
            meta = _compute_eval_metadata(sess.messages, must_retrieve)
            meta["error"] = repr(e)
            return {"messages": flat, "metadata": meta}
        flat = _flatten_for_eval(sess.messages)
        meta = _compute_eval_metadata(sess.messages, must_retrieve)
        # System/compliance erasure: run any DELETEs scheduled to fire AFTER this
        # turn (GN-004 right-to-be-forgotten). This is NOT an agent action — the
        # agent has no delete tool; the harness invokes DELETE /memory/{id} on
        # the data-subject's behalf so the NEXT turn's retrieval can't surface
        # the erased facts. See docs/11-agent-integration-guide.md.
        await _run_scheduled_erasures(req.metadata or {}, turn_index, scope)
        return {"messages": flat, "metadata": meta}
    finally:
        if scope_token is not None:
            reset_request_scope(scope_token)
        if prev_cwd:
            os.chdir(prev_cwd)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/eval/judge")
async def eval_judge(req: EvalJudgeRequest):
    """Raw LLM call — no agent loop, no system prompt injection."""
    model = DEFAULT_MODEL
    # Use OpenAI-compatible endpoint always (judge expects /chat/completions semantics).
    payload = {
        "model": model,
        "messages": req.messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{COPILOT_BASE_URL}/v1/chat/completions", json=payload)
        if r.status_code != 200:
            raise HTTPException(500, f"judge LLM error: {r.status_code} {r.text}")
        obj = r.json()
    content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"content": content}


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
