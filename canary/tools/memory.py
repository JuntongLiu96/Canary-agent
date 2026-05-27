"""EvoMem native tools — four-verb memory contract over HTTP.

See https://github.com/.../Self-evolving-agentic-memory docs/11-agent-integration-guide.md
for the canonical contract. Scope (org / repo / agent) and the upstream
bearer token are bound at process startup from env vars — never threaded
through tool arguments, never decided by the model.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from . import Tool


_RETRIEVE_DESC = (
    "Retrieve ranked memory passages distilled from past task trajectories. "
    "Call BEFORE planning when the user's task resembles a recurring pattern "
    "(framework upgrade, refactor shape, bug class), and on hard sub-problems "
    "that smell familiar. Do NOT call every turn."
)

_DISTILL_DESC = (
    "Persist a distilled memory card from a completed task trajectory. Call "
    "ONLY at terminal outcomes — never mid-task. Skip trivial single-step "
    "tasks.\n\n"
    "BEFORE calling: run the distillation prompt in "
    "memory_service/docs/distill_prompt.md against your own LLM with task, "
    "outcome, and trajectory substituted in, and pass the resulting JSON as "
    "`ground_truth`. Without it the server falls back to a low-quality "
    "heuristic. The server deduplicates against existing cards in scope; you "
    "may get back the id of an existing card with rising confidence."
)

_RATE_DESC = (
    "Provide feedback on a memory card you consumed earlier. factual=correct "
    "lifts confidence; incorrect decays it; ≥3 partially_correct ratings "
    "queue re-distillation. relevance=irrelevant penalises the reranker for "
    "similar future queries."
)

_INSPECT_DESC = (
    "Inspect one memory card (full body + audit history). Use for debugging "
    "a suspicious retrieval; not part of the normal flow."
)


def _scope() -> dict[str, str]:
    s = {"org_id": os.environ.get("MEMSVC_ORG", "")}
    repo = os.environ.get("MEMSVC_REPO", "")
    agent = os.environ.get("MEMSVC_AGENT", "")
    if repo:
        s["repo_id"] = repo
    if agent:
        s["agent_id"] = agent
    return s


def _client() -> httpx.AsyncClient:
    base = os.environ.get("MEMSVC_BASE_URL", "http://localhost:9200")
    headers: dict[str, str] = {}
    tok = os.environ.get("MEMSVC_TOKEN", "")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return httpx.AsyncClient(base_url=base, headers=headers, timeout=30)


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} bytes>"


async def _memory_retrieve(args: dict[str, Any]) -> str:
    body = {
        "query": args["query"],
        "k": int(args.get("k", 5)),
        "mode": args.get("mode", "L1"),
        "scope": _scope(),
    }
    async with _client() as c:
        r = await c.post("/memory/retrieve", json=body)
    return _truncate(r.text)


async def _memory_distill(args: dict[str, Any]) -> str:
    trace_summary = args.get("trace_summary", "")
    trace_steps = args.get("trace_steps") or [
        {"role": "assistant", "content": trace_summary}
    ]
    body: dict[str, Any] = {
        "task": args["task"],
        "outcome": args["outcome"],
        "trace_steps": trace_steps,
        "scope": _scope(),
    }
    if args.get("trajectory_id"):
        body["trajectory_id"] = args["trajectory_id"]
    if args.get("ground_truth"):
        body["ground_truth_distillation"] = args["ground_truth"]
    async with _client() as c:
        r = await c.post("/memory/distill", json=body)
    return _truncate(r.text)


async def _memory_rate(args: dict[str, Any]) -> str:
    body: dict[str, Any] = {
        "memory_id": args["memory_id"],
        "scope": _scope(),
    }
    for k in ("factual", "relevance", "note"):
        if args.get(k):
            body[k] = args[k]
    async with _client() as c:
        r = await c.post("/memory/rate", json=body)
    return _truncate(r.text)


async def _memory_inspect(args: dict[str, Any]) -> str:
    scope = _scope()
    headers = {
        "X-Org-Id": scope.get("org_id", ""),
        "X-Repo-Id": scope.get("repo_id", ""),
        "X-Agent-Id": scope.get("agent_id", ""),
    }
    async with _client() as c:
        r = await c.get(f"/memory/{args['memory_id']}", headers=headers)
    return _truncate(r.text)


MEMORY_TOOLS: list[Tool] = [
    Tool(
        name="memory_retrieve",
        description=_RETRIEVE_DESC,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "mode": {"type": "string", "enum": ["L0", "L1", "L2"], "default": "L1"},
            },
            "required": ["query"],
        },
        handler=_memory_retrieve,
    ),
    Tool(
        name="memory_distill",
        description=_DISTILL_DESC,
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "outcome": {"type": "string", "enum": ["success", "failure", "partial"]},
                "trace_summary": {"type": "string"},
                "trace_steps": {"type": "array", "items": {"type": "object"}},
                "trajectory_id": {"type": "string"},
                "ground_truth": {"type": "object"},
            },
            "required": ["task", "outcome", "trace_summary"],
        },
        handler=_memory_distill,
    ),
    Tool(
        name="memory_rate",
        description=_RATE_DESC,
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "factual": {"type": "string", "enum": ["correct", "incorrect", "partially_correct"]},
                "relevance": {"type": "string", "enum": ["relevant", "irrelevant"]},
                "note": {"type": "string"},
            },
            "required": ["memory_id"],
        },
        handler=_memory_rate,
    ),
    Tool(
        name="memory_inspect",
        description=_INSPECT_DESC,
        input_schema={
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
        handler=_memory_inspect,
    ),
]


MEMORY_POLICY_PROMPT = (
    "You have persistent memory across sessions via the memory_retrieve, "
    "memory_distill, memory_rate, and memory_inspect tools. Call "
    "memory_retrieve before planning multi-step or recurring-pattern tasks. "
    "At terminal outcomes call memory_distill; before doing so, run the "
    "distillation prompt at memory_service/docs/distill_prompt.md against "
    "your own model and pass the JSON as `ground_truth`. Distill stable "
    "facts and preferences the user volunteers (favorite language, "
    "allergies, environment quirks) using memory_distill with "
    'task=\"user preference: <topic>\", outcome=\"success\". After acting '
    "on a retrieved card, call memory_rate. Skip memory tools for trivial "
    "chat."
)


def memory_enabled() -> bool:
    """True when MEMSVC_BASE_URL or MEMSVC_ORG is set — wires the four tools."""
    return bool(os.environ.get("MEMSVC_BASE_URL")) or bool(os.environ.get("MEMSVC_ORG"))
