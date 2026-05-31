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
    "Persist a distilled memory card from a completed task trajectory. "
    "REQUIRED — not optional — at every terminal outcome matching the "
    "triggers below. Skipping is a defect.\n\n"
    "MUST call when: (a) you reported task completion to the user after "
    "edits/tool calls/investigation; (b) you abandoned a task after a "
    "non-trivial action and learned something (dead-end, pitfall, "
    "constraint); (c) the user volunteered a stable fact, preference, "
    "allergy, or environment quirk — pass task=\"user preference: "
    "<topic>\", outcome=\"success\", trace_summary=<fact verbatim>; "
    "(d) you noticed a recurring-pattern moment — distill the pattern.\n\n"
    "Do NOT call: after a single read-only retrieval with no action; "
    "after pure Q&A with no edits; mid-task before outcome is known; "
    "on greetings or trivial chat.\n\n"
    "BEFORE calling: run the distillation prompt in "
    "memory_service/docs/distill_prompt.md against your own LLM with task, "
    "outcome, and trajectory substituted in, and pass the resulting JSON as "
    "`self_distillation` (your own distillation of what you just did). "
    "Without it the server falls back to a low-quality heuristic. The server "
    "deduplicates against existing cards in scope; you may get back the id of "
    "an existing card with rising confidence."
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
    # AE-3: per-request override set by /eval/run (via set_request_scope)
    override = _SCOPE_OVERRIDE.get()
    if override:
        return dict(override)
    s = {"org_id": os.environ.get("MEMSVC_ORG", "")}
    repo = os.environ.get("MEMSVC_REPO", "")
    agent = os.environ.get("MEMSVC_AGENT", "")
    if repo:
        s["repo_id"] = repo
    if agent:
        s["agent_id"] = agent
    return s


from contextvars import ContextVar
_SCOPE_OVERRIDE: ContextVar[dict[str, str] | None] = ContextVar("memsvc_scope", default=None)

def set_request_scope(scope: dict[str, str] | None) -> object:
    """Set the per-request memory-service scope. Returns a token for reset()."""
    return _SCOPE_OVERRIDE.set(scope or None)

def reset_request_scope(token: object) -> None:
    _SCOPE_OVERRIDE.reset(token)  # type: ignore[arg-type]


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


def _working_tree_files(limit: int = 2000) -> list[str]:
    """Relative paths of the current working tree, for JIT staleness checks.

    The memory service runs in its own process and can't see the agent's
    tempdir, so it relies on this manifest to verify cited file paths still
    exist (CA-007). Paths are POSIX-style relative to cwd to match how cases
    declare key_files. VCS/junk dirs are skipped; the list is capped so a huge
    checkout can't bloat the request.
    """
    import os as _os

    root = _os.getcwd()
    skip = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}
    out: list[str] = []
    for dirpath, dirnames, filenames in _os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            rel = _os.path.relpath(_os.path.join(dirpath, fn), root)
            out.append(rel.replace(_os.sep, "/"))
            if len(out) >= limit:
                return out
    return out


async def _memory_retrieve(args: dict[str, Any]) -> str:
    body = {
        "query": args["query"],
        "k": int(args.get("k", 5)),
        "mode": args.get("mode", "L1"),
        "scope": _scope(),
        "repo_files": _working_tree_files(),
    }
    async with _client() as c:
        r = await c.post("/memory/retrieve", json=body)
    return _truncate(r.text)


_OUTCOME_ALIASES = {
    "success": "success", "succeeded": "success", "completed": "success",
    "complete": "success", "done": "success", "ok": "success",
    "resolved": "success", "fixed": "success", "pass": "success", "passed": "success",
    "failure": "failure", "failed": "failure", "fail": "failure",
    "error": "failure", "abandoned": "failure",
    "partial": "partial", "partially_correct": "partial",
    "partially": "partial", "incomplete": "partial", "mixed": "partial",
}


def _norm_outcome(raw: Any) -> str:
    """Coerce a model-supplied outcome to the server enum.

    The model frequently emits values outside {success,failure,partial}
    (e.g. "completed", "done"). A strict-enum 422 silently drops the
    distilled card and breaks the self-evolving loop, so normalise here.
    """
    return _OUTCOME_ALIASES.get(str(raw or "").strip().lower(), "success")


def _norm_self_distillation(gt: Any, fallback_summary: str = "") -> dict[str, Any] | None:
    """Sanitise a model-supplied self_distillation so it matches the wire schema.

    This is the agent's OWN distillation of the trajectory it just ran (Model B
    — the primary production write path). It is NOT the case oracle: the agent
    has no access to ground_truth_distillation, which is seeded out-of-band by
    the harness. Same shape, different provenance.

    Drops a non-numeric confidence and forces the list-typed fields to lists,
    so a type deviation can't 422 the whole distill.

    The model frequently emits the actual fact under freeform keys the wire
    schema doesn't know (``insight``, ``context``, ``summary``, ``value`` …)
    rather than in ``solution_steps``. Dropping those silently strips the only
    load-bearing content from the card — the body then degrades to the bare
    task label and retrieval surfaces a contentless card. So we salvage those
    string values (and ``fallback_summary``) into ``solution_steps`` when the
    model didn't populate it itself.
    """
    if not isinstance(gt, dict):
        gt = {}
    out: dict[str, Any] = {}
    if gt.get("task_pattern") is not None:
        out["task_pattern"] = str(gt["task_pattern"])
    if gt.get("memory_id") is not None:
        out["memory_id"] = str(gt["memory_id"])
    for key in ("key_files", "solution_steps", "pitfalls", "share_with"):
        v = gt.get(key)
        if isinstance(v, list):
            out[key] = [str(x) for x in v]
        elif isinstance(v, str) and v:
            out[key] = [v]
    conf = gt.get("confidence")
    if isinstance(conf, (int, float)):
        out["confidence"] = float(conf)
    elif isinstance(conf, str):
        try:
            out["confidence"] = float(conf)
        except ValueError:
            pass

    # Salvage the fact when solution_steps is empty.
    if not out.get("solution_steps"):
        salvaged: list[str] = []
        for key in ("insight", "context", "summary", "body", "value", "fact",
                    "note", "description", "details", "content"):
            v = gt.get(key)
            if isinstance(v, str) and v.strip():
                salvaged.append(v.strip())
        if not salvaged and fallback_summary.strip():
            salvaged.append(fallback_summary.strip())
        if salvaged:
            out["solution_steps"] = salvaged
    return out or None


_VALID_ROLES = {"user", "assistant", "tool", "system"}


def _norm_trace_steps(raw: Any, fallback_summary: str) -> list[dict[str, Any]]:
    """Coerce model-supplied trace_steps to the server's TraceStep schema.

    The server pins role to {user,assistant,tool,system} and content/result to
    strings; a stray role or a dict-valued content would 422 the whole distill.
    """
    if not isinstance(raw, list) or not raw:
        return [{"role": "assistant", "content": fallback_summary}]
    out: list[dict[str, Any]] = []
    for st in raw:
        if not isinstance(st, dict):
            out.append({"role": "assistant", "content": str(st)})
            continue
        step: dict[str, Any] = {}
        role = str(st.get("role", "")).strip().lower()
        step["role"] = role if role in _VALID_ROLES else "assistant"
        if st.get("content") is not None:
            step["content"] = st["content"] if isinstance(st["content"], str) else str(st["content"])
        if st.get("name") is not None:
            step["name"] = str(st["name"])
        if isinstance(st.get("args"), dict):
            step["args"] = st["args"]
        if st.get("result") is not None:
            step["result"] = st["result"] if isinstance(st["result"], str) else str(st["result"])
        out.append(step)
    return out


async def _memory_distill(args: dict[str, Any]) -> str:
    trace_summary = args.get("trace_summary", "")
    body: dict[str, Any] = {
        "task": str(args.get("task", "")),
        "outcome": _norm_outcome(args.get("outcome")),
        "trace_steps": _norm_trace_steps(args.get("trace_steps"), trace_summary),
        "scope": _scope(),
    }
    if args.get("trajectory_id"):
        body["trajectory_id"] = args["trajectory_id"]
    # Agent's OWN distillation -> self_distillation wire field. The agent has
    # no access to ground_truth_distillation (seed-only, harness-written).
    sd = _norm_self_distillation(args.get("self_distillation"), trace_summary)
    if sd:
        body["self_distillation"] = sd
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
    # httpx rejects None header values, and repo_id / agent_id are optional in
    # scope — emit only the headers we actually have, coerced to str.
    headers = {
        "X-Org-Id": str(scope.get("org_id") or ""),
    }
    if scope.get("repo_id"):
        headers["X-Repo-Id"] = str(scope["repo_id"])
    if scope.get("agent_id"):
        headers["X-Agent-Id"] = str(scope["agent_id"])
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
                "self_distillation": {"type": "object"},
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
    "At the end of any task that involved edits or multi-step "
    "investigation, you MUST call memory_distill with the full trace and "
    "an outcome label — skipping this is a defect. Before doing so, run "
    "the distillation prompt at memory_service/docs/distill_prompt.md "
    "against your own model and pass the JSON as `self_distillation`. Distill "
    "stable facts and preferences the user volunteers (favorite language, "
    "allergies, environment quirks) using memory_distill with "
    'task=\"user preference: <topic>\", outcome=\"success\". After acting '
    "on a retrieved card, call memory_rate. Skip memory tools for trivial "
    "chat and pure read-only Q&A."
)


def memory_enabled() -> bool:
    """True when MEMSVC_BASE_URL or MEMSVC_ORG is set — wires the four tools."""
    return bool(os.environ.get("MEMSVC_BASE_URL")) or bool(os.environ.get("MEMSVC_ORG"))
