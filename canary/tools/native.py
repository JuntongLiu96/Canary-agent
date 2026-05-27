from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from . import Tool


async def _read_file(args: dict[str, Any]) -> str:
    path = Path(args["path"])
    offset = int(args.get("offset", 0))
    limit = args.get("limit")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    end = len(lines) if limit is None else min(len(lines), offset + int(limit))
    sliced = lines[offset:end]
    return "\n".join(f"{i+1+offset:6d}\t{line}" for i, line in enumerate(sliced))


async def _write_file(args: dict[str, Any]) -> str:
    path = Path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return f"Wrote {len(args['content'])} bytes to {path}"


async def _bash(args: dict[str, Any]) -> str:
    command: str = args["command"]
    timeout_ms = int(args.get("timeout_ms", 120_000))
    if sys.platform == "win32":
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        argv = ["bash", "-lc", command]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
    except asyncio.TimeoutError:
        proc.kill()
        return f"<timeout after {timeout_ms}ms>"
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    parts = [f"<exit_code>{proc.returncode}</exit_code>"]
    if out:
        parts.append(f"<stdout>\n{out}</stdout>")
    if err:
        parts.append(f"<stderr>\n{err}</stderr>")
    return "\n".join(parts)


NATIVE_TOOLS: list[Tool] = [
    Tool(
        name="read_file",
        description="Read a UTF-8 text file and return its lines (1-indexed).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."},
                "offset": {"type": "integer", "description": "0-indexed line offset.", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to return."},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ),
    Tool(
        name="write_file",
        description="Overwrite a file with the given UTF-8 content (creates parent dirs).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
    ),
    Tool(
        name="bash",
        description=(
            "Run a shell command. On Windows uses PowerShell, elsewhere bash. "
            "Returns exit code, stdout, stderr."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer", "default": 120000},
            },
            "required": ["command"],
        },
        handler=_bash,
    ),
]
