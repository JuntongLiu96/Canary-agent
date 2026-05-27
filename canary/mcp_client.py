"""MCP hub: connects to all configured servers, exposes tools + prompts."""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import load_mcp_config

log = logging.getLogger("canary.mcp")


class MCPHub:
    def __init__(self) -> None:
        # One stack per server, so a failed connect rolls back only its own
        # resources and doesn't poison the others' lifecycle.
        self.stacks: dict[str, AsyncExitStack] = {}
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[dict[str, Any]] = []
        self.prompts: list[dict[str, Any]] = []

    async def start(self) -> None:
        cfg = load_mcp_config().get("mcpServers", {})
        for server_name, spec in cfg.items():
            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                session = await self._connect(stack, server_name, spec)
                self.sessions[server_name] = session
                self.stacks[server_name] = stack
            except Exception as e:
                log.warning("MCP server %s failed to start: %s", server_name, e)
                try:
                    await stack.__aexit__(type(e), e, e.__traceback__)
                except Exception as e2:
                    log.warning("MCP server %s cleanup error: %s", server_name, e2)
        await self.refresh()

    async def _connect(self, stack: AsyncExitStack, name: str, spec: dict[str, Any]) -> ClientSession:
        if "url" in spec:
            url: str = spec["url"]
            headers = spec.get("headers") or None
            transport = spec.get("transport") or ("sse" if url.rstrip("/").endswith("/sse") else "http")
            if transport == "sse":
                read, write = await stack.enter_async_context(sse_client(url, headers=headers))
            else:
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(url, headers=headers)
                )
        else:
            params = StdioServerParameters(
                command=spec["command"],
                args=spec.get("args", []),
                env=spec.get("env"),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        log.info("MCP connected: %s", name)
        return session

    async def refresh(self) -> None:
        self.tools = []
        self.prompts = []
        for server_name, session in self.sessions.items():
            try:
                tools_res = await session.list_tools()
                for t in tools_res.tools:
                    self.tools.append({
                        "server": server_name,
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                    })
            except Exception as e:
                log.warning("list_tools(%s) failed: %s", server_name, e)
            try:
                prompts_res = await session.list_prompts()
                for p in prompts_res.prompts:
                    self.prompts.append({
                        "server": server_name,
                        "name": p.name,
                        "description": p.description or "",
                        "arguments": [
                            {"name": a.name, "description": a.description or "", "required": bool(a.required)}
                            for a in (p.arguments or [])
                        ],
                    })
            except Exception:
                pass  # server may not support prompts

    async def call_tool(self, server: str, name: str, args: dict[str, Any]) -> str:
        session = self.sessions.get(server)
        if session is None:
            return f"MCP server '{server}' not connected"
        result = await session.call_tool(name, args)
        parts: list[str] = []
        for c in result.content:
            if getattr(c, "type", None) == "text":
                parts.append(c.text)
            else:
                parts.append(repr(c))
        text = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"<tool_error>\n{text}\n</tool_error>"
        return text

    async def get_prompt(self, server: str, name: str, args: dict[str, str]) -> list[dict[str, Any]]:
        session = self.sessions.get(server)
        if session is None:
            return []
        result = await session.get_prompt(name, args)
        out: list[dict[str, Any]] = []
        for m in result.messages:
            c = m.content
            if getattr(c, "type", None) == "text":
                out.append({"role": m.role, "content": {"type": "text", "text": c.text}})
            else:
                out.append({"role": m.role, "content": repr(c)})
        return out

    async def stop(self) -> None:
        for name, stack in list(self.stacks.items()):
            try:
                await stack.aclose()
            except Exception as e:
                log.warning("MCP server %s shutdown error: %s", name, e)
        self.stacks.clear()
        self.sessions.clear()
