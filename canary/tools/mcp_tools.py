from __future__ import annotations

from typing import Any

from . import Tool, ToolRegistry


def register_mcp_tools(registry: ToolRegistry, hub) -> None:
    for entry in hub.tools:
        server = entry["server"]
        name = entry["name"]
        tool_name = f"mcp__{server}__{name}"

        def make_handler(s: str, n: str):
            async def handler(args: dict[str, Any]) -> str:
                return await hub.call_tool(s, n, args)
            return handler

        registry.register(Tool(
            name=tool_name,
            description=entry["description"],
            input_schema=entry["input_schema"],
            handler=make_handler(server, name),
        ))
