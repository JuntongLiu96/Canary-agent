from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self, allow: list[str] | None = None) -> list[dict[str, Any]]:
        names = self._filter_names(allow)
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in (self.tools[n] for n in names)
        ]

    def _filter_names(self, allow: list[str] | None) -> list[str]:
        if allow is None:
            return list(self.tools.keys())
        out: list[str] = []
        for pat in allow:
            for name in self.tools:
                if fnmatch.fnmatchcase(name, pat) and name not in out:
                    out.append(name)
        return out
