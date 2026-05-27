"""Skills loader: scan skills/*/SKILL.md, expose a single `Skill` tool."""
from __future__ import annotations

from typing import Any

import yaml

from .config import SKILLS_DIR
from .tools import Tool, ToolRegistry


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm) or {}, body.lstrip("\n")


def load_skills() -> list[dict[str, str]]:
    """Returns [{name, description, body, path}, ...]."""
    out: list[dict[str, str]] = []
    if not SKILLS_DIR.exists():
        return out
    for skill_md in SKILLS_DIR.glob("*/SKILL.md"):
        try:
            meta, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            out.append({
                "name": meta.get("name", skill_md.parent.name),
                "description": meta.get("description", ""),
                "body": body,
                "path": str(skill_md),
            })
        except Exception:
            continue
    return out


def register_skill_tool(registry: ToolRegistry, skills: list[dict[str, str]]) -> None:
    if not skills:
        return
    by_name = {s["name"]: s for s in skills}
    desc = "Load a skill's full instructions by name. Available: " + ", ".join(
        f"{s['name']} ({s['description']})" for s in skills
    )

    async def handler(args: dict[str, Any]) -> str:
        name = args.get("name", "")
        s = by_name.get(name)
        if s is None:
            return f"Unknown skill '{name}'. Available: {', '.join(by_name)}"
        return s["body"]

    registry.register(Tool(
        name="Skill",
        description=desc,
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name to load."},
            },
            "required": ["name"],
        },
        handler=handler,
    ))
