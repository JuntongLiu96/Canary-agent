from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SKILLS_DIR = ROOT / "skills"
SUBAGENTS_DIR = ROOT / "subagents"
WEB_DIR = ROOT / "web"

# Load .env from repo root before reading any env vars below.
load_dotenv(ROOT / ".env")

COPILOT_BASE_URL = os.environ.get("COPILOT_BASE_URL", "http://localhost:4141")
DEFAULT_MODEL = os.environ.get("CANARY_MODEL", "claude-sonnet-4.5")


def load_settings() -> dict[str, Any]:
    path = CONFIG_DIR / "settings.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_mcp_config() -> dict[str, Any]:
    path = CONFIG_DIR / "mcp.json"
    if not path.exists():
        return {"mcpServers": {}}
    return json.loads(path.read_text(encoding="utf-8"))
