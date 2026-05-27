# Canary Agent

A small, hackable test-platform agent with full MCP, Skills, and subagent
support. Use it to try out new MCP servers, new skills, or new subagent
patterns without the weight of a production agent harness.

- **LLM provider:** local [`copilot-api`](../copilot-api) (`http://localhost:4141`).
  Claude models go to `/v1/messages`; everything else (`gpt-*`, `gemini-*`, …)
  goes to `/v1/chat/completions`.
- **Loop:** plain `while True` in `canary/agent.py` — same shape as Claude Code.
- **Native tools:** `read_file`, `write_file`, `bash` (PowerShell on Windows).
- **MCP:** tools + prompts. Prompts surface as `/server:prompt` slash commands
  in the UI. Configure servers in `config/mcp.json` (Claude Code shape).
- **Skills:** drop a `skills/<name>/SKILL.md` with YAML frontmatter — the model
  invokes them via a single `Skill` tool, exactly like Claude Code.
- **Subagents:** drop a `subagents/<name>.md` — the model invokes them via the
  `Agent` tool. Tool allowlists and per-subagent model overrides supported.

## Run

```sh
# 1. Start copilot-api on :4141 in another terminal
cd ../copilot-api && bun run start

# 2. Install + run canary
cd Canary-agent
uv sync                       # or: pip install -e .
uv run uvicorn canary.main:app --port 8000
# open http://localhost:8000
```

## Layout

```
canary/        agent loop, providers, tools, mcp client, skills loader
config/        mcp.json (server defs), settings.json (default model)
skills/        one folder per skill, each has SKILL.md
subagents/     one .md per subagent
web/           single-page UI (no build step)
```

## API

| route                    | description                                       |
| ------------------------ | ------------------------------------------------- |
| `POST /api/chat`         | `{session_id, message?, blocks?, model?}`         |
| `GET  /api/stream/{sid}` | SSE of normalized agent events                    |
| `GET  /api/models`       | proxies copilot-api `/v1/models`                  |
| `GET  /api/prompts`      | merged MCP prompts catalog                        |
| `POST /api/prompts/get`  | resolve a prompt → user-turn blocks               |
| `POST /api/mcp/refresh`  | re-list MCP tools + prompts                       |
| `GET  /api/state/{sid}`  | full message history (debug)                      |
| `POST /api/reset/{sid}`  | clear history                                     |
