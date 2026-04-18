# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DQE Agent is a conversational AI browser automation platform. It uses **LangGraph** for orchestration, **Playwright** for browser control, and **Native Computer Use agents** (Anthropic Claude or OpenAI) for visual page interaction. The primary use case automates NetSuite → CPQ quote creation → email approval workflows.

## Setup

```bash
pip install -e .
playwright install chromium
cp .env.example .env  # Configure credentials before starting
```

Start server (Windows):
```bash
python run.py
```

Start server (cross-platform):
```bash
uvicorn dqe_agent.api:app --reload --port 8000
```

## Running Tests

```bash
# All tests (asyncio_mode=auto is configured in pyproject.toml)
pytest

# Single test file
pytest tests/test_graph.py
```

## Linting

```bash
ruff check src/
ruff format src/
```

## Architecture

### Request Flow
1. Frontend connects via WebSocket at `/ws/{session_id}`
2. `api.py` routes messages to `MasterAgent` (per-session)
3. `MasterAgent` runs a LangGraph ReAct loop: agent node → tool node → agent node
4. Browser tools use `BrowserManager` (global singleton) to get per-session Playwright contexts
5. Visual interactions go through `dom_agent.py` (CUA): screenshot → LLM → JSON actions → Playwright execution
6. Human review gates use LangGraph `interrupt()`, suspending execution until the frontend sends a response

### Two Operating Modes
- **Conversational**: Default flow (`flows/_default.py`), free-form tool use driven by user messages
- **Workflow**: Sends `{"type": "workflow", "workflow": "opportunity_to_quote"}` to execute a deterministic YAML-defined sequence (`workflows/opportunity_to_quote.yaml`)

### Key Components

| File | Role |
|------|------|
| `src/dqe_agent/api.py` | FastAPI + WebSocket server, session management, lifespan startup |
| `src/dqe_agent/agent/master.py` | LangGraph ReAct agent, lazy per-flow graph compilation, checkpointer memory |
| `src/dqe_agent/agent/nodes.py` | LangGraph nodes: agent inference, tool execution, routing logic |
| `src/dqe_agent/state.py` | `AgentState` TypedDict — messages, extracted data fields, human review state |
| `src/dqe_agent/engine.py` | YAML workflow parser → LangGraph graph with ordered steps |
| `src/dqe_agent/browser/manager.py` | Playwright lifecycle; isolated browser contexts per session; CDP screencast |
| `src/dqe_agent/browser/dom_agent.py` | CUA wrapper: DOM extraction + screenshot → LLM → fill/click/select actions |
| `src/dqe_agent/tools/` | Auto-discovered tool modules (browser, human review, email, MCP) |
| `src/dqe_agent/flows/` | Flow registry; each flow defines system prompt, tool allowlist, reminders |
| `src/dqe_agent/schemas/models.py` | Pydantic models: OpportunityData, QuoteData, ContactInfo, EmailPayload, etc. |
| `mcp_config.yaml` | MCP server declarations (Brave Search active; Slack/GitHub/etc. commented) |

### Two AgentState Classes
There are two separate state types:
- `src/dqe_agent/agent/state.py` — simple `AgentState` used by `MasterAgent` (messages, browser_ready, current_task, error)
- `src/dqe_agent/state.py` — full workflow `AgentState` with domain fields (opportunity, quote, email, human_review, flow_data) used by `engine.py` for YAML-driven workflows

### Tool & Flow Auto-Discovery
Tools and flows are registered dynamically. Adding a new `@register_tool(name, description)`-decorated function in `tools/` or a new `FlowConfig` subclass in `flows/` makes them available automatically without wiring changes. MCP tools require `pip install -e ".[dev]"` (adds `langchain-mcp-adapters`).

### LLM Configuration
The `config.py` `Settings` class controls which provider is used:
- **General LLM** (`LLM_PROVIDER`): `azure` | `openai` | `anthropic` — used for reasoning/planning
- **Computer Use Agent** (`CUA_PROVIDER`): `anthropic` | `openai` — used for visual browser interaction

All credentials come from `.env` (see `.env.example` for all keys).

### Human Review Gates
Workflow steps marked `human_review: true` in YAML call `interrupt()`, which suspends the LangGraph execution. The session stores a queue of pending reviews. The frontend must send `{"type": "human_response", "approved": true/false, "message": "..."}` to resume.

### WebSocket Message Protocol
- Inbound: `{"type": "chat"|"workflow"|"human_response", ...}`
- Outbound: `{"type": "message"|"browser_frame"|"human_review_request"|"workflow_complete"|"error", ...}`
- Browser frames stream at ~2 fps as base64-encoded PNG via `browser_frame` messages
