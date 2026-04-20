# DQE Agent

Conversational AI agent that talks to Jira, Gmail, Google Calendar, and automates browsers (NetSuite, CPQ). Chat with it naturally — it plans steps, executes them, and confirms what it did.

---

## What it does

**Jira** (via MCP — no browser, fast)
- Query your issues, sprint status, team workload, blockers
- Create tickets, subtasks, bugs, stories
- Transition status, change priority, assign issues
- Log time, add comments, link issues
- Move issues to sprints, create/start/close sprints
- Daily standup digest

**Google Workspace** (via MCP — no browser, fast)
- Read and send Gmail
- Create, update, and query Google Calendar events
- Check free/busy availability

**Browser automation** (Playwright — visual, slower)
- NetSuite → CPQ quote creation workflow
- Any site that needs visual interaction

---

## Architecture

```
User message
    │
    ▼
Planner  (GPT-4o / large model — runs ONCE)
    │  produces a step list
    ▼
Executor (GPT-4o-mini / fast model — runs per step)
    │  calls MCP tools or browser tools
    ▼
Verifier (checks success, retries or replans)
    │
    ▼
Response streamed to frontend via WebSocket
```

**Fast path:** common Jira queries (show my issues, show blockers, etc.) bypass the LLM entirely via regex matching — response in under 2 seconds.

**Session isolation:** each WebSocket connection has its own LangGraph thread with SQLite checkpointing. Conversations are independent.

---

## WebSocket Protocol

### Connect

```
ws://localhost:8000/ws/{session_id}
```

Use any UUID as `session_id`. One session per browser tab / user.

---

### Client → Server

| Type | Fields | When |
|---|---|---|
| `chat` | `content: string` | User sends a message |
| `run_task` | `task: string` | Run a specific task string directly |
| `run_workflow` | `workflow: string`, `inputs: object` | Run a named YAML workflow |
| `human_response` | `content: string` | Reply to a `human_review` gate |
| `selection_response` | `value: string \| string[]` | Reply to a `selection_request` |
| `ping` | — | Keep-alive |

```json
{"type": "chat",              "content": "show my open issues"}
{"type": "chat",              "content": "move FLAG-42 to done"}
{"type": "chat",              "content": "log 2 hours on FLAG-42"}
{"type": "run_workflow",      "workflow": "opportunity_to_quote", "inputs": {"opportunity_id": "OP-20080"}}
{"type": "human_response",    "content": "proceed"}
{"type": "selection_response","value": "SP-23"}
{"type": "ping"}
```

---

### Server → Client

| Type | Key fields | Meaning |
|---|---|---|
| `connected` | `session_id` | Handshake confirmed |
| `agent_text` | `content` | Streaming token (append to chat bubble) |
| `agent_done` | `content` | Final complete reply |
| `tool_start` | `tool`, `args` | A tool call is starting |
| `tool_done` | `tool`, `result` | Tool call finished |
| `plan_created` | `steps: []` | Planner produced a step list |
| `step_status` | `step`, `status` | A step started / completed / failed |
| `browser_frame` | `data` (base64 PNG), `width`, `height` | Live browser screenshot at ~2 fps |
| `selection_request` | `question`, `options: [{value,label}]`, `multi_select` | Agent needs user to pick from a list |
| `human_review` | `question` | Workflow paused — needs approval |
| `workflow_done` | `summary` | Workflow completed |
| `error` | `message` | Something went wrong |
| `pong` | — | Reply to ping |

```json
{"type": "connected",         "session_id": "abc123"}
{"type": "agent_text",        "content": "Looking up your issues..."}
{"type": "plan_created",      "steps": [{"id": "jira_q", "tool": "jira_search", ...}]}
{"type": "step_status",       "step": "jira_q", "status": "running"}
{"type": "step_status",       "step": "jira_q", "status": "done"}
{"type": "agent_done",        "content": "**5 issues found:**\n\n**FLAG-42** — Login crash..."}
{"type": "selection_request", "question": "Which sprint?",
                              "options": [{"value": "23", "label": "Sprint 23 (Active)"}],
                              "multi_select": false}
{"type": "human_review",      "question": "Buyer: EQT Corp\nBandwidth: 500 Mbps\n\nType 'proceed' to continue."}
{"type": "browser_frame",     "data": "<base64>", "width": 1280, "height": 800}
{"type": "error",             "message": "Jira API token missing — set JIRA_API_TOKEN in .env"}
```

---

## REST Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server status + uptime |
| GET | `/api/v1/tools` | List all loaded tools (MCP + browser) |
| GET | `/api/v1/workflows` | List available YAML workflows |
| GET | `/api/v1/sessions` | Active WebSocket sessions |
| POST | `/api/v1/reset/{session_id}` | Clear conversation history for a session |
| GET | `/docs` | Swagger UI |

---

## Setup

### 1. Install Python dependencies

```bash
pip install -e .
```

To also enable MCP tools (Jira, Gmail, Calendar):

```bash
pip install -e ".[dev]"
```

### 2. Install Playwright browser

```bash
playwright install chromium
```

Skip this if you set `DISABLE_BROWSER_TOOLS=true` (Jira/Gmail-only mode).

### 3. Install MCP servers

```bash
# Jira
pip install mcp-atlassian

# Google Workspace (Gmail + Calendar)
pip install workspace-mcp
```

### 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your values. The minimum required for Jira features:

```bash
JIRA_URL=https://your-company.atlassian.net
JIRA_USERNAME=you@company.com
JIRA_API_TOKEN=your_api_token_here      # https://id.atlassian.com/manage-profile/security/api-tokens
```

For Gmail + Calendar, run the one-time OAuth setup:

```bash
python setup_google_auth.py
```

### 5. Start the server

```bash
# Windows
python run.py

# Cross-platform
uvicorn dqe_agent.api:app --reload --port 8000
```

Server is ready at `http://localhost:8000`. Connect your frontend to `ws://localhost:8000/ws/{any-uuid}`.

---

## Jira-only mode (no browser)

Set `DISABLE_BROWSER_TOOLS=true` in `.env`. The server starts without Playwright — faster startup, lower memory, works headless. All Jira and Gmail features still work.

---

## Project Structure

```
DQE Agent/
├── .env.example                      ← copy to .env, fill in credentials
├── mcp_config.yaml                   ← MCP server declarations
├── pyproject.toml
├── run.py                            ← Windows start script
│
├── workflows/
│   └── opportunity_to_quote.yaml    ← NetSuite → CPQ workflow definition
│
├── tests/
│   ├── test_graph.py                 ← schema / model smoke tests
│   ├── test_jira_unit.py             ← 77 unit tests (no API calls needed)
│   └── test_browser_session_input.py
│
└── src/dqe_agent/
    ├── api.py                        ← FastAPI + WebSocket server
    ├── config.py                     ← Settings loaded from .env
    ├── state.py                      ← AgentState TypedDict (LangGraph)
    ├── llm.py                        ← LLM factory (Azure / OpenAI / Anthropic)
    ├── guardrails.py                 ← Step/cost/timeout limits
    ├── observability.py              ← Tool call tracing
    │
    ├── agent/
    │   ├── master.py                 ← MasterAgent: SQLite checkpointer + graph compile
    │   ├── loop.py                   ← PEV LangGraph graph builder
    │   ├── planner.py                ← Planner node + fast-path regex bypass
    │   ├── executor.py               ← Executor node + tool normalisation
    │   └── verifier.py               ← Verifier node + retry/replan logic
    │
    ├── browser/
    │   ├── manager.py                ← Playwright lifecycle, per-session contexts
    │   ├── dom_agent.py              ← DOM extraction → LLM → Playwright actions
    │   └── webrtc.py                 ← WebRTC screen streaming
    │
    ├── tools/
    │   ├── __init__.py               ← Tool registry (auto-discovery)
    │   ├── browser_tools.py          ← login, navigate, act, extract, snapshot
    │   ├── human_tools.py            ← human_review (LangGraph interrupt)
    │   ├── selection_tool.py         ← request_selection (dropdown UI)
    │   ├── respond_tool.py           ← direct_response (no-tool replies)
    │   └── mcp_loader.py             ← Loads MCP tools at startup
    │
    ├── flows/
    │   ├── _default.py               ← Default conversational flow config
    │   └── opportunity_to_quote.py   ← Quote creation flow config
    │
    └── schemas/
        └── models.py                 ← OpportunityData, QuoteData, EmailPayload, …
```

---

## Running Tests

```bash
# All tests
pytest

# Just the Jira unit tests (no API/browser needed)
pytest tests/test_jira_unit.py -v
```

The Jira unit tests (`test_jira_unit.py`) cover 77 cases — fast-path query matching, result formatting, empty-state messages, and parameter normalisation. They run in under 1 second with no network access.

---

## Human Review Gates (workflow mode)

When the quote workflow reaches a review step, the server sends:

```json
{"type": "human_review", "question": "Please review:\n  Buyer: EQT Corporation\n  Bandwidth: 500 Mbps\n\nType 'proceed' to continue or describe corrections."}
```

The frontend shows a prompt. When the user responds:

```json
{"type": "human_response", "content": "proceed"}
```

The workflow resumes. To correct a field, the user types the correction and the agent replans.

| Gate | When | Accepted |
|---|---|---|
| `review_info` | After NetSuite extraction | `proceed` / corrections |
| `review_contact` | Before CPQ contact section | `proceed` / new contact details |
| `review_approvers` | Before CPQ finalize | `no` (on-net) / approver names |
| `review_email` | Before sending email | `send` / edits |

---

## Selection Requests (Jira workflows)

When the agent needs the user to pick from a list (e.g. which sprint, which board):

```json
{
  "type": "selection_request",
  "question": "Which sprint should I move FLAG-42 to?",
  "options": [
    {"value": "42", "label": "Sprint 24 – Active"},
    {"value": "41", "label": "Sprint 23 – Closed"}
  ],
  "multi_select": false
}
```

The frontend renders a dropdown or button group. Reply:

```json
{"type": "selection_response", "value": "42"}
```

---

## MCP Configuration

MCP servers are declared in `mcp_config.yaml`. Credentials come from `.env` via `${VAR}` substitution.

| Server | Package | What it provides |
|---|---|---|
| `jira` | `mcp-atlassian` | 170+ Jira tools (search, create, transition, sprint, …) |
| `google-workspace` | `workspace-mcp` | Gmail + Google Calendar tools |
| `brave-search` | `mcp-server-brave-search` | Web search (disabled by default) |

To enable Brave Search: set `enabled: true` in `mcp_config.yaml` and add `BRAVE_API_KEY` to `.env`.

---

## Linting

```bash
ruff check src/
ruff format src/
```
