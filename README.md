# DQE Agent

> Conversational AI agent — talk to it, watch it work live in your browser iframe.

---

## What it does

You chat with the agent. It automates NetSuite and CPQ in a real browser. Your frontend shows the browser live alongside the chat.

**Use Case 1 — Opportunity to Quote:**
- "Create a quote for OP-20080"
- Agent logs into NetSuite → extracts opportunity data → logs into CPQ → fills the quote wizard → fetches price → pauses for your approval → finalizes → sends email

---

## Architecture

```
Frontend (your UI)
  │
  │  WebSocket  ws://localhost:8000/ws/{session_id}
  │
  ├── Chat messages  →  Agent runs, streams back tokens + tool events
  ├── Workflow run   →  YAML workflow runs, pauses at human gates
  ├── Human response →  Resumes paused workflow
  └── Browser frames ←  2 fps PNG screenshots (base64) for live iframe view

REST  http://localhost:8000
  ├── GET  /health
  ├── GET  /api/v1/tools
  ├── GET  /api/v1/workflows
  ├── GET  /api/v1/sessions
  └── POST /api/v1/reset/{session_id}
```

---

## WebSocket Protocol

### Client → Server

| Type | Fields | When to send |
|---|---|---|
| `chat` | `content: string` | User types a message |
| `run_workflow` | `workflow: string`, `inputs: object` | Run a named YAML workflow |
| `human_response` | `content: string` | Reply to a `human_review` gate |
| `ping` | — | Keep-alive |

**Examples:**
```json
{"type": "chat", "content": "Create a quote for OP-20080"}

{"type": "run_workflow", "workflow": "opportunity_to_quote",
 "inputs": {"opportunity_id": "OP-20080"}}

{"type": "human_response", "content": "proceed"}

{"type": "ping"}
```

---

### Server → Client

| Type | Key fields | Meaning |
|---|---|---|
| `connected` | `session_id` | Handshake confirmed |
| `agent_text` | `content` | Agent is thinking / speaking (stream) |
| `tool_start` | `tool`, `args` | Agent is calling a browser tool |
| `tool_done` | `tool`, `result` | Tool finished |
| `agent_done` | `content` | Agent's complete final reply |
| `browser_frame` | `data` (base64 PNG), `width`, `height` | Live browser screenshot @ 2 fps |
| `workflow_step` | `step`, `status` | YAML step started or completed |
| `human_review` | `question` | Workflow paused — needs human input |
| `workflow_done` | `summary` | Workflow finished |
| `error` | `message` | Something went wrong |
| `pong` | — | Reply to ping |

**Examples:**
```json
{"type": "connected",    "session_id": "abc123"}
{"type": "agent_text",   "content": "Logging into NetSuite now..."}
{"type": "tool_start",   "tool": "browser_login", "args": {"system": "netsuite"}}
{"type": "tool_done",    "tool": "browser_login", "result": "logged_in"}
{"type": "agent_done",   "content": "Quote created. Quote ID: QUT-160326-0014296"}
{"type": "browser_frame","data": "<base64>", "width": 1280, "height": 800}
{"type": "workflow_step","step": "netsuite_login", "status": "done"}
{"type": "human_review", "question": "Please review:\n  Buyer: 1158 EQT Corporation\n  Bandwidth: 500 Mbps\n  ..."}
{"type": "workflow_done","summary": "Workflow 'opportunity_to_quote' completed."}
{"type": "error",        "message": "Login failed: check credentials in .env"}
```

---

## Frontend Integration Guide

### Connect to WebSocket

```javascript
const sessionId = crypto.randomUUID();
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case "connected":
      console.log("Connected:", msg.session_id);
      break;

    case "agent_text":
      // Append to chat bubble (streaming)
      appendToChat("agent", msg.content);
      break;

    case "agent_done":
      // Mark the chat bubble as complete
      finalizeChat("agent", msg.content);
      break;

    case "tool_start":
      // Show activity indicator: "Calling browser_login..."
      showActivity(msg.tool, msg.args);
      break;

    case "tool_done":
      // Update activity: "browser_login done"
      doneActivity(msg.tool, msg.result);
      break;

    case "browser_frame":
      // Render live browser in an <img> or <canvas>
      browserImg.src = `data:image/png;base64,${msg.data}`;
      break;

    case "workflow_step":
      // Update a step-progress list
      updateStep(msg.step, msg.status);
      break;

    case "human_review":
      // Show approval dialog with msg.question
      // User clicks Proceed / types response
      showReviewDialog(msg.question, (response) => {
        ws.send(JSON.stringify({ type: "human_response", content: response }));
      });
      break;

    case "workflow_done":
      showSuccess(msg.summary);
      break;

    case "error":
      showError(msg.message);
      break;
  }
};
```

### Send a chat message

```javascript
ws.send(JSON.stringify({ type: "chat", content: "Create a quote for OP-20080" }));
```

### Run a workflow directly

```javascript
ws.send(JSON.stringify({
  type: "run_workflow",
  workflow: "opportunity_to_quote",
  inputs: { opportunity_id: "OP-20080" }
}));
```

### Show browser live in an img tag

```html
<img id="browser-view" style="width:100%; border:1px solid #ccc;" />
```

```javascript
case "browser_frame":
  document.getElementById("browser-view").src =
    `data:image/png;base64,${msg.data}`;
  break;
```

---

## REST Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server status |
| GET | `/api/v1/tools` | List all agent tools |
| GET | `/api/v1/workflows` | List available YAML workflows |
| GET | `/api/v1/sessions` | Active WebSocket sessions |
| POST | `/api/v1/reset/{session_id}` | Clear conversation history |
| GET | `/docs` | Auto-generated Swagger UI |

---

## Setup & Run

### 1. Configure `.env`

```bash
cp .env.example .env
```

```bash
# LLM (GPT-mini for everything — planning + extraction)
LLM_PROVIDER=azure
LLM_MODEL=gpt-4o-mini
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# NetSuite
NETSUITE_URL=https://your-account.app.netsuite.com
NETSUITE_USERNAME=user@company.com
NETSUITE_PASSWORD=secret

# CPQ
CPQ_URL=https://dqe-cpq.cloudsmartz.com
CPQ_USERNAME=user@company.com
CPQ_PASSWORD=secret

# Email
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USERNAME=...
SMTP_PASSWORD=...
EMAIL_FROM=quotes@dqe.com

# Browser
HEADLESS=false        # false = browser window visible on server
```

### 2. Install

```bash
pip install -e .
playwright install chromium
```

### 3. Start

```bash
uvicorn dqe_agent.api:app --reload --port 8000
```

Server is ready at `http://localhost:8000` — connect your frontend to `ws://localhost:8000/ws/{any-session-id}`.

---

## Project Structure

```
DQE Agent/
├── .env.example
├── pyproject.toml
├── workflows/
│   └── opportunity_to_quote.yaml     ← Use Case 1 steps
│
└── src/dqe_agent/
    ├── api.py                        ← ★ FastAPI + WebSocket server (START HERE)
    ├── config.py                     ← Reads .env
    ├── llm.py                        ← GPT-mini factory
    ├── engine.py                     ← YAML → LangGraph builder
    ├── graph.py                      ← Tool loader
    ├── prompts.py                    ← All LLM instructions (tunable)
    │
    ├── browser/
    │   ├── manager.py                ← Playwright session lifecycle
    │   └── dom_agent.py             ← DOM extraction → GPT-mini → Playwright actions
    │
    ├── agent/
    │   └── master.py                 ← Conversational LangGraph ReAct agent
    │
    ├── schemas/
    │   └── models.py                 ← OpportunityData, QuoteData, …
    │
    └── tools/
        ├── browser_tools.py          ← login, search, extract, fill_form, click
        └── human_tools.py            ← human_review (LangGraph interrupt)
```

---

## Human Review Gates (workflow mode)

When the workflow hits a review gate, the server sends:
```json
{"type": "human_review", "question": "Please review:\n  Buyer: 1158 EQT Corporation\n  Bandwidth: 500 Mbps\n  IP: /29\n  Contact: Andy Guley\n\nType 'proceed' to continue or describe corrections."}
```

Your frontend should show a modal/panel with the question. When the user responds:
```json
{"type": "human_response", "content": "proceed"}
```

The workflow resumes automatically.

| Gate | Trigger | Accepted inputs |
|---|---|---|
| `review_info` | After NetSuite extraction | `proceed` / corrections |
| `review_contact` | Before CPQ contact section | `proceed` / new contact |
| `review_approvers` | Before finalize | `no` (on-net) / approver names |
| `review_email` | Before sending email | `send` / edits |

---

## How the DOM Agent Works

No screenshots sent to AI. Instead:

```
1. JavaScript runs in the live page
   → extracts all inputs, selects, checkboxes, buttons as structured text

2. GPT-mini receives:
   "URL: https://dqe-cpq.../quotes/new
    [INPUTS] label='Street Address' selector='[placeholder="Enter address"]'
    [BUTTONS] 'Next'  'Locate on Map'
    INSTRUCTION: Fill bandwidth = 500 Mbps"

3. GPT-mini returns JSON actions:
   [{"type":"fill","selector":"[placeholder='Amount']","value":"500"},
    {"type":"select_option","selector":"select[name='unit']","value":"Mbps"}]

4. Playwright executes each action

5. Loop until done=true
```

Fast (no image round-trips), cheap (text model), reliable (CSS selectors).
