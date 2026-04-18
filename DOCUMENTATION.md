# 📚 DQE Agent — Complete Documentation

> AI-powered browser automation built with **LangGraph** + **Native Computer Use** (Anthropic Claude / OpenAI GPT-5.4) + **Playwright**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Two Operating Modes](#two-operating-modes)
6. [API Reference](#api-reference)
7. [Tool System](#tool-system)
8. [MCP Integration](#mcp-integration)
9. [Hybrid Smart Click System](#hybrid-smart-click-system)
10. [Configuration](#configuration)
11. [Workflows](#workflows)
12. [Examples](#examples)

---

## 🎯 Overview

### What is DQE Agent?

DQE Agent is a browser automation platform that combines:
- **Natural language understanding** - Talk to it like a colleague
- **Visual AI** - Sees and interacts with webpages like a human
- **Modular architecture** - Easy to extend and customize
- **Production-ready** - Built for reliability and auditability

### Why Native Computer Use?

Most browser automation agents add a middle layer (Skyvern, Browser Use) that **compounds errors**:
- Middle layer makes its own LLM calls
- Double inference = information loss at each hop
- Hard to debug when things go wrong

**DQE Agent eliminates the middle layer:**
```
Screenshot → CUA Model → Playwright Actions
```
Zero information loss. Maximum accuracy.

### Key Features

| Feature | Benefit |
|---------|---------|
| **Two modes** | Conversational (flexible) + YAML workflows (deterministic) |
| **Master Agent (LangGraph ReAct)** | Agent decides which tools to call automatically |
| **YAML workflows** | Predictable, auditable execution with human approvals |
| **Modular tools** | Add capabilities without touching core code |
| **Native CUA** | LLM sees actual screen pixels → highest accuracy |
| **Provider-agnostic** | Switch between Claude and GPT-5.4 with one config change |
| **Hybrid Click System** | DOM finding + coordinate fallback for reliability |
| **MCP Integration** | Load tools from any MCP server |

---

## 🏗️ Architecture

### High-Level Flow (Conversational Mode)

```
┌─────────────────────────────────────────┐
│  You: "Create quote for OP-20080"       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Master Agent (LangGraph ReAct)         │
│  ┌──────────┐      ┌──────────┐        │
│  │  Agent   │  ←→  │  Tools   │        │
│  │  (Plan)  │      │ (Execute)│        │
│  └──────────┘      └──────────┘        │
│       ↓                   ↓             │
│  Decide tools       Run tools           │
│  Loop until done    (browser actions)   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  CUA Agent (screenshot → LLM → actions) │
│  Providers: Claude | GPT-5.4            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Playwright (browser control)           │
└─────────────────────────────────────────┘
```

### Tool Architecture

```
┌──────────────────────────────────────┐
│  Core Tools (Always Loaded)          │
│  - browser_login                     │
│  - browser_search                    │
│  - browser_navigate                  │
│  - browser_extract                   │
│  - browser_fill_form                 │
│  - browser_click                     │
│  - human_review                      │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│  Extended Tools (MCP Servers)        │
│  - brave_search (web search)         │
│  - slack_send (notifications)        │
│  - github_* (code management)        │
│  - postgres_* (database access)      │
│  - Custom tools                      │
└──────────────────────────────────────┘
```

**Core Philosophy:** DQE Agent provides browser automation as its core value. Everything else is modular via MCP.

### Project Structure

```
DQE Agent/
├── pyproject.toml                    # Dependencies
├── .env.example                      # Configuration template
├── mcp_config.yaml                   # MCP server configuration
├── DOCUMENTATION.md                  # This file
│
├── workflows/
│   └── opportunity_to_quote.yaml     # Example workflow
│
├── src/dqe_agent/
│   ├── api.py                        # ★ FastAPI server
│   ├── chat.py                       # ★ CLI chat interface
│   ├── main.py                       # Workflow runner
│   ├── config.py                     # Settings
│   ├── llm.py                        # LLM factory
│   ├── prompts.py                    # Prompt templates
│   │
│   ├── agent/                        # ★ Master Agent
│   │   ├── master.py                 # Main agent class
│   │   ├── state.py                  # Agent state
│   │   └── nodes.py                  # Agent & tool nodes
│   │
│   ├── browser/                      # Browser automation
│   │   ├── manager.py                # Playwright wrapper
│   │   ├── cua.py                    # ★ Computer Use Agent
│   │   └── smart_finder.py           # ★ Hybrid click system
│   │
│   ├── tools/                        # Tool registry
│   │   ├── browser_tools.py          # Core browser tools
│   │   ├── human_tools.py            # Human-in-the-loop
│   │   ├── mcp_loader.py             # MCP integration
│   │   ├── mcp_config_loader.py      # YAML config parser
│   │   └── mcp_startup.py            # MCP startup helper
│   │
│   └── schemas/
│       └── models.py                 # Pydantic models
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Node.js (for MCP servers, optional)
- API keys (see Configuration)

### Steps

```bash
# 1. Navigate to project
cd "DQE Agent"

# 2. Create environment file
cp .env.example .env

# 3. Edit .env with your credentials
# Required:
#   - AZURE_OPENAI_API_KEY (or OPENAI_API_KEY)
#   - ANTHROPIC_API_KEY (or use OpenAI for CUA)
#   - NetSuite credentials
#   - CPQ credentials
# Optional:
#   - BRAVE_API_KEY (for web search via MCP)
#   - GITHUB_TOKEN (for GitHub integration)

# 4. Install Python dependencies
pip install -e .

# 5. Install Playwright browser (one-time)
playwright install chromium

# 6. Optional: Install MCP dependencies
pip install -e ".[mcp]"
```

---

## 🚀 Quick Start

### Option 1: API Server (Recommended)

```bash
# Start the server
uvicorn dqe_agent.api:app --reload --port 8000

# Open interactive docs
# Browser: http://localhost:8000/docs

# Send a message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a quote for opportunity OP-20080",
    "thread_id": "user123"
  }'
```

### Option 2: CLI Chat

```bash
python -m dqe_agent.chat
```

Then type naturally:
```
You: Create a quote for OP-20080
Agent: I'll help you create that quote. Let me start by logging in to NetSuite...
```

### Option 3: YAML Workflow (Deterministic)

```bash
python -m dqe_agent --workflow workflows/opportunity_to_quote.yaml --opportunity OP-20080
```

---

## 🔀 Two Operating Modes

### Mode 1: Conversational Agent (Flexible)

**Use when:** You want natural language interaction and flexible task execution.

**How it works:**
1. You send a natural language message
2. Master Agent (LangGraph ReAct) decides which tools to call
3. Loops until task is complete
4. Returns conversational response

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Log in to NetSuite and tell me what you see",
    "thread_id": "demo"
  }'
```

**Pros:**
- Natural interaction
- Adapts to changes
- Great for exploration
- No workflow definition needed

**Cons:**
- Less predictable
- Harder to audit
- May make unexpected tool calls

### Mode 2: YAML Workflow (Deterministic)

**Use when:** You need predictable, auditable, compliance-friendly execution.

**How it works:**
1. Define steps in YAML file
2. Engine builds LangGraph
3. Executes steps in exact order
4. Human approval checkpoints guaranteed

**Example workflow:**
```yaml
name: Opportunity to Quote
steps:
  - id: login
    tool: browser_login
    params:
      system: netsuite
  
  - id: search
    tool: browser_search
    params:
      system: netsuite
      search_query: "{{ opportunity_id }}"
  
  - id: review
    tool: human_review
    params:
      review_type: info_review
```

**Pros:**
- Predictable execution
- Auditable
- Guaranteed human approvals
- Version controlled

**Cons:**
- Requires YAML definition
- Less flexible
- Manual updates needed for new processes

---

## 🔌 API Reference

### Start API Server

```bash
uvicorn dqe_agent.api:app --reload --port 8000
```

Server: **http://localhost:8000**  
Docs: **http://localhost:8000/docs**

### Endpoints

#### POST /api/v1/chat

Send a message to the agent.

**Request:**
```json
{
  "message": "Create a quote for OP-20080",
  "thread_id": "user123"
}
```

**Response:**
```json
{
  "response": "I'll help you create that quote. Let me log in to NetSuite...",
  "thread_id": "user123"
}
```

**Python Example:**
```python
import httpx
import asyncio

async def chat(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/chat",
            json={"message": message, "thread_id": "demo"}
        )
        data = response.json()
        print(data["response"])

asyncio.run(chat("Log in to NetSuite"))
```

#### POST /api/v1/chat/stream

Stream agent execution (Server-Sent Events).

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for OP-20080", "thread_id": "demo"}'
```

#### POST /api/v1/reset

Clear conversation history for a thread.

```bash
curl -X POST http://localhost:8000/api/v1/reset \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo"}'
```

#### GET /api/v1/tools

List all available tools.

```bash
curl http://localhost:8000/api/v1/tools
```

#### GET /health

Health check.

```bash
curl http://localhost:8000/health
```

---

## 🛠️ Tool System

### Core Browser Tools

Always available, no configuration needed:

| Tool | Description | Parameters |
|------|-------------|------------|
| `browser_login` | Log in to web apps | `system`: "netsuite" or "cpq" |
| `browser_search` | Search for records | `system`, `search_query` |
| `browser_navigate` | Navigate to pages | `system`, `action` (natural language) |
| `browser_extract` | Extract data from page | `extract_type`, `system` |
| `browser_fill_form` | Fill forms | `system` |
| `browser_click` | Click elements | `target`, `system` |
| `browser_click_and_extract` | Click + extract | `click_target`, `extract_prompt` |

### Human-in-the-Loop

| Tool | Description |
|------|-------------|
| `human_review` | Pause workflow for human approval |

### Creating Custom Tools

```python
# src/dqe_agent/tools/custom_tools.py

from dqe_agent.tools import register_tool

@register_tool("check_inventory", "Check warehouse inventory for a product")
async def check_inventory(sku: str, **kwargs) -> dict:
    """Check inventory for a SKU."""
    # Your logic here
    return {"sku": sku, "quantity": 42, "location": "Warehouse A"}
```

**That's it!** The tool is now available to:
- Conversational agent (will call it automatically)
- YAML workflows (reference by name)

### Tool Parameters

All tools receive:
- Explicit parameters (defined in function signature)
- `**kwargs` which includes:
  - `__state__`: Current workflow state (in workflow mode)
  - Other metadata

---

## 🔗 MCP Integration

### What is MCP?

Model Context Protocol (MCP) is a standard for loading tools from external servers.

### Configuration

Edit `mcp_config.yaml`:

```yaml
servers:
  # Brave Search (enabled by default)
  - name: brave-search
    enabled: true
    transport: stdio
    config:
      command: npx
      args:
        - "-y"
        - "@anthropic/mcp-server-brave-search"
      env:
        BRAVE_API_KEY: ${BRAVE_API_KEY}  # Expands from .env
  
  # GitHub (disabled by default)
  - name: github
    enabled: false
    transport: stdio
    config:
      command: npx
      args:
        - "-y"
        - "@anthropic/mcp-server-github"
      env:
        GITHUB_TOKEN: ${GITHUB_TOKEN}
  
  # Slack (example)
  - name: slack
    enabled: false
    transport: stdio
    config:
      command: npx
      args:
        - "-y"
        - "@anthropic/mcp-server-slack"
      env:
        SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN}
  
  # Filesystem (example)
  - name: filesystem
    enabled: false
    transport: stdio
    config:
      command: npx
      args:
        - "-y"
        - "@anthropic/mcp-server-filesystem"
        - "/path/to/allowed/directory"
```

### Setup MCP Server

**1. Get API keys:**
- Brave Search: https://brave.com/search/api/ (2,000 free queries/month)
- GitHub: https://github.com/settings/tokens
- Slack: https://api.slack.com/apps

**2. Add to `.env`:**
```bash
BRAVE_API_KEY=BSA1234567890...
GITHUB_TOKEN=ghp_1234567890...
SLACK_BOT_TOKEN=xoxb-1234567890...
```

**3. Enable in `mcp_config.yaml`:**
```yaml
- name: brave-search
  enabled: true  # ← Change to true
```

**4. Restart API:**
```bash
uvicorn dqe_agent.api:app --reload --port 8000
```

You'll see in logs:
```
📦 Loading 1 MCP server(s)...
✅ Loaded MCP server config: brave-search (stdio)
Loaded MCP tool: brave_search — Search the web using Brave Search API
```

### Using MCP Tools

**Conversational mode:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Search the web for latest pricing on Cisco routers",
    "thread_id": "demo"
  }'
```

The agent automatically detects it needs `brave_search` and calls it.

**YAML workflow:**
```yaml
- id: web_research
  tool: brave_search
  params:
    query: "Cisco router prices 2026"
```

---

## 🎯 Hybrid Smart Click System

### The Problem

Computer Use APIs (Claude, GPT) return **coordinates only**:
```python
click(450, 200)
```

This is brittle:
- Page shifts after screenshot → click wrong element
- Layout changes → coordinates invalid
- Dynamic content → unpredictable

### The Solution

**Hybrid approach** combining DOM search + coordinate fallback:

```
LLM gives coordinates (450, 200)
        ↓
Smart Element Finder
        ↓
   ┌────┴────┐
   ↓         ↓
DOM Find   Coordinate
(HTML)    (Canvas/Flutter)
   ↓         ↓
Element    Mouse
Click      Click
```

### How It Works

**1. Try DOM element finding first:**
```python
# Get element at coordinates
element = await page.element_at_coordinates(450, 200)

# Build selector: button.login-btn
selector = build_selector(element)

# Click element (survives page shifts!)
await element.click()
```

**2. Fallback to coordinates:**
```python
# If DOM finding fails (Canvas/Flutter)
await page.mouse.click(450, 200)
```

### Implementation

**Files:**
- `src/dqe_agent/browser/smart_finder.py` - Smart element finder
- `src/dqe_agent/browser/cua.py` - Integrated with CUA loop

**Methods:**
- `smart_click(x, y, description, button)` - Intelligent clicking
- `smart_type(x, y, text)` - Intelligent form filling
- `_find_element_at_coordinates()` - DOM analysis
- `_build_selector()` - Selector generation

**Benefits:**
- More reliable on HTML/React apps
- Still works on Canvas/Flutter apps
- Automatic fallback
- Logs which method was used

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# ──────────────────────────────────────
# Azure OpenAI (General LLM - Required)
# ──────────────────────────────────────
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# ──────────────────────────────────────
# Computer Use Provider (Choose One)
# ──────────────────────────────────────
# Option 1: Anthropic Claude (Recommended)
CUA_PROVIDER=anthropic
CUA_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...

# Option 2: OpenAI GPT-5.4
# CUA_PROVIDER=openai
# CUA_MODEL=gpt-5.4
# OPENAI_API_KEY=sk-...

# ──────────────────────────────────────
# Browser Settings
# ──────────────────────────────────────
HEADLESS=false
VIEWPORT_WIDTH=1280
VIEWPORT_HEIGHT=720
SCREENSHOT_DIR=screenshots

# ──────────────────────────────────────
# Application Credentials
# ──────────────────────────────────────
# NetSuite
NETSUITE_URL=https://your-account.app.netsuite.com
NETSUITE_USERNAME=your_username
NETSUITE_PASSWORD=your_password

# CPQ
CPQ_URL=https://your-cpq.example.com
CPQ_USERNAME=your_username
CPQ_PASSWORD=your_password

# ──────────────────────────────────────
# MCP Server Keys (Optional)
# ──────────────────────────────────────
BRAVE_API_KEY=BSA...
GITHUB_TOKEN=ghp_...
SLACK_BOT_TOKEN=xoxb-...

# ──────────────────────────────────────
# Email (Optional - for workflow mode)
# ──────────────────────────────────────
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Switching CUA Providers

**Use Anthropic Claude (default):**
```bash
CUA_PROVIDER=anthropic
CUA_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
```

**Use OpenAI GPT-5.4:**
```bash
CUA_PROVIDER=openai
CUA_MODEL=gpt-5.4
OPENAI_API_KEY=sk-...
```

No code changes needed!

---

## 📝 Workflows

### Creating a Workflow

Create a YAML file in `workflows/`:

```yaml
# workflows/my_workflow.yaml
name: My Custom Workflow
description: Does something useful
inputs:
  customer_name: "Acme Corp"

steps:
  - id: login
    tool: browser_login
    description: Log in to system
    params:
      system: netsuite
  
  - id: search
    tool: browser_search
    description: Search for customer
    params:
      system: netsuite
      search_query: "{{ customer_name }}"
  
  - id: extract
    tool: browser_extract
    description: Extract customer data
    params:
      system: netsuite
      extract_type: "Extract customer name, email, phone, address"
  
  - id: review
    tool: human_review
    description: Verify extracted data
    params:
      review_type: info_review
    outputs: [human_review]
```

### Running a Workflow

```bash
python -m dqe_agent --workflow workflows/my_workflow.yaml
```

### Template Variables

Use `{{ variable }}` to inject values:

```yaml
params:
  search_query: "{{ customer_name }}"
```

### Conditional Steps

```yaml
- id: send_notification
  tool: slack_send
  condition: "state.get('urgent') == True"
  params:
    message: "Urgent: Customer data extracted"
```

### Human Review Steps

Guaranteed pause points:

```yaml
- id: human_approval
  tool: human_review
  params:
    review_type: info_review
  outputs: [human_review]
```

The workflow will pause and wait for your input before continuing.

---

## 💡 Examples

### Example 1: Simple Login

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Log in to NetSuite",
    "thread_id": "demo"
  }'
```

### Example 2: Data Extraction

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Extract all information from the current page",
    "thread_id": "demo"
  }'
```

### Example 3: Multi-Step Task

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a quote for opportunity OP-20080",
    "thread_id": "demo"
  }'
```

The agent will:
1. Log in to NetSuite
2. Search for OP-20080
3. Extract opportunity details
4. Log in to CPQ
5. Fill the quote form
6. Get pricing
7. Return the quote ID

### Example 4: Web Search (MCP)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Search the web for recent news about AI agents",
    "thread_id": "demo"
  }'
```

(Requires Brave Search MCP server configured)

### Example 5: Python Client

```python
import asyncio
import httpx

class DQEClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.thread_id = "python-client"
    
    async def chat(self, message: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/chat",
                json={"message": message, "thread_id": self.thread_id},
                timeout=120.0
            )
            data = response.json()
            return data["response"]
    
    async def reset(self):
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.base_url}/api/v1/reset",
                json={"thread_id": self.thread_id}
            )

async def main():
    client = DQEClient()
    
    # Example conversation
    response = await client.chat("Log in to NetSuite")
    print(f"Agent: {response}")
    
    response = await client.chat("Search for OP-20080")
    print(f"Agent: {response}")
    
    # Reset conversation
    await client.reset()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎓 Technical Details

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (state machine, checkpointing, human-in-the-loop) |
| Browser Control | Playwright (launch, navigate, click, type, screenshot) |
| Browser AI (CUA) | Anthropic Claude / OpenAI GPT-5.4 (selectable) |
| General LLM | Azure OpenAI GPT-5 Mini (reasoning, email drafting) |
| Workflow Engine | YAML + LangGraph builder |
| Data Models | Pydantic v2 |
| API | FastAPI |
| Tools | Registry + MCP |

### Model Separation

| Model | Purpose | Cost |
|-------|---------|------|
| GPT-5 Mini (Azure) | General reasoning, email drafting | Low |
| Claude Sonnet 4 / GPT-5.4 | Browser CUA | Medium |

**Why separate?** You don't pay CUA prices for cheap tasks like email drafting.

### CUA Loop

Each browser tool runs this loop internally:

```python
1. Take screenshot (Playwright)
2. Send screenshot + instruction to CUA model
3. Model returns actions: click(x,y), type("text"), key("Enter")
4. Execute actions via Playwright
5. Take new screenshot
6. Repeat until instruction complete
```

### Constants and Configuration

Key constants in codebase:

**Browser (cua.py):**
- `DEFAULT_MAX_STEPS = 20` - Max CUA loop iterations
- `DEFAULT_TYPING_DELAY_MS = 30` - Typing speed
- `DEFAULT_ACTION_WAIT_MS = 500` - Wait after actions
- `DEFAULT_SCROLL_AMOUNT = 100` - Scroll distance
- `DEFAULT_WAIT_DURATION_SEC = 2` - Wait action duration

**Smart Finder (smart_finder.py):**
- `SMART_CLICK_TIMEOUT_MS = 2000` - Element click timeout
- `MAX_SELECTOR_DEPTH = 5` - Selector specificity limit

**Browser Tools (browser_tools.py):**
- `FORM_FILL_MAX_STEPS = 30` - Max steps for form filling
- `SEARCH_QUERY_MAX_LEN = 20` - Screenshot filename limit

---

## 🐛 Troubleshooting

### API won't start

**Check:**
1. Is `.env` file configured correctly?
2. Are all required API keys set?
3. Is Playwright installed? Run: `playwright install chromium`

### Agent not calling tools

**Check:**
1. Are tools registered properly? Look for `@register_tool` decorator
2. Check logs for errors
3. Verify API keys are valid

### MCP tools not loading

**Check:**
1. Is `mcp_config.yaml` configured correctly?
2. Are environment variables set in `.env`?
3. Is Node.js installed (for npm MCP servers)?
4. Check logs for MCP loading errors

### Browser automation failing

**Check:**
1. Is `HEADLESS=false` so you can see what's happening?
2. Are credentials correct in `.env`?
3. Check screenshots in `screenshots/` directory
4. Review CUA logs for action details

---

## 📄 License

[Your license here]

## 🤝 Contributing

[Your contribution guidelines here]

## 📞 Support

[Your support information here]

---

**Built with ❤️ using LangGraph, Anthropic Claude, OpenAI, and Playwright**
