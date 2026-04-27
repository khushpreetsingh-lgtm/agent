# DQE Agent Architecture — Hierarchical Multi-Agent System

## Overview

DQE Agent uses **hierarchical multi-agent orchestration** with **domain-based routing**. Every user message goes through:

```
User Message
    ↓
Orchestrator (classify domains via keywords)
    ↓
├─ Single domain  → PEV loop (filtered tools)
└─ Multi-domain   → Send parallel PEV instances
    ↓
Aggregator (merge results)
    ↓
Response to User
```

**Key benefit:** Tool count scales linearly per agent (13-60 tools) instead of quadratically (all 126 tools).

## Architecture Components

### 1. Agent Registry (`src/dqe_agent/agents/`)

Auto-discovery via `pkgutil.iter_modules()` — same pattern as tools/flows.

**Base class (`agents/__init__.py`):**
```python
from abc import ABC
from dataclasses import dataclass

@dataclass
class ProactiveConfig:
    interval_seconds: int
    prompt: str

class AgentConfig(ABC):
    agent_id: str
    description: str = ""
    domains: list[str] | None = None
    tools: list[str] | None = None
    system_prompt: str = ""
    proactive: ProactiveConfig | None = None
```

**Registry functions:**
- `register_agent(agent: AgentConfig)` — add to global `_AGENT_REGISTRY`
- `discover_agents()` — scan `agents/` package, import all modules
- `get_agent(agent_id: str)` — lookup by ID
- `get_all_agents()` — list all
- `build_domain_index()` — flatten keyword→agent_id map

### 2. Built-in Agents

| Agent | Domains (count) | Tools (count) | Proactive Interval |
|-------|-----------------|---------------|-------------------|
| **jira** | jira, ticket, issue, sprint, board, epic, story, bug, task, backlog, worklog, hours, logged, assignee, priority, transition, status, component, version, label (20) | jira_create_issue, jira_update_issue, jira_search, jira_get_issue, jira_add_comment, jira_transition_issue, jira_get_all_projects, jira_get_sprint_issues, jira_add_worklog, jira_get_assignable_users, jira_get_priorities, jira_get_worklogs_by_date_range, direct_response (13) | 300s |
| **calendar** | calendar, meeting, schedule, event, invite, availability, slot, appointment, standup, sync, reminder, recurring (12) | list_calendars, get_events, create_event, modify_event, delete_event (5) | 600s |
| **email** | email, mail, send, inbox, reply, draft, gmail, compose, forward, attachment, thread, unread (12) | search_gmail_messages, get_gmail_message_content, send_gmail_message, draft_gmail_message, list_gmail_labels, modify_gmail_message_labels, direct_response (7) | 900s |
| **browser** | browser, web, navigate, login, click, fill, scrape, netsuite, cpq, website, page, screenshot, extract (13) | `tools=None` (all browser tools) | None |

**Browser agent = catch-all fallback** when no domain matches.

### 3. Orchestrator (`src/dqe_agent/agent/orchestrator.py`)

#### Domain Classification

Fast keyword matching:
```python
def classify_domains(message: str, domain_index: dict[str, str]) -> list[str]:
    msg_lower = message.lower()
    matched: dict[str, bool] = {}
    for keyword, agent_id in domain_index.items():
        if keyword in msg_lower:
            matched[agent_id] = True
    if not matched:
        return ["browser"]
    return list(matched.keys())
```

#### Task Decomposition

```python
def decompose_tasks(message: str, agent_ids: list[str]) -> list[dict[str, str]]:
    if len(agent_ids) == 1:
        return [{"agent": agent_ids[0], "task": message}]
    # Multi-domain: assign full message to each agent
    return [{"agent": aid, "task": message} for aid in agent_ids]
```

Each agent's planner focuses on its domain via scoped tool list + system prompt.

#### Orchestrator Node

```python
async def orchestrator_node(state: AgentState) -> dict | list[Send]:
    message_text = <extract last HumanMessage>
    domain_index = build_domain_index()
    agent_ids = classify_domains(message_text, domain_index)
    tasks = decompose_tasks(message_text, agent_ids)

    if len(tasks) == 1:
        # Single agent — set agent_id, fall through to PEV
        return {"agent_id": tasks[0]["agent"], "task": tasks[0]["task"], ...}

    # Multi-agent — fan out via Send
    sends = []
    for t in tasks:
        sub_state = dict(state)
        sub_state["agent_id"] = t["agent"]
        sub_state["task"] = t["task"]
        sends.append(Send("pev_subgraph", sub_state))
    return sends
```

#### Aggregator Node

```python
async def aggregator_node(state: AgentState) -> dict:
    sub_results = state.get("sub_results", [])
    responses = [r.get("result") for r in sub_results if r.get("tool") == "direct_response"]
    combined = "\n\n---\n\n".join(responses)
    return {"status": "complete", "sub_results": sub_results}
```

#### Orchestrator Graph

```python
def build_orchestrator_graph():
    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("aggregator", aggregator_node)
    
    pev = build_pev_graph()
    builder.add_node("pev_subgraph", pev.compile())

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", _route_after_orchestrator, ...)
    builder.add_edge("pev_subgraph", "aggregator")
    builder.add_edge("aggregator", END)
    
    return builder
```

### 4. PEV Loop with Tool Filtering (`src/dqe_agent/agent/loop.py`)

```python
def build_pev_graph(tool_filter: list[str] | None = None) -> StateGraph:
    if tool_filter is not None:
        _planner = partial(planner_node, _tool_filter=tool_filter)
        _executor = partial(executor_node, _tool_filter=tool_filter)
    else:
        _planner = planner_node
        _executor = executor_node

    builder = StateGraph(AgentState)
    builder.add_node("planner", _planner)
    builder.add_node("executor", _executor)
    builder.add_node("verifier", verifier_node)
    # ... edges
    return builder
```

**Planner node** (`planner.py`):
```python
async def planner_node(state: AgentState, _tool_filter: list[str] | None = None) -> dict:
    # Auto-load tool filter from agent_id if not provided
    if _tool_filter is None and state.get("agent_id"):
        agent_cfg = get_agent(state["agent_id"])
        if agent_cfg.tools is not None:
            _tool_filter = agent_cfg.tools
    
    # Filter MCP tools
    if _tool_filter is not None:
        mcp_tools = [t for t in mcp_tools if t in _tool_filter]
    
    # Build tool description block + call LLM
    ...
```

**Executor node** (`executor.py`): same pattern.

### 5. State Extensions (`src/dqe_agent/state.py`)

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages

def _replace(left, right):
    return right

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_id: Annotated[str, _replace]
    orchestrator_tasks: Annotated[list, _replace]
    sub_results: Annotated[list, _replace]
    # ... existing fields
```

- `agent_id`: current sub-agent ID (jira/calendar/email/browser)
- `orchestrator_tasks`: list of `[{"agent": "jira", "task": "..."}]` for tracking
- `sub_results`: aggregated results from parallel sub-agents

### 6. Proactive Monitor (`orchestrator.py`)

Background task polls agents with `proactive` config, broadcasts alerts via WebSocket.

```python
class ProactiveMonitor:
    def __init__(self, broadcast_fn) -> None:
        self._broadcast = broadcast_fn
        self._stopped = False
        self._next_run: dict[str, float] = {}

    async def start(self) -> None:
        while not self._stopped:
            now = time.monotonic()
            for agent in get_all_agents():
                if agent.proactive is None:
                    continue
                next_run = self._next_run.get(agent.agent_id, 0.0)
                if now < next_run:
                    continue
                self._next_run[agent.agent_id] = now + agent.proactive.interval_seconds
                asyncio.create_task(self._run_agent_check(agent))
            await asyncio.sleep(10)

    async def _run_agent_check(self, agent) -> None:
        llm = get_executor_llm()
        prompt = f"{agent.proactive.prompt}\nIf noteworthy, respond. Else: NO_ALERT"
        result = await llm.ainvoke([SystemMessage(content=prompt)])
        content = result.content if hasattr(result, "content") else str(result)
        if content and "NO_ALERT" not in content:
            await self._broadcast({"type": "proactive_alert", "agent": agent.agent_id, "content": content})
```

**Wired in `api.py` lifespan:**
```python
_proactive_monitor_task: asyncio.Task | None = None
_active_websockets: set[WebSocket] = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proactive_monitor_task
    
    # Startup
    async def _broadcast_alert(msg: dict):
        for ws in _active_websockets:
            await ws.send_json(msg)
    
    monitor = ProactiveMonitor(broadcast_fn=_broadcast_alert)
    _proactive_monitor_task = asyncio.create_task(monitor.start())
    
    yield
    
    # Shutdown
    monitor.stop()
    await asyncio.wait_for(_proactive_monitor_task, timeout=2.0)
```

WebSocket tracking:
```python
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    _active_websockets.add(ws)
    try:
        # ... message handling
    finally:
        _active_websockets.discard(ws)
```

### 7. Master Agent Integration (`src/dqe_agent/agent/master.py`)

```python
def _build_graph(self) -> None:
    from dqe_agent.agent.orchestrator import build_orchestrator_graph
    graph = build_orchestrator_graph()
    self._graph = graph.compile(checkpointer=self.checkpointer)
```

## How to Add a New Agent

### Step 1: Create Agent Config

File: `src/dqe_agent/agents/slack_agent.py`

```python
from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent

class SlackAgent(AgentConfig):
    agent_id = "slack"
    description = "Slack workspace interaction agent"
    domains = [
        "slack", "message", "channel", "dm", "thread", "mention",
        "workspace", "post", "react", "emoji", "status"
    ]
    tools = [
        "slack_send_message",
        "slack_get_channels",
        "slack_get_messages",
        "slack_set_status",
        "direct_response",
    ]
    system_prompt = """You are a Slack agent. Help users:
- Send messages to channels or DMs
- Search message history
- Manage workspace presence
- React to messages

Always confirm channel names before posting."""

    proactive = ProactiveConfig(
        interval_seconds=600,  # 10 minutes
        prompt="Check for unread mentions or high-priority DMs in Slack. Alert if urgent."
    )

register_agent(SlackAgent())
```

### Step 2: Verify Auto-Discovery

Agents discovered automatically at startup via `discover_agents()` in `api.py` lifespan.

Check logs for:
```
15:20:19 [INFO] dqe_agent.agents: Registered agent: slack (domains=['slack', 'message', ...])
```

### Step 3: Test

```python
# tests/test_agents.py
def test_slack_agent_registered():
    from dqe_agent.agents import get_agent
    slack = get_agent("slack")
    assert slack.agent_id == "slack"
    assert "slack" in slack.domains
    assert "slack_send_message" in slack.tools
```

**That's it.** No wiring changes. Orchestrator auto-picks up new agent.

## How to Add Agent Tools

### Option A: Built-in Tool (Python function)

File: `src/dqe_agent/tools/slack_tools.py`

```python
from dqe_agent.tools import register_tool

@register_tool(
    name="slack_send_message",
    description="Send message to Slack channel or DM. Args: channel (str), text (str)."
)
async def slack_send_message(channel: str, text: str) -> str:
    # Implementation
    return f"Sent to {channel}: {text}"
```

### Option B: MCP Server Tool

Add to `mcp_config.yaml`:
```yaml
mcpServers:
  slack:
    command: uvx
    args:
      - mcp-server-slack
    env:
      SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN}
      SLACK_TEAM_ID: ${SLACK_TEAM_ID}
```

MCP tools auto-register at startup via `load_mcp_servers()`.

### Step 3: Reference in Agent Config

Add tool name to `SlackAgent.tools` list.

## How to Add New Domains

Edit existing agent config:

```python
class JiraAgent(AgentConfig):
    agent_id = "jira"
    domains = [
        "jira", "ticket", "issue", "sprint", "board",
        # Add new keywords:
        "confluence", "wiki", "documentation",  # if Jira agent handles Confluence too
    ]
```

Restart server. Orchestrator uses updated domain index.

## How to Add Proactive Monitoring

Add/edit `proactive` field in agent config:

```python
class CalendarAgent(AgentConfig):
    # ...
    proactive = ProactiveConfig(
        interval_seconds=300,  # 5 minutes
        prompt="""Check upcoming calendar events in next 15 minutes.
If meeting starting soon and user not in call, send reminder."""
    )
```

ProactiveMonitor calls LLM with this prompt every 300s. If LLM response != "NO_ALERT", broadcasts to all WebSocket clients.

## Testing

Run full test suite:
```bash
pytest
```

Key test files:
- `tests/test_agents.py` — registry, domain index, tool filtering
- `tests/test_orchestrator.py` — classification, decomposition, proactive monitor
- `tests/test_graph.py` — PEV loop, LangGraph execution

## Architecture Benefits

| Feature | Before (Monolithic) | After (Hierarchical) |
|---------|---------------------|----------------------|
| Tool count per request | 126 tools | 13-60 tools (per agent) |
| LLM token cost | High (all tool schemas) | Low (filtered schemas) |
| Adding new agent | Modify planner/executor | Drop file in `agents/`, auto-discovered |
| Multi-domain requests | Sequential tool calls | Parallel sub-agent execution |
| Proactive monitoring | Manual polling | Per-agent background tasks |
| Maintainability | Single 500-line planner | 4 focused agents |

## Common Operations

### View Registered Agents
```python
from dqe_agent.agents import get_all_agents
for agent in get_all_agents():
    print(f"{agent.agent_id}: {len(agent.domains)} domains, {len(agent.tools or [])} tools")
```

### View Domain Index
```python
from dqe_agent.agents import build_domain_index
index = build_domain_index()
print(index)
# {'jira': 'jira', 'ticket': 'jira', 'calendar': 'calendar', ...}
```

### Manually Test Orchestrator Classification
```python
from dqe_agent.agent.orchestrator import classify_domains
from dqe_agent.agents import build_domain_index

message = "Create a Jira ticket and schedule a meeting"
index = build_domain_index()
agents = classify_domains(message, index)
print(agents)  # ['jira', 'calendar']
```


Backward compatible:
- Existing single-agent workflows work unchanged
- Tool filter defaults to `None` (no filtering) when `agent_id` not set
- Browser tools still available when no domain matches
- All 126 tools registered and available (just filtered per agent)
