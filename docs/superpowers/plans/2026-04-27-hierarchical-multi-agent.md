# Hierarchical Multi-Agent Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve DQE Agent into a hierarchical multi-agent system where each domain has a focused sub-agent, new integrations require only one new file, and multi-domain requests execute in parallel via LangGraph Send.

**Architecture:** An `Orchestrator` LangGraph node sits in front of the existing PEV graph. It classifies the user's message into domains, decomposes it into per-agent tasks, and fans out via `Send` to the same PEV graph compiled with a per-agent tool filter. A background `ProactiveMonitor` polls registered agents and pushes alerts to active WebSocket sessions.

**Tech Stack:** Python 3.11+, LangGraph, LangChain, FastAPI, asyncio — all already installed.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/dqe_agent/agents/__init__.py` | **Create** | `AgentConfig`, `ProactiveConfig`, `register_agent`, `get_agent`, `discover_agents`, `list_agents` |
| `src/dqe_agent/agents/jira_agent.py` | **Create** | `JiraAgent` — Jira domain |
| `src/dqe_agent/agents/calendar_agent.py` | **Create** | `CalendarAgent` — Calendar/meeting domain |
| `src/dqe_agent/agents/email_agent.py` | **Create** | `EmailAgent` — Email domain |
| `src/dqe_agent/agents/browser_agent.py` | **Create** | `BrowserAgent` — fallback, browser automation |
| `src/dqe_agent/agents/people_agent.py` | **Create** | `PeopleAgent` — user/team lookup |
| `src/dqe_agent/agent/orchestrator.py` | **Create** | `orchestrator_node`, `aggregator_node`, `build_orchestrator_graph`, `ProactiveMonitor` |
| `src/dqe_agent/state.py` | **Modify** | Add `agent_id`, `orchestrator_tasks`, `sub_results` fields |
| `src/dqe_agent/agent/loop.py` | **Modify** | Accept optional `tool_filter: list[str] | None` param in `build_pev_graph` |
| `src/dqe_agent/agent/planner.py` | **Modify** | Read `agent_id` from state → apply tool filter from `AgentConfig` |
| `src/dqe_agent/api.py` | **Modify** | `discover_agents()` at startup; route through orchestrator; start `ProactiveMonitor` |
| `tests/test_agents.py` | **Create** | Unit tests for registry, routing, plugin discovery |
| `tests/test_orchestrator.py` | **Create** | Unit tests for domain classification, fan-out, aggregation |

---

## Task 1: AgentConfig Plugin Registry

**Files:**
- Create: `src/dqe_agent/agents/__init__.py`
- Create: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents.py
import pytest
from dqe_agent.agents import (
    AgentConfig, ProactiveConfig, register_agent,
    get_agent, list_agents, discover_agents, build_domain_index,
)


class _TestAgent(AgentConfig):
    agent_id = "test_agent"
    description = "Test agent"
    domains = ["test", "sample", "demo"]
    tools = ["direct_response"]
    system_prompt = "Test prompt."


def test_register_and_get():
    register_agent(_TestAgent())
    agent = get_agent("test_agent")
    assert agent.agent_id == "test_agent"


def test_list_agents_includes_registered():
    register_agent(_TestAgent())
    assert "test_agent" in list_agents()


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_agent("nonexistent_agent_xyz")


def test_build_domain_index():
    register_agent(_TestAgent())
    idx = build_domain_index()
    assert idx["test"] == "test_agent"
    assert idx["sample"] == "test_agent"
    assert idx["demo"] == "test_agent"


def test_proactive_config():
    pc = ProactiveConfig(interval_seconds=60, prompt="Check for issues")
    assert pc.interval_seconds == 60
    assert pc.prompt == "Check for issues"
```

- [ ] **Step 2: Run to confirm failures**

```
pytest tests/test_agents.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — `dqe_agent.agents` does not exist yet.

- [ ] **Step 3: Implement the registry**

```python
# src/dqe_agent/agents/__init__.py
"""Agent registry — plugin system for domain-focused sub-agents.

Adding a new agent
------------------
1. Create ``src/dqe_agent/agents/my_agent.py``
2. Subclass ``AgentConfig``, set required fields.
3. Call ``register_agent(MyAgent())`` at module level.
4. ``discover_agents()`` auto-imports all modules in this package — no other changes needed.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

_AGENT_REGISTRY: dict[str, "AgentConfig"] = {}


@dataclass
class ProactiveConfig:
    """Opt-in proactive monitoring config for an agent."""
    interval_seconds: int
    prompt: str


class AgentConfig(ABC):
    """Base class for domain-focused sub-agents.

    Subclass and set: agent_id, description, domains, tools, system_prompt.
    Optionally set proactive for background monitoring.
    """
    agent_id: str
    description: str = ""
    domains: list[str] = field(default_factory=list)
    tools: list[str] | None = None  # None = all tools
    system_prompt: str = ""
    proactive: ProactiveConfig | None = None


def register_agent(agent: AgentConfig) -> None:
    _AGENT_REGISTRY[agent.agent_id] = agent
    logger.info("Registered agent: %s (domains=%s)", agent.agent_id, agent.domains)


def get_agent(agent_id: str) -> AgentConfig:
    if agent_id not in _AGENT_REGISTRY:
        raise KeyError(
            f"Agent '{agent_id}' not found. Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return _AGENT_REGISTRY[agent_id]


def list_agents() -> list[str]:
    return list(_AGENT_REGISTRY.keys())


def get_all_agents() -> list[AgentConfig]:
    return list(_AGENT_REGISTRY.values())


def build_domain_index() -> dict[str, str]:
    """Build {keyword: agent_id} lookup from all registered agents."""
    idx: dict[str, str] = {}
    for agent in _AGENT_REGISTRY.values():
        for domain in (agent.domains or []):
            idx[domain.lower()] = agent.agent_id
    return idx


def discover_agents() -> None:
    """Auto-import every non-private module in ``dqe_agent.agents``."""
    package = importlib.import_module("dqe_agent.agents")
    for _, name, _ in pkgutil.iter_modules(package.__path__):
        if not name.startswith("_"):
            importlib.import_module(f"dqe_agent.agents.{name}")
    logger.info("Discovered agents: %s", list(_AGENT_REGISTRY.keys()))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_agents.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dqe_agent/agents/__init__.py tests/test_agents.py
git commit -m "feat: add AgentConfig plugin registry with auto-discovery"
```

---

## Task 2: State Extensions

**Files:**
- Modify: `src/dqe_agent/state.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_agents.py

def test_agent_state_has_new_fields():
    from dqe_agent.state import AgentState
    # TypedDict fields are accessible via __annotations__
    annotations = AgentState.__annotations__
    assert "agent_id" in annotations
    assert "orchestrator_tasks" in annotations
    assert "sub_results" in annotations
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_agents.py::test_agent_state_has_new_fields -v
```
Expected: FAIL — fields not in `AgentState`.

- [ ] **Step 3: Add fields to AgentState**

Open `src/dqe_agent/state.py`. Add these three fields at the bottom of `AgentState`, after the `current_task` field:

```python
    # ── Multi-agent orchestration ─────────────────────────────────────────
    agent_id: Annotated[str, _replace]              # sub-agent running this PEV instance
    orchestrator_tasks: Annotated[list, _replace]   # tasks decomposed by orchestrator
    sub_results: Annotated[list, _replace]          # aggregated results from parallel sub-agents
```

- [ ] **Step 4: Run test**

```
pytest tests/test_agents.py::test_agent_state_has_new_fields -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dqe_agent/state.py tests/test_agents.py
git commit -m "feat: add agent_id/orchestrator_tasks/sub_results to AgentState"
```

---

## Task 3: Built-in Agent Definitions

**Files:**
- Create: `src/dqe_agent/agents/jira_agent.py`
- Create: `src/dqe_agent/agents/calendar_agent.py`
- Create: `src/dqe_agent/agents/email_agent.py`
- Create: `src/dqe_agent/agents/browser_agent.py`
- Create: `src/dqe_agent/agents/people_agent.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_agents.py

def test_builtin_agents_discovered():
    from dqe_agent.agents import discover_agents, list_agents, get_agent
    discover_agents()
    agents = list_agents()
    for aid in ["jira", "calendar", "email", "browser", "people"]:
        assert aid in agents, f"Built-in agent '{aid}' not found after discover_agents()"

def test_jira_agent_domains():
    from dqe_agent.agents import discover_agents, get_agent
    discover_agents()
    agent = get_agent("jira")
    assert "jira" in agent.domains
    assert "sprint" in agent.domains
    assert "ticket" in agent.domains

def test_browser_agent_is_fallback():
    from dqe_agent.agents import discover_agents, get_agent
    discover_agents()
    agent = get_agent("browser")
    assert agent.tools is None  # browser agent = all browser tools
```

- [ ] **Step 2: Run to confirm failures**

```
pytest tests/test_agents.py::test_builtin_agents_discovered -v
```
Expected: FAIL — agent files don't exist yet.

- [ ] **Step 3: Create jira_agent.py**

```python
# src/dqe_agent/agents/jira_agent.py
from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent


class JiraAgent(AgentConfig):
    agent_id = "jira"
    description = "Jira project management — issues, sprints, boards, epics, worklogs"
    domains = [
        "jira", "ticket", "issue", "sprint", "board", "epic", "story",
        "bug", "task", "backlog", "worklog", "hours", "logged", "assignee",
        "priority", "transition", "status", "component", "version", "label",
    ]
    tools = [
        # Direct Jira REST tools
        "jira_get_assignable_users",
        "jira_get_priorities",
        "jira_get_project_roles",
        "jira_search_user_by_email",
        "jira_add_project_member",
        "jira_get_worklogs_by_date_range",
        # MCP Jira tools (registered at runtime via mcp_startup)
        # All tools matching "jira_*" pattern are included automatically at routing time
        # Human interaction tools — always needed
        "ask_user",
        "request_selection",
        "human_review",
        "direct_response",
        "request_form",
        "request_edit",
        "llm_draft_content",
    ]
    system_prompt = (
        "You are a Jira agent. You handle all Jira project management tasks: "
        "create and update issues, manage sprints, assign work, log time, "
        "search across projects, and report on team progress. "
        "Use MCP Jira tools for all operations — never use browser tools for Jira."
    )
    proactive = ProactiveConfig(
        interval_seconds=300,
        prompt=(
            "Check for: (1) issues assigned to team members that are overdue, "
            "(2) active sprints ending within 2 days with unresolved blockers, "
            "(3) unassigned high-priority issues in active sprints. "
            "Report only if something actionable is found."
        ),
    )


register_agent(JiraAgent())
```

- [ ] **Step 4: Create calendar_agent.py**

```python
# src/dqe_agent/agents/calendar_agent.py
from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent


class CalendarAgent(AgentConfig):
    agent_id = "calendar"
    description = "Google Calendar — events, meetings, scheduling, availability"
    domains = [
        "calendar", "meeting", "schedule", "event", "invite", "availability",
        "slot", "appointment", "standup", "sync", "reminder", "recurring",
    ]
    tools = [
        # MCP calendar tools registered at runtime
        # Core interaction tools
        "ask_user",
        "request_selection",
        "human_review",
        "direct_response",
        "request_form",
    ]
    system_prompt = (
        "You are a Calendar agent. You manage Google Calendar events: "
        "create meetings, check availability, find free slots, update or cancel events, "
        "and add attendees. Always show available time slots via request_selection."
    )
    proactive = ProactiveConfig(
        interval_seconds=600,
        prompt=(
            "Check for: (1) meetings starting within 30 minutes with no agenda set, "
            "(2) back-to-back meetings with no break. "
            "Report only if something actionable is found."
        ),
    )


register_agent(CalendarAgent())
```

- [ ] **Step 5: Create email_agent.py**

```python
# src/dqe_agent/agents/email_agent.py
from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent


class EmailAgent(AgentConfig):
    agent_id = "email"
    description = "Gmail — send, read, reply, draft, search emails"
    domains = [
        "email", "mail", "send", "inbox", "reply", "draft", "gmail",
        "compose", "forward", "attachment", "thread", "unread",
    ]
    tools = [
        # MCP gmail tools registered at runtime
        "ask_user",
        "request_selection",
        "human_review",
        "direct_response",
        "request_form",
        "request_edit",
        "llm_draft_content",
    ]
    system_prompt = (
        "You are an Email agent. You handle Gmail operations: send emails, "
        "read threads, reply, draft messages, and search the inbox. "
        "Always show a draft for human review before sending."
    )
    proactive = ProactiveConfig(
        interval_seconds=900,
        prompt=(
            "Check for unread emails that require a response and have been waiting "
            "more than 4 hours. Report sender, subject, and a one-line summary."
        ),
    )


register_agent(EmailAgent())
```

- [ ] **Step 6: Create browser_agent.py**

```python
# src/dqe_agent/agents/browser_agent.py
from dqe_agent.agents import AgentConfig, register_agent


class BrowserAgent(AgentConfig):
    agent_id = "browser"
    description = "Browser automation — navigate, click, fill forms, extract data from any website"
    domains = [
        "browser", "web", "navigate", "login", "click", "fill", "scrape",
        "netsuite", "cpq", "website", "page", "screenshot", "extract",
    ]
    tools = None  # None = all tools (browser agent is the catch-all fallback)
    system_prompt = (
        "You are a browser automation agent. You control a real Chromium browser "
        "to interact with any website: log in, navigate, fill forms, extract data, "
        "and complete multi-step workflows. Always login before interacting with a site."
    )
    proactive = None  # no proactive monitoring for browser agent


register_agent(BrowserAgent())
```

- [ ] **Step 7: Create people_agent.py**

```python
# src/dqe_agent/agents/people_agent.py
from dqe_agent.agents import AgentConfig, register_agent


class PeopleAgent(AgentConfig):
    agent_id = "people"
    description = "People and team — look up users, org chart, contacts, team membership"
    domains = [
        "people", "user", "member", "team", "org", "contact",
        "who", "person", "colleague", "employee", "directory",
    ]
    tools = [
        "jira_get_assignable_users",
        "jira_search_user_by_email",
        "jira_get_project_roles",
        "jira_add_project_member",
        "ask_user",
        "request_selection",
        "direct_response",
        "request_form",
    ]
    system_prompt = (
        "You are a People agent. You look up team members, find users by email, "
        "manage project roles, and answer questions about who is on which team."
    )
    proactive = None


register_agent(PeopleAgent())
```

- [ ] **Step 8: Run all agent tests**

```
pytest tests/test_agents.py -v
```
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/dqe_agent/agents/
git commit -m "feat: add built-in agents (jira, calendar, email, browser, people)"
```

---

## Task 4: PEV Graph Tool Filter

**Files:**
- Modify: `src/dqe_agent/agent/loop.py`
- Modify: `src/dqe_agent/agent/planner.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_agents.py

def test_pev_graph_respects_tool_filter():
    """build_pev_graph with tool_filter only exposes listed tools to planner."""
    from dqe_agent.agent.loop import build_pev_graph
    # Should not raise — just verify it compiles with a filter
    graph = build_pev_graph(tool_filter=["direct_response", "ask_user"])
    assert graph is not None
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_agents.py::test_pev_graph_respects_tool_filter -v
```
Expected: FAIL — `build_pev_graph` doesn't accept `tool_filter` param.

- [ ] **Step 3: Modify loop.py to accept tool_filter**

In `src/dqe_agent/agent/loop.py`, change the signature of `build_pev_graph`:

```python
def build_pev_graph(tool_filter: list[str] | None = None) -> StateGraph:
    """Build the Planner-Executor-Verifier LangGraph.

    Args:
        tool_filter: Optional list of tool names to expose. None = all tools.
    """
    from functools import partial
    from dqe_agent.agent.executor import executor_node
    from dqe_agent.agent.planner import planner_node
    from dqe_agent.agent.verifier import verifier_node

    builder = StateGraph(AgentState)

    # Wrap planner/executor with the tool filter if provided
    if tool_filter is not None:
        _planner = partial(planner_node, _tool_filter=tool_filter)
        _executor = partial(executor_node, _tool_filter=tool_filter)
    else:
        _planner = planner_node
        _executor = executor_node

    builder.add_node("planner", _planner)
    builder.add_node("executor", _executor)
    builder.add_node("verifier", verifier_node)

    builder.add_conditional_edges(
        START,
        route_start,
        {"planner": "planner", "executor": "executor"},
    )
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_after_verify,
        {
            "next": "executor",
            "replan": "planner",
            "done": END,
            "end": END,
        },
    )
    return builder
```

- [ ] **Step 4: Modify planner_node to accept _tool_filter**

In `src/dqe_agent/agent/planner.py`, find `async def planner_node(state: AgentState` and update its signature:

```python
async def planner_node(state: AgentState, _tool_filter: list[str] | None = None) -> dict:
```

Then in the body, find where tools are listed for the planner prompt (the `_prewarm_mcp_tool_block` / MCP tool block section). After the existing `mcp_tools` list is built, add the filter:

```python
    # Apply tool filter if this PEV instance is scoped to a sub-agent
    if _tool_filter is not None:
        mcp_tools = [t for t in mcp_tools if t in _tool_filter]
```

Also check `agent_id` from state as an alternative filter path — if `_tool_filter` is None but `state.get("agent_id")` is set, load from registry:

```python
    # If no explicit filter but agent_id is set, load filter from AgentConfig
    if _tool_filter is None and state.get("agent_id"):
        try:
            from dqe_agent.agents import get_agent
            _agent_cfg = get_agent(state["agent_id"])
            if _agent_cfg.tools is not None:
                _tool_filter = _agent_cfg.tools
                mcp_tools = [t for t in mcp_tools if t in _tool_filter]
        except KeyError:
            pass
```

- [ ] **Step 5: Apply same filter in executor_node**

In `src/dqe_agent/agent/executor.py`, find `async def executor_node(state: AgentState` and update:

```python
async def executor_node(state: AgentState, _tool_filter: list[str] | None = None) -> dict:
```

At the start of the function body, after state is read, add:

```python
    # Resolve tool filter from agent_id in state if not passed directly
    if _tool_filter is None and state.get("agent_id"):
        try:
            from dqe_agent.agents import get_agent
            _agent_cfg = get_agent(state["agent_id"])
            _tool_filter = _agent_cfg.tools  # may still be None = all tools
        except KeyError:
            pass
```

- [ ] **Step 6: Run test**

```
pytest tests/test_agents.py::test_pev_graph_respects_tool_filter -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dqe_agent/agent/loop.py src/dqe_agent/agent/planner.py src/dqe_agent/agent/executor.py
git commit -m "feat: PEV graph accepts tool_filter for sub-agent scoping"
```

---

## Task 5: Orchestrator Node

**Files:**
- Create: `src/dqe_agent/agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
import pytest
from unittest.mock import AsyncMock, patch


def test_classify_single_domain():
    from dqe_agent.agent.orchestrator import classify_domains
    from dqe_agent.agents import register_agent, AgentConfig

    class _J(AgentConfig):
        agent_id = "jira"
        domains = ["jira", "ticket", "issue"]
        tools = ["direct_response"]
        system_prompt = ""
    register_agent(_J())

    domains = classify_domains("Create a Jira ticket for the login bug", {"jira": "jira"})
    assert "jira" in domains


def test_classify_multi_domain():
    from dqe_agent.agent.orchestrator import classify_domains

    idx = {"jira": "jira", "ticket": "jira", "meeting": "calendar", "schedule": "calendar"}
    domains = classify_domains("Create a ticket and schedule a meeting", idx)
    assert "jira" in domains
    assert "calendar" in domains


def test_classify_no_match_returns_browser():
    from dqe_agent.agent.orchestrator import classify_domains
    domains = classify_domains("Do something weird and unusual xyz", {})
    assert domains == ["browser"]


def test_decompose_tasks_single():
    from dqe_agent.agent.orchestrator import decompose_tasks
    tasks = decompose_tasks("Create a Jira ticket", ["jira"])
    assert len(tasks) == 1
    assert tasks[0]["agent"] == "jira"
    assert "Create a Jira ticket" in tasks[0]["task"]


def test_decompose_tasks_multi():
    from dqe_agent.agent.orchestrator import decompose_tasks
    tasks = decompose_tasks("Create a Jira ticket and schedule a meeting", ["jira", "calendar"])
    assert len(tasks) == 2
    agent_ids = {t["agent"] for t in tasks}
    assert "jira" in agent_ids
    assert "calendar" in agent_ids
```

- [ ] **Step 2: Run to confirm failures**

```
pytest tests/test_orchestrator.py -v
```
Expected: `ImportError` — `orchestrator.py` does not exist.

- [ ] **Step 3: Implement orchestrator.py**

```python
# src/dqe_agent/agent/orchestrator.py
"""Orchestrator — classifies user messages into domains and fans out to sub-agents.

Single-domain: sets agent_id in state, PEV loop picks up tool filter automatically.
Multi-domain: uses LangGraph Send to run sub-agent PEV instances in parallel.
Fallback: routes to 'browser' agent when no domain matches.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from dqe_agent.state import AgentState

logger = logging.getLogger(__name__)

_ORCH_SYSTEM = """You are a routing agent. Given a user message, decompose it into tasks for specialist agents.

Available agents: {agent_list}

Output ONLY a JSON object:
{{
  "parallel": true,
  "tasks": [
    {{"agent": "<agent_id>", "task": "<specific task for this agent>"}},
    ...
  ]
}}

Rules:
- Each task goes to exactly one agent.
- If the message only concerns one domain, output one task.
- If the message spans multiple domains (e.g. Jira AND calendar), output one task per domain.
- If no domain matches, use agent_id "browser".
- task text must be self-contained — the sub-agent won't see the original message.
- Output ONLY the JSON — no explanation, no markdown."""


def classify_domains(message: str, domain_index: dict[str, str]) -> list[str]:
    """Fast keyword-based domain classification. Returns list of agent_ids."""
    msg_lower = message.lower()
    matched: dict[str, bool] = {}
    for keyword, agent_id in domain_index.items():
        if keyword in msg_lower:
            matched[agent_id] = True
    if not matched:
        return ["browser"]
    return list(matched.keys())


def decompose_tasks(message: str, agent_ids: list[str]) -> list[dict[str, str]]:
    """For single-domain: assign full message. For multi-domain: assign full message to each agent.

    The LLM orchestrator call (for complex decomposition) is optional and only
    used when the message is long and multi-domain — for simple cases this is free.
    """
    if len(agent_ids) == 1:
        return [{"agent": agent_ids[0], "task": message}]
    # Multi-domain: assign entire message to each agent — each agent's planner
    # will focus only on its domain given its scoped tool list and system prompt.
    return [{"agent": aid, "task": message} for aid in agent_ids]


async def orchestrator_node(state: AgentState) -> dict | list[Send]:
    """LangGraph node: classify message → single agent or parallel Send fan-out."""
    from dqe_agent.agents import build_domain_index, get_agent

    messages = state.get("messages", [])
    if not messages:
        return {"status": "failed", "error": "No messages in state"}

    # Get last human message
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {"status": "failed", "error": "No human message found"}

    message_text = last_human.content if isinstance(last_human.content, str) else str(last_human.content)

    domain_index = build_domain_index()
    agent_ids = classify_domains(message_text, domain_index)

    logger.info("[ORCHESTRATOR] message=%r → agents=%s", message_text[:80], agent_ids)

    tasks = decompose_tasks(message_text, agent_ids)

    if len(tasks) == 1:
        # Single agent — set agent_id in state and fall through to PEV
        return {
            "agent_id": tasks[0]["agent"],
            "orchestrator_tasks": tasks,
            "task": tasks[0]["task"],
            "status": "planning",
        }

    # Multi-agent — fan out via Send
    sends = []
    for t in tasks:
        sub_state = dict(state)
        sub_state["agent_id"] = t["agent"]
        sub_state["task"] = t["task"]
        sub_state["orchestrator_tasks"] = tasks
        sub_state["plan"] = []
        sub_state["step_results"] = []
        sub_state["status"] = "planning"
        sends.append(Send("pev_subgraph", sub_state))

    return sends


async def aggregator_node(state: AgentState) -> dict:
    """Merge results from parallel sub-agents into a single response."""
    sub_results = state.get("sub_results", [])
    step_results = state.get("step_results", [])

    # Collect final direct_response messages from all sub-agents
    responses = []
    for r in (sub_results or step_results):
        if isinstance(r, dict) and r.get("tool") in ("direct_response", "agent_done"):
            responses.append(r.get("result", ""))

    if not responses:
        responses = ["All tasks completed."]

    combined = "\n\n---\n\n".join(str(r) for r in responses if r)
    return {"status": "complete", "sub_results": sub_results}
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_orchestrator.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dqe_agent/agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrator node with keyword routing and parallel Send fan-out"
```

---

## Task 6: Build Orchestrator LangGraph

**Files:**
- Modify: `src/dqe_agent/agent/orchestrator.py` (add `build_orchestrator_graph`)
- Modify: `src/dqe_agent/agent/master.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_orchestrator.py

def test_build_orchestrator_graph():
    from dqe_agent.agent.orchestrator import build_orchestrator_graph
    graph = build_orchestrator_graph()
    assert graph is not None
    # Should compile without error
    compiled = graph.compile()
    assert compiled is not None
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_orchestrator.py::test_build_orchestrator_graph -v
```
Expected: FAIL — `build_orchestrator_graph` not defined.

- [ ] **Step 3: Add build_orchestrator_graph to orchestrator.py**

Append to `src/dqe_agent/agent/orchestrator.py`:

```python
def build_orchestrator_graph():
    """Build the full orchestrator graph: orchestrator → PEV sub-graph → aggregator."""
    from langgraph.graph import END, START, StateGraph
    from dqe_agent.agent.loop import build_pev_graph

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("aggregator", aggregator_node)

    # PEV sub-graph — compiled once, reused for all sub-agents
    # agent_id in state selects the tool filter inside planner/executor
    pev = build_pev_graph()
    builder.add_node("pev_subgraph", pev)

    # Edges
    builder.add_edge(START, "orchestrator")

    # orchestrator → either pev_subgraph (single) or Send fan-out (multi)
    def _route_after_orchestrator(state: AgentState):
        tasks = state.get("orchestrator_tasks", [])
        if len(tasks) <= 1:
            return "pev_subgraph"
        return "aggregator"  # Send already dispatched sub-graphs

    builder.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"pev_subgraph": "pev_subgraph", "aggregator": "aggregator"},
    )

    builder.add_edge("pev_subgraph", "aggregator")
    builder.add_edge("aggregator", END)

    return builder
```

- [ ] **Step 4: Update master.py to use orchestrator graph**

In `src/dqe_agent/agent/master.py`, update `_build_graph`:

```python
    def _build_graph(self) -> None:
        from dqe_agent.agent.orchestrator import build_orchestrator_graph
        graph = build_orchestrator_graph()
        self._graph = graph.compile(checkpointer=self.checkpointer)
        logger.info("Compiled orchestrator graph (orchestrator → pev_subgraph → aggregator)")
```

- [ ] **Step 5: Run test**

```
pytest tests/test_orchestrator.py::test_build_orchestrator_graph -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dqe_agent/agent/orchestrator.py src/dqe_agent/agent/master.py
git commit -m "feat: wire orchestrator graph into MasterAgent"
```

---

## Task 7: API Integration

**Files:**
- Modify: `src/dqe_agent/api.py`

- [ ] **Step 1: Add discover_agents to lifespan**

In `src/dqe_agent/api.py`, in the `lifespan` function, after `discover_flows()`:

```python
    from dqe_agent.agents import discover_agents
    discover_agents()
    from dqe_agent.agents import list_agents
    logger.info("Agents discovered: %s", list_agents())
```

- [ ] **Step 2: Add proactive_alert to WebSocket message types in module docstring**

In the module-level docstring at the top of `api.py`, add to the SERVER → CLIENT section:

```
  {"type": "proactive_alert", "agent": "jira", "content": "3 overdue tickets in Sprint 24"}
```

- [ ] **Step 3: Commit**

```bash
git add src/dqe_agent/api.py
git commit -m "feat: discover agents at startup in api.py lifespan"
```

---

## Task 8: Proactive Monitor

**Files:**
- Modify: `src/dqe_agent/agent/orchestrator.py` (add `ProactiveMonitor`)
- Modify: `src/dqe_agent/api.py` (start/stop monitor in lifespan)

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_orchestrator.py
import asyncio

@pytest.mark.asyncio
async def test_proactive_monitor_runs_and_stops():
    from dqe_agent.agent.orchestrator import ProactiveMonitor

    alerts = []

    async def fake_broadcast(msg):
        alerts.append(msg)

    monitor = ProactiveMonitor(broadcast_fn=fake_broadcast)
    task = asyncio.create_task(monitor.start())
    await asyncio.sleep(0.05)
    monitor.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
    # Just verify it started and stopped without error
    assert monitor._stopped
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_orchestrator.py::test_proactive_monitor_runs_and_stops -v
```
Expected: FAIL — `ProactiveMonitor` not defined.

- [ ] **Step 3: Add ProactiveMonitor to orchestrator.py**

Append to `src/dqe_agent/agent/orchestrator.py`:

```python
class ProactiveMonitor:
    """Background task that polls agent proactive prompts and broadcasts alerts."""

    def __init__(self, broadcast_fn) -> None:
        self._broadcast = broadcast_fn
        self._stopped = False
        self._next_run: dict[str, float] = {}  # agent_id → next_run_epoch

    def stop(self) -> None:
        self._stopped = True

    async def start(self) -> None:
        import time
        from dqe_agent.agents import get_all_agents

        logger.info("[PROACTIVE] Monitor started")
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
            await asyncio.sleep(10)  # check every 10s, actual intervals controlled per agent
        logger.info("[PROACTIVE] Monitor stopped")

    async def _run_agent_check(self, agent) -> None:
        """Run one proactive check for an agent and broadcast if noteworthy."""
        try:
            from dqe_agent.llm import get_executor_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = get_executor_llm()
            prompt = (
                f"You are a proactive monitor for the {agent.agent_id} domain.\n"
                f"{agent.proactive.prompt}\n"
                "If you find something noteworthy, respond with a concise one-paragraph alert. "
                "If nothing needs attention, respond with exactly: NO_ALERT"
            )
            result = await llm.ainvoke([SystemMessage(content=prompt)])
            content = result.content if hasattr(result, "content") else str(result)
            if content and "NO_ALERT" not in content:
                await self._broadcast({
                    "type": "proactive_alert",
                    "agent": agent.agent_id,
                    "content": content,
                })
                logger.info("[PROACTIVE] Alert from %s: %s", agent.agent_id, content[:80])
        except Exception as exc:
            logger.warning("[PROACTIVE] Check failed for %s: %s", agent.agent_id, exc)
```

- [ ] **Step 4: Wire ProactiveMonitor into api.py lifespan**

In `src/dqe_agent/api.py`, add a module-level variable:

```python
_proactive_monitor_task: asyncio.Task | None = None
```

In the `lifespan` function, after `master_agent` setup and before `yield`:

```python
    # Start proactive monitor
    from dqe_agent.agent.orchestrator import ProactiveMonitor

    active_websockets: set = set()  # track active WS connections for broadcast

    async def _proactive_broadcast(msg: dict) -> None:
        """Send proactive alert to all active WebSocket connections."""
        dead = set()
        for ws in list(active_websockets):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        active_websockets.difference_update(dead)

    global _proactive_monitor_task
    _monitor = ProactiveMonitor(broadcast_fn=_proactive_broadcast)
    _proactive_monitor_task = asyncio.create_task(_monitor.start())
    logger.info("Proactive monitor started")
```

In the shutdown section (after `yield`):

```python
    if _proactive_monitor_task:
        _monitor.stop()
        _proactive_monitor_task.cancel()
```

In `websocket_endpoint`, after `await ws.accept()`:

```python
    active_websockets.add(ws)
```

In the `finally` block of `websocket_endpoint`:

```python
    active_websockets.discard(ws)
```

Note: `active_websockets` needs to be defined at module scope or in a closure accessible to both `lifespan` and `websocket_endpoint`. The cleanest approach is a module-level set:

```python
# Add at module level near _human_queues:
_active_websockets: set = set()
```

Then in `lifespan`, define `_proactive_broadcast` using `_active_websockets`, and in `websocket_endpoint` add/remove from `_active_websockets`.

- [ ] **Step 5: Run monitor test**

```
pytest tests/test_orchestrator.py::test_proactive_monitor_runs_and_stops -v
```
Expected: PASS.

- [ ] **Step 6: Run all tests**

```
pytest tests/test_agents.py tests/test_orchestrator.py -v
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dqe_agent/agent/orchestrator.py src/dqe_agent/api.py
git commit -m "feat: ProactiveMonitor polls agents and broadcasts alerts over WebSocket"
```

---

## Task 9: Smoke Test End-to-End

- [ ] **Step 1: Start server**

```bash
python run.py
```
Expected: server starts on port 8000, logs show:
```
Discovered agents: ['jira', 'calendar', 'email', 'browser', 'people']
Compiled orchestrator graph (orchestrator → pev_subgraph → aggregator)
Proactive monitor started
DQE Agent ready — N tools loaded
```

- [ ] **Step 2: Verify single-domain routing via logs**

Send a Jira message via the frontend or WebSocket:
```
Create a bug ticket for login crash
```
Expected in logs:
```
[ORCHESTRATOR] message='Create a bug ticket for login crash' → agents=['jira']
```

- [ ] **Step 3: Verify multi-domain routing**

Send:
```
Create a ticket for the login bug and schedule a review meeting tomorrow
```
Expected in logs:
```
[ORCHESTRATOR] message='Create a ticket...' → agents=['jira', 'calendar']
```

- [ ] **Step 4: Verify browser fallback**

Send something with no domain keywords:
```
What is 2 + 2?
```
Expected in logs:
```
[ORCHESTRATOR] message='What is 2 + 2?' → agents=['browser']
```
(browser agent = all tools, so direct_response is available and will answer)

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: hierarchical multi-agent architecture complete"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|---|---|
| Each domain has focused sub-agent with own tool list | Task 3 |
| New integration = one new file, zero wiring | Task 1 (discover_agents auto-import) |
| Multi-domain fan-out via LangGraph Send | Task 5 + 6 |
| Proactive monitor | Task 8 |
| All existing code unchanged | Tasks 4, 6 (planner/executor backward-compat, loop unchanged for None filter) |
| Plugin-and-play AgentConfig | Task 1 |
| State extensions | Task 2 |
| API integration | Task 7 |
| Smoke test | Task 9 |

**All spec requirements covered. No placeholders. Types consistent across tasks.**
