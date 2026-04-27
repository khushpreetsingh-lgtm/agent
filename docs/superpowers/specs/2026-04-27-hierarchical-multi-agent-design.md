# Hierarchical Multi-Agent Architecture

**Date:** 2026-04-27
**Status:** Approved

---

## Goal

Evolve DQE Agent from a single PEV loop handling all tools into a hierarchical multi-agent system where:

- Each domain (Jira, Calendar, Email, Browser, People, etc.) has a focused sub-agent with its own tool list and prompt
- A new integration = one new file, zero wiring changes
- Multi-domain requests fan out to parallel sub-agents via LangGraph `Send`
- A proactive monitor watches connected systems and surfaces alerts unprompted
- All existing code (PEV loop, tools registry, flows, MCP) continues working unchanged

---

## Architecture

### Request Flow

```
User message (WebSocket)
        ↓
   orchestrator_node  (cheap model, runs once)
   ├── Classifies domains touched by the message
   ├── Decomposes into per-agent tasks
   └── Fan-out via LangGraph Send API
        ├── [parallel] Send("pev_subgraph", jira_state)
        ├── [parallel] Send("pev_subgraph", calendar_state)
        └── [parallel] Send("pev_subgraph", email_state)
                ↓ (all complete)
        aggregator_node → single response to user

Proactive Monitor (asyncio background task)
   ├── Polls each agent's configured endpoints on schedule
   └── Injects findings as synthetic messages → Orchestrator → user alert
```

### LangGraph Graph Structure

```
START → orchestrator_node
             ↓  (conditional: single-agent shortcut OR fan-out)
        [Send x N] → pev_subgraph (existing, reused unchanged)
             ↓
        aggregator_node → END
```

- Single-domain message: skip fan-out, call sub-agent PEV directly (no orchestrator overhead)
- Multi-domain message: `Send` API fans out in parallel, aggregator merges results
- Same compiled PEV graph reused for all sub-agents — `agent_id` in state selects tool filter

---

## Plugin System: AgentConfig

New package `src/dqe_agent/agents/` mirrors exact pattern of `tools/` and `flows/`.

### Base Class

```python
# src/dqe_agent/agents/__init__.py

class ProactiveConfig:
    interval_seconds: int        # how often to poll
    prompt: str                  # what to check for

class AgentConfig(ABC):
    agent_id: str                # unique identifier
    description: str             # shown in UI / logs
    domains: list[str]           # intent keywords for routing (e.g. ["jira", "ticket", "sprint"])
    tools: list[str] | None      # tool allowlist — None = all tools
    system_prompt: str           # injected into PEV planner for this agent
    proactive: ProactiveConfig | None = None  # opt-in proactive monitoring
```

### Adding a New Agent (complete example)

```python
# src/dqe_agent/agents/slack_agent.py  ← drop this file, done

from dqe_agent.agents import AgentConfig, register_agent

class SlackAgent(AgentConfig):
    agent_id = "slack"
    description = "Slack messaging — send, read, notify"
    domains = ["slack", "message", "channel", "notify", "dm"]
    tools = ["slack_send_message", "slack_read_channel", "slack_list_channels"]
    system_prompt = "You are a Slack agent. Send messages, read channels, manage notifications."
    proactive = ProactiveConfig(
        interval_seconds=120,
        prompt="Check for unread DMs or @mentions that need a response"
    )

register_agent(SlackAgent())
```

`discover_agents()` auto-imports all files in `agents/` on startup — same mechanism as `discover_tools()` and `discover_flows()`. No registration in any other file.

---

## Orchestrator Node

Runs on every user message. Uses cheap/fast model (same as executor).

### Routing Logic

1. Scan all registered `AgentConfig.domains` at startup → build `{keyword: agent_id}` lookup
2. On each message: score message against keyword lookup → identify domains touched
3. If one domain: skip fan-out, invoke that agent's PEV directly
4. If multiple domains: decompose into per-agent tasks, `Send` in parallel
5. If no domain matches: fallback to `BrowserAgent` (handles anything via browser)

### Orchestrator Output (JSON)

```json
{
  "parallel": true,
  "tasks": [
    {"agent": "jira",     "task": "Create bug ticket: login crash on mobile"},
    {"agent": "calendar", "task": "Schedule 30min standup tomorrow at 10am"}
  ]
}
```

---

## State Changes

Minimal addition to existing `AgentState`:

```python
class AgentState(TypedDict, total=False):
    # ... all existing fields unchanged ...
    agent_id: Annotated[str, _replace]        # which sub-agent is running this PEV
    orchestrator_tasks: Annotated[list, _replace]  # decomposed task list from orchestrator
    sub_results: Annotated[list, _replace]    # collected results from parallel sub-agents
```

---

## Proactive Monitor

Single `asyncio.Task` started in `api.py` lifespan. Iterates registered agents with `proactive` config set.

- Runs each agent's proactive prompt on its configured interval
- On finding something noteworthy: pushes `{"type": "proactive_alert", "agent": "jira", "content": "..."}` to all active WebSocket sessions
- Disabled by default (`proactive = None`). Opt-in per agent file.
- No new infrastructure — uses existing WebSocket broadcast mechanism in `api.py`

---

## Built-in Agents (Initial Set)

These replace the current single flat tool list. All tools already exist — just grouped.

| Agent | `agent_id` | Domains | Tools |
|---|---|---|---|
| Jira | `jira` | jira, ticket, issue, sprint, board, epic, story, bug, task | All `jira_*` MCP tools |
| Calendar | `calendar` | calendar, meeting, schedule, event, invite, availability | All `get_events`, `create_event`, etc. |
| Email | `email` | email, mail, send, inbox, reply, draft | All `send_email`, `read_email`, etc. |
| Browser | `browser` | browser, web, navigate, login, click, fill, scrape | All `browser_*` tools |
| People | `people` | people, user, member, assignee, team, org, contact | `jira_get_assignable_users`, `jira_search_user_by_email`, people MCP tools |

---

## What Changes vs What Stays

| Component | Change |
|---|---|
| `agent/planner.py` | Unchanged |
| `agent/executor.py` | Unchanged |
| `agent/verifier.py` | Unchanged |
| `agent/loop.py` | Minor: accept `tool_filter` param, pass to planner |
| `agent/master.py` | Minor: expose PEV as sub-graph callable |
| `tools/` registry | Unchanged |
| `flows/` registry | Unchanged — existing flows continue working |
| `state.py` | Add 3 fields: `agent_id`, `orchestrator_tasks`, `sub_results` |
| `api.py` | Route through Orchestrator node; add proactive monitor task in lifespan |
| `agents/` | **New package** — `AgentConfig` base, `discover_agents()`, built-in agents |
| `agent/orchestrator.py` | **New file** — `orchestrator_node` + `aggregator_node` LangGraph nodes |

No existing code deleted. No new dependencies. Every existing flow, tool, and MCP integration works on day one.

---

## File Layout (new files only)

```
src/dqe_agent/
  agents/
    __init__.py          # AgentConfig base, register_agent, discover_agents
    jira_agent.py        # JiraAgent
    calendar_agent.py    # CalendarAgent
    email_agent.py       # EmailAgent
    browser_agent.py     # BrowserAgent (fallback)
    people_agent.py      # PeopleAgent
  agent/
    orchestrator.py      # orchestrator_node, aggregator_node, build_orchestrator_graph
```

---

## Success Criteria

- Adding a new integration requires only one new file in `agents/`
- Multi-domain requests ("create Jira ticket AND schedule meeting") execute in parallel
- Single-domain requests have no orchestrator overhead
- All existing flows (opportunity_to_quote, default) continue working unchanged
- Proactive alerts appear in frontend without user prompting
- No new pip dependencies introduced
