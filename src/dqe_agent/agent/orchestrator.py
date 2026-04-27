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
    builder.add_node("pev_subgraph", pev.compile())

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
