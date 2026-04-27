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


async def classify_domains(
    message: str,
    conversation_history: list,
    agent_descriptions: dict[str, str],
) -> list[str]:
    """LLM-based domain classification using conversation context."""
    from dqe_agent.llm import get_planner_llm

    # Format last 3 messages for context
    history_text = ""
    human_msgs = [m for m in conversation_history if isinstance(m, HumanMessage)]
    if len(human_msgs) > 1:
        recent = human_msgs[-3:]
        history_text = "\n".join([f"User: {m.content}" for m in recent])

    # Build agent descriptions
    agent_list = "\n".join([f"- {aid}: {desc}" for aid, desc in agent_descriptions.items()])

    prompt = f"""Classify which specialist agents are needed for this user message.

Available agents:
{agent_list}

Recent conversation context:
{history_text if history_text else "(no prior context)"}

New user message: "{message}"

Instructions:
- Output agent IDs that match the user's intent
- Consider conversation context (e.g., "show more" continues previous agent)
- If message spans multiple domains, output multiple agents
- If no domain matches or very ambiguous, output ["browser"]
- Output ONLY valid JSON array of agent IDs

Output JSON: {{"agents": ["agent_id", ...]}}"""

    llm = get_planner_llm()
    try:
        result = await llm.ainvoke([SystemMessage(content=prompt)])
        content = result.content if hasattr(result, "content") else str(result)

        # Strip markdown if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)
        agents = parsed.get("agents", ["browser"])

        # Validate agent IDs
        valid_agents = [a for a in agents if a in agent_descriptions]
        if not valid_agents:
            logger.warning("[ORCHESTRATOR] LLM returned invalid agents %s, fallback to browser", agents)
            return ["browser"]

        return valid_agents
    except Exception as exc:
        logger.warning("[ORCHESTRATOR] LLM classification failed: %s, fallback to browser", exc)
        return ["browser"]


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
    from dqe_agent.agents import get_agent, get_all_agents

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

    # Build agent descriptions for LLM
    agent_descriptions = {
        agent.agent_id: agent.description or f"Handles: {', '.join((agent.domains or [])[:5])}"
        for agent in get_all_agents()
    }

    agent_ids = await classify_domains(message_text, messages, agent_descriptions)

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
            from langchain_core.messages import SystemMessage

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
