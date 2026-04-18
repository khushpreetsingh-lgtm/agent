"""LangGraph PEV loop — Planner → Executor → Verifier with retry/replan.

This is the SINGLE unified graph for ALL interactions — both chat and browser tasks.
The planner decides whether a message needs browser actions or a simple direct_response.

Graph:
  START → planner → executor → verifier ─┬─→ executor (next step / retry)
                                          ├─→ planner (replan)
                                          └─→ END (done / failed / max_steps)
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from dqe_agent.agent.executor import executor_node
from dqe_agent.agent.planner import planner_node
from dqe_agent.agent.verifier import verifier_node
from dqe_agent.state import AgentState

logger = logging.getLogger(__name__)


def route_after_verify(state: AgentState) -> str:
    """Route after verification: continue, retry, replan, or end."""
    status = state.get("status", "")
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    if status == "failed":
        return "end"

    if status == "planning":
        return "replan"

    # Check if there are more steps
    if idx >= len(plan):
        return "done"

    return "next"


def route_start(state: AgentState) -> str:
    """Initial route — if a plan already exists, skip to executor."""
    plan = state.get("plan", [])
    if plan and len(plan) > 0:
        return "executor"
    return "planner"


def build_pev_graph() -> StateGraph:
    """Build the Planner-Executor-Verifier LangGraph."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("verifier", verifier_node)

    # Entry: plan first (or skip if plan exists)
    builder.add_conditional_edges(
        START,
        route_start,
        {"planner": "planner", "executor": "executor"},
    )

    # planner → executor
    builder.add_edge("planner", "executor")

    # executor → verifier
    builder.add_edge("executor", "verifier")

    # verifier → conditional routing
    builder.add_conditional_edges(
        "verifier",
        route_after_verify,
        {
            "next": "executor",    # next step
            "replan": "planner",   # full replan
            "done": END,           # all steps complete
            "end": END,            # failed
        },
    )

    return builder
