"""Unified AgentState — single state definition for ALL modes (chat + PEV workflow).

This is the ONLY state definition in the project. The old agent/state.py
imports from here for backward compatibility.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _replace(existing: Any, new: Any) -> Any:
    """Reducer: overwrite old value with new."""
    return new


def _merge_dicts(existing: Any, new: Any) -> Any:
    """Reducer: shallow-merge two dicts."""
    if not existing:
        return new or {}
    if not new:
        return existing
    return {**existing, **new}


def _append_list(existing: list | None, new: list | None) -> list:
    """Reducer: append lists."""
    return (existing or []) + (new or [])


class StepPlan(TypedDict, total=False):
    """One step in the planner's output."""
    id: str
    type: str            # navigate, click, type, extract, login, wait, verify, custom
    description: str     # human-readable description
    tool: str            # tool name to call (browser_navigate, browser_act, etc.)
    params: dict         # params to pass to the tool
    success_criteria: str  # how to verify this step succeeded
    depends_on: list[str]  # step IDs this depends on


class StepResult(TypedDict, total=False):
    """Result of executing one step."""
    step_id: str
    step_index: int
    tool: str
    status: str          # success, failed, retried, skipped
    result: Any
    error: str
    duration_ms: float
    retries: int


class AgentState(TypedDict, total=False):
    """Full state for both ReAct (chat) and PEV (workflow) modes.

    ReAct mode uses: messages, browser_ready, current_task, error, flow_data
    PEV mode uses: all fields
    """

    # ── Chat history ─────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Session ──────────────────────────────────────────────────────────
    session_id: Annotated[str, _replace]
    task: Annotated[str, _replace]

    # ── Plan (set by Planner, read by Executor) ──────────────────────────
    plan: Annotated[list[dict], _replace]
    current_step_index: Annotated[int, _replace]

    # ── Execution tracking ───────────────────────────────────────────────
    # _replace: executor accumulates manually; planner resets to [] on new task
    step_results: Annotated[list, _replace]
    retry_count: Annotated[int, _replace]
    replan_count: Annotated[int, _replace]

    # ── Guardrails ───────────────────────────────────────────────────────
    steps_taken: Annotated[int, _replace]
    estimated_cost: Annotated[float, _replace]
    start_time: Annotated[float, _replace]

    # ── Generic data store (per-flow) ────────────────────────────────────
    # _replace: executor accumulates manually; planner resets to {} on new task
    flow_data: Annotated[dict, _replace]

    # ── Control ──────────────────────────────────────────────────────────
    status: Annotated[str, _replace]   # idle, planning, executing, verifying, complete, failed, awaiting_human
    error: Annotated[str | None, _replace]

    # ── Browser (legacy / ReAct compat) ──────────────────────────────────
    browser_ready: Annotated[bool, _replace]
    current_task: Annotated[str | None, _replace]
