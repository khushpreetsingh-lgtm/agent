"""Workflow engine — loads YAML workflow definitions and builds LangGraph instances.

This is the STABLE CORE. You never edit this file.
New workflows = new YAML file in workflows/
New capabilities = new tool file in tools/

Architecture:
  1. YAML defines step order (deterministic, guaranteed)
  2. Each step names a tool from the tool registry
  3. This engine builds a LangGraph that executes steps in order
  4. Human review steps use LangGraph's interrupt() for guaranteed pauses
  5. Conditional steps are evaluated against the current state
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from dqe_agent.state import AgentState
from dqe_agent.tools import get_tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Workflow model (parsed from YAML)
# ─────────────────────────────────────────────────────────
class WorkflowStep:
    """One step in a workflow."""

    def __init__(self, raw: dict) -> None:
        self.id: str = raw["id"]
        self.tool: str = raw["tool"]
        self.description: str = raw.get("description", "")
        self.params: dict = raw.get("params", {})
        self.outputs: list[str] = raw.get("outputs", [])
        self.condition: str | None = raw.get("condition")


class WorkflowDefinition:
    """Parsed workflow from a YAML file."""

    def __init__(self, raw: dict) -> None:
        self.name: str = raw["name"]
        self.description: str = raw.get("description", "")
        self.inputs: dict = raw.get("inputs", {})
        self.steps: list[WorkflowStep] = [WorkflowStep(s) for s in raw["steps"]]


def load_workflow(path: str | Path) -> WorkflowDefinition:
    """Load a workflow definition from a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    wf = WorkflowDefinition(raw)
    logger.info("Loaded workflow '%s' with %d steps from %s", wf.name, len(wf.steps), path.name)
    return wf


def list_workflows(workflows_dir: str | Path = "workflows") -> list[Path]:
    """List all .yaml workflow files in the workflows directory."""
    d = Path(workflows_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.yaml"))


# ─────────────────────────────────────────────────────────
# Template resolution — {{ variable }} and ${ENV_VAR} in params
# ─────────────────────────────────────────────────────────
_TEMPLATE_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_value(template: str, state: dict) -> str:
    """Resolve template references in a string value.

    Two syntaxes supported:
      ${ENV_VAR}      — substituted from environment variables / settings
      {{ state.var }} — substituted from current workflow state
    """
    if not isinstance(template, str):
        return template

    # 1. ${ENV_VAR} — environment / settings values
    #    Lets workflow YAML reference credentials without hardcoding them:
    #    instruction: "Go to ${NETSUITE_URL} and log in as ${NETSUITE_USERNAME}"
    def _env_sub(m: re.Match) -> str:
        var = m.group(1)
        # Try OS env first, then pydantic settings as fallback
        val = os.environ.get(var)
        if val is None:
            try:
                from dqe_agent.config import settings
                val = getattr(settings, var.lower(), None)
            except Exception:
                pass
        return str(val) if val is not None else ""

    template = _ENV_RE.sub(_env_sub, template)

    # 2. {{ state.path }} — values from workflow state / previous step outputs
    def _state_sub(match: re.Match) -> str:
        expr = match.group(1).strip()
        parts = expr.split(".")
        val: Any = state
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part, "")
            elif hasattr(val, part):
                val = getattr(val, part, "")
            else:
                val = ""
                break
        # Sanitize numeric totals from external tools: negative totals (e.g. -1)
        # often indicate an error or unknown count. Treat them as zero so
        # user-facing messages don't show "-1 open issues".
        try:
            if isinstance(val, int) and val < 0:
                val = 0
            # Also handle stringified integers
            if isinstance(val, str) and val.strip().lstrip("+-").isdigit():
                n = int(val.strip())
                if n < 0:
                    val = 0
        except Exception:
            pass

        return str(val) if val is not None else ""

    return _TEMPLATE_RE.sub(_state_sub, template)


def _resolve_params(params: dict, state: dict) -> dict:
    """Deep-resolve all {{ }} in a params dict."""
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str):
            resolved[key] = _resolve_value(value, state)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_value(v, state) if isinstance(v, str) else v for v in value
            ]
        elif isinstance(value, dict):
            resolved[key] = _resolve_params(value, state)
        else:
            resolved[key] = value
    return resolved


# ─────────────────────────────────────────────────────────
# Condition evaluation
# ─────────────────────────────────────────────────────────
def _evaluate_condition(condition: str | None, state: dict) -> bool:
    """Evaluate a simple condition like 'email.approved == true' against state."""
    if not condition:
        return True

    cond = condition.strip()

    # Support explicit equality conditions: "field.path == value"
    match = re.match(r"(.+?)\s*==\s*(.+)", cond)
    if match:
        left_expr = match.group(1).strip()
        right_val = match.group(2).strip().lower()

        # Resolve left side
        parts = left_expr.split(".")
        val = state
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            elif hasattr(val, part):
                val = getattr(val, part, None)
            else:
                val = None
                break

        # Compare
        if right_val in ("true", "yes"):
            return bool(val)
        elif right_val in ("false", "no"):
            return not bool(val)
        else:
            return str(val).lower() == right_val

    # Support conditions like "{{check_busy}} has events" or "check_busy has events"
    m2 = re.match(r"^\s*\{?\{?\s*([^\}\s\}]+(?:\.[^\}\s\}]+)*)\s*\}?\}?\s+has\s+events\s*$", cond, re.IGNORECASE)
    if m2:
        ref = m2.group(1).strip()
        # Resolve the referenced value from state (support nested paths)
        parts = ref.split(".")
        val = state
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            elif hasattr(val, part):
                val = getattr(val, part, None)
            else:
                val = None
                break

        # If the tool returned a list of events, truthiness is straightforward
        if isinstance(val, list):
            return bool(val)

        # If it's a dict with an 'items'/'events' key, check that
        if isinstance(val, dict):
            for key in ("items", "events", "results"):
                if key in val and isinstance(val[key], list):
                    return bool(val[key])
            # Otherwise consider non-empty dict as truthy
            return bool(val)

        # Otherwise treat strings containing 'no events' / 'no events found' as empty
        if isinstance(val, str):
            low = val.strip().lower()
            if not low:
                return False
            if "no events" in low or "no events found" in low:
                return False
            return True

        return False

    # Unknown condition format — default to True to avoid silently skipping intended steps
    return True


# ─────────────────────────────────────────────────────────
# Output saver — dumps extracted data to output/ for inspection
# ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")


def _save_step_output(step_id: str, result: dict, state: dict) -> None:
    """Write step result + relevant state fields to output/<step_id>.json."""
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        payload = {
            "step": step_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "opportunity": _to_dict(state.get("opportunity")),
            "structure": _to_dict(state.get("structure")),
            "quote": _to_dict(state.get("quote")),
        }
        out_file = OUTPUT_DIR / f"{step_id}.json"
        out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.debug("[output] saved %s", out_file)
    except Exception as exc:
        logger.warning("Could not save step output: %s", exc)


def _to_dict(obj: Any) -> Any:
    """Convert a Pydantic model or plain value to a JSON-serialisable dict."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


# ─────────────────────────────────────────────────────────
# Node factory — creates a LangGraph node for each workflow step
# ─────────────────────────────────────────────────────────
def _make_step_node(step: WorkflowStep):
    """Create an async function that executes one workflow step as a LangGraph node."""

    async def _node(state: AgentState) -> dict:
        logger.info("[Step %s] %s (tool=%s)", step.id, step.description, step.tool)

        # Check condition
        if not _evaluate_condition(step.condition, state):
            logger.info("[Step %s] Condition not met, skipping.", step.id)
            return {
                "current_step": step.id,
                "messages": [AIMessage(content=f"Skipped {step.id}: condition not met.")],
            }

        # Resolve params
        resolved_params = _resolve_params(step.params, state)

        # Inject the full state context so tools can access extracted data
        resolved_params["__state__"] = state

        # Call the registered tool
        tool = get_tool(step.tool)
        result = await tool.ainvoke(resolved_params)

        # Build return dict
        update: dict[str, Any] = {
            "current_step": step.id,
            "messages": [AIMessage(content=f"[{step.id}] {step.description} — done.")],
        }

        # Store tool outputs in state
        if isinstance(result, dict):
            for output_key in step.outputs:
                if output_key in result:
                    update[output_key] = result[output_key]
            # Also store in tool_results for generic access
            update["tool_results"] = {
                **state.get("tool_results", {}),
                step.id: result,
            }

            # Save extracted data to file for inspection
            _save_step_output(step.id, result, state)

        return update

    _node.__name__ = f"step_{step.id}"
    _node.__qualname__ = f"step_{step.id}"
    return _node


# ─────────────────────────────────────────────────────────
# Graph builder — YAML → LangGraph
# ─────────────────────────────────────────────────────────
def build_graph_from_workflow(workflow: WorkflowDefinition) -> StateGraph:
    """Build a LangGraph StateGraph from a WorkflowDefinition.

    This is the STABLE function. It takes any YAML-defined workflow and turns it
    into an executable graph with deterministic step ordering.
    """
    graph = StateGraph(AgentState)

    step_ids = []
    for step in workflow.steps:
        node_fn = _make_step_node(step)
        graph.add_node(step.id, node_fn)
        step_ids.append(step.id)

    # Wire edges: START → step1 → step2 → ... → stepN → END
    if step_ids:
        graph.add_edge(START, step_ids[0])
        for i in range(len(step_ids) - 1):
            graph.add_edge(step_ids[i], step_ids[i + 1])
        graph.add_edge(step_ids[-1], END)

    logger.info(
        "Built graph for workflow '%s': %d steps, %s → %s",
        workflow.name,
        len(step_ids),
        step_ids[0] if step_ids else "∅",
        step_ids[-1] if step_ids else "∅",
    )
    return graph
