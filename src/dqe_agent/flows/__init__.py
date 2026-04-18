"""Flow registry — plugin system for adding new conversational agent flows.

Each flow defines a system prompt and state-aware reminder logic that guides
the LLM through a specific workflow (e.g., opportunity-to-quote, customer form,
service activation).  Mirrors the tool registry pattern in ``dqe_agent.tools``.

Adding a new flow
-----------------
1. Create ``src/dqe_agent/flows/my_flow.py``
2. Subclass ``FlowConfig`` and set ``flow_id``, ``system_prompt``, and
   ``select_reminder()``.
3. Optionally set ``description`` (shown in the UI) and ``tools`` (allowlist of
   tool names the LLM may call — ``None`` means *all* tools).
4. Call ``register_flow(MyFlow())`` at module level.
5. ``discover_flows()`` auto-imports all modules in this package on startup —
   no manual imports required.

Example skeleton::

    from dqe_agent.flows import FlowConfig, register_flow

    class CustomerFormFlow(FlowConfig):
        flow_id = "customer_form"
        description = "Collect customer details and submit an opportunity form"

        # Only expose the tools this flow actually needs
        tools = ["browser_login", "browser_navigate", "browser_fill_form",
                 "ask_user", "human_review", "send_email"]

        system_prompt = \"\"\"
        You are a data collection agent...
        \"\"\"

        def select_reminder(self, messages, all_tool_contents, last_tool_content):
            # Return a hint string to steer the LLM, or None
            return None

    register_flow(CustomerFormFlow())
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Sequence

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Global registry: flow_id -> FlowConfig instance
_FLOW_REGISTRY: dict[str, FlowConfig] = {}


class FlowConfig(ABC):
    """Base class for flow-specific conversational agent configuration.

    Subclass this and override ``flow_id``, ``system_prompt``, and
    ``select_reminder()``.  The optional attributes below let you add
    metadata and constrain which tools the LLM can call.
    """

    # --- Required -----------------------------------------------------------

    flow_id: str
    """Unique identifier for this flow (e.g. ``"opportunity_to_quote"``)."""

    system_prompt: str
    """Full system prompt injected at the start of every conversation."""

    # --- Optional metadata --------------------------------------------------

    description: str = ""
    """Human-readable one-line description shown in the flows API endpoint."""

    tools: list[str] | None = None
    """Allowlist of tool names available to the LLM in this flow.

    ``None`` (default) means *all* registered tools are available.
    Set an explicit list to restrict the LLM to only the tools your flow
    needs — this makes prompt engineering easier and prevents the agent
    from accidentally calling unrelated tools.

    Example::

        tools = ["browser_login", "browser_navigate", "browser_fill_form",
                 "ask_user", "human_review"]
    """

    # --- Internal reminder machinery ----------------------------------------

    force_tool_reminder: str = (
        "[STOP — do not respond with text. The workflow is NOT complete. "
        "Read the 'next_required_step' field in the last tool result and call that tool NOW. "
        "Your response must be a tool call, not a sentence.]"
    )

    @abstractmethod
    def select_reminder(
        self,
        messages: Sequence[BaseMessage],
        all_tool_contents: list[str],
        last_tool_content: str,
    ) -> str | None:
        """Return a reminder string to inject before the next LLM call, or None.

        Called before every LLM invocation so the flow can nudge the agent
        toward the correct next action based on conversation history.

        Args:
            messages: Full message history (including system messages).
            all_tool_contents: Serialised content of every ToolMessage so far.
            last_tool_content: Content of the most recent ToolMessage (empty
                string if no tools have been called yet).

        Returns:
            A string injected as a ``SystemMessage`` immediately before the LLM
            call, or ``None`` to inject nothing.
        """
        ...

    def should_force_tool_call(self, last_tool_content: str) -> bool:
        """Return True if the LLM should be forced to make a tool call.

        Override in subclasses for flow-specific pause logic.
        Default: force when ``next_required_step`` is present and the agent
        is NOT at a user-facing pause point.
        """
        if "next_required_step" not in last_tool_content:
            return False
        pause_keywords = (
            "awaiting_confirmation",
            "awaiting_email_review",
            "email_sent_via_cpq",
        )
        return not any(kw in last_tool_content for kw in pause_keywords)


# -- Registry helpers --------------------------------------------------------

def register_flow(flow: FlowConfig) -> None:
    _FLOW_REGISTRY[flow.flow_id] = flow
    logger.info("Registered flow: %s", flow.flow_id)


def get_flow(flow_id: str) -> FlowConfig:
    if flow_id not in _FLOW_REGISTRY:
        raise KeyError(f"Flow '{flow_id}' not found. Available: {list(_FLOW_REGISTRY.keys())}")
    return _FLOW_REGISTRY[flow_id]


def get_default_flow() -> FlowConfig:
    return _FLOW_REGISTRY["default"]


def list_flows() -> list[str]:
    return list(_FLOW_REGISTRY.keys())


def list_flow_configs() -> list[dict]:
    """Return metadata for all registered flows (used by the /api/v1/flows endpoint)."""
    return [
        {
            "flow_id": f.flow_id,
            "description": f.description,
            "tools": f.tools,  # None means all tools
        }
        for f in _FLOW_REGISTRY.values()
    ]


def discover_flows() -> None:
    """Auto-import every module in ``dqe_agent.flows`` to trigger registration."""
    package = importlib.import_module("dqe_agent.flows")
    for _, name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"dqe_agent.flows.{name}")
    logger.info("Discovered flows: %s", list(_FLOW_REGISTRY.keys()))
