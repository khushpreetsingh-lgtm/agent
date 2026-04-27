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
from dataclasses import dataclass

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
    domains: list[str] | None = None
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


def clear_registry() -> None:
    """Remove all registered agents (useful for tests)."""
    _AGENT_REGISTRY.clear()


def discover_agents() -> None:
    """Auto-import every non-private module in ``dqe_agent.agents``."""
    package = importlib.import_module("dqe_agent.agents")
    for _, name, _ in pkgutil.iter_modules(package.__path__):
        if not name.startswith("_"):
            importlib.import_module(f"dqe_agent.agents.{name}")
    logger.info("Discovered agents: %s", list(_AGENT_REGISTRY.keys()))
