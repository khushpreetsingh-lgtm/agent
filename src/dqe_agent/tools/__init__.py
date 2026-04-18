"""Tool registry — a plugin system for adding new capabilities to the DQE Agent.

Any function decorated with `@register_tool` becomes available to the LangGraph
agent at runtime. Tools can be:
  1. Simple Python functions (sync or async)
  2. LangChain BaseTool instances
  3. MCP server tools (loaded dynamically)

Usage:
    from dqe_agent.tools import register_tool, get_tool, list_tools

    @register_tool(
        name="lookup_crm",
        description="Look up a customer in the CRM by account ID",
    )
    async def lookup_crm(account_id: str) -> dict:
        ...

    # Later, anywhere in the codebase:
    tool = get_tool("lookup_crm")
    result = await tool.ainvoke({"account_id": "123"})

Adding a new tool module
------------------------
1. Create ``src/dqe_agent/tools/my_tool.py``
2. Decorate your functions with ``@register_tool(...)``
3. ``discover_tools()`` auto-imports all modules in this package —
   no manual imports required (mirrors the flow discovery pattern).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Optional

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

# Global registry: name -> LangChain BaseTool
_TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(
    name: str,
    description: str = "",
    *,
    return_direct: bool = False,
) -> Callable:
    """Decorator that registers a Python function as a LangChain-compatible tool.

    Works with both sync and async functions. The function's type hints are used
    to build the tool's input schema automatically.

    Example:
        @register_tool("my_tool", "Does something useful")
        async def my_tool(query: str) -> str:
            return "result"
    """

    def decorator(fn: Callable) -> Callable:
        tool = StructuredTool.from_function(
            func=fn if not inspect.iscoroutinefunction(fn) else None,
            coroutine=fn if inspect.iscoroutinefunction(fn) else None,
            name=name,
            description=description or fn.__doc__ or f"Tool: {name}",
            return_direct=return_direct,
        )
        _TOOL_REGISTRY[name] = tool
        logger.info("Registered tool: %s", name)
        return fn

    return decorator


def register_langchain_tool(tool: BaseTool) -> None:
    """Register an existing LangChain BaseTool instance directly."""
    _TOOL_REGISTRY[tool.name] = tool
    logger.info("Registered LangChain tool: %s", tool.name)


def get_tool(name: str) -> BaseTool:
    """Retrieve a registered tool by name. Raises KeyError if not found."""
    if name not in _TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{name}' not found. Available: {list(_TOOL_REGISTRY.keys())}"
        )
    return _TOOL_REGISTRY[name]


def list_tools(names: list[str] | None = None) -> list[BaseTool]:
    """Return registered tools.

    Parameters
    ----------
    names:
        Optional allowlist of tool names. When provided only those tools are
        returned (unknown names are silently skipped).  When ``None`` (default)
        every registered tool is returned.
    """
    if names is None:
        return list(_TOOL_REGISTRY.values())
    return [_TOOL_REGISTRY[n] for n in names if n in _TOOL_REGISTRY]


def list_tool_names() -> list[str]:
    """Return names of all registered tools."""
    return list(_TOOL_REGISTRY.keys())


def clear_registry() -> None:
    """Remove all registered tools (useful for tests)."""
    _TOOL_REGISTRY.clear()


def discover_tools() -> None:
    """Auto-import every non-private module in ``dqe_agent.tools`` to trigger registration.

    Mirrors ``discover_flows()`` — drop a new file in the tools/ package and
    its ``@register_tool`` decorators will fire automatically on startup.
    Private modules (names starting with ``_``) are skipped.
    """
    package = importlib.import_module("dqe_agent.tools")
    for _, name, _ in pkgutil.iter_modules(package.__path__):
        if not name.startswith("_"):
            importlib.import_module(f"dqe_agent.tools.{name}")
    logger.info("Discovered tools: %s", list_tool_names())
