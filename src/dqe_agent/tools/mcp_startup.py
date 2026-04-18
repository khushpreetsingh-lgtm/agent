"""MCP Server Startup Helper — manages a single persistent MultiServerMCPClient.

MCP stdio tools run as subprocesses.  The MultiServerMCPClient must remain open
for the entire app lifetime — closing it kills the subprocesses and makes every
subsequent tool call fail.  This module owns the single shared client instance.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_mcp_client: Any = None  # holds the open MultiServerMCPClient between start/stop


async def load_mcp_servers(config_path: str = "mcp_config.yaml") -> int:
    """Open one MultiServerMCPClient for ALL enabled servers and register their tools.

    The client is kept alive (stored in _mcp_client) until stop_mcp_servers() is called.
    Returns the number of tools registered.
    """
    global _mcp_client

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed — MCP tools unavailable. "
                       "Run: pip install langchain-mcp-adapters")
        return 0

    try:
        from dqe_agent.tools.mcp_config_loader import load_mcp_config
        server_configs = load_mcp_config(config_path)
    except Exception as exc:
        logger.warning("Failed to read MCP config: %s", exc)
        return 0

    if not server_configs:
        logger.info("No MCP servers configured")
        return 0

    # Build the dict MultiServerMCPClient expects: {name: {transport, command, ...}}
    # We derive a unique name per server from its command + args.
    servers_dict: dict[str, dict] = {}
    for i, cfg in enumerate(server_configs):
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            import os as _os
            cmd = cfg.get("command", "")
            args = cfg.get("args", [])
            # Use last part of command + first arg as key, e.g. "uvx_mcp-atlassian"
            name = f"{cmd}_{args[0]}" if args else cmd
            name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            name = f"{name}_{i}" if name in servers_dict else name
            entry: dict = {"transport": "stdio", "command": cmd, "args": args}
            env = cfg.get("env")
            if env:
                # Merge with current process env so PATH/CONDA/system vars are preserved.
                # Passing env alone would replace the whole environment, breaking conda/uvx.
                entry["env"] = {**_os.environ, **env}
            servers_dict[name] = entry
        elif transport == "sse":
            url = cfg.get("url", "")
            name = f"sse_{i}"
            entry = {"transport": "sse", "url": url}
            hdr = cfg.get("headers")
            if hdr:
                entry["headers"] = hdr
            servers_dict[name] = entry

    if not servers_dict:
        logger.info("No usable MCP server definitions found")
        return 0

    logger.info("Starting %d MCP server(s): %s", len(servers_dict), list(servers_dict.keys()))

    try:
        # langchain-mcp-adapters >= 0.1.0: no context manager, just await get_tools()
        client = MultiServerMCPClient(servers_dict)
        tools = await client.get_tools()
        _mcp_client = client  # keep reference so tools stay callable

        from dqe_agent.tools import register_langchain_tool
        for tool in tools:
            register_langchain_tool(tool)
            logger.info("  MCP tool registered: %s", tool.name)

        logger.info("MCP ready — %d tool(s) loaded", len(tools))
        return len(tools)

    except Exception as exc:
        logger.error("Failed to start MCP servers: %s", exc, exc_info=True)
        _mcp_client = None
        return 0


async def stop_mcp_servers() -> None:
    """Release the MCP client reference (subprocesses exit naturally)."""
    global _mcp_client
    if _mcp_client is not None:
        _mcp_client = None
        logger.info("MCP servers stopped")
