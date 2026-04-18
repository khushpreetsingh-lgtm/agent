"""MCP (Model Context Protocol) client — dynamically loads tools from MCP servers.

This integrates with LangChain's official MCP adapter so that any MCP-compliant
server (local stdio or remote SSE) can expose tools that the DQE Agent can call.

Usage:
    from dqe_agent.tools.mcp_loader import load_mcp_tools

    # Load from a local stdio MCP server
    tools = await load_mcp_tools(
        transport="stdio",
        command="npx",
        args=["-y", "@anthropic/mcp-server-filesystem", "/path"],
    )

    # Load from a remote SSE MCP server
    tools = await load_mcp_tools(
        transport="sse",
        url="http://localhost:8080/sse",
    )

All loaded tools are automatically registered in the tool registry.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def load_mcp_tools(
    transport: str = "stdio",
    *,
    # stdio params
    command: str = "",
    args: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
    # sse params
    url: str = "",
    headers: Optional[dict[str, str]] = None,
) -> list[BaseTool]:
    """Load tools from an MCP server and register them in the tool registry.

    Parameters
    ----------
    transport : "stdio" or "sse"
    command / args / env : for stdio transport
    url / headers : for SSE transport

    Returns
    -------
    List of LangChain BaseTool instances loaded from the MCP server.
    """
    from dqe_agent.tools import register_langchain_tool

    tools: list[BaseTool] = []

    try:
        if transport == "stdio":
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    "mcp_server": {
                        "command": command,
                        "args": args or [],
                        "env": env,
                        "transport": "stdio",
                    }
                }
            )
            async with client:
                tools = client.get_tools()

        elif transport == "sse":
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    "mcp_server": {
                        "url": url,
                        "headers": headers or {},
                        "transport": "sse",
                    }
                }
            )
            async with client:
                tools = client.get_tools()

        else:
            raise ValueError(f"Unknown MCP transport: {transport}")

        # Register each loaded tool in our global registry
        for tool in tools:
            register_langchain_tool(tool)
            logger.info("Loaded MCP tool: %s — %s", tool.name, tool.description[:80])

    except ImportError:
        logger.warning(
            "langchain-mcp-adapters not installed. "
            "Run: pip install langchain-mcp-adapters"
        )
    except Exception:
        logger.exception("Failed to load MCP tools from %s", transport)

    return tools
