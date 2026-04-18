"""MCP smoke test: start MCP servers and call a couple of safe read-only tools.

Usage: python scripts/mcp_smoke_test.py

This script will:
 - call dqe_agent.tools.mcp_startup.load_mcp_servers()
 - attempt to invoke `jira_get_all_projects` and `list_calendars` if registered
 - print short diagnostics and outputs
"""
import asyncio
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

async def main():
    from dqe_agent.tools import list_tool_names, get_tool
    from dqe_agent.tools.mcp_startup import load_mcp_servers

    print("Starting MCP servers (may spawn subprocesses)...")
    n = await load_mcp_servers()
    print(f"MCP servers started, {n} tools loaded (approx). Registered tools: {len(list_tool_names())}")

    async def try_tool(name, params=None):
        params = params or {}
        try:
            tool = get_tool(name)
        except Exception as exc:
            print(f"Tool '{name}' not available: {exc}")
            return
        print(f"Invoking tool: {name} with params: {params}")
        try:
            res = await tool.ainvoke(params)
            try:
                # attempt to pretty-print JSON-like results
                print(name, json.dumps(res, indent=2, default=str))
            except Exception:
                print(name, str(res))
        except Exception as exc:
            print(f"Tool '{name}' invocation failed: {exc}")

    # Jira smoke: list projects
    await try_tool("jira_get_all_projects")

    # Google Calendar smoke: list calendars
    # Pass USER_GOOGLE_EMAIL from environment if available (from .env)
    user_email = os.environ.get("USER_GOOGLE_EMAIL")
    if user_email:
        await try_tool("list_calendars", {"user_google_email": user_email})
    else:
        print("Skipping 'list_calendars' — set USER_GOOGLE_EMAIL in .env or env to run this test.")

if __name__ == "__main__":
    asyncio.run(main())
