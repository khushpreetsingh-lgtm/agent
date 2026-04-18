"""Direct response tool — allows the PEV executor to deliver conversational replies.

When the planner determines a message is a simple greeting/question/chat (no browser
needed), it produces a 1-step plan with tool="direct_response". The executor calls
this tool, which simply passes the message through as the result.
"""
from dqe_agent.tools import register_tool


@register_tool("direct_response", "Return a conversational response to the user (no browser action needed).")
async def direct_response(message: str) -> str:
    """Return the message as-is. Used by the planner for non-browser responses."""
    return message
