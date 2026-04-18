"""Agent Nodes — LangGraph nodes for the ReAct (chat) graph.

Simplified: removed flow-specific reminder state machine.
Flows now just provide system prompt + optional tool allowlist.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from dqe_agent.state import AgentState
from dqe_agent.flows import FlowConfig
from dqe_agent.llm import get_llm
from dqe_agent.tools import list_tools

logger = logging.getLogger(__name__)

_CONTENT_FILTER_KEYWORDS = ("content_filter", "content management policy", "ResponsibleAIPolicyViolation")


def _tools_for_flow(flow_config: FlowConfig | None):
    """Return the tool list for a flow, honouring the optional allowlist."""
    if flow_config is not None and flow_config.tools is not None:
        return list_tools(flow_config.tools)
    return list_tools()


async def _invoke_with_content_filter_retry(llm, messages, max_retries: int = 2):
    """Invoke the LLM, retrying on Azure content filter errors."""
    for attempt in range(1 + max_retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:
            err_text = str(exc)
            is_content_filter = any(kw in err_text for kw in _CONTENT_FILTER_KEYWORDS)
            if not is_content_filter or attempt >= max_retries:
                raise
            logger.warning(
                "Azure content filter triggered (attempt %d/%d) — sanitizing",
                attempt + 1, max_retries,
            )
            sanitized = []
            for m in messages:
                if isinstance(m, ToolMessage) and len(str(m.content)) > 500:
                    truncated = str(m.content)[:500] + "... [truncated]"
                    sanitized.append(ToolMessage(content=truncated, tool_call_id=m.tool_call_id))
                else:
                    sanitized.append(m)
            messages = sanitized


def create_agent_node(flow_config: FlowConfig | None = None):
    """Create the agent reasoning node for ReAct mode."""
    llm = get_llm()
    tools = _tools_for_flow(flow_config)
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        logger.info("Agent thinking... (messages=%d)", len(state.get("messages", [])))
        messages = list(state.get("messages", []))

        # Add system prompt on first turn
        if not any(isinstance(m, AIMessage) for m in messages):
            prompt = flow_config.system_prompt if flow_config else "You are a helpful browser automation assistant."
            messages = [SystemMessage(content=prompt)] + messages

        # Fix orphaned tool_calls (after interrupt/resume)
        answered_ids: set[str] = set()
        for m in messages:
            if m.__class__.__name__ == "ToolMessage" and hasattr(m, "tool_call_id"):
                answered_ids.add(m.tool_call_id)

        patched = []
        for m in messages:
            patched.append(m)
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tc_id = tc.get("id") or tc.get("tool_call_id", "")
                    if tc_id and tc_id not in answered_ids:
                        logger.warning("Injecting placeholder for orphaned tool_call_id=%s", tc_id)
                        patched.append(ToolMessage(
                            content='{"status": "interrupted", "detail": "Resumed after interrupt."}',
                            tool_call_id=tc_id,
                        ))
                        answered_ids.add(tc_id)
        messages = patched

        # Inject flow-specific reminder (if any)
        last_msg = messages[-1] if messages else None
        has_dangling = hasattr(last_msg, "tool_calls") and bool(last_msg.tool_calls)
        if not has_dangling and flow_config is not None:
            all_tool_contents = [
                str(m.content) for m in messages
                if m.__class__.__name__ == "ToolMessage"
            ]
            last_tool = all_tool_contents[-1] if all_tool_contents else ""
            reminder = flow_config.select_reminder(messages, all_tool_contents, last_tool)
            if reminder:
                messages = messages + [SystemMessage(content=reminder)]

        response = await _invoke_with_content_filter_retry(llm_with_tools, messages)

        # Trace the ReAct agent LLM call
        from dqe_agent.observability import trace_llm_call
        trace_llm_call(model="react", role="agent")

        if response.tool_calls:
            logger.info("Agent calling tools: %s", [tc["name"] for tc in response.tool_calls])
        else:
            logger.info("Agent text response (no tools)")

        return {"messages": [response], "error": None}

    return agent_node


def create_tool_node(flow_config: FlowConfig | None = None):
    """Create the tool execution node."""
    tools = _tools_for_flow(flow_config)
    return ToolNode(tools)


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: call tools or finish."""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"
