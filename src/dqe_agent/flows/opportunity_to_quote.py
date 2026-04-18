"""Opportunity-to-Quote flow — simplified for v3 architecture.

The complex state machine is gone. Workflow logic now lives in:
  - workflows/opportunity_to_quote.yaml (deterministic steps)
  - agent/planner.py (dynamic planning)

This flow config just provides the system prompt and tool allowlist
for ReAct chat mode when the user selects this flow.
"""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage

from dqe_agent.flows import FlowConfig, register_flow


class OpportunityToQuoteFlow(FlowConfig):
    flow_id = "opportunity_to_quote"
    description = "Extract opportunity from NetSuite, create quote in CPQ, email customer"

    tools = [
        "browser_login",
        "browser_navigate",
        "browser_act",
        "browser_extract",
        "browser_click",
        "browser_type",
        "browser_wait",
        "browser_snapshot",
        "ask_user",
        "human_review",
    ]

    system_prompt = """\
You are a browser automation agent for DQE (Data Quote Engine). Your primary task is
creating quotes from NetSuite opportunities.

STANDARD WORKFLOW:
1. Log in to NetSuite (browser_login site=netsuite)
2. Search for the opportunity by ID
3. Extract opportunity details (buyer, bandwidth, term, address, contact)
4. Search and extract the Structure record for the address
5. Log in to CPQ (browser_login site=cpq)
6. Create a new quote in CPQ
7. Fill the quote wizard with extracted data
8. Get pricing
9. Request human review at key gates
10. Finalize the quote
11. Click the Send Email button in CPQ to send the quote email

RULES:
- Always start by asking for the Opportunity ID if not provided
- Login to each system before interacting with it
- Extract data before moving to CPQ
- Request human review before finalizing
- Be precise with form filling — use exact values from NetSuite"""

    def select_reminder(
        self,
        messages: Sequence[BaseMessage],
        all_tool_contents: list[str],
        last_tool_content: str,
    ) -> str | None:
        """Minimal reminders — the heavy logic is in the planner now."""
        if not all_tool_contents:
            return None
        return None


register_flow(OpportunityToQuoteFlow())
