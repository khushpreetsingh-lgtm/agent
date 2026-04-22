"""Request-edit tool — send LLM-generated content to the frontend for user editing.

Works exactly like ask_user / request_selection: uses LangGraph's interrupt()
to pause execution, sends an edit_request WebSocket message, waits for the user
to edit the text and submit (edit_response), then resumes with the final content.

The frontend should render an editable textarea pre-filled with `content`, let
the user modify it, and send:
  {"type": "edit_response", "content": "<edited text>"}
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

from dqe_agent.tools import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    "request_edit",
    "Show LLM-generated content (email body, issue description, etc.) to the user in an "
    "editable textarea. User reviews, edits if needed, then approves. Returns the final text.",
)
async def request_edit(
    label: str,
    content: str,
    question: str = "",
) -> dict:
    """Interrupt execution, send pre-generated content to the frontend for editing.

    Args:
        label:    Name of what's being edited — shown as the textarea heading.
                  e.g. "Email Body", "Email Subject", "Issue Description".
        content:  The AI-generated text to pre-fill in the editable area.
        question: Optional instruction shown above the textarea.
                  Defaults to "Review and edit the <label> below, then click Send."

    Returns:
        {"content": "<final text after edits>", "edited": bool, "original": "<original>"}
    """
    if not question:
        question = f"Review and edit the {label.lower()} below, then click Send:"

    result = interrupt({
        "type": "edit_request",
        "label": label,
        "content": content,
        "question": question,
    })

    edited_content = str(result).strip() if result is not None else content
    was_edited = edited_content != content.strip()

    logger.info(
        "[request_edit] label=%r  original_len=%d  edited_len=%d  changed=%s",
        label, len(content), len(edited_content), was_edited,
    )

    return {
        "content": edited_content,
        "edited": was_edited,
        "original": content,
    }
