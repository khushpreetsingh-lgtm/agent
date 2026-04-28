"""LLM content-generation tools.

When the user gives only a brief description (topic), these tools call the
executor-tier LLM to produce a first draft of professional content (email
body, subject, Jira issue description, sprint goal, comment text).

The draft is then handed to request_edit so the frontend can display it in
an editable textarea before the agent uses it in the final tool call.
"""
from __future__ import annotations

import logging

from dqe_agent.tools import register_tool

logger = logging.getLogger(__name__)

# ── Per-type prompt templates ─────────────────────────────────────────────────
_PROMPTS: dict[str, str] = {
    "email_body": (
        "Write a professional, concise email body. "
        "The email is about: {topic}. "
        "Additional context: {context}. "
        "Output ONLY the body text — no subject line, no 'From:' header. "
        "Start with a natural greeting if appropriate. Keep it brief and clear."
    ),
    "email_subject": (
        "Write a clear, concise email subject line for an email about: {topic}. "
        "Context: {context}. "
        "Output ONLY the subject line text — no quotes, no 'Subject:' prefix."
    ),
    "issue_description": (
        "Write a clear Jira issue description for: {topic}. "
        "Context: {context}. "
        "Include: what the issue is, why it matters, any acceptance criteria if obvious. "
        "Use plain markdown. Keep it concise."
    ),
    "sprint_goal": (
        "Write a clear sprint goal statement for a sprint about: {topic}. "
        "Context: {context}. "
        "1-2 sentences describing the business objective. No bullet points."
    ),
    "comment": (
        "Write a professional Jira comment about: {topic}. "
        "Context: {context}. "
        "Concise, informative, past tense for completed actions."
    ),
}


@register_tool(
    "llm_draft_content",
    "Use the AI to draft email body, email subject, Jira description, sprint goal, or comment text "
    "from a brief user description. Returns the generated text for user review/editing.",
)
async def llm_draft_content(
    content_type: str,
    topic: str,
    context: str = "",
) -> dict:
    """Generate professional content using the AI.

    Args:
        content_type: One of "email_body", "email_subject", "issue_description",
                      "sprint_goal", "comment".
        topic:        Short description of what the content is about (from ask_user).
        context:      Optional extra context (recipient name, project, etc.).

    Returns:
        {"content": "<generated text>"}
    """
    from dqe_agent.llm import get_executor_llm

    prompt_template = _PROMPTS.get(
        content_type,
        "Write professional content about: {topic}. Context: {context}. Be concise.",
    )
    prompt = prompt_template.format(topic=topic or context or "unspecified", context=context or topic or "none")

    try:
        llm = get_executor_llm()
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        logger.info(
            "[llm_draft_content] type=%r topic=%r → %d chars",
            content_type, topic, len(content),
        )
        return {"content": content}
    except Exception as exc:
        logger.error("[llm_draft_content] LLM call failed: %s", exc)
        # Fallback: return the raw topic so the flow doesn't break
        return {"content": topic, "error": str(exc)}
