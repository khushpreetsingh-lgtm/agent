"""Human-in-the-loop tool — pauses the workflow for human approval.

Uses LangGraph's interrupt() to guarantee the pause happens.
This is the ONLY way human reviews happen — deterministic, not LLM-decided.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from dqe_agent.schemas.models import HumanReview, OpportunityData
from dqe_agent.tools import register_tool
from dqe_agent.tools.user_selection import format_choices, parse_choice, parse_choice_llm

logger = logging.getLogger(__name__)


def _build_summary(review_type: str, state: dict, summary_fields: list[str] | None = None) -> str:
    """Build a human-readable summary depending on the review type."""

    if review_type == "info_review":
        opp = state.get("opportunity")
        struct = state.get("structure")
        if not opp:
            return "No opportunity data to review."
        lines = [
            "Please review the extracted information:\n",
            f"  Opportunity: {opp.opportunity_id}",
            f"  Buyer:       {opp.buyer}",
            f"  Bandwidth:   {opp.bandwidth_amount} {opp.bandwidth_unit}",
            f"  IP:          {opp.ip_address_assignment}",
            f"  Term:        {opp.term_amount} {opp.term_unit}",
            f"  Contact:     {opp.contact.name} — {opp.contact.email}",
        ]
        if struct:
            lines.append(f"  Address:     {struct.street}, {struct.city}, {struct.state} {struct.postcode}")
        lines.append("\nType 'proceed' to continue, or provide corrections.")
        return "\n".join(lines)

    elif review_type == "contact_review":
        opp = state.get("opportunity")
        if not opp:
            return "No contact data."
        c = opp.contact
        return (
            "Contact information:\n\n"
            f"  Name:           {c.name}\n"
            f"  Email:          {c.email}\n"
            f"  Phone:          {c.phone}\n"
            f"  Valid till:     {c.quote_valid_till}\n\n"
            "Type 'proceed' to keep, or provide new details."
        )

    elif review_type == "approver_review":
        return (
            "Do any reviewers or approvers need to be added?\n\n"
            "Type 'no' for on-net scenario (no approvals needed),\n"
            "or provide reviewer names/emails."
        )

    elif review_type == "email_review":
        email = state.get("email")
        if not email:
            return "No email to review."
        return (
            "Review the email before sending:\n\n"
            f"  To:      {', '.join(email.to)}\n"
            f"  CC:      {', '.join(email.cc) if email.cc else 'None'}\n"
            f"  Subject: {email.subject}\n\n"
            f"{email.body}\n\n"
            "Type 'send' to send, or provide modifications."
        )

    return "Please review and type 'proceed' to continue."


@register_tool("ask_user", "Pause and ask the user for any information needed to continue.")
async def ask_user(question: str, **kwargs) -> dict:
    """Interrupt execution and ask the user a question. Returns their answer.

    Use this whenever you need:
    - A 2FA / OTP / verification code
    - A missing address, name, email, or any detail not in state
    - A clarification or confirmation before a risky action

    IMPORTANT: If you are presenting a list of choices or options (e.g. meeting IDs)
    in the question, wrap each choice's key or ID in <<< and >>> markers so the
    frontend can render them as clickable options. For example: '<<<ID123>>>'
    """
    import re
    # Auto-wrap common ID formats in <<<>>> for the frontend if they aren't already wrapped.
    question = re.sub(r'\bID:\s*([a-zA-Z0-9_-]+)(?!\s*>+)', r'ID: <<<\1>>>', question)
    
    human_response = interrupt({"question": question})
    logger.info("[ask_user] Question: %s | Answer: %s", question, human_response)
    return {"answer": human_response}


@register_tool("human_review", "Pause the workflow and ask the human to review/approve.")
async def human_review(review_type: str, summary: str = "", **kwargs) -> dict:
    """Interrupt the workflow, show a summary, wait for human input.

    This uses LangGraph's interrupt() — the graph WILL pause here.
    No LLM decides whether to ask. It always asks.

    Args:
        review_type: Identifier for the type of review (e.g. "opportunity_confirm", "pricing_review")
        summary: Human-readable message shown to the user when pausing for review.
                 If empty, a default summary is built from state.
    """
    state = kwargs.get("__state__", {})
    summary_fields = kwargs.get("summary_fields", [])

    # Use the explicit summary if provided, otherwise build from state
    if not summary:
        summary = _build_summary(review_type, state, summary_fields)

    # Append clickable Yes/No choice buttons for the frontend to render
    summary += "\n\n<<<1. Yes, proceed>>>\n<<<2. No, stop>>>"

    # ── GUARANTEED PAUSE ──
    human_response = interrupt({"question": summary})

    _resp = human_response.lower().strip()
    _edit_signals = (
        "change", "chng", "chnge", "modif", "update", "edit", "fix",
        "wrong", "incorrect", "different", "go back", "back", "instead",
        "redo", "alter", "adjust", "revise", "not right", "not correct",
        "mistake", "error", "remove", "add more", "also add", "should be",
        "should say", "replace", "revert", "cancel", "start over", "try again",
        "make it", "set it", "set the", "make the", "switch", "swap",
        "months to", "mbps to", "term to", "bandwidth to",
    )
    _approve_signals = (
        "proceed", "ok", "okay", "yes", "y", "looks good", "send", "approve",
        "sure", "go ahead", "confirm", "fine", "good", "great", "no changes",
        "no change", "as is", "as-is", "correct", "right", "sounds good",
        "continue", "move on", "send it", "do it", "ship it", "go for it",
    )
    approved = (
        any(sig in _resp for sig in _approve_signals)
        and not any(sig in _resp for sig in _edit_signals)
    )

    result = {
        "human_review": HumanReview(
            step_name=review_type,
            question=summary,
            response=human_response,
            approved=approved,
        ),
    }

    # Special handling for approver review — set needs_approval flag
    if review_type == "approver_review":
        needs_approval = human_response.lower().strip() not in ("no", "n", "none", "skip")
        quote = state.get("quote")
        if quote:
            quote.needs_approval = needs_approval
            result["quote"] = quote

    # Special handling for email review — set approved flag
    if review_type == "email_review":
        email = state.get("email")
        if email:
            email.approved = approved
            result["email"] = email

    logger.info("[Human review: %s] Response: %s (approved=%s)", review_type, human_response, approved)
    return result


@register_tool("ask_user_choice", "Present a list of options and let the user choose one.")
async def ask_user_choice(title: str, options: str, **kwargs) -> dict:
    """Present numbered options to the user and return their selection.

    Args:
        title: Description of what the user is choosing (e.g., "Select a customer")
        options: Comma-separated list of options (e.g., "EQT Corporation, Acme Inc, Beta LLC")
    """
    option_list = [opt.strip() for opt in options.split(",") if opt.strip()]
    if not option_list:
        return {"selected": "", "index": -1, "error": "No options provided"}

    question = format_choices(title, option_list, allow_custom=True)
    response = interrupt({"question": question})

    idx, chosen = parse_choice(str(response), option_list)
    if idx is None:
        idx, chosen = await parse_choice_llm(str(response), option_list)
    logger.info("[ask_user_choice] title=%r response=%r → chosen=%r (idx=%s)", title, response, chosen, idx)
    return {"selected": chosen, "index": idx if idx is not None else -1, "raw_response": str(response)}
