"""User selection utilities — format choices & parse responses.

Since the frontend supports both text and voice input (mic button),
parsing must handle speech-to-text artifacts, natural language phrasing,
and anything a human might say. Fast regex paths handle obvious cases;
an LLM fallback interprets everything else.
"""

from __future__ import annotations

import logging
import re

from langgraph.types import interrupt

logger = logging.getLogger(__name__)


def format_choices(title: str, options: list[str], allow_custom: bool = False) -> str:
    """Format a numbered choice list for display in the chat.

    Options are wrapped in <<<>>> so the frontend can detect and render them
    as clickable selection buttons.
    """
    lines = [title, ""]
    for i, opt in enumerate(options, 1):
        lines.append(f"<<<{i}. {opt}>>>")
    lines.append("")
    if allow_custom:
        lines.append("Type a number to select, or type a different value.")
    else:
        lines.append("Type a number to select.")
    return "\n".join(lines)


def parse_choice(response: str, options: list[str]) -> tuple[int | None, str]:
    """Fast synchronous parse — handles obvious patterns only.

    Returns ``(index, chosen_text)`` where *index* is 0-based into *options*,
    or ``(None, raw_text)`` if no option matched (caller should use LLM fallback).
    """
    stripped = response.strip()
    lower = stripped.lower()

    # Pure digit: "1", "2", ...
    try:
        idx = int(stripped) - 1
        if 0 <= idx < len(options):
            return idx, options[idx]
    except ValueError:
        pass

    # Keyword + digit: "choose 1", "#3", "option 2"
    m = re.match(r'^(?:choose|option|pick|select|number|no\.?|#)\s*(\d+)\s*$', lower)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options):
            return idx, options[idx]

    # Exact case-insensitive match
    for i, opt in enumerate(options):
        if opt.lower() == lower:
            return i, opt

    # Substring match (handles partial names)
    for i, opt in enumerate(options):
        if lower in opt.lower() or opt.lower() in lower:
            return i, opt

    # No fast match — caller should use LLM
    return None, stripped


async def parse_choice_llm(response: str, options: list[str]) -> tuple[int | None, str]:
    """Use LLM to interpret the user's natural language / voice response.

    Handles: "the first one", "go with EQT", speech-to-text garble,
    ordinals ("second"), descriptions ("the corporation one"), etc.
    """
    from dqe_agent.llm import get_llm

    numbered = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = (
        f"The user was shown these numbered options:\n{numbered}\n\n"
        f"The user responded: \"{response}\"\n\n"
        f"Which option number (1-{len(options)}) did the user select? "
        f"Consider: spoken numbers (one, first), partial names, descriptions, "
        f"speech-to-text errors, or any natural language reference.\n"
        f"Return ONLY the number. If no option matches, return 0."
    )
    try:
        llm = get_llm()
        result = await llm.ainvoke(prompt)
        content = result.content.strip()
        # Extract first number from response
        m = re.search(r'\d+', content)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(options):
                logger.info("[user_selection] LLM interpreted '%s' as option %d: '%s'",
                            response, idx + 1, options[idx])
                return idx, options[idx]
    except Exception as exc:
        logger.debug("[user_selection] LLM parse failed: %s", exc)

    return None, response.strip()


async def ask_user_to_choose(
    title: str,
    options: list[str],
    allow_custom: bool = False,
) -> str:
    """Format choices, interrupt for user input, parse and return the selection.

    Uses fast regex matching first, falls back to LLM interpretation for
    natural language / voice input that simple patterns can't handle.

    Returns the selected option text (or the user's custom input).
    """
    question = format_choices(title, options, allow_custom)
    response = interrupt({"question": question})
    response_str = str(response)
    logger.info("[user_selection] title=%r response=%r", title, response_str)

    # Fast path: simple patterns (digit, exact match, substring)
    idx, chosen = parse_choice(response_str, options)
    if idx is not None:
        return chosen

    # Slow path: LLM interprets natural language / voice input
    idx, chosen = await parse_choice_llm(response_str, options)
    if idx is not None:
        return chosen

    # No match at all — return raw input
    return chosen
