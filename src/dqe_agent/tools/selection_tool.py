"""Selection tool — structured drill-down for multi-item results.

When the agent retrieves a list of items (Jira sprints, issues, projects,
calendar events, etc.) and needs the user to pick one before proceeding,
it calls request_selection() instead of guessing or dumping a wall of text.

The frontend receives a ``selection_request`` WebSocket message containing
structured options and renders them as interactive buttons or a dropdown.
The user's pick is returned to the agent as the selected value string,
which the agent then passes to the next tool call.

Scenarios this covers
---------------------
Jira
  - Multiple sprints returned → pick which sprint
  - Multiple projects → pick project
  - Ambiguous issue search (>1 match) → pick exact issue
  - Multiple boards → pick board
  - Available status transitions → pick target status
  - Team member list → pick assignee
  - Issue types (Bug/Story/Task/Epic/Subtask) → pick type
  - Priorities → pick priority
  - Epics list → pick epic to link to
  - Issue link types (blocks / relates to / duplicates) → pick link type
  - Fix versions / releases → pick version

Calendar
  - Multiple calendars → pick which calendar
  - Multiple matching events → pick specific event
  - Date-range ambiguity → pick Today / This Week / This Month / Custom
  - Available time slots for scheduling → pick slot

General
  - Any query returning >1 result that requires user disambiguation
  - Bulk action scope (which project? which sprint? all?) → pick scope
  - Confirmation with labelled choices (Yes / No / More info)
"""

from __future__ import annotations

import json
import logging

from langgraph.types import interrupt

from dqe_agent.tools import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="request_selection",
    description=(
        "Present the user with a structured list of options and wait for them to "
        "pick one (or several if multi_select=True). "
        "ALWAYS use this tool when a query returns multiple items and the user "
        "must choose before you can proceed. Do NOT pick an item yourself. "
        "Do NOT list options in plain text — use this tool so the frontend can "
        "render proper interactive buttons. "
        "Covers: Jira sprints, projects, issues, boards, status transitions, "
        "assignees, issue types, priorities, epics, link types, fix versions; "
        "Calendar: calendars, events, date ranges, time slots; "
        "General: disambiguation, bulk scope, Yes/No confirmations with context. "
        "Each option must have 'value' (the ID/key for the next tool call) and "
        "'label' (human-readable text shown to the user)."
    ),
)
async def request_selection(
    question: str,
    options: list[dict[str, str]],
    multi_select: bool = False,
) -> str:
    """Interrupt execution and show the user a structured selection UI.

    Args:
        question:     What the user is choosing.
                      e.g. "Which sprint would you like to explore?"
        options:      List of dicts, each with:
                        "value" — the ID/key you pass to subsequent tool calls
                        "label" — the human-readable text shown to the user
                      Example:
                        [
                          {"value": "SP-23", "label": "Sprint 23 – Active (Apr 1–14)"},
                          {"value": "SP-22", "label": "Sprint 22 – Completed (Mar 18–31)"},
                        ]
        multi_select: If True, the user may pick multiple options.
                      The return value will be a JSON-encoded list of values.
                      Default False (single pick).

    Returns:
        Single pick:  the selected ``value`` string.
        Multi-select: JSON-encoded list of selected ``value`` strings,
                      e.g. '["SP-23","SP-22"]'. Parse with json.loads().
    """
    if not options:
        logger.warning("[request_selection] called with empty options list")
        return "no_options"

    # Validate each option has required keys; fill missing gracefully
    cleaned: list[dict[str, str]] = []
    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            opt = {"value": str(opt), "label": str(opt)}
        cleaned.append({
            "value": str(opt.get("value", opt.get("label", f"option_{i}"))),
            "label": str(opt.get("label", opt.get("value", f"Option {i + 1}"))),
        })

    result = interrupt({
        "type": "selection",
        "question": question,
        "options": cleaned,
        "multi_select": multi_select,
    })

    result_str = str(result) if result is not None else ""

    # Validate returned value is one of the valid option values.
    # Frontend may send free-text chat messages while a selection is pending.
    valid_values = {str(o["value"]) for o in cleaned}
    valid_labels = {str(o["label"]).lower() for o in cleaned}
    if result_str not in valid_values:
        # Try case-insensitive label match → map back to value
        _matched = next(
            (o["value"] for o in cleaned if o["label"].lower() == result_str.lower()),
            None,
        )
        if _matched:
            result_str = _matched
        else:
            # Invalid selection — re-interrupt with same question
            logger.warning(
                "[request_selection] invalid selection %r (not in options) — re-prompting",
                result_str,
            )
            result2 = interrupt({
                "type": "selection",
                "question": question,
                "options": cleaned,
                "multi_select": multi_select,
                "error": f"'{result_str}' is not a valid option. Please select one of the options below.",
            })
            result_str = str(result2) if result2 is not None else ""
            # Accept whatever comes back (don't loop infinitely)
            if result_str not in valid_values:
                _matched2 = next(
                    (o["value"] for o in cleaned if o["label"].lower() == result_str.lower()),
                    None,
                )
                result_str = _matched2 or cleaned[0]["value"]

    logger.info(
        "[request_selection] question=%r  selected=%r  multi=%s",
        question, result_str, multi_select,
    )

    # Return a JSON dict so the executor can store it in flow_data and the
    # template resolver can access {{step_id.selected}} and {{step_id.answer}}.
    # A plain string return would cause "Unresolved template params" because
    # json.loads("FLAG") fails and result["selected"] is unreachable.
    return json.dumps({"selected": result_str, "answer": result_str})
