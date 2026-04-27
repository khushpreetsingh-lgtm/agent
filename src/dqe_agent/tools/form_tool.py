"""Form tool — ask all independent fields at once instead of one at a time.

Frontend renders a multi-field form and returns all answers in one submit.
Each field answer is accessible as {{step_id.field_id}} in subsequent steps.
"""
from __future__ import annotations

import json
import logging

from langgraph.types import interrupt

from dqe_agent.tools import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="request_form",
    description=(
        "Present the user with a multi-field form and collect all answers at once. "
        "Use this instead of multiple ask_user steps when fields are independent "
        "(don't depend on each other's answers). "
        "Each field answer is referenced as {{step_id.field_id}} in subsequent steps. "
        "Field types: text, textarea, select, multi_select, date, number. "
        "For select/multi_select fields include an 'options' list of {value, label} dicts."
    ),
)
async def request_form(
    title: str,
    fields: list[dict],
    description: str = "",
) -> str:
    """Interrupt execution and show the user a multi-field input form.

    Args:
        title:       Heading shown above the form. e.g. "Create Jira Task"
        description: Optional context shown below the title (e.g. existing meetings, current state).
        fields:      List of field definitions. Each field:
                  id          — key used in template refs ({{step_id.id}})
                  label       — human-readable field name
                  type        — text | textarea | select | multi_select | date | number
                  required    — bool (default true)
                  placeholder — hint text (optional)
                  default     — pre-filled value shown in the input (optional, use for confirmation forms)
                  options     — list of {value, label} dicts (select/multi_select only)

    Returns:
        JSON string with all submitted field values keyed by field id.
        e.g. '{"summary": "Fix login", "issue_type": "Bug", "priority": "High"}'
        Reference individual values as {{step_id.summary}}, {{step_id.issue_type}}, etc.
    """
    if not fields:
        logger.warning("[request_form] called with no fields")
        return "{}"

    cleaned_fields = []
    _used_ids: set[str] = set()
    for f in fields:
        if not isinstance(f, dict):
            continue
        raw_id = str(f.get("id") or f.get("field_id") or f.get("name") or "")
        label_str = str(f.get("label") or f.get("name") or f.get("id") or "Field")
        if not raw_id:
            # Derive id from label: lowercase, spaces/special chars → underscore
            import re as _re_id
            raw_id = _re_id.sub(r'[^a-z0-9]+', '_', label_str.lower()).strip("_") or f"field_{len(cleaned_fields)}"
        # Ensure uniqueness
        base_id = raw_id
        counter = 1
        while raw_id in _used_ids:
            raw_id = f"{base_id}_{counter}"
            counter += 1
        _used_ids.add(raw_id)
        field = {
            "id": raw_id,
            "label": label_str,
            "type": str(f.get("type", "text")),
            "required": bool(f.get("required", True)),
        }
        if f.get("placeholder"):
            field["placeholder"] = str(f["placeholder"])
        if f.get("default") is not None:
            field["default"] = str(f["default"])
        if f.get("options"):
            opts = f["options"]
            field["options"] = [
                {"value": str(o.get("value", o) if isinstance(o, dict) else o),
                 "label": str(o.get("label", o.get("value", o)) if isinstance(o, dict) else o)}
                for o in opts
            ]
        cleaned_fields.append(field)

    payload: dict = {
        "type": "form",
        "title": title,
        "fields": cleaned_fields,
    }
    if description:
        payload["description"] = description

    result = interrupt(payload)

    if isinstance(result, dict):
        values = result
    elif isinstance(result, str):
        try:
            values = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            values = {"answer": result}
    else:
        values = {}

    logger.info("[request_form] title=%r  fields=%s  values=%s",
                title, [f["id"] for f in cleaned_fields], list(values.keys()))

    return json.dumps(values)
