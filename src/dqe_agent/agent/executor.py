"""Executor node — executes ONE step from the plan using a fast/cheap model.

Uses GPT-4o-mini (or equivalent) to map plan steps to actual tool calls.
The executor doesn't think about WHAT to do — it just follows the plan.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt

from dqe_agent.guardrails import COST_PER_CALL, GuardrailError
from dqe_agent.state import AgentState

# Import project-key validator from planner (avoids circular imports — planner doesn't import executor)
from dqe_agent.agent.planner import _is_valid_jira_key  # type: ignore
from dqe_agent.agent.planner import _cache_get  # access planner cache for project options

logger = logging.getLogger(__name__)

# ── Shared constants / micro-helpers ─────────────────────────────────────────

_PERIOD_OPTIONS: list[dict[str, str]] = [
    {"value": "__ALL_TIME__",  "label": "All time (no date filter)"},
    {"value": "__CUSTOM_RANGE__", "label": "Custom range..."},
    {"value": "worklogDate = now()", "label": "Today"},
    {"value": "worklogDate = -1d",   "label": "Yesterday"},
    {"value": "worklogDate >= startOfWeek()", "label": "This week"},
    {"value": "worklogDate >= startOfMonth()", "label": "This month"},
    {"value": "worklogDate >= startOfMonth(-1) AND worklogDate <= endOfMonth(-1)", "label": "Last month"},
]


def _compute_free_slots(
    events_text: str,
    date_str: str,
    slot_minutes: int = 30,
    include_busy: bool = False,
) -> list[dict[str, str]]:
    """Parse get_events plain-text result and return IST time slots as {value, label} options.

    Parses lines like:
      - "Event Name" (Starts: 2026-04-24T16:00:00+05:30, Ends: 2026-04-24T16:30:00+05:30)
    Returns all 30-min slots across 24 hours for the given date (IST).
    By default excludes busy slots. When include_busy=True, busy slots are included but
    labelled with " (busy)" so the user can see the conflict.
    Values are IST RFC3339 (e.g. 2026-04-25T14:00:00+05:30); labels show IST 12-hour time.
    """
    import re as _re_fs
    from datetime import datetime, timedelta, timezone, date as _date

    IST = timezone(timedelta(hours=5, minutes=30))

    try:
        ref_date = _date.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        ref_date = _date.today()

    def _parse_ts(s: str) -> datetime:
        s = s.strip()
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)

    # Parse busy periods — one line per event, match Start+End on the same line only
    busy: list[tuple[datetime, datetime]] = []
    ts_pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)"
    for line in events_text.splitlines():
        m = _re_fs.search(
            r"[Ss]tarts?:\s*" + ts_pattern + r"[^|\n]*?[Ee]nds?:\s*" + ts_pattern,
            line,
        )
        if m:
            try:
                b_start, b_end = _parse_ts(m.group(1)), _parse_ts(m.group(2))
                # Only count events that overlap with ref_date in IST
                b_start_ist = b_start.astimezone(IST)
                if b_start_ist.date() == ref_date:
                    busy.append((b_start, b_end))
                    logger.info(
                        "[EXECUTOR] busy slot parsed: %s → %s",
                        b_start_ist.strftime("%I:%M %p IST"),
                        b_end.astimezone(IST).strftime("%I:%M %p IST"),
                    )
                else:
                    logger.info(
                        "[EXECUTOR] busy slot skipped (wrong date %s, expected %s): %s",
                        b_start_ist.date(), ref_date,
                        b_start_ist.strftime("%I:%M %p IST"),
                    )
            except Exception:
                continue

    # Full 24-hour range: 00:00 to 24:00 IST (next day midnight) so 23:30 slot is included
    day_start_ist = datetime(ref_date.year, ref_date.month, ref_date.day, 0, 0, tzinfo=IST)
    day_end_ist = day_start_ist + timedelta(hours=24)
    work_s = day_start_ist.astimezone(timezone.utc)
    work_e = day_end_ist.astimezone(timezone.utc)

    # For today: skip slots already past
    now_utc = datetime.now(tz=timezone.utc)
    if ref_date == _date.today() and now_utc > work_s:
        mins_past = int((now_utc - work_s).total_seconds() / 60)
        slots_past = (mins_past // slot_minutes) + 1
        work_s = work_s + timedelta(minutes=slots_past * slot_minutes)

    slots: list[dict[str, str]] = []
    cursor = work_s
    while cursor + timedelta(minutes=slot_minutes) <= work_e and len(slots) < 48:
        slot_end = cursor + timedelta(minutes=slot_minutes)
        is_busy = any(
            not (slot_end <= b_s or cursor >= b_e)
            for b_s, b_e in busy
        )
        if not is_busy or include_busy:
            cursor_ist = cursor.astimezone(IST)
            label = cursor_ist.strftime("%I:%M %p").lstrip("0") + " IST"
            if is_busy:
                label += " (busy)"
            value = cursor_ist.strftime("%Y-%m-%dT%H:%M:%S+05:30")
            slots.append({"value": value, "label": label})
        cursor = slot_end

    return slots


def _recover_project_key(results_by_id: dict, flow_data: dict) -> str:
    """Return the project key found in prior step results / flow_data, or ''."""
    try:
        val = _find_selected_value(
            results_by_id, flow_data,
            priority_words=["project", "proj", "sel_proj", "select"],
            transform=lambda v: v.strip().upper(),
            validate=lambda v: bool(_is_valid_jira_key(v)),
        )
        return val or ""
    except Exception:
        return ""


def _fuzzy_match_project_key_cached(guess: str) -> str:
    """Fuzzy-match a guessed project key/name against the planner cache.

    Returns the correct Jira project key, or '' if no match found.
    Matching priority:
      1. Exact key (case-insensitive)
      2. Key is a prefix/substring of guess or vice-versa
      3. All words from guess appear in the project name
      4. Any word from guess appears in the project name or key
    """
    try:
        from dqe_agent.agent.planner import _cache_get
        cached = _cache_get("jira_projects") or []
        if not cached:
            return ""
    except Exception:
        return ""

    guess_upper = guess.strip().upper()
    guess_words = [w.lower() for w in guess.split() if len(w) > 1]

    # Pass 1: exact key match
    for p in cached:
        if isinstance(p, dict) and p.get("value", "").upper() == guess_upper:
            return p["value"]

    # Pass 2: key prefix
    for p in cached:
        if not isinstance(p, dict):
            continue
        k = p.get("value", "").upper()
        if k and (guess_upper.startswith(k) or k.startswith(guess_upper)):
            return p["value"]

    # Pass 3: all words in project name
    for p in cached:
        if not isinstance(p, dict):
            continue
        label_lower = p.get("label", "").lower()
        if guess_words and all(w in label_lower for w in guess_words):
            return p["value"]

    # Pass 4: any word matches name or key
    for p in cached:
        if not isinstance(p, dict):
            continue
        label_lower = p.get("label", "").lower()
        k_lower = p.get("value", "").lower()
        if any(w in label_lower or w in k_lower for w in guess_words):
            return p["value"]

    # Pass 5: fuzzy/approximate match using sequence similarity
    try:
        from difflib import SequenceMatcher
        best = (None, 0.0)
        for p in cached:
            if not isinstance(p, dict):
                continue
            label_lower = p.get("label", "").lower()
            k_lower = p.get("value", "").lower()
            r1 = SequenceMatcher(None, guess.lower(), label_lower).ratio()
            r2 = SequenceMatcher(None, guess.lower(), k_lower).ratio()
            r = max(r1, r2)
            if r > best[1]:
                best = (p.get("value"), r)
        if best[0] and best[1] >= 0.68:
            return best[0]
    except Exception:
        pass

    return ""


async def _fetch_assignable_users(proj_key: str) -> list:
    """Call jira_get_assignable_users and return the unwrapped list, or []."""
    if not proj_key:
        return []
    try:
        from dqe_agent.tools import get_tool
        user_tool = get_tool("jira_get_assignable_users")
        users_raw = await user_tool.ainvoke({"project_key": proj_key})
        users = _unwrap_mcp_result(users_raw)
        return users if isinstance(users, list) and users else []
    except Exception:
        return []


def _unwrap_mcp_result(raw: Any) -> Any:
    """Unwrap LangChain MCP content-block lists to the actual payload.

    langchain-mcp-adapters sometimes returns:
      [{"type": "text", "text": "..actual json..", "id": "lc_<uuid>"}]
    instead of a plain string/dict.  Storing str() of this produces
    Python-repr (single quotes) which json.loads() cannot parse later.
    Extract and parse the text content so we store real JSON.
    """
    if not isinstance(raw, list) or not raw:
        return raw
    if all(
        isinstance(item, dict) and item.get("type") == "text" and "text" in item for item in raw
    ):
        combined = "\n".join(item["text"] for item in raw)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return combined  # return as string — still better than Python repr
    return raw


def _step_message(
    idx: int, total: int, tool: str, status: str, error: str, params: dict, result: Any
) -> str:
    """Return a human-readable progress message for a single step."""
    prefix = f"Step {idx + 1}/{total}"

    if status == "failed":
        return f"{prefix} [{tool}] failed — {error[:250]}"

    if tool == "request_selection":
        q = params.get("question", "")
        try:
            selected = (
                json.loads(result).get("selected", result) if isinstance(result, str) else result
            )
        except Exception:
            selected = result or ""
        return f"{prefix} Selected: {selected}" + (f" (for: {q})" if q else "")

    if tool == "ask_user":
        q = params.get("question", "")
        try:
            answer = json.loads(result).get("answer", result) if isinstance(result, str) else result
        except Exception:
            answer = result or ""
        return f"{prefix} {q}: {answer}"

    # For MCP/API tool calls show a brief version of the result
    brief = ""
    if result:
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict):
                # Search/list result: show issue count
                issues = parsed.get("issues") or parsed.get("_items")
                if issues is not None:
                    real_total = parsed.get("total", -1)
                    count = len(issues)
                    if real_total and real_total > 0 and real_total != count:
                        brief = f" → {count} of {real_total} item(s) returned"
                    else:
                        brief = f" → {count} item(s) returned"
                else:
                    # Show a few key fields — id, name, key, summary, etc.
                    for key in ("id", "name", "key", "summary", "sprint", "sprintId", "self"):
                        if key in parsed:
                            brief = f" → {key}: {parsed[key]}"
                            break
                    if not brief:
                        # Skip -1 / negative sentinel values as they're meaningless
                        for v in parsed.values():
                            if v is not None and v != -1 and str(v).strip() not in ("-1", ""):
                                brief = f" → {str(v)[:80]}"
                                break
            elif isinstance(parsed, list):
                brief = f" → {len(parsed)} item(s) returned"
        except Exception:
            brief = f" → {str(result)[:80]}"

    label = tool.replace("jira_", "").replace("_", " ").title()
    return f"{prefix} {label}: {status}{brief}"


def _no_results_sentence(prefix: str) -> str:
    """Turn a prefix like 'Tania's open tasks' into a clean empty-state sentence."""
    import re as _re_nrs
    raw = prefix.strip()
    p = raw.lower()

    # Extract person name from possessive prefix e.g. "Tania's open issues" → "Tania"
    _poss = _re_nrs.match(r"^([A-Z][A-Za-z ]{1,30}?)'s\b", raw)
    person = _poss.group(1).strip() if _poss else None

    # Strip common filler openers so we can inspect the subject
    for filler in ("here are your ", "here are the ", "your ", "the "):
        if p.startswith(filler):
            p = p[len(filler):]
            break

    # If a named person is in the prefix, use their name in the sentence
    if person:
        if "completed" in p or "done" in p:
            return f"{person} has no completed tasks."
        if "task" in p or "issue" in p or "ticket" in p:
            return f"{person} doesn't have any tasks right now."
        return f"No issues found for {person}."

    # Map common subjects to a natural sentence.
    # ORDER MATTERS — more specific entries must come before generic ones.
    _MAP = [
        ("blocker", "No blockers right now — all clear!"),
        ("critical", "No critical issues found."),
        ("high priority", "No high priority issues found."),
        ("medium priority", "No medium priority issues found."),
        ("low priority", "No low priority issues found."),
        ("unassigned", "No unassigned issues found in the sprint."),
        ("in progress", "Nothing is currently in progress."),
        ("in review", "Nothing is currently in review."),
        ("to do", "No items in To Do right now."),
        ("testing", "No issues are in testing right now."),
        ("backlog", "The backlog is empty."),
        ("done", "No issues marked as Done yet."),
        ("open tasks", "You don't have any open tasks right now."),
        ("open issues", "You don't have any open issues right now."),
        ("open tickets", "You don't have any open tickets right now."),
        ("tasks", "You don't have any tasks right now."),
        ("issues", "You don't have any issues right now."),
        ("tickets", "You don't have any tickets right now."),
    ]
    for key, sentence in _MAP:
        if key in p:
            return sentence
    # Fallback: build a generic sentence from the prefix
    subject = raw.rstrip(":").strip() if raw else "issues"
    return f"No {subject.lower()} found."


def _format_result_for_display(raw: str) -> str:
    """Convert a raw MCP result string into a human-readable message.

    Handles:
    - Jira search results → friendly issue list or "no results" message
    - Generic JSON → pretty-printed
    - Plain text → returned as-is
    """
    if not raw or not isinstance(raw, str):
        return raw or ""

    stripped = raw.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return raw  # plain text — no transformation needed

    extra_results: list = []
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return raw

    # ── Jira search result: {"total": N, "issues": [...]} ───────────────────
    if isinstance(data, dict) and "issues" in data:
        issues = data.get("issues", [])
        total = data.get("total", len(issues))

        if not issues:
            return "No issues found."

        def _str_field(obj: Any, *keys: str) -> str:
            for k in keys:
                v = obj.get(k)
                if isinstance(v, dict):
                    v = v.get("name") or v.get("displayName") or ""
                if v and isinstance(v, str):
                    return v
            return ""

        blocks: list[str] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            key = issue.get("key", "")
            fields = issue.get("fields", issue)
            summary = (
                _str_field(fields, "summary") or _str_field(issue, "summary") or "(no summary)"
            )
            status = _str_field(fields, "status") or _str_field(issue, "status")
            priority = _str_field(fields, "priority") or _str_field(issue, "priority")
            assignee = (
                _str_field(fields, "assignee") or _str_field(issue, "assignee") or "Unassigned"
            )
            issue_type = _str_field(fields, "issuetype") or _str_field(issue, "issuetype")

            # Line 1: key + summary (with Jira link)
            if key:
                from dqe_agent.config import settings as _cfg
                jira_base = _cfg.jira_url.rstrip("/") if _cfg.jira_url else ""
                title = f"**[{key}]({jira_base}/browse/{key})** — {summary}" if jira_base else f"**{key}** — {summary}"
            else:
                title = f"**{summary}**"
            # Line 2: meta details
            meta_parts = []
            if status:
                meta_parts.append(f"Status: {status}")
            if priority:
                meta_parts.append(f"Priority: {priority}")
            if issue_type:
                meta_parts.append(f"Type: {issue_type}")
            meta_parts.append(f"Assignee: {assignee}")
            meta_line = "   " + "   |   ".join(meta_parts)

            blocks.append(f"{title}\n{meta_line}")

        count = len(issues)
        # Determine dominant issue type from results for friendlier label
        _type_counts: dict[str, int] = {}
        for _iss in issues:
            _f = _iss.get("fields", _iss)
            _it = (_f.get("issuetype") or {})
            _itn = _it.get("name") if isinstance(_it, dict) else str(_it)
            if _itn:
                _type_counts[_itn.lower()] = _type_counts.get(_itn.lower(), 0) + 1
        _dominant = max(_type_counts, key=_type_counts.get) if _type_counts else None
        if _dominant and len(_type_counts) == 1:
            _label = _dominant  # e.g. "task", "bug", "story"
            _label_pl = _label + "s" if not _label.endswith("s") else _label
        else:
            _label = "issue"
            _label_pl = "issues"
        header = f"**{count} {_label_pl if count != 1 else _label} found:**\n\n"
        return header + "\n\n".join(blocks)

    # ── Worklog result: {"project": ..., "users": [...]} ────────────────────
    if isinstance(data, dict) and "users" in data and "project" in data:
        users = data.get("users", [])
        project = data.get("project", "")
        from_date = data.get("from", "")
        to_date = data.get("to", "")
        header = f"**Logged hours — {project}** ({from_date} → {to_date})\n\n"
        if not users:
            return header + (data.get("summary") or "No worklogs found for this period.")
        lines: list[str] = []
        for u in users:
            member = u.get("member", "Unknown")
            total = u.get("total_formatted", f"{u.get('total_hours', 0)}h")
            lines.append(f"**{member}** — {total}")
            for iss in u.get("issues", []):
                lines.append(f"  • {iss['key']} — {iss.get('formatted', '')}  _{iss.get('summary', '')}_")
        return header + "\n".join(lines)

    # ── Generic JSON: pretty-print ───────────────────────────────────────────
    return json.dumps(data, indent=2)


def _build_completion_summary(step_results: list) -> str:
    """Build a rich human-readable summary of what was accomplished."""
    _INTERACTION_TOOLS = {
        "request_selection",
        "ask_user",
        "human_review_request",
        "direct_response",
    }

    # Collect user inputs and selections
    selections: dict[str, str] = {}
    for r in step_results:
        if not isinstance(r, dict):
            continue
        tool = r.get("tool", "")
        step_id = r.get("step_id", "")
        raw = r.get("result", "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                if "answer" in parsed:
                    selections[step_id] = str(parsed["answer"])
                elif "selected" in parsed:
                    selections[step_id] = str(parsed["selected"])
        except Exception:
            if tool in ("ask_user", "request_selection"):
                selections[step_id] = str(raw)[:120]

    # Find the last successful non-interaction tool result (the "create/update" result)
    primary_result: dict = {}
    primary_tool: str = ""
    for r in reversed(step_results):
        if not isinstance(r, dict):
            continue
        if r.get("tool") in _INTERACTION_TOOLS:
            continue
        if r.get("status") != "success":
            continue
        raw = r.get("result", "")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                primary_result = parsed
                primary_tool = r.get("tool", "")
                break
        except Exception:
            pass

    lines: list[str] = []

    # Title line based on the primary tool
    if primary_tool:
        label = primary_tool.replace("jira_", "").replace("_", " ").title()
        lines.append(f"**{label} completed successfully.**")
    else:
        lines.append("**Task completed successfully.**")

    # Key fields from the primary result
    KEY_FIELDS = [
        ("id", "ID"),
        ("key", "Key"),
        ("name", "Name"),
        ("summary", "Summary"),
        ("sprintId", "Sprint ID"),
        ("boardId", "Board ID"),
        ("startDate", "Start Date"),
        ("endDate", "End Date"),
        ("state", "State"),
        ("goal", "Goal"),
        ("self", None),  # skip "self" (URL noise)
    ]
    shown: set[str] = set()
    for field, label in KEY_FIELDS:
        if label is None:
            continue
        if field in primary_result:
            lines.append(f"  {label}: {primary_result[field]}")
            shown.add(field)

    # Remaining primary result fields (limit noise)
    extras = [
        (k, v)
        for k, v in primary_result.items()
        if k not in shown and k not in ("self", "links", "_links")
    ]
    for k, v in extras[:6]:
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"  {k}: {v}")

    # User selections / answers (project, board, sprint name, etc.)
    if selections:
        lines.append("")
        lines.append("**Inputs used:**")
        for step_id, val in selections.items():
            label = step_id.replace("_", " ").title()
            lines.append(f"  {label}: {val}")

    return "\n".join(lines)


async def executor_node(state: AgentState, _tool_filter: list[str] | None = None) -> dict:
    """Execute the current step in the plan."""
    from dqe_agent.tools import get_tool

    # Resolve tool filter from agent_id if not passed directly
    if _tool_filter is None and state.get("agent_id"):
        try:
            from dqe_agent.agents import get_agent as _get_agent_cfg
            _cfg = _get_agent_cfg(state["agent_id"])
            _tool_filter = _cfg.tools  # may still be None = all tools
        except KeyError:
            pass

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    steps_taken = state.get("steps_taken", 0)
    cost = state.get("estimated_cost", 0.0)

    if idx >= len(plan):
        summary = _build_completion_summary(state.get("step_results", []))
        return {"status": "complete", "messages": [AIMessage(content=summary)]}

    step = plan[idx]
    step_id = step.get("id", f"step_{idx}")
    tool_name = step.get("tool", "")
    params = step.get("params", {})
    description = step.get("description", "")

    logger.info(
        "[EXECUTOR] Step %d/%d: [%s] %s",
        idx + 1,
        len(plan),
        tool_name,
        description or "(no description)",
    )

    # Ensure flow_data is available early — some direct_response paths
    # (replanning after MCP failures) inspect flow_data for user corrections.
    flow_data = state.get("flow_data", {})

    # ── direct_response: planner chose to reply without browser tools ─────
    if tool_name == "direct_response":
        raw_msg = params.get("message", "")

        # Sentinel: MCP failure path — replan using user correction from flow_data
        if raw_msg == "__replan_after_mcp_fix__":
            fix_steps = {k: v for k, v in flow_data.items() if k.startswith("fix_")}
            correction = ""
            for v in reversed(list(fix_steps.values())):
                if isinstance(v, dict) and v.get("answer"):
                    correction = v["answer"]
                    break
            replan_suffix = (
                f"\n\nThe previous MCP tool call failed. "
                f"The user provided this correction: {correction!r}\n"
                f"Use the corrected value(s) directly in the new plan. "
                f"Do NOT re-ask for data already collected."
            )
            logger.info("[EXECUTOR] MCP replan triggered — user correction: %r", correction)
            return {
                "steps_taken": steps_taken + 1,
                "estimated_cost": cost,
                "status": "planning",
                "task": state.get("task", "") + replan_suffix,
                "messages": [AIMessage(content="Replanning with your correction...")],
            }

        # Resolve any {{step_id.field}} templates in the message
        step_results = state.get("step_results", [])
        results_by_id = {r.get("step_id", ""): r for r in step_results if isinstance(r, dict)}
        if "{{" in raw_msg:
            response_msg = _resolve_template(raw_msg, flow_data, results_by_id)
            # Format any embedded JSON result into a human-readable string
            if response_msg and response_msg.strip().startswith(("[", "{")):
                response_msg = _format_result_for_display(response_msg)
            elif response_msg:
                # Fast path: if the whole resolved message is a Jira issues payload
                # (deeply nested — the regex approach can't handle it), format directly.
                try:
                    _quick = json.loads(response_msg.strip()) if response_msg.strip().startswith(("{", "[")) else None
                    if isinstance(_quick, dict) and "issues" in _quick:
                        response_msg = _format_result_for_display(response_msg.strip())
                        _quick = None  # handled
                except Exception:
                    pass
                # Replace ALL inline JSON blobs embedded in the message text.
                # e.g. "Total: {"issues":[],"total":5} | Done: {"issues":[],"total":2}"
                # Strategy: find JSON blobs and replace each with a formatted value.
                import re as _re_inline

                def _replace_json_blob(blob_str: str, context_before: str) -> str:
                    """Format a single JSON blob found inside a message."""
                    try:
                        _d = json.loads(blob_str)
                    except Exception:
                        _d = None

                    # For count-context fields (Total, Done, Blockers etc.) extract the number
                    ctx = context_before.lower().rstrip(": \t")
                    _is_count_ctx = any(w in ctx for w in ("total", "count", "done", "completed", "blockers", "remaining", "open", "in progress"))
                    if _is_count_ctx and isinstance(_d, dict):
                        issues = _d.get("issues") or []
                        raw_total = _d.get("total", len(issues))
                        # Jira Cloud returns -1 when count is unknown — fall back to issues list length
                        actual = len(issues) if (raw_total is None or raw_total < 0) else raw_total
                        return str(actual)

                    formatted = _format_result_for_display(blob_str)
                    if formatted == "No issues found.":
                        return "none"
                    return formatted

                # Find all top-level JSON blobs using a depth counter (handles
                # deeply nested objects that regex can't match).
                def _find_json_blobs(text: str):
                    """Yield (start, end) of each top-level JSON object/array."""
                    i = 0
                    n = len(text)
                    while i < n:
                        if text[i] in ('{', '['):
                            opener = text[i]
                            closer = '}' if opener == '{' else ']'
                            depth = 0
                            in_str = False
                            esc = False
                            start = i
                            while i < n:
                                c = text[i]
                                if esc:
                                    esc = False
                                elif in_str:
                                    if c == '\\':
                                        esc = True
                                    elif c == '"':
                                        in_str = False
                                else:
                                    if c == '"':
                                        in_str = True
                                    elif c == opener:
                                        depth += 1
                                    elif c == closer:
                                        depth -= 1
                                        if depth == 0:
                                            yield start, i + 1
                                            break
                                i += 1
                        i += 1

                _parts = []
                _last_end = 0
                _any_blob = False
                for _bstart, _bend in _find_json_blobs(response_msg):
                    _blob = response_msg[_bstart:_bend]
                    if any(k in _blob for k in ('"issues"', '"total"', '"_items"', '"users"')):
                        _any_blob = True
                        _ctx = response_msg[max(0, _bstart - 40):_bstart]
                        _parts.append(response_msg[_last_end:_bstart])
                        _parts.append(_replace_json_blob(_blob, _ctx))
                        _last_end = _bend
                if _any_blob:
                    _parts.append(response_msg[_last_end:])
                    response_msg = "".join(_parts)
                else:
                    # Fallback: handle single trailing JSON blob (original behaviour)
                    import re as _re_trail
                    _m = _re_trail.search(r"(\{[\s\S]*\}|\[[\s\S]*\])\s*$", response_msg)
                    if _m:
                        prefix = response_msg[: _m.start()].rstrip().rstrip(":").rstrip()
                        formatted = _format_result_for_display(_m.group(1))
                        if formatted == "No issues found.":
                            if prefix and "\n" in prefix:
                                response_msg = prefix
                            elif prefix:
                                response_msg = _no_results_sentence(prefix)
                            else:
                                response_msg = "No issues found."
                        elif prefix:
                            response_msg = f"{prefix}:\n\n{formatted}"
                        else:
                            response_msg = formatted

            # --- Inject Google Meet links for any unresolved {{...hangoutLink...}} placeholders ---
            import re as _re_meet
            if _re_meet.search(r"\{\{[^}]*(hangout|meet)[^}]*\}\}", response_msg, _re_meet.I):
                _meet_link = ""
                for _sid, _sres in results_by_id.items():
                    if isinstance(_sres, dict) and _sres.get("status") == "success":
                        _rraw = _sres.get("result", "")
                        if isinstance(_rraw, str):
                            try:
                                _rraw = json.loads(_rraw)
                            except Exception:
                                pass
                        if isinstance(_rraw, dict):
                            _meet_link = _rraw.get("hangoutLink") or _rraw.get("meet_link") or _rraw.get("meetLink") or ""
                        if not _meet_link and isinstance(_rraw, str):
                            _mlm2 = _re_meet.search(r"https://meet\.google\.com/[a-z0-9\-]+", _rraw)
                            if _mlm2:
                                _meet_link = _mlm2.group(0)
                        if _meet_link:
                            break
                if _meet_link:
                    response_msg = _re_meet.sub(r"\{\{[^}]*(hangout|meet)[^}]*\}\}", _meet_link, response_msg, flags=_re_meet.I)
                else:
                    # Remove the unresolved placeholder cleanly
                    response_msg = _re_meet.sub(r"\{\{[^}]*(hangout|meet)[^}]*\}\}", "(Google Meet link not available)", response_msg, flags=_re_meet.I)

            # --- Cleanup Gmail Search Output ---
            if (
                "Found " in response_msg
                and "📧 MESSAGES:" in response_msg
                and "Message ID:" in response_msg
            ):
                import re

                # Extract just the count part: "Found X messages matching 'query':"
                header_match = re.search(r"(Found \d+ messages matching '[^']+':)", response_msg)
                if header_match:
                    count_header = header_match.group(1)
                    # If it's a "how many" intent (no specific message asked for), just show the count.
                    # Or we can just strip out the ugly URLs and Thread IDs.
                    response_msg = response_msg[: response_msg.find("📧 MESSAGES:")].strip()
                    response_msg += (
                        f"\n(Run 'read my unread emails' if you want to interact with them)"
                    )
        else:
            response_msg = raw_msg

        # --- Append pagination notice if more results available ---
        _last_list = flow_data.get("_last_list_result")
        if _last_list and _last_list.get("has_more"):
            _shown = _last_list.get("shown", 0)
            _total = _last_list.get("total")
            if _total and _total > _shown:
                response_msg += f"\n\n💡 Showing {_shown} of {_total} results. Type 'show more' to see the next page."
            elif _shown > 0:
                response_msg += f"\n\n💡 Showing first {_shown} results. Type 'show more' to see the next page."

        return {
            "step_results": state.get("step_results", [])
            + [
                {
                    "step_id": step_id,
                    "step_index": idx,
                    "tool": "direct_response",
                    "status": "success",
                    "result": response_msg,
                    "error": "",
                    "duration_ms": 0,
                    "retries": 0,
                }
            ],
            "steps_taken": steps_taken + 1,
            "estimated_cost": cost,
            "status": "verifying",
            "messages": [AIMessage(content=response_msg)],
        }

    # Check guardrails
    from dqe_agent.config import settings

    if steps_taken >= settings.max_steps:
        return {
            "status": "failed",
            "error": f"Max steps ({settings.max_steps}) reached",
            "messages": [
                AIMessage(content=f"Stopped: reached {settings.max_steps}-step safety limit.")
            ],
        }

    # Resolve template references in params using flow_data and step_results
    flow_data = state.get("flow_data", {})
    step_results = state.get("step_results", [])
    results_by_id = {r.get("step_id", ""): r for r in step_results if isinstance(r, dict)}

    # ── Pre-resolve request_selection.options BEFORE template resolution ──────
    # If the planner put a {{ref}} template in options, resolve it by looking up
    # the EXACT referenced step first. Only fall back to scanning recent results
    # if that specific step is not found — prevents picking the wrong step's data.
    _INTERACTION_TOOLS = {"request_selection", "ask_user", "human_review_request"}
    if (
        tool_name == "request_selection"
        and isinstance(params.get("options"), str)
        and "{{" in params["options"]
    ):
        import re as _re_temp

        _templ_m = _re_temp.match(r"^\{\{(.+?)\}\}$", params["options"].strip())
        _resolved_sel_opts = None

        if _templ_m:
            _ref_full = _templ_m.group(1).strip()          # e.g. "get_roles" or "fetch.boards"
            _ref_step = _ref_full.split(".")[0]             # step id portion
            _ref_field = _ref_full.split(".", 1)[1] if "." in _ref_full else None

            # ── Priority 1: look up the exact referenced step in results_by_id ──
            _sr = results_by_id.get(_ref_step)
            _ref_step_exists = _sr is not None
            if _sr and _sr.get("status") == "success":
                _raw = _sr.get("result", "")
                if isinstance(_raw, str):
                    try:
                        _raw = json.loads(_raw)
                    except (json.JSONDecodeError, TypeError):
                        _raw = None
                if _raw is not None:
                    if _ref_field and isinstance(_raw, dict) and _ref_field in _raw:
                        _raw = _raw[_ref_field]
                    items = (
                        _raw if isinstance(_raw, list)
                        else _extract_items_from_response(_raw)
                    )
                    _resolved_sel_opts = _items_to_options(items) if items else []
                    logger.info(
                        "[EXECUTOR] Pre-resolved request_selection options: %d items from step '%s'",
                        len(_resolved_sel_opts), _ref_step,
                    )

            # ── Priority 2: scan recent results only when exact step was never run ──
            # If the referenced step ran but failed, do NOT substitute data from a
            # different step — that produces the wrong option type (e.g. user accountIds
            # used as role IDs when get_roles returned 401).
            if _resolved_sel_opts is None and not _ref_step_exists:
                _suffix = _ref_field
                for sr in reversed(step_results):
                    if not isinstance(sr, dict):
                        continue
                    if sr.get("tool") in _INTERACTION_TOOLS:
                        continue
                    if sr.get("status") != "success":
                        continue
                    # Skip the step the template references if it succeeded but
                    # returned nothing useful — don't pick a completely different step
                    # unless it has value/label options.
                    raw = sr.get("result", "")
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    items = []
                    if _suffix and isinstance(raw, dict) and _suffix in raw and isinstance(raw[_suffix], list):
                        items = raw[_suffix]
                    else:
                        items = _extract_items_from_response(raw)
                    # Only use this fallback result if it has proper {value,label} options
                    opts = _items_to_options(items)
                    if opts and all(o.get("value") and o.get("label") for o in opts[:3]):
                        _resolved_sel_opts = opts
                        logger.info(
                            "[EXECUTOR] Pre-resolved request_selection options (fallback): %d items from step '%s'",
                            len(opts), sr.get("step_id", "?"),
                        )
                        break

        # If we found concrete options, replace; if the referenced step ran but
        # returned no items, DO NOT replace the original template string — this
        # preserves the explicit-ref marker so later logic will NOT fall back to
        # unrelated step results.
        if _resolved_sel_opts:
            params = dict(params)
            params["options"] = _resolved_sel_opts
        else:
            if _templ_m and _resolved_sel_opts == []:
                logger.warning(
                    "[EXECUTOR] request_selection: referenced step '%s' returned no options — leaving template for explicit-ref handling",
                    _ref_step,
                )

    # ── Pre-resolve request_form field options BEFORE template resolution ──────
    # _resolve_params only handles top-level strings; it won't recurse into the
    # nested field dicts.  Scan each field's options string for {{ref}} templates
    # and resolve them the same way as request_selection.options above.
    if tool_name == "request_form" and isinstance(params.get("fields"), list):
        import re as _re_form

        _pure_ref_form = _re_form.compile(r"^\{\{(.+?)\}\}$")
        resolved_fields = []
        for _field in params["fields"]:
            if not isinstance(_field, dict):
                resolved_fields.append(_field)
                continue
            _field = dict(_field)
            _opts = _field.get("options")
            if isinstance(_opts, str) and _opts.strip().startswith("<<"):
                _resolved_pre = _resolve_prefetched_sentinel(_opts.strip())
                if _resolved_pre:
                    _field["options"] = _resolved_pre
                    logger.info(
                        "[EXECUTOR] Pre-resolved form field '%s' options from sentinel: %d items",
                        _field.get("id", "?"),
                        len(_resolved_pre),
                    )
            if isinstance(_opts, str) and "{{" in _opts:
                _m = _pure_ref_form.match(_opts.strip())
                if _m:
                    _ref = _m.group(1).strip()
                    _obj = _resolve_ref_to_object(_ref, flow_data, results_by_id)
                    if _obj is None:
                        # Fallback: scan recent non-interaction step results
                        _suffix_f = _ref.split(".")[-1] if "." in _ref else None
                        for sr in reversed(step_results):
                            if not isinstance(sr, dict) or sr.get("tool") in _INTERACTION_TOOLS:
                                continue
                            if sr.get("status") != "success":
                                continue
                            _raw = sr.get("result", "")
                            if isinstance(_raw, str):
                                try:
                                    _raw = json.loads(_raw)
                                except (json.JSONDecodeError, TypeError):
                                    continue
                            if (
                                isinstance(_raw, dict)
                                and _suffix_f
                                and _suffix_f in _raw
                                and isinstance(_raw[_suffix_f], list)
                            ):
                                _obj = _raw[_suffix_f]
                            else:
                                _obj = _extract_items_from_response(_raw)
                            if _obj:
                                break
                    if _obj is not None:
                        _resolved_opts = (
                            _items_to_options(_obj)
                            if not (
                                isinstance(_obj, list)
                                and _obj
                                and isinstance(_obj[0], dict)
                                and "value" in _obj[0]
                                and "label" in _obj[0]
                            )
                            else _obj
                        )
                        # For member/assignee fields, prepend "All members" and "My hours only"
                        _fid = _field.get("id", "")
                        if any(k in _fid.lower() for k in ("member", "assignee")):
                            _defaults = [
                                {"value": "__all__", "label": "All members"},
                                {"value": "__me__", "label": "My hours only"},
                            ]
                            _resolved_opts = _defaults + [
                                o for o in _resolved_opts
                                if o.get("value") not in ("__all__", "__me__")
                            ]
                        _field["options"] = _resolved_opts
                        logger.info(
                            "[EXECUTOR] Pre-resolved form field '%s' options: %d items",
                            _field.get("id", "?"),
                            len(_field["options"]),
                        )
            resolved_fields.append(_field)
        params = dict(params)
        params["fields"] = resolved_fields


    resolved_params = _resolve_params(params, flow_data, results_by_id)

    # Strip MCP-internal params that the framework injects — callers must not pass them
    for _internal in ("ctx", "kwargs"):
        resolved_params.pop(_internal, None)

    # Warn and STRIP params that still contain unresolved {{}} templates.
    # Passing raw template strings to tools causes Pydantic validation errors.
    _unresolved = [k for k, v in resolved_params.items() if isinstance(v, str) and "{{" in v]
    if _unresolved:
        logger.warning(
            "[EXECUTOR] Unresolved template params for step '%s': %s — stripping before tool call",
            step_id,
            _unresolved,
        )
        for _uk in _unresolved:
            resolved_params.pop(_uk, None)

    logger.info(
        "[EXECUTOR] Resolved params for '%s': %s",
        step_id,
        json.dumps(resolved_params, default=str)[:500],
    )

    resolved_params = await _normalize_tool_params(tool_name, resolved_params, flow_data, results_by_id)

    # ── Person-not-found: abort plan and tell user when name has 0 matches ──────
    if tool_name == "request_selection" and resolved_params.get("_person_not_found"):
        _not_found_name = resolved_params["_person_not_found"]
        # Try to figure out which project was selected from recent step results
        _proj_label = ""
        for _sr in reversed(state.get("step_results", [])):
            try:
                _sr_parsed = json.loads(_sr.get("result", "{}")) if isinstance(_sr.get("result"), str) else (_sr.get("result") or {})
                _sel = _sr_parsed.get("selected", "") if isinstance(_sr_parsed, dict) else ""
                if _sel and len(_sel) <= 20:  # project keys / names are short
                    _proj_label = f" in the selected project ({_sel})"
                    break
            except Exception:
                pass
        _msg = f"**{_not_found_name}** was not found{_proj_label}. Please check the name or try a different project."
        logger.info("[EXECUTOR] request_selection: person not found — aborting with message: %s", _msg)
        return {
            "step_results": state.get("step_results", []) + [{
                "step_id": step_id,
                "step_index": idx,
                "tool": "direct_response",
                "status": "success",
                "result": _msg,
                "error": "",
                "duration_ms": 0,
                "retries": 0,
            }],
            "steps_taken": steps_taken + 1,
            "estimated_cost": cost,
            "status": "complete",
            "messages": [AIMessage(content=_msg)],
            "flow_data": dict(state.get("flow_data", {})),
        }

    # ── Auto-select: skip interrupt when request_selection has exactly 1 option ──
    if tool_name == "request_selection":
        opts = resolved_params.get("options", [])
        if isinstance(opts, list) and len(opts) == 1:
            auto_val = opts[0].get("value", "")
            auto_label = opts[0].get("label", auto_val)
            logger.info(
                "[EXECUTOR] request_selection: single option — auto-selecting %r", auto_val
            )
            result_json = json.dumps({"selected": auto_val, "answer": auto_val})
            merged_flow = dict(state.get("flow_data", {}))
            merged_flow[step_id] = {"selected": auto_val, "answer": auto_val}
            return {
                "step_results": state.get("step_results", [])
                + [
                    {
                        "step_id": step_id,
                        "step_index": idx,
                        "tool": tool_name,
                        "status": "success",
                        "result": result_json,
                        "error": "",
                        "duration_ms": 0,
                        "retries": 0,
                    }
                ],
                "steps_taken": steps_taken + 1,
                "estimated_cost": cost,
                "status": "verifying",
                "flow_data": merged_flow,
                "messages": [AIMessage(content=f"Auto-selected: {auto_label}")],
            }

    # Special handling for jira_delete_issue with multiple issue keys
    if tool_name == "jira_delete_issue" and "issue_key" in resolved_params:
        issue_key_param = resolved_params["issue_key"]
        if isinstance(issue_key_param, str):
            try:
                # Try to parse as JSON array
                parsed_keys = json.loads(issue_key_param)
                if isinstance(parsed_keys, list):
                    if len(parsed_keys) > 1:
                        logger.info(
                            "[EXECUTOR] jira_delete_issue: detected %d issue keys, processing individually",
                            len(parsed_keys),
                        )

                        # Execute multiple delete operations
                        start = time.time()
                        results = []
                        errors = []

                        for issue_key in parsed_keys:
                            single_params = resolved_params.copy()
                            # Recover full Jira key (e.g. "FLAG-148756") from numeric ID if needed
                            single_params["issue_key"] = _recover_jira_issue_key(
                                issue_key, flow_data, results_by_id
                            )
                            try:
                                tool = get_tool(tool_name)
                                if tool is None:
                                    raise ValueError(f"Tool '{tool_name}' not found")

                                result_raw = await tool.ainvoke(single_params)
                                result_raw = _unwrap_mcp_result(result_raw)
                                results.append(f"Deleted {issue_key}: {str(result_raw)}")
                            except Exception as exc:
                                import traceback as _tb

                                tb_str = "".join(
                                    _tb.format_exception(type(exc), exc, exc.__traceback__)
                                )
                                error_msg = f"Failed to delete {issue_key}: {str(exc)}\nTraceback:\n{tb_str}"
                                logger.error(
                                    "[EXECUTOR] Delete failed for %s: %s\n%s",
                                    issue_key,
                                    str(exc),
                                    tb_str,
                                )
                                errors.append(error_msg)

                        duration_ms = int((time.time() - start) * 1000)
                        all_results = results + errors
                        combined_result = "; ".join(all_results)

                        # Return combined result
                        step_result = {
                            "step_id": step_id,
                            "step_index": idx,
                            "tool": tool_name,
                            "status": "success" if not errors else "partial",
                            "result": combined_result,
                            "error": "; ".join(errors) if errors else "",
                            "duration_ms": duration_ms,
                            "retries": 0,
                        }

                        return {
                            "step_results": state.get("step_results", []) + [step_result],
                            "steps_taken": steps_taken + 1,
                            "estimated_cost": cost,
                            "status": "verifying",
                            "flow_data": dict(state.get("flow_data", {})),
                            "messages": [
                                AIMessage(
                                    content=f"Delete operation completed: {len(results)} successful, {len(errors)} failed"
                                )
                            ],
                        }
                    elif len(parsed_keys) == 1:
                        # Single item from JSON array: extract it and recover full key if needed
                        recovered = _recover_jira_issue_key(
                            parsed_keys[0], flow_data, results_by_id
                        )
                        logger.info(
                            "[EXECUTOR] jira_delete_issue: extracted single issue key %r → %r",
                            parsed_keys[0],
                            recovered,
                        )
                        resolved_params["issue_key"] = recovered
            except (json.JSONDecodeError, TypeError):
                # Not a JSON array — recover key for plain string too
                recovered = _recover_jira_issue_key(issue_key_param, flow_data, results_by_id)
                if recovered != issue_key_param:
                    logger.info(
                        "[EXECUTOR] jira_delete_issue: recovered issue key %r → %r",
                        issue_key_param,
                        recovered,
                    )
                    resolved_params["issue_key"] = recovered

    # Execute the tool directly
    start = time.time()
    status = "success"
    result: Any = None
    error_msg = ""

    try:
        tool = get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")
    except KeyError:
        # Tool doesn't exist — this is a hard failure, not recoverable.
        try:
            from dqe_agent.tools import list_tool_names

            available = list_tool_names()
        except Exception:
            available = []
        logger.error("[EXECUTOR] Tool '%s' not found — available tools: %s", tool_name, available)

        # Provide helpful hint for common misnamed tools
        hint = ""
        if tool_name == "jira_get_active_sprints":
            hint = " (hint: use 'jira_get_sprints_from_board' with state='active' after selecting a board)"
        elif tool_name == "jira_delete_comment":
            hint = " (note: jira_delete_comment may not be available in your mcp-atlassian version)"

        error_msg = f"Tool '{tool_name}' is not available in this environment.{hint}"
        return {
            "step_results": state.get("step_results", [])
            + [
                {
                    "step_id": step_id,
                    "step_index": idx,
                    "tool": tool_name,
                    "status": "failed",
                    "result": "",
                    "error": error_msg,
                    "duration_ms": 0,
                    "retries": 0,
                }
            ],
            "steps_taken": steps_taken + 1,
            "estimated_cost": cost,
            "status": "verifying",
            "flow_data": dict(state.get("flow_data", {})),
            "messages": [AIMessage(content=f"Error: {error_msg}")],
        }

    try:
        # Sanitize JQL sentinel values before invoking Jira search tools so we
        # never pass invalid JQL like "project = AMA AND __ALL_TIME__".
        if isinstance(resolved_params.get("jql"), str):
            import re

            jql = resolved_params["jql"]
            if "__ALL_TIME__" in jql:
                # Remove any surrounding AND/OR + sentinel token
                jql = re.sub(r"\b(AND|OR)\s+__ALL_TIME__\b", "", jql, flags=re.IGNORECASE)
                jql = re.sub(r"__ALL_TIME__\s+\b(AND|OR)\b", "", jql, flags=re.IGNORECASE)
                jql = jql.replace("__ALL_TIME__", "").strip()
                # Trim trailing conjunctions
                jql = re.sub(r"\b(AND|OR)\s*$", "", jql, flags=re.IGNORECASE).strip()
                resolved_params["jql"] = jql

            if "__CUSTOM_RANGE__" in jql:
                try:
                    from langgraph.types import interrupt

                    form_result = interrupt(
                        {
                            "type": "form",
                            "title": "Custom date range",
                            "fields": [
                                {"id": "start_date", "label": "Start date", "type": "date", "required": True},
                                {"id": "end_date", "label": "End date", "type": "date", "required": True},
                            ],
                        }
                    )
                    if isinstance(form_result, str):
                        try:
                            form_vals = json.loads(form_result)
                        except Exception:
                            form_vals = {}
                    else:
                        form_vals = form_result or {}

                    sd = form_vals.get("start_date")
                    ed = form_vals.get("end_date")
                    if sd and ed:
                        # Ensure dates are simple YYYY-MM-DD strings for JQL
                        # Users' inputs should be ISO dates already from the form
                        clause = f"worklogDate >= '{sd}' AND worklogDate <= '{ed}'"
                        jql = jql.replace("__CUSTOM_RANGE__", clause)
                        resolved_params["jql"] = jql
                    else:
                        # If user didn't provide both dates, strip the sentinel
                        resolved_params["jql"] = jql.replace("__CUSTOM_RANGE__", "")
                except Exception:
                    # On any failure, remove the sentinel to avoid invalid JQL
                    resolved_params["jql"] = jql.replace("__CUSTOM_RANGE__", "")

        # ── Busy-slot gate for create_event ─────────────────────────────────
        # Check the entire start→end window for overlap with any existing event.
        if tool_name in ("create_event", "manage_event"):
            _create_start = str(resolved_params.get("start_time", ""))
            _create_end = str(resolved_params.get("end_time", ""))
            if _create_start and _create_end:
                _all_ev_text = "\n".join(
                    _sr.get("result", "") for _sr in step_results
                    if isinstance(_sr, dict) and _sr.get("tool") == "get_events"
                    and isinstance(_sr.get("result"), str)
                )
                if _all_ev_text.strip():
                    from datetime import datetime, timedelta, timezone
                    import re as _re_bsg
                    IST = timezone(timedelta(hours=5, minutes=30))
                    try:
                        _st = datetime.fromisoformat(_create_start).astimezone(timezone.utc)
                        _et = datetime.fromisoformat(_create_end).astimezone(timezone.utc)

                        # Parse ALL busy periods from the collected events text
                        _bsg_ts = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)"

                        def _bsg_parse(s: str) -> datetime:
                            s = s.strip()
                            if s.endswith("Z"):
                                return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
                            return datetime.fromisoformat(s).astimezone(timezone.utc)

                        _conflicting: list[tuple[str, datetime, datetime]] = []
                        for _line in _all_ev_text.splitlines():
                            _bm = _re_bsg.search(
                                r'["\-]\s*(.*?)\s*"?\s*\([Ss]tarts?:\s*' + _bsg_ts + r'[^|\n]*?[Ee]nds?:\s*' + _bsg_ts,
                                _line,
                            )
                            if _bm:
                                try:
                                    _ev_name = _bm.group(1).strip(' "')
                                    _b_s = _bsg_parse(_bm.group(2))
                                    _b_e = _bsg_parse(_bm.group(3))
                                    # Check overlap: new meeting window vs this event
                                    if not (_et <= _b_s or _st >= _b_e):
                                        _conflicting.append((_ev_name, _b_s, _b_e))
                                except Exception:
                                    continue

                        if _conflicting:
                            _conflict_lines = "\n".join(
                                f"• {_n} ({_bs.astimezone(IST).strftime('%I:%M %p').lstrip('0')} – "
                                f"{_be.astimezone(IST).strftime('%I:%M %p').lstrip('0')} IST)"
                                for _n, _bs, _be in _conflicting
                            )
                            _st_label = _st.astimezone(IST).strftime("%I:%M %p").lstrip("0") + " IST"
                            _et_label = _et.astimezone(IST).strftime("%I:%M %p").lstrip("0") + " IST"
                            logger.info(
                                "[EXECUTOR] create_event: %d conflict(s) for %s–%s — triggering confirm gate",
                                len(_conflicting), _st_label, _et_label,
                            )
                            _busy_answer = interrupt({
                                "question": (
                                    f"⚠️ The time **{_st_label} – {_et_label}** conflicts with:\n\n"
                                    f"{_conflict_lines}\n\nProceed and create the meeting anyway?"
                                ),
                                "type": "human_review",
                            })
                            if isinstance(_busy_answer, str) and _busy_answer.strip().lower() in (
                                "no", "n", "cancel", "stop", "nope",
                            ):
                                return {
                                    "status": "complete",
                                    "messages": [AIMessage(content="Meeting creation cancelled.")],
                                }
                    except Exception as _bge:
                        logger.warning("[EXECUTOR] busy-slot gate error (skipping): %s", _bge)

        result_raw = await tool.ainvoke(resolved_params)
        # Unwrap LangChain MCP content blocks so downstream code (pre-resolve,
        # flow_data storage, template resolution) always sees real JSON-serialisable data.
        result_raw = _unwrap_mcp_result(result_raw)
        # Log raw result for write-action tools so silent errors are visible in logs
        if tool_name in (
            "jira_transition_issue", "transition_issue",
            "jira_update_issue", "update_issue",
            "jira_create_issue", "create_issue",
        ):
            logger.info("[EXECUTOR] %s raw result: %s", tool_name, str(result_raw)[:400])

        def _serialize(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        if isinstance(result_raw, dict):
            # If human_review came back rejected, abort remaining steps
            hr = result_raw.get("human_review")
            if hr is not None and hasattr(hr, "approved") and not hr.approved:
                logger.info(
                    "[EXECUTOR] Human review rejected at step %s — stopping workflow", step_id
                )
                return {
                    "status": "complete",
                    "step_results": state.get("step_results", [])
                    + [
                        {
                            "step_id": step_id,
                            "step_index": idx,
                            "tool": tool_name,
                            "status": "rejected",
                            "result": "User rejected — workflow stopped",
                            "error": "",
                            "duration_ms": 0,
                            "retries": 0,
                        }
                    ],
                    "steps_taken": steps_taken + 1,
                    "estimated_cost": cost,
                    "flow_data": dict(state.get("flow_data", {})),
                    "messages": [
                        AIMessage(
                            content=f"Workflow stopped at step '{step_id}' — user did not approve."
                        )
                    ],
                }
            # Serialize — Pydantic models (e.g. HumanReview) need model_dump() first
            result = json.dumps(result_raw, default=_serialize)
        elif isinstance(result_raw, list):
            # Serialize lists as proper JSON (not Python repr which uses single quotes)
            try:
                result = json.dumps(result_raw, default=_serialize)
            except (TypeError, ValueError):
                result = str(result_raw)
        elif result_raw is not None:
            result = str(result_raw)
        else:
            result = "done"

    except GraphInterrupt:
        # interrupt() is a pause signal, not an error — must bubble up to the
        # graph runner so api.py's astream loop sees event["__interrupt__"].
        raise

    except Exception as exc:
        status = "failed"
        cause = exc.__cause__ or exc.__context__
        error_msg = str(exc) + (f" — caused by: {cause}" if cause else "")
        logger.error("[EXECUTOR] Step %s failed: %s", step_id, error_msg, exc_info=True)

    duration = (time.time() - start) * 1000
    step_cost = COST_PER_CALL.get("executor", 0.002)

    # Trace the tool execution
    from dqe_agent.observability import trace_tool_call

    trace_tool_call(
        tool=tool_name,
        args=resolved_params,
        result_status=status,
        duration_ms=round(duration, 1),
        session_id=state.get("session_id", ""),
    )

    step_result = {
        "step_id": step_id,
        "step_index": idx,
        "tool": tool_name,
        "status": status,
        "result": result,
        "error": error_msg,
        "duration_ms": round(duration, 1),
        "retries": state.get("retry_count", 0),
    }

    # Store extracted data into flow_data — merge with existing so earlier steps' data is preserved
    merged_flow = dict(state.get("flow_data", {}))
    if status == "success" and result:
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict):
                # For create/update tools, hoist nested issue fields to the top level so that
                # {{step.key}}, {{step.id}}, {{step.summary}} resolve without path errors.
                if tool_name in (
                    "jira_create_issue", "create_issue",
                    "jira_update_issue", "update_issue",
                    "jira_transition_issue", "transition_issue",
                ):
                    _nested = parsed.get("issue", {})
                    if isinstance(_nested, dict):
                        for _hk in ("key", "id", "summary", "url", "status"):
                            if _nested.get(_hk) and _hk not in parsed:
                                parsed[_hk] = _nested[_hk]
                    # Persist the enriched dict back so results_by_id is also up-to-date
                    result = json.dumps(parsed, default=str)
                    step_result["result"] = result
                merged_flow[step_id] = parsed
            elif isinstance(parsed, list):
                # Wrap lists so template references like {{step._items}} work.
                wrapped = {"_items": parsed, "_list": parsed}
                # Hoist key/id from first issue element for create_issue list responses
                if tool_name in ("jira_create_issue", "create_issue") and parsed:
                    _issue = parsed[0].get("issue", {}) if isinstance(parsed[0], dict) else {}
                    if isinstance(_issue, dict):
                        for _k in ("key", "id", "summary"):
                            if _issue.get(_k):
                                wrapped[_k] = _issue[_k]
                merged_flow[step_id] = wrapped
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Pagination tracking: persist last list-result so the planner can page ──
    # When a search/list tool returns results, write a _last_list_result summary
    # into flow_data. On the next user turn ("show more", "next 50"), the planner
    # sees this in AVAILABLE DATA and generates a plan with the correct start_at.
    _LIST_TOOLS = {
        "jira_search", "search_issues",
        "jira_get_sprint_issues", "get_sprint_issues",
    }
    if status == "success" and tool_name in _LIST_TOOLS:
        try:
            _raw = result if isinstance(result, str) else json.dumps(result, default=str)
            _parsed = json.loads(_raw) if isinstance(_raw, str) else _raw
            _issues = []
            _total = None
            _start_at = int(params.get("start_at", 0))
            _total_is_real = False  # True when Jira returned an actual count (not -1)
            if isinstance(_parsed, dict):
                _issues = _parsed.get("issues") or _parsed.get("_items") or []
                _raw_total = _parsed.get("total")
                if _raw_total is not None and _raw_total >= 0:
                    _total = _raw_total
                    _total_is_real = True
                else:
                    # Jira Cloud returns -1 for unknown totals — use len as fallback display value
                    _total = len(_issues)
            elif isinstance(_parsed, list):
                _issues = _parsed
                _total = len(_issues)
            _shown = len(_issues)
            _limit = int(params.get("limit", 50))
            _next_start = _start_at + _shown
            # Jira Cloud cursor pagination: nextPageToken is the authoritative signal.
            # jira_search also accepts page_token param (Cloud only).
            _next_page_token = _parsed.get("nextPageToken") if isinstance(_parsed, dict) else None
            # has_more priority:
            # 1. nextPageToken present → definitely more (cursor pagination)
            # 2. real total known → compare next_start vs total
            # 3. unknown total: full first page may have more; subsequent pages → done
            if _next_page_token:
                _has_more = True
            elif _total_is_real:
                _has_more = _next_start < _total
            else:
                # total=-1 (unknown): no nextPageToken means Jira has no more pages
                _has_more = False
            _pagination_summary = {
                "tool": tool_name,
                "query": params.get("jql") or params.get("sprint_id") or "",
                "project": params.get("project_key") or "",
                "sprint_id": params.get("sprint_id") or "",
                "start_at": _start_at,
                "limit": _limit,
                "shown": _shown,
                "total": _total,
                "next_start_at": _next_start,
                "has_more": _has_more,
                "next_page_token": _next_page_token or "",
                # page_token to pass directly to jira_search on next call
                "page_token_for_next": _next_page_token or "",
            }
            merged_flow["_last_list_result"] = _pagination_summary
            logger.info(
                "[EXECUTOR] Pagination state: shown=%d, total=%s, next_start_at=%d, has_more=%s, page_token=%s",
                _shown, _total, _next_start, _has_more, bool(_next_page_token),
            )
        except Exception:
            pass

    extra_results: list = []

    # ---- Executor-side fallback: if planner asked for project/period via a
    # request_form but did NOT collect team members, and the original task
    # mentioned members/team, prompt the user now with a multi-select of
    # assignable users for the selected project.
    try:
        if tool_name == "request_form":
            task_text = str(state.get("task", "")).lower()
            if any(k in task_text for k in ("member", "members", "team", "assignee", "assignees")):
                # Check whether the form already provided member data
                provided = merged_flow.get(step_id, {}) or {}
                has_members = any(k for k in provided.keys() if any(s in k.lower() for s in ("member", "assignee", "assignees", "team")))
                if not has_members:
                    # try to infer project key from the form results or flow_data
                    proj = None
                    # common field names used in forms
                    for candidate in ("project_key", "project", "proj", "projectId"):
                        if candidate in provided and provided.get(candidate):
                            proj = str(provided.get(candidate)).strip().upper()
                            break
                    if not proj:
                        proj = _recover_project_key(results_by_id, flow_data)
                    if proj:
                        try:
                            users = await _fetch_assignable_users(proj)
                            if users:
                                from dqe_agent.tools import get_tool
                                sel_tool = get_tool("request_selection")
                                sel_params = {
                                    "question": "Which team members should I include? You can pick one or more.",
                                    "options": users,
                                    "multi_select": True,
                                }
                                sel_raw = await sel_tool.ainvoke(sel_params)
                                sel_res = _unwrap_mcp_result(sel_raw)
                                # Merge selection into flow under a clear key
                                sel_key = f"{step_id}_members"
                                merged_flow[sel_key] = sel_res
                                # Append a synthetic step_result so downstream sees it
                                extra_result = {
                                    "step_id": sel_key,
                                    "step_index": idx + 0.1,
                                    "tool": "request_selection",
                                    "status": "success",
                                    "result": json.dumps(sel_res, default=str),
                                    "error": "",
                                    "duration_ms": 0,
                                    "retries": 0,
                                }
                                extra_results.append(extra_result)
                                # Store merged_flow back to state variable for return
                        except Exception:
                            logger.debug("[EXECUTOR] follow-up member selection failed or no users available")
    except Exception:
        pass

    msg_content = _step_message(
        idx, len(plan), tool_name, status, error_msg, resolved_params, result
    )

    return {
        "step_results": state.get("step_results", []) + [step_result] + extra_results,
        "steps_taken": steps_taken + 1,
        "estimated_cost": cost + step_cost,
        "status": "verifying",
        "flow_data": merged_flow,
        "messages": [AIMessage(content=msg_content)],
    }


def _resolve_prefetched_sentinel(sentinel: str) -> list | None:
    """Resolve <<SENTINEL>> tokens from planner cache (sync, cache-only)."""
    s = sentinel.strip()
    if s == "<<JIRA_PROJECTS_PREFETCHED>>":
        try:
            from dqe_agent.agent.planner import _cache_get
            return _cache_get("jira_projects")
        except Exception:
            return None
    return None


def _first_project_key() -> str:
    """Return the first Jira project key from planner cache (used before user picks a project)."""
    try:
        from dqe_agent.agent.planner import _cache_get

        projects = _cache_get("jira_projects")
        if projects and isinstance(projects, list) and projects:
            first = projects[0]
            if isinstance(first, dict):
                return str(first.get("value") or first.get("key") or "")
    except Exception:
        pass
    return ""


def _extract_ask_user_answer(step_result: dict) -> str:
    """Pull the 'answer' value out of an ask_user step result dict."""
    raw = step_result.get("result", "")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw  # plain string answer
    if isinstance(raw, dict):
        return str(raw.get("answer", ""))
    return ""


def _items_to_options(items: list) -> list[dict[str, str]]:
    """Convert a list of arbitrary objects into [{value, label}] for request_selection.

    Handles Jira boards, sprints, projects, users, and any generic id/name dicts.
    Prefers 'key' (Jira issue/project key) over numeric 'id' when both exist.
    """
    options: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            s = str(item)
            options.append({"value": s, "label": s})
            continue
        # Already formatted
        if "value" in item and "label" in item:
            options.append({"value": str(item["value"]), "label": str(item["label"])})
            continue
        # Jira board / sprint / Google Calendar: {id, name/displayName/summary, type/state}
        # For Jira users the VALUE must be the display name (not the accountId), because
        # mcp-atlassian resolves assignees via _get_account_id() which searches by name/email.
        # Passing a raw accountId string causes the lookup to fail silently.
        # For Jira issues and projects, prefer the 'key' field (e.g. 'FLAG-123') over numeric 'id'.
        # Boards/sprints have 'boardId' or 'id' but not 'key'.
        _is_user = "accountId" in item or "account_id" in item or "displayName" in item or "display_name" in item
        if _is_user:
            logger.info("[ITEMS_TO_OPTIONS] user object keys=%s raw=%s", list(item.keys()), str(item)[:300])
            # Use display name as the value — MCP resolves names to accountIds internally
            name = str(
                item.get("displayName") or item.get("display_name")
                or item.get("name") or item.get("emailAddress") or item.get("email_address") or ""
            ).strip()
            logger.info("[ITEMS_TO_OPTIONS] user name resolved='%s'", name)
            item_id = name  # value == label for users
        else:
            item_id = str(
                item.get("key")             # Jira project/issue key
                or item.get("boardId")      # Jira board
                or item.get("id")           # generic id
                or ""
            ).strip()
            name = str(item.get("name", item.get("displayName", item.get("summary", item_id)))).strip()
        extra = item.get("type") or item.get("state") or item.get("projectTypeKey") or ""
        label = f"{name} ({extra})" if extra else name
        if item_id:
            options.append({"value": item_id, "label": label})
        elif name:
            options.append({"value": name, "label": name})
    return options


def _extract_items_from_response(raw: Any) -> list:
    """Pull a usable list out of whatever a Jira/MCP tool returned."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # Try common wrapper keys
        for k in (
            "values",
            "boards",
            "sprints",
            "items",
            "data",
            "results",
            "issues",
            "projects",
            "_items",
            "_list",
        ):
            if k in raw and isinstance(raw[k], list):
                return raw[k]
        # Single-object dict — wrap it
        if raw:
            return [raw]
    return []


def _strip_invalid_params(tool_name: str, params: dict) -> dict:
    """Remove params the tool's schema doesn't declare.

    Prevents 'unexpected keyword argument' errors when the planner invents
    param names (e.g. board_filter, auth_check, ctx) that don't exist.
    Falls back to the original params if the schema cannot be read.
    """
    from dqe_agent.tools import get_tool as _get_tool

    try:
        tool_obj = _get_tool(tool_name)
    except KeyError:
        return params
    if tool_obj is None:
        return params

    try:
        schema = {}
        if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
            schema = tool_obj.args_schema.model_json_schema()
        elif hasattr(tool_obj, "args") and isinstance(tool_obj.args, dict):
            schema = {"properties": tool_obj.args}

        props = schema.get("properties", {})
        valid = set(props.keys()) - {"ctx", "kwargs"}

        if not valid:
            # If no properties found but schema exists, check if it's a root type (rare)
            return params

        invalid = [k for k in params if k not in valid]
        if invalid:
            logger.warning(
                "[EXECUTOR] '%s': stripping unknown params %s (valid keys: %s)",
                tool_name,
                invalid,
                list(valid),
            )
            params = {k: v for k, v in params.items() if k in valid}
    except Exception as e:
        logger.debug("[EXECUTOR] _strip_invalid_params failed for %s: %s", tool_name, e)
        pass
    return params


def _pre_strip_remap(tool_name: str, params: dict, flow_data: dict | None = None, results_by_id: dict | None = None) -> dict:
    """Remap wrong param names to correct ones BEFORE _strip_invalid_params runs.

    _strip_invalid_params removes anything not in the tool's schema. If the planner
    sends 'start' instead of 'start_time', the strip step would delete 'start' before
    tool-specific normalization can remap it. This function runs first.
    """
    params = dict(params)  # don't mutate caller's dict

    # ── project_key: fuzzy-match if LLM guessed a wrong/invented key ────────────
    _pk = params.get("project_key", "")
    if isinstance(_pk, str) and _pk and "{{" not in _pk and not _pk.strip().startswith("__"):
        _matched = _fuzzy_match_project_key_cached(_pk)
        if _matched and _matched.upper() != _pk.strip().upper():
            logger.info(
                "[EXECUTOR] project_key fuzzy matched %r → %r", _pk, _matched
            )
            params["project_key"] = _matched

    # ── human_review: remap 'question' → 'summary', inject default review_type ──
    if tool_name == "human_review":
        if "question" in params and "summary" not in params:
            params["summary"] = params.pop("question")
        if "message" in params and "summary" not in params:
            params["summary"] = params.pop("message")
        if not params.get("review_type"):
            params["review_type"] = "confirm"

    # ── create_event / modify_event / manage_event: fix wrong param names ───────
    if tool_name in ("create_event", "modify_event", "manage_event"):
        for alias in ("title", "name", "event_title", "event_name", "event_summary"):
            if alias in params and "summary" not in params:
                params["summary"] = params.pop(alias)
                break
        for alias in (
            "start",
            "start_datetime",
            "start_date",
            "date_start",
            "start_at",
            "begins_at",
        ):
            if alias in params and "start_time" not in params:
                params["start_time"] = params.pop(alias)
                break
        for alias in ("end", "end_datetime", "end_date", "date_end", "end_at", "ends_at"):
            if alias in params and "end_time" not in params:
                params["end_time"] = params.pop(alias)
                break

    # ── get_events / list_events: fix wrong time param names ─────────────────
    _CALENDAR_QUERY_TOOLS = {
        "get_events",
        "list_events",
        "calendar_get_events",
        "calendar_list_events",
    }
    if tool_name in _CALENDAR_QUERY_TOOLS:
        for alias in ("start_datetime", "start_date", "start", "date_start", "from_date", "from"):
            if alias in params and "time_min" not in params:
                params["time_min"] = params.pop(alias)
                break
        for alias in ("end_datetime", "end_date", "end", "date_end", "to_date", "to"):
            if alias in params and "time_max" not in params:
                params["time_max"] = params.pop(alias)
                break

    # ── jira_search / jira_get_sprint_issues: clamp limit to Jira's max of 50 ────
    if tool_name in ("jira_search", "jira_get_sprint_issues", "get_sprint_issues"):
        if isinstance(params.get("limit"), int) and params["limit"] > 50:
            params["limit"] = 50
            logger.info("[EXECUTOR] %s: clamped limit to 50", tool_name)

    # ── jira_search: fix single-word assignee + inject project from flow_data ────
    if tool_name == "jira_search":
        import re as _re_jql
        _jql = params.get("jql", "")

        # Helper: find a selected project key from flow_data / step results
        def _find_selected_project() -> str:
            _proj_pat = _re_jql.compile(r'^[A-Z][A-Z0-9]{1,9}$')
            for _fd_val in (flow_data or {}).values():
                if isinstance(_fd_val, dict):
                    _sel = str(_fd_val.get("selected") or _fd_val.get("answer") or "")
                    if _proj_pat.match(_sel):
                        return _sel
            for _sr in (results_by_id or {}).values():
                try:
                    _srp = json.loads(_sr.get("result", "{}")) if isinstance(_sr.get("result"), str) else (_sr.get("result") or {})
                    _sel = str(_srp.get("selected") or _srp.get("answer") or "") if isinstance(_srp, dict) else ""
                    if _proj_pat.match(_sel):
                        return _sel
                except Exception:
                    pass
            return ""

        # 1. Expand single-word assignee → full name
        _single_word_m = _re_jql.search(r'assignee\s*=\s*["\']([A-Za-z]+)["\']', _jql)
        if _single_word_m:
            _bare_name = _single_word_m.group(1)
            _bare_lower = _bare_name.lower()
            _full_name: str | None = None
            for _fd_val in (flow_data or {}).values():
                if isinstance(_fd_val, dict):
                    _sel = str(_fd_val.get("selected") or _fd_val.get("answer") or "")
                    if _sel.lower().startswith(_bare_lower) and " " in _sel:
                        _full_name = _sel
                        break
                elif isinstance(_fd_val, str) and _fd_val.lower().startswith(_bare_lower) and " " in _fd_val:
                    _full_name = _fd_val
                    break
            if not _full_name:
                for _sr in (results_by_id or {}).values():
                    try:
                        _srp = json.loads(_sr.get("result", "{}")) if isinstance(_sr.get("result"), str) else (_sr.get("result") or {})
                        _sel = str(_srp.get("selected") or _srp.get("answer") or "") if isinstance(_srp, dict) else ""
                        if _sel.lower().startswith(_bare_lower) and " " in _sel:
                            _full_name = _sel
                            break
                    except Exception:
                        pass
            if _full_name:
                _jql = _re_jql.sub(f'assignee = "{_full_name}"', _jql, count=1)
                logger.info("[EXECUTOR] jira_search: expanded single-word assignee %r → %r", _bare_name, _full_name)
            else:
                logger.warning("[EXECUTOR] jira_search: bare single-word assignee %r — no full name in flow_data", _bare_name)

        # 2. Inject project filter when a project was selected but not in JQL
        # Only apply when JQL has an assignee clause (person-specific query) and
        # no project clause is present, to avoid breaking cross-project queries.
        if "assignee" in _jql and "project" not in _jql.lower():
            _proj = _find_selected_project()
            if _proj:
                _jql = _jql.rstrip().rstrip("ORDER").rstrip()
                # Append before ORDER BY if present, else at end
                _order_m = _re_jql.search(r'(\s+ORDER\s+BY\s+.+)$', _jql, _re_jql.IGNORECASE)
                if _order_m:
                    _jql = _jql[:_order_m.start()] + f" AND project = {_proj}" + _order_m.group(1)
                else:
                    _jql = _jql + f" AND project = {_proj}"
                logger.info("[EXECUTOR] jira_search: injected project=%r into JQL", _proj)

        params["jql"] = _jql

    # ── sprint_id coercion: extract string id from sprint object/list ───────────
    # jira_get_sprints_from_board returns [{id, name, state, ...}].
    # When the planner writes sprint_id:"{{active}}" the template resolves to the
    # whole list/dict instead of just the numeric id string. Extract it here.
    _SPRINT_ID_TOOLS = {
        "jira_get_sprint_issues", "get_sprint_issues",
        "jira_update_sprint", "update_sprint",
        "jira_rank_backlog_issues", "rank_backlog_issues",
        "jira_move_issues_to_sprint", "move_issues_to_sprint",
    }
    if tool_name in _SPRINT_ID_TOOLS and "sprint_id" in params:
        sid = params["sprint_id"]
        if isinstance(sid, list) and sid:
            # list of sprint objects — take the first one's id
            first = sid[0]
            if isinstance(first, dict):
                params["sprint_id"] = str(first.get("id") or first.get("sprint_id") or first.get("value") or "")
            else:
                params["sprint_id"] = str(first)
            logger.info("[EXECUTOR] %s: extracted sprint_id=%r from list", tool_name, params["sprint_id"])
        elif isinstance(sid, dict):
            params["sprint_id"] = str(sid.get("id") or sid.get("sprint_id") or sid.get("selected") or sid.get("value") or "")
            logger.info("[EXECUTOR] %s: extracted sprint_id=%r from dict", tool_name, params["sprint_id"])

    # ── create_sprint / jira_create_sprint: fix sprint_name → name ─────────────
    if tool_name in ("create_sprint", "jira_create_sprint"):
        if "sprint_name" in params and "name" not in params:
            params["name"] = params.pop("sprint_name")

    # ── update_issue / jira_update_issue: issue_id → issue_key ──────────────
    if tool_name in ("update_issue", "jira_update_issue"):
        for alias in ("issue_id", "id", "key", "ticket_id", "ticket"):
            if alias in params and "issue_key" not in params:
                params["issue_key"] = params.pop(alias)
                break
        # If fields arrived as a JSON string, parse to dict for processing below
        # (the deep normalization block converts it back to JSON string for MCP)
        if isinstance(params.get("fields"), str):
            try:
                params["fields"] = json.loads(params["fields"])
            except (json.JSONDecodeError, TypeError):
                pass

    # ── get_issue / jira_get_issue: issue_id → issue_key ─────────────────────
    if tool_name in ("get_issue", "jira_get_issue"):
        for alias in ("issue_id", "id", "ticket_id", "ticket"):
            if alias in params and "issue_key" not in params:
                params["issue_key"] = params.pop(alias)
                break

    # ── transition_issue: keep both issue_key and issue_id so schema validation
    # can decide which one to use (some mcp-atlassian versions use issue_id) ──
    if tool_name in ("jira_transition_issue", "transition_issue"):
        for alias in ("issue_id", "id", "key", "ticket_id", "ticket"):
            if alias in params and "issue_key" not in params:
                params["issue_key"] = params.pop(alias)
                break

    # ── batch_modify_gmail_message_labels: fix synonyms ─────────────────────
    if tool_name == "batch_modify_gmail_message_labels":
        for alias in ("ids", "id_list"):
            if alias in params and "message_ids" not in params:
                params["message_ids"] = params.pop(alias)
                break
        for alias in ("remove_labels", "removeLabelIds", "labels_to_remove"):
            if alias in params and "remove_label_ids" not in params:
                params["remove_label_ids"] = params.pop(alias)
                break
        for alias in ("add_labels", "addLabelIds", "labels_to_add"):
            if alias in params and "add_label_ids" not in params:
                params["add_label_ids"] = params.pop(alias)
                break

    return params


def _recover_jira_issue_key(issue_key: str, flow_data: dict, results_by_id: dict) -> str:
    """Recover the full Jira issue key (e.g. 'FLAG-148756') from a numeric ID.

    The jira_search tool returns issues with both an 'id' (numeric DB id) and
    a 'key' (e.g. 'FLAG-148756'). The request_selection options use the 'id'
    as the value by default, but jira_delete_issue requires the full key.
    This helper looks up the cached search results to map the numeric ID back
    to the full key.
    """
    # Already looks like a full key (e.g. "FLAG-123")?
    import re as _re_issue

    if _re_issue.match(r"^[A-Z][A-Z0-9]+-\d+$", issue_key.strip()):
        return issue_key

    # Try to interpret as a numeric ID
    try:
        numeric_id = int(issue_key.strip())
    except (ValueError, AttributeError):
        return issue_key  # not numeric, return as-is

    # Search cached step results (results_by_id → flow_data)
    for container in (results_by_id, flow_data):
        for step_id, step_result in container.items():
            # In results_by_id, step_result is a dict with a "result" field.
            # In flow_data, step_result is the raw object itself.
            if isinstance(step_result, dict) and "result" in step_result:
                raw = step_result.get("result", "")
            else:
                raw = step_result

            # Parse JSON if needed
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

            # Extract list of issues from common wrapper keys
            issues: list = []
            if isinstance(raw, dict):
                for k in ("issues", "items", "data", "results", "_items", "_list"):
                    if k in raw and isinstance(raw[k], list):
                        issues = raw[k]
                        break
                else:
                    # Single issue object?
                    if "key" in raw and "fields" in raw:
                        issues = [raw]
            elif isinstance(raw, list):
                issues = raw

            # Scan for matching issue
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                # Match by numeric 'id'
                issue_id = issue.get("id")
                if isinstance(issue_id, (int, float)) and int(issue_id) == numeric_id:
                    full_key = issue.get("key")
                    if isinstance(full_key, str) and full_key.strip():
                        return full_key
                # Also match by suffix of 'key' (e.g. key="FLAG-148756" ends with "148756")
                key_str = issue.get("key", "")
                if isinstance(key_str, str):
                    m = _re_issue.search(r"-(\d+)$", key_str)
                    if m and int(m.group(1)) == numeric_id:
                        return key_str

    # No match found — return original
    return issue_key


async def _normalize_tool_params(
    tool_name: str,
    params: dict,
    flow_data: dict | None = None,
    results_by_id: dict | None = None,
) -> dict:
    """Apply tool-specific normalization to resolved params before invocation."""
    flow_data = flow_data or {}
    results_by_id = results_by_id or {}

    # ── Pre-strip: remap wrong param names BEFORE schema validation strips them ─
    # These run first so wrong-named params survive into the correct-named slots.
    params = _pre_strip_remap(tool_name, params, flow_data, results_by_id)

    # ── Save out-of-schema hints before stripping so specific blocks can use them ─
    # _target_status is passed by the planner as a hint for transition matching.
    _target_status_hint = str(
        params.pop("_target_status", "") or params.pop("target_status", "") or ""
    ).lower().strip()

    # ── Universal: strip params not in the tool's schema ─────────────────────
    # Must run BEFORE tool-specific logic so the specific blocks see clean params.
    params = _strip_invalid_params(tool_name, params)

    # ── Auto-recover project_key from cached projects when LLM guessed a name ──
    # The planner sometimes infers a plain project name (e.g. "insure tech") and passes
    # it as project_key. If it's not a valid Jira key, try to map it to the real key
    # using the jira_projects cache built at startup.
    if (
        tool_name
        in (
            "jira_create_issue",
            "jira_get_assignable_users",
            "jira_get_issues_in_project",
            "jira_search_issues",
            "jira_update_issue",
            "jira_delete_issue",
        )
        and "project_key" in params
    ):
        raw_pk = params["project_key"]
        if isinstance(raw_pk, str):
            pk = raw_pk.strip()
            # Not a valid Jira key (must be 2-10 uppercase alphanumeric)?
            if not _is_valid_jira_key(pk):
                try:
                    from dqe_agent.agent.planner import _cache_get as _planner_cache_get

                    cached_projects = _planner_cache_get("jira_projects") or []
                    # If the key is already an exact match to a cache value, keep it as-is
                    # (handles keys that look long/unusual but are real Jira project keys)
                    if any(p.get("value", "").upper() == pk.upper() for p in cached_projects):
                        logger.debug(
                            "[EXECUTOR] project_key %r is a known cache value — no recovery needed",
                            pk,
                        )
                    else:
                        # Find a project whose label contains the given string (case-insensitive)
                        matches = [
                            p for p in cached_projects if pk.lower() in p.get("label", "").lower()
                        ]
                        if len(matches) == 1:
                            new_key = matches[0]["value"]
                            logger.info(
                                "[EXECUTOR] project_key recovery: %r → %r (matched label %r)",
                                raw_pk,
                                new_key,
                                matches[0]["label"],
                            )
                            params["project_key"] = new_key
                except Exception as e:
                    logger.debug("[EXECUTOR] project_key recovery error: %s", e)

    # ── request_selection: ensure options is a proper [{value,label}] list ──
    if tool_name == "request_selection":
        # ── RULE: always allow multi-select when the user picks issues to delete ──
        # The LLM sometimes incorrectly sets multi_select=false for single-issue deletion
        # phrasing ("Which issue do you want to delete?"). Deletion must ALWAYS permit
        # selecting multiple issues because a user may want to delete more than one.
        # Override any false value if the question clearly references deletion.
        q_raw = params.get("question", "")
        if isinstance(q_raw, str):
            q_lower = q_raw.lower()
            if ("delete" in q_lower or "remove" in q_lower) and params.get("multi_select") is False:
                logger.info(
                    "[EXECUTOR] request_selection: forcing multi_select=True for deletion question: %r",
                    q_raw[:80],
                )
                params["multi_select"] = True

        # Step 0: resolve <<SENTINEL>> tokens from planner cache

        def _is_valid_options(v: Any) -> bool:
            """True only if v is already a non-empty [{value,label}] list."""
            return (
                isinstance(v, list)
                and bool(v)
                and isinstance(v[0], dict)
                and "value" in v[0]
                and "label" in v[0]
            )

        opts = params.get("options")

        # Step 0: resolve <<SENTINEL>> tokens from planner cache (or live fetch)
        if isinstance(opts, str) and opts.strip().startswith("<<"):
            _raw_sentinel = opts.strip()
            _sentinel_resolved = _resolve_prefetched_sentinel(_raw_sentinel)
            # Cache miss → live fetch for known sentinels
            if not _sentinel_resolved and _raw_sentinel == "<<JIRA_PROJECTS_PREFETCHED>>":
                try:
                    from dqe_agent.tools import get_tool as _gt_s
                    _proj_tool = _gt_s("jira_get_all_projects")
                    _proj_raw = await _proj_tool.ainvoke({})
                    _proj_items = _extract_items_from_response(_proj_raw)
                    _sentinel_resolved = _items_to_options(_proj_items) or None
                    if _sentinel_resolved:
                        logger.info("[EXECUTOR] sentinel live-fetch: %d projects", len(_sentinel_resolved))
                except Exception as _se:
                    logger.warning("[EXECUTOR] sentinel live-fetch failed: %s", _se)
            if _sentinel_resolved:
                params["options"] = _sentinel_resolved
                opts = _sentinel_resolved
                logger.info(
                    "[EXECUTOR] request_selection: resolved sentinel %r → %d items",
                    _raw_sentinel,
                    len(_sentinel_resolved),
                )

        # Track whether options came from a specific {{ref}} template so we can
        # skip the last-resort scan when the template named a concrete step.
        _opts_had_explicit_ref = False

        # Step 1: string → try to resolve to actual object
        if isinstance(opts, str):
            import re as _re

            m = _re.match(r"^\{\{(.+?)\}\}$", opts.strip())
            raw = None
            if m:
                _opts_had_explicit_ref = True
                ref = m.group(1).strip()
                raw = _resolve_ref_to_object(ref, flow_data, results_by_id)
                # If the exact ref path wasn't found, try the parent step's full result
                if raw is None:
                    parent_step = ref.split(".")[0]
                    raw = _resolve_ref_to_object(parent_step, flow_data, results_by_id)
            else:
                try:
                    raw = json.loads(opts)
                except (json.JSONDecodeError, TypeError):
                    pass
            if raw is not None:
                items = _extract_items_from_response(raw)
                transformed = _items_to_options(items)
                if transformed:
                    params["options"] = transformed
                    logger.info(
                        "[EXECUTOR] request_selection: resolved %d options from template %r",
                        len(transformed),
                        opts,
                    )
                    opts = transformed

        # Step 2: list of raw objects → transform to {value,label}
        if isinstance(opts, list) and opts and not _is_valid_options(opts):
            params["options"] = _items_to_options(opts)
            opts = params["options"]

        # Step 3: still not valid — scan ALL recent step results as last resort.
        # Skip when the options template referenced a specific step (e.g. {{get_roles}});
        # if that step failed or returned nothing, do NOT substitute unrelated data
        # from a different step (e.g. find_user), which would produce a wrong value.
        if not _is_valid_options(params.get("options")) and not _opts_had_explicit_ref:
            for sid, entry in reversed(list(results_by_id.items())):
                raw_result = entry.get("result", "")
                if isinstance(raw_result, str):
                    try:
                        raw_result = json.loads(raw_result)
                    except (json.JSONDecodeError, TypeError):
                        continue
                items = _extract_items_from_response(raw_result)
                transformed = _items_to_options(items)
                if transformed:  # use any non-empty list
                    logger.warning(
                        "[EXECUTOR] request_selection: fallback — using %d items from step '%s'",
                        len(transformed),
                        sid,
                    )
                    params["options"] = transformed
                    break

        # Step 4: name-fragment filter — "Which Tania do you mean?" → keep only Tanias
        # Extract the queried name from patterns like "Which X do you mean?" / "Which X?"
        # Then filter options to only entries whose label contains that fragment.
        # 0 matches → show all (don't filter). 1+ matches → filter to subset.
        # Single-match auto-select happens at call site (needs state).
        if _is_valid_options(params.get("options")):
            import re as _re_frag
            _q_for_filter = params.get("question", "")
            _name_frag: str | None = None
            for _pat in (
                r"[Ww]hich\s+([A-Za-z][A-Za-z\s'-]{0,30}?)\s+do you mean",
                r"[Ww]hich\s+([A-Za-z][A-Za-z\s'-]{0,30}?)\?",
                r"multiple\s+([A-Za-z][A-Za-z\s'-]{1,30})s?\b",
            ):
                _m2 = _re_frag.search(_pat, _q_for_filter)
                if _m2:
                    _name_frag = _m2.group(1).strip()
                    break
            if _name_frag:
                _frag_lower = _name_frag.lower()
                _all_opts = params["options"]
                _filtered = [
                    o for o in _all_opts
                    if _frag_lower in str(o.get("label", "")).lower()
                    or _frag_lower in str(o.get("value", "")).lower()
                ]
                if _filtered:
                    logger.info(
                        "[EXECUTOR] request_selection: name-frag %r → %d matches (filtered from %d)",
                        _name_frag, len(_filtered), len(_all_opts),
                    )
                    params["options"] = _filtered
                else:
                    # Person not found in the fetched members list — signal the call
                    # site to abort and tell the user instead of showing all options.
                    logger.info(
                        "[EXECUTOR] request_selection: name-frag %r → 0 matches in %d options — person not found",
                        _name_frag, len(_all_opts),
                    )
                    params["_person_not_found"] = _name_frag

        return params

    # ── Jira board/sprint fetch tools: auto-inject project_key ───────────────
    # The planner sometimes passes wrong param names (board_filter, project, etc.).
    # After stripping invalid params above, re-inject project_key if the tool
    # accepts it and we already have the key from a prior selection step.
    _JIRA_LIST_TOOLS = {
        "jira_get_agile_boards",
        "jira_get_boards",
        "jira_list_boards",
        "jira_get_sprints",
        "jira_list_sprints",
        "jira_get_active_sprints",
        "jira_get_issues_in_project",
        "jira_search_issues",
    }
    if tool_name in _JIRA_LIST_TOOLS and not params.get("project_key"):
        import re as _re2

        _key_pat2 = _re2.compile(r"^[A-Z][A-Z0-9]{1,9}$")
        recovered_key = _find_selected_value(
            results_by_id,
            flow_data,
            priority_words=["select", "project", "key"],
            validate=lambda v: bool(_key_pat2.match(v.strip().upper())),
            transform=lambda v: v.strip().upper(),
        )
        if recovered_key:
            # Only inject if the tool schema actually has this param
            from dqe_agent.tools import get_tool as _gt

            _to = _gt(tool_name)
            if _to:
                try:
                    _schema = (
                        _to.args_schema.model_json_schema()
                        if hasattr(_to, "args_schema") and _to.args_schema
                        else {}
                    )
                    if "project_key" in _schema.get("properties", {}):
                        logger.info(
                            "[EXECUTOR] '%s': injecting project_key=%r from prior selection",
                            tool_name,
                            recovered_key,
                        )
                        params["project_key"] = recovered_key
                except Exception:
                    pass

        # Hard guard: if jira_get_agile_boards still has no project_key after recovery,
        # interrupt to ask the user — calling it with no project returns workspace-wide
        # boards which is useless and confusing.
        if tool_name == "jira_get_agile_boards" and not params.get("project_key"):
            from langgraph.types import interrupt as _interrupt_boards

            cached_projs = _cache_get("jira_projects") or []
            if cached_projs:
                proj_options = cached_projs
            else:
                proj_options = [{"value": "FLAG", "label": "FLAG"}]

            logger.warning(
                "[EXECUTOR] jira_get_agile_boards called with no project_key — "
                "interrupting to ask user"
            )
            sel_result = _interrupt_boards({
                "type": "selection",
                "question": "Which project's boards do you want to see?",
                "options": proj_options,
                "multi_select": False,
            })
            selected_proj = (
                sel_result.get("selected") if isinstance(sel_result, dict) else str(sel_result)
            )
            if selected_proj:
                params["project_key"] = selected_proj
                logger.info(
                    "[EXECUTOR] jira_get_agile_boards: user selected project_key=%r",
                    selected_proj,
                )

    if tool_name == "jira_create_sprint":
        from datetime import datetime, timezone, timedelta
        import re as _red

        now_utc = datetime.now(timezone.utc)
        tomorrow_utc = now_utc + timedelta(days=1)
        default_start = tomorrow_utc.strftime("%Y-%m-%dT00:00:00+00:00")
        default_end = (tomorrow_utc + timedelta(days=14)).strftime("%Y-%m-%dT00:00:00+00:00")

        def _normalise_sprint_date(v: Any) -> str | None:
            """Convert user-provided date string to Jira ISO format, or return None if blank/missing."""
            if not v or not isinstance(v, str):
                return None
            v = v.strip()
            if not v:
                return None
            # Already ISO: 2026-05-01 or 2026-05-01T...
            if _red.match(r"^\d{4}-\d{2}-\d{2}", v):
                # Ensure full datetime with offset
                if "T" not in v:
                    return v + "T00:00:00+00:00"
                return v
            # Natural language
            v_low = v.lower()
            if v_low in ("today",):
                return now_utc.strftime("%Y-%m-%dT00:00:00+00:00")
            if v_low in ("tomorrow",):
                return tomorrow_utc.strftime("%Y-%m-%dT00:00:00+00:00")
            if "next week" in v_low:
                return (now_utc + timedelta(days=7)).strftime("%Y-%m-%dT00:00:00+00:00")
            # Try dateutil if available, else fall back to None
            try:
                from dateutil import parser as _du
                parsed = _du.parse(v, default=now_utc.replace(tzinfo=timezone.utc))
                return parsed.strftime("%Y-%m-%dT00:00:00+00:00")
            except Exception:
                return None

        sd = _normalise_sprint_date(params.get("start_date"))
        ed = _normalise_sprint_date(params.get("end_date"))

        if sd:
            params["start_date"] = sd
        else:
            logger.warning(
                "[EXECUTOR] jira_create_sprint: start_date not provided — falling back to %s",
                default_start,
            )
            params["start_date"] = default_start

        if ed:
            params["end_date"] = ed
        else:
            logger.warning(
                "[EXECUTOR] jira_create_sprint: end_date not provided — falling back to %s",
                default_end,
            )
            params["end_date"] = default_end

        # Remove goal if empty — it's optional and some Jira instances reject empty strings
        if "goal" in params and not params["goal"]:
            params.pop("goal")

        # board_id: recover from prior selection if still a template
        if isinstance(params.get("board_id"), str) and "{{" in params["board_id"]:
            recovered_board = _find_selected_value(
                results_by_id,
                flow_data,
                priority_words=["board", "select"],
                validate=lambda v: v.strip().lstrip("-").isdigit(),
                transform=lambda v: v.strip(),
            )
            if recovered_board:
                logger.info(
                    "[EXECUTOR] jira_create_sprint: recovered board_id=%r from step results",
                    recovered_board,
                )
                params["board_id"] = recovered_board

    if tool_name == "jira_create_issue":
        flow_data = flow_data or {}
        results_by_id = results_by_id or {}

        # ── project_key ───────────────────────────────────────────────────
        # Always uppercase whatever was provided
        if isinstance(params.get("project_key"), str):
            params["project_key"] = params["project_key"].strip().upper()

        # Valid Jira project key: 2-10 uppercase letters/digits, no hyphens (not a UUID)
        import re as _re

        _key_pat = _re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")  # Look for a key anywhere in the string

        orig_key = params.get("project_key", "")
        match_orig = (
            _key_pat.search(orig_key.strip().upper()) if isinstance(orig_key, str) else None
        )

        if match_orig:
            # Successfully extracted a key from the provided string (handles "KEY - Name" cases)
            params["project_key"] = match_orig.group(1)
        else:
            # Scan step results for a valid key — prefer steps named "select*" or "project*"
            # Also check the "selected" field (from request_selection JSON result)
            _strict_pat = _re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
            recovered = _find_selected_value(
                results_by_id,
                flow_data,
                priority_words=["select", "project", "key"],
                validate=lambda v: bool(_strict_pat.match(v.strip().upper())),
                transform=lambda v: v.strip().upper(),
            )
            if recovered:
                logger.info(
                    "[EXECUTOR] jira_create_issue: recovered project_key=%r from step results",
                    recovered,
                )
                params["project_key"] = recovered
            else:
                logger.warning(
                    "[EXECUTOR] jira_create_issue: project_key=%r is invalid and could not be recovered",
                    orig_key,
                )

        # ── issue_type ────────────────────────────────────────────────────
        if not params.get("issue_type"):
            recovered_type = _find_ask_answer(
                results_by_id,
                flow_data,
                priority_words=["issue_type", "type"],
                transform=lambda v: v.strip().capitalize(),
            )
            if recovered_type:
                logger.info(
                    "[EXECUTOR] jira_create_issue: recovered issue_type=%r from step results",
                    recovered_type,
                )
                params["issue_type"] = recovered_type

        # ── summary ───────────────────────────────────────────────────────
        if not params.get("summary"):
            recovered_summary = _find_ask_answer(
                results_by_id,
                flow_data,
                priority_words=["summary", "title"],
            )
            if recovered_summary:
                logger.info(
                    "[EXECUTOR] jira_create_issue: recovered summary=%r from step results",
                    recovered_summary,
                )
                params["summary"] = recovered_summary

    # ── jira_add_comment: comment field aliases ──────────────────────────────
    if tool_name == "jira_add_comment":
        for alias in ("text", "message", "body", "content", "note"):
            if alias in params and "comment" not in params:
                params["comment"] = params.pop(alias)
                break

    # ── jira_add_worklog: time_spent conversion + aliases ───────────────────
    if tool_name == "jira_add_worklog":
        for alias in ("hours", "time", "duration", "logged_time", "work_time"):
            if alias in params and "time_spent" not in params:
                params["time_spent"] = params.pop(alias)
                break
        # Convert plain-English durations to Jira format
        ts = params.get("time_spent", "")
        if ts and isinstance(ts, str):
            import re as _rts

            # "1.5 hours" / "1.5h" → "1h 30m"
            frac = _rts.match(r"^(\d+(?:\.\d+))\s*h", ts.strip(), _rts.IGNORECASE)
            if frac:
                total_minutes = int(float(frac.group(1)) * 60)
                h, m = divmod(total_minutes, 60)
                params["time_spent"] = f"{h}h {m}m" if m else f"{h}h"
            else:
                # "3 hours" → "3h"
                ts2 = _rts.sub(r"(\d+)\s*hours?", r"\1h", ts, flags=_rts.IGNORECASE)
                # "30 minutes" / "30 mins" → "30m"
                ts2 = _rts.sub(r"(\d+)\s*min(?:utes?)?", r"\1m", ts2, flags=_rts.IGNORECASE)
                params["time_spent"] = ts2.strip()

    # ── jira_create_issue_link: field aliases + direction normalization ───────
    if tool_name == "jira_create_issue_link":
        for alias in ("type", "relationship", "link", "link_type_name"):
            if alias in params and "link_type" not in params:
                params["link_type"] = params.pop(alias)
                break
        for alias in ("from_issue", "source", "source_issue", "issue", "issue_key"):
            if alias in params and "inward_issue" not in params:
                params["inward_issue"] = params.pop(alias)
                break
        for alias in ("to_issue", "target", "target_issue", "related_issue"):
            if alias in params and "outward_issue" not in params:
                params["outward_issue"] = params.pop(alias)
                break

    # ── jira_update_sprint: state aliases ───────────────────────────────────
    if tool_name == "jira_update_sprint":
        state_val = params.get("state", "")
        if isinstance(state_val, str):
            _state_map = {
                "start": "active",
                "begin": "active",
                "open": "active",
                "close": "closed",
                "end": "closed",
                "complete": "closed",
                "finish": "closed",
            }
            params["state"] = _state_map.get(state_val.lower(), state_val)

    # ── jira_search: resolve form sentinels + sanitize JQL ───────────────────
    if tool_name == "jira_search":
        import re as _re_jql_s

        jql = params.get("jql", "") or ""

        # ── Resolve period sentinels from the logged-hours form ───────────
        jql_lower = jql.lower()
        # "__all_time__" → remove the entire worklogDate clause AND the sentinel itself
        if "__all_time__" in jql_lower:
            jql = _re_jql_s.sub(
                r"\s*AND\s+worklogDate\s*[><=!]+[^A-Za-z]*(?:AND\s+worklogDate[^A-Za-z]*)*",
                "",
                jql,
            )
            jql = _re_jql_s.sub(r"\s*AND\s+__all_time__\b|\b__all_time__\b\s*AND\s*|\b__all_time__\b", "", jql, flags=_re_jql_s.IGNORECASE).strip()
            logger.info("[EXECUTOR] jira_search: __all_time__ → removed worklogDate filter")

        # "__custom__" → swap in start/end dates from flow_data
        elif "__custom__" in jql_lower:
            _sd, _ed = None, None
            for _fv in flow_data.values():
                if isinstance(_fv, dict):
                    _sd = _fv.get("start_date") or _sd
                    _ed = _fv.get("end_date") or _ed
            if _sd and _ed:
                _date_jql = f'worklogDate >= "{_sd}" AND worklogDate <= "{_ed}"'
            elif _sd:
                _date_jql = f'worklogDate >= "{_sd}"'
            elif _ed:
                _date_jql = f'worklogDate <= "{_ed}"'
            else:
                _date_jql = ""
            if _date_jql:
                jql = _re_jql_s.sub(r"\b__custom__\b", _date_jql, jql, flags=_re_jql_s.IGNORECASE)
                logger.info("[EXECUTOR] jira_search: __custom__ → %r", _date_jql)
            else:
                jql = _re_jql_s.sub(r"\s*AND\s+__custom__|\b__custom__\b", "", jql, flags=_re_jql_s.IGNORECASE).strip()
                logger.warning("[EXECUTOR] jira_search: __custom__ but no dates in flow_data — removed")

        # ── Resolve member sentinels ──────────────────────────────────────
        jql_lower = jql.lower()
        if "__all__" in jql_lower:
            jql = _re_jql_s.sub(
                r"\s*AND\s+worklogAuthor\s*=\s*['\"]?__all__['\"]?|worklogAuthor\s*=\s*['\"]?__all__['\"]?\s*AND\s*",
                "", jql, flags=_re_jql_s.IGNORECASE,
            ).strip()
            logger.info("[EXECUTOR] jira_search: __all__ → removed worklogAuthor filter")

        if "__me__" in jql_lower:
            jql = _re_jql_s.sub(
                r"worklogAuthor\s*=\s*['\"]?__me__['\"]?",
                "worklogAuthor = currentUser()",
                jql,
                flags=_re_jql_s.IGNORECASE,
            )
            logger.info("[EXECUTOR] jira_search: __me__ → worklogAuthor = currentUser()")

        # Normalize whitespace and clean up orphaned AND/OR
        jql = _re_jql_s.sub(r"\s+AND\s+AND\s+", " AND ", jql)
        jql = _re_jql_s.sub(r"^\s*AND\s+|\s+AND\s*$", "", jql).strip()
        params["jql"] = jql

        jql = params.get("jql", "")
        if not isinstance(jql, str) or not jql.strip():
            import re as _re_jql

            proj_key = _first_project_key()
            params["jql"] = (
                f"project = {proj_key} ORDER BY created DESC"
                if proj_key
                else "ORDER BY created DESC"
            )
            logger.warning("[EXECUTOR] jira_search: empty jql — defaulted to %r", params["jql"])
        elif "{{" in jql:
            # Unresolved template — substitute project key
            import re as _re_jql

            proj_key = _first_project_key()
            cleaned = _re_jql.sub(r"\{\{[^}]+\}\}", proj_key if proj_key else "", jql)
            cleaned = cleaned.strip().strip("AND").strip("OR").strip()
            params["jql"] = cleaned or f"project = {proj_key} ORDER BY created DESC"
            logger.warning(
                "[EXECUTOR] jira_search: cleaned unresolved JQL template → %r", params["jql"]
            )
        # mcp-atlassian v0.11.10 bug: Jira Cloud v3 returns description as ADF dict,
        # but JiraIssue.description expects str → Pydantic validation error.
        # Fix: always exclude 'description' from search fields.
        if "fields" not in params or not params.get("fields"):
            params["fields"] = "summary,status,assignee,priority,issuetype,labels,updated,created"
        elif isinstance(params.get("fields"), str) and "description" in params["fields"]:
            params["fields"] = ",".join(
                f for f in params["fields"].split(",") if f.strip() != "description"
            )
        # Ensure limit is an integer >= 1
        _lim = params.get("limit", 10)
        try:
            params["limit"] = max(1, int(_lim))
        except (TypeError, ValueError):
            params["limit"] = 10

    # ── jira_get_worklogs_by_date_range: normalise sentinels + recover project ────
    if tool_name == "jira_get_worklogs_by_date_range":
        mn = params.get("member_name", "")
        if isinstance(mn, str) and mn in ("__all__", "__all_time__"):
            params["member_name"] = ""
            logger.info("[EXECUTOR] jira_get_worklogs_by_date_range: member_name cleared (all)")
        # Recover missing project_key from flow_data / prior steps
        _wl_pk = params.get("project_key", "")
        if not _wl_pk or (isinstance(_wl_pk, str) and ("{{" in _wl_pk or not _wl_pk.strip())):
            _wl_pk = _recover_project_key(results_by_id, flow_data) or _first_project_key()
            if _wl_pk:
                params["project_key"] = _wl_pk
                logger.info("[EXECUTOR] jira_get_worklogs_by_date_range: recovered project_key=%r", _wl_pk)

        # Executor no longer performs member disambiguation here.
        # Planner-level planning is responsible for inserting fetch + selection
        # steps when a user names a person ambiguously. Leave member_name as-is
        # so the planner's plan can include explicit request_selection steps.

    # ── jira_get_assignable_users: resolve missing/unresolved project_key ──────
    if tool_name == "jira_get_assignable_users":
        pk = params.get("project_key", "")
        if not pk or (isinstance(pk, str) and ("{{" in pk or "<<" in pk or not pk.strip())):
            # Try flow_data / recent results first (user may have just picked a project)
            pk = _recover_project_key(results_by_id, flow_data)
            if not pk:
                pk = _first_project_key()
            if pk:
                params["project_key"] = pk
                logger.info("[EXECUTOR] jira_get_assignable_users: recovered project_key=%r", pk)

    # ── jira_assign_issue: resolve "me" to configured username ──────────────
    if tool_name == "jira_assign_issue":
        acct = params.get("account_id", "")
        if isinstance(acct, str) and acct.lower() in (
            "me",
            "myself",
            "currentuser",
            "current_user",
        ):
            from dqe_agent.config import settings as _cfg

            if _cfg.jira_username:
                params["account_id"] = _cfg.jira_username
                logger.info("[EXECUTOR] jira_assign_issue: resolved 'me' to %s", _cfg.jira_username)

    # ── jira_search_users: query aliases ─────────────────────────────────────
    if tool_name == "jira_search_users":
        for alias in ("name", "username", "user", "person", "search"):
            if alias in params and "query" not in params:
                params["query"] = params.pop(alias)
                break

    # ── jira_transition_issue: extract transition_id from get_transitions result ──
    if tool_name == "jira_transition_issue":
        tid = params.get("transition_id", "")
        if isinstance(tid, (list, dict)) or (
            isinstance(tid, str) and (tid.startswith("[") or tid.startswith("{"))
        ):
            # The transition_id is a raw result from jira_get_transitions — extract by target status
            # Look at the task context (step description) for the target status name
            try:
                transitions = json.loads(tid) if isinstance(tid, str) else tid
                if not isinstance(transitions, list):
                    transitions = transitions.get("transitions", transitions.get("values", []))

                # Try to find by matching known status names from context
                _TARGET_NAMES = {
                    "done": ["done", "complete", "closed", "resolved", "finish"],
                    "in progress": ["in progress", "inprogress", "start", "working", "active", "doing"],
                    "in review": ["in review", "review", "pr review", "code review"],
                    "to do": ["to do", "todo", "backlog", "open", "reopen"],
                    "testing": ["testing", "qa", "test", "in testing"],
                    "on hold": ["on hold", "onhold", "blocked", "impediment", "hold"],
                    "cancelled": ["cancelled", "canceled", "cancel"],
                }

                # Prefer the saved _target_status hint (planner-set), then flow_data, then nothing
                target_hint = (
                    _target_status_hint
                    or str((flow_data or {}).get("_target_status", "")).lower()
                ).strip()
                logger.info(
                    "[EXECUTOR] jira_transition_issue: target_hint=%r  transitions=%s",
                    target_hint,
                    [(t.get("id"), t.get("name")) for t in transitions if isinstance(t, dict)],
                )

                best_id = None

                # 1. Try alias matching against target_hint
                if target_hint:
                    for t in transitions:
                        if not isinstance(t, dict):
                            continue
                        t_name = (t.get("name") or t.get("to", {}).get("name") or "").lower()
                        t_id = str(t.get("id", ""))
                        if not t_id:
                            continue
                        for canonical, aliases in _TARGET_NAMES.items():
                            if any(a in target_hint for a in aliases):
                                if any(a in t_name for a in aliases):
                                    best_id = t_id
                                    break
                        if best_id:
                            break

                # 2. Direct name match (target_hint == transition name exactly)
                if not best_id and target_hint:
                    for t in transitions:
                        if not isinstance(t, dict):
                            continue
                        t_name = (t.get("name") or t.get("to", {}).get("name") or "").lower()
                        t_id = str(t.get("id", ""))
                        if t_id and target_hint in t_name:
                            best_id = t_id
                            break

                if not best_id and transitions:
                    # No hint available — cannot guess, leave transition_id unresolved
                    # so the caller can use request_selection instead
                    logger.warning(
                        "[EXECUTOR] jira_transition_issue: no target_hint — cannot pick transition automatically"
                    )
                    best_id = None

                if best_id:
                    logger.info(
                        "[EXECUTOR] jira_transition_issue: extracted transition_id=%s", best_id
                    )
                    params["transition_id"] = best_id

            except Exception as _exc:
                logger.warning(
                    "[EXECUTOR] jira_transition_issue: could not extract transition_id: %s", _exc
                )

    # ── Google Calendar read tools: defaults + injection ─────────────────────
    _CALENDAR_EVENT_TOOLS = {
        "get_events",
        "list_events",
        "calendar_get_events",
        "calendar_list_events",
        "list_calendars",
        "calendar_list",
        "get_calendars",
    }
    if tool_name in _CALENDAR_EVENT_TOOLS:
        from datetime import date

        today = date.today()

        # Auto-inject user_google_email from settings if missing or unresolved
        _email_val = str(params.get("user_google_email", ""))
        if not _email_val or "{{" in _email_val:
            from dqe_agent.config import settings as _cs

            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] %s: injected user_google_email from settings", tool_name)

        # list_calendars / get_calendars don't use time filters — stop here
        _CALENDAR_LIST_TOOLS = {"list_calendars", "calendar_list", "get_calendars"}
        if tool_name in _CALENDAR_LIST_TOOLS:
            for bad in [
                k for k in list(params.keys()) if k not in {"user_google_email", "max_results"}
            ]:
                params.pop(bad)
            return params

        # Default time_min/time_max to today if missing or unresolved.
        # Then, if flow_data has a user-chosen meeting date (ask_date / date / meeting_date),
        # use that instead of today so multi-calendar fetches all stay on the same date.
        def _is_valid_dt(v: Any) -> bool:
            return isinstance(v, str) and len(v) >= 10 and v[:4].isdigit()

        _meeting_date = ""
        for _mdk in ("ask_date", "date", "meeting_date"):
            _mdv = flow_data.get(_mdk, {})
            if isinstance(_mdv, dict):
                _mda = _mdv.get("answer") or _mdv.get("result", "")
                if _mda and isinstance(_mda, str) and len(_mda) >= 10 and _mda[:4].isdigit():
                    _meeting_date = _mda.strip()[:10]
                    break

        if not _is_valid_dt(params.get("time_min")):
            _ref_date = _meeting_date or today.isoformat()
            params["time_min"] = f"{_ref_date}T00:00:00Z"
            logger.info("[EXECUTOR] %s: defaulted time_min to %s", tool_name, _ref_date)
        if not _is_valid_dt(params.get("time_max")):
            _ref_date = _meeting_date or today.isoformat()
            params["time_max"] = f"{_ref_date}T23:59:59Z"
            logger.info("[EXECUTOR] %s: defaulted time_max to %s", tool_name, _ref_date)

        # If time_min/time_max resolved to today but flow_data has a different meeting date,
        # override — planner may have hardcoded today for a secondary get_events step
        if _meeting_date and _meeting_date != today.isoformat():
            _tmin = str(params.get("time_min", ""))
            _tmax = str(params.get("time_max", ""))
            if _tmin.startswith(today.isoformat()) and not _tmax.startswith(_meeting_date):
                params["time_min"] = f"{_meeting_date}T00:00:00Z"
                params["time_max"] = f"{_meeting_date}T23:59:59Z"
                logger.info(
                    "[EXECUTOR] %s: corrected date from today to meeting_date=%s",
                    tool_name, _meeting_date,
                )

        # Resolve {{user_google_email}} in calendar_id before defaulting
        _cal_id = str(params.get("calendar_id", ""))
        if "{{user_google_email}}" in _cal_id:
            from dqe_agent.config import settings as _cs_ge
            if _cs_ge.user_google_email:
                _cal_id = _cal_id.replace("{{user_google_email}}", _cs_ge.user_google_email)
                params["calendar_id"] = _cal_id
                logger.info("[EXECUTOR] %s: resolved calendar_id to '%s'", tool_name, _cal_id)

        # Default calendar_id to 'primary' if missing or still unresolved
        _cal_id = str(params.get("calendar_id", ""))
        if not _cal_id or "{{" in _cal_id:
            params["calendar_id"] = "primary"
            logger.info("[EXECUTOR] %s: defaulted calendar_id to 'primary'", tool_name)

    # ── create_event / modify_event: defaults + injection ────────────────────────
    if tool_name in ("create_event", "modify_event", "manage_event"):
        from dqe_agent.config import settings as _cs

        # Auto-inject user_google_email
        _ev_email = str(params.get("user_google_email", ""))
        if not _ev_email or "{{" in _ev_email:
            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] %s: injected user_google_email from settings", tool_name)

        # Default calendar_id to 'primary'
        _ev_cal = str(params.get("calendar_id", ""))
        if not _ev_cal or "{{" in _ev_cal:
            params["calendar_id"] = "primary"

        # Ensure start_time/end_time are RFC3339 (YYYY-MM-DDTHH:MM:SSZ)
        def _parse_nl_datetime(val: str, known_date: str = "") -> str:
            """Convert natural-language time/datetime to RFC3339.
            known_date: YYYY-MM-DD to combine with time-only input like '2pm', '14:00'.
            """
            import re as _re_ev
            from datetime import datetime, date as _date, timedelta

            s = str(val).strip() if val else ""
            if not s or "{{" in s:
                return s

            s_lower = s.lower().strip()

            # "now" / "asap" → current time rounded up to next 5 min
            if s_lower in ("now", "asap", "right now", "immediately"):
                now = datetime.utcnow()
                # Round up to next 5-minute boundary
                minutes = (now.minute // 5 + 1) * 5
                if minutes >= 60:
                    now = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    now = now.replace(minute=minutes, second=0, microsecond=0)
                return now.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Already RFC3339
            if "T" in s and _re_ev.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s):
                if not s.endswith("Z") and "+" not in s:
                    s = s[:19] + "Z"
                return s

            # Plain date only: YYYY-MM-DD
            if _re_ev.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return f"{s}T00:00:00Z"

            # Natural language date resolution
            today = _date.today()
            ref_date = None

            # Try to extract a known date from the string
            if "today" in s_lower:
                ref_date = today
            elif "tomorrow" in s_lower:
                ref_date = today + timedelta(days=1)
            else:
                # Try YYYY-MM-DD embedded in a longer string
                dm = _re_ev.search(r"(\d{4}-\d{2}-\d{2})", s)
                if dm:
                    try:
                        ref_date = _date.fromisoformat(dm.group(1))
                    except ValueError:
                        pass

            # Fall back to known_date if no date found in the string
            if ref_date is None and known_date:
                try:
                    ref_date = _date.fromisoformat(known_date[:10])
                except ValueError:
                    pass
            if ref_date is None:
                ref_date = today

            # Extract time from string
            # Matches: 2pm, 2:30pm, 14:00, 2:30 PM, 14:30:00
            tm = _re_ev.search(
                r"(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm)?",
                s_lower,
            )
            if tm:
                hour = int(tm.group(1))
                minute = int(tm.group(2) or 0)
                second = int(tm.group(3) or 0)
                meridiem = tm.group(4)
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                try:
                    dt = datetime(ref_date.year, ref_date.month, ref_date.day, hour, minute, second)
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    pass

            return s  # give up, return as-is

        # Pull the known date from flow_data (ask_date step answer)
        _known_date = ""
        for _fd_key in ("ask_date", "date", "meeting_date"):
            _fd_val = flow_data.get(_fd_key, {})
            if isinstance(_fd_val, dict):
                _fd_ans = _fd_val.get("answer") or _fd_val.get("result", "")
                if _fd_ans and isinstance(_fd_ans, str) and len(_fd_ans) >= 8:
                    _known_date = _fd_ans.strip()
                    break
            elif isinstance(_fd_val, str) and len(_fd_val) >= 8:
                _known_date = _fd_val.strip()
                break

        if params.get("start_time"):
            params["start_time"] = _parse_nl_datetime(params["start_time"], _known_date)
        if params.get("end_time"):
            params["end_time"] = _parse_nl_datetime(params["end_time"], _known_date)

        # If the caller provided a human-friendly duration ("30 minutes", "1 hour"), parse it
        # compute end_time from duration when not provided (create_event / manage_event create)
        _is_create = tool_name == "create_event" or params.get("action") == "create"
        if _is_create and params.get("start_time") and not params.get("end_time"):
            try:
                from datetime import datetime, timedelta

                dt = datetime.fromisoformat(params["start_time"].replace("Z", "+00:00"))
                dur = params.get("duration")
                if dur and isinstance(dur, str):
                    import re as _re_dur

                    m = _re_dur.search(r"(\d+)\s*(h|hr|hour|hours)", dur, _re_dur.I)
                    if m:
                        params["end_time"] = (dt + timedelta(hours=int(m.group(1)))).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    else:
                        m = _re_dur.search(r"(\d+)\s*(m|min|minute|minutes)", dur, _re_dur.I)
                        if m:
                            params["end_time"] = (dt + timedelta(minutes=int(m.group(1)))).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            )
                if not params.get("end_time"):
                    params["end_time"] = (dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    logger.info("[EXECUTOR] %s: defaulted end_time to 1h after start", tool_name)
                else:
                    logger.info("[EXECUTOR] %s: computed end_time from duration", tool_name)
            except Exception:
                pass

        # Normalize attendees: plain string → list
        _att = params.get("attendees")
        if isinstance(_att, str):
            _att_lower = _att.strip().lower()
            if not _att_lower or _att_lower in (
                "none", "just me", "no one", "only me", "myself",
                "no attendees", "no one else", "just myself",
            ):
                params.pop("attendees", None)
                logger.info("[EXECUTOR] %s: removed 'just me' attendees", tool_name)
            else:
                import re as _re_att
                _emails = [e.strip() for e in _re_att.split(r"[,;]+", _att) if e.strip()]
                if _emails:
                    params["attendees"] = _emails
                    logger.info("[EXECUTOR] %s: parsed %d attendee(s)", tool_name, len(_emails))

    # ── get_events: auto-inject email, defaults, NL time parsing ────────────────
    if tool_name in ("get_events", "query_freebusy"):
        from dqe_agent.config import settings as _cs
        from datetime import date as _dt_date, datetime as _dt_datetime
        import re as _re_ge

        _fb_email = str(params.get("user_google_email", ""))
        if not _fb_email or "{{" in _fb_email:
            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] %s: injected user_google_email", tool_name)

        # Pull known date from flow_data for NL parsing
        _ge_known_date = ""
        for _gek in ("ask_date", "date", "meeting_date"):
            _ge_fd = flow_data.get(_gek, {})
            if isinstance(_ge_fd, dict):
                _ge_ans = _ge_fd.get("answer") or _ge_fd.get("result", "")
                if _ge_ans and isinstance(_ge_ans, str) and len(_ge_ans) >= 8:
                    _ge_known_date = _ge_ans.strip()
                    break

        def _ge_parse_time(val: str, is_end: bool = False) -> str:
            s = str(val).strip() if val else ""
            if not s or "{{" in s:
                return s
            # "now" / "asap" → current UTC time
            if s.lower() in ("now", "asap", "right now", "immediately"):
                from datetime import datetime as _dtnow
                return _dtnow.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            if _re_ge.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s):
                if not s.endswith("Z") and "+" not in s:
                    s = s[:19] + "Z"
                return s
            # Natural language: resolve via same logic as create_event block
            from datetime import date as _d, timedelta as _td
            today = _d.today()
            s_low = s.lower()
            ref_date = None
            if "today" in s_low:
                ref_date = today
            elif "tomorrow" in s_low:
                ref_date = today + _td(days=1)
            else:
                dm = _re_ge.search(r"(\d{4}-\d{2}-\d{2})", s)
                if dm:
                    try:
                        ref_date = _d.fromisoformat(dm.group(1))
                    except ValueError:
                        pass
            if ref_date is None and _ge_known_date:
                try:
                    ref_date = _d.fromisoformat(_ge_known_date[:10])
                except ValueError:
                    pass
            if ref_date is None:
                ref_date = today
            tm = _re_ge.search(r"(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm)?", s_low)
            if tm:
                h = int(tm.group(1)); mi = int(tm.group(2) or 0); sec = int(tm.group(3) or 0)
                mer = tm.group(4)
                if mer == "pm" and h != 12: h += 12
                elif mer == "am" and h == 12: h = 0
                try:
                    return _dt_datetime(ref_date.year, ref_date.month, ref_date.day, h, mi, sec).strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    pass
            default = "23:59:59" if is_end else "00:00:00"
            return f"{ref_date.isoformat()}T{default}Z"

        today_str = _dt_date.today().isoformat()
        if not params.get("time_min"):
            params["time_min"] = f"{today_str}T00:00:00Z"
        elif "{{" not in str(params["time_min"]):
            params["time_min"] = _ge_parse_time(params["time_min"], is_end=False)

        if not params.get("time_max"):
            params["time_max"] = f"{today_str}T23:59:59Z"
        elif "{{" not in str(params["time_max"]):
            params["time_max"] = _ge_parse_time(params["time_max"], is_end=True)

        if not params.get("calendar_id"):
            params["calendar_id"] = "primary"

    # ── Jira update_issue: extract issue_key + ensure fields is dict ────────────
    if tool_name in ("update_issue", "jira_update_issue"):
        # Log tool schema once so we know what params it accepts
        try:
            from dqe_agent.tools import get_tool as _gt_schema

            _t = _gt_schema(tool_name)
            if _t and hasattr(_t, "args_schema") and _t.args_schema:
                _schema_keys = list(_t.args_schema.model_json_schema().get("properties", {}).keys())
                logger.info("[EXECUTOR] jira_update_issue schema keys: %s", _schema_keys)
        except Exception:
            pass
        _ik = params.get("issue_key")
        # Extract key string from complex create result objects
        if not isinstance(_ik, str) or not _ik or "{{" in str(_ik):
            _extracted = ""
            raw = _ik
            if isinstance(raw, list) and raw:
                raw = raw[0]
            if isinstance(raw, dict):
                _extracted = (
                    raw.get("key")
                    or (raw.get("issue") or {}).get("key")
                    or (raw.get("_items") or [{}])[0].get("issue", {}).get("key", "")
                    if isinstance(raw.get("_items"), list)
                    else ""
                )
            if not _extracted:
                # Fallback: scan all prior step results for a successful create
                import re as _re_key

                _key_pat = _re_key.compile(r"\b([A-Z][A-Z0-9]{0,9}-\d+)\b")
                for sr in reversed(list((results_by_id or {}).values())):
                    if sr.get("tool") not in ("jira_create_issue", "create_issue"):
                        continue
                    _res_str = str(sr.get("result", ""))
                    _km = _key_pat.search(_res_str)
                    if _km:
                        _extracted = _km.group(1)
                        break
            if _extracted:
                params["issue_key"] = _extracted
                logger.info("[EXECUTOR] jira_update_issue: resolved issue_key=%r", _extracted)
        # Parse fields string to dict for template resolution + flattening below
        if isinstance(params.get("fields"), str):
            try:
                params["fields"] = json.loads(params["fields"])
            except (json.JSONDecodeError, TypeError):
                pass
        if not params.get("fields"):
            return params  # nothing to update — skip

        # Deep-resolve {{template}} placeholders inside the fields dict AND flatten
        # nested Jira-style dicts to plain strings.
        #
        # The mcp-atlassian update_issue tool expects a FLAT fields dict:
        #   {"priority": "P1", "assignee": "accountId..."}
        # NOT the Jira REST API nested form:
        #   {"priority": {"name": "P1"}, "assignee": {"accountId": "..."}}
        # Passing nested dicts causes _get_account_id() to crash with AttributeError.
        if isinstance(params.get("fields"), dict):
            _rf: dict = {}
            for _fk, _fv in params["fields"].items():
                logger.info("[EXECUTOR] jira_update_issue field[%r] raw_value=%r (type=%s)", _fk, str(_fv)[:200], type(_fv).__name__)
                # Resolve any template strings first
                if isinstance(_fv, str) and "{{" in _fv:
                    _fv = _resolve_template(_fv, flow_data, results_by_id)
                    logger.info("[EXECUTOR] jira_update_issue field[%r] after_template=%r", _fk, str(_fv)[:200])
                elif isinstance(_fv, dict):
                    # Resolve templates in nested dict values
                    _fv = {
                        _ik2: _resolve_template(_iv, flow_data, results_by_id)
                        if isinstance(_iv, str) and "{{" in _iv
                        else _iv
                        for _ik2, _iv in _fv.items()
                    }

                # Safety net: if template resolved to a JSON-encoded list/object, extract the value
                logger.info("[EXECUTOR] jira_update_issue field[%r] pre_safety_net=%r (type=%s)", _fk, str(_fv)[:200], type(_fv).__name__)
                if isinstance(_fv, str) and _fv.strip().startswith(("[", "{")):
                    try:
                        _parsed_fv = json.loads(_fv)
                        if isinstance(_parsed_fv, list) and _parsed_fv:
                            _first = _parsed_fv[0]
                            if isinstance(_first, dict):
                                # Selection option {"value":..,"label":..} → use "value"
                                if "value" in _first and "label" in _first:
                                    _fv = str(_first.get("value", "")).strip()
                                else:
                                    # Form result → try to extract this specific field name
                                    _fv = str(_first.get(_fk, "")).strip()
                            else:
                                _fv = str(_first).strip()
                        elif isinstance(_parsed_fv, dict):
                            _fv = _parsed_fv  # hand off to flatten block below
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Flatten {"name": "..."} / {"accountId": "..."} / {"id": "..."} to plain string
                if isinstance(_fv, dict):
                    _flat = (
                        _fv.get("accountId")  # assignee Cloud
                        or _fv.get("name")  # priority / status / etc.
                        or _fv.get("id")  # numeric id fallback
                        or _fv.get("key")  # project/parent key fallback
                    )
                    if _flat is not None:
                        _fv = str(_flat).strip()

                # Drop empty values (user left field blank)
                if isinstance(_fv, str) and not _fv.strip():
                    logger.info("[EXECUTOR] jira_update_issue: skipping empty field %r", _fk)
                    continue

                _rf[_fk] = _fv
            params["fields"] = _rf

        # MCP update_issue expects fields as a JSON string, not a dict
        if isinstance(params.get("fields"), dict):
            params["fields"] = json.dumps(params["fields"])

        logger.info(
            "[EXECUTOR] jira_update_issue: final issue_key=%r  fields=%s",
            params.get("issue_key"),
            params.get("fields", "{}"),
        )

    # ── jira_get_transitions + jira_transition_issue: coerce list params to str ─
    if tool_name in (
        "jira_transition_issue",
        "transition_issue",
        "jira_get_transitions",
        "get_transitions",
    ):
        # issue_key might be a multi-select result list: [{'selected': 'FLAG-33', ...}]
        ikey = params.get("issue_key")
        if isinstance(ikey, list):
            # Extract string from first element's 'selected' or 'answer' field
            extracted_key = ""
            for item in ikey:
                if isinstance(item, dict):
                    extracted_key = str(
                        item.get("selected") or item.get("answer") or item.get("key") or ""
                    ).strip()
                elif isinstance(item, str):
                    extracted_key = item.strip()
                if extracted_key:
                    break
            if extracted_key:
                logger.info(
                    "[EXECUTOR] %s: extracted issue_key=%r from multi-select list (%d items)",
                    tool_name,
                    extracted_key,
                    len(ikey),
                )
                params["issue_key"] = extracted_key
            else:
                logger.warning(
                    "[EXECUTOR] %s: could not extract issue_key from list: %s", tool_name, ikey
                )

        # transition_id coercion — only for transition_issue
        if tool_name in ("jira_transition_issue", "transition_issue"):
            tid = params.get("transition_id")
            if isinstance(tid, list):
                # Planner passed the full transitions list instead of a single ID string.
                # Use _target_status_hint (saved before stripping) to pick the right transition.
                _STATUS_ALIASES: dict[str, list[str]] = {
                    "done": ["done", "complete", "closed", "resolved", "finish"],
                    "in progress": ["in progress", "inprogress", "start", "working", "active", "doing"],
                    "in review": ["in review", "review", "pr review", "code review"],
                    "to do": ["to do", "todo", "backlog", "open", "reopen"],
                    "testing": ["testing", "qa", "test", "in testing"],
                    "on hold": ["on hold", "onhold", "blocked", "impediment", "hold"],
                    "cancelled": ["cancelled", "canceled", "cancel"],
                }
                matched_id = None
                if _target_status_hint:
                    for entry in tid:
                        if not isinstance(entry, dict):
                            continue
                        t_name = str(entry.get("name", "")).lower()
                        t_id = str(entry.get("id", ""))
                        if not t_id:
                            continue
                        # Direct substring match first
                        if _target_status_hint in t_name or t_name in _target_status_hint:
                            matched_id = t_id
                            break
                        # Alias match
                        for canonical, aliases in _STATUS_ALIASES.items():
                            if any(a in _target_status_hint for a in aliases):
                                if any(a in t_name for a in aliases):
                                    matched_id = t_id
                                    break
                        if matched_id:
                            break
                if matched_id:
                    logger.info(
                        "[EXECUTOR] jira_transition_issue: matched transition_id=%r (hint=%r) from %d-item list",
                        matched_id, _target_status_hint, len(tid),
                    )
                    params["transition_id"] = matched_id
                else:
                    logger.warning(
                        "[EXECUTOR] jira_transition_issue: transition list received but could not match hint=%r — "
                        "planner should use request_selection for status",
                        _target_status_hint,
                    )
            elif isinstance(tid, (int, float)):
                params["transition_id"] = str(int(tid))

    # ── Gmail tools: auto-inject user_google_email ───────────────────────────
    _GMAIL_TOOLS = {
        "send_gmail_message",
        "search_gmail_messages",
        "get_gmail_message_content",
        "get_gmail_messages_content_batch",
        "draft_gmail_message",
        "get_gmail_thread_content",
        "modify_gmail_message_labels",
        "batch_modify_gmail_message_labels",
    }
    if tool_name in _GMAIL_TOOLS:
        from dqe_agent.config import settings as _cs

        _gm_email = str(params.get("user_google_email", ""))
        if not _gm_email or "{{" in _gm_email:
            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] %s: injected user_google_email from settings", tool_name)

    # ── send_gmail_message: safe recipient handling + meet-link extraction ────
    if tool_name == "send_gmail_message":
        _to = params.get("to", "")
        # If 'to' is a list, join to comma-separated string
        if isinstance(_to, list):
            _to = ", ".join(str(e) for e in _to if e)
            params["to"] = _to
        # Strip any unresolved templates from body/subject
        _body = str(params.get("body", ""))
        # Replace unresolved {{create_evt.hangoutLink}} with actual link from results
        if "hangoutLink" in _body or "meet_link" in _body or "meetLink" in _body:
            _meet_link = ""
            for _sid, _sres in results_by_id.items():
                if isinstance(_sres, dict) and _sres.get("status") == "success":
                    _raw_res = _sres.get("result", "")
                    if isinstance(_raw_res, str):
                        try:
                            _raw_res = json.loads(_raw_res)
                        except Exception:
                            pass
                    if isinstance(_raw_res, dict):
                        _meet_link = _raw_res.get("hangoutLink") or _raw_res.get("meet_link") or ""
                    if not _meet_link and isinstance(_raw_res, str):
                        import re as _re_ml
                        _mlm = _re_ml.search(r"https://meet\.google\.com/[a-z0-9\-]+", _raw_res)
                        if _mlm:
                            _meet_link = _mlm.group(0)
                    if _meet_link:
                        break
            if _meet_link:
                import re as _re_ml2
                _body = _re_ml2.sub(r"\{\{[^}]*hangoutLink[^}]*\}\}", _meet_link, _body)
                _body = _re_ml2.sub(r"\{\{[^}]*meet_?[Ll]ink[^}]*\}\}", _meet_link, _body)
                params["body"] = _body
                logger.info("[EXECUTOR] send_gmail_message: injected meet link %r", _meet_link)

    # ── llm_draft_content: ensure context and topic are strings ──────────────
    if tool_name == "llm_draft_content":

        def _to_str(val: Any) -> str:
            if isinstance(val, str):
                return val
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            if isinstance(val, dict):
                return "; ".join(
                    f"{k}: {v}" for k, v in val.items() if k not in ("_items", "_list") and v
                )
            return str(val) if val is not None else ""

        if not isinstance(params.get("context"), str):
            params["context"] = _to_str(params.get("context"))
        if not isinstance(params.get("topic"), str):
            params["topic"] = _to_str(params.get("topic"))
        # Replace accountId in context/topic with display name from fetch_users result
        import re as _re_aid

        _AID_PAT = _re_aid.compile(r"\d+:[0-9a-f\-]{30,}")

        def _replace_account_ids(text: str) -> str:
            if not _AID_PAT.search(text):
                return text
            _users_entry = (results_by_id or {}).get("fetch_users", {})
            _users_raw = _users_entry.get("result", "") if isinstance(_users_entry, dict) else ""
            try:
                _users_list = json.loads(_users_raw) if isinstance(_users_raw, str) else _users_raw
            except Exception:
                _users_list = []
            if not isinstance(_users_list, list):
                _items = _users_list.get("_items", []) if isinstance(_users_list, dict) else []
                _users_list = _items
            _aid_map = {
                u["value"]: u["label"]
                for u in _users_list
                if isinstance(u, dict) and "value" in u and "label" in u
            }
            return _AID_PAT.sub(lambda m: _aid_map.get(m.group(0), m.group(0)), text)

        if params.get("context"):
            params["context"] = _replace_account_ids(params["context"])
        if params.get("topic"):
            params["topic"] = _replace_account_ids(params["topic"])

    return params


def _find_selected_value(
    results_by_id: dict,
    flow_data: dict,
    priority_words: list[str],
    transform=None,
    validate=None,
) -> str:
    """Like _find_ask_answer but also checks the 'selected' field from request_selection results."""

    def _score(sid: str) -> int:
        sid_lower = sid.lower()
        return sum(1 for w in priority_words if w in sid_lower)

    candidates = sorted(results_by_id.keys(), key=_score, reverse=True)
    for sid in candidates:
        if _score(sid) == 0:
            break
        r = results_by_id[sid]
        raw = r.get("result", "")
        # Parse JSON result (request_selection returns {"selected": "FLAG", "answer": "FLAG"})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        # Try "selected" first, then "answer"
        if isinstance(raw, dict):
            for field in ("selected", "answer"):
                val = str(raw.get(field, "")).strip()
                if not val:
                    continue
                if transform:
                    val = transform(val)
                if validate and not validate(val):
                    continue
                return val
        # Plain string result
        elif isinstance(raw, str) and raw:
            val = raw.strip()
            if transform:
                val = transform(val)
            if not validate or validate(val):
                return val

    # Fallback: flow_data (which mirrors step results)
    fd_candidates = sorted(flow_data.keys(), key=_score, reverse=True)
    for sid in fd_candidates:
        if _score(sid) == 0:
            break
        entry = flow_data[sid]
        if not isinstance(entry, dict):
            continue
        for field in ("selected", "answer"):
            val = str(entry.get(field, "")).strip()
            if not val:
                continue
            if transform:
                val = transform(val)
            if validate and not validate(val):
                continue
            return val

    return ""


def _find_ask_answer(
    results_by_id: dict,
    flow_data: dict,
    priority_words: list[str],
    transform=None,
    validate=None,
) -> str:
    """Search step results for an ask_user answer whose step-id contains the priority words.

    Falls back to flow_data if results_by_id doesn't yield a match.
    Returns the first matching answer (optionally transformed and validated), or ''.
    """

    def _score(sid: str) -> int:
        sid_lower = sid.lower()
        return sum(1 for w in priority_words if w in sid_lower)

    # Sort step ids by how many priority words appear in them (descending)
    candidates = sorted(results_by_id.keys(), key=_score, reverse=True)
    for sid in candidates:
        if _score(sid) == 0:
            break  # no more matching step ids
        answer = _extract_ask_user_answer(results_by_id[sid])
        if not answer:
            continue
        if transform:
            answer = transform(answer)
        if validate and not validate(answer):
            continue
        return answer

    # Fallback: scan flow_data (which mirrors step results as dicts)
    fd_candidates = sorted(flow_data.keys(), key=_score, reverse=True)
    for sid in fd_candidates:
        if _score(sid) == 0:
            break
        entry = flow_data[sid]
        if not isinstance(entry, dict):
            continue
        answer = str(entry.get("answer", ""))
        if not answer:
            continue
        if transform:
            answer = transform(answer)
        if validate and not validate(answer):
            continue
        return answer

    return ""


def _resolve_params(params: dict, flow_data: dict, results_by_id: dict) -> dict:
    """Replace {{step_id.field}} references in params with actual values.

    For pure {{ref}} templates (no surrounding text), returns the actual Python
    object (list / dict) rather than a stringified version, so that params like
    ``options: "{{fetch_boards.boards}}"`` resolve to an actual list instead of
    a string, avoiding Pydantic validation errors downstream.
    """
    import re as _re

    _pure_ref = _re.compile(r"^\{\{(.+?)\}\}$")

    resolved = {}
    for key, val in params.items():
        if isinstance(val, str) and "{{" in val:
            m = _pure_ref.match(val.strip())
            if m:
                # Pure reference — try to get the real object first
                obj = _resolve_ref_to_object(m.group(1).strip(), flow_data, results_by_id)
                if obj is not None:
                    resolved[key] = obj
                    continue
            # Fallback: string interpolation
            resolved[key] = _resolve_template(val, flow_data, results_by_id)
        else:
            resolved[key] = val
    return resolved


def _resolve_ref_to_object(ref: str, flow_data: dict, results_by_id: dict) -> Any:
    """Resolve a dot-path reference to the actual Python object (list/dict/str).

    Tries step_results first, then flow_data.
    Returns None if the path cannot be resolved.
    """
    parts = ref.split(".")
    step_id = parts[0]

    def _navigate(root: Any, path_parts: list) -> Any:
        obj = root
        for p in path_parts:
            if isinstance(obj, dict):
                val = obj.get(p)
                if val is None:
                    # Key not found at top level — try nested "issue" dict first
                    # (e.g. jira_create_issue returns {"message":..., "issue":{"key":...}})
                    _nested = obj.get("issue", {})
                    if isinstance(_nested, dict) and _nested.get(p) is not None:
                        obj = _nested[p]
                        continue
                    # Key truly not found — try to extract a list from the dict
                    # (e.g. template says .boards but data is under .values / ._items)
                    extracted = _extract_items_from_response(obj)
                    return extracted if extracted else None
                obj = val
            elif isinstance(obj, list) and p.isdigit():
                obj = obj[int(p)]
            elif isinstance(obj, list):
                # Path segment is not an index but root is already a list —
                # the template named a sub-key that doesn't exist on a list.
                # The list itself IS the data (e.g. boards list from MCP).
                return obj
            else:
                return None
        return obj

    # ── Try results_by_id ────────────────────────────────────────────────────
    if step_id in results_by_id:
        raw = results_by_id[step_id].get("result", None)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        if len(parts) == 1:
            return raw
        result = _navigate(raw, parts[1:])
        if result is not None:
            return result

    # ── Try flow_data ────────────────────────────────────────────────────────
    obj = flow_data
    for p in parts:
        if isinstance(obj, dict):
            obj = obj.get(p)
        elif isinstance(obj, list) and p.isdigit():
            obj = obj[int(p)]
        else:
            return None
    return obj


def _resolve_template(template: str, flow_data: dict, results_by_id: dict) -> str:
    """Replace all {{ref}} placeholders in a string."""
    import re

    def replacer(match):
        ref = match.group(1).strip()
        parts = ref.split(".")
        # Try step_results first
        if parts[0] in results_by_id:
            r = results_by_id[parts[0]].get("result", {})
            # Parse JSON string to dict/list if possible
            if isinstance(r, str):
                try:
                    r = json.loads(r)
                except (json.JSONDecodeError, TypeError):
                    pass  # r stays as plain string
            if len(parts) == 1:
                # Whole result — return as-is
                if isinstance(r, (dict, list)):
                    return json.dumps(r, indent=2, ensure_ascii=False)
                return str(r) if r else match.group(0)
            # Sub-path navigation
            if isinstance(r, dict):
                nav: Any = r
                _key_missing = False
                for p in parts[1:]:
                    if isinstance(nav, dict):
                        if p in nav:
                            nav = nav[p]
                        else:
                            _key_missing = True
                            break
                    else:
                        _key_missing = True
                        break
                if not _key_missing:
                    # Key found — return value even if empty string (don't fall back to full JSON)
                    if isinstance(nav, (dict, list)):
                        return json.dumps(nav, indent=2, ensure_ascii=False)
                    return str(nav) if nav is not None else ""
                # Key not found — fall back to full result so user sees the data
                items = _extract_items_from_response(r)
                if items:
                    return json.dumps(items, indent=2, ensure_ascii=False)
                return json.dumps(r, indent=2, ensure_ascii=False) if r else match.group(0)
            elif isinstance(r, list):
                # If sub-path requested (e.g. {{find.accountId}}) and result is a list of dicts,
                # pull the field from the first element — common for jira_search_users results.
                if len(parts) > 1:
                    field = parts[-1]
                    for item in r:
                        if isinstance(item, dict) and field in item:
                            return str(item[field])
                return json.dumps(r, indent=2, ensure_ascii=False)
            elif isinstance(r, str) and r:
                # Plain text result — sub-path not applicable, return the text directly
                return r
        # Try flow_data
        val: Any = flow_data
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p, "")
            else:
                return match.group(0)
        return str(val) if val else match.group(0)

    return re.sub(r"\{\{(.+?)\}\}", replacer, template)
