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

logger = logging.getLogger(__name__)


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
                # Show a few key fields — id, name, key, summary, etc.
                for key in ("id", "name", "key", "summary", "sprint", "sprintId", "self"):
                    if key in parsed:
                        brief = f" → {key}: {parsed[key]}"
                        break
                if not brief:
                    first_val = next(iter(parsed.values()), "")
                    brief = f" → {str(first_val)[:80]}"
            elif isinstance(parsed, list):
                brief = f" → {len(parsed)} item(s) returned"
        except Exception:
            brief = f" → {str(result)[:80]}"

    label = tool.replace("jira_", "").replace("_", " ").title()
    return f"{prefix} {label}: {status}{brief}"


def _no_results_sentence(prefix: str) -> str:
    """Turn a prefix like 'Here are your open tasks' into a clean empty-state sentence."""
    p = prefix.lower().strip()
    # Strip common filler openers so we can inspect the subject
    for filler in ("here are your ", "here are the ", "your ", "the "):
        if p.startswith(filler):
            p = p[len(filler) :]
            break
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
    subject = prefix.strip().rstrip(":").strip() if prefix else "issues"
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

            # Line 1: key + summary
            title = f"**{key}** — {summary}" if key else f"**{summary}**"
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

        count = len(issues) if total < 0 else total
        header = f"**{count} issue{'s' if count != 1 else ''} found:**\n\n"
        return header + "\n\n".join(blocks)

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


async def executor_node(state: AgentState) -> dict:
    """Execute the current step in the plan."""
    from dqe_agent.tools import get_tool

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
                # Message may embed a JSON blob mid-text — format any trailing JSON block
                # e.g. "Your open tasks are: {...}"
                import re as _re_inline

                _m = _re_inline.search(r"(\{[\s\S]*\}|\[[\s\S]*\])\s*$", response_msg)
                if _m:
                    prefix = response_msg[: _m.start()].rstrip().rstrip(":").rstrip()
                    formatted = _format_result_for_display(_m.group(1))
                    if formatted == "No issues found.":
                        # Merge prefix + empty result into one natural sentence
                        response_msg = _no_results_sentence(prefix)
                    elif prefix:
                        response_msg = f"{prefix}:\n\n{formatted}"
                    else:
                        response_msg = formatted

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
    # If the planner put a {{ref}} template in options but the step-id in the
    # template doesn't exactly match any result, _resolve_params returns the raw
    # string and Pydantic validation fails.  Bypass that entirely: scan the
    # most recent successful non-interaction step result and use its list data.
    _INTERACTION_TOOLS = {"request_selection", "ask_user", "human_review_request"}
    if (
        tool_name == "request_selection"
        and isinstance(params.get("options"), str)
        and "{{" in params["options"]
    ):
        # If planner used a different step id in the template (e.g. '{{fetch_boards.boards}}')
        # try to extract the referenced field name (suffix) so we can match it
        # against any recent non-interaction step result that contains that key.
        import re as _re_temp

        _templ_m = _re_temp.match(r"^\{\{(.+?)\}\}$", params["options"].strip())
        _suffix = None
        if _templ_m and "." in _templ_m.group(1):
            _suffix = _templ_m.group(1).split(".")[-1]
        for sr in reversed(step_results):
            if not isinstance(sr, dict):
                continue
            if sr.get("tool") in _INTERACTION_TOOLS:
                continue
            if sr.get("status") != "success":
                continue
            raw = sr.get("result", "")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            # If the recent step returned a dict and the planner referenced a
            # specific sub-field (e.g. 'boards'), prefer that sub-field when
            # it's present to better match templates that used different ids.
            items = []
            if (
                isinstance(raw, dict)
                and _suffix
                and _suffix in raw
                and isinstance(raw[_suffix], list)
            ):
                items = raw[_suffix]
            else:
                items = _extract_items_from_response(raw)
            opts = _items_to_options(items)
            if opts:
                params = dict(params)
                params["options"] = opts
                logger.info(
                    "[EXECUTOR] Pre-resolved request_selection options: %d items from step '%s'",
                    len(opts),
                    sr.get("step_id", "?"),
                )
                break

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
                        _field["options"] = (
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

    # Warn about params that still contain unresolved {{}} templates
    _unresolved = [k for k, v in resolved_params.items() if isinstance(v, str) and "{{" in v]
    if _unresolved:
        logger.warning(
            "[EXECUTOR] Unresolved template params for step '%s': %s", step_id, _unresolved
        )

    logger.info(
        "[EXECUTOR] Resolved params for '%s': %s",
        step_id,
        json.dumps(resolved_params, default=str)[:500],
    )

    # Tool-specific param normalization (pass context so recovery can scan step results)
    resolved_params = _normalize_tool_params(tool_name, resolved_params, flow_data, results_by_id)

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
        # Tool doesn't exist — return a graceful direct response instead of failing the task.
        logger.warning("[EXECUTOR] Tool '%s' not found — responding gracefully", tool_name)
        msg = f"I don't have a tool to perform '{tool_name.replace('_', ' ')}' directly. I'll answer based on available information."
        return {
            "step_results": state.get("step_results", [])
            + [
                {
                    "step_id": step_id,
                    "step_index": idx,
                    "tool": "direct_response",
                    "status": "success",
                    "result": msg,
                    "error": "",
                    "duration_ms": 0,
                    "retries": 0,
                }
            ],
            "steps_taken": steps_taken + 1,
            "estimated_cost": cost,
            "status": "verifying",
            "flow_data": dict(state.get("flow_data", {})),
            "messages": [AIMessage(content=msg)],
        }

    try:
        result_raw = await tool.ainvoke(resolved_params)
        # Unwrap LangChain MCP content blocks so downstream code (pre-resolve,
        # flow_data storage, template resolution) always sees real JSON-serialisable data.
        result_raw = _unwrap_mcp_result(result_raw)

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
                merged_flow[step_id] = parsed
            elif isinstance(parsed, list):
                # Wrap lists so template references like {{step._items}} work.
                # Also hoist commonly-accessed fields to the top level.
                wrapped = {"_items": parsed, "_list": parsed}
                # For jira_create_issue: hoist key/id so {{step.key}} resolves
                if tool_name in ("jira_create_issue", "create_issue") and parsed:
                    _issue = parsed[0].get("issue", {}) if isinstance(parsed[0], dict) else {}
                    if isinstance(_issue, dict):
                        for _k in ("key", "id", "summary"):
                            if _issue.get(_k):
                                wrapped[_k] = _issue[_k]
                merged_flow[step_id] = wrapped
        except (json.JSONDecodeError, TypeError):
            pass

    msg_content = _step_message(
        idx, len(plan), tool_name, status, error_msg, resolved_params, result
    )

    return {
        "step_results": state.get("step_results", []) + [step_result],
        "steps_taken": steps_taken + 1,
        "estimated_cost": cost + step_cost,
        "status": "verifying",
        "flow_data": merged_flow,
        "messages": [AIMessage(content=msg_content)],
    }


def _resolve_prefetched_sentinel(sentinel: str) -> list | None:
    """Resolve <<SENTINEL>> tokens injected by the planner into actual option lists."""
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
        # For Jira issues and projects, prefer the 'key' field (e.g. 'FLAG-123') over numeric 'id'.
        # Boards/sprints have 'boardId' or 'id' but not 'key'.
        item_id = str(item.get("key", item.get("boardId", item.get("id", "")))).strip()
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

    # HARDCODED FIXES for known persistent hallucinations
    if tool_name == "search_gmail_messages":
        # search_gmail_messages ONLY supports 'query'
        if "limit" in params:
            logger.warning(
                "[EXECUTOR] Stripping persistent hallucinated 'limit' from search_gmail_messages"
            )
            params = {k: v for k, v in params.items() if k != "limit"}

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


def _pre_strip_remap(tool_name: str, params: dict) -> dict:
    """Remap wrong param names to correct ones BEFORE _strip_invalid_params runs.

    _strip_invalid_params removes anything not in the tool's schema. If the planner
    sends 'start' instead of 'start_time', the strip step would delete 'start' before
    tool-specific normalization can remap it. This function runs first.
    """
    params = dict(params)  # don't mutate caller's dict

    # ── manage_event: fix wrong param names ──────────────────────────────────
    if tool_name == "manage_event":
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
        # fields MUST be a dict — never convert to JSON string
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


def _normalize_tool_params(
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
    params = _pre_strip_remap(tool_name, params)

    # ── Universal: strip params not in the tool's schema ─────────────────────
    # Must run BEFORE tool-specific logic so the specific blocks see clean params.
    params = _strip_invalid_params(tool_name, params)

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

        # Step 0: resolve <<SENTINEL>> tokens from planner cache
        if isinstance(opts, str) and opts.strip().startswith("<<"):
            _sentinel_resolved = _resolve_prefetched_sentinel(opts.strip())
            if _sentinel_resolved:
                params["options"] = _sentinel_resolved
                opts = _sentinel_resolved
                logger.info(
                    "[EXECUTOR] request_selection: resolved sentinel %r → %d items",
                    opts,
                    len(_sentinel_resolved),
                )

        # Step 1: string → try to resolve to actual object
        if isinstance(opts, str):
            import re as _re

            m = _re.match(r"^\{\{(.+?)\}\}$", opts.strip())
            raw = None
            if m:
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

        # Step 3: still not valid — scan ALL recent step results as last resort
        if not _is_valid_options(params.get("options")):
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

    if tool_name == "jira_create_sprint":
        # start_date / end_date are required by the MCP schema.
        # Default to tomorrow → tomorrow+14 days to avoid Jira rejecting today as "in the past".
        # Use timezone-aware ISO format (UTC) to avoid offset-naive vs offset-aware comparison errors.
        from datetime import datetime, timezone, timedelta

        now_utc = datetime.now(timezone.utc)
        tomorrow_utc = now_utc + timedelta(days=1)
        default_start = tomorrow_utc.strftime("%Y-%m-%dT00:00:00+00:00")
        default_end = (tomorrow_utc + timedelta(days=14)).strftime("%Y-%m-%dT00:00:00+00:00")

        def _is_date(v: Any) -> bool:
            if not isinstance(v, str):
                return False
            import re as _red

            return bool(_red.match(r"^\d{4}-\d{2}-\d{2}", v.strip()))

        if not _is_date(params.get("start_date")):
            logger.info(
                "[EXECUTOR] jira_create_sprint: defaulting start_date to %s (tomorrow)",
                default_start,
            )
            params["start_date"] = default_start
        if not _is_date(params.get("end_date")):
            logger.info(
                "[EXECUTOR] jira_create_sprint: defaulting end_date to %s (tomorrow+14)",
                default_end,
            )
            params["end_date"] = default_end

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
            # Try to find an ask_user answer for issue type
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
            else:
                params["issue_type"] = "Task"
                logger.info("[EXECUTOR] jira_create_issue: defaulted issue_type to 'Task'")

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

    # ── jira_search: sanitize JQL + exclude description field (ADF bug in mcp-atlassian) ──
    if tool_name == "jira_search":
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

    # ── jira_get_assignable_users: resolve <<FIRST_PROJECT_KEY>> sentinel ───────
    if tool_name == "jira_get_assignable_users":
        pk = params.get("project_key", "")
        if not pk or (isinstance(pk, str) and ("{{" in pk or "<<" in pk or not pk.strip())):
            pk = _first_project_key()
            if pk:
                params["project_key"] = pk
                logger.info("[EXECUTOR] jira_get_assignable_users: using first project key %r", pk)

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
                    "in progress": ["in progress", "inprogress", "start", "working", "active"],
                    "in review": ["in review", "review", "pr review", "code review"],
                    "to do": ["to do", "todo", "backlog", "open", "reopen"],
                    "testing": ["testing", "qa", "test", "in testing"],
                    "blocked": ["blocked", "impediment"],
                }

                # Try to match against task description for target status
                target_hint = (flow_data or {}).get("_target_status", "")
                if not target_hint:
                    # Check description field from step
                    target_hint = str(params.get("_description", "")).lower()

                best_id = None
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

                if not best_id and transitions:
                    # Fallback: take the first non-initial transition
                    for t in transitions:
                        if isinstance(t, dict) and t.get("id"):
                            best_id = str(t["id"])
                            break

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

        # Default time_min/time_max to today if missing or wrong date
        def _is_valid_dt(v: Any) -> bool:
            return isinstance(v, str) and len(v) >= 10 and v[:4].isdigit()

        if not _is_valid_dt(params.get("time_min")):
            params["time_min"] = f"{today.isoformat()}T00:00:00Z"
            logger.info("[EXECUTOR] %s: defaulted time_min to today", tool_name)
        if not _is_valid_dt(params.get("time_max")):
            params["time_max"] = f"{today.isoformat()}T23:59:59Z"
            logger.info("[EXECUTOR] %s: defaulted time_max to today", tool_name)

        # Default calendar_id to 'primary' if missing or unresolved
        _cal_id = str(params.get("calendar_id", ""))
        if not _cal_id or "{{" in _cal_id:
            params["calendar_id"] = "primary"
            logger.info("[EXECUTOR] %s: defaulted calendar_id to 'primary'", tool_name)

    # ── manage_event: defaults + injection (remapping done in _pre_strip_remap)
    if tool_name == "manage_event":
        from dqe_agent.config import settings as _cs

        # Default action to "create" if missing
        if not params.get("action"):
            params["action"] = "create"
            logger.info("[EXECUTOR] manage_event: defaulted action to 'create'")

        # Auto-inject user_google_email
        _ev_email = str(params.get("user_google_email", ""))
        if not _ev_email or "{{" in _ev_email:
            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] manage_event: injected user_google_email from settings")

        # Default calendar_id to 'primary'
        _ev_cal = str(params.get("calendar_id", ""))
        if not _ev_cal or "{{" in _ev_cal:
            params["calendar_id"] = "primary"

        # Ensure start_time/end_time are RFC3339 (YYYY-MM-DDTHH:MM:SSZ)
        def _ensure_rfc3339(val: Any, default_time: str) -> str:
            import re as _re_ev

            s = str(val) if val else ""
            if not s or "{{" in s:
                return s
            if "T" in s:
                if not s.endswith("Z") and "+" not in s:
                    s = s[:19] + "Z" if len(s) > 19 else s + "Z"
                return s
            if _re_ev.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return f"{s}T{default_time}Z"
            return s

        if params.get("start_time"):
            params["start_time"] = _ensure_rfc3339(params["start_time"], "00:00:00")
        if params.get("end_time"):
            params["end_time"] = _ensure_rfc3339(params["end_time"], "01:00:00")

        # If the caller provided a human-friendly duration ("30 minutes", "1 hour"), parse it
        # and compute end_time from start_time. Otherwise default end_time to 1h after start.
        if (
            params.get("action") == "create"
            and params.get("start_time")
            and not params.get("end_time")
        ):
            try:
                from datetime import datetime, timedelta

                dt = datetime.fromisoformat(params["start_time"].replace("Z", "+00:00"))
                dur = params.get("duration")
                if dur and isinstance(dur, str):
                    import re as _re_dur

                    m = _re_dur.search(r"(\d+)\s*(h|hr|hour|hours)", dur, _re_dur.I)
                    if m:
                        hours = int(m.group(1))
                        params["end_time"] = (dt + timedelta(hours=hours)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    else:
                        m = _re_dur.search(r"(\d+)\s*(m|min|minute|minutes)", dur, _re_dur.I)
                        if m:
                            mins = int(m.group(1))
                            params["end_time"] = (dt + timedelta(minutes=mins)).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            )
                if not params.get("end_time"):
                    params["end_time"] = (dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    logger.info("[EXECUTOR] manage_event: defaulted end_time to 1h after start")
                else:
                    logger.info("[EXECUTOR] manage_event: computed end_time from duration")
            except Exception:
                pass

        # Normalize attendees: if it came back as a plain string from ask_user, parse it
        _att = params.get("attendees")
        if isinstance(_att, str):
            _att_lower = _att.strip().lower()
            if not _att_lower or _att_lower in (
                "none",
                "just me",
                "no one",
                "only me",
                "myself",
                "no attendees",
                "no one else",
                "just myself",
            ):
                params.pop("attendees", None)
                logger.info("[EXECUTOR] manage_event: removed 'just me' attendees")
            else:
                # Parse comma/semicolon-separated emails into a list
                import re as _re_att

                _emails = [e.strip() for e in _re_att.split(r"[,;]+", _att) if e.strip()]
                if _emails:
                    params["attendees"] = _emails
                    logger.info("[EXECUTOR] manage_event: parsed %d attendee(s)", len(_emails))

    # ── query_freebusy: auto-inject email + default to today working hours ─────
    if tool_name == "query_freebusy":
        from dqe_agent.config import settings as _cs
        from datetime import date

        _fb_email = str(params.get("user_google_email", ""))
        if not _fb_email or "{{" in _fb_email:
            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] query_freebusy: injected user_google_email")
        today = date.today().isoformat()
        if not params.get("time_min"):
            params["time_min"] = f"{today}T09:00:00Z"
        if not params.get("time_max"):
            params["time_max"] = f"{today}T18:00:00Z"
        # Default calendar_ids to primary if missing
        if not params.get("calendar_ids"):
            params["calendar_ids"] = ["primary"]

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
        # fields MUST be a dict — if it arrived as a string, parse it
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
                # Resolve any template strings first
                if isinstance(_fv, str) and "{{" in _fv:
                    _fv = _resolve_template(_fv, flow_data, results_by_id)
                elif isinstance(_fv, dict):
                    # Resolve templates in nested dict values
                    _fv = {
                        _ik2: _resolve_template(_iv, flow_data, results_by_id)
                        if isinstance(_iv, str) and "{{" in _iv
                        else _iv
                        for _ik2, _iv in _fv.items()
                    }

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

        logger.info(
            "[EXECUTOR] jira_update_issue: final issue_key=%r  fields=%s",
            params.get("issue_key"),
            json.dumps(params.get("fields", {})),
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
                # Auto-pick by matching "done/close/resolve" names, else first item.
                _DONE_NAMES = {
                    "done",
                    "closed",
                    "close",
                    "complete",
                    "completed",
                    "resolve",
                    "resolved",
                    "finish",
                    "finished",
                }
                matched_id = None
                for entry in tid:
                    if isinstance(entry, dict):
                        name = str(entry.get("name", "")).lower()
                        if any(d in name for d in _DONE_NAMES):
                            matched_id = str(entry.get("id", ""))
                            break
                if not matched_id and tid and isinstance(tid[0], dict):
                    matched_id = str(tid[0].get("id", ""))
                if matched_id:
                    logger.info(
                        "[EXECUTOR] jira_transition_issue: auto-extracted transition_id=%r from %d-item list",
                        matched_id,
                        len(tid),
                    )
                    params["transition_id"] = matched_id
                else:
                    logger.warning(
                        "[EXECUTOR] jira_transition_issue: could not extract id from list: %s", tid
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
                    # Key not found — try to extract a list from the dict instead
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
                nav = r
                for p in parts[1:]:
                    nav = nav.get(p, "") if isinstance(nav, dict) else ""
                if nav:
                    return (
                        str(nav)
                        if not isinstance(nav, (dict, list))
                        else json.dumps(nav, indent=2, ensure_ascii=False)
                    )
                # Sub-key not found or empty — return full result so user sees the data
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
