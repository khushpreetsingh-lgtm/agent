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
    if all(isinstance(item, dict) and item.get("type") == "text" and "text" in item for item in raw):
        combined = "\n".join(item["text"] for item in raw)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return combined  # return as string — still better than Python repr
    return raw


async def executor_node(state: AgentState) -> dict:
    """Execute the current step in the plan."""
    from dqe_agent.tools import get_tool

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    steps_taken = state.get("steps_taken", 0)
    cost = state.get("estimated_cost", 0.0)

    if idx >= len(plan):
        return {"status": "complete", "messages": [AIMessage(content="All steps completed.")]}

    step = plan[idx]
    step_id = step.get("id", f"step_{idx}")
    tool_name = step.get("tool", "")
    params = step.get("params", {})
    description = step.get("description", "")

    logger.info("[EXECUTOR] Step %d/%d: [%s] %s", idx + 1, len(plan), tool_name, description)

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
            # If the template resolved to raw JSON, pretty-print it for the user
            if response_msg and response_msg.strip().startswith(("[", "{")):
                try:
                    parsed = json.loads(response_msg)
                    response_msg = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            response_msg = raw_msg

        return {
            "step_results": [{
                "step_id": step_id, "step_index": idx, "tool": "direct_response",
                "status": "success", "result": response_msg, "error": "",
                "duration_ms": 0, "retries": 0,
            }],
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
            "messages": [AIMessage(content=f"Stopped: reached {settings.max_steps}-step safety limit.")],
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
        _templ_m = _re_temp.match(r'^\{\{(.+?)\}\}$', params["options"].strip())
        _suffix = None
        if _templ_m and "." in _templ_m.group(1):
            _suffix = _templ_m.group(1).split('.')[-1]
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
            if isinstance(raw, dict) and _suffix and _suffix in raw and isinstance(raw[_suffix], list):
                items = raw[_suffix]
            else:
                items = _extract_items_from_response(raw)
            opts = _items_to_options(items)
            if opts:
                params = dict(params)
                params["options"] = opts
                logger.info(
                    "[EXECUTOR] Pre-resolved request_selection options: %d items from step '%s'",
                    len(opts), sr.get("step_id", "?"),
                )
                break

    resolved_params = _resolve_params(params, flow_data, results_by_id)

    # Strip MCP-internal params that the framework injects — callers must not pass them
    for _internal in ("ctx", "kwargs"):
        resolved_params.pop(_internal, None)

    # Warn about params that still contain unresolved {{}} templates
    _unresolved = [k for k, v in resolved_params.items() if isinstance(v, str) and "{{" in v]
    if _unresolved:
        logger.warning("[EXECUTOR] Unresolved template params for step '%s': %s", step_id, _unresolved)

    logger.debug("[EXECUTOR] Resolved params for '%s': %s", step_id, resolved_params)

    # Tool-specific param normalization (pass context so recovery can scan step results)
    resolved_params = _normalize_tool_params(tool_name, resolved_params, flow_data, results_by_id)

    # Execute the tool directly
    start = time.time()
    status = "success"
    result: Any = None
    error_msg = ""

    try:
        tool = get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")

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
                logger.info("[EXECUTOR] Human review rejected at step %s — stopping workflow", step_id)
                return {
                    "status": "complete",
                    "step_results": [{
                        "step_id": step_id, "step_index": idx, "tool": tool_name,
                        "status": "rejected", "result": "User rejected — workflow stopped",
                        "error": "", "duration_ms": 0, "retries": 0,
                    }],
                    "steps_taken": steps_taken + 1,
                    "estimated_cost": cost,
                    "flow_data": dict(state.get("flow_data", {})),
                    "messages": [AIMessage(content=f"Workflow stopped at step '{step_id}' — user did not approve.")],
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
        error_msg = str(exc)
        logger.error("[EXECUTOR] Step %s failed: %s", step_id, error_msg)

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
                # Wrap lists so template references like {{step._items}} and
                # the whole list is accessible to _resolve_ref_to_object.
                merged_flow[step_id] = {"_items": parsed, "_list": parsed}
        except (json.JSONDecodeError, TypeError):
            pass

    msg_content = f"Step {idx+1}/{len(plan)} [{tool_name}]: {status}"
    if status == "failed":
        msg_content += f" — {error_msg[:200]}"

    return {
        "step_results": [step_result],
        "steps_taken": steps_taken + 1,
        "estimated_cost": cost + step_cost,
        "status": "verifying",
        "flow_data": merged_flow,
        "messages": [AIMessage(content=msg_content)],
    }


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
        item_id = str(item.get("id", item.get("boardId", item.get("key", "")))).strip()
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
        for k in ("values", "boards", "sprints", "items", "data",
                  "results", "issues", "projects", "_items", "_list"):
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
    tool_obj = _get_tool(tool_name)
    if tool_obj is None:
        return params
    try:
        schema = (
            tool_obj.args_schema.model_json_schema()
            if hasattr(tool_obj, "args_schema") and tool_obj.args_schema
            else {}
        )
        valid = set(schema.get("properties", {}).keys()) - {"ctx", "kwargs"}
        if not valid:
            return params
        invalid = [k for k in params if k not in valid]
        if invalid:
            logger.warning(
                "[EXECUTOR] '%s': stripping unknown params %s (not in schema)",
                tool_name, invalid,
            )
            params = {k: v for k, v in params.items() if k in valid}
    except Exception:
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
        for alias in ("start", "start_datetime", "start_date", "date_start",
                      "start_at", "begins_at"):
            if alias in params and "start_time" not in params:
                params["start_time"] = params.pop(alias)
                break
        for alias in ("end", "end_datetime", "end_date", "date_end",
                      "end_at", "ends_at"):
            if alias in params and "end_time" not in params:
                params["end_time"] = params.pop(alias)
                break

    # ── get_events / list_events: fix wrong time param names ─────────────────
    _CALENDAR_QUERY_TOOLS = {
        "get_events", "list_events", "calendar_get_events", "calendar_list_events",
    }
    if tool_name in _CALENDAR_QUERY_TOOLS:
        for alias in ("start_datetime", "start_date", "start", "date_start",
                      "from_date", "from"):
            if alias in params and "time_min" not in params:
                params["time_min"] = params.pop(alias)
                break
        for alias in ("end_datetime", "end_date", "end", "date_end",
                      "to_date", "to"):
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
        # If fields is a dict, convert to JSON string (tool expects str)
        if isinstance(params.get("fields"), dict):
            params["fields"] = json.dumps(params["fields"])

    # ── get_issue / jira_get_issue: issue_id → issue_key ─────────────────────
    if tool_name in ("get_issue", "jira_get_issue"):
        for alias in ("issue_id", "id", "ticket_id", "ticket"):
            if alias in params and "issue_key" not in params:
                params["issue_key"] = params.pop(alias)
                break

    return params


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

        def _is_valid_options(v: Any) -> bool:
            """True only if v is already a non-empty [{value,label}] list."""
            return (
                isinstance(v, list) and bool(v) and
                isinstance(v[0], dict) and "value" in v[0] and "label" in v[0]
            )

        opts = params.get("options")

        # Step 1: string → try to resolve to actual object
        if isinstance(opts, str):
            import re as _re
            m = _re.match(r'^\{\{(.+?)\}\}$', opts.strip())
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
                        len(transformed), opts,
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
                        len(transformed), sid,
                    )
                    params["options"] = transformed
                    break

        return params

    # ── Jira board/sprint fetch tools: auto-inject project_key ───────────────
    # The planner sometimes passes wrong param names (board_filter, project, etc.).
    # After stripping invalid params above, re-inject project_key if the tool
    # accepts it and we already have the key from a prior selection step.
    _JIRA_LIST_TOOLS = {
        "jira_get_agile_boards", "jira_get_boards", "jira_list_boards",
        "jira_get_sprints", "jira_list_sprints", "jira_get_active_sprints",
        "jira_get_issues_in_project", "jira_search_issues",
    }
    if tool_name in _JIRA_LIST_TOOLS and not params.get("project_key"):
        import re as _re2
        _key_pat2 = _re2.compile(r'^[A-Z][A-Z0-9]{1,9}$')
        recovered_key = _find_selected_value(
            results_by_id, flow_data,
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
                    _schema = _to.args_schema.model_json_schema() if hasattr(_to, "args_schema") and _to.args_schema else {}
                    if "project_key" in _schema.get("properties", {}):
                        logger.info(
                            "[EXECUTOR] '%s': injecting project_key=%r from prior selection",
                            tool_name, recovered_key,
                        )
                        params["project_key"] = recovered_key
                except Exception:
                    pass

    if tool_name == "jira_create_sprint":
        # start_date / end_date are required by the MCP schema.
        # Default to today → today+14 days if not provided or unresolved.
        from datetime import date, timedelta
        today = date.today()
        default_start = today.isoformat()
        default_end = (today + timedelta(days=14)).isoformat()

        def _is_date(v: Any) -> bool:
            if not isinstance(v, str):
                return False
            import re as _red
            return bool(_red.match(r'^\d{4}-\d{2}-\d{2}', v.strip()))

        if not _is_date(params.get("start_date")):
            logger.info("[EXECUTOR] jira_create_sprint: defaulting start_date to %s", default_start)
            params["start_date"] = default_start
        if not _is_date(params.get("end_date")):
            logger.info("[EXECUTOR] jira_create_sprint: defaulting end_date to %s", default_end)
            params["end_date"] = default_end

        # board_id: recover from prior selection if still a template
        if isinstance(params.get("board_id"), str) and "{{" in params["board_id"]:
            recovered_board = _find_selected_value(
                results_by_id, flow_data,
                priority_words=["board", "select"],
                validate=lambda v: v.strip().lstrip("-").isdigit(),
                transform=lambda v: v.strip(),
            )
            if recovered_board:
                logger.info("[EXECUTOR] jira_create_sprint: recovered board_id=%r from step results", recovered_board)
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
        _key_pat = _re.compile(r'^[A-Z][A-Z0-9]{1,9}$')
        if not _key_pat.match(params.get("project_key", "")):
            # Scan step results for a valid key — prefer steps named "select*" or "project*"
            # Also check the "selected" field (from request_selection JSON result)
            recovered = _find_selected_value(
                results_by_id, flow_data,
                priority_words=["select", "project", "key"],
                validate=lambda v: bool(_key_pat.match(v.strip().upper())),
                transform=lambda v: v.strip().upper(),
            )
            if recovered:
                logger.info("[EXECUTOR] jira_create_issue: recovered project_key=%r from step results", recovered)
                params["project_key"] = recovered
            else:
                logger.warning(
                    "[EXECUTOR] jira_create_issue: project_key=%r is invalid and could not be recovered",
                    params.get("project_key"),
                )

        # ── issue_type ────────────────────────────────────────────────────
        if not params.get("issue_type"):
            # Try to find an ask_user answer for issue type
            recovered_type = _find_ask_answer(
                results_by_id, flow_data,
                priority_words=["issue_type", "type"],
                transform=lambda v: v.strip().capitalize(),
            )
            if recovered_type:
                logger.info("[EXECUTOR] jira_create_issue: recovered issue_type=%r from step results", recovered_type)
                params["issue_type"] = recovered_type
            else:
                params["issue_type"] = "Task"
                logger.info("[EXECUTOR] jira_create_issue: defaulted issue_type to 'Task'")

        # ── summary ───────────────────────────────────────────────────────
        if not params.get("summary"):
            recovered_summary = _find_ask_answer(
                results_by_id, flow_data,
                priority_words=["summary", "title"],
            )
            if recovered_summary:
                logger.info("[EXECUTOR] jira_create_issue: recovered summary=%r from step results", recovered_summary)
                params["summary"] = recovered_summary

    # ── Google Calendar read tools: defaults + injection ─────────────────────
    _CALENDAR_EVENT_TOOLS = {"get_events", "list_events", "calendar_get_events", "calendar_list_events",
                             "list_calendars", "calendar_list", "get_calendars"}
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
            for bad in [k for k in list(params.keys()) if k not in {"user_google_email", "max_results"}]:
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
        if params.get("action") == "create" and params.get("start_time") and not params.get("end_time"):
            try:
                from datetime import datetime, timedelta
                dt = datetime.fromisoformat(params["start_time"].replace("Z", "+00:00"))
                dur = params.get("duration")
                if dur and isinstance(dur, str):
                    import re as _re_dur
                    m = _re_dur.search(r"(\d+)\s*(h|hr|hour|hours)", dur, _re_dur.I)
                    if m:
                        hours = int(m.group(1))
                        params["end_time"] = (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    else:
                        m = _re_dur.search(r"(\d+)\s*(m|min|minute|minutes)", dur, _re_dur.I)
                        if m:
                            mins = int(m.group(1))
                            params["end_time"] = (dt + timedelta(minutes=mins)).strftime("%Y-%m-%dT%H:%M:%SZ")
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
            if not _att_lower or _att_lower in ("none", "just me", "no one", "only me", "myself",
                                                 "no attendees", "no one else", "just myself"):
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

    # ── Jira update_issue: defaults + fields-as-JSON-string ──────────────────
    if tool_name in ("update_issue", "jira_update_issue"):
        if isinstance(params.get("fields"), dict):
            params["fields"] = json.dumps(params["fields"])
        if not params.get("fields"):
            params["fields"] = "{}"  # tool requires fields even if empty

    # ── jira_get_transitions + jira_transition_issue: coerce list params to str ─
    if tool_name in ("jira_transition_issue", "transition_issue",
                     "jira_get_transitions", "get_transitions"):

        # issue_key might be a multi-select result list: [{'selected': 'FLAG-33', ...}]
        ikey = params.get("issue_key")
        if isinstance(ikey, list):
            # Extract string from first element's 'selected' or 'answer' field
            extracted_key = ""
            for item in ikey:
                if isinstance(item, dict):
                    extracted_key = str(item.get("selected") or item.get("answer") or item.get("key") or "").strip()
                elif isinstance(item, str):
                    extracted_key = item.strip()
                if extracted_key:
                    break
            if extracted_key:
                logger.info(
                    "[EXECUTOR] %s: extracted issue_key=%r from multi-select list (%d items)",
                    tool_name, extracted_key, len(ikey),
                )
                params["issue_key"] = extracted_key
            else:
                logger.warning("[EXECUTOR] %s: could not extract issue_key from list: %s", tool_name, ikey)

        # transition_id coercion — only for transition_issue
        if tool_name in ("jira_transition_issue", "transition_issue"):
            tid = params.get("transition_id")
            if isinstance(tid, list):
                # Planner passed the full transitions list instead of a single ID string.
                # Auto-pick by matching "done/close/resolve" names, else first item.
                _DONE_NAMES = {"done", "closed", "close", "complete", "completed",
                               "resolve", "resolved", "finish", "finished"}
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
                        matched_id, len(tid),
                    )
                    params["transition_id"] = matched_id
                else:
                    logger.warning("[EXECUTOR] jira_transition_issue: could not extract id from list: %s", tid)
            elif isinstance(tid, (int, float)):
                params["transition_id"] = str(int(tid))

    # ── Gmail tools: auto-inject user_google_email ───────────────────────────
    _GMAIL_TOOLS = {
        "send_gmail_message", "search_gmail_messages", "get_gmail_message_content",
        "get_gmail_messages_content_batch", "draft_gmail_message",
        "get_gmail_thread_content", "modify_gmail_message_labels",
        "batch_modify_gmail_message_labels",
    }
    if tool_name in _GMAIL_TOOLS:
        from dqe_agent.config import settings as _cs
        _gm_email = str(params.get("user_google_email", ""))
        if not _gm_email or "{{" in _gm_email:
            if _cs.user_google_email:
                params["user_google_email"] = _cs.user_google_email
                logger.info("[EXECUTOR] %s: injected user_google_email from settings", tool_name)

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
    _pure_ref = _re.compile(r'^\{\{(.+?)\}\}$')

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
                    return str(nav) if not isinstance(nav, (dict, list)) else json.dumps(nav, indent=2, ensure_ascii=False)
                # Sub-key not found or empty — return full result so user sees the data
                items = _extract_items_from_response(r)
                if items:
                    return json.dumps(items, indent=2, ensure_ascii=False)
                return json.dumps(r, indent=2, ensure_ascii=False) if r else match.group(0)
            elif isinstance(r, list):
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
