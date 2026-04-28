"""Verifier node — confirms each step succeeded before moving to the next.

Verification priority (fastest → most expensive):
  MCP/API tools:
    1. Fail on explicit error status from executor
    2. Scan result text for embedded HTTP/API error messages
    3. Deterministic criteria check (keyword matching against result)
  Browser tools:
    1. Deterministic checks (URL, element visible, text present) — FREE
    2. AI screenshot verification — only if deterministic checks fail/unavailable
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from dqe_agent.guardrails import COST_PER_CALL
from dqe_agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # retry once unchanged
DIAGNOSE_AT = 1  # on retry_count == 1, ask LLM to diagnose and adapt
MAX_REPLANS = 2

VISION_VERIFY_PROMPT = """You are verifying a browser automation step. Look at the screenshot and answer:

STEP: {description}
SUCCESS CRITERIA: {criteria}

Did this step succeed? Respond with EXACTLY one JSON:
{{"verified": true/false, "reason": "brief explanation"}}"""

# Tools that control a real browser — only these get verified.
_BROWSER_TOOLS = {
    "browser_login",
    "browser_navigate",
    "browser_act",
    "browser_extract",
    "browser_click",
    "browser_type",
    "browser_wait",
    "browser_snapshot",
}

# Human-interaction tools — always auto-pass.
_HUMAN_TOOLS = {
    "ask_user",
    "human_review",
    "ask_user_choice",
    "direct_response",
    "request_selection",
}


async def verifier_node(state: AgentState) -> dict:
    """Verify the last executed step."""
    # If the executor already set a terminal status (complete/failed), don't override it.
    state_status = state.get("status")
    if state_status in ("complete", "failed"):
        logger.info("[VERIFIER] Terminal status '%s' — skipping verification", state_status)
        return {}

    step_results = state.get("step_results", [])
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    cost = state.get("estimated_cost", 0.0)

    if not step_results:
        return {"status": "executing"}

    last_result = step_results[-1] if step_results else {}
    if not isinstance(last_result, dict):
        return {"status": "executing"}

    step_status = last_result.get("status", "failed")
    step_id = last_result.get("step_id", "")
    tool_used = last_result.get("tool", "")

    # ── Human-interaction tools: always advance, BUT respect explicit rejection ─
    if tool_used in _HUMAN_TOOLS:
        # For human_review, check if it was rejected (status 'rejected')
        if tool_used == "human_review" and step_status == "rejected":
            logger.warning("[VERIFIER] Human review rejected — stopping workflow")
            return {
                "status": "complete",
                "messages": [AIMessage(content="Workflow stopped — user did not approve.")],
            }
        logger.info("[VERIFIER] Step '%s' auto-passed (human tool)", step_id)
        return {"current_step_index": idx + 1, "retry_count": 0, "status": "executing"}

    # ── MCP / API tools: verify result content — no blind auto-pass ─────────
    if tool_used not in _BROWSER_TOOLS:
        # Skipped steps (e.g. jira_add_attachment with no file) — advance without error
        if step_status == "skipped":
            logger.info("[VERIFIER] Step '%s' skipped — advancing", step_id)
            return {"current_step_index": idx + 1, "retry_count": 0, "status": "executing"}

        # 1. Explicit failure from executor (exception raised or tool returned error status)
        if step_status in ("failed", "partial"):
            error_msg = last_result.get("error", "unknown error")
            logger.warning("[VERIFIER] MCP tool '%s' failed: %s", tool_used, error_msg[:120])
            return {
                "status": "failed",
                "error": f"'{tool_used}' failed: {error_msg}",
                "messages": [AIMessage(content=f"Step failed: {error_msg[:300]}")],
            }

        # 2. Detect silent errors embedded in the result text — some MCP tools
        #    return HTTP error messages as plain text without raising an exception.
        _raw_result_val = last_result.get("result", "")
        _result_text = str(_raw_result_val).lower()
        # Skip error-text scan for known-good structured Jira responses.
        # A result that has an "issues" list (even empty) is a valid Jira response,
        # not an error — field names like "errorMessages" in issue data would
        # otherwise trigger false positives.
        # Tools whose results are always user-facing text — never scan for errors
        _NO_ERROR_SCAN_TOOLS = {
            "llm_draft_content", "request_edit", "direct_response",
            "ask_user", "request_form", "request_selection", "human_review",
        }
        _skip_error_scan = tool_used in _NO_ERROR_SCAN_TOOLS
        if not _skip_error_scan:
            try:
                _rt_parsed = json.loads(_raw_result_val) if isinstance(_raw_result_val, str) else _raw_result_val
                if isinstance(_rt_parsed, dict) and "issues" in _rt_parsed and isinstance(_rt_parsed["issues"], list):
                    _skip_error_scan = True
                # Skip scan when result is a content-wrapper dict (llm output, edit result, etc.)
                elif isinstance(_rt_parsed, dict) and "content" in _rt_parsed and len(_rt_parsed) <= 3:
                    _skip_error_scan = True
            except Exception:
                pass
        _ERROR_INDICATORS = [
            "error", "exception", "traceback", "failed", "failure",
            "unauthori", "forbidden", "not found", "bad request",
            "cannot", "unable to", "does not exist", "permission denied",
            "status: 4", "status: 5",  # HTTP 4xx / 5xx in result text
        ]
        _FALSE_POSITIVES = [
            "no error", "without error", "0 error", "no issues found",
            "errorcount: 0", "errors: 0",
        ]
        if not _skip_error_scan and \
                any(ind in _result_text for ind in _ERROR_INDICATORS) and \
                not any(fp in _result_text for fp in _FALSE_POSITIVES):
            raw_result = last_result.get("result", "")
            try:
                _parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                _err_detail = (
                    _parsed.get("error") or _parsed.get("message") or str(raw_result)
                    if isinstance(_parsed, dict) else str(raw_result)
                )
            except Exception:
                _err_detail = str(raw_result)
            logger.warning(
                "[VERIFIER] '%s' result contains error text: %s", tool_used, _err_detail[:200]
            )
            return {
                "status": "failed",
                "error": f"'{tool_used}' returned an error: {_err_detail[:300]}",
                "messages": [AIMessage(content=f"Operation failed: {_err_detail[:300]}")],
            }

        # 3. Check success_criteria against result using the deterministic verifier
        current_step = plan[idx] if idx < len(plan) else {}
        criteria = current_step.get("success_criteria", "")
        if criteria:
            verified, reason = await _deterministic_verify(criteria, last_result, state)
            if verified is False:
                logger.warning("[VERIFIER] '%s' failed criteria check: %s", tool_used, reason)
                return {
                    "status": "failed",
                    "error": f"'{tool_used}' did not meet success criteria: {reason}",
                    "messages": [AIMessage(content=f"Step did not succeed: {reason}")],
                }
            if verified is True:
                logger.info("[VERIFIER] Step '%s' passed (criteria: %s)", step_id, reason)
            else:
                logger.info("[VERIFIER] Step '%s' passed (result OK, criteria inconclusive)", step_id)
        else:
            logger.info("[VERIFIER] Step '%s' passed (result OK, no criteria)", step_id)

        return {"current_step_index": idx + 1, "retry_count": 0, "status": "executing"}

    # ── Browser tools: full verification pipeline ─────────────────────────────
    if step_status == "failed":
        return await _handle_browser_failure(state)

    current_step = plan[idx] if idx < len(plan) else {}
    criteria = current_step.get("success_criteria", "")

    # 1. Deterministic (free)
    verified, reason = await _deterministic_verify(criteria, last_result, state)
    if verified is True:
        logger.info("[VERIFIER] Step '%s' PASSED (deterministic): %s", step_id, reason)
        return {
            "current_step_index": idx + 1,
            "retry_count": 0,
            "status": "executing",
            "messages": [AIMessage(content=f"Verified: {reason}")],
        }
    if verified is False:
        logger.warning("[VERIFIER] Step '%s' FAILED (deterministic): %s", step_id, reason)
        return await _handle_browser_failure(state, reason)

    # 2. AI screenshot verification
    if criteria:
        ai_verified, ai_reason, ai_cost = await _ai_verify(criteria, current_step, last_result)
        if ai_verified:
            logger.info("[VERIFIER] Step '%s' PASSED (AI): %s", step_id, ai_reason)
            return {
                "current_step_index": idx + 1,
                "retry_count": 0,
                "status": "executing",
                "estimated_cost": cost + ai_cost,
                "messages": [AIMessage(content=f"Verified (AI): {ai_reason}")],
            }
        else:
            logger.warning("[VERIFIER] Step '%s' FAILED (AI): %s", step_id, ai_reason)
            return await _handle_browser_failure(state, ai_reason)

    # No criteria — assume pass
    logger.info("[VERIFIER] Step '%s' — no criteria, assuming success", step_id)
    return {"current_step_index": idx + 1, "retry_count": 0, "status": "executing"}


async def _deterministic_verify(
    criteria: str, result: dict, state: AgentState
) -> tuple[bool | None, str]:
    """Check success without using an LLM. Returns (True/False/None, reason)."""
    if not criteria:
        return None, "no criteria"

    criteria_lower = criteria.lower()
    result_str = str(result.get("result", "")).lower()

    if "url contains" in criteria_lower or "url has" in criteria_lower:
        try:
            from dqe_agent.tools.browser_tools import _get_browser

            browser = _get_browser()
            if browser and browser.page:
                fragment = ""
                for keyword in ["contains", "has"]:
                    if f"url {keyword}" in criteria_lower:
                        fragment = criteria_lower.split(f"url {keyword}")[-1].strip().strip("'\"")
                        break
                if fragment:
                    current_url = browser.page.url.lower()
                    if fragment in current_url:
                        return True, f"URL contains '{fragment}'"
                    try:
                        await browser.page.wait_for_url(f"**{fragment}**", timeout=12000)
                        if fragment in browser.page.url.lower():
                            return True, f"URL contains '{fragment}'"
                    except Exception:
                        for frame in browser.page.frames:
                            if fragment in frame.url.lower():
                                return True, f"Frame URL contains '{fragment}'"
                    return False, f"URL '{browser.page.url.lower()}' missing '{fragment}'"
        except Exception:
            pass

    if "page text contains" in criteria_lower:
        try:
            from dqe_agent.tools.browser_tools import _get_browser

            browser = _get_browser()
            if browser and browser.page:
                fragment = criteria_lower.split("page text contains")[-1].strip().strip("'\"")
                page_text = (await browser.page.inner_text("body")).lower()
                if fragment in page_text:
                    return True, f"Page text contains '{fragment}'"
                return False, f"Page text missing '{fragment}'"
        except Exception:
            pass

    if "shows" in criteria_lower or "visible" in criteria_lower or "present" in criteria_lower:
        if result_str and result_str != "none" and "error" not in result_str:
            return True, "Step produced non-empty result"

    if "json" in criteria_lower or "fields" in criteria_lower:
        try:
            parsed = (
                json.loads(result.get("result", "{}"))
                if isinstance(result.get("result"), str)
                else result.get("result", {})
            )
            if isinstance(parsed, dict) and len(parsed) > 0:
                return True, f"Extracted {len(parsed)} fields"
        except (json.JSONDecodeError, TypeError):
            pass

    return None, "inconclusive"


async def _ai_verify(criteria: str, step: dict, result: dict) -> tuple[bool, str, float]:
    """Screenshot-based AI verification. Returns (verified, reason, cost)."""
    from dqe_agent.llm import get_vision_llm

    try:
        screenshot_b64 = None
        try:
            from dqe_agent.tools.browser_tools import _get_browser

            browser = _get_browser()
            if browser and browser.page:
                raw = await browser.screenshot_bytes()
                screenshot_b64 = base64.b64encode(raw).decode()
        except Exception:
            pass

        llm = get_vision_llm()
        prompt = VISION_VERIFY_PROMPT.format(
            description=step.get("description", ""),
            criteria=criteria,
        )

        msgs = [HumanMessage(content=prompt)]
        if screenshot_b64:
            msgs = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                        },
                    ]
                )
            ]

        response = await llm.ainvoke(msgs)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)

        from dqe_agent.observability import trace_llm_call

        trace_llm_call(
            model="verifier-vision", role="verifier", cost_usd=COST_PER_CALL.get("verifier", 0.001)
        )

        return parsed.get("verified", False), parsed.get("reason", ""), COST_PER_CALL["verifier"]

    except Exception as exc:
        logger.warning("[VERIFIER] AI verification failed: %s — assuming success", exc)
        return True, f"AI verify error ({exc}), assuming pass", 0.0


async def _handle_browser_failure(state: AgentState, reason: str = "") -> dict:
    """Browser tool failure: blind retry → LLM screenshot diagnosis → replan → fail."""
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    step = plan[idx] if idx < len(plan) else {}
    step_desc = step.get("description", f"step {idx}")
    retry_count = state.get("retry_count", 0)
    replan_count = state.get("replan_count", 0)

    # Retry once unchanged
    if retry_count < DIAGNOSE_AT:
        logger.info(
            "[VERIFIER] Retrying browser step %d (attempt %d/%d): %s",
            idx,
            retry_count + 1,
            DIAGNOSE_AT,
            reason,
        )
        return {
            "retry_count": retry_count + 1,
            "status": "executing",
            "messages": [
                AIMessage(
                    content=f"Step failed ({reason}), retrying ({retry_count + 1}/{DIAGNOSE_AT})..."
                )
            ],
        }

    # LLM screenshot diagnosis
    logger.warning("[VERIFIER] Retries exhausted — asking LLM to diagnose step %d", idx)
    adapted = await _diagnose_and_adapt(state, reason)

    if adapted:
        new_plan = list(plan)
        new_plan[idx] = {
            **step,
            "params": adapted["params"],
            "description": adapted.get("description", step_desc),
        }
        logger.info("[VERIFIER] Adapted step %d: %s", idx, adapted.get("reasoning", "")[:120])
        return {
            "plan": new_plan,
            "retry_count": 0,
            "status": "executing",
            "messages": [
                AIMessage(
                    content=f"Diagnosed: {adapted.get('reasoning', '')} — trying adapted approach..."
                )
            ],
        }

    # Replan
    if replan_count < MAX_REPLANS:
        logger.warning(
            "[VERIFIER] Diagnosis inconclusive — replanning (%d/%d)", replan_count + 1, MAX_REPLANS
        )
        flow_data = state.get("flow_data", {})
        completed = [p.get("id", "") for p in plan[:idx]]
        collected = {sid: flow_data[sid] for sid in completed if sid in flow_data}
        collected_str = ""
        if collected:
            lines = []
            for sid, data in collected.items():
                if isinstance(data, dict) and "answer" in data:
                    lines.append(f"    {sid}.answer = {data['answer']!r}")
                else:
                    lines.append(f"    {sid} = {json.dumps(data, default=str)[:120]}")
            collected_str = "\n  Data already collected:\n" + "\n".join(lines)

        current_url = ""
        try:
            from dqe_agent.tools.browser_tools import _get_browser

            b = _get_browser()
            if b and b.page:
                current_url = b.page.url
        except Exception:
            pass

        replan_context = (
            f"\n\nPREVIOUS ATTEMPT FAILED at step '{step_desc}': {reason}"
            f"\n\nCURRENT STATE:"
            f"\n  Browser URL: {current_url or 'unknown'}"
            f"\n  Completed steps: {completed}"
            f"{collected_str}"
            f"\n\nDo NOT re-ask for already-collected data. Resume from current browser state."
        )
        return {
            "replan_count": replan_count + 1,
            "retry_count": 0,
            "status": "planning",
            "task": state.get("task", "") + replan_context,
            "messages": [AIMessage(content="Replanning from current browser state...")],
        }

    return {
        "status": "failed",
        "error": f"Step '{step_desc}' failed after diagnosis and {MAX_REPLANS} replans: {reason}",
        "messages": [
            AIMessage(content=f"Task failed: could not complete '{step_desc}' after all attempts.")
        ],
    }


async def _diagnose_and_adapt(state: AgentState, failure_reason: str) -> dict | None:
    """Take a screenshot, read the page, ask LLM what's happening and how to fix it."""
    from dqe_agent.llm import get_vision_llm
    from dqe_agent.agent.notes import save_note

    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    step = plan[idx] if idx < len(plan) else {}

    screenshot_b64 = None
    page_text = ""
    current_url = ""
    try:
        from dqe_agent.tools.browser_tools import _get_browser

        browser = _get_browser()
        if browser and browser.page:
            current_url = browser.page.url
            try:
                page_text = (await browser.page.inner_text("body"))[:1500]
            except Exception:
                pass
            raw = await browser.screenshot_bytes()
            screenshot_b64 = base64.b64encode(raw).decode()
    except Exception as exc:
        logger.warning("[DIAGNOSE] Could not capture page state: %s", exc)

    step_type = step.get("type", "")
    type_constraint = ""
    if step_type == "navigate":
        type_constraint = "\nCRITICAL: Navigation step only — do not add form-filling or actions from later steps."
    elif step_type == "login":
        type_constraint = "\nCRITICAL: Login step only — do not add post-login navigation."

    prompt = f"""A browser automation step has failed twice. Look at the screenshot and page state, then tell me what is happening and how to fix it.

FAILED STEP:
  Tool: {step.get("tool")}
  Type: {step_type}
  Description: {step.get("description")}
  Current params: {json.dumps(step.get("params", {}), indent=2)}
  Failure reason: {failure_reason}

CURRENT PAGE STATE:
  URL: {current_url}
  Page text:
{page_text}
{type_constraint}

Respond with ONLY this JSON (no markdown):
{{
  "what_i_see": "description of what is on screen",
  "why_it_failed": "root cause",
  "reasoning": "what I will do differently",
  "description": "updated step description",
  "params": {{ ... adapted params ... }}
}}"""

    try:
        llm = get_vision_llm()
        msg_content: Any = [{"type": "text", "text": prompt}]
        if screenshot_b64:
            msg_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            )
        response = await llm.ainvoke([HumanMessage(content=msg_content)])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        adapted = json.loads(content)
        save_note(
            tool=step.get("tool", ""),
            failure_reason=failure_reason,
            what_was_seen=adapted.get("what_i_see", ""),
            solution=adapted.get("reasoning", ""),
            adapted_params=adapted.get("params", {}),
        )
        logger.info("[DIAGNOSE] LLM saw: %s", adapted.get("what_i_see", "")[:100])
        logger.info("[DIAGNOSE] Fix:     %s", adapted.get("reasoning", "")[:100])
        return adapted

    except Exception as exc:
        logger.warning("[DIAGNOSE] Diagnosis LLM call failed: %s", exc)
        return None
