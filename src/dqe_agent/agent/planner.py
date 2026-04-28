"""Planner node — generates a structured step-by-step plan from a task description.

Uses a LARGE model (Opus / GPT-4o / o1) but runs only ONCE per task.
This is where the expensive thinking happens — the executor just follows the plan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from time import monotonic
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dqe_agent.guardrails import COST_PER_CALL
from dqe_agent.state import AgentState

logger = logging.getLogger(__name__)

# ── In-memory cache for common data (Jira projects, sprints, etc.) ────────────
_CACHE: dict[str, tuple[float, Any]] = {}  # key -> (fetch_time, data)
_CACHE_LOCK = asyncio.Lock()
_CACHE_TTL = 600  # seconds — refresh every 10 minutes

# ── MCP tool description block cache (frozenset of tool names → prebuilt string) ──
# Building schema for 170 tools on every planner call is expensive (~30s).
# This cache makes it instant after the first call.
_MCP_DESC_CACHE: dict[frozenset, str] = {}


def _cache_get(key: str) -> Any | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, data = entry
    if monotonic() - ts > _CACHE_TTL:
        del _CACHE[key]
        return None
    return data


def _cache_set(key: str, data: Any) -> None:
    _CACHE[key] = (monotonic(), data)


async def warm_cache() -> None:
    """Pre-fetch common Jira data at server startup so the first task is instant."""
    logger.info("[PLANNER] Warming cache: fetching Jira projects …")
    try:
        dummy: list = []
        await _prefetch_jira_projects(dummy)
        cached = _cache_get("jira_projects")
        if cached:
            logger.info("[PLANNER] Cache warm: %d Jira projects cached", len(cached))
        else:
            logger.warning("[PLANNER] Cache warm: no Jira projects found")
    except Exception as exc:
        logger.warning("[PLANNER] Cache warm-up failed: %s", exc)


async def _prewarm_mcp_tool_block() -> None:
    """Build & cache the MCP tool description block at startup.

    Iterating 170 tools and calling model_json_schema() on each takes ~30s.
    Running this once at startup means every subsequent planner call gets
    an instant cache hit instead.
    """
    from dqe_agent.tools import list_tool_names, get_tool as _get_tool

    _browser_tool_set = {
        "browser_login",
        "browser_navigate",
        "browser_act",
        "browser_extract",
        "browser_click",
        "browser_type",
        "browser_wait",
        "browser_snapshot",
        "ask_user",
        "human_review",
        "ask_user_choice",
        "direct_response",
        "request_selection",
    }
    mcp_tools = [t for t in list_tool_names() if t not in _browser_tool_set]
    if not mcp_tools:
        logger.warning("[PLANNER] MCP tool block pre-warm: no MCP tools found")
        return
    tool_key = frozenset(mcp_tools)
    if tool_key in _MCP_DESC_CACHE:
        logger.info("[PLANNER] MCP tool block already cached — skip pre-warm")
        return
    _t0 = time.monotonic()
    mcp_lines = []
    for t in mcp_tools:
        try:
            tool_obj = _get_tool(t)
            desc = tool_obj.description or ""
            param_str = ""
            try:
                schema = (
                    tool_obj.args_schema.model_json_schema()
                    if hasattr(tool_obj, "args_schema") and tool_obj.args_schema
                    else {}
                )
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                if props:
                    parts = []
                    for p, info in props.items():
                        if p in ("kwargs", "ctx"):
                            continue
                        req = "*" if p in required else ""
                        ptype = info.get("type", info.get("anyOf", [{}])[0].get("type", "any"))
                        parts.append(f"{p}{req}: {ptype}")
                    param_str = f"({', '.join(parts)})"
            except Exception:
                pass
            mcp_lines.append(f"  - {t}{param_str}: {desc[:100]}")
        except Exception:
            mcp_lines.append(f"  - {t}")
    mcp_block = (
        "AVAILABLE_MCP_TOOLS — call these directly as step tools, no browser needed.\n"
        "Parameters marked * are required. Use ONLY the listed param names — never invent extras like 'ctx', 'priority', 'auth_check'.\n"
        "{{step_id.field}} templates reference PREVIOUS STEP RESULTS only, never AVAILABLE_MCP_TOOLS itself.\n"
        + "\n".join(mcp_lines)
    )
    _MCP_DESC_CACHE[tool_key] = mcp_block
    elapsed = (time.monotonic() - _t0) * 1000
    logger.info("[PLANNER] MCP tool block pre-warmed in %.0fms (%d tools)", elapsed, len(mcp_tools))


async def _prefetch_selection_options(task: str, context_parts: list) -> None:
    """Pre-fetch Jira project options and inject them into planner context.

    Always runs — projects are cached (10 min TTL) so repeated calls are free.
    The planner receives the actual project list and can use <<JIRA_PROJECTS_PREFETCHED>>
    in request_selection steps without guessing IDs.
    """
    await _prefetch_jira_projects(context_parts)


async def _prefetch_jira_projects(context_parts: list) -> None:
    """Return Jira project options from cache (or fetch + cache on first call)."""
    # ── Fast path: serve from cache ──────────────────────────────────────────
    cached = _cache_get("jira_projects")
    if cached is not None:
        logger.info("[PLANNER] Jira projects served from cache (%d projects)", len(cached))
        _inject_jira_projects(cached, context_parts)
        return

    # ── Slow path: fetch from MCP tool (only one concurrent fetch at a time) ─
    async with _CACHE_LOCK:
        # Re-check after acquiring lock (another coroutine may have just populated it)
        cached = _cache_get("jira_projects")
        if cached is not None:
            _inject_jira_projects(cached, context_parts)
            return

        await _fetch_and_cache_jira_projects(context_parts)


def _inject_jira_projects(options: list, context_parts: list) -> None:
    """Append pre-fetched project data into the planner context."""
    options_json = json.dumps(options, indent=2)
    context_parts.append(
        "JIRA_PROJECTS_PREFETCHED — use these directly when the project is unknown.\n"
        "DO NOT call any project listing tool again.\n"
        f"{options_json}\n\n"
        "When project IS already known from the task: scan the list above to find the matching\n"
        "  'value' key (e.g. user says 'insuretech' → find label containing 'insuretech' → use its 'value').\n"
        "  Use that value directly in jira_get_assignable_users and jira_create_issue.\n"
        "When project is NOT known: use request_selection with id='sel_proj' and\n"
        "  options=<<JIRA_PROJECTS_PREFETCHED>>. Reference result as {{sel_proj.selected}}."
    )


def _unwrap_mcp_result(raw: Any) -> Any:
    """Unwrap LangChain MCP content-block lists to the actual payload.

    langchain-mcp-adapters sometimes returns tool results as:
      [{"type": "text", "text": "..actual json..", "id": "lc_<uuid>"}]
    instead of the raw JSON string / dict. This extracts the text content
    and parses it so downstream code sees real data, not content blocks.
    """
    if not isinstance(raw, list) or not raw:
        return raw
    first = raw[0]
    if not isinstance(first, dict):
        return raw
    # Detect content-block format: every item has {"type": "text", "text": ...}
    if all(
        isinstance(item, dict) and item.get("type") == "text" and "text" in item for item in raw
    ):
        combined = "\n".join(item["text"] for item in raw)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return combined  # return string — fallback parsers can handle it
    return raw


async def _fetch_and_cache_jira_projects(context_parts: list) -> None:
    """Actually call the MCP tool, cache the result, and inject into context."""
    from dqe_agent.tools import get_tool, list_tool_names

    _DISQUALIFIERS = {
        "issue",
        "sprint",
        "board",
        "comment",
        "worklog",
        "member",
        "component",
        "version",
        "field",
        "attachment",
        "link",
        "transition",
    }

    def _score(name: str) -> int:
        n = name.lower()
        if "jira" not in n:
            return -1
        if any(d in n for d in _DISQUALIFIERS):
            return -1
        if "project" not in n:
            return -1
        score = 0
        if n.endswith("projects") or "_projects" in n:
            score += 10
        if "list" in n or "get_all" in n or "search" in n:
            score += 5
        if "get" in n:
            score += 2
        return score

    candidates = sorted(
        [(name, _score(name)) for name in list_tool_names()],
        key=lambda x: x[1],
        reverse=True,
    )
    candidates = [(name, sc) for name, sc in candidates if sc > 0]

    if not candidates:
        logger.debug("[PLANNER] No Jira project listing tool found — skipping pre-fetch")
        return

    for tool_name, score in candidates:
        tool_obj = get_tool(tool_name)
        if tool_obj is None:
            continue
        logger.info("[PLANNER] Pre-fetching Jira projects via '%s' (score=%d)", tool_name, score)
        try:
            result_raw = await tool_obj.ainvoke({})
            result_str = str(result_raw) if result_raw is not None else ""
            options = _parse_jira_projects(result_raw, result_str)

            if options:
                _cache_set("jira_projects", options)
                logger.info("[PLANNER] Cached %d Jira projects", len(options))
                _inject_jira_projects(options, context_parts)
                return
            else:
                logger.warning(
                    "[PLANNER] '%s' returned no parseable projects — trying next", tool_name
                )

        except Exception as exc:
            logger.warning("[PLANNER] '%s' failed (%s) — trying next candidate", tool_name, exc)

    logger.warning("[PLANNER] All Jira project listing candidates failed — planner will ask_user")


import re as _re

_VALID_JIRA_KEY = _re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def _is_valid_jira_key(raw: str) -> bool:
    """True if the string looks like a real Jira project key (e.g. FLAG, DEV, PROJ)."""
    return bool(_VALID_JIRA_KEY.match(raw.strip().upper()))


def _parse_jira_projects(result_raw: Any, result_str: str) -> list[dict]:
    """Extract [{value: key, label: name}] from whatever the MCP tool returned.

    Rejects UUID/long-hex values that mcp-atlassian sometimes returns as 'key'.
    Only accepts proper Jira project keys: 2-10 uppercase letters/digits, no hyphens.
    """
    options: list[dict] = []

    # Unwrap LangChain MCP content blocks before any other processing
    result_raw = _unwrap_mcp_result(result_raw)

    # ── Parse structured data ────────────────────────────────────────────────
    data = None
    if isinstance(result_raw, (list, dict)):
        data = result_raw
    elif isinstance(result_raw, str):
        try:
            data = json.loads(result_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                logger.debug("[PLANNER] Project item %d is not a dict: %s", idx, item)
                continue
            # Try short fields first (most likely to be the real project key)
            # mcp-atlassian may return {key, name, id} or {projectKey, projectName, id}
            candidate_keys = [
                item.get("key", ""),
                item.get("projectKey", ""),
                item.get("project_key", ""),
            ]
            key = ""
            for ck in candidate_keys:
                ck_str = str(ck).strip().upper()
                if _is_valid_jira_key(ck_str):
                    key = ck_str
                    break

            if not key:
                # Skip — all candidate keys are UUIDs, numeric IDs, or empty
                logger.debug(
                    "[PLANNER] Skipping project item %d with no valid key: keys=%s",
                    idx,
                    candidate_keys,
                )
                continue

            name = item.get("name") or item.get("projectName") or item.get("displayName") or key
            # Heuristic: if the key is just the uppercased name (e.g. key="INSURETECH", name="InsureTech"),
            # it's likely the server mistakenly used the name as the key. Skip this item.
            if key.upper() == name.strip().upper():
                logger.debug(
                    "[PLANNER] Skipping project item %d where key '%s' equals name '%s'",
                    idx,
                    key,
                    name,
                )
                continue

            # Ensure value is JUST the key, label is for display
            options.append({"value": key, "label": f"{key} — {name}"})

    elif isinstance(data, dict):
        for container_key in ("projects", "values", "items", "data", "results"):
            if container_key in data and isinstance(data[container_key], list):
                nested = data[container_key]
                logger.debug(
                    "[PLANNER] Found %d projects in container '%s'", len(nested), container_key
                )
                # Recurse but avoid infinite loops by passing empty string for result_str
                return _parse_jira_projects(nested, "")
        logger.debug(
            "[PLANNER] No project list container found in dict keys: %s", list(data.keys())
        )

    # ── String fallback: scan for KEY — Name patterns ────────────────────────
    if not options and result_str:
        for m in _re.finditer(r"\b([A-Z][A-Z0-9]{1,9})\b", result_str):
            key = m.group(1)
            if key not in {o["value"] for o in options}:
                options.append({"value": key, "label": key})

    logger.debug("[PLANNER] Parsed %d project options total", len(options))
    # Do NOT arbitrarily truncate — return all discovered projects (up to a safe max).
    # The UI can scroll; the user needs to find their project. 100 is a reasonable cap.
    return options[:100]


# ── Master system prompt — all tasks ─────────────────────────────────────────
MASTER_SYSTEM_PROMPT = """You are a task-execution agent. Given a task you output a COMPLETE plan as a JSON array of steps, then an executor runs each step in order.

OUTPUT RULES:
1. Output ONLY a JSON array. No markdown, no explanation, no prose.
2. Plan ALL steps upfront in one array. If an early step pauses for input, still include every subsequent step after it.
3. Every step must have: id, type, description, tool, params, success_criteria (plain quoted string, never nested JSON).
4. Reference prior step results as {{step_id.field}} or {{step_id.answer}}.
5. Max 25 steps. Merge trivial adjacent actions where safe.
6. Do not invent IDs, values, names, or keys the user did not explicitly provide.
7. Greetings / status queries → one direct_response step only.
8. ⚑ DO NOT ASK FOR INFORMATION ALREADY IN THE MESSAGE.
   If the task contains a Jira project key (e.g. "in FLAG", "for FLAG", "FLAG project"),
   use it directly — skip any project selection or ask_user step.
   Same rule applies to: issue key, sprint name, assignee, priority, status, comment text,
   time duration, issue type, board name. Extract from the message first.
   Only ask / show selection when the value is genuinely absent.
9. ⚑ DATE HANDLING RULES:
   • When the user gives ANY date reference ("today", "this week", "last month", "April 20",
     "all time", "custom", specific dates) → YOU convert it directly to valid JQL in the plan.
     TODAY'S DATE is injected in context — use it to resolve relative references.
   • "all time" / "ever" / "from the beginning" / no date mentioned → omit the date filter entirely.
   • "custom" / two specific dates given → put the actual date range in the JQL.
   [Rule] If NO date reference is given → DO NOT add any date/updated/created filter to JQL.
      "what work did X do?" with no date → JQL has NO date clause. Return all results.
      NEVER default to startOfDay(), -1d, -7d, or any date when the user did not ask for it.

10. ⚑ PAGINATION RULES (CRITICAL — READ CAREFULLY):
   • jira_search and jira_get_sprint_issues return at most 50 results at a time.
   • When PAGINATION CONTEXT is present in AVAILABLE DATA (look for "PAGINATION CONTEXT" section):
     - If user says "next" / "more" / "show more" / "continue":
       COPY the JQL string from "Query/sprint:" in PAGINATION CONTEXT VERBATIM — character for character.
       DO NOT shorten it, DO NOT remove any AND clauses, DO NOT reconstruct it from memory.
       Use the start_at or page_token value from PAGINATION CONTEXT.
     - DO NOT re-ask for project, assignee, or any other parameters — ALL values are in PAGINATION CONTEXT
     - DO NOT create multi-step plans — pagination is a SINGLE tool call
     - Example: if PAGINATION CONTEXT shows Query/sprint: assignee = "X" AND project = AMA ORDER BY updated DESC
       → plan MUST be: [{"id":"fetch","tool":"jira_search","params":{"jql":"assignee = \"X\" AND project = AMA ORDER BY updated DESC","start_at":50,"limit":50}}]
       NEVER drop the AND project = AMA part. NEVER simplify.
   • For sprint issues pagination: jira_get_sprint_issues has no start_at — use jira_search with
     jql="sprint = <sprint_id>" + start_at for any page after the first.
   • If no PAGINATION CONTEXT exists, treat as new query (full plan needed).

═══════════════════════════════════════════════════════════════════
TOOLS
═══════════════════════════════════════════════════════════════════

── USER INTERACTION (no browser, no API call) ──
  direct_response(message)
      Reply to the user. No side effects. Use for greetings, answers, status messages.

  ask_user(question)
      Pause for a free-text reply. Use ONLY when no finite option list exists in the system.
      IMPORTANT: If the question presents a list of options (e.g. IDs), wrap each option/ID in <<< and >>> markers to make them clickable.

  request_selection(question, options, multi_select)
      Show the user clickable buttons to pick from a list. ALWAYS prefer this over ask_user
      when the system can return a finite set of valid choices.
      options format: [{"value": "KEY", "label": "KEY — Human readable name"}, ...]
      The selected value is returned as {{step_id.selected}}.

  human_review(review_type, summary)
      Pause for explicit human approval before a consequential action.

── MCP / API TOOLS (no browser needed) ──
  Listed in AVAILABLE_MCP_TOOLS injected into the context below.
  Call them directly as a step tool. Param names must exactly match the schema.
  Never add extra params like "ctx", "priority", "auth_check" — only schema params.
  [Rule] Do not invent tool names. If a capability is not in AVAILABLE_MCP_TOOLS, use direct_response to explain the limitation instead of fabricating a tool name.

── BROWSER TOOLS (web automation only — see BROWSER INSTRUCTIONS section if present) ──
  browser_login(site)                   log in to a configured site
  browser_navigate(url)                 go to a URL
  browser_act(instruction)              click, type, fill, scroll, interact
  browser_extract(instruction, schema)  read data from the current page
  browser_click(target)                 click a specific element
  browser_type(text, target)            type text into a field
  browser_wait(condition, timeout_ms)   wait for a page condition
  browser_snapshot()                    capture current page state

═══════════════════════════════════════════════════════════════════
SELECTION RULE — applies to every tool, every domain
═══════════════════════════════════════════════════════════════════

SHORTCUT RULE — skip the selection flow entirely when the value is already in the task:
  • "Create a sprint in FLAG"         → project_key = "FLAG", skip project selection
  • "Create a bug in MA"              → project_key = "MA", skip project selection
  • "Move FLAG-42 to done"            → issue_key = "FLAG-42", skip ask_user
  • "Assign FLAG-42 to Hrithik"       → issue_key and assignee both known, skip both asks
  • "Log 3 hours on FLAG-42"          → issue_key and duration known, skip both asks
  • "Add sprint named Q3 Sprint 1"    → sprint_name known, skip ask_user for name
  • "Set FLAG-42 priority to high"    → key and priority known, skip both asks
  Extract the value from the message and use it directly in the step params.

Only when the value is genuinely absent:
  STEP A  Call the appropriate MCP tool to fetch the valid options list.
  STEP B  Present with request_selection — user clicks a button, no typing needed.
  STEP C  Use {{stepB_id.selected}} in the next step's params.

NEVER use ask_user for system-defined values. NEVER hard-code options you haven't fetched.
NEVER ask the user to type a key, ID, or code that the system can provide as a list.

[Rule] Do not ASSUME any Jira field. This means:
  • Project: if stated in task → look up the real key from JIRA_PROJECTS_PREFETCHED and use it directly.
             if NOT stated → use request_selection with JIRA_PROJECTS_PREFETCHED options.
             NEVER invent a project key from the user's words without matching it to the cache.
  • NEVER assume the sprint — ALWAYS fetch boards then sprints, present with request_selection.
  • NEVER assume the assignee — ALWAYS fetch project members (for the confirmed project) and
    present with request_selection via the form.
  • NEVER assume the issue type — if not 100% explicit in user message, fetch and ask.
  • NEVER assume the board — ALWAYS fetch boards for the selected/known project.

═══════════════════════════════════════════════════════════════════
JIRA / ATLASSIAN
═══════════════════════════════════════════════════════════════════

Use MCP tools from AVAILABLE_MCP_TOOLS. Do NOT use browser_* tools for Jira.

Verifier note: Jira steps are verified by response content only — no screenshot.
Use plain success_criteria like "Issue created successfully" or "Projects listed".

── CREATE ISSUE — MANDATORY STEP ORDER (NEVER skip or reorder) ──

  MANDATORY ORDER for every create-issue plan:
  1. sel_proj (request_selection) — ONLY if project not stated; skip if project is known
  2. get_fields (jira_get_project_fields) — ALWAYS, no exceptions, BEFORE get_pri/fetch_users/form
  3. get_pri (jira_get_priorities) — AFTER get_fields
  4. fetch_users (jira_get_assignable_users) — AFTER get_fields
  5. collect_info (request_form) — AFTER get_fields, using supports.* to decide which fields
  6. gen_desc → edit_desc → create_issue → upd_issue → attach (if file) → get_created → done

  NEVER output a create-issue plan without get_fields as step 1 (or step 2 if sel_proj needed).
  NEVER call jira_get_priorities or request_form before jira_get_project_fields completes.

⚑ KEY RULE — resolve the REAL project key from JIRA_PROJECTS_PREFETCHED BEFORE fetching
  assignees or creating anything. NEVER use <<FIRST_PROJECT_KEY>>. NEVER guess a key from
  the user's words alone.
  When user says "insuretech" → scan JIRA_PROJECTS_PREFETCHED labels → use the matching "value"
  (e.g. look for "insuretech" in label "NSRTCH — InsureTech" → real key is "NSRTCH").

⚑ Assignees MUST be fetched AFTER the project key is known — they are project-specific.

⚑ FIELD DISCOVERY RULE — ALWAYS call jira_get_project_fields first to discover which fields
  the project actually supports. Then show ONLY fields the project supports:
  - If supports.priority=false → OMIT priority from form AND from jira_update_issue
  - If supports.story_points=true → ADD story_points number field to form
  - If supports.sprint=true → ADD sprint selection to form
  - If supports.components=true AND components exist → ADD components multi-select
  - If supports.labels=true → ADD labels text field
  Never show a field the project does not support.

CASE A — project IS stated in the task (e.g. "create issue in FLAG", "create bug in insuretech"):
  • Scan JIRA_PROJECTS_PREFETCHED to find the real key ("value") for the project the user named.
  • Use that real key throughout — DO NOT put a project selection field in the form.
  • Replace <REAL_KEY> below with the actual "value" from JIRA_PROJECTS_PREFETCHED.

⚑ SMART FORM RULE — only ask for what is NOT already known from the user's message:
  • summary:    user gave a clear title/topic → pre-fill as default, required:false (confirm only)
                user gave NO title at all → required:true, no default
  • issue_type: user said "bug"/"task"/"story"/"epic" → OMIT this field entirely, hardcode in create_issue
                user gave no type → include as select, required:true
  • assignee:   user named a person → OMIT this field, resolve the name from fetch_users and hardcode
                user gave no assignee → include as select, required:false
  • priority:   user gave an exact priority → OMIT this field, hardcode in upd_issue
                user gave no priority AND supports.priority=true → include as select, required:true
  • desc_topic: always optional — pre-fill with user's description if they gave one, otherwise leave blank

  EXAMPLE — user says "create a task in FLAG for making UI with hero UI, assign to hrithik":
    Known: project=FLAG, summary≈"Making UI with Hero UI", type=task (implied), assignee≈"hrithik"
    Form should only show: priority (if project supports it), desc_topic (optional)
    Assignee is resolved by matching "hrithik" against fetch_users options (case-insensitive first-name match)
    Hardcode: issue_type="Task", assignee="Hrithik <FullName>" from fetch_users

  EXACT JSON for CASE A (replace every <REAL_KEY> with the matched project value):
  {"id":"get_fields","tool":"jira_get_project_fields","params":{"project_key":"<REAL_KEY>","issue_type":"<ISSUE_TYPE_OR_Task>"},"success_criteria":"Project fields discovered"},
  {"id":"get_pri","tool":"jira_get_priorities","params":{"project_key":"<REAL_KEY>"},"success_criteria":"Priorities fetched"},
  {"id":"fetch_users","tool":"jira_get_assignable_users","params":{"project_key":"<REAL_KEY>"},"success_criteria":"Users fetched"},
  {"id":"collect_info","tool":"request_form","params":{"title":"Create Jira Issue","fields":[
    {"id":"summary","label":"Summary","type":"text","required":<true_if_unknown|false_if_known>,"placeholder":"Short one-line title","default":"<known_summary_or_omit>"},
    <OMIT issue_type field if type is known — use hardcoded value in create_issue>,
    {"id":"issue_type","label":"Issue Type","type":"select","required":true,"options":[{"value":"Task","label":"Task"},{"value":"Bug","label":"Bug"},{"value":"Story","label":"Story"},{"value":"Epic","label":"Epic"}]},
    <INCLUDE priority ONLY if get_fields result shows supports.priority=true AND user did not state priority>,
    {"id":"priority","label":"Priority","type":"select","required":true,"options":"{{get_pri}}"},
    <OMIT assignee field if assignee name was stated — resolve from fetch_users and hardcode>,
    {"id":"assignee","label":"Assignee","type":"select","required":false,"options":"{{fetch_users}}"},
    <INCLUDE story_points ONLY if get_fields result shows supports.story_points=true>,
    {"id":"story_points","label":"Story Points","type":"number","required":false,"placeholder":"e.g. 3"},
    <INCLUDE sprint ONLY if get_fields result shows supports.sprint=true>,
    {"id":"sprint","label":"Sprint","type":"text","required":false,"placeholder":"Sprint name or leave blank"},
    <INCLUDE labels ONLY if get_fields result shows supports.labels=true>,
    {"id":"labels","label":"Labels","type":"text","required":false,"placeholder":"comma-separated labels"},
    {"id":"desc_topic","label":"Description Topic","type":"textarea","required":false,"placeholder":"What should the description cover?","default":"<known_description_if_any>"},
    {"id":"attachment","label":"Attach file? (enter local file path or leave blank)","type":"text","required":false,"placeholder":"/path/to/file.png"}
  ]},"success_criteria":"All task details collected"},
  {"id":"gen_desc","tool":"llm_draft_content","params":{"content_type":"issue_description","topic":"{{collect_info.desc_topic}}","context":"{{collect_info.issue_type}} in <REAL_KEY>: {{collect_info.summary}}"},"success_criteria":"Description drafted"},
  {"id":"edit_desc","tool":"request_edit","params":{"label":"Issue Description","content":"{{gen_desc.content}}"},"success_criteria":"Description reviewed"},
  {"id":"create_issue","tool":"jira_create_issue","params":{"project_key":"<REAL_KEY>","issue_type":"{{collect_info.issue_type}}","summary":"{{collect_info.summary}}","description":"{{edit_desc.content}}"},"success_criteria":"Issue created"},
  {"id":"upd_issue","tool":"jira_update_issue","params":{"issue_key":"{{create_issue.key}}","fields":{"priority":"{{collect_info.priority}}","assignee":"{{collect_info.assignee}}"}},"success_criteria":"Priority and assignee set"},
  <INCLUDE attachment step ONLY if user provided a file path in collect_info.attachment>,
  {"id":"attach","tool":"jira_add_attachment","params":{"issue_key":"{{create_issue.key}}","file_path":"{{collect_info.attachment}}"},"success_criteria":"Attachment uploaded"},
  {"id":"get_created","tool":"jira_get_issue","params":{"issue_key":"{{create_issue.key}}"},"success_criteria":"Issue details fetched"},
  {"id":"done","tool":"direct_response","params":{"message":"Issue created successfully!\n\nKey: {{create_issue.key}}\nSummary: {{collect_info.summary}}\nType: {{collect_info.issue_type}}\nPriority: {{collect_info.priority}}\nAssignee: {{collect_info.assignee}}\n\nFull details:\n{{get_created}}"}}

CASE B — project NOT stated in the task:
  First show project selection, then fetch fields+assignees for the CHOSEN project.
  Apply the same SMART FORM RULE — only include fields not already known.

  EXACT JSON for CASE B:
  {"id":"sel_proj","tool":"request_selection","params":{"question":"Which Jira project should this be created in?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
  {"id":"get_fields","tool":"jira_get_project_fields","params":{"project_key":"{{sel_proj.selected}}","issue_type":"Task"},"success_criteria":"Project fields discovered"},
  {"id":"get_pri","tool":"jira_get_priorities","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Priorities fetched"},
  {"id":"fetch_users","tool":"jira_get_assignable_users","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Users fetched"},
  {"id":"collect_info","tool":"request_form","params":{"title":"Create Jira Issue","fields":[
    {"id":"summary","label":"Summary","type":"text","required":<true_if_unknown|false_if_known>,"placeholder":"Short one-line title","default":"<known_summary_or_omit>"},
    <OMIT issue_type field if type is known>,
    {"id":"issue_type","label":"Issue Type","type":"select","required":true,"options":[{"value":"Task","label":"Task"},{"value":"Bug","label":"Bug"},{"value":"Story","label":"Story"},{"value":"Epic","label":"Epic"}]},
    <INCLUDE priority ONLY if get_fields shows supports.priority=true>,
    {"id":"priority","label":"Priority","type":"select","required":true,"options":"{{get_pri}}"},
    <OMIT assignee field if assignee name was stated — resolve from fetch_users>,
    {"id":"assignee","label":"Assignee","type":"select","required":false,"options":"{{fetch_users}}"},
    <INCLUDE story_points ONLY if get_fields shows supports.story_points=true>,
    {"id":"story_points","label":"Story Points","type":"number","required":false,"placeholder":"e.g. 3"},
    <INCLUDE labels ONLY if get_fields shows supports.labels=true>,
    {"id":"labels","label":"Labels","type":"text","required":false,"placeholder":"comma-separated labels"},
    {"id":"desc_topic","label":"Description Topic","type":"textarea","required":false,"placeholder":"What should the description cover?"},
    {"id":"attachment","label":"Attach file? (enter local file path or leave blank)","type":"text","required":false,"placeholder":"/path/to/file.png"}
  ]},"success_criteria":"All task details collected"},
  {"id":"gen_desc","tool":"llm_draft_content","params":{"content_type":"issue_description","topic":"{{collect_info.desc_topic}}","context":"{{collect_info.issue_type}} in {{sel_proj.selected}}: {{collect_info.summary}}"},"success_criteria":"Description drafted"},
  {"id":"edit_desc","tool":"request_edit","params":{"label":"Issue Description","content":"{{gen_desc.content}}"},"success_criteria":"Description reviewed"},
  {"id":"create_issue","tool":"jira_create_issue","params":{"project_key":"{{sel_proj.selected}}","issue_type":"{{collect_info.issue_type}}","summary":"{{collect_info.summary}}","description":"{{edit_desc.content}}"},"success_criteria":"Issue created"},
  {"id":"upd_issue","tool":"jira_update_issue","params":{"issue_key":"{{create_issue.key}}","fields":{"priority":"{{collect_info.priority}}","assignee":"{{collect_info.assignee}}"}},"success_criteria":"Priority and assignee set"},
  <INCLUDE attachment step ONLY if user provided a file path in collect_info.attachment>,
  {"id":"attach","tool":"jira_add_attachment","params":{"issue_key":"{{create_issue.key}}","file_path":"{{collect_info.attachment}}"},"success_criteria":"Attachment uploaded"},
  {"id":"get_created","tool":"jira_get_issue","params":{"issue_key":"{{create_issue.key}}"},"success_criteria":"Issue details fetched"},
  {"id":"done","tool":"direct_response","params":{"message":"Issue created successfully!\n\nKey: {{create_issue.key}}\nSummary: {{collect_info.summary}}\nType: {{collect_info.issue_type}}\nPriority: {{collect_info.priority}}\nAssignee: {{collect_info.assignee}}\n\nFull details:\n{{get_created}}"}}

⚑ EVERY field dict MUST include an "id" key.
⚑ summary: pre-fill as default when user gave a clear title; only require when truly absent.
⚑ priority: include ONLY if get_fields shows supports.priority=true AND user did not state priority.
⚑ story_points: include ONLY if get_fields shows supports.story_points=true.
⚑ labels: include ONLY if get_fields shows supports.labels=true.
⚑ attachment: ALWAYS include as optional text field — user can leave blank if no file.
⚑ attachment step (jira_add_attachment): ONLY add to plan if collect_info.attachment is non-empty.
⚑ assignee: when user names a person, match against fetch_users and hardcode — do NOT show assignee field.
⚑ issue_type: when user states type ("bug", "task", "story", "epic") → omit field, hardcode in create_issue.
⚑ NEVER show a field for something the user already told you.

project_key rules:
  • MUST be a short ALL-UPPERCASE key like FLAG, DEV, PROJ.
  • NEVER pass a UUID, numeric ID, lowercase string, or sentence as project_key.
  • ALWAYS use the exact "value" from JIRA_PROJECTS_PREFETCHED — NEVER the "label" or the user's raw words.

issue_type rules:
  • "task"/"bug"/"story"/"epic"/"subtask" stated by user → hardcode it.
  • Default: "Task" when ambiguous.

── CREATE SPRINT ──

  Required: board_id (from board selection), sprint_name (from ask_user or task).
  Boards MUST be fetched per-project using jira_get_agile_boards(project_key=...).
  The executor pre-resolves the board list into request_selection options automatically.
  start_date / end_date are auto-filled by executor — NEVER ask the user for them.

  [Rule] STEP ORDER IS STRICT — a step that references {{X.selected}} MUST come AFTER the step with id "X".
     NEVER place jira_get_agile_boards before the project selection step it depends on.

  CASE A — project key IS in the task (e.g. "Create a sprint in FLAG"):
    EXACT JSON (copy verbatim — step order must not change):
    [{"id":"get_boards","tool":"jira_get_agile_boards","params":{"project_key":"FLAG"},"success_criteria":"Boards listed"},
     {"id":"sel_board","tool":"request_selection","params":{"question":"Which board?","options":"{{get_boards}}","multi_select":false},"success_criteria":"Board selected"},
     {"id":"sprint_info","tool":"request_form","params":{"title":"New Sprint Details","fields":[
       {"id":"name","label":"Sprint Name","type":"text","required":true,"placeholder":"e.g. Sprint 5"},
       {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
       {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
       {"id":"goal","label":"Sprint Goal","type":"textarea","required":false,"placeholder":"What is the goal of this sprint? (optional)"}
     ]},"success_criteria":"Sprint details collected"},
     {"id":"cr_sprint","tool":"jira_create_sprint","params":{"board_id":"{{sel_board.selected}}","name":"{{sprint_info.name}}","start_date":"{{sprint_info.start_date}}","end_date":"{{sprint_info.end_date}}","goal":"{{sprint_info.goal}}"},"success_criteria":"Sprint created"},
     {"id":"done","tool":"direct_response","params":{"message":"Sprint '{{sprint_info.name}}' created successfully."}}]

  CASE B — project key NOT in task (e.g. "Create a sprint"):
    EXACT JSON (sel_proj MUST be first — jira_get_agile_boards MUST be after it):
    [{"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
     {"id":"get_boards","tool":"jira_get_agile_boards","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Boards listed"},
     {"id":"sel_board","tool":"request_selection","params":{"question":"Which board?","options":"{{get_boards}}","multi_select":false},"success_criteria":"Board selected"},
     {"id":"sprint_info","tool":"request_form","params":{"title":"New Sprint Details","fields":[
       {"id":"name","label":"Sprint Name","type":"text","required":true,"placeholder":"e.g. Sprint 5"},
       {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
       {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
       {"id":"goal","label":"Sprint Goal","type":"textarea","required":false,"placeholder":"What is the goal of this sprint? (optional)"}
     ]},"success_criteria":"Sprint details collected"},
     {"id":"cr_sprint","tool":"jira_create_sprint","params":{"board_id":"{{sel_board.selected}}","name":"{{sprint_info.name}}","start_date":"{{sprint_info.start_date}}","end_date":"{{sprint_info.end_date}}","goal":"{{sprint_info.goal}}"},"success_criteria":"Sprint created"},
     {"id":"done","tool":"direct_response","params":{"message":"Sprint '{{sprint_info.name}}' created successfully."}}]

  ⚑ ALWAYS collect start_date, end_date, and goal from the user via the form above.
     NEVER skip the form or leave date fields empty — the executor no longer auto-fills them.

── GET ACTIVE SPRINTS ──
To retrieve active sprints (e.g. "what is the active sprint for me?", "status of current sprint"):
  ⚑ ALWAYS select project first — boards are project-specific. NEVER call jira_get_agile_boards with no project_key.
  1. If project is NOT stated → request_selection from <<JIRA_PROJECTS_PREFETCHED>> first.
  2. Fetch boards for that project → jira_get_agile_boards(project_key=<key>).
  3. request_selection so user picks the board (executor auto-skips if only one board).
  4. Call jira_get_sprints_from_board(board_id=..., state="active") → returns sprint list.
  5. request_selection so user picks which sprint (executor auto-skips if only one active sprint).
  6. direct_response with the sprint result OR call jira_get_sprint_issues(sprint_id="{{sel_sprint.selected}}").

  ⚑ sprint_id must ALWAYS be a plain string ID like "1454" — NEVER pass the full sprint object or list.
     Use request_selection after jira_get_sprints_from_board so the user picks one sprint,
     then reference {{sel_sprint.selected}} as the sprint_id string.

  Example — show sprint issues, project NOT stated ("show current sprint board", "what tasks are in current sprint"):
  [
    {"id":"sel_proj","tool":"request_selection","params":{"question":"Which project's sprint do you want to check?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
    {"id":"boards","tool":"jira_get_agile_boards","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Boards retrieved"},
    {"id":"sel_board","tool":"request_selection","params":{"question":"Which board?","options":"{{boards}}","multi_select":false},"success_criteria":"Board selected"},
    {"id":"active","tool":"jira_get_sprints_from_board","params":{"board_id":"{{sel_board.selected}}","state":"active"},"success_criteria":"Active sprints loaded"},
    {"id":"sel_sprint","tool":"request_selection","params":{"question":"Which active sprint?","options":"{{active}}","multi_select":false},"success_criteria":"Sprint selected"},
    {"id":"issues","tool":"jira_get_sprint_issues","params":{"sprint_id":"{{sel_sprint.selected}}","limit":50},"success_criteria":"Sprint issues fetched"},
    {"id":"show","tool":"direct_response","params":{"message":"**Sprint:** {{active}}\n\n{{issues}}"}}
  ]

  ⚑ Sprint board message MUST include sprint name/dates from {{active}} AND the issue list from {{issues}}.
  The {{active}} resolves to the sprint object — it contains name, startDate, endDate, state.
  The executor formats the issue list with status breakdown automatically via _format_result_for_display.

  Example — just show sprint info (no issues needed), project IS stated ("current sprint in FLAG"):
  [
    {"id":"boards","tool":"jira_get_agile_boards","params":{"project_key":"FLAG"},"success_criteria":"Boards retrieved"},
    {"id":"sel_board","tool":"request_selection","params":{"question":"Which board?","options":"{{boards}}","multi_select":false},"success_criteria":"Board selected"},
    {"id":"active","tool":"jira_get_sprints_from_board","params":{"board_id":"{{sel_board.selected}}","state":"active"},"success_criteria":"Active sprints loaded"},
    {"id":"show","tool":"direct_response","params":{"message":"Active sprints:\n{{active}}"}}
  ]

  [Rule] Do not call jira_get_agile_boards with empty params {} — always pass project_key.
  [Rule] jira_get_sprint_issues limit MUST be ≤ 50 — never pass limit > 50.
  [Rule] Do not pass {{active}} directly as sprint_id — {{active}} is a list of sprint objects, NOT a string ID.
     Always add a request_selection step after jira_get_sprints_from_board to get a single sprint_id string.
  ⛠ There is NO tool named 'jira_get_active_sprints' — use jira_get_sprints_from_board with state='active'.



[Rule] Do not pass jira_get_transitions result directly as transition_id (it's a list, not a string).

  Required params for jira_transition_issue:
    issue_key      (string, e.g. "FLAG-42")
    transition_id  (string — a SINGLE numeric ID like "31", NOT a list)
    _target_status (string — ALWAYS include the target status name from the user's request,
                   e.g. "in progress", "done", "to do", "testing". The executor uses this
                   to pick the correct transition from the fetched list.)

  ⚑ ALWAYS include _target_status in jira_transition_issue params when the target is stated by the user.
  ⚑ _target_status must be a plain lowercase string: "in progress", "done", "to do", "testing", "on hold".

  DECISION TREE — follow this exactly:

  A) MULTIPLE issues in context AND user didn't specify which ones
     → request_selection(multi_select=true) to let user pick which issues to act on FIRST.
     Then loop the transition steps for each selected issue.

  B) TARGET STATUS already known from user message ("move to In Progress", "mark as Done", "close")
     → jira_get_transitions(issue_key=...) → jira_transition_issue(transition_id="{{get_tr}}", _target_status="<status>")
     The executor uses _target_status to match the correct ID. SKIP request_selection for status.

  C) TARGET STATUS is ambiguous (user just said "update status" / "change state")
     → jira_get_transitions → request_selection for status → jira_transition_issue(transition_id="{{sel_tr.selected}}")

  EXAMPLE A — "mark them as done" (multiple open tasks returned from previous step):
  [
    {"id":"sel_issues","tool":"request_selection",
     "params":{"question":"Which tasks should I mark as Done?",
               "options":"{{search_tasks}}","multi_select":true},
     "success_criteria":"Issues selected"},
    {"id":"get_tr_1","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions fetched"},
    {"id":"tr_1","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{get_tr_1}}","_target_status":"done"},"success_criteria":"FLAG-33 marked Done"},
    {"id":"get_tr_2","tool":"jira_get_transitions","params":{"issue_key":"FLAG-32"},"success_criteria":"Transitions fetched"},
    {"id":"tr_2","tool":"jira_transition_issue","params":{"issue_key":"FLAG-32","transition_id":"{{get_tr_2}}","_target_status":"done"},"success_criteria":"FLAG-32 marked Done"}
  ]

  ⚑ ALWAYS add jira_get_issue AFTER jira_transition_issue + direct_response showing the actual current status. This confirms the transition actually happened.

  EXAMPLE B — "mark FLAG-33 as Done" (single issue, status known):
  [
    {"id":"get_tr","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions listed"},
    {"id":"do_tr","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{get_tr}}","_target_status":"done"},"success_criteria":"Transitioned to Done"},
    {"id":"verify","tool":"jira_get_issue","params":{"issue_key":"FLAG-33"},"success_criteria":"Issue fetched"},
    {"id":"done","tool":"direct_response","params":{"message":"FLAG-33 has been moved to Done.\n\nCurrent issue details:\n{{verify}}"}}
  ]

  EXAMPLE B2 — "move FLAG-145 to In Progress" (single issue, status known):
  [
    {"id":"get_tr","tool":"jira_get_transitions","params":{"issue_key":"FLAG-145"},"success_criteria":"Transitions listed"},
    {"id":"do_tr","tool":"jira_transition_issue","params":{"issue_key":"FLAG-145","transition_id":"{{get_tr}}","_target_status":"in progress"},"success_criteria":"Transitioned to In Progress"},
    {"id":"verify","tool":"jira_get_issue","params":{"issue_key":"FLAG-145"},"success_criteria":"Issue fetched"},
    {"id":"done","tool":"direct_response","params":{"message":"FLAG-145 has been moved to In Progress.\n\nCurrent issue details:\n{{verify}}"}}
  ]

  EXAMPLE C — "change the status of FLAG-33" (ambiguous — ask):
  [
    {"id":"get_tr","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions listed"},
    {"id":"sel_tr","tool":"request_selection","params":{"question":"Which status would you like to move FLAG-33 to?","options":"{{get_tr}}","multi_select":false},"success_criteria":"Status selected"},
    {"id":"do_tr","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{sel_tr.selected}}"},"success_criteria":"Transitioned"},
    {"id":"verify","tool":"jira_get_issue","params":{"issue_key":"FLAG-33"},"success_criteria":"Issue fetched"},
    {"id":"done","tool":"direct_response","params":{"message":"FLAG-33 status updated.\n\nCurrent issue details:\n{{verify}}"}}
  ]

── DIRECT-ANSWER QUERIES (count / filter / list) ──
  Use jira_search(jql=..., limit=N) then direct_response. Always exactly 2 steps.
  NEVER add project-selection steps for read-only queries — JQL is sufficient.

  JQL reference (use these exactly):
    My open issues:         assignee = currentUser() AND status != Done
    By priority:            priority = Critical  (or Blocker, High, Medium, Low)
    By status:              status = "In Progress"  (or "In Review", "To Do", "Done", "Testing")
    Unassigned:             assignee is EMPTY
    In active sprint:       sprint in openSprints()
    Blocked:                priority = Blocker AND status != Done
    Done recently:          status = Done AND updated >= -1d
    Combine freely:         assignee = currentUser() AND status = "In Progress"
    Worklog on a date:      worklogDate = "YYYY-MM-DD" AND worklogAuthor = currentUser()

  ⚑ DATE RULE FOR JQL: Convert any date reference the user gives you directly to JQL in the plan.
    "all time" → omit date clause. "custom" → collect via form. "today"/"this week"/etc. → JQL function.
    If NO date was mentioned at all → ask_user or show a period-select form before querying.

  Count query ("how many X do I have?"):
  [{"id":"s","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status != Done","limit":1},"success_criteria":"Count returned"},
   {"id":"r","tool":"direct_response","params":{"message":"You have {{s.total}} open issues."}}]

  List query ("show all critical issues"):
  [{"id":"s","tool":"jira_search","params":{"jql":"priority = Critical AND status != Done","limit":50},"success_criteria":"Issues listed"},
   {"id":"r","tool":"direct_response","params":{"message":"Critical open issues:\n{{s}}"}}]

  ⚑ FOLLOW-UP "list them" / "show them" / "list it" rule:
    When the user says a short follow-up like "list them", "show them", "show me", "list it",
    "display them" after a prior search was already run in this session:
    → Re-run the SAME jira_search with the SAME JQL from PAGINATION CONTEXT (or infer from history).
    → Pass the result directly to direct_response with the full list: "{{s}}"
    → Do NOT show just a count. Do NOT use {{s.total}}. Show the full issue list.
    Example: user asked "give me all tasks assigned" → saw count → says "list them":
    [{"id":"s","tool":"jira_search","params":{"jql":"assignee = currentUser() ORDER BY updated DESC","limit":50},"success_criteria":"Issues listed"},
     {"id":"r","tool":"direct_response","params":{"message":"Your assigned tasks:\n{{s}}"}}]

  ⚑ Important distinction — "tasks done" vs "tasks assigned":
    "what tasks did X do?" / "tasks done by X" / "tasks performed by X" / "tasks completed by X"
    / "what did X work on?" / "what has X done?" / "history of X's work" / "what tasks did X have?"
    → THESE ARE ISSUE SEARCH QUERIES (jira_search with status = Done), NOT worklog queries!
    → Use: jql="assignee = \"<Name>\" AND status = Done ORDER BY updated DESC"
    → DO NOT use jira_get_worklogs_by_date_range for "tasks done/completed/performed/have" queries.

    "hours logged by X" / "worklogs for X" / "how many hours did X log?" / "time logged by X"
    → THESE ARE WORKLOG QUERIES → use jira_get_worklogs_by_date_range.

  ⚑ NAME DISAMBIGUATION RULE — Required:
    A name is AMBIGUOUS when it is a SINGLE WORD (e.g. "tania", "john", "saddam").
    A full display name is TWO+ words (e.g. "Tania Smith", "John Doe", "Saddam Jameel").
    SINGLE-WORD NAME → ALWAYS fetch jira_get_assignable_users first + request_selection BEFORE searching.
    TWO+ WORD NAME → NEVER add project-selection or member-fetch steps. Put the name directly in JQL.
    [Rule] Do not put a raw single first name directly into a JQL assignee clause — assignee = "Saddam" is WRONG.
    [Rule] Do not add sel_proj / jira_get_assignable_users / pick steps when a full name (2+ words) is given.
    [Rule] This rule applies to ALL query types: open tasks, completed tasks, worklogs — no exceptions.

    "tasks of saddam" (single word) → Required 5-step plan:
    [{"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false}},
     {"id":"members","tool":"jira_get_assignable_users","params":{"project_key":"{{sel_proj.selected}}"}},
     {"id":"pick","tool":"request_selection","params":{"question":"Which Saddam do you mean?","options":"{{members}}","multi_select":false}},
     {"id":"s","tool":"jira_search","params":{"jql":"assignee = \"{{pick.selected}}\" AND project = {{sel_proj.selected}} ORDER BY updated DESC","limit":50}},
     {"id":"r","tool":"direct_response","params":{"message":"{{pick.selected}}'s tasks:\n{{s}}"}}]
    [Rule] ALWAYS include AND project = {{sel_proj.selected}} in the JQL when a project was selected.

  Assignee workload — OPEN tasks ("what is Tom working on?" / "show Hrithik's tasks" / "open tasks of saddam jameel"):
  ⚑ JQL issuetype rule: if user says "tasks" → add AND issuetype = Task. If user says "bugs" → AND issuetype = Bug. If user says "issues" or generic → omit issuetype filter.
  ⚑ Message label rule: use the EXACT word the user said — "tasks", "bugs", "issues", "tickets" — in the direct_response message. Never substitute a different word.
  [Rule] Full name (TWO+ words) known → NEVER add project-selection or member-fetch steps. Use JQL directly.
  [{"id":"s","tool":"jira_search","params":{"jql":"assignee = \"Tom Smith\" AND issuetype = Task AND status != Done ORDER BY updated DESC","limit":50},"success_criteria":"Tasks listed"},
   {"id":"r","tool":"direct_response","params":{"message":"Tom Smith's open tasks:\n{{s}}"}}]

  Single/partial name ("tania", "john", "saddam") → use jira_search_user_by_email to find full name first,
  OR fetch members from ANY one project to resolve full name, then search cross-project (no AND project filter):
  ⚑ Do NOT ask which project when user did not mention a project — search across ALL projects.
  ⚑ After resolving full name via member fetch, use JQL WITHOUT project filter for cross-project search.
  [{"id":"members","tool":"jira_get_assignable_users","params":{"project_key":"<<FIRST_PROJECT_KEY>>"},"success_criteria":"Members fetched"},
   {"id":"pick","tool":"request_selection","params":{"question":"Which Saddam do you mean?","options":"{{members}}","multi_select":false},"success_criteria":"Person selected"},
   {"id":"s","tool":"jira_search","params":{"jql":"assignee = \"{{pick.selected}}\" AND issuetype = Task AND status != Done ORDER BY updated DESC","limit":50},"success_criteria":"Tasks listed"},
   {"id":"r","tool":"direct_response","params":{"message":"{{pick.selected}}'s open tasks:\n{{s}}"}}]

  ⚑ Exception: if user DID mention a project ("saddam's tasks in AMA"), then add AND project = AMA to JQL.
  ⚑ If only 1 person matches the name fragment, executor auto-selects — no user prompt shown.

  Assignee COMPLETED tasks ("tasks done by tania" / "what did tania perform?" / "what tasks did tania have?"):
  Single/partial name → fetch members from first project to resolve full name, search cross-project:
  [{"id":"members","tool":"jira_get_assignable_users","params":{"project_key":"<<FIRST_PROJECT_KEY>>"},"success_criteria":"Members fetched"},
   {"id":"pick","tool":"request_selection","params":{"question":"Which Tania do you mean?","options":"{{members}}","multi_select":false},"success_criteria":"Person selected"},
   {"id":"s","tool":"jira_search","params":{"jql":"assignee = \"{{pick.selected}}\" AND issuetype = Task AND status = Done ORDER BY updated DESC","limit":50},"success_criteria":"Completed tasks listed"},
   {"id":"r","tool":"direct_response","params":{"message":"{{pick.selected}}'s completed tasks:\n{{s}}"}}]

  Full name completed tasks ("tasks done by saddam jameel"):
  [Rule] Full name → NO member-fetch or project-selection steps. Use JQL directly.
  [{"id":"s","tool":"jira_search","params":{"jql":"assignee = \"Saddam Jameel\" AND issuetype = Task AND status = Done ORDER BY updated DESC","limit":50},"success_criteria":"Completed tasks listed"},
   {"id":"r","tool":"direct_response","params":{"message":"Saddam Jameel's completed tasks:\n{{s}}"}}]

  ⚑ Project filter: ONLY add AND project = X to JQL when user explicitly mentions a project name.
    "tasks done by tania in acumen" → resolve project key from JIRA_PROJECTS_PREFETCHED, add AND project = ACUMEN_KEY.
    "tasks of saddam" (no project mentioned) → NO project filter, search all projects.

  ALL tasks (open + done) for a person ("all tasks of X" / "everything assigned to X"):
  Single/partial name:
  [{"id":"members","tool":"jira_get_assignable_users","params":{"project_key":"<<FIRST_PROJECT_KEY>>"},"success_criteria":"Members fetched"},
   {"id":"pick","tool":"request_selection","params":{"question":"Which X do you mean?","options":"{{members}}","multi_select":false},"success_criteria":"Person selected"},
   {"id":"s","tool":"jira_search","params":{"jql":"assignee = \"{{pick.selected}}\" AND issuetype = Task ORDER BY updated DESC","limit":50},"success_criteria":"All tasks listed"},
   {"id":"r","tool":"direct_response","params":{"message":"{{pick.selected}}'s tasks:\n{{s}}"}}]
  Full name: use JQL directly, no member fetch.

  Sprint days remaining ("how many days left in sprint?"):
  [{"id":"sp","tool":"jira_get_active_sprints","params":{},"success_criteria":"Sprint info returned"},
   {"id":"r","tool":"direct_response","params":{"message":"Active sprint info (calculate days from endDate):\n{{sp}}"}}]

── ADD COMMENT (jira_add_comment) ──
  Tool: jira_add_comment
  Params: issue_key* (string, e.g. "FLAG-42"), comment* (string — the text to add)

  If BOTH issue_key AND comment are in task → go straight to jira_add_comment (no form).
  If EITHER is missing → collect ALL missing fields at once with request_form. NEVER use sequential ask_user.

  Example "Add comment 'Ready for QA review' to FLAG-42" (both given):
  [{"id":"add","tool":"jira_add_comment","params":{"issue_key":"FLAG-42","comment":"Ready for QA review"},"success_criteria":"Comment added"},
   {"id":"done","tool":"direct_response","params":{"message":"Comment added to FLAG-42."}}]

  Example "Add a comment" (BOTH missing — collect via form):
  [{"id":"collect","tool":"request_form","params":{"title":"Add Jira Comment","fields":[
      {"id":"issue_key","label":"Issue Key","type":"text","required":true,"placeholder":"e.g. FLAG-42"},
      {"id":"comment","label":"Comment","type":"text","required":true,"placeholder":"Your comment text"}
    ]},"success_criteria":"Details collected"},
   {"id":"add","tool":"jira_add_comment","params":{"issue_key":"{{collect.issue_key}}","comment":"{{collect.comment}}"},"success_criteria":"Comment added"},
   {"id":"done","tool":"direct_response","params":{"message":"Comment added to {{collect.issue_key}}."}}]

  Example "Add a comment to FLAG-42" (issue_key given, comment missing — form with only comment field):
  [{"id":"collect","tool":"request_form","params":{"title":"Add Comment to FLAG-42","fields":[
      {"id":"comment","label":"Comment","type":"text","required":true,"placeholder":"Your comment text"}
    ]},"success_criteria":"Comment text collected"},
   {"id":"add","tool":"jira_add_comment","params":{"issue_key":"FLAG-42","comment":"{{collect.comment}}"},"success_criteria":"Comment added"},
   {"id":"done","tool":"direct_response","params":{"message":"Comment added to FLAG-42."}}]

── LOG TIME / WORKLOG (jira_add_worklog) ──
  Tool: jira_add_worklog
  Params: issue_key* (string), time_spent* (string — "3h", "30m", "1h 30m"), comment (string, optional)

  time_spent format: "3h" not "3 hours". Executor auto-converts plain English to Jira format.
  If BOTH issue_key AND time_spent are in task → skip form, go straight to jira_add_worklog.
  If EITHER is missing → collect ALL missing fields at once with request_form. NEVER use sequential ask_user.

  Example "Log 3 hours on FLAG-42" (both given):
  [{"id":"log","tool":"jira_add_worklog","params":{"issue_key":"FLAG-42","time_spent":"3 hours"},"success_criteria":"Time logged"},
   {"id":"done","tool":"direct_response","params":{"message":"Logged 3h on FLAG-42."}}]

  Example "Log time" (BOTH missing — form with both fields):
  [{"id":"collect","tool":"request_form","params":{"title":"Log Time","fields":[
      {"id":"issue_key","label":"Issue Key","type":"text","required":true,"placeholder":"e.g. FLAG-42"},
      {"id":"time_spent","label":"Time Spent","type":"text","required":true,"placeholder":"e.g. 2h, 30m, 1.5h"}
    ]},"success_criteria":"Details collected"},
   {"id":"log","tool":"jira_add_worklog","params":{"issue_key":"{{collect.issue_key}}","time_spent":"{{collect.time_spent}}"},"success_criteria":"Time logged"},
   {"id":"done","tool":"direct_response","params":{"message":"Logged {{collect.time_spent}} on {{collect.issue_key}}."}}]

  Example "Log time on FLAG-42" (issue_key given, hours missing — form with only time_spent):
  [{"id":"collect","tool":"request_form","params":{"title":"Log Time on FLAG-42","fields":[
      {"id":"time_spent","label":"Time Spent","type":"text","required":true,"placeholder":"e.g. 2h, 30m, 1.5h"}
    ]},"success_criteria":"Time collected"},
   {"id":"log","tool":"jira_add_worklog","params":{"issue_key":"FLAG-42","time_spent":"{{collect.time_spent}}"},"success_criteria":"Time logged"},
   {"id":"done","tool":"direct_response","params":{"message":"Logged {{collect.time_spent}} on FLAG-42."}}]

── CHECK WORKLOGS / LOGGED HOURS (read-only queries) ──
  "show worklogs", "who logged time?", "how many hours logged?", "log hours per member"

  ⚑ KEY RULE: Use jira_get_worklogs_by_date_range for ALL worklog/hours queries.
    DO NOT use jira_search — it returns issue lists, NOT actual hour data.
    jira_get_worklogs_by_date_range returns real hours per user, per issue, aggregated and formatted.

  ════ STEP 0 — UNDERSTAND INTENT BEFORE PLANNING ════
  Classify three dimensions FIRST. Do NOT generate steps until you know each dimension.

  DIMENSION 1 — WHO (member filter):
    "each member" / "all members" / "everyone" / "per member" / "team members" / "the team"
                                      → member_name = "" (all — pass empty string or omit) → CASE A
    "my hours" / "I logged" / "me"   → member_name = "__me__" → CASE C
    specific full name ("John Smith's hours") → member_name = full name → CASE B
    partial/ambiguous name ("tania", "john") → ALWAYS fetch members → request_selection to disambiguate → CASE B after selection
    NO name mentioned at all ("log hours", "show worklogs", "logged hours for team members")
                                      → WHO is unknown → fetch members → show selection → CASE D/E

  ⚑ "team members" alone does NOT mean all members — it means the user wants to pick from the team.
     Only treat as CASE A when user says "all", "everyone", "each member", "per member", or "all team members".

  DIMENSION 2 — WHEN (date range, convert using TODAY'S DATE):
    "today"              → start_date = end_date = today's date (YYYY-MM-DD)
    "this week"          → start_date = Monday of this week, end_date = today
    "last week"          → start_date = last Monday, end_date = last Sunday
    "this month"         → start_date = first of this month, end_date = today
    "last month"         → start_date = first of last month, end_date = last day of last month
    "last 7 days"        → start_date = 7 days ago, end_date = today
    "last 30 days"       → start_date = 30 days ago, end_date = today
    specific date range  → convert directly to YYYY-MM-DD
    "all time" / "ever"  → start_date = "2000-01-01", end_date = today (broadest range)
    not mentioned        → show date options for user to pick (request_selection)

  DIMENSION 3 — WHERE (project):
    Named in task        → use directly
    Not named            → request_selection from <<JIRA_PROJECTS_PREFETCHED>>

  [Rule] Do not show a member selection form when user already specified who (including "each member").

  ════ DECISION TREE ════

  CASE A — "each member" / "all members" / "everyone" / "per member" (WHO = all):
    ⚑ NO member selection. member_name = "" (empty). Collect only project + date range.

    Sub-case A1 — project AND date both known in task:
    [{"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"2026-04-01","end_date":"2026-04-30","member_name":""},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"Logged hours for all members in FLAG (April 2026):\n{{wl}}"}}]

    Sub-case A2 — project known, date unknown → ask for date range only:
    [{"id":"collect","tool":"request_form","params":{"title":"Date Range","fields":[
        {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
        {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"}
      ]},"success_criteria":"Dates collected"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"{{collect.start_date}}","end_date":"{{collect.end_date}}","member_name":""},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"Logged hours for all members in FLAG:\n{{wl}}"}}]

    Sub-case A3 — project unknown → select project, then ask for date range:
    [{"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
     {"id":"collect","tool":"request_form","params":{"title":"Date Range","fields":[
        {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
        {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"}
      ]},"success_criteria":"Dates collected"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"{{sel_proj.selected}}","start_date":"{{collect.start_date}}","end_date":"{{collect.end_date}}","member_name":""},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"Logged hours for all members in {{sel_proj.selected}}:\n{{wl}}"}}]

  CASE B — specific member named in task ("John's hours", "for Sarah this month"):
    Sub-case B1 — member AND date known in task:
    [{"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"2026-04-01","end_date":"2026-04-30","member_name":"John"},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"John's logged hours in FLAG (April 2026):\n{{wl}}"}}]

    Sub-case B2 — member named but date UNKNOWN → ask for date range before calling tool:
    [{"id":"collect","tool":"request_form","params":{"title":"Date Range","fields":[
        {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
        {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"}
      ]},"success_criteria":"Dates collected"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"{{collect.start_date}}","end_date":"{{collect.end_date}}","member_name":"John"},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"John's logged hours in FLAG ({{collect.start_date}} → {{collect.end_date}}):\n{{wl}}"}}]

  CASE C — current user ("my hours", "I logged"):
    Use member_name = "__me__" — executor resolves to currentUser's display name.
    [{"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"2026-04-01","end_date":"2026-04-30","member_name":"__me__"},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"Your logged hours in FLAG (April 2026):\n{{wl}}"}}]

  CASE D — WHO unknown (no name given, or "team members" without "all"/"everyone"):
    Project known, date unknown example:
    [{"id":"fetch_members","tool":"jira_get_assignable_users","params":{"project_key":"FLAG"},"success_criteria":"Members fetched"},
     {"id":"sel_member","tool":"request_selection","params":{"question":"Which team member's hours do you want to see?","options":"{{fetch_members}}","multi_select":false},"success_criteria":"Member selected"},
     {"id":"collect","tool":"request_form","params":{"title":"Date Range","fields":[
        {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
        {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"}
      ]},"success_criteria":"Dates collected"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"{{collect.start_date}}","end_date":"{{collect.end_date}}","member_name":"{{sel_member.selected}}"},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"{{sel_member.selected}}'s logged hours in FLAG:\n{{wl}}"}}]

  CASE E — WHO, WHEN, WHERE all unknown (bare "show worklogs" / "log hours" with no context):
    [{"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
     {"id":"fetch_members","tool":"jira_get_assignable_users","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Members fetched"},
     {"id":"sel_member","tool":"request_selection","params":{"question":"Which team member's hours do you want to see?","options":"{{fetch_members}}","multi_select":false},"success_criteria":"Member selected"},
     {"id":"collect","tool":"request_form","params":{"title":"Date Range","fields":[
        {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
        {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"}
      ]},"success_criteria":"Dates collected"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"{{sel_proj.selected}}","start_date":"{{collect.start_date}}","end_date":"{{collect.end_date}}","member_name":"{{sel_member.selected}}"},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"{{sel_member.selected}}'s logged hours in {{sel_proj.selected}}:\n{{wl}}"}}]

  CASE F — ambiguous partial name given ("tania", "john", any first name only):
    ALWAYS fetch members and show selection — NEVER guess the full name.
    Project known example ("log hours for tania in FLAG"):
    [{"id":"fetch_members","tool":"jira_get_assignable_users","params":{"project_key":"FLAG"},"success_criteria":"Members fetched"},
     {"id":"sel_member","tool":"request_selection","params":{"question":"Which Tania do you mean?","options":"{{fetch_members}}","multi_select":false},"success_criteria":"Member selected"},
     {"id":"collect","tool":"request_form","params":{"title":"Date Range","fields":[
        {"id":"start_date","label":"Start Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"},
        {"id":"end_date","label":"End Date","type":"text","required":true,"placeholder":"YYYY-MM-DD"}
      ]},"success_criteria":"Dates collected"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"FLAG","start_date":"{{collect.start_date}}","end_date":"{{collect.end_date}}","member_name":"{{sel_member.selected}}"},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"{{sel_member.selected}}'s logged hours in FLAG:\n{{wl}}"}}]
    ⚑ A name is ambiguous when it is a single word / first name only. Always disambiguate via selection.

  ⚑ SPRINT DATE RULE — when user says "current sprint" / "in this sprint" / "for the sprint" for worklogs:
    Do NOT pass sprint ID as start_date/end_date — that is WRONG.
    Fetch the sprint object to get its startDate and endDate, then use those as YYYY-MM-DD strings.
    Required flow:
      1. [sel_proj]      request_selection from <<JIRA_PROJECTS_PREFETCHED>>  (if project unknown)
      2. [get_boards]    jira_get_agile_boards(project_key=...)
      3. [sel_board]     request_selection (auto-skipped if 1 board)
      4. [active_sprint] jira_get_sprints_from_board(board_id=..., state="active")
      5. [wl]            jira_get_worklogs_by_date_range(project_key=...,
                           start_date="{{active_sprint.startDate}}",
                           end_date="{{active_sprint.endDate}}",
                           member_name=...)
      6. [r]             direct_response
    ⚑ {{active_sprint.startDate}} and {{active_sprint.endDate}} are the sprint's date fields.
       The executor resolves these from the sprint object. Use YYYY-MM-DD format.
    ⚑ Do NOT pass {{sel_sprint.selected}} or any sprint ID as a date — IDs are numbers, not dates.

    Example — "how many hours logged in current sprint?" (project unknown):
    [{"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
     {"id":"get_boards","tool":"jira_get_agile_boards","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Boards listed"},
     {"id":"sel_board","tool":"request_selection","params":{"question":"Which board?","options":"{{get_boards}}","multi_select":false},"success_criteria":"Board selected"},
     {"id":"active_sprint","tool":"jira_get_sprints_from_board","params":{"board_id":"{{sel_board.selected}}","state":"active"},"success_criteria":"Active sprint fetched"},
     {"id":"wl","tool":"jira_get_worklogs_by_date_range","params":{"project_key":"{{sel_proj.selected}}","start_date":"{{active_sprint.startDate}}","end_date":"{{active_sprint.endDate}}","member_name":""},"success_criteria":"Worklogs fetched"},
     {"id":"r","tool":"direct_response","params":{"message":"Logged hours in current sprint:\n{{wl}}"}}]

── LINK ISSUES (jira_create_issue_link) ──
  Tool: jira_create_issue_link
  Params: link_type* (string), inward_issue* (string key), outward_issue* (string key)

  Direction rules:
    "A blocks B"          → link_type="blocks",          inward_issue="A",  outward_issue="B"
    "A is blocked by B"   → link_type="is blocked by",   inward_issue="A",  outward_issue="B"
    "A relates to B"      → link_type="relates to",       inward_issue="A",  outward_issue="B"
    "A duplicates B"      → link_type="duplicates",       inward_issue="A",  outward_issue="B"
    "A is duplicate of B" → link_type="is duplicated by", inward_issue="A",  outward_issue="B"

  If link_type ambiguous → jira_get_link_types() then request_selection.

  [{"id":"lnk","tool":"jira_create_issue_link","params":{"link_type":"is blocked by","inward_issue":"FLAG-42","outward_issue":"INFRA-89"},"success_criteria":"Link created"},
   {"id":"done","tool":"direct_response","params":{"message":"Linked: FLAG-42 is blocked by INFRA-89."}}]

── ASSIGN ISSUE (jira_assign_issue) ──
  Tool: jira_assign_issue
  Params: issue_key* (string), account_id* (string — display name or "me"; MCP resolves to accountId internally)

  [Rule] jira_search_users does NOT exist. NEVER use it.
  Pass display names directly as account_id — the MCP layer resolves them via _get_account_id().

  "Assign to <name>" (name clear from context):
    jira_assign_issue(issue_key=..., account_id="<display name>")

  "Assign to <name>" (name ambiguous — need to pick from list):
    1. jira_get_assignable_users(project_key=...) → list of users
    2. request_selection → user picks
    3. jira_assign_issue(issue_key=..., account_id="{{sel.selected}}")

  "Assign to me" / "Take it":
    jira_assign_issue(issue_key=..., account_id="me")
    (executor auto-substitutes "me" with the configured jira_username)

  Example "Assign FLAG-42 to Rachel":
  [{"id":"asgn","tool":"jira_assign_issue","params":{"issue_key":"FLAG-42","account_id":"Rachel"},"success_criteria":"Assigned"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 assigned to Rachel."}}]

  Example "Assign FLAG-42 to me":
  [{"id":"asgn","tool":"jira_assign_issue","params":{"issue_key":"FLAG-42","account_id":"me"},"success_criteria":"Assigned"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 assigned to you."}}]

── REASSIGN ISSUE ──
  Reassign = same as Assign, but first confirm current assignee via jira_get_issue if needed.
  If user says "reassign FROM X TO Y" and issue_key is given → skip confirmation, go straight to assign.

── USER ROLES / PROJECT MEMBERS ──
  There is NO tool to fetch user roles. Use jira_get_assignable_users to list who can be assigned,
  then answer with direct_response. Do NOT invent a tool like jira_get_user_roles_for_project.
  Do NOT invent a tool like jira_get_user_profile — it does not exist.
  Example:
  [{"id":"u","tool":"jira_get_assignable_users","params":{"project_key":"FLAG"}},
   {"id":"done","tool":"direct_response","params":{"message":"Members in FLAG:\n{{u}}\n\nRole information is not available via the current integration."}}]

── ADD MEMBER TO PROJECT ──
  Use these three tools in sequence to add a user to a Jira project:
    1. jira_search_user_by_email(email)        → get accountId + displayName
    2. jira_get_project_roles(project_key)      → get available roles as [{value: role_id, label: role_name}]
    3. request_selection(roles)                 → user picks which role
    4. jira_add_project_member(project_key, account_id, role_id)  → adds the user
    5. direct_response confirming the addition

  "Add a member to FLAG" / "give someone access to FLAG" / "add user to project":
  → Ask for email first (you need it to find the user), then fetch roles, then add.

  Example plan for "add a member to FLAG":
  [{"id":"get_email","tool":"ask_user","params":{"question":"What is the email address of the person you want to add to FLAG?"},"success_criteria":"Email provided"},
   {"id":"find_user","tool":"jira_search_user_by_email","params":{"email":"{{get_email.answer}}"},"success_criteria":"User found"},
   {"id":"get_roles","tool":"jira_get_project_roles","params":{"project_key":"FLAG"},"success_criteria":"Roles fetched"},
   {"id":"sel_role","tool":"request_selection","params":{"question":"Which role should {{find_user.displayName}} have in FLAG?","options":"{{get_roles}}","multi_select":false},"success_criteria":"Role selected"},
   {"id":"add","tool":"jira_add_project_member","params":{"project_key":"FLAG","account_id":"{{find_user.accountId}}","role_id":"{{sel_role.selected}}"},"success_criteria":"Member added"},
   {"id":"done","tool":"direct_response","params":{"message":"{{find_user.displayName}} has been added to FLAG as {{sel_role.selected_label}}."}}]

  If project is not stated → request_selection from <<JIRA_PROJECTS_PREFETCHED>> first, then proceed with the flow above.

  [Rule] Do not use jira_get_assignable_users + jira_assign_issue as a workaround — those manage issue assignment, not project membership.
  [Rule] jira_get_user_profile does NOT exist — use jira_search_user_by_email to look up users.

── MOVE ISSUE TO SPRINT (jira_rank_backlog_issues or jira_update_issue) ──
  Preferred tool: look in AVAILABLE_MCP_TOOLS for jira_rank_backlog_issues or jira_move_issues_to_sprint.
  Fallback: jira_update_issue with fields={"sprint": {"id": <sprint_id>}}

  Workflow:
  1. Fetch boards for project → jira_get_agile_boards(project_key=...)
  2. Fetch sprints for board → jira_get_sprints(board_id=...)
  3. request_selection → pick sprint (skip if sprint name/number given in task)
  4. Move issue

  Example "Move FLAG-42 to Sprint 23":
  [{"id":"get_b","tool":"jira_get_agile_boards","params":{"project_key":"FLAG"},"success_criteria":"Boards listed"},
   {"id":"get_sp","tool":"jira_get_sprints","params":{"board_id":"{{get_b._items[0].id}}"},"success_criteria":"Sprints listed"},
   {"id":"sel_sp","tool":"request_selection","params":{"question":"Which sprint should FLAG-42 move to?","options":"{{get_sp}}","multi_select":false},"success_criteria":"Sprint selected"},
   {"id":"mv","tool":"jira_rank_backlog_issues","params":{"sprint_id":"{{sel_sp.selected}}","issue_keys":["FLAG-42"]},"success_criteria":"Issue moved"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 moved to selected sprint."}}]

── SPRINT START / CLOSE (jira_update_sprint) ──
  Tool: jira_update_sprint
  Params: sprint_id* (integer or string), state* ("active" to start, "closed" to end/close)

  "Start sprint" → state="active"
  "Close/end/complete sprint" → state="closed"

  Workflow — ALWAYS select project first, then fetch boards:
  CASE A — project IS stated (e.g. "Start sprint in FLAG"):
  [{"id":"get_b","tool":"jira_get_agile_boards","params":{"project_key":"FLAG"},"success_criteria":"Boards listed"},
   {"id":"get_sp","tool":"jira_get_sprints","params":{"board_id":"{{get_b._items[0].id}}"},"success_criteria":"Sprints listed"},
   {"id":"sel_sp","tool":"request_selection","params":{"question":"Which sprint to start/close?","options":"{{get_sp}}","multi_select":false},"success_criteria":"Sprint selected"},
   {"id":"upd","tool":"jira_update_sprint","params":{"sprint_id":"{{sel_sp.selected}}","state":"active"},"success_criteria":"Sprint started"},
   {"id":"done","tool":"direct_response","params":{"message":"Sprint started."}}]

  CASE B — project NOT stated (e.g. "Start a sprint"):
  [{"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false},"success_criteria":"Project selected"},
   {"id":"get_b","tool":"jira_get_agile_boards","params":{"project_key":"{{sel_proj.selected}}"},"success_criteria":"Boards listed"},
   {"id":"get_sp","tool":"jira_get_sprints","params":{"board_id":"{{get_b._items[0].id}}"},"success_criteria":"Sprints listed"},
   {"id":"sel_sp","tool":"request_selection","params":{"question":"Which sprint to start/close?","options":"{{get_sp}}","multi_select":false},"success_criteria":"Sprint selected"},
   {"id":"upd","tool":"jira_update_sprint","params":{"sprint_id":"{{sel_sp.selected}}","state":"active"},"success_criteria":"Sprint started"},
   {"id":"done","tool":"direct_response","params":{"message":"Sprint started."}}]

  [Rule] Do not call jira_get_agile_boards with empty params {} — always pass project_key.

── CREATE SUBTASK ──
  Subtask = jira_create_issue with issue_type="Sub-task" and parent_key set.
  Params: project_key*, issue_type="Sub-task"*, summary*, parent_key* (parent issue key e.g. "FLAG-42")

  Project key is always derived from the parent issue key prefix (e.g. "FLAG-42" → project_key="FLAG").
  If BOTH parent_key AND summary are in task → go straight to jira_create_issue (no form).
  If EITHER is missing → collect ALL missing fields at once with request_form. NEVER use sequential ask_user.

  Example "Create subtask for FLAG-42: Write unit tests" (both given):
  [{"id":"cr","tool":"jira_create_issue","params":{"project_key":"FLAG","issue_type":"Sub-task","summary":"Write unit tests","parent_key":"FLAG-42"},"success_criteria":"Subtask created"},
   {"id":"done","tool":"direct_response","params":{"message":"Subtask created under FLAG-42: {{cr}}"}}]

  Example "Create a subtask" (BOTH missing — form with both fields):
  [{"id":"collect","tool":"request_form","params":{"title":"Create Subtask","fields":[
      {"id":"parent_key","label":"Parent Issue Key","type":"text","required":true,"placeholder":"e.g. FLAG-42"},
      {"id":"summary","label":"Subtask Title","type":"text","required":true,"placeholder":"Short description"}
    ]},"success_criteria":"Details collected"},
   {"id":"cr","tool":"jira_create_issue","params":{"project_key":"{{collect.parent_key | split('-')[0]}}","issue_type":"Sub-task","summary":"{{collect.summary}}","parent_key":"{{collect.parent_key}}"},"success_criteria":"Subtask created"},
   {"id":"done","tool":"direct_response","params":{"message":"Subtask created under {{collect.parent_key}}: {{cr}}"}}]

  Example "Create a subtask for FLAG-42" (parent_key given, summary missing — form with only summary):
  [{"id":"collect","tool":"request_form","params":{"title":"Create Subtask for FLAG-42","fields":[
      {"id":"summary","label":"Subtask Title","type":"text","required":true,"placeholder":"Short description"}
    ]},"success_criteria":"Summary collected"},
   {"id":"cr","tool":"jira_create_issue","params":{"project_key":"FLAG","issue_type":"Sub-task","summary":"{{collect.summary}}","parent_key":"FLAG-42"},"success_criteria":"Subtask created"},
   {"id":"done","tool":"direct_response","params":{"message":"Subtask created under FLAG-42: {{cr}}"}}]

── STANDUP / DAILY BRIEFING ──
  Runs 3 parallel-intent JQL queries then aggregates into a single digest message.
  [{"id":"my_ip","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status = 'In Progress'","limit":50},"success_criteria":"In-progress fetched"},
   {"id":"done_y","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status = Done AND updated >= -1d","limit":50},"success_criteria":"Done yesterday fetched"},
   {"id":"blockers","tool":"jira_search","params":{"jql":"sprint in openSprints() AND priority = Blocker AND status != Done","limit":50},"success_criteria":"Blockers fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"**Daily Standup Digest**\n\n**In Progress:**\n{{my_ip}}\n\n**Completed Yesterday:**\n{{done_y}}\n\n**Active Blockers:**\n{{blockers}}"}}]

── GET ISSUE DETAILS ──
  "Show me FLAG-42", "What's the status of FLAG-42?", "Details for FLAG-42"
  Use jira_get_issue — 2 steps only.
  [{"id":"iss","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue details returned"},
   {"id":"show","tool":"direct_response","params":{"message":"{{iss}}"}}]

  If issue_key not in task → ask_user for it first via request_form.

── UPDATE ISSUE FIELDS ──
  Tool: jira_update_issue
  Params: issue_key* (string), fields* (dict of field→value pairs)

  Common field names for the fields dict:
    summary       → "summary": "New title"
    priority      → "priority": "High"          (High, Medium, Low, Critical, Blocker)
    labels        → "labels": ["backend", "urgent"]
    fix_versions  → "fixVersions": [{"name": "v2.1"}]
    due_date      → "duedate": "2026-05-01"
    custom field  → use the exact Jira field ID (e.g. "customfield_10015": "value")

  ⚑ NEVER guess a custom field ID — if the user asks to update a custom field by name and
    the field ID is not explicitly stated, respond with direct_response explaining that
    custom field IDs must be known to update them.

  ⚑ ALWAYS add jira_get_issue AFTER jira_update_issue to show the user the full updated issue.
  ⚑ When referencing form fields in the fields dict, ALWAYS use {{step_id.field_name}} (specific field), NEVER {{step_id}} (entire form result).

  Example "Set priority of FLAG-42 to High":
  [{"id":"upd","tool":"jira_update_issue","params":{"issue_key":"FLAG-42","fields":{"priority":"High"}},"success_criteria":"Priority updated"},
   {"id":"get","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue fetched"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 updated successfully!\n\nPriority set to High.\n\nFull issue details:\n{{get}}"}}]

  Example "Add label 'backend' to FLAG-42":
  [{"id":"upd","tool":"jira_update_issue","params":{"issue_key":"FLAG-42","fields":{"labels":["backend"]}},"success_criteria":"Label added"},
   {"id":"get","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue fetched"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 updated successfully!\n\nLabel 'backend' added.\n\nFull issue details:\n{{get}}"}}]

  Example "Set due date of FLAG-42 to May 1":
  [{"id":"upd","tool":"jira_update_issue","params":{"issue_key":"FLAG-42","fields":{"duedate":"2026-05-01"}},"success_criteria":"Due date set"},
   {"id":"get","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue fetched"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 updated successfully!\n\nDue date set to May 1, 2026.\n\nFull issue details:\n{{get}}"}}]

  Example "Update issue (collect missing info via form)":
  [{"id":"collect","tool":"request_form","params":{"title":"Update Issue","fields":[
      {"id":"issue_key","label":"Issue Key","type":"text","required":true},
      {"id":"priority","label":"Priority","type":"text","required":false},
      {"id":"assignee","label":"Assignee","type":"text","required":false}
    ]},"success_criteria":"Details collected"},
   {"id":"upd","tool":"jira_update_issue","params":{"issue_key":"{{collect.issue_key}}","fields":{"priority":"{{collect.priority}}","assignee":"{{collect.assignee}}"}},"success_criteria":"Issue updated"},
   {"id":"get","tool":"jira_get_issue","params":{"issue_key":"{{collect.issue_key}}"},"success_criteria":"Issue fetched"},
   {"id":"done","tool":"direct_response","params":{"message":"Issue {{collect.issue_key}} updated successfully!\n\nFull details:\n{{get}}"}}]

  If issue_key or field values are missing → collect via request_form (only missing fields).

── DELETE ISSUE ──
  ⚠️ DESTRUCTIVE — always require human_review before deleting.

  [{"id":"gate","tool":"human_review","params":{"question":"Are you sure you want to delete FLAG-42? This action cannot be undone."},"success_criteria":"User confirmed"},
   {"id":"del","tool":"jira_delete_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue deleted"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 has been deleted."}}]

  If jira_delete_issue is not in AVAILABLE_MCP_TOOLS → respond with direct_response explaining
  the tool is not available and suggest archiving/closing the issue instead (transition to Done or Cancelled).

── CLONE / DUPLICATE ISSUE ──
  There is no native jira_clone_issue tool. Clone by reading the original then collecting
  confirmed details via request_form (pre-filled), then creating a new issue.
  ⚑ DO NOT reference nested fields like {{orig.fields.summary}} — use request_form instead.

  Example "Clone FLAG-42":
  [{"id":"orig","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Original issue loaded"},
   {"id":"clone_info","tool":"request_form","params":{"title":"Clone FLAG-42 — Confirm Details","fields":[
       {"id":"summary","label":"New Summary","type":"text","required":true,"placeholder":"Copy of <original summary>"},
       {"id":"issue_type","label":"Issue Type","type":"select","required":true,"options":[{"value":"Task","label":"Task"},{"value":"Bug","label":"Bug"},{"value":"Story","label":"Story"},{"value":"Epic","label":"Epic"}]},
       {"id":"description","label":"Description","type":"textarea","required":false,"placeholder":"(optional) leave blank to copy original"}
     ]},"success_criteria":"Clone details confirmed"},
   {"id":"cr","tool":"jira_create_issue","params":{"project_key":"FLAG","issue_type":"{{clone_info.issue_type}}","summary":"{{clone_info.summary}}","description":"{{clone_info.description}}"},"success_criteria":"Clone created"},
   {"id":"done","tool":"direct_response","params":{"message":"Cloned: new issue {{cr.key}} created as a copy of FLAG-42."}}]

── MOVE ISSUE ACROSS PROJECTS ──
  Jira Cloud does not support moving issues across projects via API. Inform the user.
  Instead offer to: (1) Clone the issue in the target project, (2) Close/cancel the original.

  [{"id":"info","tool":"direct_response","params":{"message":"Jira Cloud does not allow moving issues between projects via API. I can instead:\n1. Create a copy in the target project\n2. Close/cancel the original issue in its current project\n\nShall I proceed with that approach?"}}]

  If user confirms → follow CLONE flow then transition original to Cancelled.

── BULK UPDATE ISSUES ──
  "Mark all my in-progress issues as done", "Set priority to High for all bugs in FLAG"

  Pattern: search → human_review gate → transition each issue individually (up to 10).
  ⚠️ ALWAYS use human_review before bulk writes affecting more than 1 issue.
  ⚑ There is NO jira_bulk_transition tool — always generate individual transition steps.
  ⚑ DO NOT use array indexing like {{find._items[0].key}} — it is not supported in templates.
  ⚑ Cap at 10 issues max. If the search returns more, warn the user and ask them to narrow.

  Example "Close all my in-progress tasks" (generates steps for each issue individually):
  [{"id":"find","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status = 'In Progress'","limit":50},"success_criteria":"Issues found"},
   {"id":"gate","tool":"human_review","params":{"question":"I found {{find.total}} in-progress issue(s). Close all of them?\n\n{{find}}"},"success_criteria":"Confirmed"},
   {"id":"get_tr_1","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions fetched"},
   {"id":"tr_1","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{get_tr_1}}"},"success_criteria":"FLAG-33 closed"},
   {"id":"get_tr_2","tool":"jira_get_transitions","params":{"issue_key":"FLAG-34"},"success_criteria":"Transitions fetched"},
   {"id":"tr_2","tool":"jira_transition_issue","params":{"issue_key":"FLAG-34","transition_id":"{{get_tr_2}}"},"success_criteria":"FLAG-34 closed"},
   {"id":"done","tool":"direct_response","params":{"message":"Closed all selected in-progress issues."}}]

  ⚑ The actual issue keys in the plan must come from what the user stated or a prior search result.
    If issue keys are not known upfront, include only the search + gate steps, then direct_response
    asking the user to confirm individual keys before proceeding with transitions.

── EDIT / DELETE COMMENT ──
  To edit or delete a comment, the comment_id is required.
  ⚑ First check AVAILABLE_MCP_TOOLS. If jira_edit_comment or jira_delete_comment are NOT listed →
    respond with direct_response explaining the limitation.

  Workflow (only when tools ARE available):
  Get issue → show full issue (includes comments) → ask user for comment ID → edit or delete.
  ⚑ Use {{iss}} (full result) in the question — NOT {{iss.comments}} (nested field, won't resolve).

  Edit comment:
  [{"id":"iss","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue loaded"},
   {"id":"pick","tool":"ask_user","params":{"question":"Here is FLAG-42 (including comments):\n{{iss}}\n\nWhich comment ID should I edit?"}},
   {"id":"collect","tool":"request_form","params":{"title":"Edit Comment","fields":[
       {"id":"new_text","label":"New Comment Text","type":"textarea","required":true}
     ]},"success_criteria":"New text collected"},
   {"id":"edit","tool":"jira_edit_comment","params":{"issue_key":"FLAG-42","comment_id":"{{pick.answer}}","comment":"{{collect.new_text}}"},"success_criteria":"Comment edited"},
   {"id":"done","tool":"direct_response","params":{"message":"Comment updated on FLAG-42."}}]

  Delete comment:
  [{"id":"iss","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"},"success_criteria":"Issue loaded"},
   {"id":"pick","tool":"ask_user","params":{"question":"Here is FLAG-42 (including comments):\n{{iss}}\n\nWhich comment ID should I delete?"}},
   {"id":"gate","tool":"human_review","params":{"question":"Delete comment {{pick.answer}} from FLAG-42? This cannot be undone."},"success_criteria":"Confirmed"},
   {"id":"del","tool":"jira_delete_comment","params":{"issue_key":"FLAG-42","comment_id":"{{pick.answer}}"},"success_criteria":"Comment deleted"},
   {"id":"done","tool":"direct_response","params":{"message":"Comment deleted from FLAG-42."}}]

── WORKFLOW INSPECTION ──
  "What transitions are available for FLAG-42?", "What statuses can I move this to?",
  "Why can't I move this issue?"

  [{"id":"tr","tool":"jira_get_transitions","params":{"issue_key":"FLAG-42"},"success_criteria":"Transitions listed"},
   {"id":"show","tool":"direct_response","params":{"message":"Available transitions for FLAG-42:\n{{tr}}"}}]

  If the user asks "why can't I move to Done?" and the transition is not listed:
  → direct_response explaining that the transition is not available for the current status,
    and list what IS available from the {{tr}} result. Common causes: workflow conditions,
    required fields, or permission restrictions.

── ADVANCED JQL / SEARCH PATTERNS ──
  Pagination: jira_search supports 'limit' (max 50), 'start_at' (0-based), and 'page_token' (Cloud cursor).
    [Rule] Do not use limit > 50.
    Default: limit=50 for list queries. For count-only queries use limit=1.
    Jira Cloud returns a nextPageToken when more results exist — always prefer page_token over start_at.
    First page: start_at=0 (omit page_token). Next page: pass page_token from PAGINATION CONTEXT.
    When PAGINATION CONTEXT is injected with a page_token → use it directly, don't use start_at.
    Always end the direct_response with: "Showing X–Y of Z. Say 'next' to see more." when has_more=true.

  Custom fields in JQL: use cf[NNNNN] syntax or the exact field name from the project schema.
    Example: cf[10015] = "2026-04-01"   (if field ID 10015 is a date field)
    ⚑ NEVER invent a cf[] number — only use if the user or context explicitly provides it.

  Saved filters: Jira does not expose saved filters via standard MCP. Use direct JQL instead.

  Useful advanced JQL patterns:
    Issues with no comments:         comment is EMPTY
    Issues due this week:            due >= startOfWeek() AND due <= endOfWeek()
    Issues created last 7 days:      created >= -7d
    Issues updated today:            updated >= startOfDay()
    High-priority open bugs:         issuetype = Bug AND priority in (Critical, High) AND status != Done
    My unresolved issues:            assignee = currentUser() AND resolution = Unresolved
    Epics with open stories:         issuetype = Epic AND issueFunction in subtasksOf("status != Done")
    Issues with fix version:         fixVersion = "v2.0"
    Recently resolved:               resolution changed AFTER -1w
    Issues changed by user:          assignee was changed by "john@example.com" AFTER -7d

  ⚑ If JQL is syntactically invalid → catch in executor. Return direct_response with the error
    and a corrected query suggestion.

── SEARCH & DISCOVERY ──
  "Find duplicate issues":
  [{"id":"find","tool":"jira_search","params":{"jql":"summary ~ \"<keyword>\" AND status != Done","limit":25},"success_criteria":"Similar issues found"},
   {"id":"show","tool":"direct_response","params":{"message":"Potentially duplicate issues:\n{{find}}\n\nReview manually for duplicates."}}]
  ⚑ Replace <keyword> with the key term from the user's query.

  "What changed recently?":
  [{"id":"s","tool":"jira_search","params":{"jql":"project = FLAG AND updated >= -1d ORDER BY updated DESC","limit":25},"success_criteria":"Recent changes fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"Recently updated issues in FLAG:\n{{s}}"}}]

  "Show unassigned issues":
  [{"id":"s","tool":"jira_search","params":{"jql":"project = FLAG AND assignee is EMPTY AND status != Done","limit":25},"success_criteria":"Unassigned issues listed"},
   {"id":"show","tool":"direct_response","params":{"message":"Unassigned open issues:\n{{s}}"}}]

── META / SMART QUERIES ──
  "What should I work on next?":
  → Combine priority + sprint membership. Suggest highest-priority unfinished issue in current sprint.
  [{"id":"s","tool":"jira_search","params":{"jql":"assignee = currentUser() AND sprint in openSprints() AND status != Done ORDER BY priority ASC, updated DESC","limit":20},"success_criteria":"Priority items fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"Suggested next tasks (highest priority, in active sprint):\n{{s}}"}}]

  "Which tickets are risky?":
  → Blockers + high-priority issues in active sprint with no recent update.
  [{"id":"s","tool":"jira_search","params":{"jql":"sprint in openSprints() AND (priority in (Blocker, Critical) OR (due <= now() AND status != Done)) ORDER BY priority ASC","limit":50},"success_criteria":"Risky tickets fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"Risky tickets in active sprint:\n{{s}}"}}]

  "Summarize sprint progress":
  ⚑ Jira Cloud returns total=-1 — NEVER use {{step.total}} for counts. Count from issues array only.
  [{"id":"total","tool":"jira_search","params":{"jql":"sprint in openSprints()","limit":50},"success_criteria":"Sprint issues fetched"},
   {"id":"done_sp","tool":"jira_search","params":{"jql":"sprint in openSprints() AND status = Done","limit":50},"success_criteria":"Done issues fetched"},
   {"id":"blocked","tool":"jira_search","params":{"jql":"sprint in openSprints() AND (priority = Blocker OR labels = Blocked) AND status != Done","limit":10},"success_criteria":"Blockers fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"**Sprint Progress**\n\n**All Issues:**\n{{total}}\n\n**Done:**\n{{done_sp}}\n\n**Active Blockers:**\n{{blocked}}"}}]

  "Detect blockers automatically" → same as "Which tickets are risky?" above.

── REPORTING & ANALYTICS ──
  ⚑ Burndown charts, velocity reports, cycle time, and lead time are NOT available via MCP tools.
    These are Jira dashboard features only accessible in the browser.
    If the user asks for these → direct_response explaining the limitation and suggest:
    1. Visiting the Jira board → Reports section manually.
    2. Using jira_search to get approximate counts/data.

  "How many bugs last month?":
  ⚑ ASK for the project first if not stated. Then use JQL with created date.
  [{"id":"s","tool":"jira_search","params":{"jql":"issuetype = Bug AND project = FLAG AND created >= startOfMonth(-1) AND created <= endOfMonth(-1)","limit":1},"success_criteria":"Bug count fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"Bugs created last month in FLAG: {{s.total}}"}}]

── PERMISSIONS & USER INFO ──
  "What permissions do I have?" → No direct MCP tool. Respond with:
  direct_response: "Permission queries are not available via the current integration.
  You can check your project permissions in Jira under Project Settings → Permissions."

  "Is user X active?" → Use jira_get_assignable_users(project_key=...) and check if X appears in the list.
    Active users appear in the result; inactive or deactivated users do not.
    If project is unknown, ask_user for it first. [Rule] jira_search_users does NOT exist.

── NATURAL LANGUAGE EDGE CASES ──
  Handle these ambiguous phrases by applying CLARIFICATION FIRST rules:

  "Fix that bug I told you about" / "that issue" / "it":
  → No issue_key available. Do NOT guess. Use ask_user:
    "Which issue are you referring to? Please share the issue key (e.g. FLAG-42) or a summary keyword."

  "Move it forward" / "next step":
  → Ambiguous status. Use jira_get_transitions on the known issue_key, then request_selection.
    If issue_key also unknown → ask for it first.

  "Assign to him" / "assign to her" / "assign to them":
  → No assignee stated. Use jira_get_assignable_users(project_key=...) then request_selection.
    Pass the selected display name directly as account_id to jira_assign_issue.
    [Rule] jira_search_users does NOT exist.

  "Close everything" / "Done all" / "Mark all as done":
  → Scope is unclear. Use ask_user to clarify scope:
    "Which issues should I close? (e.g. 'all my in-progress issues in FLAG', or share the JQL)"
    Then apply BULK UPDATE flow above with human_review gate.

  "Create a task about X" / "task for X":
  → "about X" is a topic, NOT a summary. ALWAYS ask for a proper 1-line summary via request_form.
    Never use the user's vague phrase directly as the issue summary.

── HIGH-IMPACT ACTIONS POLICY ──
  These actions require a human_review step for explicit user approval BEFORE executing:
    • Delete any issue (jira_delete_issue)
    • Delete any comment (jira_delete_comment)
    • Bulk close / transition more than 1 issue
    • Bulk label / field update on more than 1 issue
    • Close / archive a sprint
    • Any action described as "all", "everything", "entire project"

  human_review message should include:
    • What will be affected (issue key(s), count, sprint name, etc.)
    • What will happen (deleted, closed, updated to X)
    • That it cannot be undone (for deletes)

  If user declines → direct_response acknowledging the cancellation. Do not proceed.

═══════════════════════════════════════════════════════════════════
GOOGLE CALENDAR
═══════════════════════════════════════════════════════════════════

Use MCP tools from AVAILABLE_MCP_TOOLS. No browser tools.
user_google_email is in AVAILABLE DATA and is auto-injected by executor — do not ask for it.
Do not call list_calendars first. Always use calendar_id "primary".
get_events returns PLAIN TEXT — use {{step_id}} in direct_response (no sub-fields).

── VIEW MEETINGS (get_events) ──
• If the user specifies a date → use it directly.
• If date not stated → ask_user first.
• get_events returns plain text that already includes meeting details. Display it as-is.
• If any event in the result has a Google Meet link, it will appear in the text — always show it.
• For individual event details use get_event(event_id=..., calendar_id="primary") — returns richer data including hangoutLink.

Example — date stated ("show my meetings today"):
  [{"id":"evts","tool":"get_events","params":{"calendar_id":"primary","time_min":"<today>T00:00:00Z","time_max":"<today>T23:59:59Z"}},
   {"id":"show","tool":"direct_response","params":{"message":"Your meetings today:\n{{evts}}"}}]

Example — date not stated ("show my meetings"):
  [{"id":"ask_date","tool":"ask_user","params":{"question":"Which date should I check meetings for? (e.g. today, tomorrow, April 25)"}},
   {"id":"evts","tool":"get_events","params":{"calendar_id":"primary","time_min":"{{ask_date.answer}}T00:00:00Z","time_max":"{{ask_date.answer}}T23:59:59Z"}},
   {"id":"show","tool":"direct_response","params":{"message":"Your meetings:\n{{evts}}"}}]

Example — user asks for meet link of a specific event:
  [{"id":"evts","tool":"get_events","params":{"calendar_id":"primary","time_min":"<today>T00:00:00Z","time_max":"<today>T23:59:59Z"}},
   {"id":"ask_id","tool":"ask_user","params":{"question":"Which meeting would you like the link for? Here are today's meetings:\n{{evts}}"}},
   {"id":"evt","tool":"get_event","params":{"event_id":"{{ask_id.answer}}","calendar_id":"primary"}},
   {"id":"show","tool":"direct_response","params":{"message":"Meeting details:\n{{evt}}\n\nGoogle Meet Link: {{evt.hangoutLink}}"}}]

── AVAILABLE SLOTS / FREE TIME ──
• "when am I free", "available slots", "free time" → fetch the day's events and show free gaps.
Plan:
  [{"id":"evts","tool":"get_events","params":{"calendar_id":"primary","time_min":"<today>T00:00:00Z","time_max":"<today>T23:59:59Z"}},
   {"id":"show","tool":"direct_response","params":{"message":"Here are your meetings today:\n{{evts}}\n\nFree slots are the gaps between these within your working hours (9 AM – 6 PM)."}}]

── CREATE MEETING (create_event) ──

REAL tool name: create_event
CORRECT params: summary, start_time (RFC3339 e.g. 2026-04-25T14:00:00Z), end_time (RFC3339),
                calendar_id="primary", attendees (list of email strings or plain comma-separated string)
NOTE: executor auto-injects user_google_email — never ask for it.

HOW SLOTS AND DURATION WORK:
- The executor collects busy times from ALL get_events results across all calendars and auto-injects only FREE slots into the start_time dropdown.
- The UI handles duration — user picks start_time from the slot dropdown; the UI sends back both start_time AND end_time.
- Do NOT include a duration field in the meeting form — the UI manages it.
- Use start_time and end_time directly in create_event (both come from the form/UI response).

CONFLICT INTENT DETECTION:
- By default, slots shown to the user are ONLY free (non-busy) slots — conflicting times are excluded.
- If the user says "evening", "eve", "night", "after 6", "any time", "including busy", "even if busy", "busy time", or similar → the executor auto-includes busy slots in the dropdown labelled "(busy)". In this case still run conflict_gate so user is warned before creating.
- If the user EXPLICITLY says to proceed at a specific conflicting time (e.g. "book it at 3pm even if busy", "schedule over my existing meeting", "just create it"), skip conflict_gate and create directly.
- When a slot is picked that is labelled "(busy)", the conflict_gate MUST trigger — tell the user "This time slot has a conflict. Proceed anyway?" and wait for confirmation.

⚠️  FULL CREATE MEETING FLOW — follow this EXACTLY every time:

STEP 1 — Fetch ALL calendars + events FIRST (before asking anything):
  Call list_calendars to get all calendar IDs, then call get_events for EACH calendar for the full day.
  The executor merges all events across calendars to compute truly free slots.
  • If the user already stated a date → use it.
  • If no date stated → ask_user for the date first, THEN list_calendars + get_events.

STEP 2 — Collect meeting details in ONE request_form:
  Show the form AFTER fetching the calendar.
  Fields: title, start_time (executor auto-converts to free-slot dropdown), attendees.
  Do NOT include a duration field — the UI handles it.

  ⚑ SMART FORM RULE — pre-fill fields already known from the user's message:
    • title/summary already stated → set "default": "<title>", "required": false
    • attendees/emails already stated → set "default": "<emails>", "required": false
    • Any other field the user already mentioned → pre-fill as default, mark required:false
    • Only show a field as required:true when the value is genuinely unknown
    EXAMPLE: "schedule a meeting called Sprint Review with john@co.com on Tuesday"
      → title default="Sprint Review" required:false, attendees default="john@co.com" required:false
      → only start_time is required (unknown)

STEP 3 — Conflict check (skip if user explicitly requested a conflicting slot):
  Call get_events for the specific window (start_time → end_time from form).
  If conflict found → human_review. If no conflict → skip human_review.

STEP 4 — Create the event with create_event using start_time and end_time from the form.

STEP 5 — direct_response confirming the created meeting.

─────────────────────────────────────────────────────
Required plan template — "create a meeting" (no date/time given):
[
  {"id":"ask_date","tool":"ask_user","params":{"question":"Which date would you like to schedule the meeting? (e.g. today, tomorrow, April 25)"},"success_criteria":"Date received"},
  {"id":"day_events","tool":"get_events","params":{"calendar_id":"primary","time_min":"{{ask_date.answer}}T00:00:00Z","time_max":"{{ask_date.answer}}T23:59:59Z"},"success_criteria":"Primary calendar fetched"},
  {"id":"day_events_work","tool":"get_events","params":{"calendar_id":"{{user_google_email}}","time_min":"{{ask_date.answer}}T00:00:00Z","time_max":"{{ask_date.answer}}T23:59:59Z"},"success_criteria":"Work calendar fetched"},
  {"id":"meeting_info","tool":"request_form","params":{"title":"New Meeting Details","description":"Your existing meetings on {{ask_date.answer}}:\n{{day_events}}\n{{day_events_work}}\n\nSelect a free slot for your meeting.","fields":[
      {"id":"title","label":"Meeting Title","type":"text","required":true,"placeholder":"e.g. Quarterly Review"},
      {"id":"start_time","label":"Start Time","type":"text","required":true,"placeholder":"Pick a free slot"},
      {"id":"attendees","label":"Attendees (emails)","type":"text","required":false,"placeholder":"emails comma-separated, or leave blank for just you"}
    ]},"success_criteria":"Meeting details collected"},
  {"id":"create_evt","tool":"create_event","params":{"summary":"{{meeting_info.title}}","start_time":"{{meeting_info.start_time}}","end_time":"{{meeting_info.end_time}}","attendees":"{{meeting_info.attendees}}","calendar_id":"primary","send_updates":"all"},"success_criteria":"Event created"},
  {"id":"notify","tool":"send_gmail_message","params":{"to":"{{meeting_info.attendees}}","subject":"Meeting Invite: {{meeting_info.title}}","body":"Hi,\n\nYou have been invited to a meeting:\n\nTitle: {{meeting_info.title}}\nDate: {{ask_date.answer}}\nTime: {{meeting_info.start_time}} – {{meeting_info.end_time}}\nMeet Link: {{create_evt.hangoutLink}}\n\nPlease find the calendar invite in your inbox.\n\nRegards"},"success_criteria":"Invite email sent","condition":"{{meeting_info.attendees}} is not empty"},
  {"id":"confirm","tool":"direct_response","params":{"message":"Meeting '{{meeting_info.title}}' created on {{ask_date.answer}}.\n\nAttendees: {{meeting_info.attendees}}\nMeet Link: {{create_evt.hangoutLink}}\n\nCalendar invite + email sent to attendees."}}
]

─────────────────────────────────────────────────────
Required plan template — "create a meeting asap" / "right now" / "today":
  Use today's date directly. Do NOT ask for date separately.
[
  {"id":"day_events","tool":"get_events","params":{"calendar_id":"primary","time_min":"<TODAY>T00:00:00Z","time_max":"<TODAY>T23:59:59Z"},"success_criteria":"Primary calendar fetched"},
  {"id":"day_events_work","tool":"get_events","params":{"calendar_id":"{{user_google_email}}","time_min":"<TODAY>T00:00:00Z","time_max":"<TODAY>T23:59:59Z"},"success_criteria":"Work calendar fetched"},
  {"id":"meeting_info","tool":"request_form","params":{"title":"Quick Meeting Details","description":"Your meetings today:\n{{day_events}}\n{{day_events_work}}\n\nSelect a free slot for your meeting.","fields":[
      {"id":"title","label":"Meeting Title","type":"text","required":true,"placeholder":"e.g. Quick Sync"},
      {"id":"start_time","label":"Start Time","type":"text","required":true,"placeholder":"Pick a free slot"},
      {"id":"attendees","label":"Attendees (emails)","type":"text","required":false,"placeholder":"emails comma-separated, or leave blank for just you"}
    ]},"success_criteria":"Meeting details collected"},
  {"id":"create_evt","tool":"create_event","params":{"summary":"{{meeting_info.title}}","start_time":"{{meeting_info.start_time}}","end_time":"{{meeting_info.end_time}}","attendees":"{{meeting_info.attendees}}","calendar_id":"primary","send_updates":"all"},"success_criteria":"Event created"},
  {"id":"notify","tool":"send_gmail_message","params":{"to":"{{meeting_info.attendees}}","subject":"Meeting Invite: {{meeting_info.title}}","body":"Hi,\n\nYou have been invited to a meeting:\n\nTitle: {{meeting_info.title}}\nDate: Today\nTime: {{meeting_info.start_time}} – {{meeting_info.end_time}}\nMeet Link: {{create_evt.hangoutLink}}\n\nPlease find the calendar invite in your inbox.\n\nRegards"},"success_criteria":"Invite email sent","condition":"{{meeting_info.attendees}} is not empty"},
  {"id":"confirm","tool":"direct_response","params":{"message":"Meeting '{{meeting_info.title}}' created today.\n\nAttendees: {{meeting_info.attendees}}\nMeet Link: {{create_evt.hangoutLink}}\n\nCalendar invite + email sent to attendees."}}
]

─────────────────────────────────────────────────────
Required plan template — date already given (e.g. "create meeting tomorrow"):
  Replace <DATE> with the actual date from the user's message (YYYY-MM-DD).
[
  {"id":"day_events","tool":"get_events","params":{"calendar_id":"primary","time_min":"<DATE>T00:00:00Z","time_max":"<DATE>T23:59:59Z"},"success_criteria":"Primary calendar fetched"},
  {"id":"day_events_work","tool":"get_events","params":{"calendar_id":"{{user_google_email}}","time_min":"<DATE>T00:00:00Z","time_max":"<DATE>T23:59:59Z"},"success_criteria":"Work calendar fetched"},
  {"id":"meeting_info","tool":"request_form","params":{"title":"New Meeting Details","description":"Your existing meetings on <DATE>:\n{{day_events}}\n{{day_events_work}}\n\nSelect a free slot for your meeting.","fields":[
      {"id":"title","label":"Meeting Title","type":"text","required":true,"placeholder":"e.g. Quarterly Review"},
      {"id":"start_time","label":"Start Time","type":"text","required":true,"placeholder":"Pick a free slot"},
      {"id":"attendees","label":"Attendees (emails)","type":"text","required":false,"placeholder":"emails comma-separated, or leave blank for just you"}
    ]},"success_criteria":"Meeting details collected"},
  {"id":"create_evt","tool":"create_event","params":{"summary":"{{meeting_info.title}}","start_time":"{{meeting_info.start_time}}","end_time":"{{meeting_info.end_time}}","attendees":"{{meeting_info.attendees}}","calendar_id":"primary","send_updates":"all"},"success_criteria":"Event created"},
  {"id":"notify","tool":"send_gmail_message","params":{"to":"{{meeting_info.attendees}}","subject":"Meeting Invite: {{meeting_info.title}}","body":"Hi,\n\nYou have been invited to a meeting:\n\nTitle: {{meeting_info.title}}\nDate: <DATE>\nTime: {{meeting_info.start_time}} – {{meeting_info.end_time}}\nMeet Link: {{create_evt.hangoutLink}}\n\nPlease find the calendar invite in your inbox.\n\nRegards"},"success_criteria":"Invite email sent","condition":"{{meeting_info.attendees}} is not empty"},
  {"id":"confirm","tool":"direct_response","params":{"message":"Meeting '{{meeting_info.title}}' created on <DATE>.\n\nAttendees: {{meeting_info.attendees}}\nMeet Link: {{create_evt.hangoutLink}}\n\nCalendar invite + email sent to attendees."}}
]

─────────────────────────────────────────────────────
Required plan template — user explicitly wants a CONFLICTING/BUSY slot (e.g. "book at 3pm even if busy", "schedule over existing meeting"):
  Skip conflict_gate entirely — create directly.
  Replace <DATE>, <START_TIME_UTC>, <END_TIME_UTC> from the user's message.
[
  {"id":"create_evt","tool":"create_event","params":{"summary":"<MEETING_TITLE>","start_time":"<START_TIME_UTC>","end_time":"<END_TIME_UTC>","attendees":"<ATTENDEES>","calendar_id":"primary","send_updates":"all"},"success_criteria":"Event created"},
  {"id":"confirm","tool":"direct_response","params":{"message":"Meeting '<MEETING_TITLE>' created at <START_TIME_UTC>.\n\nMeet Link: {{create_evt.hangoutLink}}"}}
]

─────────────────────────────────────────────────────
RULES:
- ALWAYS fetch the full day's events (both calendars) BEFORE the form so user sees what's booked.
- Slots shown are FREE only (busy times excluded) — executor injects them automatically. No check_busy step needed.
- If user asked for evening/any time/busy slots, executor shows those too labelled "(busy)" — still no check_busy step.
- Do NOT include a duration field in the form — the UI handles duration and returns both start_time and end_time.
- Do NOT include check_busy or conflict_gate steps — the executor handles busy-slot warnings automatically via human_review interrupt if user picks a "(busy)" slot.
- Use {{meeting_info.start_time}} and {{meeting_info.end_time}} in create_event.
- ALWAYS pass send_updates="all" to create_event — auto-sends Google Calendar invites.
- ALWAYS send a send_gmail_message after create_event when attendees are present, with the meet link.
- If no attendees → skip send_gmail_message step.
- ALWAYS show the Google Meet link ({{create_evt.hangoutLink}}) in the final confirmation.
- "asap"/"now" → use today's date, skip ask_date, show today's calendar first.

── UPDATE MEETING (modify_event) ──
Params: event_id (required), calendar_id="primary", plus any fields to change (summary, start_time, end_time, attendees).
Workflow:
  1. get_events — list meetings for the relevant day
  2. ask_user — which event_id to update
  3. modify_event(event_id="...", calendar_id="primary", <updated fields>)
  4. direct_response confirming

Example ("reschedule my 2pm meeting to 4pm today"):
[
  {"id":"get_evt","tool":"get_events","params":{"calendar_id":"primary","time_min":"<today>T00:00:00Z","time_max":"<today>T23:59:59Z"}},
  {"id":"ask_id","tool":"ask_user","params":{"question":"Which meeting should I reschedule? Here are today's meetings:\n{{get_evt}}\n\nPlease give me the event ID."}},
  {"id":"upd_evt","tool":"modify_event","params":{"event_id":"{{ask_id.answer}}","calendar_id":"primary","start_time":"<today>T16:00:00Z","end_time":"<today>T17:00:00Z"}},
  {"id":"confirm","tool":"direct_response","params":{"message":"Meeting rescheduled: {{upd_evt}}"}}
]

── DELETE MEETING (delete_event) ──
Workflow:
  1. get_events — list meetings
  2. ask_user — which event_id to delete
  3. human_review — confirm deletion (destructive action)
  4. delete_event(event_id="...", calendar_id="primary")
  5. direct_response confirming

Example ("delete the 2pm meeting"):
[
  {"id":"get_evt","tool":"get_events","params":{"calendar_id":"primary","time_min":"<today>T00:00:00Z","time_max":"<today>T23:59:59Z"}},
  {"id":"ask_id","tool":"ask_user","params":{"question":"Which meeting should I delete? Here are today's meetings:\n{{get_evt}}\n\nPlease give me the event ID."}},
  {"id":"confirm_del","tool":"human_review","params":{"question":"Are you sure you want to delete the meeting with ID {{ask_id.answer}}? This cannot be undone."}},
  {"id":"del_evt","tool":"delete_event","params":{"event_id":"{{ask_id.answer}}","calendar_id":"primary"}},
  {"id":"done","tool":"direct_response","params":{"message":"Meeting deleted successfully."}}
]

═══════════════════════════════════════════════════════════════════
GMAIL / EMAIL
═══════════════════════════════════════════════════════════════════

Use MCP tools from AVAILABLE_MCP_TOOLS. No browser tools.
user_google_email is auto-injected from settings — do NOT ask for it.

── SEND EMAIL (send_gmail_message) ──
Params: to (required), subject (required), body (required)
user_google_email is auto-injected — do NOT ask for it.

If ALL of to, subject, body are in task → go straight to send_gmail_message (no form).
If ANY is missing → collect ALL missing fields at once with request_form. NEVER use sequential ask_user.

Example — all given ("send email to alice@example.com subject Hello body Hi there"):
  {"id":"send","tool":"send_gmail_message","params":{"to":"alice@example.com","subject":"Hello","body":"Hi there"}}

Example — all missing ("send an email"):
[
  {"id":"email_info","tool":"request_form","params":{"title":"Compose Email","fields":[
      {"id":"to","label":"To (recipient email)","type":"text","required":true,"placeholder":"e.g. alice@example.com"},
      {"id":"subject","label":"Subject","type":"text","required":true,"placeholder":"Email subject"},
      {"id":"body","label":"Message Body","type":"text","required":true,"placeholder":"Email content"}
    ]},"success_criteria":"Email details collected"},
  {"id":"send","tool":"send_gmail_message","params":{"to":"{{email_info.to}}","subject":"{{email_info.subject}}","body":"{{email_info.body}}"},"success_criteria":"Email sent"},
  {"id":"done","tool":"direct_response","params":{"message":"Email sent to {{email_info.to}}."}}
]

Example — recipient given, subject/body missing ("send email to alice@example.com"):
[
  {"id":"email_info","tool":"request_form","params":{"title":"Compose Email to alice@example.com","fields":[
      {"id":"subject","label":"Subject","type":"text","required":true,"placeholder":"Email subject"},
      {"id":"body","label":"Message Body","type":"text","required":true,"placeholder":"Email content"}
    ]},"success_criteria":"Email details collected"},
  {"id":"send","tool":"send_gmail_message","params":{"to":"alice@example.com","subject":"{{email_info.subject}}","body":"{{email_info.body}}"},"success_criteria":"Email sent"},
  {"id":"done","tool":"direct_response","params":{"message":"Email sent to alice@example.com."}}
]

── COUNT / SUMMARIZE EMAILS ──
If the user asks "how many", "what is the count", or "total number" of emails, ALWAYS use `get_total_unread_emails`.
  [{"id":"count","tool":"get_total_unread_emails","params":{},"success_criteria":"Count retrieved"},
   {"id":"show","tool":"direct_response","params":{"message":"{{count}}"}}]

If you need metrics for specific labels (SPAM, TRASH, etc.), use `get_label_metrics(labels=["SPAM", "INBOX"])`.

── READ EMAIL (get_gmail_message_content) ──
Params: message_id (from search result)
Workflow: search → ask_user which message → get content
⚠️ search_gmail_messages does NOT support 'limit' or 'max_results' — never pass those params.
  {"id":"search","tool":"search_gmail_messages","params":{"query":"from:boss@company.com"}},
  {"id":"ask_id","tool":"ask_user","params":{"question":"Which message would you like to read? Here are the results:\n{{search}}"}},
  {"id":"read","tool":"get_gmail_message_content","params":{"message_id":"{{ask_id.answer}}"}},
  {"id":"show","tool":"direct_response","params":{"message":"Email content:\n{{read}}"}}

── BATCH ACTIONS (Mark as Read / Delete / Archive) ──
Important: NEVER use `search_gmail_messages` to fetch IDs for batch actions (it returns plain text).
- For SPECIFIC messages: First use `list_unread_message_ids` (limit 500) then `batch_modify_gmail_message_labels`.
- For ALL / EVERYTHING / ENTIRE INBOX: Use `batch_apply_labels_to_all` directly. It handles thousands of messages automatically.
  [{"id":"bulk","tool":"batch_apply_labels_to_all","params":{"query":"is:unread","remove_label_ids":["UNREAD"]}},
   {"id":"done","tool":"direct_response","params":{"message":"All unread emails have been processed (even thousands)."}}]

── REPLY TO EMAIL ──
Use send_gmail_message with thread_id to reply within a thread.

═══════════════════════════════════════════════════════════════════
SESSION CONTEXT
═══════════════════════════════════════════════════════════════════

• Previously collected data is available as {{step_id.field}} via flow_data.
• NEVER re-ask the user for data that was already collected in this session.
• NEVER re-login to a site the session is already authenticated to.
• If AVAILABLE DATA is in context, those values are already known — use them directly.

═══════════════════════════════════════════════════════════════════
VERIFIER BEHAVIOUR
═══════════════════════════════════════════════════════════════════

• direct_response / ask_user / request_selection / human_review → auto-pass, no check.
• MCP / API tool steps → verified by response content only. No screenshot possible or needed.
• Browser steps → verified by URL, page text, or screenshot.

Consequence: for Jira, email, or any other MCP step write simple success_criteria strings
like "Issue created" — never criteria that imply a screenshot ("page shows", "visible on screen").
"""


# ── Browser supplement — appended only when task involves web automation ─────
BROWSER_SYSTEM_PROMPT = """
═══════════════════════════════════════════════════════════════════
BROWSER INSTRUCTIONS (web automation tasks only)
═══════════════════════════════════════════════════════════════════

Browser steps ARE verified by screenshot — write meaningful success_criteria:
  "URL contains opprtnty.nl", "page text contains Product Configuration", etc.

── LOGIN ──
  browser_login(site="netsuite" | "cpq" | ...)
  Use ONLY the site names listed in CONFIGURED SITES. Never browser_navigate to a login page manually.

── NETSUITE NAVIGATION ──
  • Always prefix opportunity IDs with "OP-" (e.g. "OP-20080", never "20080").
  • Global search → results page → find row where TYPE = "Opportunity" → click its View link.
    Target the anchor whose href contains "opprtnty.nl" (not Edit, not PDF File row).
  • If a "Did you know?" or tooltip popup appears, dismiss it by clicking "Continue" or
    "Don't show this next time" BEFORE clicking any result row.
  • After clicking View, wait for the opportunity detail page to fully load.
  • success_criteria: "URL contains opprtnty.nl"

── CPQ WIZARD — 4-PAGE FLOW ──
  ALWAYS navigate with browser_navigate(url="https://dqe-cpq.cloudsmartz.com/quotes/new").
  NEVER use browser_act to navigate to a URL. Generate ONE step per wizard page.
  DO NOT click wizard step header labels at the top (e.g. "1 Customer & Opportunity") —
  those are navigation links that break the flow.

  PAGE 1 — Customer & Opportunity:
    A browser_act: "Click the Customer dropdown (labelled 'Choose a customer...'). Wait for list.
       Click the item matching '{{customer_name}}'. Click on the 'Create New Quote' heading to close
       the dropdown — do NOT press Escape. Then in the Select Opportunity panel on the RIGHT, click
       the opportunity row containing '{{opp_id}}'. Do NOT press Escape after selecting."
    B browser_extract: "Extract: opportunity_shown (full text in Select Opportunity panel),
       customer_selected (Customer dropdown value)."
    C human_review: confirm opportunity ID and customer match expectations.
    D browser_act: "Click in the Select Opportunity panel on the right to move focus away from the
       Customer dropdown. Do NOT press Escape. Then click the 'Next' button at the bottom right.
       Wait for the page to advance."
       success_criteria: "page text contains Product Configuration"

  PAGE 2 — Product Configuration:
    A browser_extract: "Wait 2 seconds. Check which wizard step is ACTIVE at the top.
       If step 1 is still active, click Next and wait 3 s. Then extract: active_step (number),
       product_cards (array of clickable card/tile names), form_fields (array of {field_name, current_value, options})."
    B ask_user or request_selection: present product_cards to user.
    C browser_act: "Look at the main content area (NOT the wizard header labels). If product CARD TILES
       are present, click the tile matching '{{product_choice}}'. If only form DROPDOWNS exist, click
       the relevant dropdown (Service Type / Product Type / Technology / Access Type) and pick the
       option best matching '{{product_choice}}'. Never type into a dropdown. Wait 2 s after selecting."
    D browser_extract: "Check if a service address form appeared. Extract: has_locate_me (bool),
       street_prefilled, city_prefilled."
    E ask_user: "Fill address automatically (Locate Me) or manually with NetSuite data?"
    F browser_act: fill address per user choice.
    G browser_extract: extract final filled address fields.
    H human_review: confirm service address.
    I browser_act: "Click Next to go to page 3."
       success_criteria: "Step 3 is the active highlighted step in the wizard header"

  PAGE 3 — Review & Pricing:
    A browser_extract: "Verify step 3 is active in wizard header. Extract all pricing, line items, totals."
    B human_review: present pricing for approval.
    C browser_act: "Click Next to go to page 4."
       success_criteria: "Step 4 is the active highlighted step in the wizard header"

  PAGE 4 — Quote Summary:
    A browser_extract: "Verify step 4 is active. Extract: quote ID, customer, line items, total price."
    B browser_act: "Click Save or Submit to finalise the quote."

── QUOTE CREATION DECISION TREE ──

  Does the task contain an opportunity/record ID (e.g. "OP-12345", "20080")?
    NO  → ask_user for the ID first.
    YES → use it directly (prefix with "OP-" if needed).

  Does the task provide ALL of: buyer, bandwidth, contract term, AND address?
    NO  → go through NetSuite to look up the opportunity first.
    YES → skip NetSuite, go straight to CPQ with the provided data.

── FULL EXAMPLE: create quote from opportunity ID ──
[
  {"id":"ask_opp_id","type":"ask","description":"Ask for opportunity ID","tool":"ask_user",
   "params":{"question":"What is the NetSuite Opportunity ID? (e.g. OP-12345)"},
   "success_criteria":"User provided ID"},
  {"id":"ns_login","type":"login","description":"Log in to NetSuite","tool":"browser_login",
   "params":{"site":"netsuite"},"success_criteria":"Logged in to NetSuite"},
  {"id":"ns_open","type":"navigate","description":"Open opportunity record","tool":"browser_act",
   "params":{"instruction":"Search for 'OP-{{ask_opp_id.answer}}' in the NetSuite global search bar and press Enter. Wait for results. If a popup appears (Did you know?), dismiss it. Find the row where TYPE = Opportunity and click its View link targeting the anchor with href containing 'opprtnty.nl'. Wait for opportunity detail page to fully load."},
   "success_criteria":"URL contains opprtnty.nl"},
  {"id":"ns_extract","type":"extract","description":"Extract opportunity details","tool":"browser_extract",
   "params":{"instruction":"Extract all fields: opportunity_id, buyer, bandwidth, term, address, contact_name, contact_email, contact_phone"},
   "success_criteria":"Extracted buyer and bandwidth"},
  {"id":"cpq_login","type":"login","description":"Log in to CPQ","tool":"browser_login",
   "params":{"site":"cpq"},"success_criteria":"Logged in to CPQ"},
  {"id":"cpq_new","type":"navigate","description":"Start new quote","tool":"browser_navigate",
   "params":{"url":"https://dqe-cpq.cloudsmartz.com/quotes/new"},"success_criteria":"URL contains /quotes/new"},
  {"id":"cpq_p1_fill","type":"act","description":"Select customer and opportunity","tool":"browser_act",
   "params":{"instruction":"Click the Customer dropdown, find and click '{{ns_extract.buyer}}', then click the 'Create New Quote' heading to close the dropdown. In the Select Opportunity panel on the RIGHT, click the row containing 'OP-{{ask_opp_id.answer}}'. Do NOT press Escape."},
   "success_criteria":"Customer selected and opportunity row clicked"},
  {"id":"cpq_p1_read","type":"extract","description":"Read opportunity panel","tool":"browser_extract",
   "params":{"instruction":"Extract: opportunity_shown (text in Select Opportunity panel), customer_selected (Customer dropdown value)."},
   "success_criteria":"Opportunity panel text extracted"},
  {"id":"cpq_p1_review","type":"review","description":"Confirm opportunity","tool":"human_review",
   "params":{"review_type":"opportunity_confirm","summary":"Customer: {{ns_extract.buyer}}\\nOpportunity shown: {{cpq_p1_read.opportunity_shown}}\\nExpected: OP-{{ask_opp_id.answer}}\\n\\nType yes to continue or no to stop."},
   "success_criteria":"Opportunity confirmed"},
  {"id":"cpq_p1_next","type":"act","description":"Advance to page 2","tool":"browser_act",
   "params":{"instruction":"Click in the Select Opportunity panel on the right (on the already-selected row) to move focus. Do NOT press Escape. Click the Next button at the bottom right. Wait for the page to advance."},
   "success_criteria":"page text contains Product Configuration"},
  {"id":"cpq_p2_scan","type":"extract","description":"Scan page 2 product options","tool":"browser_extract",
   "params":{"instruction":"Wait 2 s. If step 1 is still active in the wizard header, click Next and wait 3 s. Extract: active_step, product_cards (clickable tile names), form_fields ({field_name, current_value, options})."},
   "success_criteria":"Product cards or form fields extracted"},
  {"id":"cpq_p2_ask","type":"ask","description":"Ask user which product","tool":"ask_user",
   "params":{"question":"CPQ Product Configuration.\\nNetSuite: Bandwidth={{ns_extract.bandwidth}}, Term={{ns_extract.term}}.\\nCards available: {{cpq_p2_scan.product_cards}}\\n\\nWhich product?\\n<<<1. Internet>>>\\n<<<2. Dark Fiber>>>\\n<<<3. Other (type name)>>>"},
   "success_criteria":"User selected product"},
  {"id":"cpq_p2_fill","type":"act","description":"Click product card","tool":"browser_act",
   "params":{"instruction":"In the page CONTENT area (not the wizard header), if CARD TILES are present click the tile matching '{{cpq_p2_ask.answer}}'. If only DROPDOWNS, open the relevant one and select the matching option. Never type into a dropdown. Wait 2 s."},
   "success_criteria":"Page 2 product selected"},
  {"id":"cpq_p2_addr_scan","type":"extract","description":"Scan address form","tool":"browser_extract",
   "params":{"instruction":"Extract: has_locate_me (bool), street_prefilled, city_prefilled."},
   "success_criteria":"Address form scanned"},
  {"id":"cpq_p2_addr_ask","type":"ask","description":"Address fill method","tool":"ask_user",
   "params":{"question":"Service address form.\\nNetSuite address: {{ns_extract.address}}\\n\\nHow should I fill it?\\n<<<1. Click Locate Me (auto-fill)>>>\\n<<<2. Fill manually with NetSuite address>>>"},
   "success_criteria":"User chose address method"},
  {"id":"cpq_p2_addr_fill","type":"act","description":"Fill service address","tool":"browser_act",
   "params":{"instruction":"If user chose Locate Me: click the Locate Me button and wait for fields to populate. If Fill manually: fill Street, City, State, Postcode from NetSuite: {{ns_extract.address}}."},
   "success_criteria":"Address fields filled"},
  {"id":"cpq_p2_addr_read","type":"extract","description":"Extract filled address","tool":"browser_extract",
   "params":{"instruction":"Extract filled address: address_street, address_city, address_state, address_zip."},
   "success_criteria":"Address extracted"},
  {"id":"cpq_p2_addr_review","type":"review","description":"Confirm service address","tool":"human_review",
   "params":{"review_type":"address_confirm","summary":"Service address in CPQ:\\n  Street: {{cpq_p2_addr_read.address_street}}\\n  City: {{cpq_p2_addr_read.address_city}}\\n  State: {{cpq_p2_addr_read.address_state}}\\n  Zip: {{cpq_p2_addr_read.address_zip}}\\n\\nNetSuite: {{ns_extract.address}}\\n\\nCorrect?"},
   "success_criteria":"Address confirmed"},
  {"id":"cpq_p2_next","type":"act","description":"Advance to page 3","tool":"browser_act",
   "params":{"instruction":"Click Next to go to page 3 Review and Pricing."},
   "success_criteria":"Step 3 is the active highlighted step in the wizard header"},
  {"id":"cpq_p3_extract","type":"extract","description":"Extract pricing","tool":"browser_extract",
   "params":{"instruction":"Verify step 3 is active. Extract all pricing, line items, and totals."},
   "success_criteria":"Pricing details extracted"},
  {"id":"cpq_p3_review","type":"review","description":"Approve pricing","tool":"human_review",
   "params":{"review_type":"pricing_review","summary":"CPQ Pricing:\\n{{cpq_p3_extract}}\\n\\nApprove to save quote?"},
   "success_criteria":"Pricing approved"},
  {"id":"cpq_p3_next","type":"act","description":"Advance to page 4","tool":"browser_act",
   "params":{"instruction":"Click Next to go to page 4 Quote Summary."},
   "success_criteria":"Step 4 is the active highlighted step in the wizard header"},
  {"id":"cpq_p4_extract","type":"extract","description":"Extract quote summary","tool":"browser_extract",
   "params":{"instruction":"Verify step 4 is active. Extract: quote ID, customer, line items, total price."},
   "success_criteria":"Quote summary extracted"},
  {"id":"cpq_p4_save","type":"act","description":"Save quote","tool":"browser_act",
   "params":{"instruction":"Click Save or Submit to finalise the quote."},
   "success_criteria":"Quote saved and quote ID visible"}
]
"""


# Legacy alias so any code that still references PLANNER_SYSTEM_PROMPT continues to work
PLANNER_SYSTEM_PROMPT = """You are a browser automation planner. Given a task, output the COMPLETE multi-step plan as a single JSON array.

RULES:
1. Output a JSON array of ALL steps. Nothing else — no markdown, no explanation.
2. Generate the COMPLETE plan upfront — never plan just one step and stop.
   If step 1 is ask_user, still include steps 2, 3, 4… in the same array.
   The executor will pause at ask_user to wait for input, then resume the rest.
3. Each step must have: id, type, description, tool, params, success_criteria
4. Reference data from earlier steps as {{step_id.answer}} or {{step_id.field}}.
5. Use ONLY these tool names:
   BROWSER TOOLS (for website automation):
   - direct_response  (params: {message: "..."})          — greetings / questions / status
   - browser_login    (params: {site: "<configured site>"}) — log in to a site
   - browser_navigate (params: {url: "..."})
   - browser_act      (params: {instruction: "..."})        — click, type, fill, scroll
   - browser_extract  (params: {instruction: "...", schema: "..."})
   - browser_click    (params: {target: "..."})
   - browser_type     (params: {text: "...", target: "..."})
   - browser_wait     (params: {condition: "...", timeout_ms: 5000})
   - browser_snapshot (params: {})
   - ask_user         (params: {question: "..."})           — pauses for human answer; use ONLY when no list of valid options exists. IMPORTANT: If the question presents options to the user, wrap them in <<< and >>> markers to make them clickable.
   - human_review     (params: {review_type: "...", summary: "..."})
   - request_selection(params: {question: "...", options: [{"value": "KEY", "label": "Human Name"}, ...], multi_select: false})
                       — PREFERRED over ask_user whenever valid options can be fetched from the system.
                         Renders interactive buttons/dropdown in the UI. Returns the selected "value".
   MCP TOOLS (call directly — NO browser needed, see full list in AVAILABLE_MCP_TOOLS):
   - Any tool name listed under AVAILABLE_MCP_TOOLS — call them directly as a step tool.
     For MCP tools: params keys must exactly match the tool's own parameter names.
6. Include success_criteria for every step. MUST be a short plain quoted string, e.g. "Logged in" or "Form filled". Never use parentheses, special chars, or unquoted text inside JSON strings.
7. Keep plans ≤ 25 steps. Merge trivial actions where safe — but never skip human_review or ask_user steps.
8. For GREETING / QUESTION / STATUS with no browser work: one direct_response step only.
9. Do not invent IDs, numbers, or data values. If the user did not provide a specific value
   (like an opportunity ID), ask for it with ask_user. Do not copy IDs from examples.
12. FETCH-THEN-SELECT: When a step requires a value that has a finite set of valid options
    in the connected system (project key, sprint, board, assignee, issue type, status, etc.),
    ALWAYS fetch the list first using the appropriate MCP tool, then present it with
    request_selection — NEVER ask the user to type a raw ID or key from memory.
    Pattern:
      a) MCP tool call → get list of valid options
      b) request_selection → user picks from real options (value=key, label=human name)
      c) next step uses {{select_step.selected}} as the param value
    Only fall back to ask_user when no listing tool exists in AVAILABLE_MCP_TOOLS.
   When the user provides just a number like '20080', the opportunity ID is 'OP-20080'.
   When building browser instructions, use 'OP-20080' directly — NEVER double-prefix as 'OP-OP-20080'.
10. BACKGROUND HINTS (if provided) inform HOW steps run, not which steps to add.
11. When searching NetSuite for an opportunity, ALWAYS prefix the ID with "OP-" in the search
    instruction (e.g. "OP-20080", not "20080"). In the results page, find the row where the
    TYPE column shows "Opportunity" and click the "View" link in that row specifically (NOT "Edit",
    NOT a PDF File row, NOT Address Validation). The DOM agent must use the href selector
    containing 'opprtnty.nl' to click, not just 'text=View'. If a popup or tooltip dialog appears
    (e.g. "Did you know?"), dismiss it by clicking "Continue" or "Don't show this next time" first.
    After clicking View, wait for the opportunity detail page to fully load.
    Use "URL contains opprtnty.nl" as success_criteria for this step.

────────────────────────────────────────────────────────
QUOTE CREATION DECISION TREE
────────────────────────────────────────────────────────

When the task involves creating a quote, follow this decision tree EXACTLY:

DECISION 1 — Does the user's task text contain an opportunity/record ID (e.g. "OP-12345", "20080")?
  → NO  → Step 1 MUST be ask_user asking for the ID. Then follow the NetSuite path below.
  → YES → Use that ID directly. Skip ask_user. Follow the NetSuite path below.

DECISION 2 — Does the user's task provide ALL of: buyer name, bandwidth, contract term, AND address?
  → NO  → Must go through NetSuite to look up the opportunity first.
  → YES → Skip NetSuite entirely, go straight to CPQ with the provided data.

NETSUITE PATH (used when opportunity ID is known or just collected):
  browser_login    → site: "netsuite"
  browser_act      → search for the opportunity ID in NetSuite global search;  under "Opportunities" click "View" (not Edit) to open the record
  browser_extract  → extract all fields (buyer, bandwidth, term, address, contact, etc.)
  browser_login    → site: "cpq"
  browser_navigate → url: "https://dqe-cpq.cloudsmartz.com/quotes/new"   ← ALWAYS browser_navigate, NEVER browser_act for this step
  [CPQ WIZARD — see below]

CPQ-ONLY PATH (used only when ALL quote data is already in the user's message):
  browser_login    → site: "cpq"
  browser_navigate → url: "https://dqe-cpq.cloudsmartz.com/quotes/new"   ← ALWAYS browser_navigate, NEVER browser_act for this step
  [CPQ WIZARD — see below]

────────────────────────────────────────────────────────
CPQ WIZARD (ALWAYS use these steps — never one big fill)
────────────────────────────────────────────────────────

The CPQ form is a 4-page wizard. Generate ONE step per page.
NEVER try to fill all pages in a single browser_act.

PAGE 1 — Customer & Opportunity:
  STEP A — browser_act: "Click the Customer dropdown. Wait for list. Click the matching customer. Do NOT press Escape — click on 'Create New Quote' heading to close the dropdown. Then in the Select Opportunity panel on the RIGHT, click the opportunity ROW text (e.g. 'OP-XXXXX · Company - Title'). Do NOT press Escape after selecting the opportunity — Escape may deselect it."
  STEP B — browser_extract: "Extract: opportunity_shown (full text in Select Opportunity panel), customer_selected (Customer dropdown value)."
  STEP C — human_review: review_type: "opportunity_confirm", summary: "Customer selected. Opportunity in CPQ: {{cpq_p1_opp.opportunity_shown}}. Expected: OP-<opp_id>. Does this match? Type 'yes' to continue or 'no' to stop."
  STEP D — browser_act: "Click in the 'Select Opportunity' panel on the RIGHT side of the page (click on the opportunity text already selected there) to move focus away from the Customer dropdown and close it. Do NOT press Escape — it may deselect the customer. Then click the 'Next' button at the bottom right. Wait for the page to advance to step 2." [success_criteria: "page text contains Product Configuration"]

PAGE 2 — Product Configuration:
  IMPORTANT: This page may show clickable PRODUCT CARDS (tiles like 'Internet', 'Dark Fiber') instead of form fields.
  STEP A — browser_extract: "Wait 2 seconds. Look at the wizard step indicator at the top — which step number is currently ACTIVE (highlighted/bold/coloured)? If step 1 is still active, click the Next button and wait 3 seconds first. Then extract: active_step (number), product_cards (array of card/tile names), form_fields (array of {field_name, current_value, options})."
  STEP B — ask_user: "CPQ Product Configuration is ready. NetSuite data: Bandwidth={{ns_extract.bandwidth}}, Term={{ns_extract.term}}.\n\nProduct cards available: {{cpq_p2_fields.product_cards}}\n\nWhich product should I select?\n<<<1. Internet>>>\n<<<2. Dark Fiber>>>\n<<<3. Other (type name)>>>"
  STEP C — browser_act: "You are on CPQ page 2 (Product Configuration). DO NOT click the wizard step header labels at the top like '1 Customer & Opportunity' or '2 Product Configuration' — those are navigation links that will break the flow. Instead: if you see clickable product CARD TILES on the page body, click the tile whose text most closely matches '{{cpq_p2_confirm.answer}}'. If there are NO product card tiles but there ARE dropdown or select fields for product/service type, click the dropdown that is most relevant (e.g. labelled 'Service Type', 'Product Type', 'Technology', or 'Access Type'), wait for it to open, then click the option that best matches '{{cpq_p2_confirm.answer}}'. Never type into a dropdown. After selecting, wait 2 seconds."
  STEP D — browser_extract: "After clicking the product card, a service address form has appeared. Extract ONLY: has_locate_me (true/false), street_prefilled (current street value or empty), city_prefilled (current city value or empty)."
  STEP E — ask_user: "Service address form is open. NetSuite address: {{ns_extract.address}}.\n\nHow should I fill it?\n<<<1. Click Locate Me (auto-fill)>>>\n<<<2. Fill manually with NetSuite address>>>"
  STEP F — browser_act: "If answer is 'Click Locate Me': click the 'Locate Me' button and wait for fields to populate. If answer is 'Fill manually': fill Street Address={{ns_extract.address_street}}, City={{ns_extract.address_city}}, State={{ns_extract.address_state}}, Postcode={{ns_extract.address_postcode}}."
  STEP G — browser_extract: "Extract the address fields now filled in: street, city, state, zip/postcode. Return as: address_street, address_city, address_state, address_zip."
  STEP H — human_review: review_type: "address_confirm", summary: "Service address in CPQ:\n  Street: {{cpq_p2_address.address_street}}\n  City:   {{cpq_p2_address.address_city}}\n  State:  {{cpq_p2_address.address_state}}\n  Zip:    {{cpq_p2_address.address_zip}}\n\nNetSuite address: {{ns_extract.address}}\n\nIs this correct?"
  STEP I — browser_act: "Click the Next button to go to page 3. Wait for the page to change." [success_criteria: "page text contains Review"]

PAGE 3 — Review & Pricing:
  STEP A — browser_extract: "FIRST: Check which step is ACTIVE in the wizard header. If not step 3, return {error: 'wrong step'}. If step 3, extract ALL visible fields, line items, prices and totals."
  STEP B — human_review: review_type: "pricing_review", summary: "CPQ Review and Pricing summary:\n{{cpq_p3_pricing}}\n\nApprove to save the quote, or type 'no' to stop."
  STEP C — browser_act: "Click the Next button to go to page 4 Quote Summary. Wait for the page to change." [success_criteria: "Step 4 is now the ACTIVE/HIGHLIGHTED step in the wizard header"]

PAGE 4 — Quote Summary:
  STEP A — browser_extract: "FIRST: Check which step is ACTIVE in the wizard header. If not step 4, return {error: 'wrong step'}. If step 4, extract the final quote summary: quote ID, customer, all line items, total price."
  STEP B — browser_act: "Click Save or Submit to finalise and save the quote."

────────────────────────────────────────────────────────
OTHER WORKFLOWS
────────────────────────────────────────────────────────

JIRA / ATLASSIAN ("create jira", "jira sprint", "jira issue", "jira ticket", "confluence"):
  → Use MCP tools from AVAILABLE_MCP_TOOLS directly. NO browser_login, NO browser_act.

  ── FETCH-THEN-SELECT RULE (apply at EVERY step where a list of valid values exists) ──
  NEVER ask the user to type a value they cannot know from memory.
  Instead: fetch the available options from Jira first, then use request_selection so the
  user can pick from real, valid choices. This prevents invalid-value errors entirely.

  For project_key:
    Step 1 — call the MCP tool that lists accessible Jira projects
             (look for "jira_get_projects", "jira_list_projects", or similar in AVAILABLE_MCP_TOOLS).
    Step 2 — extract each project's key and name from the result.
    Step 3 — call request_selection with options like:
             [{"value": "DEMO", "label": "DEMO — Demo Project"},
              {"value": "DEV",  "label": "DEV  — Development"}]
             question: "Which Jira project should this be created in?"
    Step 4 — use {{select_project.selected}} (or the step id you chose) as project_key.

  For issue_type:
    • If the user's message contains "task", "bug", "story", "epic", "subtask" → hardcode it.
      "create a task" → issue_type="Task". Do NOT ask or fetch.
    • If ambiguous → call the MCP tool that lists issue types for the selected project
      (look for "jira_get_issue_types", "jira_get_project" which includes types, etc.),
      then use request_selection.
    • Default to "Task" if no listing tool exists.

  For summary:
    • Extract from the user's message if a clear description is present. Only ask if absent.

  → Required fields for jira_create_issue: project_key, issue_type, summary.
       • project_key — always obtained via fetch → request_selection, NOT ask_user.
         Must be ALL UPPERCASE; the selected "value" from request_selection is already correct.
       • issue_type — from message, or request_selection if ambiguous.
       • summary — from message, or ask_user if absent.
  → Use ONLY the parameter names listed in AVAILABLE_MCP_TOOLS. NEVER add extra params
     like "ctx", "priority", "auth_check", or anything not in the schema.
  → Each MCP tool call is ONE step. params must exactly match the tool's own parameter names.

  ── APPLY THE SAME FETCH-THEN-SELECT PATTERN for ALL Jira operations ──
  • Jira sprints       → fetch sprints for the board → request_selection
  • Jira assignees     → fetch team members / project members → request_selection
  • Jira boards        → fetch boards → request_selection
  • Status transitions → fetch available transitions for the issue → request_selection
  • Epics              → fetch epics in the project → request_selection
  Never ask the user to type an ID or key that Jira can return as a list.

GMAIL / EMAIL ("send email", "check email", "gmail"):
  → Use MCP tools from AVAILABLE_MCP_TOOLS directly. NO browser tools.
  → Collect recipient, subject, body with ask_user if not provided.
  → Then call the Gmail MCP tool.

────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────

EXAMPLE A — "create a quote for me" (no ID given → must ask first):
[
  {"id": "ask_opp_id", "type": "ask", "description": "Ask for Opportunity ID", "tool": "ask_user",
   "params": {"question": "What is the NetSuite Opportunity ID? (e.g. OP-12345)"},
   "success_criteria": "User provided ID"},
  {"id": "ns_login", "type": "login", "description": "Log in to NetSuite", "tool": "browser_login",
   "params": {"site": "netsuite"}, "success_criteria": "Logged in to NetSuite"},
  {"id": "ns_open", "type": "navigate", "description": "Open opportunity record", "tool": "browser_act",
   "params": {"instruction": "Search for 'OP-{{ask_opp_id.answer}}' in the NetSuite global search bar and press Enter. Wait for the search results page to load. If a popup or tooltip appears (e.g. 'Did you know?'), close it by clicking 'Continue' or 'Don't show this next time'. Then on the results page find the row where the TYPE column says 'Opportunity' (not PDF File, not Address Validation) and click its View link by targeting the anchor whose href contains 'opprtnty.nl'. Wait for the opportunity detail page to fully load."},
   "success_criteria": "URL contains opprtnty.nl"},
  {"id": "ns_extract", "type": "extract", "description": "Extract opportunity details", "tool": "browser_extract",
   "params": {"instruction": "Extract all visible fields from this opportunity record including: opportunity_id, buyer, bandwidth, term, address, contact_name, contact_email, contact_phone"},
   "success_criteria": "Extracted at least buyer and bandwidth"},
  {"id": "cpq_login", "type": "login", "description": "Log in to CPQ", "tool": "browser_login",
   "params": {"site": "cpq"}, "success_criteria": "Logged in to CPQ"},
  {"id": "cpq_new", "type": "navigate", "description": "Start new quote", "tool": "browser_navigate",
   "params": {"url": "https://dqe-cpq.cloudsmartz.com/quotes/new"},
   "success_criteria": "Page URL contains /quotes/new"},
  {"id": "cpq_p1_cust", "type": "act", "description": "Select customer from dropdown", "tool": "browser_act",
   "params": {"instruction": "Click the Customer dropdown (labelled 'Choose a customer...'). Wait for the list to open. Find and click the item matching '{{ns_extract.buyer}}'. After clicking it, click on the 'Create New Quote' page heading to close the dropdown. Do NOT press Escape."},
   "success_criteria": "Customer dropdown closed with customer selected"},
  {"id": "cpq_p1_opp_click", "type": "act", "description": "Click opportunity row to select it", "tool": "browser_act",
   "params": {"instruction": "In the 'Select Opportunity' panel on the RIGHT side, find and click the row that contains the opportunity ID '{{ask_opp_id.answer}}'. Click the row text directly. Do NOT press Escape."},
   "success_criteria": "Opportunity row selected"},
  {"id": "cpq_p1_opp", "type": "extract", "description": "Read opportunity panel", "tool": "browser_extract",
   "params": {"instruction": "Extract the text shown in the Select Opportunity panel on the right. Return: opportunity_shown (full text in the panel), customer_selected (value shown in Customer dropdown)."},
   "success_criteria": "Opportunity panel text extracted"},
  {"id": "cpq_p1_review", "type": "review", "description": "Confirm opportunity", "tool": "human_review",
   "params": {"review_type": "opportunity_confirm", "summary": "Customer '{{ns_extract.buyer}}' selected.\n\nOpportunity shown in CPQ: {{cpq_p1_opp.opportunity_shown}}\nExpected opportunity: OP-{{ask_opp_id.answer}}\n\nDoes this opportunity match what you want? Type 'yes' to continue to page 2, or 'no' to stop."},
   "success_criteria": "User confirmed opportunity"},
  {"id": "cpq_p1_next", "type": "act", "description": "Next to page 2", "tool": "browser_act",
   "params": {"instruction": "Click in the 'Select Opportunity' panel on the right side of the page (click on the opportunity row text already shown there) to move focus away from the Customer dropdown and close it. Do NOT press Escape — it may deselect the customer. Then click the 'Next' button at the bottom right of the page. Wait for the page to advance."},
   "success_criteria": "page text contains Product Configuration"},
  {"id": "cpq_p2_fields", "type": "extract", "description": "Extract page 2 product cards and fields", "tool": "browser_extract",
   "params": {"instruction": "Wait 2 seconds. Look at the wizard step indicator at the top — which step number is currently ACTIVE (highlighted/bold/coloured)? If step 1 is still active, click the Next button at the bottom right and wait 3 seconds. Then extract: active_step (number), product_cards (array of clickable card/tile names), form_fields (array of {field_name, current_value, options})."},
   "success_criteria": "product cards or form fields extracted"},
  {"id": "cpq_p2_confirm", "type": "ask", "description": "Ask user what to do on page 2", "tool": "ask_user",
   "params": {"question": "CPQ Product Configuration is ready.\nNetSuite: Bandwidth={{ns_extract.bandwidth}}, Term={{ns_extract.term}}.\n\nProduct cards available: {{cpq_p2_fields.product_cards}}\n\nWhich product should I select?\n<<<1. Internet>>>\n<<<2. Dark Fiber>>>\n<<<3. Other (type product name)>>>"},
   "success_criteria": "User selected a product"},
  {"id": "cpq_p2_fill", "type": "act", "description": "Fill page 2 per user instruction", "tool": "browser_act",
   "params": {"instruction": "You are on CPQ page 2 (Product Configuration). DO NOT click wizard step labels at the top of the page such as '1 Customer & Opportunity' or '2 Product Configuration' — those navigate away and will break the flow. Look at the main CONTENT area of the page (not the header). If the page shows product card TILES (large clickable cards with service names), find and click the tile whose text matches '{{cpq_p2_confirm.answer}}'. If instead the page shows form DROPDOWNS or SELECT fields (e.g. 'Service Type', 'Access Type', 'Technology', 'Product'), click the most relevant dropdown to open it, then click the option that best matches '{{cpq_p2_confirm.answer}}'. Never type into a dropdown. After selecting, wait 2 seconds for the form to update."},
   "success_criteria": "Page 2 product selected"},
  {"id": "cpq_p2_addr_scan", "type": "extract", "description": "Scan address form after product selected", "tool": "browser_extract",
   "params": {"instruction": "A service address or location form has appeared after clicking the product card. Extract ONLY: has_locate_me (true/false), street_prefilled (current value of street field or empty string), city_prefilled (current city value or empty string)."},
   "success_criteria": "Address form scanned"},
  {"id": "cpq_p2_addr_ask", "type": "ask", "description": "Ask user how to fill address", "tool": "ask_user",
   "params": {"question": "Service address form is open.\nNetSuite address: {{ns_extract.address}}\n\nHow should I fill it?\n<<<1. Click Locate Me (auto-fill)>>>\n<<<2. Fill manually with NetSuite address>>>"},
   "success_criteria": "User chose address fill method"},
  {"id": "cpq_p2_addr_fill", "type": "act", "description": "Fill address per user instruction", "tool": "browser_act",
   "params": {"instruction": "If the user chose 'Click Locate Me': click the 'Locate Me' button and wait for address fields to populate automatically. If the user chose 'Fill manually': fill Street Address, City, State and Postcode fields using the NetSuite address values: {{ns_extract.address}}."},
   "success_criteria": "Address fields filled"},
  {"id": "cpq_p2_address", "type": "extract", "description": "Extract filled address", "tool": "browser_extract",
   "params": {"instruction": "Extract the address now showing in the service location form fields: street, city, state, zip/postcode. Return as: address_street, address_city, address_state, address_zip."},
   "success_criteria": "Address extracted"},
  {"id": "cpq_p2_addr_review", "type": "review", "description": "Confirm service address", "tool": "human_review",
   "params": {"review_type": "address_confirm", "summary": "Service address in CPQ:\n  Street: {{cpq_p2_address.address_street}}\n  City:   {{cpq_p2_address.address_city}}\n  State:  {{cpq_p2_address.address_state}}\n  Zip:    {{cpq_p2_address.address_zip}}\n\nNetSuite address: {{ns_extract.address}}\n\nCorrect? Type 'yes' to continue or provide corrections."},
   "success_criteria": "User confirmed address"},
  {"id": "cpq_p2_next", "type": "act", "description": "Next to page 3", "tool": "browser_act",
   "params": {"instruction": "Press Escape to close any open panel. Click the Next button to go to page 3 Review and Pricing. Wait for the page to change."},
   "success_criteria": "Step 3 is now the active highlighted step in the wizard header"},
  {"id": "cpq_p3_pricing", "type": "extract", "description": "Extract pricing", "tool": "browser_extract",
   "params": {"instruction": "IMPORTANT: First verify step 3 (Review and Pricing) is the active step in the wizard header. If step 2 is still active, return {error: 'still on step 2'}. If step 3 is active, extract ALL visible fields, line items, prices and totals on this page."},
   "success_criteria": "Pricing details extracted from step 3"},
  {"id": "cpq_p3_review", "type": "review", "description": "Approve pricing", "tool": "human_review",
   "params": {"review_type": "pricing_review", "summary": "CPQ Review and Pricing:\n{{cpq_p3_pricing}}\n\nApprove to proceed to Quote Summary, or type 'no' to stop."},
   "success_criteria": "User approved pricing"},
  {"id": "cpq_p3_next", "type": "act", "description": "Next to page 4", "tool": "browser_act",
   "params": {"instruction": "Click the Next button to go to page 4 Quote Summary. Wait for the page to change."},
   "success_criteria": "Step 4 is now the active highlighted step in the wizard header"},
  {"id": "cpq_p4_extract", "type": "extract", "description": "Extract quote summary", "tool": "browser_extract",
   "params": {"instruction": "IMPORTANT: First verify step 4 (Quote Summary) is the active step. If not on step 4, return {error: 'wrong step'}. If step 4 is active, extract the final quote summary: quote ID, customer, all line items, total price."},
   "success_criteria": "Quote summary extracted from step 4"},
  {"id": "cpq_p4_save", "type": "act", "description": "Save quote", "tool": "browser_act",
   "params": {"instruction": "Click Save or Submit to finalise and save the quote."},
   "success_criteria": "Quote saved and quote ID visible"}
]

EXAMPLE B — "create quote for opportunity 54321" (ID given in task → no ask_user for ID):
[
  {"id": "ns_login", "tool": "browser_login", "type": "login", "description": "Log in to NetSuite",
   "params": {"site": "netsuite"}, "success_criteria": "Logged in to NetSuite"},
  {"id": "ns_open", "tool": "browser_act", "type": "navigate", "description": "Open opportunity record",
   "params": {"instruction": "Search for 'OP-54321' in the NetSuite global search bar and press Enter. Wait for the search results page. If a popup or tooltip appears (e.g. 'Did you know?'), close it by clicking 'Continue' or 'Don't show this next time'. Then find the row where TYPE says 'Opportunity' and click its View link by targeting the anchor whose href contains 'opprtnty.nl'. Do NOT click any other View link. Wait for the opportunity detail page to fully load."},
   "success_criteria": "URL contains opprtnty.nl"},
  {"id": "ns_extract", "tool": "browser_extract", "type": "extract", "description": "Extract opportunity details",
   "params": {"instruction": "Extract all visible fields: opportunity_id, buyer, bandwidth, term, address, contact_name, contact_email, contact_phone"},
   "success_criteria": "Data extracted"},
  {"id": "cpq_login", "tool": "browser_login", "type": "login", "description": "Log in to CPQ",
   "params": {"site": "cpq"}, "success_criteria": "Logged in to CPQ"},
  {"id": "cpq_new", "tool": "browser_navigate", "type": "navigate", "description": "Start new quote",
   "params": {"url": "https://dqe-cpq.cloudsmartz.com/quotes/new"},
   "success_criteria": "Page URL contains /quotes/new"},
  {"id": "cpq_p1_select", "tool": "browser_act", "type": "act", "description": "Select customer",
   "params": {"instruction": "Click the Customer dropdown. When the list opens, click '{{ns_extract.buyer}}'. Press Escape to close the dropdown. Then click on the opportunity shown in the Select Opportunity panel on the right to select it."},
   "success_criteria": "Customer chosen from dropdown and dropdown is closed"},
  {"id": "cpq_p1_opp", "tool": "browser_extract", "type": "extract", "description": "Read opportunity panel",
   "params": {"instruction": "Extract the text shown in the Select Opportunity panel on the right. Return: opportunity_shown (full text), customer_selected (customer dropdown value)."},
   "success_criteria": "Opportunity panel text extracted"},
  {"id": "cpq_p1_review", "tool": "human_review", "type": "review", "description": "Confirm opportunity",
   "params": {"review_type": "opportunity_confirm", "summary": "Customer '{{ns_extract.buyer}}' selected.\n\nOpportunity shown in CPQ: {{cpq_p1_opp.opportunity_shown}}\nExpected: OP-54321\n\nDoes this match? Type 'yes' to continue, or 'no' to stop."},
   "success_criteria": "Opportunity confirmed"},
  {"id": "cpq_p1_next", "tool": "browser_act", "type": "act", "description": "Next to page 2",
   "params": {"instruction": "Click in the 'Select Opportunity' panel on the right side of the page (click on the opportunity row text already shown there) to move focus away from the Customer dropdown and close it. Do NOT press Escape. Then click the 'Next' button at the bottom right of the page. Wait for the page to advance."},
   "success_criteria": "page text contains Product Configuration"},
  {"id": "cpq_p2_fields", "tool": "browser_extract", "type": "extract", "description": "Extract page 2 fields",
   "params": {"instruction": "Extract ALL fields. For each dropdown: field_name, current_value, available_options (every option). For text inputs: field_name, current_value."},
   "success_criteria": "All page 2 fields and options extracted"},
  {"id": "cpq_p2_confirm", "tool": "ask_user", "type": "ask", "description": "Confirm page 2 values",
   "params": {"question": "CPQ Product Configuration fields and options:\n{{cpq_p2_fields}}\n\nNetSuite: Bandwidth={{ns_extract.bandwidth}}, Term={{ns_extract.term}}.\n\nType 'ok' if NetSuite values match available options, or specify corrections."},
   "success_criteria": "Values confirmed"},
  {"id": "cpq_p2_fill", "tool": "browser_act", "type": "act", "description": "Fill page 2",
   "params": {"instruction": "You are on CPQ page 2 (Product Configuration). DO NOT click wizard step labels at the top like '1 Customer & Opportunity' or '2 Product Configuration'. Look at the main content area. The user wants: {{cpq_p2_confirm.answer}}. If you see product card TILES, click the tile matching the answer. If you see form DROPDOWNS (Service Type, Access Type, Technology, Product), click the relevant dropdown to open it and select the best matching option. Never type into a dropdown. After selecting, wait 2 seconds."},
   "success_criteria": "Page 2 product selected"},
  {"id": "cpq_p2_next", "tool": "browser_act", "type": "act", "description": "Next to page 3",
   "params": {"instruction": "Press Escape to close any open dropdown. Click Next to go to page 3 Review and Pricing."},
   "success_criteria": "Page text contains Review and Pricing"},
  {"id": "cpq_p3_pricing", "tool": "browser_extract", "type": "extract", "description": "Extract pricing",
   "params": {"instruction": "Extract ALL pricing, line items, and totals on this Review and Pricing page."},
   "success_criteria": "Pricing extracted"},
  {"id": "cpq_p3_review", "tool": "human_review", "type": "review", "description": "Approve pricing",
   "params": {"review_type": "pricing_review", "summary": "CPQ Review and Pricing:\n{{cpq_p3_pricing}}\n\nApprove to proceed to Quote Summary, or type 'no' to stop."},
   "success_criteria": "Pricing approved"},
  {"id": "cpq_p3_next", "tool": "browser_act", "type": "act", "description": "Next to page 4",
   "params": {"instruction": "Click Next to go to Quote Summary."},
   "success_criteria": "Page text contains Quote Summary"},
  {"id": "cpq_p4_extract", "tool": "browser_extract", "type": "extract", "description": "Extract quote summary",
   "params": {"instruction": "Extract quote summary: quote ID, customer, line items, total price."},
   "success_criteria": "Quote summary extracted"},
  {"id": "cpq_p4_save", "tool": "browser_act", "type": "act", "description": "Save quote",
   "params": {"instruction": "Click Save or Submit to finalise the quote."},
   "success_criteria": "Quote saved"}
]

EXAMPLE C — simple chat:
[{"id": "respond", "type": "respond", "description": "Respond to user", "tool": "direct_response",
  "params": {"message": "Hello! I can create quotes, search Jira, or automate any browser task. What would you like me to do?"},
  "success_criteria": "Response delivered"}]
"""


def _salvage_partial_plan(content: str) -> list:
    """Extract complete JSON objects from a truncated/malformed plan array.
    Uses brace-counting to correctly handle nested objects like params: {...}.
    """
    steps = []
    i = 0
    n = len(content)
    while i < n:
        # Find start of next top-level object
        if content[i] != "{":
            i += 1
            continue
        # Walk forward counting braces, respecting strings
        depth = 0
        in_str = False
        esc = False
        start = i
        j = i
        while j < n:
            ch = content[j]
            if esc:
                esc = False
            elif ch == "\\" and in_str:
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        # Complete object found
                        try:
                            obj = json.loads(content[start : j + 1])
                            if isinstance(obj, dict) and "tool" in obj:
                                if not isinstance(obj.get("success_criteria"), str):
                                    obj["success_criteria"] = "Step completed"
                                steps.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            break  # truncation — stop here
    return steps


def _fix_json_control_chars(s: str) -> str:
    """Escape bare control characters (newline, tab, etc.) that appear inside
    JSON string literals — a common LLM output bug.  Structural whitespace
    between JSON tokens is left untouched."""
    result: list[str] = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        elif in_string and ord(ch) < 0x20:
            result.append(f"\\u{ord(ch):04x}")
        else:
            result.append(ch)
    return "".join(result)




async def planner_node(state: AgentState, _tool_filter: list[str] | None = None) -> dict:
    """Generate a plan from the task. Runs ONCE at the start."""
    from dqe_agent.llm import get_planner_llm

    task = state.get("task", "")
    if not task:
        # Extract task from last user message
        messages = state.get("messages", [])
        for m in reversed(messages):
            if hasattr(m, "type") and m.type == "human":
                task = m.content
                break
            if isinstance(m, tuple) and m[0] == "user":
                task = m[1]
                break

    if not task:
        return {"status": "failed", "error": "No task provided to planner"}

    logger.info("[PLANNER] Planning task: %s", task[:100])

    llm = get_planner_llm()

    # Inject configured sites so the planner knows what browser_login(site=?) accepts
    from dqe_agent.config import settings

    configured_sites = list(settings.sites.keys())
    site_names = (
        " | ".join(f'"{s}"' for s in configured_sites) if configured_sites else "(none configured)"
    )

    from datetime import date as _date

    _today = _date.today().isoformat()  # e.g. 2026-04-18

    context_parts = [f"TASK: {task}"]
    context_parts.append(
        f"TODAY'S DATE: {_today}  "
        "Use this to resolve ALL natural-language date references the user gives you. "
        "Examples: 'today' → {_today}, 'yesterday' → one day before, 'this week' → startOfWeek(), "
        "'last month' → startOfMonth(-1)/endOfMonth(-1), 'April 20' → the nearest April 20, "
        "'all time' / 'ever' → omit date filter entirely, 'custom range' → collect via form. "
        "NEVER auto-fill a date the user did NOT mention — ask_user if no date reference is given."
    )

    context_parts.append(f"CONFIGURED SITES (use these names with browser_login): [{site_names}]")

    # Inject MCP/non-browser tool names so planner knows what's available
    from dqe_agent.tools import list_tool_names
    from dqe_agent.config import settings as _settings

    _browser_tool_set = {
        "browser_login",
        "browser_navigate",
        "browser_act",
        "browser_extract",
        "browser_click",
        "browser_type",
        "browser_wait",
        "browser_snapshot",
        "ask_user",
        "human_review",
        "ask_user_choice",
        "direct_response",
        "request_selection",
    }
    if _settings.disable_browser_tools:
        context_parts.append(
            "IMPORTANT: Browser tools are DISABLED for this session. "
            "Do NOT plan any browser_* steps. "
            "Use ONLY direct_response, ask_user, human_review, request_selection, "
            "and the MCP tools listed in AVAILABLE_MCP_TOOLS."
        )
    mcp_tools = [t for t in list_tool_names() if t not in _browser_tool_set]

    # Apply agent tool filter if set
    if _tool_filter is None and state.get("agent_id"):
        try:
            from dqe_agent.agents import get_agent as _get_agent_cfg
            _cfg = _get_agent_cfg(state["agent_id"])
            if _cfg.tools is not None:
                _tool_filter = _cfg.tools
        except KeyError:
            pass
    if _tool_filter is not None:
        mcp_tools = [t for t in mcp_tools if t in _tool_filter]

    if mcp_tools:
        # ── Fast path: serve cached tool description block ───────────────────
        _tool_key = frozenset(mcp_tools)
        cached_mcp_block = _MCP_DESC_CACHE.get(_tool_key)
        if cached_mcp_block:
            context_parts.append(cached_mcp_block)
        else:
            # ── Slow path: build schema for all MCP tools (runs once) ────────
            _t0 = time.monotonic()
            from dqe_agent.tools import get_tool as _get_tool

            mcp_lines = []
            for t in mcp_tools:
                try:
                    tool_obj = _get_tool(t)
                    desc = tool_obj.description or ""
                    param_str = ""
                    try:
                        schema = (
                            tool_obj.args_schema.model_json_schema()
                            if hasattr(tool_obj, "args_schema") and tool_obj.args_schema
                            else {}
                        )
                        props = schema.get("properties", {})
                        required = set(schema.get("required", []))
                        if props:
                            parts = []
                            for p, info in props.items():
                                if p in ("kwargs", "ctx"):
                                    continue
                                req = "*" if p in required else ""
                                ptype = info.get(
                                    "type", info.get("anyOf", [{}])[0].get("type", "any")
                                )
                                parts.append(f"{p}{req}: {ptype}")
                            param_str = f"({', '.join(parts)})"
                    except Exception:
                        pass
                    mcp_lines.append(f"  - {t}{param_str}: {desc[:100]}")
                except Exception:
                    mcp_lines.append(f"  - {t}")
            mcp_block = (
                "AVAILABLE_MCP_TOOLS — call these directly as step tools, no browser needed.\n"
                "Parameters marked * are required. Use ONLY the listed param names — never invent extras like 'ctx', 'priority', 'auth_check'.\n"
                "{{step_id.field}} templates reference PREVIOUS STEP RESULTS only, never AVAILABLE_MCP_TOOLS itself.\n"
                + "\n".join(mcp_lines)
            )
            _MCP_DESC_CACHE[_tool_key] = mcp_block
            logger.info(
                "[PLANNER] MCP tool block built in %.0fms and cached (%d tools)",
                (time.monotonic() - _t0) * 1000,
                len(mcp_tools),
            )
            context_parts.append(mcp_block)
    else:
        context_parts.append("AVAILABLE_MCP_TOOLS: (none loaded yet)")

    # ── Pre-fetch structured options so the planner doesn't guess ───────────────
    # For any domain where the system can return a finite list of valid values,
    # fetch them here and inject as ready-made request_selection options.
    # The planner then just uses request_selection with the pre-supplied data.
    await _prefetch_selection_options(task, context_parts)

    # Inject agent's own notes from past diagnoses
    from dqe_agent.agent.notes import format_notes_for_prompt

    notes_text = format_notes_for_prompt()
    if notes_text:
        context_parts.append(notes_text)

    # Inject known user credentials from config so the planner never re-asks for them
    known_data: dict = {}
    if _settings.user_google_email:
        known_data["user_google_email"] = _settings.user_google_email
    if _settings.jira_username:
        known_data["jira_username"] = _settings.jira_username
    flow_data = state.get("flow_data", {})
    merged_data = {**known_data, **flow_data}
    if merged_data:
        # Remove sensitive/PII-like fields before injecting into planner context
        redacted = dict(merged_data)
        redacted.pop("user_google_email", None)
        # Separate out cross-turn context keys for clearer injection
        _pagination = redacted.pop("_last_list_result", None)
        _last_action = redacted.pop("_last_action", None)
        if _last_action and isinstance(_last_action, dict):
            _prev_task = _last_action.get('task', '')
            _prev_tools = _last_action.get('tools', [])
            context_parts.append(
                f"Context from prior turn: previous request was about [{_prev_task[:80]}], "
                f"tools used: {_prev_tools}. "
                f"Short follow-up messages like 'in flag?' or 'and for X?' refer to repeating "
                f"the same operation for the newly mentioned target."
            )
        if redacted:
            context_parts.append(
                f"AVAILABLE DATA (already known — do NOT ask the user for these): {json.dumps(redacted, default=str)[:2000]}"
            )
        if _pagination and isinstance(_pagination, dict):
            _shown = _pagination.get("shown", 0)
            _total = _pagination.get("total")
            _next = _pagination.get("next_start_at", _shown)
            _has_more = _pagination.get("has_more", False)
            _tool = _pagination.get("tool", "")
            _query = _pagination.get("query", "")
            _sprint_id = _pagination.get("sprint_id", "")
            _limit = _pagination.get("limit", 50)
            _next_page_token = _pagination.get("page_token_for_next", "") or _pagination.get("next_page_token", "")
            _total_str = f" of {_total}" if _total is not None else ""
            _more_str = f" ({_total - _next} more available)" if (_total is not None and _has_more) else (" (more available — cursor pagination)" if _has_more else " (no more results)")
            if _next_page_token:
                _next_page_note = (
                    f"  ⚑ Jira returned a page_token — use it for the next page (more reliable than start_at).\n"
                    f"  Next page: jira_search(jql=\"{_query}\", page_token=\"{_next_page_token}\", limit={_limit})\n"
                    f"  Do NOT use start_at when page_token is available — start_at is ignored by Jira Cloud cursor pagination.\n"
                )
            else:
                _next_page_note = f"  To show NEXT page: use start_at={_next} with limit={_limit}\n"
            context_parts.append(
                f"PAGINATION CONTEXT — last list result:\n"
                f"  Tool used: {_tool}\n"
                f"  Query/sprint: {_query or _sprint_id}\n"
                f"  Shown so far: {_shown}{_total_str} items{_more_str}\n"
                f"{_next_page_note}"
                f"  To show from beginning: use start_at=0 (omit page_token)\n"
                f"\n"
                f"  If the user says 'next', 'more', 'show more', 'continue' etc. → use the page_token above if present.\n"
                f"  If user says 'previous' or 'go back' → use start_at={max(0, _next - _limit * 2)} (no page_token).\n"
                f"  For sprint issues page 2+: use jira_search with jql='sprint = {_sprint_id}' + page_token or start_at."
            )

    user_msg = "\n\n".join(context_parts)

    # Always give the LLM full tool context — let it decide what's needed.
    system_prompt = MASTER_SYSTEM_PROMPT
    if not _settings.disable_browser_tools:
        system_prompt = MASTER_SYSTEM_PROMPT + BROWSER_SYSTEM_PROMPT

    start = time.time()

    # ── Build multi-turn history so the LLM is aware of prior conversation ──
    # state["messages"] accumulates all turns via LangGraph add_messages reducer.
    # We include the last N exchanges (excluding the very latest user msg which
    # is already encoded in the TASK context_part above).
    _MAX_HISTORY_TURNS = 10  # configurable — covers ~5 Q&A exchanges
    raw_messages = state.get("messages", [])
    history_msgs: list = []

    # Collect prior turns (skip the last user message — already in TASK)
    prior = raw_messages[:-1] if raw_messages else []
    # Take last _MAX_HISTORY_TURNS messages for context window efficiency
    prior = prior[-_MAX_HISTORY_TURNS:]

    for m in prior:
        if hasattr(m, "type"):
            if m.type == "human":
                history_msgs.append(HumanMessage(content=str(m.content)[:800]))
            elif m.type == "ai":
                # Summarise AI planning output as brief assistant context
                content = str(m.content)
                # Trim very long plan confirmations
                history_msgs.append(AIMessage(content=content[:400]))
        elif isinstance(m, tuple) and len(m) == 2:
            role, content = m
            if role == "user":
                history_msgs.append(HumanMessage(content=str(content)[:800]))
            elif role == "assistant":
                history_msgs.append(AIMessage(content=str(content)[:400]))

    if history_msgs:
        logger.debug(
            "[PLANNER] Injecting %d prior context messages into LLM call", len(history_msgs)
        )

    # Final message list: system → history → current task
    llm_messages = (
        [SystemMessage(content=system_prompt)] + history_msgs + [HumanMessage(content=user_msg)]
    )

    try:
        response = await llm.ainvoke(llm_messages)
    except Exception as _llm_exc:
        _exc_str = str(_llm_exc)
        if "content_filter" in _exc_str or "ResponsibleAIPolicyViolation" in _exc_str:
            logger.warning("[PLANNER] Azure content filter triggered: %s", _exc_str[:300])
            return {
                "status": "failed",
                "error": "content_filter",
                "messages": [
                    AIMessage(
                        content=(
                            "I wasn't able to process that request — the content moderation "
                            "policy blocked it. Please rephrase your request and try again."
                        )
                    )
                ],
            }
        raise

    duration = (time.time() - start) * 1000

    # Parse plan JSON from response
    content = response.content.strip()
    # Strip markdown code fences if present (handle leading whitespace before ```)
    if "```" in content:
        content = content[content.index("```") :]
        content = content.split("\n", 1)[-1]
        if "```" in content:
            content = content[: content.rindex("```")]
        content = content.strip()

    # Fix bare control characters inside JSON strings (common LLM bug)
    content = _fix_json_control_chars(content)

    try:
        plan = json.loads(content)
        if not isinstance(plan, list):
            plan = [plan]
    except json.JSONDecodeError as exc:
        logger.warning("[PLANNER] JSON parse failed (%s) — attempting repair", exc)
        # Repair attempt: the LLM sometimes puts unquoted prose in string fields.
        # Try to salvage complete step objects before the first malformed one.
        plan = _salvage_partial_plan(content)
        if not plan:
            logger.error("[PLANNER] Could not salvage plan — asking LLM to retry")
            # Retry once with a stricter prompt
            repair_response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=user_msg
                        + "\n\nImportant: Output ONLY valid JSON array. No markdown fences, no explanation. Every string value must use \\n for newlines, never bare newlines."
                    ),
                ]
            )
            repair_content = repair_response.content.strip()
            if "```" in repair_content:
                repair_content = repair_content[repair_content.index("```") :]
                repair_content = repair_content.split("\n", 1)[-1]
                if "```" in repair_content:
                    repair_content = repair_content[: repair_content.rindex("```")]
                repair_content = repair_content.strip()
            repair_content = _fix_json_control_chars(repair_content)
            try:
                plan = json.loads(repair_content)
                if not isinstance(plan, list):
                    plan = [plan]
            except json.JSONDecodeError as exc2:
                logger.error("[PLANNER] Retry also failed: %s", exc2)
                return {
                    "status": "failed",
                    "error": f"Planner produced invalid JSON: {exc2}",
                    "messages": [
                        AIMessage(
                            content="I couldn't generate a valid plan. Please try rephrasing your request."
                        )
                    ],
                }

    # Post-process plans: if a meeting-day get_events was produced but no ask_date
    # step exists and the user's task did not explicitly request "today/asap/now",
    # insert an `ask_date` step before the day-scoped get_events and template the
    # get_events to use the collected date. This prevents silently assuming today.
    try:
      import re as _re_post
      _has_ask = any(isinstance(s, dict) and s.get("id") == "ask_date" for s in plan)
      # Find first get_events step
      _ge_idx = None
      for _i, _s in enumerate(plan):
        if isinstance(_s, dict) and _s.get("tool") == "get_events":
          _ge_idx = _i
          break
      if _ge_idx is not None and not _has_ask:
        # Only inject ask_date when the user's task doesn't already imply today/asap
        if not _re_post.search(r"\b(today|asap|now|right now|this (?:morning|afternoon|evening)|tonight)\b", task, _re_post.I):
          _ge_step = plan[_ge_idx]
          _params = _ge_step.get("params", {}) or {}
          _time_min = str(_params.get("time_min", ""))
          # If get_events was requested for a concrete date (e.g. 2026-04-24T00...) or <TODAY>,
          # replace with templated date and insert ask_date before it.
          if ("<TODAY>" in _time_min) or _re_post.match(r"\d{4}-\d{2}-\d{2}", _time_min):
            ask_step = {
              "id": "ask_date",
              "tool": "ask_user",
              "params": {"question": "Which date would you like to schedule the meeting? (e.g. today, tomorrow, April 25)"},
              "success_criteria": "Date received",
            }
            # Update get_events to use the collected date
            _params["time_min"] = "{{ask_date.answer}}T00:00:00Z"
            _params["time_max"] = "{{ask_date.answer}}T23:59:59Z"
            _ge_step["params"] = _params
            plan.insert(_ge_idx, ask_step)
    except Exception:
      # If meeting post-processing fails, continue with original plan
      pass

    # Post-process worklog plans: ensure member disambiguation and date collection
    try:
      for _i in range(len(plan)):
        _s = plan[_i]
        if not isinstance(_s, dict):
          continue
        if _s.get("tool") != "jira_get_worklogs_by_date_range":
          continue
        _params = _s.get("params", {}) or {}
        # If member_name present as a plain string (not template) and not a special token,
        # insert jira_get_assignable_users + request_selection steps to let user pick the exact member.
        mn = _params.get("member_name", "")
        if isinstance(mn, str) and mn and not ("{{" in mn) and mn not in ("__me__", "__all__"):
          # Create unique ids based on step index
          fetch_id = f"fetch_members_{_i}"
          sel_id = f"sel_member_{_i}"
          proj_key_val = _params.get("project_key", "") or ""
          fetch_step = {
            "id": fetch_id,
            "tool": "jira_get_assignable_users",
            "params": {"project_key": proj_key_val},
            "success_criteria": "Members fetched",
          }
          sel_step = {
            "id": sel_id,
            "tool": "request_selection",
            "params": {"question": f"Which team member? (you said: {mn})","options": f"{{{{{fetch_id}}}}}", "multi_select": False},
            "success_criteria": "Member selected",
          }
          # Replace member_name in the worklogs step with the selection reference
          _params["member_name"] = f"{{{{{sel_id}.selected}}}}"
          _s["params"] = _params
          # Insert fetch and selection steps before the current worklog step
          plan.insert(_i, fetch_step)
          plan.insert(_i + 1, sel_step)
          # Advance index to skip over newly inserted steps
          # (loop will continue and handle further steps)
        # Ensure date range is collected if start_date/end_date are missing or defaulted to today
        sd = _params.get("start_date", "")
        ed = _params.get("end_date", "")
        if (not sd or not ed or (isinstance(sd, str) and sd.strip() == _today)):
          # Insert a date-range collection form before this worklog step
          collect_id = f"collect_dates_{_i}"
          collect_step = {
            "id": collect_id,
            "tool": "request_form",
            "params": {
              "title": "Date Range",
              "fields": [
                {"id": "start_date", "label": "Start Date", "type": "text", "required": True, "placeholder": "YYYY-MM-DD"},
                {"id": "end_date", "label": "End Date", "type": "text", "required": True, "placeholder": "YYYY-MM-DD"},
              ],
            },
            "success_criteria": "Dates collected",
          }
          _params["start_date"] = f"{{{{{collect_id}.start_date}}}}"
          _params["end_date"] = f"{{{{{collect_id}.end_date}}}}"
          _s["params"] = _params
          plan.insert(_i, collect_step)
    except Exception:
      # If worklog post-processing fails, prefer to continue with original plan
      pass

    # Post-process project-member additions: ensure role_id is numeric by
    # fetching project roles and asking the user to pick if planner supplied
    # a role name or otherwise non-numeric value.
    try:
      for _i in range(len(plan)):
        _s = plan[_i]
        if not isinstance(_s, dict):
          continue
        if _s.get("tool") != "jira_add_project_member":
          continue
        _params = _s.get("params", {}) or {}
        role_val = _params.get("role_id", "")
        proj_key_val = _params.get("project_key", "") or ""
        # If role_id is present but not numeric, insert fetch+selection steps
        if role_val and not str(role_val).strip().isdigit():
          fetch_id = f"get_roles_{_i}"
          sel_id = f"sel_role_{_i}"
          fetch_step = {
            "id": fetch_id,
            "tool": "jira_get_project_roles",
            "params": {"project_key": proj_key_val},
            "success_criteria": "Roles fetched",
          }
          sel_step = {
            "id": sel_id,
            "tool": "request_selection",
            "params": {"question": f"Which project role should the user have? (you said: {role_val})", "options": f"{{{{{fetch_id}}}}}", "multi_select": False},
            "success_criteria": "Role selected",
          }
          _params["role_id"] = f"{{{{{sel_id}.selected}}}}"
          _s["params"] = _params
          plan.insert(_i, fetch_step)
          plan.insert(_i + 1, sel_step)
    except Exception:
      pass

    logger.info("[PLANNER] Generated %d-step plan in %.0fms", len(plan), duration)
    for i, step in enumerate(plan):
        logger.info("  Step %d: [%s] %s", i, step.get("tool", "?"), step.get("description", ""))

    # Trace the planner LLM call
    from dqe_agent.observability import trace_llm_call

    trace_llm_call(
        model="planner",
        role="planner",
        cost_usd=COST_PER_CALL["planner"],
        session_id=state.get("session_id", ""),
    )

    cost = COST_PER_CALL["planner"]
    # Build new flow_data: reset stale state but carry forward cross-turn context keys.
    _prev_flow = state.get("flow_data", {})
    _new_flow_data = {k: v for k, v in _prev_flow.items() if k in ("_last_list_result", "_last_action")}
    _action_tools = [s.get("tool", "") for s in plan if s.get("tool") != "direct_response"]
    _new_flow_data["_last_action"] = {"task": task[:120], "tools": _action_tools[:5]}
    return {
        "plan": plan,
        "current_step_index": 0,
        "status": "executing",
        "retry_count": 0,
        "replan_count": state.get("replan_count", 0),
        "steps_taken": state.get("steps_taken", 0),
        "estimated_cost": state.get("estimated_cost", 0.0) + cost,
        "step_results": [],
        "flow_data": _new_flow_data,
        "messages": [
            AIMessage(content=f"Plan created with {len(plan)} steps. Starting execution...")
        ],
    }
