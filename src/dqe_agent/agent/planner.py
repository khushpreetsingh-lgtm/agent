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

# ── Keywords that signal Jira involvement ────────────────────────────────────
_JIRA_KEYWORDS = {
    "jira",
    "ticket",
    "issue",
    "sprint",
    "board",
    "epic",
    "story",
    "backlog",
    # also trigger for common creation verbs when Jira context is implied
    "task",
    "bug",
    "subtask",
    "create ticket",
    "create issue",
    "create task",
    "create sprint",
    "create story",
    "create epic",
}

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
    """For tasks involving connected systems, pre-fetch valid option lists and inject
    them into the planner context as ready-made request_selection data.

    This avoids the LLM having to guess tool names or the user having to type
    raw IDs. The planner receives the actual options and just wires request_selection.
    """
    task_lower = task.lower()

    # ── Jira: fetch projects if task is Jira-related ─────────────────────────
    # Strong keywords → always trigger Jira prefetch
    _JIRA_STRONG = {
        "jira",
        "ticket",
        "sprint",
        "board",
        "epic",
        "backlog",
        "create ticket",
        "create issue",
        "create sprint",
        "create story",
        "create epic",
        "jira issue",
    }
    # Weak keywords → trigger only if a strong keyword is also present
    _JIRA_WEAK = {"issue", "task", "bug", "subtask", "story"}

    _has_strong = any(kw in task_lower for kw in _JIRA_STRONG)
    _has_weak = any(kw in task_lower for kw in _JIRA_WEAK)
    if _has_strong or (_has_weak and _has_strong):
        await _prefetch_jira_projects(context_parts)

    # NOTE: Boards are NOT pre-fetched globally because jira_get_agile_boards
    # without a project filter returns workspace-wide boards which may miss the
    # user's target project boards.  Instead the plan selects a project first,
    # then fetches boards for that specific project as a plan step.  The
    # executor's pre-resolution logic converts the step result into options.


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
        "JIRA_PROJECTS_PREFETCHED — use these directly in request_selection, "
        "do NOT call any project listing tool again:\n"
        f"{options_json}\n\n"
        "For the project selection step use:\n"
        '  tool: "request_selection"\n'
        '  params: {"question": "Which Jira project should this be created in?", '
        '"options": <copy the array above exactly>, "multi_select": false}\n'
        "Reference the result as {{select_project.selected}} in jira_create_issue params."
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
        for item in data:
            if not isinstance(item, dict):
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
                logger.debug("[PLANNER] Skipping project item with no valid key: %s", item)
                continue

            name = item.get("name") or item.get("projectName") or item.get("displayName") or key
            # Ensure value is JUST the key, label is for display
            options.append({"value": key, "label": f"{key} — {name}"})

    elif isinstance(data, dict):
        for container_key in ("projects", "values", "items", "data", "results"):
            if container_key in data and isinstance(data[container_key], list):
                return _parse_jira_projects(data[container_key], "")

    # ── String fallback: scan for KEY — Name patterns ────────────────────────
    if not options and result_str:
        for m in _re.finditer(r"\b([A-Z][A-Z0-9]{1,9})\b", result_str):
            key = m.group(1)
            if key not in {o["value"] for o in options}:
                options.append({"value": key, "label": key})

    return options[:20]


_BROWSER_TASK_KEYWORDS = {
    "netsuite",
    "cpq",
    "quote",
    "browser",
    "website",
    "web",
    "page",
    "login",
    "navigate",
    "click",
    "fill",
    "scroll",
    "form",
    "screenshot",
    "http",
    "https",
    "url",
}


def _needs_browser(task: str) -> bool:
    """Return True if this task likely requires browser tools."""
    t = task.lower()
    return any(kw in t for kw in _BROWSER_TASK_KEYWORDS)


# ── Master system prompt — all tasks ─────────────────────────────────────────
MASTER_SYSTEM_PROMPT = """You are a task-execution agent. Given a task you output a COMPLETE plan as a JSON array of steps, then an executor runs each step in order.

CRITICAL OUTPUT RULES:
1. Output ONLY a JSON array. No markdown, no explanation, no prose.
2. Plan ALL steps upfront in one array. If an early step pauses for input, still include every subsequent step after it.
3. Every step must have: id, type, description, tool, params, success_criteria (plain quoted string, never nested JSON).
4. Reference prior step results as {{step_id.field}} or {{step_id.answer}}.
5. Max 25 steps. Merge trivial adjacent actions where safe.
6. NEVER invent IDs, values, names, or keys the user did not explicitly provide.
7. Greetings / status queries → one direct_response step only.
8. ⚑ DO NOT ASK FOR INFORMATION ALREADY IN THE MESSAGE.
   If the task contains a Jira project key (e.g. "in FLAG", "for FLAG", "FLAG project"),
   use it directly — skip any project selection or ask_user step.
   Same rule applies to: issue key, sprint name, assignee, priority, status, comment text,
   time duration, issue type, board name. Extract from the message first.
   Only ask / show selection when the value is genuinely absent.

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
  ⛔ NEVER invent tool names. If a capability is not in AVAILABLE_MCP_TOOLS, use direct_response to explain the limitation instead of fabricating a tool name.

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

EXCEPTION FIRST — skip the selection flow entirely when the value is already in the task:
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

⛔ NEVER ASSUME any Jira field. This means:
  • NEVER assume the project — ALWAYS fetch and present with request_selection.
  • NEVER assume the sprint — ALWAYS fetch boards then sprints, present with request_selection.
  • NEVER assume the assignee — ALWAYS fetch project members and present with request_selection
    (or ask_user to confirm if no listing tool is available).
  • NEVER assume the issue type — if not 100% explicit in user message, fetch and ask.
  • NEVER assume the board — ALWAYS fetch boards for the selected project.
  Even if the user says "create a task", you MUST still fetch and confirm the project
  before creating anything. Assume nothing.

═══════════════════════════════════════════════════════════════════
JIRA / ATLASSIAN
═══════════════════════════════════════════════════════════════════

Use MCP tools from AVAILABLE_MCP_TOOLS. Do NOT use browser_* tools for Jira.

Verifier note: Jira steps are verified by response content only — no screenshot.
Use plain success_criteria like "Issue created successfully" or "Projects listed".

── CREATE ISSUE — MANDATORY step order (ALWAYS follow this — no exceptions) ──

⛔ NEVER skip or reorder these phases. NEVER use request_selection alone for project — it must be
   part of a request_form that also collects summary and priority in ONE interaction.

  PHASE 1 — Fetch data (no user interaction — TWO steps before the form):
    Step A: jira_get_priorities()           → id: "get_pri"    (needed for priority dropdown)
    Step B: jira_get_assignable_users()     → id: "fetch_users" (needed for assignee dropdown)
            project_key: "<<FIRST_PROJECT_KEY>>" — executor resolves this automatically

  PHASE 2 — Collect ALL info in ONE request_form (NEVER split into multiple interactions):
    ⚑ ALL of the following go in ONE form — user fills and submits once.
    ⚑ EVERY field dict MUST include an "id" key — this is how values are referenced later.
    ⚑ summary is ALWAYS required. "about X" or a topic phrase is NOT a summary — always ask.
    ⚑ priority is ALWAYS required unless user stated an exact priority name.
    ⚑ assignee MUST be a select field using {{fetch_users}} — NEVER a text field.
      ⛔ NEVER use separate jira_get_assignable_users + request_selection steps for assignee.

    EXACT JSON for PHASE 1 + PHASE 2 (copy verbatim — do NOT change ids):
    {"id":"get_pri","tool":"jira_get_priorities","params":{},"success_criteria":"Priorities fetched"},
    {"id":"fetch_users","tool":"jira_get_assignable_users","params":{"project_key":"<<FIRST_PROJECT_KEY>>"},"success_criteria":"Users fetched"},
    {"id":"collect_info","tool":"request_form","params":{"title":"Create Jira Task","fields":[
      {"id":"project","label":"Project","type":"select","required":true,"options":"<<JIRA_PROJECTS_PREFETCHED>>"},
      {"id":"summary","label":"Task Summary","type":"text","required":true,"placeholder":"Short one-line title"},
      {"id":"issue_type","label":"Issue Type","type":"select","required":true,"options":[{"value":"Task","label":"Task"},{"value":"Bug","label":"Bug"},{"value":"Story","label":"Story"},{"value":"Epic","label":"Epic"}]},
      {"id":"priority","label":"Priority","type":"select","required":true,"options":"{{get_pri}}"},
      {"id":"assignee","label":"Assignee","type":"select","required":false,"options":"{{fetch_users}}"},
      {"id":"desc_topic","label":"Description Topic","type":"textarea","required":false,"placeholder":"What should the description cover?"}
    ]},"success_criteria":"All task details collected"}

  PHASE 3 — Generate description (AFTER form submitted):
    llm_draft_content(content_type="issue_description",
                      topic="{{collect_info.desc_topic}}",
                      context="{{collect_info.issue_type}} in {{collect_info.project}}: {{collect_info.summary}}, priority {{collect_info.priority}}")

  PHASE 4 — Let user review/edit the description:
    request_edit(label="Issue Description", content="{{gen_desc.content}}")

  PHASE 5 — Create + update:
    jira_create_issue(project_key="{{collect_info.project}}", issue_type="{{collect_info.issue_type}}",
                      summary="{{collect_info.summary}}", description="{{edit_desc.content}}")
    jira_update_issue(issue_key="{{create_issue.key}}", fields={"priority":"{{collect_info.priority}}","assignee":"{{collect_info.assignee}}"})

  project_key rules:
    • MUST be a short ALL-UPPERCASE key like FLAG, DEV, PROJ.
    • NEVER pass a UUID, numeric ID, lowercase string, or sentence as project_key.

  issue_type rules:
    • "task"/"bug"/"story"/"epic"/"subtask" stated by user → hardcode it.
    • Default: "Task" when ambiguous.

── CREATE SPRINT ──

  Required: board_id (from board selection), sprint_name (from ask_user or task).
  Boards MUST be fetched per-project using jira_get_agile_boards(project_key=...).
  The executor pre-resolves the board list into request_selection options automatically.
  start_date / end_date are auto-filled by executor — NEVER ask the user for them.

  CASE A — project key IS in the task (e.g. "Create a sprint in FLAG"):
    1. jira_get_agile_boards   — project_key: "FLAG"                     → board list
    2. request_selection       — board (options: "{{get_boards}}") → {{sel_board.selected}}
    3. ask_user                — "What should the new sprint be named?"  → {{ask_name.answer}}
    4. jira_create_sprint      — board_id: {{sel_board.selected}}, name: {{ask_name.answer}}

  CASE B — project key NOT in task (e.g. "Create a sprint"):
    1. request_selection       — project (JIRA_PROJECTS_PREFETCHED)    → {{sel_proj.selected}}
    2. jira_get_agile_boards   — project_key: {{sel_proj.selected}}     → board list
    3. request_selection       — board (options: "{{get_boards}}") → {{sel_board.selected}}
    4. ask_user                — "What should the new sprint be named?"  → {{ask_name.answer}}
    5. jira_create_sprint      — board_id: {{sel_board.selected}}, name: {{ask_name.answer}}

── SEARCH / QUERY (jira_search) ──
  Tool: jira_search  (NOT jira_search_issues)
  Params: jql (required — JQL string), limit (optional, default 10)
  Returns JSON list of issues — use {{step_id}} in direct_response.

  Examples:
    jira_search(jql="project = FLAG AND status = Open")
    jira_search(jql="assignee = currentUser() ORDER BY created DESC")
    jira_search(jql="project = DEV AND issuetype = Bug")

  Plan for "show open issues in FLAG":
  [
    {"id":"search","tool":"jira_search","params":{"jql":"project = FLAG AND status != Done","limit":10}},
    {"id":"show","tool":"direct_response","params":{"message":"Open issues in FLAG:\n{{search}}"}}
  ]

── DELETE ISSUE (jira_delete_issue) ──
  Tool: jira_delete_issue
  Params: issue_key* (string, e.g. "FLAG-42")
  ⛔ ALWAYS ask the user to confirm before deleting — deletion is permanent.
  ALWAYS search first so the user picks from a real list (never ask for a key by text).

   Workflow for "delete some tasks in my project":
  [
    {"id":"sel_proj","tool":"request_selection","params":{"question":"Which project?","options":"<<JIRA_PROJECTS_PREFETCHED>>","multi_select":false}},
    {"id":"search","tool":"jira_search","params":{"jql":"project = {{sel_proj.selected}} ORDER BY created DESC","limit":100}},
    {"id":"sel_issue","tool":"request_selection","params":{"question":"Which issue(s) to delete?","options":"{{search}}","multi_select":true}},
    {"id":"confirm","tool":"human_review","params":{"review_type":"delete","summary":"Delete issue(s): {{sel_issue.selected}}"}},
    {"id":"del","tool":"jira_delete_issue","params":{"issue_key":"{{sel_issue.selected}}"}},
    {"id":"done","tool":"direct_response","params":{"message":"Deleted: {{sel_issue.selected}}"}}
  ]

  If project key is already known (e.g. user says "delete tasks in FLAG"):
  Skip sel_proj step, use project key directly in jql.

── GET ISSUE (jira_get_issue) ──
  Tool: jira_get_issue
  Params: issue_key (required, e.g. "FLAG-42")
  Returns JSON with issue details — use {{step_id}} in direct_response.

  • Ask the user for issue_key if not given in the task.
  Plan for "show details of FLAG-42":
  [
    {"id":"get","tool":"jira_get_issue","params":{"issue_key":"FLAG-42"}},
    {"id":"show","tool":"direct_response","params":{"message":"Issue details:\n{{get}}"}}
  ]

── UPDATE ISSUE (jira_update_issue) ──
  Tool: jira_update_issue
  Params:
    issue_key  (required — e.g. "FLAG-42") — ask_user if not in task
    fields     (required — JSON string of fields to update)

  IMPORTANT: fields must be a dict/object (NOT a JSON string). The executor passes it directly to Jira.
  For assignee, always use {"accountId": "<id>"} — never {"name": "..."} (Jira Cloud requires accountId).
  Executor resolves the accountId automatically from the display name when needed.

  Common field names: summary, description, priority ({"name":"High"}), assignee ({"accountId":"..."}),
                      status (use jira_transition_issue for status changes)

  Workflow for "update issue title":
  [
    {"id":"ask_key","tool":"ask_user","params":{"question":"What is the Jira issue key to update? (e.g. FLAG-42)"}},
    {"id":"ask_val","tool":"ask_user","params":{"question":"What should the new title/summary be?"}},
    {"id":"upd","tool":"jira_update_issue","params":{"issue_key":"{{ask_key.answer}}","fields":{"summary":"{{ask_val.answer}}"}}},
    {"id":"done","tool":"direct_response","params":{"message":"Issue updated: {{upd}}"}}
  ]

  Workflow for "set FLAG-42 priority to high" (both given — skip all asks):
  [
    {"id":"upd","tool":"jira_update_issue","params":{"issue_key":"FLAG-42","fields":{"priority":{"name":"High"}}},"success_criteria":"Priority updated"},
    {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 priority set to High."}}
  ]

  Workflow for "set priority to high" (no issue key — ask first):
  [
    {"id":"ask_key","tool":"ask_user","params":{"question":"Which issue key? (e.g. FLAG-42)"}},
    {"id":"upd","tool":"jira_update_issue","params":{"issue_key":"{{ask_key.answer}}","fields":{"priority":{"name":"High"}}},"success_criteria":"Priority updated"},
    {"id":"done","tool":"direct_response","params":{"message":"Priority set to High on {{ask_key.answer}}."}}
  ]

  Workflow for "change the priority of FLAG-42" (key given, priority unknown — show selection):
  [
    {"id":"sel_pri","tool":"request_selection","params":{"question":"Select new priority:","options":["Blocker","Critical","High","Medium","Low"],"multi_select":false}},
    {"id":"upd","tool":"jira_update_issue","params":{"issue_key":"FLAG-42","fields":{"priority":{"name":"{{sel_pri.selected}}"}}},"success_criteria":"Priority updated"},
    {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 priority updated to {{sel_pri.selected}}."}}
  ]

⛔ For CREATE ISSUE user flows, ALWAYS follow PHASE 1-6 below — never skip the form or priority fetch.
   The full PHASE-compliant examples are in the CREATE ISSUE section further below.

── EXAMPLE A: "Create a new sprint in FLAG" (project key given — 4 steps) ──
[
  {"id": "get_boards", "tool": "jira_get_agile_boards", "params": {"project_key": "FLAG"}, "success_criteria": "Boards listed"},
  {"id": "sel_board", "tool": "request_selection", "params": {"question": "Which board?", "options": "{{get_boards}}", "multi_select": false}, "success_criteria": "Board selected"},
  {"id": "ask_name", "tool": "ask_user", "params": {"question": "What should the new sprint be named?"}, "success_criteria": "Sprint name provided"},
  {"id": "create_sprint", "tool": "jira_create_sprint", "params": {"board_id": "{{sel_board.selected}}", "name": "{{ask_name.answer}}"}, "success_criteria": "Sprint created"}
]

── EXAMPLE B: "Create a sprint" (no project — 5 steps) ──
[
  {"id": "sel_proj", "tool": "request_selection", "params": {"question": "Which Jira project?", "options": "<<JIRA_PROJECTS_PREFETCHED>>", "multi_select": false}, "success_criteria": "Project selected"},
  {"id": "get_boards", "tool": "jira_get_agile_boards", "params": {"project_key": "{{sel_proj.selected}}"}, "success_criteria": "Boards listed"},
  {"id": "sel_board", "tool": "request_selection", "params": {"question": "Which board?", "options": "{{get_boards}}", "multi_select": false}, "success_criteria": "Board selected"},
  {"id": "ask_name", "tool": "ask_user", "params": {"question": "What should the new sprint be named?"}, "success_criteria": "Sprint name provided"},
  {"id": "create_sprint", "tool": "jira_create_sprint", "params": {"board_id": "{{sel_board.selected}}", "name": "{{ask_name.answer}}"}, "success_criteria": "Sprint created"}
]
Note: start_date and end_date are auto-filled by the executor. Do NOT ask for them.


⛔ NEVER pass jira_get_transitions result directly as transition_id (it's a list, not a string).

  Required params for jira_transition_issue:
    issue_key     (string, e.g. "FLAG-42")
    transition_id (string — a SINGLE numeric ID like "31", NOT a list)

  DECISION TREE — follow this exactly:

  A) MULTIPLE issues in context AND user didn't specify which ones
     → request_selection(multi_select=true) to let user pick which issues to act on FIRST.
     Then loop the transition steps for each selected issue.

  B) TARGET STATUS already known from user message ("mark as Done", "close", "complete")
     → jira_get_transitions(issue_key=...) → jira_transition_issue(transition_id="{{get_tr}}")
     The executor auto-extracts the matching ID by name. SKIP request_selection for status.

  C) TARGET STATUS is ambiguous (user just said "update status" / "change state")
     → jira_get_transitions → request_selection for status → jira_transition_issue

  EXAMPLE A — "mark them as done" (multiple open tasks returned from previous step):
  [
    {"id":"sel_issues","tool":"request_selection",
     "params":{"question":"Which tasks should I mark as Done?",
               "options":"{{search_tasks}}","multi_select":true},
     "success_criteria":"Issues selected"},
    {"id":"get_tr_1","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions fetched"},
    {"id":"tr_1","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{get_tr_1}}"},"success_criteria":"FLAG-33 marked Done"},
    {"id":"get_tr_2","tool":"jira_get_transitions","params":{"issue_key":"FLAG-32"},"success_criteria":"Transitions fetched"},
    {"id":"tr_2","tool":"jira_transition_issue","params":{"issue_key":"FLAG-32","transition_id":"{{get_tr_2}}"},"success_criteria":"FLAG-32 marked Done"}
  ]
  Note: passing {{get_tr_N}} directly — the executor finds and extracts the "Done" ID automatically.

  EXAMPLE B — "mark FLAG-33 as Done" (single issue, status known):
  [
    {"id":"get_tr","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions listed"},
    {"id":"do_tr","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{get_tr}}"},"success_criteria":"Done"}
  ]

  EXAMPLE C — "change the status of FLAG-33" (ambiguous — ask):
  [
    {"id":"get_tr","tool":"jira_get_transitions","params":{"issue_key":"FLAG-33"},"success_criteria":"Transitions listed"},
    {"id":"sel_tr","tool":"request_selection","params":{"question":"Which status?","options":"{{get_tr}}","multi_select":false},"success_criteria":"Status selected"},
    {"id":"do_tr","tool":"jira_transition_issue","params":{"issue_key":"FLAG-33","transition_id":"{{sel_tr.selected}}"},"success_criteria":"Done"}
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

  Count query ("how many X do I have?"):
  [{"id":"s","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status != Done","limit":1},"success_criteria":"Count returned"},
   {"id":"r","tool":"direct_response","params":{"message":"You have {{s.total}} open issues."}}]

  List query ("show all critical issues"):
  [{"id":"s","tool":"jira_search","params":{"jql":"priority = Critical AND status != Done","limit":25},"success_criteria":"Issues listed"},
   {"id":"r","tool":"direct_response","params":{"message":"Critical open issues:\n{{s}}"}}]

  Assignee workload ("what is Tom working on?" / "show Hrithik's tasks" / "check tasks of Alice"):
  RULE: NEVER use assignee = "display name" directly — Jira Cloud requires an account ID.
        ALWAYS call jira_search_users first to resolve the name to an account ID, then use it in JQL.

  [{"id":"find","tool":"jira_search_users","params":{"query":"Tom"},"success_criteria":"User account found"},
   {"id":"s","tool":"jira_search","params":{"jql":"assignee = \"{{find.accountId}}\" AND status != Done ORDER BY updated DESC","limit":15},"success_criteria":"Issues listed"},
   {"id":"r","tool":"direct_response","params":{"message":"Tom's open issues:\n{{s}}"}}]

  If jira_search_users returns multiple users → add request_selection step before jira_search:
  [{"id":"find","tool":"jira_search_users","params":{"query":"Hrithik"}},
   {"id":"pick","tool":"request_selection","params":{"question":"Which Hrithik?","options":"{{find}}"}},
   {"id":"s","tool":"jira_search","params":{"jql":"assignee = \"{{pick.selected}}\" AND status != Done","limit":15}},
   {"id":"r","tool":"direct_response","params":{"message":"{{pick.selected}}'s open issues:\n{{s}}"}}]

  Sprint days remaining ("how many days left in sprint?"):
  [{"id":"sp","tool":"jira_get_active_sprints","params":{},"success_criteria":"Sprint info returned"},
   {"id":"r","tool":"direct_response","params":{"message":"Active sprint info (calculate days from endDate):\n{{sp}}"}}]

── ADD COMMENT (jira_add_comment) ──
  Tool: jira_add_comment
  Params: issue_key* (string, e.g. "FLAG-42"), comment* (string — the text to add)

  If issue_key not in task → ask_user.
  If comment text not in task → ask_user.
  If issue_key IS in task and comment IS in task (after colon, or in quotes) → skip ask_user steps.

  [{"id":"add","tool":"jira_add_comment","params":{"issue_key":"FLAG-42","comment":"Ready for QA review"},"success_criteria":"Comment added"},
   {"id":"done","tool":"direct_response","params":{"message":"Comment added to FLAG-42."}}]

── LOG TIME / WORKLOG (jira_add_worklog) ──
  Tool: jira_add_worklog
  Params: issue_key* (string), time_spent* (string — "3h", "30m", "1h 30m"), comment (string, optional)

  time_spent format: "3h" not "3 hours". Executor auto-converts plain English to Jira format.
  If issue_key not in task → ask_user.
  If hours not in task → ask_user ("How many hours? e.g. 2h, 30m, 1.5h").
  If BOTH are in task → skip all ask_user steps, go straight to jira_add_worklog.

  Example "Log 3 hours on FLAG-42" (both given — 2 steps):
  [{"id":"log","tool":"jira_add_worklog","params":{"issue_key":"FLAG-42","time_spent":"3 hours"},"success_criteria":"Time logged"},
   {"id":"done","tool":"direct_response","params":{"message":"Logged 3h on FLAG-42."}}]

  Example "Log time on FLAG-42" (hours missing — ask):
  [{"id":"ask_h","tool":"ask_user","params":{"question":"How many hours to log? (e.g. 2h, 30m, 1.5h)"}},
   {"id":"log","tool":"jira_add_worklog","params":{"issue_key":"FLAG-42","time_spent":"{{ask_h.answer}}"},"success_criteria":"Time logged"},
   {"id":"done","tool":"direct_response","params":{"message":"Logged {{ask_h.answer}} on FLAG-42."}}]

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
  Params: issue_key* (string), account_id* (string — Jira accountId, NOT a display name)

  NEVER pass a display name as account_id. Always resolve first.

  "Assign to <name>":
    1. jira_search_users(query="<name>") → get accountId
    2. jira_assign_issue(issue_key=..., account_id="{{find_user.accountId}}")

  "Assign to me" / "Take it":
    jira_assign_issue(issue_key=..., account_id="me")
    (executor auto-substitutes "me" with the configured jira_username)

  Example "Assign FLAG-42 to Rachel":
  [{"id":"find","tool":"jira_search_users","params":{"query":"Rachel"},"success_criteria":"User found"},
   {"id":"asgn","tool":"jira_assign_issue","params":{"issue_key":"FLAG-42","account_id":"{{find.accountId}}"},"success_criteria":"Assigned"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 assigned to Rachel ({{find.displayName}})."}}]

  Example "Assign FLAG-42 to me":
  [{"id":"asgn","tool":"jira_assign_issue","params":{"issue_key":"FLAG-42","account_id":"me"},"success_criteria":"Assigned"},
   {"id":"done","tool":"direct_response","params":{"message":"FLAG-42 assigned to you."}}]

── REASSIGN ISSUE ──
  Reassign = same as Assign, but first confirm current assignee via jira_get_issue if needed.
  If user says "reassign FROM X TO Y" and issue_key is given → skip confirmation, go straight to assign.

── USER ROLES / PROJECT MEMBERS ──
  There is NO tool to fetch user roles. Use jira_get_assignable_users to list who can be assigned,
  then answer with direct_response. Do NOT invent a tool like jira_get_user_roles_for_project.
  Example:
  [{"id":"u","tool":"jira_get_assignable_users","params":{"project_key":"FLAG"}},
   {"id":"done","tool":"direct_response","params":{"message":"Members in FLAG:\n{{u}}\n\nRole information is not available via the current integration."}}]

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

  Workflow — fetch sprint list to get sprint_id:
  [{"id":"get_b","tool":"jira_get_agile_boards","params":{},"success_criteria":"Boards listed"},
   {"id":"get_sp","tool":"jira_get_sprints","params":{"board_id":"{{get_b._items[0].id}}"},"success_criteria":"Sprints listed"},
   {"id":"sel_sp","tool":"request_selection","params":{"question":"Which sprint to start/close?","options":"{{get_sp}}","multi_select":false},"success_criteria":"Sprint selected"},
   {"id":"upd","tool":"jira_update_sprint","params":{"sprint_id":"{{sel_sp.selected}}","state":"active"},"success_criteria":"Sprint started"},
   {"id":"done","tool":"direct_response","params":{"message":"Sprint started."}}]

── CREATE SUBTASK ──
  Subtask = jira_create_issue with issue_type="Sub-task" and parent_key set.
  Params: project_key*, issue_type="Sub-task"*, summary*, parent_key* (parent issue key e.g. "FLAG-42")

  If parent_key not in task → ask_user.
  If summary not in task → ask_user.
  Project key is always derived from the parent issue key prefix.

  Example "Create subtask for FLAG-42: Write unit tests":
  [{"id":"cr","tool":"jira_create_issue","params":{"project_key":"FLAG","issue_type":"Sub-task","summary":"Write unit tests","parent_key":"FLAG-42"},"success_criteria":"Subtask created"},
   {"id":"done","tool":"direct_response","params":{"message":"Subtask created under FLAG-42: {{cr}}"}}]

── STANDUP / DAILY BRIEFING ──
  Runs 3 parallel-intent JQL queries then aggregates into a single digest message.
  [{"id":"my_ip","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status = 'In Progress'","limit":10},"success_criteria":"In-progress fetched"},
   {"id":"done_y","tool":"jira_search","params":{"jql":"assignee = currentUser() AND status = Done AND updated >= -1d","limit":10},"success_criteria":"Done yesterday fetched"},
   {"id":"blockers","tool":"jira_search","params":{"jql":"sprint in openSprints() AND priority = Blocker AND status != Done","limit":10},"success_criteria":"Blockers fetched"},
   {"id":"show","tool":"direct_response","params":{"message":"**Daily Standup Digest**\n\n**In Progress:**\n{{my_ip}}\n\n**Completed Yesterday:**\n{{done_y}}\n\n**Active Blockers:**\n{{blockers}}"}}]

═══════════════════════════════════════════════════════════════════
GOOGLE CALENDAR
═══════════════════════════════════════════════════════════════════

Use MCP tools from AVAILABLE_MCP_TOOLS. No browser tools.
user_google_email is in AVAILABLE DATA and auto-injected — do NOT ask for it.

── VIEW MEETINGS (get_events) ──
• NEVER call list_calendars first. Always use calendar_id "primary".
• get_events returns PLAIN TEXT — use {{step_id}} in direct_response (NO sub-field).
Plan: get_events(calendar_id="primary", time_min="<today>T00:00:00Z", time_max="<today>T23:59:59Z")
      direct_response(message="Your meetings today:\n{{step_id}}")

── AVAILABLE SLOTS / FREE TIME (query_freebusy) ──
• "Available slots", "when am I free", "free time" → use query_freebusy, NOT get_events.
• query_freebusy returns BUSY periods as plain text. FREE time = everything else in working hours.
• Default working hours: 9 AM to 6 PM (local time). Executor auto-injects email + defaults to today.
Plan: query_freebusy()  — no params needed, all auto-filled by executor
      direct_response(message="Your busy periods today:\n{{step_id}}\n\nYour available slots are the gaps between the busy periods above, within your working hours (9 AM – 6 PM).")

── CREATE MEETING (manage_event) ──
CORRECT param names:
  action="create", summary (title), start_time (RFC3339), end_time (RFC3339),
  calendar_id="primary", attendees (list of emails or plain string — executor auto-parses)

⚠️  CONFLICT CHECK IS MANDATORY before creating any meeting.
  Always call query_freebusy for the requested time window FIRST.
  If the slot is busy, use human_review to warn the user and ask if they want to proceed anyway.
  Only create the meeting after the user confirms.

ALWAYS ask for missing info (BEFORE the freebusy check):
  • summary — ask if not given in task
  • start_time — ask if not given, then convert to RFC3339 (e.g. "today at 2pm" → "<today>T08:30:00Z")
  • attendees — ALWAYS ask who to invite (may be solo or group)
  • duration — ask if not given

MANDATORY plan template for "create a meeting":
[
  {"id":"ask_title","tool":"ask_user","params":{"question":"What should the meeting be called?"}},
  {"id":"ask_attendees","tool":"ask_user","params":{"question":"Who would you like to invite? (emails, or 'just me')"}},
  {"id":"ask_duration","tool":"ask_user","params":{"question":"How long? (e.g. '30 minutes', '1 hour')"}},
  {"id":"check_busy","tool":"query_freebusy","params":{"time_min":"<start_time>","time_max":"<end_time>"},
   "success_criteria":"Freebusy checked"},
  {"id":"conflict_gate","tool":"human_review","params":{"question":"⚠️ You have a conflict at this time:\n{{check_busy}}\n\nDo you still want to create the meeting?"},
   "condition":"{{check_busy}} contains busy period",
   "success_criteria":"User confirmed or slot is free — skip this step if no conflict"},
  {"id":"create_evt","tool":"manage_event","params":{"action":"create","summary":"{{ask_title.answer}}","start_time":"<start_time>","duration":"{{ask_duration.answer}}","attendees":"{{ask_attendees.answer}}","calendar_id":"primary"}},
  {"id":"confirm","tool":"direct_response","params":{"message":"✅ Meeting created: {{create_evt}}"}}
]

IMPORTANT:
- If query_freebusy returns NO busy periods overlapping the requested slot → skip human_review, proceed directly to manage_event.
- If there IS a conflict → show human_review with the conflict details. Only proceed if user says yes.
- If the task already contains title AND attendees, skip those ask_user steps.


── UPDATE / DELETE MEETING (manage_event) ──
To update or delete, you need the event_id. Workflow:
  1. get_events — find the meeting (returns plain text with event IDs)
  2. ask_user — ask which meeting to update/delete (show the list from step 1)
  3. manage_event(action="update"|"delete", event_id="<from user>", ...)

Example for "delete the 2pm meeting":
[
  {"id":"get_evt","tool":"get_events","params":{"calendar_id":"primary","time_min":"<today>T00:00:00Z","time_max":"<today>T23:59:59Z"}},
  {"id":"ask_id","tool":"ask_user","params":{"question":"Which meeting ID should I delete? Here are your meetings:\n{{get_evt}}"}},
  {"id":"del_evt","tool":"manage_event","params":{"action":"delete","event_id":"{{ask_id.answer}}","calendar_id":"primary"}},
  {"id":"confirm","tool":"direct_response","params":{"message":"Done: {{del_evt}}"}}
]

═══════════════════════════════════════════════════════════════════
GMAIL / EMAIL
═══════════════════════════════════════════════════════════════════

Use MCP tools from AVAILABLE_MCP_TOOLS. No browser tools.
user_google_email is auto-injected from settings — do NOT ask for it.

── SEND EMAIL (send_gmail_message) ──
Params: to (required), subject (required), body (required)
Ask for any of these if not given. user_google_email is auto-injected.
  {"id":"send","tool":"send_gmail_message","params":{"to":"alice@example.com","subject":"Hello","body":"Hi there"}}

  {"id":"search","tool":"search_gmail_messages","params":{"query":"is:unread"}}
  (Wait: search_gmail_messages DOES NOT support 'limit' or 'max_results'. Never pass those.)

── COUNT / SUMMARIZE EMAILS ──
If the user asks "how many", "what is the count", or "total number" of emails, ALWAYS use `get_total_unread_emails`.
  [{"id":"count","tool":"get_total_unread_emails","params":{},"success_criteria":"Count retrieved"},
   {"id":"show","tool":"direct_response","params":{"message":"{{count}}"}}]

If you need metrics for specific labels (SPAM, TRASH, etc.), use `get_label_metrics(labels=["SPAM", "INBOX"])`.

── READ EMAIL (get_gmail_message_content) ──
Params: message_id (from search result)
Workflow: search → ask_user which message → get content
  {"id":"search","tool":"search_gmail_messages","params":{"query":"from:boss@company.com"}},
  {"id":"ask_id","tool":"ask_user","params":{"question":"Which message would you like to read? Here are the results:\n{{search}}"}},
  {"id":"read","tool":"get_gmail_message_content","params":{"message_id":"{{ask_id.answer}}"}},
  {"id":"show","tool":"direct_response","params":{"message":"Email content:\n{{read}}"}}

── BATCH ACTIONS (Mark as Read / Delete / Archive) ──
CRITICAL: NEVER use `search_gmail_messages` to fetch IDs for batch actions (it returns plain text).
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

CRITICAL RULES:
1. Output a JSON array of ALL steps. NOTHING else — no markdown, no explanation.
2. ALWAYS generate the COMPLETE plan upfront — never plan just one step and stop.
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
9. NEVER invent IDs, numbers, or data values. If the user did not provide a specific value
   (like an opportunity ID), you MUST ask for it with ask_user. Do NOT copy IDs from examples.
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


import re as _re_fast

# ── Fast-path patterns for common Jira queries (bypass LLM, instant response) ─
# Each entry: (compiled_regex, jql_template, response_template, limit)
# jql_template may reference group(1) from the regex match.
_FAST_JIRA_DIRECT: list[tuple] = [
    # "how many tickets/issues do I have" / "how many issues do we have" (personal or team)
    (
        _re_fast.compile(
            r"\b(how many|count)\b.{0,40}\b(tickets?|issues?|tasks?|blockers?)\b.{0,40}\b(i have|do i have|do we have|are there|assigned to me|mine)\b",
            _re_fast.IGNORECASE,
        ),
        "assignee = currentUser() AND status != Done",
        "You have {total} open issues assigned to you.",
        1,
    ),
    # "count my open issues" / "count my tickets" (no trailing clause)
    (
        _re_fast.compile(
            r"\b(count|how many)\b.{0,15}\bmy\b.{0,25}\b(tickets?|issues?|tasks?)\b",
            _re_fast.IGNORECASE,
        ),
        "assignee = currentUser() AND status != Done",
        "You have {total} open issues assigned to you.",
        1,
    ),
    # "which (all) open tasks/issues do i have"
    (
        _re_fast.compile(
            r"\bwhich\b.{0,30}\b(tasks?|issues?|tickets?)\b.{0,30}\b(i have|do i have)\b",
            _re_fast.IGNORECASE,
        ),
        "assignee = currentUser() AND status != Done ORDER BY updated DESC",
        "Your open issues:\n{results}",
        25,
    ),
    # "show/list/give me my tickets/issues" — all "my X" phrasings
    (
        _re_fast.compile(
            r"\b(show|list|display|view|give\s+me|get\s+me|fetch)\b.{0,20}\bmy\b.{0,20}\b(open\s+)?(tickets?|issues?|tasks?)\b",
            _re_fast.IGNORECASE,
        ),
        "assignee = currentUser() AND status != Done ORDER BY updated DESC",
        "Your open issues:\n{results}",
        25,
    ),
    # "what are my tasks / what's on my plate / what do i have"
    (
        _re_fast.compile(
            r"\b(what\s+(are|is)\s+my|what.{0,10}i\s+(have|need\s+to)|what.{0,10}on\s+my\s+plate)\b.{0,20}\b(tickets?|issues?|tasks?|work|todo)?\b",
            _re_fast.IGNORECASE,
        ),
        "assignee = currentUser() AND status != Done ORDER BY updated DESC",
        "Your open issues:\n{results}",
        25,
    ),
    # "show unassigned issues" — unassigned before type noun
    (
        _re_fast.compile(
            r"\b(show|list|display|what|are there)\b.{0,20}\bunassigned\b.{0,20}\b(tickets?|issues?|tasks?)\b",
            _re_fast.IGNORECASE,
        ),
        "assignee is EMPTY AND sprint in openSprints() AND status != Done",
        "Unassigned issues in the active sprint:\n{results}",
        25,
    ),
    # "what tasks are unassigned?" — type noun before unassigned
    (
        _re_fast.compile(
            r"\b(tasks?|issues?|tickets?)\b.{0,25}\bare\s+unassigned\b",
            _re_fast.IGNORECASE,
        ),
        "assignee is EMPTY AND sprint in openSprints() AND status != Done",
        "Unassigned issues in the active sprint:\n{results}",
        25,
    ),
]

# Priority-filter fast paths (one per priority level)
# Pattern uses s? so both "blocker" and "blockers" match.
for _pri in ("Blocker", "Critical", "High", "Medium", "Low"):
    _FAST_JIRA_DIRECT.append(
        (
            _re_fast.compile(
                rf"\b(show|list|display|are\s+there)\b.{{0,25}}\b{_pri.lower()}s?\b.{{0,25}}\b(tickets?|issues?|tasks?|priority)?\b",
                _re_fast.IGNORECASE,
            ),
            f"priority = {_pri} AND status != Done ORDER BY updated DESC",
            f"{_pri} open issues:\n{{results}}",
            25,
        )
    )

# Status-filter fast paths
_STATUS_MAP = {
    "in progress": "In Progress",
    "in review": "In Review",
    "to do": "To Do",
    "done": "Done",
    "testing": "Testing",
    "backlog": "Backlog",
}
for _slug, _jira_status in _STATUS_MAP.items():
    _FAST_JIRA_DIRECT.append(
        (
            _re_fast.compile(
                rf"\b(show|list|display|what)\b.{{0,20}}\b({_slug.replace(' ', r'[- ]')})\b.{{0,20}}\b(tickets?|issues?|tasks?)?\b",
                _re_fast.IGNORECASE,
            ),
            f'status = "{_jira_status}" AND sprint in openSprints() ORDER BY updated DESC',
            f"{_jira_status} issues:\n{{results}}",
            25,
        )
    )


def _fast_jira_plan(task: str) -> list | None:
    """Return a pre-built 2-step plan for simple Jira queries, bypassing the LLM.

    Returns None if the task doesn't match a known fast-path pattern,
    in which case the caller should fall through to the full planner.
    """
    for pattern, jql, response_tmpl, limit in _FAST_JIRA_DIRECT:
        if pattern.search(task):
            logger.info("[PLANNER] Fast-path match for Jira query: %s", pattern.pattern[:60])
            return [
                {
                    "id": "jira_q",
                    "tool": "jira_search",
                    "description": f"Search Jira: {jql[:80]}",
                    "params": {"jql": jql, "limit": limit},
                    "success_criteria": "Issues retrieved",
                },
                {
                    "id": "show",
                    "tool": "direct_response",
                    "description": "Present results",
                    "params": {"message": "{{jira_q}}"},
                    "success_criteria": "Results shown",
                },
            ]
    return None


# ── Fast-path patterns for common Gmail read queries ─────────────────────────
# Each entry: (compiled_regex, gmail_query_string, description)
_FAST_GMAIL_READ: list[tuple] = [
    # "show my unread emails" / "any new emails?" / "any unread?"
    (
        _re_fast.compile(
            r"\b(show|check|list|any|get|fetch|display)\b.{0,20}\bunread\b.{0,20}\b(emails?|messages?|mail)?\b"
            r"|\bany\s+(new|unread)\s*(emails?|messages?|mail)?\b",
            _re_fast.IGNORECASE,
        ),
        "is:unread in:inbox",
        "Your unread emails",
    ),
    # "check my inbox" / "what's in my inbox"
    (
        _re_fast.compile(
            r"\b(check|show|open|view|what.{0,10}in)\b.{0,15}\bmy\b.{0,10}\binbox\b"
            r"|\bmy\s+inbox\b",
            _re_fast.IGNORECASE,
        ),
        "in:inbox",
        "Your inbox",
    ),
    # "show my sent emails"
    (
        _re_fast.compile(
            r"\b(show|list|display|get)\b.{0,15}\b(my\s+)?(sent|outbox)\b.{0,15}\b(emails?|messages?|mail)?\b",
            _re_fast.IGNORECASE,
        ),
        "in:sent",
        "Your sent emails",
    ),
    # "show starred emails"
    (
        _re_fast.compile(
            r"\b(show|list|display|get)\b.{0,15}\bstarred\b.{0,15}\b(emails?|messages?|mail)?\b",
            _re_fast.IGNORECASE,
        ),
        "is:starred",
        "Your starred emails",
    ),
    # "show important emails"
    (
        _re_fast.compile(
            r"\b(show|list|display|get)\b.{0,15}\bimportant\b.{0,15}\b(emails?|messages?|mail)?\b",
            _re_fast.IGNORECASE,
        ),
        "is:important",
        "Your important emails",
    ),
    # "show emails with attachments"
    (
        _re_fast.compile(
            r"\b(show|list|display|get|find)\b.{0,20}\b(emails?|messages?|mail)\b.{0,20}\battachment(s)?\b"
            r"|\battachment(s)?\b.{0,20}\b(emails?|messages?|mail)\b",
            _re_fast.IGNORECASE,
        ),
        "has:attachment in:inbox",
        "Emails with attachments",
    ),
    # "show emails from today" / "show emails I sent today"
    (
        _re_fast.compile(
            r"\b(show|list|get|check)\b.{0,20}\b(emails?|messages?|mail)\b.{0,20}\btoday\b"
            r"|\btoday.{0,10}\b(emails?|messages?|mail)\b",
            _re_fast.IGNORECASE,
        ),
        "after:today in:inbox",
        "Today's emails",
    ),
    # "show spam" / "show emails in spam"
    (
        _re_fast.compile(
            r"\b(show|list|check)\b.{0,20}\b(in\s+)?(spam|junk)\b",
            _re_fast.IGNORECASE,
        ),
        "in:spam",
        "Spam folder",
    ),
]


def _fast_gmail_plan(task: str) -> list | None:
    """Return a pre-built 2-step plan for simple Gmail read queries, bypassing the LLM.

    Only covers pure read/search operations (Sections 1, partial 11).
    Send/reply/forward/label flows are too variable — those go to the LLM.
    Returns None if not a fast-path match.
    """
    # Don't fast-path if task mentions a specific sender/subject — those need the LLM
    # to compose the right query string.
    _SPECIFIC_PATTERNS = _re_fast.compile(
        r"\bfrom\b|\bsubject\b|\babout\b|\bregarding\b|\bsent by\b|\bby\b\s+[A-Z]",
        _re_fast.IGNORECASE,
    )
    if _SPECIFIC_PATTERNS.search(task):
        return None

    # Don't fast-path send/reply/forward/draft actions
    _ACTION_PATTERNS = _re_fast.compile(
        r"\b(send|reply|forward|draft|compose|write|respond|email\s+\w+@|ping\b|drop.{0,5}message)\b",
        _re_fast.IGNORECASE,
    )
    if _ACTION_PATTERNS.search(task):
        return None

    for pattern, gmail_query, description in _FAST_GMAIL_READ:
        if pattern.search(task):
            logger.info("[PLANNER] Gmail fast-path match: %s", pattern.pattern[:60])
            return [
                {
                    "id": "gmail_q",
                    "tool": "search_gmail_messages",
                    "description": description,
                    "params": {"query": gmail_query},
                    "success_criteria": "Emails retrieved",
                },
                {
                    "id": "show",
                    "tool": "direct_response",
                    "description": "Present results",
                    "params": {"message": "{{gmail_q}}"},
                    "success_criteria": "Results shown",
                },
            ]
    return None


async def planner_node(state: AgentState) -> dict:
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

    # ── Special case: "what's in the current sprint" ────────────────────────
    import re as _re_sprint

    if _re_sprint.search(
        r"\b(what.?s|what.?is)\b.{0,20}\b(in|current)\b.{0,10}\bsprint\b",
        task,
        _re_sprint.IGNORECASE,
    ):
        logger.info("[PLANNER] Special case: current sprint query")
        # This requires a multi-step workflow that the LLM planner gets wrong
        sprint_plan = [
            {
                "id": "sel_proj",
                "tool": "request_selection",
                "description": "Select project to check sprints",
                "params": {
                    "question": "Which project do you want to check the current sprint for?",
                    "options": "{{projects}}",  # This will be resolved from cached projects
                    "multi_select": False,
                },
                "success_criteria": "Project selected",
            },
            {
                "id": "get_boards",
                "tool": "jira_get_agile_boards",
                "description": "Get agile boards for selected project",
                "params": {"project_key": "{{sel_proj.selected}}"},
                "success_criteria": "Boards retrieved",
            },
            {
                "id": "sel_board",
                "tool": "request_selection",
                "description": "Select board to check sprints",
                "params": {
                    "question": "Which board do you want to check?",
                    "options": "{{get_boards}}",
                    "multi_select": False,
                },
                "success_criteria": "Board selected",
            },
            {
                "id": "get_sprints",
                "tool": "jira_get_sprints_from_board",
                "description": "Get sprints for selected board",
                "params": {"board_id": "{{sel_board.selected}}"},
                "success_criteria": "Sprints retrieved",
            },
            {
                "id": "sel_sprint",
                "tool": "request_selection",
                "description": "Select which sprint to view",
                "params": {
                    "question": "Which sprint do you want to see issues for?",
                    "options": "{{get_sprints}}",
                    "multi_select": False,
                },
                "success_criteria": "Sprint selected",
            },
            {
                "id": "get_issues",
                "tool": "jira_get_sprint_issues",
                "description": "Get issues in selected sprint",
                "params": {"sprint_id": "{{sel_sprint.selected}}"},
                "success_criteria": "Sprint issues retrieved",
            },
            {
                "id": "response",
                "tool": "direct_response",
                "description": "Show sprint issues",
                "params": {"message": "{{get_issues}}"},
                "success_criteria": "Results displayed",
            },
        ]
        cost = COST_PER_CALL["planner"]
        return {
            "plan": sprint_plan,
            "current_step_index": 0,
            "status": "executing",
            "retry_count": 0,
            "replan_count": state.get("replan_count", 0),
            "steps_taken": state.get("steps_taken", 0),
            "estimated_cost": state.get("estimated_cost", 0.0),
            "step_results": [],
            "flow_data": {},
            "messages": [
                AIMessage(
                    content=f"Plan created with {len(sprint_plan)} steps. Starting execution..."
                )
            ],
        }

    # ── Fast path: bypass LLM for simple Jira or Gmail queries ──────────────
    fast_plan = _fast_jira_plan(task) or _fast_gmail_plan(task)
    if fast_plan:
        logger.info("[PLANNER] Fast-path plan returned (%d steps)", len(fast_plan))
        cost = COST_PER_CALL["planner"]
        return {
            "plan": fast_plan,
            "current_step_index": 0,
            "status": "executing",
            "retry_count": 0,
            "replan_count": state.get("replan_count", 0),
            "steps_taken": state.get("steps_taken", 0),
            "estimated_cost": state.get("estimated_cost", 0.0),  # no LLM cost
            "step_results": [],
            "flow_data": {},
            "messages": [
                AIMessage(
                    content=f"Plan created with {len(fast_plan)} steps. Starting execution..."
                )
            ],
        }

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
    context_parts.append(f"TODAY'S DATE: {_today}  (use this for any date-relative queries)")

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
        context_parts.append(
            f"AVAILABLE DATA (already known — do NOT ask the user for these): {json.dumps(merged_data, default=str)[:2000]}"
        )

    user_msg = "\n\n".join(context_parts)

    # Choose system prompt: master prompt always; browser supplement only for browser tasks.
    system_prompt = MASTER_SYSTEM_PROMPT
    if not _settings.disable_browser_tools and _needs_browser(task):
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

    response = await llm.ainvoke(llm_messages)

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
                        + "\n\nCRITICAL: Output ONLY valid JSON array. No markdown fences, no explanation. Every string value must use \\n for newlines, never bare newlines."
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
    return {
        "plan": plan,
        "current_step_index": 0,
        "status": "executing",
        "retry_count": 0,
        "replan_count": state.get("replan_count", 0),
        "steps_taken": state.get("steps_taken", 0),
        "estimated_cost": state.get("estimated_cost", 0.0) + cost,
        # Reset per-task context so stale step results / flow_data from a previous
        # plan don't bleed into this one (e.g. board_id skipping the selection step).
        "step_results": [],
        "flow_data": {},
        "messages": [
            AIMessage(content=f"Plan created with {len(plan)} steps. Starting execution...")
        ],
    }
