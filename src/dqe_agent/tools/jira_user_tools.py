"""Jira user-lookup tools.

Provides assignable-user listing for a Jira project so the planner can
present real user options via request_selection instead of asking for a
free-text name.

Calls the Jira REST API directly using the credentials in settings.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from dqe_agent.tools import register_tool

logger = logging.getLogger(__name__)


def _jira_auth() -> tuple[str, str] | None:
    """Return (username, api_token) or None if Jira is not configured."""
    from dqe_agent.config import settings
    if settings.jira_url and settings.jira_username and settings.jira_api_token:
        return (settings.jira_username, settings.jira_api_token)
    if settings.jira_url and settings.jira_username and settings.jira_password:
        return (settings.jira_username, settings.jira_password)
    return None


async def _fuzzy_match_project_key(
    guess: str,
    base_url: str,
    auth: tuple[str, str],
) -> str:
    """Fetch all Jira projects and return the key whose name/key best matches `guess`.

    Matching priority:
      1. Exact key match (case-insensitive)
      2. Key starts with guess or guess starts with key
      3. Every word in guess appears in the project name (case-insensitive)
      4. Any word in guess appears in the project name
    Returns "" if nothing matches.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url}/rest/api/3/project/search",
                params={"maxResults": 100},
                auth=auth,
            )
            resp.raise_for_status()
            projects = resp.json().get("values", [])
    except Exception as exc:
        logger.warning("[_fuzzy_match_project_key] Failed to fetch projects: %s", exc)
        # Fall back to planner cache
        try:
            from dqe_agent.agent.planner import _cache_get
            cached = _cache_get("jira_projects") or []
            projects = [{"key": p.get("value", ""), "name": p.get("label", "")} for p in cached]
        except Exception:
            return ""

    guess_clean = guess.strip().upper()
    guess_words = [w.lower() for w in guess.split() if len(w) > 1]

    # Pass 1: exact key
    for p in projects:
        if p.get("key", "").upper() == guess_clean:
            return p["key"]

    # Pass 2: key prefix / substring
    for p in projects:
        k = p.get("key", "").upper()
        if k.startswith(guess_clean) or guess_clean.startswith(k):
            return p["key"]

    # Pass 3: all words in guess appear in project name
    for p in projects:
        name_lower = p.get("name", "").lower()
        if guess_words and all(w in name_lower for w in guess_words):
            return p["key"]

    # Pass 4: any word matches
    for p in projects:
        name_lower = p.get("name", "").lower()
        k_lower = p.get("key", "").lower()
        if any(w in name_lower or w in k_lower for w in guess_words):
            return p["key"]

    return ""


@register_tool(
    "jira_get_assignable_users",
    "Fetch the list of members in a Jira project. "
    "Returns a list of {value, label} options ready for request_selection. "
    "Use this before showing an assignee selection to the user.",
)
async def jira_get_assignable_users(project_key: str) -> list[dict[str, str]]:
    """Return project members as [{value, label}] options.

    Strategy:
    1. Fetch all project roles and collect members from each role (actual project members).
    2. Fall back to /user/assignable/search if role-based fetch returns nothing.
    """
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        logger.warning("[jira_get_assignable_users] Jira not configured — returning empty list")
        return []

    base_url = settings.jira_url.rstrip("/")

    # Fuzzy-match project key if needed
    async def _key_exists(key: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{base_url}/rest/api/3/project/{key}", auth=auth)
                return r.status_code == 200
        except Exception:
            return False

    if not await _key_exists(project_key):
        matched_key = await _fuzzy_match_project_key(project_key, base_url, auth)
        if matched_key:
            logger.info("[jira_get_assignable_users] fuzzy matched %r → %r", project_key, matched_key)
            project_key = matched_key
        else:
            logger.warning("[jira_get_assignable_users] no fuzzy match found for %r", project_key)
            return [{"value": "__error__", "label": f"Project '{project_key}' not found in Jira. Please check the project key."}]

    seen: dict[str, str] = {}  # display_name → label

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # Step 1: get all roles for the project
            roles_resp = await client.get(
                f"{base_url}/rest/api/3/project/{project_key}/role",
                auth=auth,
            )
            roles_resp.raise_for_status()
            roles: dict = roles_resp.json()  # {role_name: role_url}

            # Step 2: fetch members of each role
            for role_name, role_url in roles.items():
                try:
                    r = await client.get(str(role_url), auth=auth)
                    if r.status_code != 200:
                        continue
                    role_data = r.json()
                    for actor in role_data.get("actors", []):
                        # actorType "atlassian-user-role-actor" = real user
                        if actor.get("actorUser") or actor.get("type") == "atlassian-user-role-actor":
                            name = actor.get("displayName", "")
                            acct = actor.get("actorUser", {})
                            email = acct.get("emailAddress", "") if isinstance(acct, dict) else ""
                            if not name:
                                continue
                            if name not in seen:
                                seen[name] = f"{name} ({email})" if email else name
                except Exception as _re:
                    logger.debug("[jira_get_assignable_users] role %r fetch error: %s", role_name, _re)
                    continue

    except httpx.HTTPStatusError as exc:
        logger.warning("[jira_get_assignable_users] roles HTTP %s — %s", exc.response.status_code, exc.response.text[:200])
    except Exception as exc:
        logger.warning("[jira_get_assignable_users] roles fetch failed: %s", exc)

    # Fallback: if roles gave nothing, use /user/assignable/search (may return more than actual members)
    if not seen:
        logger.info("[jira_get_assignable_users] roles returned 0 — falling back to assignable/search")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{base_url}/rest/api/3/user/assignable/search",
                    params={"project": project_key, "maxResults": 50},
                    auth=auth,
                )
                resp.raise_for_status()
                for u in resp.json():
                    if not u.get("accountId") or u.get("accountType") == "app":
                        continue
                    name = u.get("displayName", "")
                    email = u.get("emailAddress", "")
                    if name and name not in seen:
                        seen[name] = f"{name} ({email})" if email else name
        except Exception as exc:
            logger.warning("[jira_get_assignable_users] assignable/search fallback failed: %s", exc)

    options = [{"value": name, "label": label} for name, label in sorted(seen.items())]
    logger.info("[jira_get_assignable_users] project=%s → %d users", project_key, len(options))
    return options


@register_tool(
    "jira_get_priorities",
    "Fetch the list of valid priority names for a Jira project. "
    "Pass the project_key to get project-specific valid priorities (recommended). "
    "Returns a list of {value, label} options ready for request_selection. "
    "ALWAYS call this before showing a priority selection — never hardcode priority names "
    "because different Jira projects use different schemes (e.g. P1/P2/P3 vs High/Medium/Low).",
)
async def jira_get_priorities(project_key: str = "") -> list[dict[str, str]]:
    """Return valid Jira priorities as [{value, label}] options.

    When project_key is given, fetches the project-specific allowed priority values
    via the issue create-meta API (most accurate). Falls back to the global priority
    list if project-specific data is unavailable.
    """
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        logger.warning("[jira_get_priorities] Jira not configured — returning defaults")
        return [
            {"value": "Highest", "label": "Highest"},
            {"value": "High",    "label": "High"},
            {"value": "Medium",  "label": "Medium"},
            {"value": "Low",     "label": "Low"},
            {"value": "Lowest",  "label": "Lowest"},
        ]

    base_url = settings.jira_url.rstrip("/")

    async with httpx.AsyncClient(timeout=15) as client:
        # --- Approach 1: Priority scheme API (project-specific, most accurate) ---
        if project_key:
            project_priorities = await _get_priorities_from_scheme(client, base_url, auth, project_key)
            if project_priorities:
                logger.info(
                    "[jira_get_priorities] project=%s scheme → %d priorities: %s",
                    project_key, len(project_priorities), [o["value"] for o in project_priorities],
                )
                return project_priorities

        # --- Approach 2: Global priority list (fallback) ---
        try:
            resp = await client.get(f"{base_url}/rest/api/3/priority", auth=auth)
            resp.raise_for_status()
            priorities: list[dict] = resp.json()
        except Exception as exc:
            logger.warning("[jira_get_priorities] global list failed: %s", exc)
            return []

    options = [
        {"value": p["name"], "label": p["name"]}
        for p in priorities
        if p.get("name")
    ]
    logger.info("[jira_get_priorities] global → %d priorities: %s", len(options), [o["value"] for o in options])
    return options


async def _get_priorities_from_scheme(
    client: "httpx.AsyncClient",
    base_url: str,
    auth: tuple[str, str],
    project_key: str,
) -> list[dict[str, str]]:
    """Fetch the priority scheme assigned to the given project and return its priorities."""
    try:
        # Get all priority schemes with their project associations and their priorities
        resp = await client.get(
            f"{base_url}/rest/api/3/priorityscheme",
            params={"expand": "priorities,projects", "maxResults": 50},
            auth=auth,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("[jira_get_priorities] priorityscheme API failed: %s", exc)
        return []

    schemes = data.get("values", [])
    if not schemes:
        return []

    # Find the scheme explicitly assigned to this project; fall back to the default scheme
    matched_scheme: dict | None = None
    default_scheme: dict | None = None
    for scheme in schemes:
        if scheme.get("isDefault"):
            default_scheme = scheme
        project_keys_in_scheme = [
            p.get("key") for p in scheme.get("projects", {}).get("values", [])
        ]
        if project_key in project_keys_in_scheme:
            matched_scheme = scheme
            break

    target = matched_scheme or default_scheme
    if not target:
        return []

    # Extract priorities from the scheme
    scheme_priorities = target.get("priorities", {}).get("values", [])
    options = [
        {"value": p["name"], "label": p["name"]}
        for p in scheme_priorities
        if p.get("name")
    ]
    logger.info(
        "[jira_get_priorities] scheme '%s' (default=%s) → %d priorities for project %s",
        target.get("name"), target.get("isDefault"), len(options), project_key,
    )
    return options


@register_tool(
    "jira_get_project_roles",
    "Fetch the list of roles available in a Jira project (e.g. Developer, Viewer, Service Desk Team). "
    "Returns [{value, label}] options ready for request_selection. "
    "Call this before asking the user which role to assign to a new member.",
)
async def jira_get_project_roles(project_key: str) -> list[dict[str, str]]:
    """Return project roles as [{value: role_id, label: role_name}]."""
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        logger.warning("[jira_get_project_roles] Jira not configured")
        return []

    base_url = settings.jira_url.rstrip("/")
    url = f"{base_url}/rest/api/3/project/{project_key}/role"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, auth=auth)
            resp.raise_for_status()
            data: dict = resp.json()
    except Exception as exc:
        logger.warning("[jira_get_project_roles] Failed: %s", exc)
        return []

    options: list[dict[str, str]] = []
    for role_name, role_url in data.items():
        # Extract role ID from the URL (last path segment)
        role_id = str(role_url).rstrip("/").split("/")[-1]
        options.append({"value": role_id, "label": role_name})

    logger.info("[jira_get_project_roles] project=%s → %d roles", project_key, len(options))
    return options


@register_tool(
    "jira_search_user_by_email",
    "Search for a Jira user by their email address. "
    "Returns the user's accountId and displayName if found, or an error message. "
    "Use this before adding a member to a project — you need the accountId.",
)
async def jira_search_user_by_email(email: str) -> dict[str, str]:
    """Return {accountId, displayName, emailAddress} for the matching user, or {error}."""
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        return {"error": "Jira not configured"}

    base_url = settings.jira_url.rstrip("/")
    url = f"{base_url}/rest/api/3/user/search"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params={"query": email, "maxResults": 5}, auth=auth)
            resp.raise_for_status()
            users: list[dict] = resp.json()
    except Exception as exc:
        logger.warning("[jira_search_user_by_email] Failed: %s", exc)
        return {"error": str(exc)}

    # Find exact email match first, then fall back to first result
    matched = next((u for u in users if u.get("emailAddress", "").lower() == email.lower()), None)
    if not matched and users:
        matched = users[0]

    if not matched:
        return {"error": f"No Jira user found for email '{email}'"}

    result = {
        "accountId": matched.get("accountId", ""),
        "displayName": matched.get("displayName", ""),
        "emailAddress": matched.get("emailAddress", ""),
    }
    logger.info("[jira_search_user_by_email] email=%s → %s (%s)", email, result["displayName"], result["accountId"])
    return result


@register_tool(
    "jira_add_project_member",
    "Add a user to a Jira project role. "
    "Requires: project_key (e.g. FLAG), account_id (from jira_search_user_by_email), role_id (from jira_get_project_roles). "
    "Returns success or error message.",
)
async def jira_add_project_member(project_key: str, account_id: str, role_id: str) -> dict[str, str]:
    """Add account_id to the given role in the project. Returns {status, message}."""
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        return {"status": "error", "message": "Jira not configured"}

    # role_id must be numeric — reject display names passed by mistake
    if not str(role_id).strip().isdigit():
        return {
            "status": "error",
            "message": (
                f"role_id must be a numeric Jira role ID (got '{role_id}'). "
                "Call jira_get_project_roles first to get the correct numeric ID."
            ),
        }

    base_url = settings.jira_url.rstrip("/")
    url = f"{base_url}/rest/api/3/project/{project_key}/role/{role_id}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json={"user": [account_id]}, auth=auth)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        msg = exc.response.text[:300]
        logger.warning("[jira_add_project_member] HTTP %s: %s", exc.response.status_code, msg)
        return {"status": "error", "message": f"Jira API error {exc.response.status_code}: {msg}"}
    except Exception as exc:
        logger.warning("[jira_add_project_member] Failed: %s", exc)
        return {"status": "error", "message": str(exc)}

    logger.info("[jira_add_project_member] project=%s role=%s accountId=%s → added", project_key, role_id, account_id)
    return {"status": "success", "message": f"User added to project {project_key} successfully."}


def _seconds_to_hm(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"


@register_tool(
    "jira_get_worklogs_by_date_range",
    "Fetch actual hours logged by team members in a Jira project within a date range. "
    "Returns a per-user breakdown (total hours + per-issue detail). "
    "Parameters: project_key (e.g. AMA), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD), "
    "member_name (optional — display name to filter a specific user; omit or pass '' for all members). "
    "ALWAYS use this tool for 'log hours', 'time logged', 'worklogs' requests — "
    "jira_search only returns issue lists, not actual hour data.",
)
async def jira_get_worklogs_by_date_range(
    project_key: str,
    start_date: str,
    end_date: str,
    member_name: str = "",
) -> dict:
    """Aggregate worklog hours per user for project in [start_date, end_date]."""
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        return {"error": "Jira not configured"}

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = (
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            + timedelta(days=1)
            - timedelta(seconds=1)
        )
    except ValueError as exc:
        return {"error": f"Invalid date format (expected YYYY-MM-DD): {exc}"}

    started_after_ms = int(start_dt.timestamp() * 1000)
    started_before_ms = int(end_dt.timestamp() * 1000)

    # JQL — member filter uses worklogAuthor only when member is known;
    # local displayName filtering applied afterwards for safety.
    jql_parts = [
        f'project = {project_key}',
        f'worklogDate >= "{start_date}"',
        f'worklogDate <= "{end_date}"',
    ]
    if member_name and member_name not in ("__all__", ""):
        if member_name == "__me__":
            jql_parts.append("worklogAuthor = currentUser()")
        else:
            jql_parts.append(f'worklogAuthor = "{member_name}"')

    jql = " AND ".join(jql_parts)
    base_url = settings.jira_url.rstrip("/")

    def _in_range(wl: dict) -> bool:
        """True if the worklog's started timestamp falls in [start_dt, end_dt]."""
        started_str = wl.get("started", "")
        if not started_str:
            return True  # can't tell — include it
        try:
            # Jira format: "2026-04-24T10:00:00.000+0000"
            started = datetime.strptime(started_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            return start_dt <= started <= end_dt
        except ValueError:
            return True

    def _accept_author(display_name: str) -> bool:
        if not member_name or member_name in ("__all__", "", "__me__"):
            return True
        return display_name.lower() == member_name.lower()

    def _accumulate(user_totals: dict, key: str, summary: str, wl: dict) -> None:
        author_info = wl.get("author") or {}
        display_name = author_info.get("displayName", "Unknown")
        secs = int(wl.get("timeSpentSeconds", 0))
        if not secs or not _in_range(wl) or not _accept_author(display_name):
            return
        if display_name not in user_totals:
            user_totals[display_name] = {"total_seconds": 0, "issues": {}}
        user_totals[display_name]["total_seconds"] += secs
        if key not in user_totals[display_name]["issues"]:
            user_totals[display_name]["issues"][key] = {"summary": summary, "seconds": 0}
        user_totals[display_name]["issues"][key]["seconds"] += secs

    all_issues: list[dict] = []
    user_totals: dict[str, dict] = {}

    try:
        async with httpx.AsyncClient(timeout=30) as client:

            # ── Step 1: search/jql with embedded worklogs (up to 20 per issue) ──
            start_at = 0
            while len(all_issues) < 200:
                resp = await client.get(
                    f"{base_url}/rest/api/3/search/jql",
                    params={
                        "jql": jql,
                        "fields": "summary,worklog",
                        "startAt": start_at,
                        "maxResults": 50,
                    },
                    auth=auth,
                )
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("issues", [])
                all_issues.extend(batch)
                # search/jql may omit "total" — use isLast flag when present
                is_last = data.get("isLast")
                total = data.get("total")
                if not batch:
                    break
                if is_last is not None:
                    if is_last:
                        break
                elif total is not None and len(all_issues) >= total:
                    break
                start_at += 50

            logger.info(
                "[jira_get_worklogs_by_date_range] project=%s %s→%s — %d issues",
                project_key, start_date, end_date, len(all_issues),
            )

            # ── Step 2: process embedded worklogs; fetch more when total > 20 ──
            for issue in all_issues:
                key = issue["key"]
                fields = issue.get("fields") or {}
                summary = fields.get("summary", "")[:80]
                wl_block = fields.get("worklog") or {}
                embedded = wl_block.get("worklogs", [])
                wl_total = wl_block.get("total", len(embedded))

                for wl in embedded:
                    _accumulate(user_totals, key, summary, wl)

                # If there are more worklogs than the 20 returned, fetch the rest
                if wl_total > len(embedded):
                    wl_start = len(embedded)
                    while wl_start < wl_total:
                        wl_resp = await client.get(
                            f"{base_url}/rest/api/3/issue/{key}/worklog",
                            params={
                                "startAt": wl_start,
                                "maxResults": 100,
                                "startedAfter": started_after_ms,
                                "startedBefore": started_before_ms,
                            },
                            auth=auth,
                        )
                        wl_resp.raise_for_status()
                        wl_data = wl_resp.json()
                        more = wl_data.get("worklogs", [])
                        for wl in more:
                            _accumulate(user_totals, key, summary, wl)
                        if not more:
                            break
                        wl_start += len(more)

    except httpx.HTTPStatusError as exc:
        return {"error": f"Jira API {exc.response.status_code}: {exc.response.text[:300]}"}
    except Exception as exc:
        return {"error": str(exc)}

    if not user_totals:
        hint = ""
        if len(all_issues) > 0:
            hint = (
                f" {len(all_issues)} issue(s) matched the date filter but contained no worklogs "
                f"in that period. The worklogs may be logged under a different date range — "
                f"try broadening the date range (e.g. 2024-01-01 to today)."
            )
        return {
            "project": project_key,
            "from": start_date,
            "to": end_date,
            "issues_scanned": len(all_issues),
            "summary": f"No worklogs found for {project_key} between {start_date} and {end_date}.{hint}",
            "users": [],
        }

    # ── Step 3: format result ──
    users_out = []
    for name, data in sorted(user_totals.items(), key=lambda x: -x[1]["total_seconds"]):
        issue_list = [
            {
                "key": k,
                "summary": v["summary"],
                "hours": round(v["seconds"] / 3600, 2),
                "formatted": _seconds_to_hm(v["seconds"]),
            }
            for k, v in sorted(data["issues"].items(), key=lambda x: -x[1]["seconds"])
        ]
        users_out.append({
            "member": name,
            "total_hours": round(data["total_seconds"] / 3600, 2),
            "total_formatted": _seconds_to_hm(data["total_seconds"]),
            "issues": issue_list,
        })

    logger.info(
        "[jira_get_worklogs_by_date_range] → %d members, top: %s (%s)",
        len(users_out),
        users_out[0]["member"] if users_out else "—",
        users_out[0]["total_formatted"] if users_out else "0h",
    )

    return {
        "project": project_key,
        "from": start_date,
        "to": end_date,
        "member_filter": member_name or "all",
        "issues_scanned": len(all_issues),
        "users": users_out,
    }


@register_tool(
    name="jira_get_project_fields",
    description=(
        "Get available fields for a Jira project and issue type. "
        "Returns which fields exist (priority, story_points, components, sprint, labels, etc.) "
        "and which are required. Call this BEFORE creating an issue to know what to ask the user."
    ),
)
async def jira_get_project_fields(
    project_key: str,
    issue_type: str = "Task",
) -> dict:
    """Fetch create-metadata for a project/issue-type to discover available fields."""
    auth = _jira_auth()
    if not auth:
        return {"error": "Jira not configured"}

    from dqe_agent.config import settings
    base_url = settings.jira_url.rstrip("/")

    async with httpx.AsyncClient(timeout=20) as client:
        # Jira Cloud createmeta endpoint
        resp = await client.get(
            f"{base_url}/rest/api/3/issue/createmeta",
            params={
                "projectKeys": project_key,
                "issuetypeNames": issue_type,
                "expand": "projects.issuetypes.fields",
            },
            auth=auth,
        )
        if resp.status_code == 404:
            # Jira Cloud newer API
            resp = await client.get(
                f"{base_url}/rest/api/3/issue/createmeta/{project_key}/issuetypes",
                auth=auth,
            )

        resp.raise_for_status()
        data = resp.json()

    # Parse out fields from createmeta response
    fields_out: dict[str, dict] = {}

    projects = data.get("projects", [])
    if projects:
        for issue_type_data in projects[0].get("issuetypes", []):
            if issue_type_data.get("name", "").lower() == issue_type.lower():
                for field_id, field_info in issue_type_data.get("fields", {}).items():
                    name = field_info.get("name", field_id)
                    required = field_info.get("required", False)
                    schema = field_info.get("schema", {})
                    allowed = field_info.get("allowedValues", [])

                    # Only surface fields useful for creation (skip system internals)
                    _SKIP = {"issuetype", "project", "reporter", "summary", "description",
                              "attachment", "issuelinks", "subtasks", "creator", "created",
                              "updated", "status", "resolution", "resolutiondate", "votes",
                              "watches", "workratio", "lastViewed", "timespent", "progress",
                              "aggregateprogress", "comment", "worklog", "changelog"}
                    if field_id in _SKIP:
                        continue

                    field_entry: dict = {
                        "id": field_id,
                        "name": name,
                        "required": required,
                        "type": schema.get("type", "string"),
                    }
                    if allowed:
                        field_entry["options"] = [
                            v.get("name") or v.get("value") or str(v)
                            for v in allowed[:30]
                        ]
                    fields_out[field_id] = field_entry

    # Fallback: parse newer API format
    if not fields_out and "fields" in data:
        for f in data.get("fields", []):
            field_id = f.get("fieldId", "")
            _SKIP = {"issuetype", "project", "reporter", "summary", "description"}
            if field_id in _SKIP:
                continue
            field_entry = {
                "id": field_id,
                "name": f.get("name", field_id),
                "required": f.get("required", False),
                "type": f.get("schema", {}).get("type", "string"),
            }
            allowed = f.get("allowedValues", [])
            if allowed:
                field_entry["options"] = [
                    v.get("name") or v.get("value") or str(v)
                    for v in allowed[:30]
                ]
            fields_out[field_id] = field_entry

    # Categorise for easy planner consumption
    has_priority = "priority" in fields_out
    has_story_points = any(k in fields_out for k in ("story_points", "customfield_10016", "customfield_10028", "storyPoints"))
    has_sprint = any(k in fields_out for k in ("sprint", "customfield_10020"))
    has_components = "components" in fields_out
    has_labels = "labels" in fields_out
    has_fix_versions = "fixVersions" in fields_out

    logger.info(
        "[jira_get_project_fields] %s/%s → %d fields (priority=%s, sp=%s, sprint=%s)",
        project_key, issue_type, len(fields_out), has_priority, has_story_points, has_sprint,
    )

    return {
        "project_key": project_key,
        "issue_type": issue_type,
        "fields": fields_out,
        "supports": {
            "priority": has_priority,
            "story_points": has_story_points,
            "sprint": has_sprint,
            "components": has_components,
            "labels": has_labels,
            "fix_versions": has_fix_versions,
        },
    }


@register_tool(
    name="jira_add_attachment",
    description=(
        "Upload a file attachment to an existing Jira issue. "
        "Provide the issue_key and the local file_path to upload. "
        "Returns the attachment URL on success."
    ),
)
async def jira_add_attachment(
    issue_key: str,
    file_path: str,
) -> dict:
    """Upload a local file to a Jira issue as an attachment."""
    import os
    auth = _jira_auth()
    if not auth:
        return {"error": "Jira not configured"}

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    from dqe_agent.config import settings
    base_url = settings.jira_url.rstrip("/")
    file_name = os.path.basename(file_path)

    async with httpx.AsyncClient(timeout=60) as client:
        with open(file_path, "rb") as f:
            resp = await client.post(
                f"{base_url}/rest/api/3/issue/{issue_key}/attachments",
                headers={"X-Atlassian-Token": "no-check"},
                auth=auth,
                files={"file": (file_name, f, "application/octet-stream")},
            )
        resp.raise_for_status()
        result = resp.json()

    attachments = result if isinstance(result, list) else [result]
    urls = [a.get("content", "") for a in attachments if a.get("content")]

    logger.info("[jira_add_attachment] %s ← %s ✅", issue_key, file_name)
    return {
        "issue_key": issue_key,
        "file_name": file_name,
        "attachment_url": urls[0] if urls else "",
        "status": "uploaded",
    }
