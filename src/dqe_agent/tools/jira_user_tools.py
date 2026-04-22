"""Jira user-lookup tools.

Provides assignable-user listing for a Jira project so the planner can
present real user options via request_selection instead of asking for a
free-text name.

Calls the Jira REST API directly using the credentials in settings.
"""
from __future__ import annotations

import logging
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


@register_tool(
    "jira_get_assignable_users",
    "Fetch the list of users who can be assigned issues in a Jira project. "
    "Returns a list of {value, label} options ready for request_selection. "
    "Use this before showing an assignee selection to the user.",
)
async def jira_get_assignable_users(project_key: str) -> list[dict[str, str]]:
    """Return assignable users for a Jira project as [{value, label}] options."""
    from dqe_agent.config import settings

    auth = _jira_auth()
    if not auth or not settings.jira_url:
        logger.warning("[jira_get_assignable_users] Jira not configured — returning empty list")
        return []

    base_url = settings.jira_url.rstrip("/")
    url = f"{base_url}/rest/api/3/user/assignable/search"
    params: dict[str, Any] = {"project": project_key, "maxResults": 50}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, auth=auth)
            resp.raise_for_status()
            users: list[dict] = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "[jira_get_assignable_users] HTTP %s from Jira — %s",
            exc.response.status_code, exc.response.text[:200],
        )
        return []
    except Exception as exc:
        logger.warning("[jira_get_assignable_users] Failed: %s", exc)
        return []

    options: list[dict[str, str]] = []
    for u in users:
        account_id = u.get("accountId", "")
        display_name = u.get("displayName", account_id)
        if not account_id or u.get("accountType") == "app":
            continue  # skip bot/service accounts
        options.append({"value": account_id, "label": display_name})

    logger.info(
        "[jira_get_assignable_users] project=%s → %d users", project_key, len(options)
    )
    return options


@register_tool(
    "jira_get_priorities",
    "Fetch the list of valid priority names configured in this Jira instance. "
    "Returns a list of {value, label} options ready for request_selection. "
    "ALWAYS call this before showing a priority selection — never hardcode priority names "
    "because different Jira projects use different schemes (e.g. P1/P2/P3 vs High/Medium/Low).",
)
async def jira_get_priorities() -> list[dict[str, str]]:
    """Return all Jira priorities as [{value, label}] options."""
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
    url = f"{base_url}/rest/api/3/priority"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, auth=auth)
            resp.raise_for_status()
            priorities: list[dict] = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "[jira_get_priorities] HTTP %s — %s",
            exc.response.status_code, exc.response.text[:200],
        )
        return []
    except Exception as exc:
        logger.warning("[jira_get_priorities] Failed: %s", exc)
        return []

    options = [
        {"value": p["name"], "label": p["name"]}
        for p in priorities
        if p.get("name")
    ]
    logger.info("[jira_get_priorities] → %d priorities: %s", len(options), [o["value"] for o in options])
    return options
