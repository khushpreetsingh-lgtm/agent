"""Tool → human-readable progress labels for frontend UX.

Shared between executor.py (step messages) and api.py (WebSocket status events).
Each entry: (phase, in_progress_label, done_label)
  phase          — category string the frontend can use for icons/colours
  in_progress    — shown WHILE the step is running
  done           — shown AFTER the step completes successfully
"""
from __future__ import annotations

# ── Master mapping ────────────────────────────────────────────────────────────
# (phase, in_progress_label, done_label)
# in_progress → shown while the step is running  (natural sentence)
# done        → shown after success               (natural past tense)
# fmt: off
_MAP: dict[str, tuple[str, str, str]] = {

    # ── Interaction tools — frontend handles the UI; no text needed ──────────
    "ask_user":           ("waiting",  "Just a quick question for you...",         ""),
    "request_selection":  ("waiting",  "Please pick one of the options below...",  ""),
    "request_edit":       ("waiting",  "Here's a draft — feel free to edit it...", ""),
    "human_review":       ("waiting",  "Please review and approve to continue...", ""),
    "direct_response":    ("done",     "",                                         ""),

    # ── AI generation ────────────────────────────────────────────────────────
    "llm_draft_content":  ("thinking", "Writing a draft for you...",               "Draft is ready for your review"),

    # ── Jira reads ───────────────────────────────────────────────────────────
    "jira_search":                  ("reading",  "Pulling up your issues...",                "Here are the results"),
    "jira_get_issue":               ("reading",  "Fetching the issue details for you...",    "Got the issue details"),
    "jira_get_agile_boards":        ("reading",  "Loading the available boards...",          "Boards are ready"),
    "jira_get_boards":              ("reading",  "Loading the available boards...",          "Boards are ready"),
    "jira_get_sprints":             ("reading",  "Fetching the sprints for you...",          "Sprints are ready"),
    "jira_list_sprints":            ("reading",  "Fetching the sprints for you...",          "Sprints are ready"),
    "jira_get_active_sprints":      ("reading",  "Checking the active sprint...",            "Got the active sprint info"),
    "jira_search_users":            ("reading",  "Looking up the team member...",            "Found the team member"),
    "jira_get_transitions":         ("reading",  "Fetching the available statuses...",       "Got the available statuses"),
    "jira_get_all_projects":        ("reading",  "Loading your Jira projects...",            "Projects are ready"),
    "jira_search_projects":         ("reading",  "Searching for the project...",             "Found the project"),
    "jira_get_issue_types":         ("reading",  "Loading the issue types...",               "Issue types are ready"),
    "jira_get_project_components":  ("reading",  "Loading project components...",            "Components are ready"),
    "jira_get_link_types":          ("reading",  "Fetching the link types...",               "Link types are ready"),
    "jira_get_worklogs":            ("reading",  "Fetching the time logs...",                "Got the time logs"),

    # ── Jira writes ──────────────────────────────────────────────────────────
    "jira_create_issue":            ("creating", "Creating your issue...",                   "Your issue has been created"),
    "jira_create_sprint":           ("creating", "Creating your sprint...",                  "Your sprint has been created"),
    "jira_update_issue":            ("updating", "Updating your issue...",                   "Your issue has been updated"),
    "jira_assign_issue":            ("updating", "Assigning the issue for you...",           "Issue assigned successfully"),
    "jira_transition_issue":        ("updating", "Moving the issue to the new status...",    "Status has been updated"),
    "jira_add_comment":             ("creating", "Adding your comment...",                   "Your comment has been added"),
    "jira_add_worklog":             ("creating", "Logging your time...",                     "Time logged successfully"),
    "jira_create_issue_link":       ("creating", "Linking the issues together...",           "Issues are now linked"),
    "jira_rank_backlog_issues":     ("updating", "Moving the issue to the sprint...",        "Issue moved to the sprint"),
    "jira_update_sprint":           ("updating", "Updating your sprint...",                  "Sprint has been updated"),
    "jira_delete_issue":            ("updating", "Deleting the issue...",                    "Issue has been deleted"),

    # ── Gmail reads ──────────────────────────────────────────────────────────
    "search_gmail_messages":             ("reading", "Searching through your emails...",         "Found your emails"),
    "get_gmail_message_content":         ("reading", "Opening the email for you...",             "Email is ready"),
    "get_gmail_messages_content_batch":  ("reading", "Loading your emails...",                   "Emails are ready"),
    "get_gmail_thread_content":          ("reading", "Loading the full conversation...",         "Conversation is ready"),
    "get_total_unread_emails":           ("reading", "Counting your unread emails...",           "Got your unread count"),
    "list_unread_message_ids":           ("reading", "Fetching your unread messages...",         "Messages fetched"),
    "get_label_metrics":                 ("reading", "Checking your inbox stats...",             "Got your inbox stats"),

    # ── Gmail writes ─────────────────────────────────────────────────────────
    "send_gmail_message":                ("sending", "Sending your email...",                    "Your email has been sent"),
    "draft_gmail_message":               ("sending", "Saving your draft...",                     "Your draft has been saved"),
    "modify_gmail_message_labels":       ("updating","Updating the email labels...",             "Email updated"),
    "batch_modify_gmail_message_labels": ("updating","Updating your emails...",                  "Emails have been updated"),
    "batch_apply_labels_to_all":         ("updating","Processing all your emails...",            "All done"),

    # ── Google Calendar reads ─────────────────────────────────────────────────
    "get_events":         ("reading", "Checking your calendar...",               "Got your calendar events"),
    "list_events":        ("reading", "Checking your calendar...",               "Got your calendar events"),
    "query_freebusy":     ("reading", "Checking when you are free...",           "Got your availability"),
    "list_calendars":     ("reading", "Loading your calendars...",               "Calendars are ready"),
    "get_calendars":      ("reading", "Loading your calendars...",               "Calendars are ready"),

    # ── Google Calendar writes ────────────────────────────────────────────────
    "manage_event":       ("creating", "Saving the event to your calendar...",   "Event added to your calendar"),

    # ── Browser ──────────────────────────────────────────────────────────────
    "browser_login":      ("browser", "Logging you in...",                       "Logged in successfully"),
    "browser_navigate":   ("browser", "Opening the page...",                     "Page is ready"),
    "browser_act":        ("browser", "Working on the page for you...",          "Done"),
    "browser_extract":    ("browser", "Reading the information from the page...", "Information extracted"),
    "browser_snapshot":   ("browser", "Taking a screenshot...",                  "Screenshot captured"),
    "browser_click":      ("browser", "Clicking...",                             "Done"),
    "browser_type":       ("browser", "Filling in the details...",               "Done"),
    "browser_wait":       ("browser", "Waiting for the page to load...",         "Page is ready"),
}
# fmt: on

# Tools whose step messages should be completely silent (they drive their own UI)
SILENT_TOOLS: frozenset[str] = frozenset({
    "ask_user", "request_selection", "request_edit", "human_review",
    "direct_response",
})


def tool_phase(tool: str) -> str:
    """Return the phase category for a tool (reading / creating / updating / etc.)."""
    return _MAP.get(tool, ("working", "", ""))[0]


def tool_in_progress(tool: str) -> str:
    """Return the 'currently running' label for a tool.

    Returns empty string for silent / unknown tools.
    """
    entry = _MAP.get(tool)
    if entry:
        return entry[1]
    if tool in SILENT_TOOLS:
        return ""
    # Fallback: humanise the tool name
    return tool.replace("jira_", "").replace("gmail_", "").replace("_", " ").title() + "..."


def tool_done(tool: str) -> str:
    """Return the completion label for a tool.

    Returns empty string for silent tools (no completion message needed).
    """
    entry = _MAP.get(tool)
    if entry:
        return entry[2]
    if tool in SILENT_TOOLS:
        return ""
    return tool.replace("jira_", "").replace("gmail_", "").replace("_", " ").title()
