from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent


class JiraAgent(AgentConfig):
    agent_id = "jira"
    description = "Jira project management — issues, sprints, boards, epics, worklogs"
    domains = [
        "jira", "ticket", "issue", "sprint", "board", "epic", "story",
        "bug", "task", "backlog", "worklog", "hours", "logged", "assignee",
        "priority", "transition", "status", "component", "version", "label",
    ]
    tools = [
        "jira_get_assignable_users",
        "jira_get_priorities",
        "jira_get_project_roles",
        "jira_search_user_by_email",
        "jira_add_project_member",
        "jira_get_worklogs_by_date_range",
        "jira_get_project_fields",
        "jira_add_attachment",
        "ask_user",
        "request_selection",
        "human_review",
        "direct_response",
        "request_form",
        "request_edit",
        "llm_draft_content",
    ]
    system_prompt = (
        "You are a Jira agent. You handle all Jira project management tasks: "
        "create and update issues, manage sprints, assign work, log time, "
        "search across projects, and report on team progress. "
        "Use MCP Jira tools for all operations — never use browser tools for Jira."
    )
    proactive = ProactiveConfig(
        interval_seconds=300,
        prompt=(
            "Check for: (1) issues assigned to team members that are overdue, "
            "(2) active sprints ending within 2 days with unresolved blockers, "
            "(3) unassigned high-priority issues in active sprints. "
            "Report only if something actionable is found."
        ),
    )


register_agent(JiraAgent())
