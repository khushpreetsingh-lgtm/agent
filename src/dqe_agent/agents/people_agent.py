from dqe_agent.agents import AgentConfig, register_agent


class PeopleAgent(AgentConfig):
    agent_id = "people"
    description = "People and team — look up users, org chart, contacts, team membership"
    domains = [
        "people", "user", "member", "team", "org", "contact",
        "who", "person", "colleague", "employee", "directory",
    ]
    tools = [
        "jira_get_assignable_users",
        "jira_search_user_by_email",
        "jira_get_project_roles",
        "jira_add_project_member",
        "ask_user",
        "request_selection",
        "direct_response",
        "request_form",
    ]
    system_prompt = (
        "You are a People agent. You look up team members, find users by email, "
        "manage project roles, and answer questions about who is on which team."
    )
    proactive = None


register_agent(PeopleAgent())
