from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent


class EmailAgent(AgentConfig):
    agent_id = "email"
    description = "Gmail — send, read, reply, draft, search emails"
    domains = [
        "email", "mail", "send", "inbox", "reply", "draft", "gmail",
        "compose", "forward", "attachment", "thread", "unread",
    ]
    tools = [
        "ask_user",
        "request_selection",
        "human_review",
        "direct_response",
        "request_form",
        "request_edit",
        "llm_draft_content",
    ]
    system_prompt = (
        "You are an Email agent. You handle Gmail operations: send emails, "
        "read threads, reply, draft messages, and search the inbox. "
        "Always show a draft for human review before sending."
    )
    proactive = ProactiveConfig(
        interval_seconds=900,
        prompt=(
            "Check for unread emails that require a response and have been waiting "
            "more than 4 hours. Report sender, subject, and a one-line summary."
        ),
    )


register_agent(EmailAgent())
