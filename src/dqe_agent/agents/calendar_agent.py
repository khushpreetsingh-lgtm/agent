from dqe_agent.agents import AgentConfig, ProactiveConfig, register_agent


class CalendarAgent(AgentConfig):
    agent_id = "calendar"
    description = "Google Calendar — events, meetings, scheduling, availability"
    domains = [
        "calendar", "meeting", "schedule", "event", "invite", "availability",
        "slot", "appointment", "standup", "sync", "reminder", "recurring",
    ]
    tools = [
        "ask_user",
        "request_selection",
        "human_review",
        "direct_response",
        "request_form",
    ]
    system_prompt = (
        "You are a Calendar agent. You manage Google Calendar events: "
        "create meetings, check availability, find free slots, update or cancel events, "
        "and add attendees. Always show available time slots via request_selection."
    )
    proactive = ProactiveConfig(
        interval_seconds=600,
        prompt=(
            "Check for: (1) meetings starting within 30 minutes with no agenda set, "
            "(2) back-to-back meetings with no break. "
            "Report only if something actionable is found."
        ),
    )


register_agent(CalendarAgent())
