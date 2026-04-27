from dqe_agent.agents import AgentConfig, register_agent


class BrowserAgent(AgentConfig):
    agent_id = "browser"
    description = "Browser automation — navigate, click, fill forms, extract data from any website"
    domains = [
        "browser", "web", "navigate", "login", "click", "fill", "scrape",
        "netsuite", "cpq", "website", "page", "screenshot", "extract",
    ]
    tools = None  # None = all tools — browser agent is the catch-all fallback
    system_prompt = (
        "You are a browser automation agent. You control a real Chromium browser "
        "to interact with any website: log in, navigate, fill forms, extract data, "
        "and complete multi-step workflows. Always login before interacting with a site."
    )
    proactive = None


register_agent(BrowserAgent())
