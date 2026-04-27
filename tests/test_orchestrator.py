import pytest
from unittest.mock import AsyncMock, patch


def test_classify_single_domain():
    from dqe_agent.agent.orchestrator import classify_domains
    from dqe_agent.agents import register_agent, AgentConfig

    class _J(AgentConfig):
        agent_id = "jira"
        domains = ["jira", "ticket", "issue"]
        tools = ["direct_response"]
        system_prompt = ""
    register_agent(_J())

    domains = classify_domains("Create a Jira ticket for the login bug", {"jira": "jira"})
    assert "jira" in domains


def test_classify_multi_domain():
    from dqe_agent.agent.orchestrator import classify_domains

    idx = {"jira": "jira", "ticket": "jira", "meeting": "calendar", "schedule": "calendar"}
    domains = classify_domains("Create a ticket and schedule a meeting", idx)
    assert "jira" in domains
    assert "calendar" in domains


def test_classify_no_match_returns_browser():
    from dqe_agent.agent.orchestrator import classify_domains
    domains = classify_domains("Do something weird and unusual xyz", {})
    assert domains == ["browser"]


def test_decompose_tasks_single():
    from dqe_agent.agent.orchestrator import decompose_tasks
    tasks = decompose_tasks("Create a Jira ticket", ["jira"])
    assert len(tasks) == 1
    assert tasks[0]["agent"] == "jira"
    assert "Create a Jira ticket" in tasks[0]["task"]


def test_decompose_tasks_multi():
    from dqe_agent.agent.orchestrator import decompose_tasks
    tasks = decompose_tasks("Create a Jira ticket and schedule a meeting", ["jira", "calendar"])
    assert len(tasks) == 2
    agent_ids = {t["agent"] for t in tasks}
    assert "jira" in agent_ids
    assert "calendar" in agent_ids


def test_build_orchestrator_graph():
    from dqe_agent.agent.orchestrator import build_orchestrator_graph
    graph = build_orchestrator_graph()
    assert graph is not None
    # Should compile without error
    compiled = graph.compile()
    assert compiled is not None
