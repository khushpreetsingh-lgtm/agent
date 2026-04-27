import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_classify_single_domain():
    from dqe_agent.agent.orchestrator import classify_domains
    from dqe_agent.agents import register_agent, AgentConfig
    from langchain_core.messages import HumanMessage, AIMessage
    from unittest.mock import AsyncMock, patch

    class _J(AgentConfig):
        agent_id = "jira"
        description = "Jira ticket management"
        domains = ["jira", "ticket", "issue"]
        tools = ["direct_response"]
        system_prompt = ""
    register_agent(_J())

    messages = [HumanMessage(content="Create a Jira ticket for the login bug")]
    agent_descriptions = {"jira": "Jira ticket management", "browser": "Web navigation"}

    # Mock LLM response
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"agents": ["jira"]}'))

    with patch("dqe_agent.llm.get_planner_llm", return_value=mock_llm):
        domains = await classify_domains("Create a Jira ticket for the login bug", messages, agent_descriptions)
        assert "jira" in domains


@pytest.mark.asyncio
async def test_classify_multi_domain():
    from dqe_agent.agent.orchestrator import classify_domains
    from langchain_core.messages import HumanMessage, AIMessage
    from unittest.mock import AsyncMock, patch

    messages = [HumanMessage(content="Create a ticket and schedule a meeting")]
    agent_descriptions = {
        "jira": "Jira tickets and issues",
        "calendar": "Calendar events and meetings",
        "browser": "Web navigation"
    }

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"agents": ["jira", "calendar"]}'))

    with patch("dqe_agent.llm.get_planner_llm", return_value=mock_llm):
        domains = await classify_domains("Create a ticket and schedule a meeting", messages, agent_descriptions)
        assert "jira" in domains
        assert "calendar" in domains


@pytest.mark.asyncio
async def test_classify_no_match_returns_browser():
    from dqe_agent.agent.orchestrator import classify_domains
    from langchain_core.messages import HumanMessage, AIMessage
    from unittest.mock import AsyncMock, patch

    messages = [HumanMessage(content="Do something weird and unusual xyz")]
    agent_descriptions = {"jira": "Jira tickets", "browser": "Web navigation"}

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"agents": ["browser"]}'))

    with patch("dqe_agent.llm.get_planner_llm", return_value=mock_llm):
        domains = await classify_domains("Do something weird and unusual xyz", messages, agent_descriptions)
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


@pytest.mark.asyncio
async def test_proactive_monitor_runs_and_stops():
    import asyncio
    from dqe_agent.agent.orchestrator import ProactiveMonitor

    alerts = []

    async def fake_broadcast(msg):
        alerts.append(msg)

    monitor = ProactiveMonitor(broadcast_fn=fake_broadcast)
    task = asyncio.create_task(monitor.start())
    await asyncio.sleep(0.05)
    monitor.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
    # Just verify it started and stopped without error
    assert monitor._stopped
