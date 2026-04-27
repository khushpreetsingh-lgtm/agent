import pytest
from dqe_agent.agents import (
    AgentConfig, ProactiveConfig, register_agent,
    get_agent, list_agents, discover_agents, build_domain_index, clear_registry,
)


@pytest.fixture(autouse=True)
def _clear_agent_registry():
    from dqe_agent.agents import clear_registry
    clear_registry()
    yield
    clear_registry()


class _TestAgent(AgentConfig):
    agent_id = "test_agent"
    description = "Test agent"
    domains = ["test", "sample", "demo"]
    tools = ["direct_response"]
    system_prompt = "Test prompt."


def test_register_and_get():
    register_agent(_TestAgent())
    agent = get_agent("test_agent")
    assert agent.agent_id == "test_agent"


def test_list_agents_includes_registered():
    register_agent(_TestAgent())
    assert "test_agent" in list_agents()


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_agent("nonexistent_agent_xyz")


def test_build_domain_index():
    register_agent(_TestAgent())
    idx = build_domain_index()
    assert idx["test"] == "test_agent"
    assert idx["sample"] == "test_agent"
    assert idx["demo"] == "test_agent"


def test_proactive_config():
    pc = ProactiveConfig(interval_seconds=60, prompt="Check for issues")
    assert pc.interval_seconds == 60
    assert pc.prompt == "Check for issues"
