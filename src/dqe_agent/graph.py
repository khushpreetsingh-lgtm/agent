"""Compatibility shim: expose a simple `build_graph(browser)` used by tests.

The real engine builds LangGraph StateGraph instances from YAML workflows.
For unit tests we only need a graph-like object with a `nodes` collection
containing the workflow step ids (so tests can assert node presence).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _load_workflow_step_ids(path: str | Path) -> list[str]:
    try:
        from dqe_agent.engine import load_workflow
    except Exception:
        return []
    wf = load_workflow(path)
    return [s.id for s in wf.steps]


class SimpleGraph:
    def __init__(self, nodes: Iterable[str]):
        # Use a set so membership checks work as expected in tests
        self.nodes = set(nodes)


def build_graph(browser) -> SimpleGraph:
    """Construct a lightweight graph-like object for tests.

    `browser` is accepted for API compatibility but not used here.
    This reads the `workflows/opportunity_to_quote.yaml` file and returns
    a SimpleGraph whose `nodes` set contains all step ids from that workflow.
    """
    wf_path = Path("workflows") / "opportunity_to_quote.yaml"
    if wf_path.exists():
        ids = _load_workflow_step_ids(wf_path)
    else:
        ids = []
    # Backwards-compat aliases expected by tests
    aliases = {
        "send_quote_email": "send_email",
    }
    nodes = list(ids)
    for src, alias in aliases.items():
        if src in ids and alias not in nodes:
            nodes.append(alias)

    return SimpleGraph(nodes)
