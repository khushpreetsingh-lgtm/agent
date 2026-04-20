"""Master Agent — unified PEV graph for ALL interactions.

Every user message goes through: Planner → Executor → Verifier.
Simple chat messages get a single "direct_response" step from the planner.
No separate ReAct graph — one path for everything.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from dqe_agent.state import AgentState

logger = logging.getLogger(__name__)

_CHECKPOINT_DB = Path("data/checkpoints.db")


class MasterAgent:
    """Unified PEV agent — planner→executor→verifier for all messages."""

    def __init__(self) -> None:
        _CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        self._aio_conn: aiosqlite.Connection | None = None
        self.checkpointer: AsyncSqliteSaver | None = None
        self._graph: Any = None
        logger.info("MasterAgent created — call await setup() to initialise")

    async def setup(self) -> None:
        """Open async SQLite connection and compile graph."""
        self._aio_conn = await aiosqlite.connect(str(_CHECKPOINT_DB))
        self.checkpointer = AsyncSqliteSaver(self._aio_conn)
        await self.checkpointer.setup()
        self._build_graph()
        logger.info("MasterAgent ready (unified PEV) with AsyncSqliteSaver")

    def _build_graph(self) -> None:
        from dqe_agent.agent.loop import build_pev_graph
        graph = build_pev_graph()
        self._graph = graph.compile(checkpointer=self.checkpointer)
        logger.info("Compiled unified PEV graph")

    def get_app(self) -> Any:
        """Return the compiled PEV graph."""
        if self._graph is None:
            self._build_graph()
        return self._graph

    async def reset(self, thread_id: str = "default") -> None:
        """Clear checkpoint rows for a specific thread without touching other sessions."""
        logger.info("Reset thread: %s", thread_id)
        if self._aio_conn:
            try:
                # Delete only this thread's rows — other sessions are unaffected.
                await self._aio_conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                )
                await self._aio_conn.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,)
                )
                await self._aio_conn.commit()
                logger.info("Cleared checkpoints for thread %s", thread_id)
            except Exception as exc:
                logger.warning("Could not clear checkpoints for thread %s: %s", thread_id, exc)
