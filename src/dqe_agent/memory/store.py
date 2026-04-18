"""SQLite-based task store — persists tasks, logs, and results (replaces PostgreSQL)."""
from __future__ import annotations
import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path("data/tasks.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id        TEXT PRIMARY KEY,
            session_id     TEXT NOT NULL,
            workflow       TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'pending',
            state_json     TEXT,
            result_json    TEXT,
            error          TEXT,
            steps_taken    INTEGER DEFAULT 0,
            created_at     REAL NOT NULL,
            updated_at     REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS task_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT NOT NULL,
            step        TEXT,
            level       TEXT DEFAULT 'info',
            message     TEXT,
            data_json   TEXT,
            ts          REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tool_cache (
            cache_key   TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at  REAL NOT NULL,
            ttl_seconds INTEGER DEFAULT 300
        );

        CREATE INDEX IF NOT EXISTS idx_task_session ON tasks(session_id);
        CREATE INDEX IF NOT EXISTS idx_logs_task    ON task_logs(task_id);
    """)
    conn.commit()

    # Migrate: add new columns to tasks table if they don't exist yet
    for col, defn in [
        ("title",          "TEXT"),
        ("task_type",      "TEXT DEFAULT 'generic'"),
        ("source",         "TEXT DEFAULT 'chat'"),
        ("result_summary", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.close()
    logger.info("TaskStore DB initialised at %s", DB_PATH)


class TaskStore:
    """Synchronous SQLite-backed store — wraps calls in a thread executor for async use."""

    def __init__(self) -> None:
        init_db()
        self._conn = _get_conn()
        self._lock = asyncio.Lock()

    # ── Tasks ──────────────────────────────────────────────────────────────

    def create_task(
        self,
        task_id: str,
        session_id: str,
        workflow: str,
        initial_state: dict | None = None,
        *,
        title: str = "",
        source: str = "chat",
        task_type: str = "generic",
    ) -> None:
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, session_id, workflow, status, state_json, created_at, updated_at,
                title, task_type, source)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (task_id, session_id, workflow, "pending", json.dumps(initial_state or {}),
             now, now, title, task_type, source),
        )
        self._conn.commit()

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        state: dict | None = None,
        result: dict | None = None,
        error: str | None = None,
        steps_taken: int | None = None,
        result_summary: str | None = None,
        task_type: str | None = None,
    ) -> None:
        fields, vals = [], []
        if status is not None:
            fields.append("status=?"); vals.append(status)
        if state is not None:
            fields.append("state_json=?"); vals.append(json.dumps(state))
        if result is not None:
            fields.append("result_json=?"); vals.append(json.dumps(result))
        if error is not None:
            fields.append("error=?"); vals.append(error)
        if steps_taken is not None:
            fields.append("steps_taken=?"); vals.append(steps_taken)
        if result_summary is not None:
            fields.append("result_summary=?"); vals.append(result_summary)
        if task_type is not None:
            fields.append("task_type=?"); vals.append(task_type)
        if not fields:
            return
        fields.append("updated_at=?"); vals.append(time.time())
        vals.append(task_id)
        self._conn.execute(f"UPDATE tasks SET {','.join(fields)} WHERE task_id=?", vals)
        self._conn.commit()

    def get_task(self, task_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["state"] = json.loads(d.pop("state_json") or "{}")
        d["result"] = json.loads(d.pop("result_json") or "null")
        return d

    def get_pending_tasks(self) -> list[dict]:
        """Return tasks that were interrupted — used for resume on startup."""
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status IN ('pending','running') ORDER BY created_at"
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["state"] = json.loads(d.pop("state_json") or "{}")
            d["result"] = json.loads(d.pop("result_json") or "null")
            out.append(d)
        return out

    # ── Logs ───────────────────────────────────────────────────────────────

    def log(self, task_id: str, message: str, *, step: str = "", level: str = "info", data: dict | None = None) -> None:
        self._conn.execute(
            "INSERT INTO task_logs (task_id, step, level, message, data_json, ts) VALUES (?,?,?,?,?,?)",
            (task_id, step, level, message, json.dumps(data) if data else None, time.time()),
        )
        self._conn.commit()

    def get_logs(self, task_id: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM task_logs WHERE task_id=? ORDER BY ts DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Tool result cache ──────────────────────────────────────────────────

    def cache_get(self, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT result_json, created_at, ttl_seconds FROM tool_cache WHERE cache_key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        if time.time() - row["created_at"] > row["ttl_seconds"]:
            self._conn.execute("DELETE FROM tool_cache WHERE cache_key=?", (key,))
            self._conn.commit()
            return None
        return json.loads(row["result_json"])

    def cache_set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tool_cache (cache_key, result_json, created_at, ttl_seconds) VALUES (?,?,?,?)",
            (key, json.dumps(value), time.time(), ttl_seconds),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# Global singleton
task_store = TaskStore()
