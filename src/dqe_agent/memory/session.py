"""In-memory session store — per-session key/value cache (replaces Redis)."""
from __future__ import annotations
import asyncio
import time
from typing import Any

class SessionMemory:
    """Lightweight in-memory store for active session data.

    Stores arbitrary key-value data per session_id. Data is lost when the
    process restarts (use TaskStore for persistence). One global instance
    shared across all sessions.
    """
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._data.get(session_id, {}).get(key, default)

    async def set(self, session_id: str, key: str, value: Any) -> None:
        async with self._lock:
            if session_id not in self._data:
                self._data[session_id] = {}
            self._data[session_id][key] = value

    async def get_session(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            return dict(self._data.get(session_id, {}))

    async def set_session(self, session_id: str, data: dict[str, Any]) -> None:
        async with self._lock:
            if session_id not in self._data:
                self._data[session_id] = {}
            self._data[session_id].update(data)

    async def delete(self, session_id: str, key: str) -> None:
        async with self._lock:
            self._data.get(session_id, {}).pop(key, None)

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            self._data.pop(session_id, None)

    def get_sync(self, session_id: str, key: str, default: Any = None) -> Any:
        return self._data.get(session_id, {}).get(key, default)

    def set_sync(self, session_id: str, key: str, value: Any) -> None:
        if session_id not in self._data:
            self._data[session_id] = {}
        self._data[session_id][key] = value

    def get_session_sync(self, session_id: str) -> dict[str, Any]:
        return dict(self._data.get(session_id, {}))

    def clear_session_sync(self, session_id: str) -> None:
        self._data.pop(session_id, None)

# Global singleton
session_memory = SessionMemory()
