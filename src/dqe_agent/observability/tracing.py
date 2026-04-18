"""Observability — structured logging + lightweight span tracing (no external service needed).

Writes structured JSON logs to logs/traces.jsonl for local observability.
Compatible with OpenTelemetry conventions so you can switch to a real OTLP
exporter later with minimal changes.
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

TRACE_LOG = Path("logs/traces.jsonl")


def _write(record: dict) -> None:
    try:
        TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # never break the agent because of logging


class Tracer:
    def __init__(self, service: str = "dqe-agent") -> None:
        self.service = service

    @contextmanager
    def span(self, name: str, attributes: dict | None = None) -> Generator[dict, None, None]:
        span_id = str(uuid.uuid4())[:8]
        start = time.time()
        record: dict[str, Any] = {
            "span_id": span_id,
            "name": name,
            "service": self.service,
            "start": start,
            "attributes": attributes or {},
            "status": "ok",
        }
        try:
            yield record
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            raise
        finally:
            record["duration_ms"] = round((time.time() - start) * 1000, 1)
            _write(record)
            logger.debug("[trace] %s %.0fms %s", name, record["duration_ms"], record["status"])


_default_tracer = Tracer()


def get_tracer() -> Tracer:
    return _default_tracer


def trace_llm_call(
    model: str,
    role: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    session_id: str = "",
    task_id: str = "",
) -> None:
    _write({
        "type": "llm_call",
        "ts": time.time(),
        "model": model,
        "role": role,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "session_id": session_id,
        "task_id": task_id,
    })


def trace_tool_call(
    tool: str,
    args: dict,
    result_status: str,
    duration_ms: float,
    session_id: str = "",
    task_id: str = "",
) -> None:
    _write({
        "type": "tool_call",
        "ts": time.time(),
        "tool": tool,
        "args_keys": list(args.keys()),
        "result_status": result_status,
        "duration_ms": duration_ms,
        "session_id": session_id,
        "task_id": task_id,
    })
