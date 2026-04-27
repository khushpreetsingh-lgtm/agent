"""DQE Agent — Backend API

Two transports:
  WebSocket /ws/{session_id}       — live chat + browser screenshot stream
  REST      /api/v1/*              — session info, tools list, health

WebSocket message protocol
──────────────────────────
CLIENT → SERVER
  {"type": "chat",           "content": "Create a quote for OP-20080"}
  {"type": "run_task",       "task": "Log in to Jira and create ticket..."}
  {"type": "run_workflow",   "workflow": "opportunity_to_quote",
                             "inputs":   {"opportunity_id": "OP-20080"}}
  {"type": "human_response", "content": "proceed"}
  {"type": "ping"}

SERVER → CLIENT
  {"type": "connected",       "session_id": "..."}
  {"type": "agent_text",      "content": "I'll log in first..."}
  {"type": "tool_start",      "tool": "browser_login", "args": {...}}
  {"type": "tool_done",       "tool": "browser_login", "result": "logged_in"}
  {"type": "agent_done",      "content": "<full final reply>"}
  {"type": "browser_frame",   "data": "<base64>", "width": 1280, "height": 800}
  {"type": "plan_created",    "steps": [...]}
  {"type": "step_status",     "step": "...", "status": "running|done|failed"}
  {"type": "human_review",       "question": "Does this look correct? ..."}
  {"type": "selection_request",  "question": "Which sprint?",
                                 "options": [{"value": "SP-1", "label": "Sprint 23 – Active"},
                                             {"value": "SP-2", "label": "Sprint 22 – Closed"}],
                                 "multi_select": false}
  {"type": "workflow_done",      "summary": "..."}
  {"type": "error",              "message": "..."}
  {"type": "proactive_alert",    "agent": "jira", "content": "3 overdue tickets in Sprint 24"}
  {"type": "pong"}

CLIENT → SERVER (additional)
  {"type": "selection_response", "value": "SP-1"}           -- single pick
  {"type": "selection_response", "value": ["SP-1","SP-2"]}  -- multi-pick
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from dqe_agent.agent import MasterAgent
from dqe_agent.browser.manager import BrowserManager
from dqe_agent.browser.webrtc import cleanup_webrtc, handle_webrtc_offer
from dqe_agent.config import settings
from dqe_agent.tools import list_tool_names, list_tools
from dqe_agent.tools.browser_tools import set_browser

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path("workflows")


# ── Task response helpers ────────────────────────────────────────────────────

def _epoch_to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _map_status(status: str) -> str:
    """Normalise internal status values to the frontend contract."""
    return {"complete": "completed", "failed_restart": "failed"}.get(status, status)


def _infer_task_type(tool_name: str) -> str:
    t = (tool_name or "").lower()
    if "jira" in t:
        return "jira_create"
    if "email" in t or "gmail" in t or "send_email" in t:
        return "email_send"
    if "calendar" in t:
        return "calendar_event"
    if "search" in t:
        return "web_search"
    return "generic"


def _format_task(row: dict) -> dict:
    state = json.loads(row.get("state_json") or "{}")
    return {
        "id":             row["task_id"],
        "title":          row.get("title") or state.get("task", "Task"),
        "status":         _map_status(row["status"]),
        "type":           row.get("task_type") or "generic",
        "source":         row.get("source") or "chat",
        "created_at":     _epoch_to_iso(row.get("created_at")),
        "updated_at":     _epoch_to_iso(row.get("updated_at")),
        "result_summary": row.get("result_summary"),
        "error":          row.get("error"),
    }

# ── Shared singletons ───────────────────────────────────────────────────────
browser_manager: BrowserManager | None = None
master_agent: MasterAgent | None = None
_proactive_monitor_task: asyncio.Task | None = None

# ── Per-session state ────────────────────────────────────────────────────────
_human_queues: dict[str, asyncio.Queue] = {}
_awaiting_input: set[str] = set()
_session_browsers: dict[str, Any] = {}
_session_tasks: dict[str, asyncio.Task] = {}
_active_websockets: set[WebSocket] = set()


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_manager, master_agent

    # Configure structured logging
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    # Discover tools and flows
    from dqe_agent.tools import discover_tools
    discover_tools()

    from dqe_agent.flows import discover_flows
    discover_flows()

    from dqe_agent.agents import discover_agents
    discover_agents()

    # Log configured sites (from settings.sites — no Python files needed)
    from dqe_agent.config import settings as _s
    logger.info("Sites configured: %s", list(_s.sites.keys()))

    # Init task store DB
    from dqe_agent.memory.store import task_store  # noqa: F401

    # Mark interrupted tasks from previous runs
    for t in task_store.get_pending_tasks():
        logger.info("Marking interrupted task %s as failed_restart", t["task_id"])
        task_store.update_task(t["task_id"], status="failed_restart", error="Server restarted")

    # Load MCP servers (jira, gmail, brave-search, etc.)
    from dqe_agent.tools.mcp_startup import load_mcp_servers
    await load_mcp_servers()

    # ── Pre-warm ALL caches so the FIRST request is just as fast as subsequent ──
    # 1. Jira projects (used by planner for project selection)
    from dqe_agent.agent.planner import warm_cache, _prewarm_mcp_tool_block
    logger.info("Pre-warming Jira project cache...")
    await warm_cache()

    # 2. MCP tool description block (the ~30s schema-build loop for 170 tools)
    logger.info("Pre-warming MCP tool schema block...")
    await _prewarm_mcp_tool_block()

    # Start browser (skip when browser tools are disabled)
    browser_manager = BrowserManager()
    if not settings.disable_browser_tools:
        await browser_manager.start()

    # Init agent
    master_agent = MasterAgent()
    await master_agent.setup()

    tools = list_tool_names()
    logger.info("DQE Agent ready — %d tools loaded", len(tools))

    # Start proactive monitor
    global _proactive_monitor_task
    from dqe_agent.agent.orchestrator import ProactiveMonitor

    async def _broadcast_alert(msg: dict) -> None:
        for ws in _active_websockets:
            try:
                await ws.send_json(msg)
            except Exception as exc:
                logger.warning("Failed to broadcast proactive alert: %s", exc)

    monitor = ProactiveMonitor(broadcast_fn=_broadcast_alert)
    _proactive_monitor_task = asyncio.create_task(monitor.start())
    logger.info("ProactiveMonitor started")

    yield

    # Stop proactive monitor
    if _proactive_monitor_task:
        monitor.stop()
        try:
            await asyncio.wait_for(_proactive_monitor_task, timeout=2.0)
        except asyncio.TimeoutError:
            _proactive_monitor_task.cancel()
        logger.info("ProactiveMonitor stopped")

    if browser_manager:
        await browser_manager.stop()
    if master_agent and master_agent._aio_conn:
        await master_agent._aio_conn.close()
    from dqe_agent.tools.mcp_startup import stop_mcp_servers
    await stop_mcp_servers()
    logger.info("DQE Agent shut down.")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DQE Agent API",
    version="3.0.0",
    description="Browser automation agent with Planner-Executor-Verifier architecture",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DIST_DIR = Path(__file__).resolve().parent.parent.parent / "dist"


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — main entry point
# ══════════════════════════════════════════════════════════════════════════════

async def _run_with_browser(bm: Any, session_id: str, coro) -> None:
    if bm is not None:
        set_browser(bm, session_id)
    await coro


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    logger.info("[%s] WebSocket connected", session_id)

    _active_websockets.add(ws)

    human_q: asyncio.Queue = asyncio.Queue()
    _human_queues[session_id] = human_q

    # Skip browser creation when browser tools are disabled (e.g. Jira/Calendar demo mode)
    if settings.disable_browser_tools:
        session_bm = None
    else:
        session_bm = await browser_manager.create_session()
        _session_browsers[session_id] = session_bm
        set_browser(session_bm, session_id)

    await _send(ws, {"type": "connected", "session_id": session_id})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg: dict = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, {"type": "error", "message": "Invalid JSON"})
                continue

            mtype = msg.get("type", "")

            if mtype == "ping":
                await _send(ws, {"type": "pong"})

            elif mtype == "chat":
                content = msg.get("content", "").strip()
                if content:
                    if session_id in _awaiting_input:
                        await human_q.put(content)
                    else:
                        _cancel_session_task(session_id)
                        task = asyncio.create_task(
                            _run_with_browser(session_bm, session_id,
                                              _handle_message(ws, session_id, content))
                        )
                        _session_tasks[session_id] = task

            elif mtype == "run_task":
                # Alias — same unified PEV handler
                task_desc = msg.get("task", "").strip()
                if task_desc:
                    _cancel_session_task(session_id)
                    task = asyncio.create_task(
                        _run_with_browser(session_bm, session_id,
                                          _handle_message(ws, session_id, task_desc))
                    )
                    _session_tasks[session_id] = task

            elif mtype == "run_workflow":
                workflow_name = msg.get("workflow", "opportunity_to_quote")
                inputs = msg.get("inputs", {})
                _cancel_session_task(session_id)
                task = asyncio.create_task(
                    _run_with_browser(session_bm, session_id,
                                      _handle_workflow(ws, session_id, workflow_name, inputs))
                )
                _session_tasks[session_id] = task

            elif mtype == "human_response":
                response = msg.get("content", "").strip()
                await human_q.put(response)

            elif mtype == "selection_response":
                # User picked from a selection_request UI — route into the same queue
                value = msg.get("value", "")
                if isinstance(value, list):
                    value = json.dumps(value)
                await human_q.put(str(value))

            elif mtype == "form_response":
                # User submitted a multi-field form — route all values into the queue as JSON
                values = msg.get("values", {})
                await human_q.put(json.dumps(values) if isinstance(values, dict) else str(values))

            elif mtype == "edit_response":
                # User submitted edited content — route final text into the queue
                content = msg.get("content", "")
                await human_q.put(str(content))

            elif mtype == "browser_input":
                asyncio.create_task(_handle_browser_input(msg, session_id))

            elif mtype == "webrtc_offer":
                # WebRTC signaling — skip when browser is disabled
                if session_bm is None:
                    pass  # browser disabled — no video stream
                else:
                    sdp = msg.get("sdp", "")
                    if sdp:
                        answer = await handle_webrtc_offer(session_id, sdp, session_bm)
                        await _send(ws, answer)

            else:
                await _send(ws, {"type": "error", "message": f"Unknown type: {mtype!r}"})

    except (WebSocketDisconnect, RuntimeError):
        logger.info("[%s] WebSocket disconnected", session_id)
    finally:
        _active_websockets.discard(ws)
        # Guard: only clean up state owned by THIS connection.
        # If the frontend reconnected with the same session_id, a new connection
        # already registered fresh objects — don't clobber them.
        _awaiting_input.discard(session_id)
        if _human_queues.get(session_id) is human_q:
            _human_queues.pop(session_id, None)
        if session_bm is not None and _session_browsers.get(session_id) is session_bm:
            _cancel_session_task(session_id)
            _session_browsers.pop(session_id, None)
        await cleanup_webrtc(session_id)
        if session_bm is not None:
            await session_bm.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Unified handler: ALL messages go through Planner → Executor → Verifier
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_message(ws: WebSocket, session_id: str, message: str) -> None:
    """Run the unified PEV graph for any user message (chat or task)."""
    from dqe_agent.memory.store import task_store

    if not master_agent:
        await _send(ws, {"type": "error", "message": "Agent not initialised"})
        return

    task_id = f"pev-{session_id}-{uuid.uuid4().hex[:8]}"
    task_store.create_task(
        task_id, session_id, "pev",
        initial_state={"task": message},
        title=message[:200],
        source="chat",
    )

    session_bm = _session_browsers.get(session_id)
    config = {"configurable": {"thread_id": session_id}}

    input_state: dict[str, Any] = {
        "messages": [("user", message)],
        "task": message,
        "session_id": session_id,
        "browser_ready": True,
        "status": "planning",
        "plan": [],
        "current_step_index": 0,
        "step_results": [],
        "retry_count": 0,
        "replan_count": 0,
        "steps_taken": 0,
        "estimated_cost": 0.0,
        "flow_data": {},
    }

    current_input: Any = input_state
    steps_completed = 0
    last_summary: str = ""
    task_type_set = False

    try:
        task_store.update_task(task_id, status="running")

        # ── Signal 'planning' immediately so the frontend shows activity ──
        await _send(ws, {"type": "status", "status": "planning", "message": "Planning your request..."})
        task_store.log(task_id, "Planning started", step="planner", level="info")

        while True:
            interrupted = False
            if session_bm:
                set_browser(session_bm, session_id)

            async for event in master_agent.get_app().astream(
                current_input, config, stream_mode="updates"
            ):
                if "__interrupt__" in event:
                    interrupt_val = event["__interrupt__"][0].value
                    interrupt_type = interrupt_val.get("type", "human_review")

                    human_q = _human_queues.get(session_id)
                    if human_q is None:
                        logger.warning("[%s] interrupt: queue not found — skipping", session_id)
                        break

                    # Mark BEFORE sending so any inbound message arriving between
                    # the send and the queue.get() is routed to the queue.
                    _awaiting_input.add(session_id)

                    if interrupt_type == "selection":
                        # Structured drill-down — frontend renders buttons/dropdown
                        logger.info(
                            "[%s] selection_request: %r (%d options, multi=%s)",
                            session_id,
                            interrupt_val.get("question", "")[:80],
                            len(interrupt_val.get("options", [])),
                            interrupt_val.get("multi_select", False),
                        )
                        await _send(ws, {
                            "type": "selection_request",
                            "question": interrupt_val.get("question", "Please select an option:"),
                            "options": interrupt_val.get("options", []),
                            "multi_select": interrupt_val.get("multi_select", False),
                        })
                    elif interrupt_type == "form":
                        # Multi-field form — frontend renders all fields at once
                        logger.info(
                            "[%s] form_request: %r (%d fields)",
                            session_id,
                            interrupt_val.get("title", "")[:60],
                            len(interrupt_val.get("fields", [])),
                        )
                        await _send(ws, {
                            "type": "form_request",
                            "title": interrupt_val.get("title", ""),
                            "fields": interrupt_val.get("fields", []),
                        })
                    elif interrupt_type == "edit_request":
                        # Editable textarea — frontend pre-fills content and lets user modify
                        logger.info(
                            "[%s] edit_request: label=%r content_len=%d",
                            session_id,
                            interrupt_val.get("label", "")[:60],
                            len(interrupt_val.get("content", "")),
                        )
                        await _send(ws, {
                            "type": "edit_request",
                            "label": interrupt_val.get("label", ""),
                            "content": interrupt_val.get("content", ""),
                            "question": interrupt_val.get("question", ""),
                        })
                    else:
                        # Plain human review / ask_user
                        question = interrupt_val.get("question", "Please provide input:")
                        logger.info(
                            "[%s] human_review: sending prompt to frontend (len=%d)",
                            session_id, len(question),
                        )
                        await _send(ws, {"type": "human_review", "question": question})

                    try:
                        response = await asyncio.wait_for(human_q.get(), timeout=300)
                    finally:
                        _awaiting_input.discard(session_id)

                    current_input = Command(resume=response)
                    interrupted = True
                    break

                for node_name, node_output in event.items():
                    if node_name.startswith("__"):
                        continue

                    msgs = node_output.get("messages", [])
                    plan = node_output.get("plan")

                    # Emit human-readable phase events for each node
                    if node_name == "planner":
                        pass  # planning status already sent before the loop
                    elif node_name == "executor":
                        step_desc = ""
                        if plan := node_output.get("plan"):
                            idx = node_output.get("current_step_index", 0)
                            if 0 <= idx < len(plan):
                                step_desc = plan[idx].get("description", "") or plan[idx].get("tool", "")
                        if not step_desc and node_output.get("step_results"):
                            last_sr = node_output["step_results"][-1]
                            step_desc = last_sr.get("step_id", "") or last_sr.get("tool", "")
                        await _send(ws, {"type": "status", "status": "executing",
                                        "message": f"Executing: {step_desc}" if step_desc else "Executing..."})
                    elif node_name == "verifier":
                        await _send(ws, {"type": "status", "status": "verifying", "message": "Verifying result..."})

                    # Send plan info
                    if plan and node_name == "planner":
                        await _send(ws, {
                            "type": "plan_created",
                            "steps": [
                                {
                                    "id": s.get("id", ""),
                                    "label": s.get("description", "") or s.get("tool", "") or s.get("id", ""),
                                    "description": s.get("description", "") or s.get("tool", "") or s.get("id", ""),
                                    "tool": s.get("tool", ""),
                                }
                                for s in plan
                            ],
                        })
                        task_store.log(task_id, f"Plan created with {len(plan)} steps", step="planner")
                        # Infer task type from the primary step's tool (first non-generic tool)
                        if not task_type_set and plan:
                            first_tool = next(
                                (s.get("tool", "") for s in plan if s.get("tool")), ""
                            )
                            if first_tool:
                                task_store.update_task(task_id, task_type=_infer_task_type(first_tool))
                                task_type_set = True

                    # Send step status — with human-readable label
                    step_results = node_output.get("step_results", [])
                    # Build a quick id→description map from the plan for labelling
                    _plan_map = {s.get("id", ""): s for s in (node_output.get("plan") or [])}
                    for sr in step_results:
                        if isinstance(sr, dict):
                            steps_completed += 1
                            step_id = sr.get("step_id", "")
                            tool = sr.get("tool", "")
                            step_meta = _plan_map.get(step_id, {})
                            label = (
                                step_meta.get("description")
                                or step_meta.get("tool")
                                or tool
                                or step_id
                            )
                            status = sr.get("status", "")
                            await _send(ws, {
                                "type": "step_status",
                                "step": step_id,
                                "tool": tool,
                                "label": label,
                                "status": status,
                                "result": str(sr.get("result", ""))[:200],
                            })
                            # If this was a successful extraction, send full data to frontend
                            if sr.get("tool") == "browser_extract" and status == "success":
                                raw = sr.get("result", "")
                                try:
                                    import json as _json
                                    extracted = _json.loads(raw) if isinstance(raw, str) else raw
                                    if isinstance(extracted, dict):
                                        await _send(ws, {
                                            "type": "extraction_result",
                                            "step": step_id,
                                            "data": extracted,
                                        })
                                except Exception:
                                    pass
                            # Store structured tool call data for the detail endpoint
                            log_msg = f"[{label}] {status}" if label != tool else f"[{tool}] {status}"
                            task_store.log(
                                task_id,
                                log_msg,
                                step=step_id,
                                level="info" if status == "success" else "warning",
                                data={
                                    "tool":        tool,
                                    "label":       label,
                                    "result":      str(sr.get("result", ""))[:500],
                                    "duration_ms": sr.get("duration_ms"),
                                    "retries":     sr.get("retries", 0),
                                    "error":       sr.get("error"),
                                },
                            )
                            task_store.update_task(task_id, steps_taken=steps_completed)

                    # Send agent text and track last meaningful summary
                    for m in msgs:
                        content = getattr(m, "content", "")
                        if content:
                            await _send(ws, {"type": "agent_text", "content": content})
                            last_summary = str(content)[:300]

            if not interrupted:
                break

        task_store.update_task(
            task_id,
            status="complete",
            result_summary=last_summary or None,
        )
        await _send(ws, {"type": "status", "status": "completed", "message": "Done"})
        await _send(ws, {"type": "agent_done", "content": ""})


    except asyncio.TimeoutError:
        _awaiting_input.discard(session_id)
        task_store.update_task(task_id, status="failed", error="Timed out waiting for input")
        await _send(ws, {"type": "error", "message": "Timed out waiting for input."})
    except Exception as exc:
        _awaiting_input.discard(session_id)
        task_store.update_task(task_id, status="failed", error=str(exc))
        logger.error("[%s] message error: %s", session_id, exc, exc_info=True)
        await _send(ws, {"type": "error", "message": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
# Handler: YAML workflow execution (legacy, still supported)
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_workflow(
    ws: WebSocket,
    session_id: str,
    workflow_name: str,
    inputs: dict[str, Any],
) -> None:
    """Execute a named YAML workflow."""
    from dqe_agent.engine import build_graph_from_workflow, load_workflow

    yaml_path = WORKFLOW_DIR / f"{workflow_name}.yaml"
    if not yaml_path.exists():
        await _send(ws, {"type": "error", "message": f"Workflow not found: {yaml_path}"})
        return

    await _send(ws, {"type": "step_status", "step": "__start__", "status": "running"})

    try:
        workflow = load_workflow(yaml_path)
        graph = build_graph_from_workflow(workflow)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": f"wf-{session_id}"}}
        current_input: Any = inputs

        while True:
            interrupted = False
            async for event in compiled.astream(current_input, config, stream_mode="updates"):
                if "__interrupt__" in event:
                    interrupt_val = event["__interrupt__"][0].value
                    question = interrupt_val.get("question", "Please review and respond.")
                    await _send(ws, {"type": "human_review", "question": question})
                    human_q = _human_queues.get(session_id)
                    if human_q is None:
                        break
                    response = await asyncio.wait_for(human_q.get(), timeout=300)
                    current_input = Command(resume=response)
                    interrupted = True
                    break
                else:
                    for step_id in event:
                        if not step_id.startswith("__"):
                            await _send(ws, {"type": "step_status", "step": step_id, "status": "done"})
            if not interrupted:
                break

        await _send(ws, {"type": "workflow_done", "summary": f"Workflow '{workflow_name}' completed."})

    except asyncio.TimeoutError:
        await _send(ws, {"type": "error", "message": "Human review timed out."})
    except Exception as exc:
        logger.error("[%s] workflow error: %s", session_id, exc, exc_info=True)
        await _send(ws, {"type": "error", "message": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
# Handler: browser input forwarding
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_browser_input(msg: dict, session_id: str) -> None:
    session_bm = _session_browsers.get(session_id)
    if not session_bm:
        return
    try:
        action = msg.get("action", "")
        if action == "click":
            await session_bm.send_click(float(msg.get("x", 0)), float(msg.get("y", 0)))
        elif action == "dblclick":
            await session_bm.send_click(float(msg.get("x", 0)), float(msg.get("y", 0)), double=True)
        elif action == "type":
            text = msg.get("text", "")
            if text:
                await session_bm.send_type(text)
        elif action == "key":
            key = msg.get("key", "")
            if key:
                await session_bm.send_key(key)
        elif action == "scroll":
            await session_bm.send_scroll(
                float(msg.get("x", 0)), float(msg.get("y", 0)), float(msg.get("delta", 0))
            )
    except Exception as exc:
        logger.warning("browser_input error (%s): %s", msg.get("action", "?"), exc)


# ══════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    index = _DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"service": "DQE Agent API", "version": "3.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "browser": "ready" if browser_manager else "not started",
        "agent": "ready" if master_agent else "not started",
        "active_sessions": len(_human_queues),
    }


@app.get("/api/v1/tools")
async def get_tools():
    return {
        "tools": [{"name": t.name, "description": t.description} for t in list_tools()],
        "count": len(list_tools()),
    }


@app.get("/api/v1/connectors")
async def get_connectors():
    """Return configured sites (from .env). No Python connector files needed."""
    from dqe_agent.config import settings
    return {"connectors": list(settings.sites.keys())}


@app.get("/api/v1/workflows")
async def get_workflows():
    if not WORKFLOW_DIR.exists():
        return {"workflows": []}
    files = list(WORKFLOW_DIR.glob("*.yaml"))
    return {"workflows": [{"name": f.stem, "file": f.name} for f in sorted(files)]}


@app.get("/api/v1/flows")
async def get_flows():
    from dqe_agent.flows import list_flows
    return {"flows": list_flows()}


@app.get("/api/v1/sessions")
async def get_sessions():
    return {"sessions": list(_human_queues.keys()), "count": len(_human_queues)}


@app.post("/api/v1/reset/{session_id}")
async def reset_session(session_id: str):
    _cancel_session_task(session_id)
    _awaiting_input.discard(session_id)
    q = _human_queues.get(session_id)
    if q:
        await q.put("__reset__")
    if master_agent:
        await master_agent.reset(session_id)
    # Clear session memory
    from dqe_agent.memory.session import session_memory
    await session_memory.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get("/api/v1/tasks/{session_id}")
async def get_tasks(session_id: str, limit: int = 50, offset: int = 0):
    """List tasks for a session, most-recent first."""
    from dqe_agent.memory.store import task_store
    rows = task_store._conn.execute(
        "SELECT * FROM tasks WHERE session_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (session_id, limit, offset),
    ).fetchall()
    tasks = [_format_task(dict(r)) for r in rows]
    return {"tasks": tasks, "count": len(tasks)}


# NOTE: /stats must be registered BEFORE /{task_id} so FastAPI matches the
# literal path segment "stats" before the wildcard.
@app.get("/api/v1/tasks/{session_id}/stats")
async def get_task_stats(session_id: str):
    """Aggregate counts for the session header badges."""
    from dqe_agent.memory.store import task_store
    from datetime import datetime, timezone

    today_start = datetime(
        *datetime.now(timezone.utc).timetuple()[:3], tzinfo=timezone.utc
    ).timestamp()

    conn = task_store._conn

    total = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE session_id=?", (session_id,)
    ).fetchone()[0]

    running = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE session_id=? AND status='running'",
        (session_id,),
    ).fetchone()[0]

    completed_today = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE session_id=? AND status='complete' AND updated_at >= ?",
        (session_id, today_start),
    ).fetchone()[0]

    failed_today = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE session_id=? AND status IN ('failed','failed_restart') AND updated_at >= ?",
        (session_id, today_start),
    ).fetchone()[0]

    return {
        "running":         running,
        "completed_today": completed_today,
        "failed_today":    failed_today,
        "total":           total,
    }


@app.get("/api/v1/tasks/{session_id}/{task_id}")
async def get_task_detail(session_id: str, task_id: str):
    """Single task with tool_calls timeline."""
    from dqe_agent.memory.store import task_store

    row = task_store._conn.execute(
        "SELECT * FROM tasks WHERE task_id=? AND session_id=?",
        (task_id, session_id),
    ).fetchone()
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")

    d = dict(row)
    base = _format_task(d)
    state = json.loads(d.get("state_json") or "{}")
    base["user_message"] = state.get("task", "")

    # Reconstruct tool_calls from log entries that carry structured data
    log_rows = task_store._conn.execute(
        "SELECT * FROM task_logs WHERE task_id=? AND data_json IS NOT NULL ORDER BY ts ASC",
        (task_id,),
    ).fetchall()

    tool_calls = []
    for lr in log_rows:
        data = json.loads(lr["data_json"])
        if data.get("tool"):
            tool_calls.append({
                "tool":        data["tool"],
                "args":        data.get("args", {}),
                "result":      data.get("result"),
                "duration_ms": data.get("duration_ms"),
                "status":      lr["level"],
            })

    base["tool_calls"] = tool_calls
    return base


@app.get("/api/v1/tasks/{session_id}/{task_id}/logs")
async def get_task_logs(session_id: str, task_id: str):
    """Structured log entries for a task."""
    from dqe_agent.memory.store import task_store

    rows = task_store._conn.execute(
        "SELECT * FROM task_logs WHERE task_id=? ORDER BY ts ASC LIMIT 500",
        (task_id,),
    ).fetchall()

    logs = []
    for r in rows:
        entry = {
            "ts":      _epoch_to_iso(r["ts"]),
            "level":   r["level"] or "info",
            "message": r["message"] or "",
            "step":    r["step"] or "",
        }
        if r["data_json"]:
            entry["data"] = json.loads(r["data_json"])
        logs.append(entry)

    return {"logs": logs, "count": len(logs)}


@app.get("/api/v1/traces")
async def get_traces(limit: int = 100):
    """Read recent trace entries from the JSONL log."""
    trace_file = Path("logs/traces.jsonl")
    if not trace_file.exists():
        return {"traces": [], "count": 0}
    try:
        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")[-limit:]
        traces = [json.loads(line) for line in lines if line.strip()]
        return {"traces": traces, "count": len(traces)}
    except Exception:
        return {"traces": [], "count": 0}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cancel_session_task(session_id: str) -> None:
    old = _session_tasks.pop(session_id, None)
    if old and not old.done():
        old.cancel()


async def _send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_json(data)
    except Exception:
        pass


def _safe_args(args: Any) -> Any:
    if not isinstance(args, dict):
        return {}
    HIDDEN = {"password", "pwd", "secret", "token", "api_key"}
    return {
        k: "***" if any(h in k.lower() for h in HIDDEN) else v
        for k, v in args.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# Static files
# ══════════════════════════════════════════════════════════════════════════════
if _DIST_DIR.exists():
    # Mount whichever static dirs exist (Next.js uses _next/, Vite uses assets/)
    _assets_dir = _DIST_DIR / "assets"
    _next_dir = _DIST_DIR / "_next"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="static-assets")
    if _next_dir.exists():
        app.mount("/_next", StaticFiles(directory=_next_dir), name="static-next")

    @app.get("/favicon.svg")
    async def favicon():
        f = _DIST_DIR / "favicon.svg"
        return FileResponse(f) if f.exists() else Response(status_code=204)

    @app.get("/icons.svg")
    async def icons():
        f = _DIST_DIR / "icons.svg"
        return FileResponse(f) if f.exists() else Response(status_code=204)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "ws/", "docs", "openapi.json", "health")):
            return {"detail": "Not Found"}
        file_path = _DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_DIST_DIR / "index.html")
else:
    logger.warning("dist/ not found at %s — UI will not be served", _DIST_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dqe_agent.api:app", host="0.0.0.0", port=8001, reload=True)
