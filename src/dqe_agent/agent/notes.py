"""Agent notes — persistent memory of solutions the agent discovered itself.

When the agent diagnoses a failure and adapts to fix it, it writes a note here.
The planner reads these notes on every task so the agent doesn't repeat mistakes.

Format: JSONL (one JSON object per line) in data/agent_notes.jsonl
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_NOTES_FILE = Path("data/agent_notes.jsonl")


def load_notes() -> list[dict]:
    """Load all saved notes. Returns [] if file doesn't exist."""
    if not _NOTES_FILE.exists():
        return []
    notes = []
    try:
        for line in _NOTES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                notes.append(json.loads(line))
    except Exception as exc:
        logger.warning("[notes] Failed to load notes: %s", exc)
    return notes


def save_note(tool: str, failure_reason: str, what_was_seen: str, solution: str, adapted_params: dict) -> None:
    """Append a new note. Called when diagnosis + adaptation succeeded."""
    _NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    note = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "failure_reason": failure_reason,
        "what_was_seen": what_was_seen,
        "solution": solution,
        "adapted_params": adapted_params,
    }
    try:
        with _NOTES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(note) + "\n")
        logger.info("[notes] Saved note: %s — %s", tool, solution[:80])
    except Exception as exc:
        logger.warning("[notes] Failed to save note: %s", exc)


def format_notes_for_prompt() -> str:
    """Format saved notes as background hints for the planner.

    These are CONTEXTUAL HINTS only — do NOT add extra steps based on them.
    They inform how existing steps should behave, not what steps to create.
    """
    notes = load_notes()
    if not notes:
        return ""
    lines = [
        "BACKGROUND HINTS (past learnings — use to inform HOW steps run, NOT to add new steps or change the plan structure):"
    ]
    for n in notes[-10:]:  # last 10 notes
        lines.append(
            f"- [{n['tool']}] Issue: {n['failure_reason'][:80]} | "
            f"Fix applied: {n['solution'][:120]}"
        )
    return "\n".join(lines)
