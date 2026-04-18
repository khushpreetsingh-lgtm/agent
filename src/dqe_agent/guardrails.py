"""Guardrails — enforce MAX_STEPS, MAX_COST, TIMEOUT limits on agent execution."""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Hard limits (override via config if needed)
MAX_STEPS: int = 20
MAX_COST_USD: float = 2.0
TIMEOUT_SECONDS: float = 60.0  # 60 seconds per task (per spec)

# Cost estimates per model call (rough USD)
COST_PER_CALL = {
    "planner":  0.04,   # Large model (opus/o1)
    "executor": 0.002,  # Small model (4o-mini)
    "verifier": 0.015,  # Deterministic free; AI vision fallback ~$0.01-0.03
    "dom":      0.003,  # DOM agent LLM call
}


class GuardrailError(Exception):
    """Raised when an agent hits a hard limit."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class GuardrailState:
    steps_taken: int = 0
    estimated_cost: float = 0.0
    start_time: float = field(default_factory=time.time)

    def check(self) -> None:
        """Raise GuardrailError if any limit is exceeded."""
        elapsed = time.time() - self.start_time

        if self.steps_taken >= MAX_STEPS:
            msg = f"MAX_STEPS limit reached ({self.steps_taken}/{MAX_STEPS})"
            logger.warning("GUARDRAIL: %s", msg)
            raise GuardrailError(msg)

        if self.estimated_cost >= MAX_COST_USD:
            msg = f"MAX_COST limit reached (${self.estimated_cost:.3f} / ${MAX_COST_USD})"
            logger.warning("GUARDRAIL: %s", msg)
            raise GuardrailError(msg)

        if elapsed >= TIMEOUT_SECONDS:
            msg = f"TIMEOUT limit reached ({elapsed:.0f}s / {TIMEOUT_SECONDS}s)"
            logger.warning("GUARDRAIL: %s", msg)
            raise GuardrailError(msg)

    def record_step(self, call_type: str = "executor") -> None:
        self.steps_taken += 1
        self.estimated_cost += COST_PER_CALL.get(call_type, 0.002)
        logger.debug(
            "Guardrails: steps=%d cost=$%.4f elapsed=%.1fs",
            self.steps_taken, self.estimated_cost, time.time() - self.start_time,
        )
        self.check()

    def summary(self) -> dict:
        return {
            "steps_taken": self.steps_taken,
            "estimated_cost_usd": round(self.estimated_cost, 4),
            "elapsed_seconds": round(time.time() - self.start_time, 1),
        }
