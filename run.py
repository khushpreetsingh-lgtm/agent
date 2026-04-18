"""Windows-safe server launcher.

On Windows the default SelectorEventLoop does not support subprocess creation,
which Playwright requires. This script switches to ProactorEventLoop before
uvicorn starts so Playwright can launch Chromium without errors.

Usage:
    python run.py
"""

import sys
import asyncio
import logging

# ── Windows fix: Playwright needs ProactorEventLoop ──────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ── Configure app logging so logger.info() calls appear in the terminal ───────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.ERROR)   # suppress link-local bind noise

import uvicorn  # noqa: E402 (must be imported after policy is set)

if __name__ == "__main__":
    uvicorn.run(
        "dqe_agent.api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,   # --reload uses multiprocessing which breaks Playwright on Windows
        log_level="info",
    )
