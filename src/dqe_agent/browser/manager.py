"""Playwright browser session with CDP Screencast for live streaming.

Instead of polling screenshots, we use Chrome DevTools Protocol (CDP)
Page.startScreencast which pushes a JPEG frame on every visual change.
This gives a smooth, truly live view — same as tools like Browserless.io.

Input events (click, type, scroll) are also forwarded via CDP so the
browser panel in the UI is fully interactive.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Awaitable

from playwright.async_api import Browser, BrowserContext, CDPSession, Page, async_playwright

from dqe_agent.config import settings

logger = logging.getLogger(__name__)

# Called with (jpeg_b64: str) for every new frame from CDP screencast
FrameCallback = Callable[[str], Awaitable[None]]

# Shared CDP screencast parameters used by both BrowserSession and BrowserManager
_SCREENCAST_PARAMS = {
    "format": "jpeg",
    "quality": 80,          # up from 50 — clearer image on local connections
    "maxWidth": settings.viewport_width,
    "maxHeight": settings.viewport_height,
    "everyNthFrame": 1,     # every frame instead of every 2nd — doubles framerate (~30fps)
}


class BrowserSession:
    """Owns one isolated Playwright context + page + CDP session for a single WebSocket session."""

    def __init__(self, context: BrowserContext, page: Page, settings_ref: Any) -> None:
        self._context = context
        self._page = page
        self._settings = settings_ref
        self._cdp: CDPSession | None = None
        self._frame_callbacks: list[FrameCallback] = []

    # ── CDP Screencast ───────────────────────────────────────────────────────

    async def start_screencast(self) -> None:
        """Attach a CDP session and start streaming JPEG frames."""
        try:
            logger.info("[BrowserSession] Creating CDP session …")
            self._cdp = await self._context.new_cdp_session(self._page)

            frame_count = 0

            async def _on_frame(event: dict) -> None:
                nonlocal frame_count
                b64 = event.get("data", "")
                session_id = event.get("sessionId")
                frame_count += 1
                if frame_count <= 3 or frame_count % 50 == 0:
                    logger.info("[BrowserSession] screencastFrame #%d received (len=%d, callbacks=%d)",
                                frame_count, len(b64), len(self._frame_callbacks))
                try:
                    await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
                except Exception as e:
                    logger.warning("[BrowserSession] screencastFrameAck failed: %s", e)
                for cb in list(self._frame_callbacks):
                    try:
                        await cb(b64)
                    except Exception as e:
                        logger.warning("[BrowserSession] frame callback error: %s", e)

            self._cdp.on("Page.screencastFrame", lambda e: asyncio.create_task(_on_frame(e)))

            async def _restart_on_nav(frame) -> None:
                if frame != self._page.main_frame:
                    return
                url = frame.url if hasattr(frame, "url") else "?"
                logger.info("[BrowserSession] framenavigated → %s — restarting screencast", url)
                if self._cdp:
                    try:
                        await self._cdp.send("Page.startScreencast", _SCREENCAST_PARAMS)
                        logger.info("[BrowserSession] screencast restarted after navigation.")
                    except Exception as e:
                        logger.warning("[BrowserSession] screencast restart failed: %s", e)

            self._page.on("framenavigated", lambda f: asyncio.create_task(_restart_on_nav(f)))

            await self._cdp.send("Page.startScreencast", _SCREENCAST_PARAMS)
            logger.info("[BrowserSession] CDP screencast started. viewport=%dx%d",
                        self._settings.viewport_width, self._settings.viewport_height)
        except Exception as exc:
            logger.warning("[BrowserSession] CDP screencast unavailable for session: %s", exc)
            self._cdp = None

    async def stop(self) -> None:
        """Stop screencast, close page, close context."""
        if self._cdp:
            try:
                await self._cdp.send("Page.stopScreencast")
            except Exception:
                pass
        try:
            await self._page.close()
        except Exception:
            pass
        try:
            await self._context.close()
        except Exception:
            pass

    # ── Properties / helpers ─────────────────────────────────────────────────

    @property
    def page(self) -> Page:
        return self._page

    def add_frame_callback(self, cb: FrameCallback) -> None:
        """Register a callback to receive each JPEG frame (base64 string)."""
        self._frame_callbacks.append(cb)

    def remove_frame_callback(self, cb: FrameCallback) -> None:
        """Unregister a frame callback."""
        try:
            self._frame_callbacks.remove(cb)
        except ValueError:
            pass

    async def navigate(self, url: str) -> None:
        await self._page.goto(url, wait_until="domcontentloaded")

    async def screenshot(self, name: str) -> Path | None:
        if not self._settings.screenshot_enabled:
            return None
        dest = self._settings.screenshot_dir
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{name}.png"
        await self._page.screenshot(path=str(path))
        return path

    async def screenshot_bytes(self) -> bytes:
        """Return a PNG screenshot as raw bytes."""
        return await self._page.screenshot(type="png")

    # ── Input forwarding via CDP ─────────────────────────────────────────────

    async def send_click(self, x: float, y: float, double: bool = False) -> None:
        """Send a mouse click to the browser at viewport coordinates (x, y)."""
        if not self._cdp:
            await self._page.mouse.click(x, y)
            return
        count = 2 if double else 1
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x, "y": y,
            "button": "none",
            "buttons": 0,
            "modifiers": 0,
        })
        for event_type in ("mousePressed", "mouseReleased"):
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": event_type,
                "x": x, "y": y,
                "button": "left",
                "buttons": 1 if event_type == "mousePressed" else 0,
                "clickCount": count,
                "modifiers": 0,
            })

    async def send_type(self, text: str) -> None:
        """Type a string into the focused element."""
        await self._page.keyboard.type(text)

    async def send_key(self, key: str) -> None:
        """Press a special key (Enter, Tab, Escape, ArrowDown, etc.)."""
        await self._page.keyboard.press(key)

    async def send_scroll(self, x: float, y: float, delta_y: float) -> None:
        """Scroll at viewport coordinates."""
        if self._cdp:
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseWheel",
                "x": x, "y": y,
                "deltaX": 0,
                "deltaY": delta_y,
                "modifiers": 0,
            })
        else:
            await self._page.mouse.wheel(0, delta_y)


class BrowserManager:
    """Owns a single Playwright browser session with live CDP screencast."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._cdp: CDPSession | None = None
        self._frame_callbacks: list[FrameCallback] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch Chromium and start CDP screencast."""
        logger.info("Launching Playwright Chromium (headless=%s) …", settings.headless)
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=settings.headless,
        )
        self._context = await self._browser.new_context(
            viewport={"width": settings.viewport_width, "height": settings.viewport_height},
        )
        self._page = await self._context.new_page()
        logger.info("Browser ready (%dx%d).", settings.viewport_width, settings.viewport_height)
        await self._start_screencast()

    async def stop(self) -> None:
        """Gracefully stop screencast and close browser."""
        if self._cdp:
            try:
                await self._cdp.send("Page.stopScreencast")
            except Exception:
                pass
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        logger.info("Browser closed.")

    # ── CDP Screencast ───────────────────────────────────────────────────────

    async def _start_screencast(self) -> None:
        """Attach a CDP session and start streaming JPEG frames."""
        try:
            logger.info("[BrowserManager] Creating CDP session on page …")
            self._cdp = await self._context.new_cdp_session(self._page)
            logger.info("[BrowserManager] CDP session created OK.")

            frame_count = 0

            async def _on_frame(event: dict) -> None:
                nonlocal frame_count
                b64 = event.get("data", "")
                session_id = event.get("sessionId")
                frame_count += 1
                if frame_count <= 3 or frame_count % 50 == 0:
                    logger.info("[BrowserManager] screencastFrame #%d received (len=%d, callbacks=%d)",
                                frame_count, len(b64), len(self._frame_callbacks))
                # Acknowledge so Chrome keeps sending frames
                try:
                    await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
                except Exception as e:
                    logger.warning("[BrowserManager] screencastFrameAck failed: %s", e)
                # Broadcast to all registered callbacks (one per WS session)
                for cb in list(self._frame_callbacks):
                    try:
                        await cb(b64)
                    except Exception as e:
                        logger.warning("[BrowserManager] frame callback error: %s", e)

            self._cdp.on("Page.screencastFrame", lambda e: asyncio.create_task(_on_frame(e)))

            async def _restart_on_nav(frame) -> None:
                url = frame.url if hasattr(frame, "url") else "?"
                logger.info("[BrowserManager] framenavigated → %s — restarting screencast", url)
                if self._cdp:
                    try:
                        await self._cdp.send("Page.startScreencast", _SCREENCAST_PARAMS)
                        logger.info("[BrowserManager] screencast restarted after navigation.")
                    except Exception as e:
                        logger.warning("[BrowserManager] screencast restart failed: %s", e)

            self._page.on(
                "framenavigated",
                lambda f: asyncio.create_task(_restart_on_nav(f)) if f == self._page.main_frame else None,
            )

            await self._cdp.send("Page.startScreencast", _SCREENCAST_PARAMS)
            logger.info("[BrowserManager] CDP screencast started. viewport=%dx%d",
                        settings.viewport_width, settings.viewport_height)
        except Exception as exc:
            logger.warning("[BrowserManager] CDP screencast unavailable, falling back to polling: %s", exc)
            self._cdp = None

    def add_frame_callback(self, cb: FrameCallback) -> None:
        """Register a callback to receive each JPEG frame (base64 string)."""
        self._frame_callbacks.append(cb)

    def remove_frame_callback(self, cb: FrameCallback) -> None:
        """Unregister a frame callback."""
        try:
            self._frame_callbacks.remove(cb)
        except ValueError:
            pass

    async def screenshot_bytes(self) -> bytes:
        """Fallback: return a PNG screenshot as raw bytes."""
        return await self.page.screenshot(type="png")

    # ── Input forwarding via CDP ─────────────────────────────────────────────

    async def send_click(self, x: float, y: float, double: bool = False) -> None:
        """Send a mouse click to the browser at viewport coordinates (x, y)."""
        if not self._cdp:
            await self._page.mouse.click(x, y)
            return
        count = 2 if double else 1
        # Chrome requires a mouseMoved event to position the cursor before pressing
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x, "y": y,
            "button": "none",
            "buttons": 0,
            "modifiers": 0,
        })
        for event_type in ("mousePressed", "mouseReleased"):
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": event_type,
                "x": x, "y": y,
                "button": "left",
                "buttons": 1 if event_type == "mousePressed" else 0,
                "clickCount": count,
                "modifiers": 0,
            })

    async def send_type(self, text: str) -> None:
        """Type a string into the focused element."""
        await self._page.keyboard.type(text)

    async def send_key(self, key: str) -> None:
        """Press a special key (Enter, Tab, Escape, ArrowDown, etc.)."""
        await self._page.keyboard.press(key)

    async def send_scroll(self, x: float, y: float, delta_y: float) -> None:
        """Scroll at viewport coordinates."""
        if self._cdp:
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseWheel",
                "x": x, "y": y,
                "deltaX": 0,
                "deltaY": delta_y,
                "modifiers": 0,
            })
        else:
            await self._page.mouse.wheel(0, delta_y)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserManager.start() has not been called yet.")
        return self._page

    async def screenshot(self, name: str) -> Path | None:
        if not settings.screenshot_enabled:
            return None
        dest = settings.screenshot_dir
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{name}.png"
        await self.page.screenshot(path=str(path))
        return path

    async def navigate(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded")

    async def create_session(self) -> "BrowserSession":
        """Create an isolated browser context+page for one WebSocket session."""
        logger.info("[BrowserManager] create_session: creating new browser context …")
        context = await self._browser.new_context(
            viewport={"width": settings.viewport_width, "height": settings.viewport_height},
        )
        page = await context.new_page()
        logger.info("[BrowserManager] create_session: context+page ready, starting screencast …")
        session = BrowserSession(context, page, settings)
        await session.start_screencast()
        logger.info("[BrowserManager] create_session: session ready.")
        return session

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()
