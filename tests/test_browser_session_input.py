"""Tests that BrowserSession has the input-forwarding methods required by api._handle_browser_input.

The critical bug was that _handle_browser_input called send_click / send_type /
send_key / send_scroll on a BrowserSession, but those methods only existed on
BrowserManager — causing silent AttributeErrors and making all user clicks in
the browser panel do nothing.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqe_agent.browser.manager import BrowserSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_session(cdp=None):
    """Create a BrowserSession with mocked Playwright objects."""
    context = MagicMock()
    page = MagicMock()
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.keyboard = MagicMock()
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()

    settings = MagicMock()
    settings.screenshot_enabled = False

    session = BrowserSession(context, page, settings)
    # Inject a mock CDP session if requested
    session._cdp = cdp
    return session


# ---------------------------------------------------------------------------
# send_click
# ---------------------------------------------------------------------------

async def test_send_click_exists():
    """BrowserSession must expose send_click (was missing before fix)."""
    session = _make_session()
    assert hasattr(session, "send_click"), "BrowserSession is missing send_click"


async def test_send_click_without_cdp_uses_playwright():
    """Without a CDP session, send_click should fall back to page.mouse.click."""
    session = _make_session(cdp=None)
    await session.send_click(100.0, 200.0)
    session._page.mouse.click.assert_awaited_once_with(100.0, 200.0)


async def test_send_click_with_cdp_dispatches_events():
    """With a CDP session, send_click should dispatch mouse events via CDP."""
    cdp = MagicMock()
    cdp.send = AsyncMock()
    session = _make_session(cdp=cdp)

    await session.send_click(50.0, 75.0)

    # Should have sent: mouseMoved, mousePressed, mouseReleased = 3 calls
    assert cdp.send.await_count == 3
    call_args_list = [c.args for c in cdp.send.await_args_list]
    event_types = [a[0] for a in call_args_list]
    assert event_types == [
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
    ]


async def test_send_click_double():
    """Double-click should set clickCount=2 in CDP events."""
    cdp = MagicMock()
    cdp.send = AsyncMock()
    session = _make_session(cdp=cdp)

    await session.send_click(10.0, 20.0, double=True)

    # Find the mousePressed call and check clickCount
    for call in cdp.send.await_args_list:
        args = call.args
        if isinstance(args[1], dict) and args[1].get("type") == "mousePressed":
            assert args[1]["clickCount"] == 2


# ---------------------------------------------------------------------------
# send_type
# ---------------------------------------------------------------------------

async def test_send_type_exists():
    session = _make_session()
    assert hasattr(session, "send_type"), "BrowserSession is missing send_type"


async def test_send_type_delegates_to_keyboard():
    session = _make_session()
    await session.send_type("hello world")
    session._page.keyboard.type.assert_awaited_once_with("hello world")


# ---------------------------------------------------------------------------
# send_key
# ---------------------------------------------------------------------------

async def test_send_key_exists():
    session = _make_session()
    assert hasattr(session, "send_key"), "BrowserSession is missing send_key"


async def test_send_key_delegates_to_keyboard():
    session = _make_session()
    await session.send_key("Enter")
    session._page.keyboard.press.assert_awaited_once_with("Enter")


# ---------------------------------------------------------------------------
# send_scroll
# ---------------------------------------------------------------------------

async def test_send_scroll_exists():
    session = _make_session()
    assert hasattr(session, "send_scroll"), "BrowserSession is missing send_scroll"


async def test_send_scroll_without_cdp_uses_playwright():
    session = _make_session(cdp=None)
    await session.send_scroll(100.0, 200.0, -120.0)
    session._page.mouse.wheel.assert_awaited_once_with(0, -120.0)


async def test_send_scroll_with_cdp_dispatches_wheel():
    cdp = MagicMock()
    cdp.send = AsyncMock()
    session = _make_session(cdp=cdp)

    await session.send_scroll(50.0, 60.0, -200.0)

    cdp.send.assert_awaited_once()
    args = cdp.send.await_args.args
    assert args[0] == "Input.dispatchMouseEvent"
    assert args[1]["type"] == "mouseWheel"
    assert args[1]["deltaY"] == -200.0


# ---------------------------------------------------------------------------
# Integration-style: simulate what api._handle_browser_input does
# ---------------------------------------------------------------------------

async def test_handle_browser_input_click_simulation():
    """Simulate the exact call path from api._handle_browser_input for a click."""
    session = _make_session()
    x, y = float(100), float(200)
    await session.send_click(x, y)
    session._page.mouse.click.assert_awaited_once_with(100.0, 200.0)


async def test_handle_browser_input_type_simulation():
    """Simulate the exact call path from api._handle_browser_input for typing."""
    session = _make_session()
    await session.send_type("test input")
    session._page.keyboard.type.assert_awaited_once_with("test input")


async def test_handle_browser_input_key_simulation():
    """Simulate the exact call path from api._handle_browser_input for key press."""
    session = _make_session()
    await session.send_key("Tab")
    session._page.keyboard.press.assert_awaited_once_with("Tab")


async def test_handle_browser_input_scroll_simulation():
    """Simulate the exact call path from api._handle_browser_input for scroll."""
    session = _make_session()
    await session.send_scroll(100.0, 200.0, -120.0)
    session._page.mouse.wheel.assert_awaited_once_with(0, -120.0)
