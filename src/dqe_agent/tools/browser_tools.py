"""Generic browser tools — zero website-specific logic.

All site knowledge lives in connectors/ (login URLs, credentials, selectors).
These tools work on ANY website: NetSuite, CPQ, Jira, Salesforce, etc.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import struct
import time
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool
from langgraph.types import interrupt

from dqe_agent.tools import register_tool

logger = logging.getLogger(__name__)

# ── Per-session state (keyed by session_id) ─────────────────────────────────
_session_browser: dict[str, Any] = {}     # session_id → BrowserSession
_session_agent: dict[str, Any] = {}       # session_id → DOMAgent
_session_data: dict[str, dict] = {}       # session_id → extracted data
_active_session_id: str = ""


# ── Session management ──────────────────────────────────────────────────────

def set_browser(browser_session: Any, session_id: str) -> None:
    """Bind a BrowserSession + DOMAgent pair to a session_id.

    Always creates a fresh DOMAgent so it holds the NEW browser_session reference,
    not a stale one from a previous connection with the same session_id.
    """
    global _active_session_id
    _active_session_id = session_id
    _session_browser[session_id] = browser_session

    # Always recreate — never reuse a stale DOMAgent from a previous connection
    from dqe_agent.browser.dom_agent import DOMAgent
    from dqe_agent.llm import get_dom_llm
    _session_agent[session_id] = DOMAgent(browser_session, get_dom_llm())

    if session_id not in _session_data:
        _session_data[session_id] = {}


def _get_browser() -> Any:
    return _session_browser.get(_active_session_id)


def _agent() -> Any:
    return _session_agent.get(_active_session_id)


def _get_session_data() -> dict:
    if _active_session_id not in _session_data:
        _session_data[_active_session_id] = {}
    return _session_data[_active_session_id]


# ── TOTP helper (generic, used by any 2FA-enabled site) ─────────────────────

def _generate_totp_code(secret: str, digits: int = 6, period: int = 30) -> str:
    """Generate a TOTP code from a base32 secret."""
    import base64 as b64
    key = b64.b32decode(secret.upper().replace(" ", ""), casefold=True)
    counter = int(time.time()) // period
    msg = struct.pack(">Q", counter)
    h = hmac.new(key, msg, hashlib.sha1).digest()
    offset = h[-1] & 0x0F
    code = (struct.unpack(">I", h[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 1: browser_login
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_login", "Log in to a configured site by name (netsuite, cpq, jira) or any URL with explicit credentials.")
async def browser_login(site: str = "", url: str = "", username: str = "", password: str = "") -> str:
    """Log in to a website.

    Two modes:
      1. By site name: browser_login(site="netsuite") — looks up credentials from .env config
      2. Explicit:     browser_login(url="https://...", username="...", password="...")
    """
    from dqe_agent.config import settings

    browser = _get_browser()
    dom = _agent()
    if not browser or not dom:
        return json.dumps({"status": "failed", "error": "No browser session"})

    # ── Resolve site config ────────────────────────────────────────────────────
    if site and not url:
        site_config = settings.get_site(site.lower())
        if not site_config:
            configured = list(settings.sites.keys())
            return json.dumps({
                "status": "failed",
                "error": f"Site '{site}' not in .env. Configured sites: {configured}",
            })
        login_url      = site_config["url"]
        login_user     = site_config["username"]
        login_pass     = site_config["password"]
        totp_secret    = site_config.get("totp_secret", "")
        success_frag   = site_config.get("success_url_fragment", "")
        display_name   = site_config.get("display_name", site)
    else:
        login_url    = url
        login_user   = username
        login_pass   = password
        totp_secret  = ""
        success_frag = ""
        display_name = site or url

    if not login_url:
        return json.dumps({"status": "failed", "error": "No URL provided"})

    page = browser.page
    login_path = urlparse(login_url).path

    # Already logged in?
    current_url = page.url if page else ""
    if success_frag and success_frag in current_url:
        logger.info("[browser_login] Already logged in to %s", display_name)
        return json.dumps({"status": "already_logged_in", "site": display_name})

    # ── Skip navigation if already on a challenge/post-login page ────────────
    # LangGraph re-runs the entire tool after interrupt() resumes, so we must
    # not navigate back to the login page if we're already past it.
    current_path = urlparse(current_url).path.lower()
    already_past_login = (
        "loginchallenge" in current_path
        or "myroles" in current_path
        or (success_frag and success_frag in current_url.lower())
    )

    if not already_past_login:
        # ── Navigate to login page ─────────────────────────────────────────────
        logger.info("[browser_login] Navigating to %s: %s", display_name, login_url)
        await page.goto(login_url, wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        try:
            await page.wait_for_timeout(500)
        except Exception:
            pass

        # ── Fill credentials — React _valueTracker JS trick ──────────────────────
        # All Playwright fill / press_sequentially approaches fail on React Hook Form
        # SPAs because React's internal change-detector (_valueTracker) thinks the
        # value is already set and skips onChange.  The fix: reset _valueTracker to ''
        # first, then set the native value + fire input/change/blur — React then sees
        # a real change and updates its form state, enabling the submit button.
        from dqe_agent.browser.dom_agent import NeedsHumanInput as _NeedsHumanInputEarly

        _REACT_FILL_JS = """(args) => {
            function setReactInput(selector, value) {
                var el = document.querySelector(selector);
                if (!el) { return 'no-element:' + selector; }
                // Reset React's internal value tracker so it sees a change
                var tracker = el._valueTracker;
                if (tracker) { tracker.setValue(''); }
                // Set the actual DOM value via native setter (bypasses React override)
                var proto = Object.getPrototypeOf(el);
                var desc  = Object.getOwnPropertyDescriptor(proto, 'value');
                if (desc && desc.set) { desc.set.call(el, value); }
                else { el.value = value; }
                // Fire events: React listens via delegated handlers on the root
                el.dispatchEvent(new Event('input',  { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.dispatchEvent(new FocusEvent('blur', { bubbles: true }));
                return 'ok';
            }
            var r1 = setReactInput(args.emailSel, args.email);
            var r2 = setReactInput(args.passSel,  args.pass);
            return r1 + '|' + r2;
        }"""

        filled_directly = False
        js_result = None
        try:
            logger.info("[browser_login] Attempting React _valueTracker fill for %s", display_name)
            js_result = await page.evaluate(_REACT_FILL_JS, {
                "emailSel": "#email",
                "passSel":  "#password",
                "email":    login_user,
                "pass":     login_pass,
            })
            logger.info("[browser_login] JS fill result: %s", js_result)
            await asyncio.sleep(0.5)  # let React re-render + enable submit button

            # Click the submit button
            submit_loc = page.locator(
                "button[type='submit'], input[type='submit'], "
                "button:has-text('Sign in'), button:has-text('Log in'), button:has-text('Login')"
            ).first
            await submit_loc.click(timeout=5000)
            filled_directly = True
            logger.info("[browser_login] JS fill + submit click done for %s", display_name)
        except BaseException as direct_exc:
            logger.warning("[browser_login] React JS fill failed (%s) — falling back to DOM agent", direct_exc)

        if not filled_directly:
            # If JS returned "no-element" for BOTH fields, the login form is not on
            # this page — we are likely already logged in.  Do a quick text check
            # before sending the DOM agent into the live app UI.
            if js_result and js_result.startswith("no-element:") and "no-element:" in js_result.split("|", 1)[-1]:
                page_text_check = ""
                try:
                    page_text_check = (await page.inner_text("body"))[:500].lower()
                except Exception:
                    pass
                login_kws = ["log in", "sign in", "email address", "password", "forgot your password"]
                if not any(kw in page_text_check for kw in login_kws):
                    logger.info("[browser_login] No login form and no login keywords — already logged in to %s", display_name)
                    return json.dumps({"status": "already_logged_in", "site": display_name})

            instruction = (
                f"Log in to {display_name}. "
                f"Find the username or email field and enter '{login_user}'. "
                f"Find the password field and enter '{login_pass}'. "
                f"Click the Login or Sign In button. Do NOT ask for any credentials."
            )
            try:
                await dom.act(instruction)
            except _NeedsHumanInputEarly:
                logger.info("[browser_login] NeedsHumanInput during credential fill — deferring to challenge loop")
            except BaseException as exc:
                err_str = str(exc)
                if any(msg in err_str for msg in ("Target page", "Connection closed", "Browser.close")):
                    logger.warning("[browser_login] Browser closed during credential fill")
                else:
                    return json.dumps({"status": "failed", "error": err_str, "site": display_name})

        # Press Enter as final fallback if page hasn't navigated
        await asyncio.sleep(0.5)
        if urlparse(page.url).path == login_path:
            logger.info("[browser_login] Page still on login path after fill — pressing Enter to submit")
            try:
                await page.keyboard.press("Enter")
            except Exception:
                pass

        # Wait for the browser to navigate AWAY from the login page.
        # For SPAs the URL path may not change — also accept success_frag appearing.
        for _ in range(30):  # up to 15 seconds
            await asyncio.sleep(0.5)
            new_url = page.url
            if urlparse(new_url).path != login_path:
                break
            if success_frag and success_frag in new_url.lower():
                break
        # Extra settle time for SPAs (React/Vue apps that render async after login)
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        await asyncio.sleep(1.5)
    else:
        logger.info("[browser_login] Already past login page (currently at %s) — skipping navigation", current_path)

    # ── Post-login challenge handling (up to 3 rounds) ─────────────────────────
    from dqe_agent.browser.dom_agent import NeedsHumanInput as _NeedsHumanInput

    for _round in range(5):
        # Wait for page to fully load before inspecting it
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

        current_full = page.url
        # Check path ONLY — the login URL embeds "myroles" in its redirect query param
        # which would cause false positives if we match the full URL string.
        _path = urlparse(current_full).path.lower()
        current = current_full.lower()  # kept for success_frag check (fragment is in path)

        try:
            page_text = await page.inner_text("body")
        except Exception:
            page_text = ""

        # Already succeeded — URL fragment match OR large page (SPA loaded the app)
        if success_frag and success_frag in current:
            break
        # SPA fallback: if page text is substantial and no login/challenge keywords, assume logged in
        login_keywords = ["log in", "sign in", "email address", "password", "forgot your password"]
        if len(page_text) > 2000 and not any(kw in page_text.lower() for kw in login_keywords):
            logger.info("[browser_login] SPA login detected — page loaded with %d chars, no login form", len(page_text))
            break

        # Unified 2FA / code keyword check — covers both loginchallenge path and page content
        code_keywords = ["verification code", "security code", "one-time", "otp", "enter code",
                         "check your email", "sent a code", "authenticator", "two-factor", "2fa",
                         "authentication code", "6-digit"]
        is_challenge_path = "loginchallenge" in _path or "myroles" in _path
        needs_code = any(kw in page_text.lower() for kw in code_keywords)

        logger.info("[browser_login] Challenge round %d: path=%s, needs_code=%s, page_text_len=%d",
                    _round, _path, needs_code, len(page_text))

        if is_challenge_path and needs_code:
            logger.info("[browser_login] 2FA/verification code required — asking user")
            code = interrupt({"question": "NetSuite is asking for a verification code. Open your authenticator app, get a FRESH unused code, and enter it:"})

            # Use direct Playwright fill — much faster than DOM agent (avoids LLM round-trip),
            # critical because TOTP codes expire every 30 s and can't be reused.
            code_str = str(code).strip()
            submitted = False
            try:
                # NetSuite loginchallenge uses SAP UI5 — IDs like uif58, uif74 are stable
                inp = page.locator(
                    "#uif58_input, "
                    "input[placeholder*='digit' i], "
                    "input[type='text']:visible, "
                    "input[type='number']:visible"
                ).first
                await inp.fill("", timeout=3000)
                await inp.fill(code_str, timeout=3000)
                btn = page.locator(
                    "#uif74, "
                    "button:has-text('Submit'), "
                    "input[type='submit']"
                ).first
                await btn.click(timeout=3000)
                submitted = True
                logger.info("[browser_login] Direct Playwright 2FA submit done (code len=%d)", len(code_str))
            except Exception as fill_exc:
                logger.warning("[browser_login] Direct fill failed (%s) — falling back to dom.act", fill_exc)
                try:
                    await dom.act(f"Enter the verification code '{code_str}' in the code field and click Submit or Verify.")
                    submitted = True
                except _NeedsHumanInput:
                    await dom.act(f"The code is '{code_str}'. Enter it now in the verification code field and click Submit.")
                    submitted = True

            if submitted:
                # Wait up to 15 s for NetSuite to verify the code and navigate away.
                # A successful code causes a redirect (loginchallenge disappears from URL).
                # Only fall through to the next round if the page genuinely didn't move.
                try:
                    await page.wait_for_url(
                        lambda url: "loginchallenge" not in url.lower(),
                        timeout=15000,
                    )
                    logger.info("[browser_login] 2FA accepted — navigated away from challenge page")
                    break  # login progressing — exit challenge loop
                except Exception:
                    logger.info("[browser_login] Still on challenge page after 15 s — code may have been rejected")
            continue

        if is_challenge_path and not needs_code:
            # Role selection page — click first role
            logger.info("[browser_login] Role selection page — clicking first role")
            try:
                await dom.act(
                    "You are on a NetSuite role selection page. "
                    "Find the list of roles and click the 'Log In' link or button next to the first role shown. "
                    "Do not click any other links."
                )
            except _NeedsHumanInput:
                pass  # ignore ask_human from role selection — just continue
            try:
                await page.wait_for_timeout(2000)
            except Exception:
                pass
            continue

        # Page text has 2FA keywords but not a loginchallenge URL — still ask
        if needs_code and not is_challenge_path:
            logger.info("[browser_login] 2FA detected on non-challenge page — asking user")
            if totp_secret:
                totp_code = _generate_totp_code(totp_secret)
                await dom.act(f"Enter the verification/TOTP code: {totp_code} and click verify or submit")
            else:
                code = interrupt({"question": f"2FA required for {display_name}. Enter the code:"})
                await dom.act(f"Enter the verification code: {code} and click verify or submit")
            try:
                await page.wait_for_timeout(2000)
            except Exception:
                pass
            continue

        # No recognised challenge — stop looping
        break

    final_url = page.url
    logged_in = success_frag and success_frag in final_url.lower()
    status = "logged_in" if logged_in else "login_attempted"
    logger.info("[browser_login] %s — final URL: %s (status=%s)", display_name, final_url, status)
    return json.dumps({"status": status, "site": display_name, "url": final_url})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 2: browser_navigate
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_navigate", "Navigate the browser to a specific URL.")
async def browser_navigate(url: str) -> str:
    """Navigate to a URL."""
    browser = _get_browser()
    if not browser or not browser.page:
        return json.dumps({"status": "failed", "error": "No browser session"})

    logger.info("[browser_navigate] → %s", url)
    try:
        await browser.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await browser.page.wait_for_timeout(500)
        final_url = browser.page.url
        title = await browser.page.title()
        return json.dumps({"status": "navigated", "url": final_url, "title": title})
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 3: browser_act
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_act", "Perform any action on the current page using natural language instruction.")
async def browser_act(instruction: str) -> str:
    """Execute an instruction on the current page via DOM agent.

    This is the most flexible tool — it can click, type, scroll, select,
    fill forms, navigate menus, etc. based on the instruction.
    """
    dom = _agent()
    if not dom:
        return json.dumps({"status": "failed", "error": "No DOM agent"})

    logger.info("[browser_act] %s", instruction[:100])
    try:
        result = await dom.act(instruction)
        return json.dumps({"status": "success", "result": str(result)[:1000] if result else "done"})
    except Exception as exc:
        if "NeedsHumanInput" in type(exc).__name__:
            answer = interrupt({"question": str(exc)})
            return json.dumps({"status": "human_input_received", "answer": answer})
        return json.dumps({"status": "failed", "error": str(exc)[:500]})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 4: browser_extract
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_extract", "Extract structured data from the current page.")
async def browser_extract(instruction: str, schema: str = "") -> str:
    """Extract data from the current page based on an instruction.

    Args:
        instruction: What to extract (e.g. "Extract all opportunity details: buyer, bandwidth, address")
        schema: Optional JSON schema hint for the expected output format
    """
    dom = _agent()
    if not dom:
        return json.dumps({"status": "failed", "error": "No DOM agent"})

    logger.info("[browser_extract] %s", instruction[:100])
    try:
        # Build extraction instruction
        full_instruction = instruction
        if schema:
            full_instruction += f"\n\nExpected output format: {schema}"

        result = await dom.extract(full_instruction)

        # Store in session data
        if isinstance(result, dict):
            data = _get_session_data()
            data.update(result)

        return json.dumps(result, default=str) if result else json.dumps({"status": "empty"})
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)[:500]})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 5: browser_click
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_click", "Click a specific element on the page.")
async def browser_click(target: str) -> str:
    """Click an element identified by description, text, or selector.

    Args:
        target: What to click (e.g. "Submit button", "the Login link", "#save-btn", "text=OP-")
    """
    logger.info("[browser_click] %s", target)

    # ── Direct Playwright path (bypasses DOM-agent LLM) ──────────────────────
    # Used when target looks like an explicit selector so the LLM can't pick
    # the wrong parent container element (e.g. opportunity row vs its wrapper).
    browser = _get_browser()
    if browser and browser.page:
        try:
            if target.startswith("text="):
                text_val = target[5:]
                loc = browser.page.get_by_text(text_val, exact=False).first
            elif target.startswith("#") or target.startswith(".") or target.startswith("["):
                loc = browser.page.locator(target).first
            else:
                loc = None

            if loc is not None:
                await loc.click(timeout=8000)
                logger.info("[browser_click] Direct Playwright click succeeded for: %s", target)
                return json.dumps({"status": "clicked", "target": target, "method": "playwright_direct"})
        except Exception as direct_exc:
            logger.warning("[browser_click] Direct Playwright click failed (%s), falling back to DOM agent", direct_exc)

    # ── DOM-agent fallback ────────────────────────────────────────────────────
    dom = _agent()
    if not dom:
        return json.dumps({"status": "failed", "error": "No DOM agent"})

    try:
        result = await dom.act(f"Click on: {target}")
        return json.dumps({"status": "clicked", "target": target, "method": "dom_agent"})
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)[:500]})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 6: browser_type
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_type", "Type text into a field on the page.")
async def browser_type(text: str, target: str = "") -> str:
    """Type text into a field.

    Args:
        text: The text to type
        target: Which field to type into (e.g. "search box", "email field", "#username")
    """
    dom = _agent()
    if not dom:
        return json.dumps({"status": "failed", "error": "No DOM agent"})

    instruction = f"Type '{text}'"
    if target:
        instruction += f" into the {target}"

    logger.info("[browser_type] %s", instruction)
    try:
        result = await dom.act(instruction)
        return json.dumps({"status": "typed", "text": text, "target": target})
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)[:500]})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 7: browser_wait
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_wait", "Wait for a condition on the page (element visible, URL change, etc.).")
async def browser_wait(condition: str, timeout_ms: int = 5000) -> str:
    """Wait for a condition to be met on the page.

    Args:
        condition: What to wait for (e.g. "page loads", "button becomes visible", "URL contains dashboard")
        timeout_ms: Maximum wait time in milliseconds
    """
    browser = _get_browser()
    if not browser or not browser.page:
        return json.dumps({"status": "failed", "error": "No browser session"})

    page = browser.page
    logger.info("[browser_wait] %s (timeout=%dms)", condition, timeout_ms)

    try:
        cond_lower = condition.lower()

        # URL-based waits
        if "url" in cond_lower:
            for fragment in condition.split("'"):
                if fragment and fragment not in ("url contains", "url has", " "):
                    await page.wait_for_url(f"**{fragment}**", timeout=timeout_ms)
                    return json.dumps({"status": "condition_met", "url": page.url})

        # Selector-based waits
        if condition.startswith("#") or condition.startswith(".") or condition.startswith("["):
            await page.wait_for_selector(condition, timeout=timeout_ms)
            return json.dumps({"status": "condition_met", "selector": condition})

        # Text-based waits
        if "visible" in cond_lower or "appears" in cond_lower or "shows" in cond_lower:
            await page.wait_for_timeout(min(timeout_ms, 5000))
            return json.dumps({"status": "waited", "duration_ms": min(timeout_ms, 5000)})

        # Generic wait
        await page.wait_for_timeout(min(timeout_ms, 5000))
        return json.dumps({"status": "waited", "duration_ms": min(timeout_ms, 5000)})

    except Exception as exc:
        return json.dumps({"status": "timeout", "error": str(exc)[:200]})


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 8: browser_snapshot
# ═══════════════════════════════════════════════════════════════════════════

@register_tool("browser_snapshot", "Get the current page text content and URL.")
async def browser_snapshot() -> str:
    """Return the current page URL, title, and visible text content."""
    browser = _get_browser()
    if not browser or not browser.page:
        return json.dumps({"status": "failed", "error": "No browser session"})

    page = browser.page
    try:
        url = page.url
        title = await page.title()
        # Get visible text (capped at 5000 chars)
        text = await page.inner_text("body")
        text = text[:5000] if text else ""

        return json.dumps({
            "status": "success",
            "url": url,
            "title": title,
            "text": text,
        })
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)[:200]})
