"""DOM-based browser agent — Playwright DOM extraction + GPT-mini → Playwright actions.

Replaces the screenshot-based CUA for standard web apps (NetSuite, CPQ).

Flow per act() call:
  1. Extract interactive elements from page DOM as structured text
  2. Send DOM snapshot + instruction to GPT-mini (cheap text model)
  3. LLM returns JSON list of actions (fill, click, select_option, etc.)
  4. Execute actions via Playwright using CSS/text selectors
  5. Loop until LLM signals done=true or max_steps reached

Benefits over CUA:
  - ~10x faster  : no image encoding/decoding round trips
  - ~90% cheaper : text model vs vision model
  - More reliable: CSS selectors survive layout changes
  - Works perfectly for standard web apps (NetSuite, CPQ)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from dqe_agent.browser.manager import BrowserManager
from dqe_agent.prompts import DOM_ACTION_SYSTEM, DOM_EXTRACT_SYSTEM

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 8


class NeedsHumanInput(Exception):
    """Raised by DOMAgent when the page requires human input to continue.

    The agent (not keywords) decided it cannot proceed — e.g. OTP screen,
    CAPTCHA, ambiguous choice. The caller must call interrupt() and then
    retry the action with the user's answer injected into the instruction.
    """
    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question
DEFAULT_ACTION_TIMEOUT_MS = 7000
DEFAULT_WAIT_AFTER_CLICK_MS = 150  # reduced: 300→150ms — saves ~150ms per click action
DEFAULT_WAIT_SELECTOR_TIMEOUT_MS = 2000
# First click attempt uses a shorter timeout so failures bail out quickly.
# If the element is found but intercepted (Radix overlay), the Escape+retry
# path then uses the full DEFAULT_ACTION_TIMEOUT_MS.
# Raised from 3000 → 5000 to handle slow-rendering SPAs (e.g. CPQ wizard).
_CLICK_FIRST_ATTEMPT_MS = 5000

# Error strings that mean the browser/page is gone — re-raise immediately
_BROWSER_CLOSED_MSGS = (
    "Target page, context or browser has been closed",
    "Connection closed while reading",
    "Browser.close: Connection closed",
)


# ── JavaScript that extracts interactive elements from the live DOM ──────────
_SNAPSHOT_JS = r"""() => {
    const results = [];
    const seenSel = new Set();
    const seenText = new Set();

    function getLabel(el) {
        // aria-label wins
        const aria = el.getAttribute('aria-label');
        if (aria) return aria.trim();
        // <label for="id">
        if (el.id) {
            try {
                const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                if (lbl) return lbl.innerText.trim().replace(/\s+/g, ' ').slice(0, 80);
            } catch(_) {}
        }
        // placeholder
        const ph = el.getAttribute('placeholder');
        if (ph) return ph.trim();
        // wrapping label
        const wl = el.closest('label');
        if (wl) return wl.innerText.trim().replace(/\s+/g, ' ').slice(0, 80);
        // name attribute
        return (el.getAttribute('name') || '').trim();
    }

    function getBestSelector(el) {
        if (el.id && /^[a-zA-Z_]/.test(el.id)) return '#' + el.id;
        const testId = el.getAttribute('data-testid') || el.getAttribute('data-test-id');
        if (testId) return '[data-testid="' + testId + '"]';
        if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
        const ph = el.getAttribute('placeholder');
        if (ph) return '[placeholder="' + ph + '"]';
        const aria = el.getAttribute('aria-label');
        if (aria) return '[aria-label="' + aria + '"]';
        // Fallback: type-based selector with position index (handles bare login forms)
        if (el.tagName === 'INPUT' && el.type) {
            const t = el.type.toLowerCase();
            const siblings = Array.from(document.querySelectorAll('input[type="' + t + '"]:not([type=hidden]):not([disabled])'));
            const idx = siblings.indexOf(el);
            if (idx >= 0) return 'input[type="' + t + '"]:nth-of-type(' + (idx + 1) + ')';
            return 'input[type="' + t + '"]';
        }
        return null;
    }

    function isVisible(el) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return false;
        const s = window.getComputedStyle(el);
        return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
    }

    const TEXT_INPUTS = new Set(['text','email','tel','number','password','search','date','datetime-local','url','time','']);

    // ── text inputs & textareas ───────────────────────────────────────────────
    for (const el of document.querySelectorAll('input:not([type=hidden]):not([disabled]), textarea:not([disabled])')) {
        if (!isVisible(el)) continue;
        const t = (el.type || '').toLowerCase();
        if (el.tagName === 'INPUT' && !TEXT_INPUTS.has(t)) continue;
        const sel = getBestSelector(el);
        if (!sel) continue;  // truly no selector possible
        if (seenSel.has(sel)) continue;
        seenSel.add(sel);
        results.push({
            kind: el.tagName === 'TEXTAREA' ? 'textarea' : 'input',
            inputType: t || 'text',
            selector: sel,
            label: getLabel(el),
            value: el.value || '',
        });
    }

    // ── native <select> ───────────────────────────────────────────────────────
    for (const el of document.querySelectorAll('select:not([disabled])')) {
        if (!isVisible(el)) continue;
        const sel = getBestSelector(el);
        if (!sel || seenSel.has(sel)) continue;
        seenSel.add(sel);
        const opts = Array.from(el.options).map(o => o.text.trim()).filter(Boolean).slice(0, 20);
        results.push({
            kind: 'select',
            selector: sel,
            label: getLabel(el),
            current: el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : '',
            options: opts,
        });
    }

    // ── checkboxes ───────────────────────────────────────────────────────────
    for (const el of document.querySelectorAll('input[type=checkbox]:not([disabled])')) {
        if (!isVisible(el)) continue;
        const sel = getBestSelector(el);
        if (!sel || seenSel.has(sel)) continue;
        seenSel.add(sel);
        results.push({ kind: 'checkbox', selector: sel, label: getLabel(el), checked: el.checked });
    }

    // ── buttons ───────────────────────────────────────────────────────────────
    for (const el of document.querySelectorAll('button, input[type="submit"], a[role="button"]')) {
        if (!isVisible(el)) continue;
        const raw = (el.innerText || el.getAttribute('value') || el.getAttribute('aria-label') || '').trim();
        const text = raw.replace(/\s+/g, ' ').slice(0, 80);
        if (!text || seenText.has(text)) continue;
        seenText.add(text);
        const sel = getBestSelector(el) || null;
        const isDisabled = el.disabled || el.getAttribute('aria-disabled') === 'true';
        results.push({ kind: 'button', selector: sel, text, disabled: isDisabled });
    }

    // ── links (<a href>) — critical for Edit|View and popup dismissal ─────────
    for (const el of document.querySelectorAll('a[href]')) {
        if (!isVisible(el)) continue;
        const raw = (el.innerText || el.getAttribute('aria-label') || '').trim();
        const text = raw.replace(/\s+/g, ' ').slice(0, 80);
        if (!text || seenText.has(text)) continue;
        seenText.add(text);
        const href = el.getAttribute('href') || '';
        const sel = el.id ? '#' + el.id : (href && href !== '#' ? 'a[href="' + href + '"]' : null);
        results.push({ kind: 'link', selector: sel, text, href: href.slice(0, 120) });
    }

    // ── ARIA role-based elements (React / Angular / custom UI frameworks) ────
    // These won't appear as native <input>/<button> but are still interactive.

    // role="textbox" / contenteditable — custom text inputs
    for (const el of document.querySelectorAll('[role="textbox"], [contenteditable="true"]')) {
        if (!isVisible(el)) continue;
        const sel = getBestSelector(el) || (el.id ? '#' + el.id : null);
        if (!sel || seenSel.has(sel)) continue;
        seenSel.add(sel);
        results.push({
            kind: 'input',
            inputType: 'text',
            selector: sel,
            label: getLabel(el),
            value: el.innerText || el.getAttribute('value') || '',
        });
    }

    // role="button" / role="link" — custom clickable elements
    for (const el of document.querySelectorAll('[role="button"], [role="link"]')) {
        if (!isVisible(el)) continue;
        const raw = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
        const text = raw.replace(/\s+/g, ' ').slice(0, 80);
        if (!text || seenText.has(text)) continue;
        seenText.add(text);
        const sel = getBestSelector(el) || (el.id ? '#' + el.id : null);
        results.push({ kind: 'button', selector: sel, text });
    }

    // role="option" / role="listbox" — custom dropdown items
    for (const el of document.querySelectorAll('[role="combobox"], [role="listbox"]')) {
        if (!isVisible(el)) continue;
        const sel = getBestSelector(el) || (el.id ? '#' + el.id : null);
        if (!sel || seenSel.has(sel)) continue;
        seenSel.add(sel);
        const opts = Array.from(document.querySelectorAll('[role="option"]'))
            .map(o => o.innerText.trim()).filter(Boolean).slice(0, 20);
        results.push({
            kind: 'select',
            selector: sel,
            label: getLabel(el),
            current: el.getAttribute('aria-activedescendant') || '',
            options: opts,
        });
    }

    // ── Radix UI / shadcn custom <select> triggers ──────────────────────────
    // These render as <button> with a child <span data-slot="select-value">
    // and are NOT native <select> or ARIA combobox, so the above loops miss them.
    for (const btn of document.querySelectorAll('button')) {
        if (!isVisible(btn)) continue;
        const valSpan = btn.querySelector('[data-slot="select-value"], span[data-placeholder]');
        if (!valSpan) continue;
        // Skip if we already captured this element via another path
        const sel = getBestSelector(btn) || (btn.id ? '#' + btn.id : null);
        if (sel && seenSel.has(sel)) continue;
        // Derive label from a nearby <label>, aria-label, or preceding text node
        let label = getLabel(btn);
        if (!label) {
            // Walk up to the closest wrapper and look for a label-like sibling
            const wrapper = btn.closest('[class*="field"], [class*="form-group"], div');
            if (wrapper) {
                const lbl = wrapper.querySelector('label');
                if (lbl) label = lbl.innerText.trim().replace(/\s+/g, ' ').slice(0, 80);
            }
        }
        const current = valSpan.innerText.trim();
        const selectorToUse = sel || 'text=' + current;
        if (sel) seenSel.add(sel);
        results.push({
            kind: 'custom_select',
            selector: selectorToUse,
            label: label || '',
            current: current,
            hint: 'Radix/shadcn select — click to open, then click the desired option text',
        });
    }

    // ── Clickable cards / tiles (product selectors, option cards) ────────────
    // These are often <div> or <article> elements with click handlers, NOT <button>.
    // Capture them so the LLM can interact with card-based UIs.
    for (const el of document.querySelectorAll('[class*="card"], [class*="tile"], [class*="product"], [class*="option-card"], [data-clickable], [data-selectable]')) {
        if (!isVisible(el)) continue;
        const raw = (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 80);
        if (!raw || seenText.has(raw)) continue;
        // Only include if it looks like a card (reasonable size, not too big)
        const rect = el.getBoundingClientRect();
        if (rect.width < 50 || rect.height < 30 || rect.width > 800) continue;
        seenText.add(raw);
        const sel = getBestSelector(el) || null;
        results.push({ kind: 'button', selector: sel, text: raw, hint: 'clickable card' });
    }

    return results;
}"""


# ── LLM system prompts (imported from dqe_agent.prompts) ─────────────────────
_ACTION_SYSTEM = DOM_ACTION_SYSTEM
_EXTRACT_SYSTEM = DOM_EXTRACT_SYSTEM


# ── DOMAgent ─────────────────────────────────────────────────────────────────
class DOMAgent:
    """DOM-based agent: extracts page structure as text, uses GPT-mini to plan actions."""

    def __init__(self, browser: BrowserManager, llm) -> None:
        self._browser = browser
        self._llm = llm

    # ── Public API ────────────────────────────────────────────────────────────

    async def act(self, instruction: str, *, max_steps: int = DEFAULT_MAX_STEPS) -> str:
        """Execute a natural-language instruction via iterative DOM understanding."""
        log_lines: list[str] = []

        for step_num in range(1, max_steps + 1):
            try:
                snapshot = await self._get_page_snapshot()
            except Exception as exc:
                if any(msg in str(exc) for msg in _BROWSER_CLOSED_MSGS):
                    logger.warning("[DOM-AGENT] browser closed while getting snapshot — aborting act()")
                    return " | ".join(log_lines) if log_lines else "browser_closed"
                raise
            logger.debug("[DOMAgent step %d] snapshot (%d chars)", step_num, len(snapshot))

            plan = await self._plan_actions(snapshot, instruction)
            actions: list[dict] = plan.get("actions", [])
            done: bool = plan.get("done", False)

            logger.info("[DOM-AGENT step %d] %d actions  done=%s", step_num, len(actions), done)

            if not actions and done:
                break

            for act in actions:
                if act.get("type") == "ask_human":
                    # Agent decided it needs human input — surface to caller
                    question = act.get("question", "I need your input to continue.")
                    logger.info("[DOM-AGENT] ask_human: %s", question)
                    raise NeedsHumanInput(question)

                try:
                    result = await self._execute_action(act)
                except Exception as exc:
                    # Browser/page closed — stop the loop immediately
                    exc_str = str(exc)
                    if any(msg in exc_str for msg in _BROWSER_CLOSED_MSGS):
                        logger.warning("[DOM-AGENT] browser closed mid-loop — aborting act()")
                        return " | ".join(log_lines) if log_lines else "browser_closed"
                    raise
                log_lines.append(result)
                logger.debug("  ↳ %s", result)

            if done:
                break
        else:
            logger.warning("[DOM-AGENT] hit max_steps=%d without done=true", max_steps)

        return " | ".join(log_lines) if log_lines else "completed"

    async def extract(
        self,
        instruction: str,
        schema: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Extract structured data from the current page via LLM text parsing."""
        page_text = await self._get_page_text()

        schema_hint = ""
        if schema:
            schema_hint = f"\n\nReturn a JSON object with exactly these fields:\n{json.dumps(schema, indent=2)}"

        messages = [
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(
                content=(
                    f"PAGE CONTENT:\n{page_text}\n\n"
                    f"INSTRUCTION: {instruction}"
                    f"{schema_hint}\n\n"
                    "Return ONLY JSON:"
                )
            ),
        ]
        response = await self._llm.ainvoke(messages)
        return self._parse_json(response.content)

    # ── DOM snapshot ──────────────────────────────────────────────────────────

    async def _get_page_snapshot(self) -> str:
        """Render the page's interactive elements as structured text for the LLM."""
        page = self._browser.page
        try:
            title = await page.title()
            url = page.url
        except Exception:
            title, url = "", ""

        try:
            elements: list[dict] = await page.evaluate(_SNAPSHOT_JS)
        except Exception as exc:
            exc_str = str(exc)
            if any(msg in exc_str for msg in _BROWSER_CLOSED_MSGS):
                logger.warning("[DOM-AGENT] snapshot JS failed (browser closed): %s", exc)
                raise  # propagate so act() can stop
            logger.warning("[DOM-AGENT] snapshot JS failed: %s", exc)
            elements = []

        # If no elements found, wait briefly and retry once — SPA may still be rendering
        if not elements:
            try:
                await page.wait_for_timeout(600)  # reduced: 1500→600ms; SPA usually renders in <500ms
                elements = await page.evaluate(_SNAPSHOT_JS)
            except Exception:
                elements = []

        lines: list[str] = [f"URL: {url}", f"Title: {title}", ""]

        inputs = [e for e in elements if e["kind"] in ("input", "textarea")]
        selects = [e for e in elements if e["kind"] == "select"]
        custom_selects = [e for e in elements if e["kind"] == "custom_select"]
        checkboxes = [e for e in elements if e["kind"] == "checkbox"]
        buttons = [e for e in elements if e["kind"] == "button"]
        links = [e for e in elements if e["kind"] == "link"]

        if inputs:
            lines.append("[INPUTS / TEXTAREAS]")
            for e in inputs:
                val = f"  current='{e['value']}'" if e.get("value") else ""
                lines.append(
                    f"  label='{e['label']}'  type={e.get('inputType','text')}"
                    f"  selector='{e['selector']}'{val}"
                )

        if selects:
            lines.append("[DROPDOWNS — native <select>]")
            for e in selects:
                opts_str = " | ".join(e.get("options", [])[:12])
                lines.append(
                    f"  label='{e['label']}'  selector='{e['selector']}'"
                    f"  current='{e.get('current','')}'"
                    f"  options=[{opts_str}]"
                )

        if custom_selects:
            lines.append("[CUSTOM DROPDOWNS — click to open, then click option text]")
            for e in custom_selects:
                lines.append(
                    f"  label='{e['label']}'  selector='{e['selector']}'"
                    f"  current='{e.get('current','')}'"
                    f"  *** DO NOT use fill() — click selector to open, then click the option text ***"
                )

        if checkboxes:
            lines.append("[CHECKBOXES]")
            for e in checkboxes:
                lines.append(
                    f"  label='{e['label']}'  selector='{e['selector']}'  checked={e['checked']}"
                )

        if buttons:
            lines.append("[BUTTONS]")
            for e in buttons:
                sel_hint = f"  selector='{e['selector']}'" if e.get("selector") else ""
                lines.append(f"  '{e['text']}'{sel_hint}")

        if links:
            # In NetSuite, suppress record-creation links so the agent doesn't click them
            _netsuite_blocked = {"create quote", "create order", "new quote", "create sale", "create invoice"}
            filtered_links = [
                e for e in links
                if e["text"].lower().strip() not in _netsuite_blocked
                or "cpq" in url.lower() or "cloudsmartz" in url.lower()
            ]
            if filtered_links:
                lines.append("[LINKS]")
                for e in filtered_links:
                    sel_hint = f"  selector='{e['selector']}'" if e.get("selector") else ""
                    lines.append(f"  '{e['text']}'{sel_hint}")

        if not any([inputs, selects, custom_selects, checkboxes, buttons, links]):
            # Fallback: include raw visible text for context
            try:
                raw = await page.inner_text("body")
                lines.append("[VISIBLE TEXT]")
                lines.append(raw[:2000])
            except Exception:
                pass

        return "\n".join(lines)

    async def _get_page_text(self) -> str:
        """Return readable text of the full page for extraction.
        Scrolls to bottom first so lazy-loaded content is included."""
        try:
            page = self._browser.page
            # Only scroll if the page has significant off-screen content (lazy loading).
            # Skipping the scroll on short pages saves ~550ms (scroll + wait + scroll back).
            needs_scroll = await page.evaluate(
                "() => document.body.scrollHeight > window.innerHeight * 1.5"
            )
            if needs_scroll:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(150)  # reduced: 400→150ms
                await page.evaluate("window.scrollTo(0, 0)")
            text = await page.inner_text("body")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            return text[:20000]  # increased from 8000 — NetSuite pages are content-heavy
        except Exception:
            return ""

    # ── LLM action planning ───────────────────────────────────────────────────

    async def _plan_actions(self, snapshot: str, instruction: str) -> dict:
        messages = [
            SystemMessage(content=_ACTION_SYSTEM),
            HumanMessage(
                content=(
                    f"CURRENT PAGE STATE:\n{snapshot}\n\n"
                    f"INSTRUCTION: {instruction}\n\n"
                    "Return JSON actions:"
                )
            ),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            return self._parse_json(response.content)
        except Exception as exc:
            exc_str = str(exc)
            # Azure OpenAI content filter — retry with a sanitized (truncated) snapshot
            if "content_filter" in exc_str or "content management policy" in exc_str or "ResponsibleAIPolicyViolation" in exc_str:
                logger.warning("[DOM-AGENT] Content filter triggered — retrying with sanitized snapshot")
                # Strip the snapshot to just element labels/selectors (no raw page text)
                sanitized_lines = []
                for line in snapshot.split("\n"):
                    # Keep structural lines (section headers, element metadata) but skip raw text
                    stripped = line.strip()
                    if (
                        stripped.startswith("[")
                        or stripped.startswith("URL:")
                        or stripped.startswith("Title:")
                        or stripped.startswith("label=")
                        or stripped.startswith("'")
                        or stripped == ""
                    ):
                        sanitized_lines.append(line)
                sanitized_snapshot = "\n".join(sanitized_lines)
                retry_messages = [
                    SystemMessage(content=_ACTION_SYSTEM),
                    HumanMessage(
                        content=(
                            f"CURRENT PAGE STATE:\n{sanitized_snapshot}\n\n"
                            f"INSTRUCTION: {instruction}\n\n"
                            "Return JSON actions:"
                        )
                    ),
                ]
                try:
                    response = await self._llm.ainvoke(retry_messages)
                    return self._parse_json(response.content)
                except Exception as retry_exc:
                    logger.error("[DOM-AGENT] Content filter retry also failed: %s", retry_exc)
                    # Return a no-op so the caller can continue
                    return {"actions": [], "done": True}
            raise

    # ── Action execution ──────────────────────────────────────────────────────

    async def _execute_action(self, action: dict) -> str:
        """Execute a single action dict via Playwright. Returns a log string."""
        import asyncio

        page = self._browser.page
        atype: str = action.get("type", "")
        selector: str = action.get("selector", "")
        value: str = action.get("value", action.get("key", ""))
        timeout = DEFAULT_ACTION_TIMEOUT_MS

        logger.info("[DOM-AGENT exec] %s  selector=%r  value=%r", atype, selector, value)

        # ── Label-to-input resolver ──────────────────────────────────────────
        # GPT-mini often generates text=Label as the selector for fill/clear_fill.
        # Labels are not fillable — resolve to the nearby <input>, <textarea>,
        # <select>, or custom dropdown <button> via JavaScript.
        _is_label_selector = (
            selector.startswith("text=")
            or selector.startswith(":has-text(")
        )
        # Only resolve label→input for fill/clear_fill.
        # For click, text= / :has-text() selectors should target the element
        # directly (e.g. a tab, button, or link) — not redirect to a nearby input.
        if atype in ("fill", "clear_fill") and _is_label_selector:
            # Extract the label text from the selector
            if selector.startswith("text="):
                label_text = selector[5:]
            else:
                # :has-text("...") → extract inner text
                import re as _re
                m = _re.search(r':has-text\(["\'](.+?)["\']\)', selector)
                label_text = m.group(1) if m else selector
            try:
                resolved = await page.evaluate(r"""(labelText) => {
                    function bestSel(el) {
                        if (!el) return null;
                        if (el.id && /^[a-zA-Z_]/.test(el.id)) return '#' + el.id;
                        if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                        const ph = el.getAttribute('placeholder');
                        if (ph) return '[placeholder="' + ph + '"]';
                        const al = el.getAttribute('aria-label');
                        if (al) return '[aria-label="' + al + '"]';
                        const tid = el.getAttribute('data-testid') || el.getAttribute('data-test-id');
                        if (tid) return '[data-testid="' + tid + '"]';
                        return null;
                    }

                    // Find label containing this text (case-insensitive)
                    const lt = labelText.toLowerCase();
                    const labels = Array.from(document.querySelectorAll('label'));
                    let label = labels.find(l => l.innerText.trim().toLowerCase().includes(lt));

                    // Also try span/div that acts as label
                    if (!label) {
                        const spans = Array.from(document.querySelectorAll('span, div, p'));
                        label = spans.find(s => {
                            const t = s.innerText.trim().toLowerCase();
                            return t === lt || t.startsWith(lt);
                        });
                    }

                    if (!label) return null;

                    // Strategy 1: label[for] → target element by id
                    const forId = label.getAttribute('for');
                    if (forId) {
                        const target = document.getElementById(forId);
                        const s = bestSel(target);
                        if (s) return s;
                    }

                    // Strategy 2: input/textarea/select INSIDE the label
                    const inner = label.querySelector('input, textarea, select');
                    if (inner) {
                        const s = bestSel(inner);
                        if (s) return s;
                    }

                    // Strategy 3: next sibling is the input or a wrapper containing it
                    let sib = label.nextElementSibling;
                    if (sib) {
                        const tags = ['INPUT', 'TEXTAREA', 'SELECT'];
                        if (tags.includes(sib.tagName)) {
                            const s = bestSel(sib);
                            if (s) return s;
                        }
                        // Might be a Radix button trigger
                        if (sib.tagName === 'BUTTON') {
                            const s = bestSel(sib);
                            if (s) return s;
                        }
                        // Check inside wrapper
                        const nested = sib.querySelector('input, textarea, select, button[role="combobox"], button');
                        if (nested) {
                            const s = bestSel(nested);
                            if (s) return s;
                        }
                    }

                    // Strategy 4: walk UP to closest wrapper div, then look for form controls
                    for (let el = label.parentElement; el; el = el.parentElement) {
                        if (el.tagName === 'BODY') break;
                        // Only check divs that are small wrappers, not the whole page
                        if (el.children.length > 10) continue;
                        const controls = el.querySelectorAll(
                            'input:not([type="hidden"]), textarea, select, ' +
                            'button[role="combobox"], ' +
                            'button:has(span[data-slot="select-value"]), ' +
                            'button:has(span[data-placeholder])'
                        );
                        for (const ctrl of controls) {
                            if (ctrl === label || ctrl.contains(label)) continue;
                            const s = bestSel(ctrl);
                            if (s) return s;
                        }
                        // Stop at the first meaningful wrapper that has controls
                        if (controls.length > 0) break;
                    }

                    // Strategy 5: broader search — walk ALL siblings of the label
                    // (handles layouts where input is not the immediate next sibling)
                    if (label.parentElement) {
                        const siblings = Array.from(label.parentElement.children);
                        for (const sib of siblings) {
                            if (sib === label) continue;
                            const tags = ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'];
                            if (tags.includes(sib.tagName)) {
                                const s = bestSel(sib);
                                if (s) return s;
                            }
                            const nested = sib.querySelector(
                                'input:not([type="hidden"]), textarea, select, ' +
                                'button[role="combobox"], button'
                            );
                            if (nested) {
                                const s = bestSel(nested);
                                if (s) return s;
                            }
                        }
                    }

                    // Strategy 6: last resort — find ANY input/button whose aria-label
                    // or preceding text matches
                    const allInputs = document.querySelectorAll(
                        'input:not([type="hidden"]), textarea, select, button[role="combobox"]'
                    );
                    for (const inp of allInputs) {
                        const al = (inp.getAttribute('aria-label') || '').toLowerCase();
                        if (al && al.includes(lt)) {
                            const s = bestSel(inp);
                            if (s) return s;
                        }
                    }

                    return null;
                }""", label_text)

                if resolved:
                    logger.info("[DOM-AGENT exec] Resolved label %r → %r", selector, resolved)
                    selector = resolved
                    action = {**action, "selector": resolved}
                else:
                    logger.debug("[DOM-AGENT exec] Could not resolve label %r — keeping original", selector)
                    # For fill/clear_fill, text= selectors target the label not the input.
                    # Try a Playwright-based fallback: find label, then locate sibling input.
                    if atype in ("fill", "clear_fill"):
                        try:
                            fallback_sel = await page.evaluate(r"""(labelText) => {
                                const lt = labelText.toLowerCase();
                                // Try to find any input whose closest label-like ancestor contains the text
                                for (const inp of document.querySelectorAll('input:not([type="hidden"]), textarea')) {
                                    // Check parent chain for a label-like element containing the text
                                    let el = inp.parentElement;
                                    for (let depth = 0; el && depth < 5; depth++, el = el.parentElement) {
                                        const txt = (el.innerText || '').toLowerCase();
                                        if (txt.includes(lt) && txt.length < 200) {
                                            // Found a wrapper containing both the label text and the input
                                            if (inp.id && /^[a-zA-Z_]/.test(inp.id)) return '#' + inp.id;
                                            if (inp.name) return inp.tagName.toLowerCase() + '[name="' + inp.name + '"]';
                                            const ph = inp.getAttribute('placeholder');
                                            if (ph) return '[placeholder="' + ph + '"]';
                                            return null;
                                        }
                                    }
                                }
                                return null;
                            }""", label_text)
                            if fallback_sel:
                                logger.info("[DOM-AGENT exec] Fallback resolved label %r → %r", selector, fallback_sel)
                                selector = fallback_sel
                                action = {**action, "selector": fallback_sel}
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("[DOM-AGENT exec] Label resolution failed for %r: %s", selector, exc)

        # SAFETY: Prevent accidental record-creation clicks in NetSuite.
        try:
            cur_url = page.url.lower()
        except Exception:
            cur_url = ""
        _netsuite_blocked = ("create quote", "create order", "create sale", "new quote", "add to quote")
        # If we're on NetSuite and the planned action would click a blocked target, skip it.
        sel_text = (selector or "").lower()
        action_text = (action.get("text", "") or "").lower()
        if (
            "netsuite" in cur_url
            and atype == "click"
            and any(kw in sel_text or kw in action_text for kw in _netsuite_blocked)
        ):
            logger.warning("[DOM-AGENT] blocked click on NetSuite creation action: selector=%r text=%r", selector, action.get("text"))
            return f"SKIPPED_BLOCKED_CLICK({selector!r})"

        try:
            if atype == "fill":
                value = action.get("value", "")
                try:
                    loc = page.locator(selector).first
                    await loc.click(timeout=timeout)
                    # For CSS selectors (#id, .class) use the React nativeInputValueSetter trick:
                    # This directly updates React's controlled component state by bypassing React's
                    # batching and firing the synthetic onChange that React actually listens to.
                    # press_sequentially alone is insufficient for some React SPA login forms.
                    if selector.startswith("#") or selector.startswith("."):
                        try:
                            await page.evaluate("""(args) => {
                                const el = document.querySelector(args.sel);
                                if (!el) return;
                                const desc = Object.getOwnPropertyDescriptor(
                                    window.HTMLInputElement.prototype, 'value'
                                ) || Object.getOwnPropertyDescriptor(
                                    window.HTMLTextAreaElement.prototype, 'value'
                                );
                                if (desc && desc.set) desc.set.call(el, args.val);
                                el.dispatchEvent(new Event('input', {bubbles: true}));
                                el.dispatchEvent(new Event('change', {bubbles: true}));
                            }""", {"sel": selector, "val": value})
                        except Exception:
                            # JS approach failed — fall through to press_sequentially
                            await loc.press("Control+a")
                            await loc.press_sequentially(value, delay=30)
                    else:
                        # Non-CSS selector — use press_sequentially (fires real keyboard events)
                        await loc.press("Control+a")
                        await loc.press_sequentially(value, delay=30)
                except Exception as fill_exc:
                    if "not an <input>" in str(fill_exc) or "not editable" in str(fill_exc).lower():
                        # Non-editable element (likely a Radix/custom select) —
                        # fallback: click to open dropdown, then click the option text
                        logger.info("[DOM-AGENT exec] fill() on non-editable — fallback to click→option for %r", selector)
                        await page.locator(selector).first.click(timeout=_CLICK_FIRST_ATTEMPT_MS)
                        await page.wait_for_timeout(200)
                        option_text = value
                        await page.locator(f"text={option_text}").first.click(timeout=timeout)
                        await page.wait_for_timeout(DEFAULT_WAIT_AFTER_CLICK_MS)
                        return f"fill_fallback_click({selector!r}, {option_text!r})"
                    raise
                return f"fill({selector!r}, {value!r})"

            elif atype == "clear_fill":
                loc = page.locator(selector).first
                try:
                    await loc.click(timeout=_CLICK_FIRST_ATTEMPT_MS)
                except Exception as click_exc:
                    if "intercepts pointer events" in str(click_exc):
                        logger.info("[DOM-AGENT exec] pointer-event interceptor on clear_fill — pressing Escape, then retrying")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(200)  # reduced: 400→200ms
                        await loc.click(timeout=DEFAULT_ACTION_TIMEOUT_MS)
                    else:
                        raise
                try:
                    await loc.press("Control+a")
                    await loc.press("Delete")
                    await loc.fill(action.get("value", ""), timeout=timeout)
                except Exception as fill_exc:
                    if "not an <input>" in str(fill_exc) or "not editable" in str(fill_exc).lower():
                        # Non-editable element — the click already opened the dropdown,
                        # now click the desired option text
                        logger.info("[DOM-AGENT exec] clear_fill() on non-editable — clicking option text for %r", selector)
                        await page.wait_for_timeout(400)
                        option_text = action.get("value", "")
                        await page.locator(f"text={option_text}").first.click(timeout=timeout)
                        await page.wait_for_timeout(DEFAULT_WAIT_AFTER_CLICK_MS)
                        return f"clear_fill_fallback_click({selector!r}, {option_text!r})"
                    raise
                return f"clear_fill({selector!r}, {action.get('value','')!r})"

            elif atype == "click":
                # First attempt uses a shorter timeout so failures in CPQ/Radix bail quickly.
                try:
                    await page.locator(selector).first.click(timeout=_CLICK_FIRST_ATTEMPT_MS, no_wait_after=True)
                except Exception as click_exc:
                    exc_str = str(click_exc)
                    if "intercepts pointer events" in exc_str:
                        # Radix overlay — dismiss with Escape and retry with full timeout
                        logger.info("[DOM-AGENT exec] pointer-event interceptor — pressing Escape, retrying click")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(400)
                        try:
                            await page.locator(selector).first.click(timeout=DEFAULT_ACTION_TIMEOUT_MS, no_wait_after=True)
                        except Exception:
                            # Final fallback: force=True bypasses the overlay entirely
                            logger.info("[DOM-AGENT exec] retry also intercepted — using force=True")
                            await page.locator(selector).first.click(timeout=DEFAULT_ACTION_TIMEOUT_MS, no_wait_after=True, force=True)
                    else:
                        raise
                await page.wait_for_timeout(DEFAULT_WAIT_AFTER_CLICK_MS)
                return f"click({selector!r})"

            elif atype == "select_option":
                value = action.get("value", "")
                try:
                    # Try by visible label first
                    await page.select_option(selector, label=value, timeout=timeout)
                except Exception:
                    # Fallback: by value attribute
                    await page.select_option(selector, value=value, timeout=timeout)
                return f"select_option({selector!r}, {value!r})"

            elif atype == "check":
                await page.check(selector, timeout=timeout)
                return f"check({selector!r})"

            elif atype == "uncheck":
                await page.uncheck(selector, timeout=timeout)
                return f"uncheck({selector!r})"

            elif atype == "press":
                key = action.get("key", "Enter")
                await page.keyboard.press(key)
                await page.wait_for_timeout(200)  # reduced: 500→200ms
                return f"press({key!r})"

            elif atype == "wait":
                ms = int(action.get("ms", 1000))
                await asyncio.sleep(ms / 1000)
                return f"wait({ms}ms)"

            elif atype == "wait_selector":
                # Text-based selectors (text=Home, :has-text(...)) are unreliable as
                # page-load indicators — they can match arbitrary data content on the
                # page (e.g. "Exceptional Home Care" matching text=Home).
                # Skip them silently; URL/DOM checks in the login loop handle readiness.
                if selector.startswith("text=") or ":has-text(" in selector:
                    logger.debug("[DOM-AGENT] skipping text-based wait_selector: %r", selector)
                    return f"wait_selector_skipped({selector!r})"
                await page.wait_for_selector(selector, timeout=DEFAULT_WAIT_SELECTOR_TIMEOUT_MS)
                return f"wait_selector({selector!r})"

            else:
                logger.warning("[DOM-AGENT] unknown action type: %s", atype)
                return f"unknown({atype!r})"

        except Exception as exc:
            exc_str = str(exc)
            if any(msg in exc_str for msg in _BROWSER_CLOSED_MSGS):
                logger.warning("[DOM-AGENT] browser/page closed during [%s] — stopping", atype)
                raise  # propagate up so act() can stop cleanly
            logger.debug("[DOM-AGENT] action failed [%s] selector=%r: %s", atype, selector, exc)
            return f"FAILED:{atype}({selector!r}) — {exc}"

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try extracting the outermost JSON object (model may add preamble/postamble)
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.error("[DOM-AGENT] JSON parse failed: %s", raw[:400])
        # Return done=False so the loop retries rather than silently stopping
        return {"actions": [], "done": False}
