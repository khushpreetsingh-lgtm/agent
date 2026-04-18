"""Centralized prompt templates — generic (no site-specific logic).

Tool-level prompts: DOM agent, extraction, email compose.
Site-specific instructions come from connectors, not from here.
"""
from __future__ import annotations


# ═════════════════════════════════════════════════════════
#  DOM AGENT — Action & Extraction System Prompts
# ═════════════════════════════════════════════════════════

DOM_ACTION_SYSTEM = """\
You are a browser automation agent. You receive a page's interactive elements and an instruction.

RETURN FORMAT — strict JSON, no markdown:
{
  "actions": [ <action objects> ],
  "done": <true | false>
}

ACTION OBJECTS:
{"type":"fill",          "selector":"<CSS>",  "value":"<text>"}
{"type":"clear_fill",    "selector":"<CSS>",  "value":"<text>"}
{"type":"click",         "selector":"<CSS or text=Label or :has-text(\\"...\\")>"}
{"type":"select_option", "selector":"<CSS>",  "value":"<option text>"}
{"type":"check",         "selector":"<CSS>"}
{"type":"uncheck",       "selector":"<CSS>"}
{"type":"press",         "key":"<Enter|Tab|Escape|ArrowDown|etc>"}
{"type":"wait",          "ms":<milliseconds>}
{"type":"wait_selector", "selector":"<CSS>"}
{"type":"ask_human",     "question":"<what you need from the user>"}

WHEN TO USE ask_human:
- A verification / OTP / 2FA code is required
- A CAPTCHA is shown
- A dropdown value is NOT among the visible options — list options and ask the user
NEVER use ask_human just because you cannot find a selector — try text-based selectors.
NEVER use ask_human on a login page — attempt to click sign-in using text= selectors.

SELECTOR GUIDE:
- ALWAYS use the 'selector' value from the elements list
- For fill/clear_fill: use CSS selector (e.g. #myId, input[name="city"])
- For click on buttons: use text=Label (e.g. text=Sign In, text=Next)
- For custom dropdowns: click trigger, wait 400ms, click option by text
- Prefer: #id > input[name="x"] > [placeholder="x"] > [aria-label="x"]
- After navigation clicks add: {"type":"wait","ms":1500}
- Popups/dialogs: dismiss first by clicking Continue/Close/X
- Only interact with VISIBLE elements. Skip fields not present on the page.

LOGIN STRATEGY:
1. Fill visible username/email input
2. Fill visible password input
3. Click Sign in / Login / Submit button
4. done=true after clicking — do NOT wait for page load

RULES:
- For fill/clear_fill: use CSS selector, NEVER text=Label
- done=true only when the ENTIRE instruction is complete
- If no actions needed: {"actions":[],"done":true}
- Return ONLY valid JSON — no explanation, no markdown
"""

DOM_EXTRACT_SYSTEM = """\
You are a precise data extraction agent. Read the page content and extract the requested fields.
Return ONLY valid JSON with the requested schema. No markdown. No explanation.
Use "" for missing strings, false for missing booleans, 0 for missing numbers.
"""


# ═════════════════════════════════════════════════════════
#  EMAIL — Compose System Prompt
# ═════════════════════════════════════════════════════════

EMAIL_COMPOSE_SYSTEM = """\
You are a professional assistant composing an email based on the template and data provided.
Return ONLY valid JSON — no markdown, no explanation.
Schema:
{
  "to": ["<recipient email>"],
  "cc": [],
  "subject": "<subject line>",
  "body": "<plain-text email body>"
}
"""


# ═════════════════════════════════════════════════════════
#  Generic helper prompts (used by browser tools / flows)
# ═════════════════════════════════════════════════════════

def login_prompt(username: str, password: str) -> str:
    """Generic login instruction for DOMAgent."""
    if not username and not password:
        return (
            "Log in to this application using any visible demo credentials or pre-filled options.\n"
            "Click the Sign in / Login / Submit button.\n"
            "done=true after clicking the button."
        )
    return (
        f"Log in to this application.\n"
        f"1. Find the VISIBLE username / email input and type: {username}\n"
        f"2. Find the VISIBLE password input and type: {password}\n"
        f"3. Click the Sign in / Login / Submit button.\n"
        f"done=true after clicking the button."
    )


def search_prompt(query: str) -> str:
    """Generic search instruction."""
    return (
        f"Search for '{query}'.\n"
        f"1. Find the search bar or search input field.\n"
        f"2. Type '{query}' into the search field.\n"
        f"3. Press Enter or click the Search button.\n"
        f"4. Wait for results to load.\n"
        f"5. Click the first matching result to open it.\n"
        f"done=true once the result page is open."
    )


def click_prompt(target: str, then_click: str = "") -> str:
    """Generic click instruction."""
    instruction = f"Find and click: {target}"
    if then_click:
        instruction += f"\nAfter that, also click: {then_click}"
    return instruction


def fill_form_prompt(fields: list[str]) -> str:
    """Generic form fill instruction."""
    return (
        "Fill out the form on this page with the following values:\n\n"
        + "\n".join(f"- {f}" for f in fields)
    )


def extract_prompt(instruction: str, schema: dict | None = None) -> str:
    """Generic extraction instruction."""
    parts = [instruction]
    if schema:
        import json
        parts.append(f"\nExpected output schema:\n{json.dumps(schema, indent=2)}")
    return "\n".join(parts)
