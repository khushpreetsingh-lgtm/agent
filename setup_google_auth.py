"""One-time Google OAuth setup for workspace-mcp.

Run this script BEFORE starting the DQE Agent (so port 8000 is free).
It opens a browser, completes OAuth, and saves credentials to
~/.google_workspace_mcp/credentials/ where workspace-mcp picks them up.

Usage:
    python setup_google_auth.py
"""

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

# ── Config (read from .env so we never hard-code secrets here) ────────────────
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID     = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
USER_EMAIL    = os.environ.get("USER_GOOGLE_EMAIL", "")
REDIRECT_URI  = "http://localhost:8000/oauth2callback"

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ── OAuth callback handler ────────────────────────────────────────────────────
_auth_code: list[str] = []
_auth_done = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/oauth2callback":
            params = parse_qs(parsed.query)
            if "code" in params:
                _auth_code.append(params["code"][0])
                html = b"<h1 style='font-family:sans-serif;color:green'>Authorization successful! You can close this tab and return to the terminal.</h1>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)
                _auth_done.set()
            else:
                error = params.get("error", ["unknown"])[0]
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Authorization failed: {error}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress HTTP log noise


def _build_auth_url() -> str:
    params = {
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         " ".join(SCOPES),
        "access_type":   "offline",
        "prompt":        "consent",
    }
    return "https://accounts.google.com/o/oauth2/auth?" + urlencode(params)


def _exchange_code(code: str) -> dict:
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "grant_type":    "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _save_credentials(token_data: dict, email: str) -> None:
    creds_dir = Path.home() / ".google_workspace_mcp" / "credentials"
    creds_dir.mkdir(parents=True, exist_ok=True)

    creds = {
        "token":          token_data.get("access_token", ""),
        "refresh_token":  token_data.get("refresh_token", ""),
        "token_uri":      "https://oauth2.googleapis.com/token",
        "client_id":      CLIENT_ID,
        "client_secret":  CLIENT_SECRET,
        "scopes":         SCOPES,
        "universe_domain": "googleapis.com",
    }

    # Save under every filename pattern workspace-mcp might look for
    for name in [f"{email}.json", f"{email}_credentials.json", "credentials.json"]:
        path = creds_dir / name
        path.write_text(json.dumps(creds, indent=2))
        print(f"  Saved: {path}")

    # Also save the refresh token back to .env for reference
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        refresh = token_data.get("refresh_token", "")
        if "GOOGLE_REFRESH_TOKEN=" in text:
            lines = []
            for line in text.splitlines():
                if line.startswith("GOOGLE_REFRESH_TOKEN="):
                    lines.append(f"GOOGLE_REFRESH_TOKEN={refresh}")
                else:
                    lines.append(line)
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  Updated GOOGLE_REFRESH_TOKEN in .env")


def main():
    print("=" * 60)
    print("Google OAuth Setup for workspace-mcp")
    print("=" * 60)
    print(f"Client ID : {CLIENT_ID[:40]}...")
    print(f"User email: {USER_EMAIL}")
    print()

    # Start local callback server on port 8000
    try:
        server = HTTPServer(("localhost", 8000), _Handler)
    except OSError as e:
        print(f"ERROR: Cannot bind to port 8000: {e}")
        print("Make sure the DQE Agent is stopped before running this script.")
        sys.exit(1)

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print("Callback server started on http://localhost:8000")

    auth_url = _build_auth_url()
    print("\nOpening browser for Google authorization...")
    print(f"(If browser doesn't open, copy this URL manually:)\n{auth_url}\n")
    webbrowser.open(auth_url)

    print("Waiting for you to authorize in the browser (timeout: 3 min)...")
    _auth_done.wait(timeout=180)
    server.shutdown()

    if not _auth_code:
        print("\nERROR: Timed out waiting for authorization.")
        sys.exit(1)

    print("\nAuthorization code received. Exchanging for tokens...")
    try:
        token_data = _exchange_code(_auth_code[0])
    except Exception as e:
        print(f"ERROR exchanging code: {e}")
        sys.exit(1)

    if not token_data.get("refresh_token"):
        print("WARNING: No refresh_token in response. This may happen if you already")
        print("authorized before. Try revoking access at https://myaccount.google.com/permissions")
        print("then run this script again.")

    print("\nSaving credentials...")
    email = USER_EMAIL or "default"
    _save_credentials(token_data, email)

    print("\n✓ Done! Start the DQE Agent and ask for your meetings.")


if __name__ == "__main__":
    main()
