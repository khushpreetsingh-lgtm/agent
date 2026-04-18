"""Centralised settings — loaded once from .env at import time.

Sites (formerly 'connectors') are configured here via env vars.
No per-site Python files needed — just add entries to .env and they appear.
"""
from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM: general reasoning ────────────────────────────────────────────
    llm_provider: str = Field("azure", description="azure | openai | anthropic")
    llm_model: str = Field("gpt-4o-mini", description="Default (executor) model")

    # ── LLM: planner (large/expensive — runs once per task) ───────────────
    planner_model: str = Field("gpt-4o", description="Model for Planner node")
    planner_provider: str = Field("", description="Override provider for planner (empty = use llm_provider)")

    # ── LLM: executor (small/fast — runs per step) ────────────────────────
    executor_model: str = Field("gpt-4o-mini", description="Model for Executor node")

    # ── LLM: vision fallback ──────────────────────────────────────────────
    vision_model: str = Field("gpt-4o", description="Model for screenshot-based verification")

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Azure OpenAI ──────────────────────────────────────────────────────
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = Field("2025-01-01")
    azure_openai_deployment: str = Field("gpt-4o-mini")

    # ── Browser viewport ──────────────────────────────────────────────────
    viewport_width: int = Field(1280)
    viewport_height: int = Field(800)

    # ── NetSuite ──────────────────────────────────────────────────────────
    netsuite_url: str = ""
    netsuite_username: str = ""
    netsuite_password: str = ""
    netsuite_totp_secret: str = ""

    # ── CPQ ───────────────────────────────────────────────────────────────
    cpq_url: str = ""
    cpq_username: str = ""
    cpq_password: str = ""

    # ── Jira ──────────────────────────────────────────────────────────────
    jira_url: str = ""
    jira_username: str = ""
    jira_password: str = ""
    jira_api_token: str = ""

    # ── Guardrails ────────────────────────────────────────────────────────
    max_steps: int = Field(20, description="Hard limit on agent steps per task")
    max_cost_usd: float = Field(2.0, description="Hard cost limit per task in USD")
    timeout_seconds: float = Field(60.0, description="Hard timeout per task in seconds")

    # ── Google / Workspace ───────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""
    user_google_email: str = ""

    # ── Demo / feature flags ──────────────────────────────────────────────
    disable_browser_tools: bool = Field(False, description="Disable browser tools — use only MCP tools (Jira/Gmail/Calendar)")

    # ── Browser ───────────────────────────────────────────────────────────
    headless: bool = False
    screenshot_enabled: bool = True
    screenshot_dir: Path = Path("./screenshots")

    # ── Site registry (replaces connectors/ Python files) ─────────────────
    # All site knowledge lives here + .env. No per-site Python files needed.
    # To add a new site: add SITE_URL / _USERNAME / _PASSWORD to .env only.
    @property
    def sites(self) -> dict[str, dict]:
        """All configured sites with login details. Keyed by site name."""
        result: dict[str, dict] = {}
        if self.netsuite_url:
            result["netsuite"] = {
                "display_name": "NetSuite",
                "url": self.netsuite_url,
                "username": self.netsuite_username,
                "password": self.netsuite_password,
                "totp_secret": self.netsuite_totp_secret,
                "success_url_fragment": "app/center",
            }
        if self.cpq_url:
            result["cpq"] = {
                "display_name": "CPQ",
                "url": self.cpq_url,
                "username": self.cpq_username,
                "password": self.cpq_password,
                "totp_secret": "",
                "success_url_fragment": "dashboard",
            }
        if self.jira_url:
            result["jira"] = {
                "display_name": "Jira",
                "url": self.jira_url,
                "username": self.jira_username,
                "password": self.jira_password,
                "totp_secret": "",
                "success_url_fragment": "jira/software",
            }
        return result

    def get_site(self, name: str) -> "dict | None":
        """Look up a site config by name. Returns None if not configured."""
        return self.sites.get(name.lower())


settings = Settings()
