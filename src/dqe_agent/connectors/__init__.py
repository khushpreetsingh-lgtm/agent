"""Site registry — thin shim over settings.sites.

Per-site Python files (netsuite.py, cpq.py, jira.py) are gone.
All site config now lives in .env + config.py settings.sites dict.

To add a new site: just add MYSITE_URL / _USERNAME / _PASSWORD to .env.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def list_connectors() -> list[str]:
    from dqe_agent.config import settings
    return list(settings.sites.keys())


def get_connector(site_id: str) -> dict:
    from dqe_agent.config import settings
    site = settings.get_site(site_id)
    if site is None:
        raise KeyError(f"Site '{site_id}' not configured. Add {site_id.upper()}_URL to .env")
    return site


def discover_connectors() -> None:
    """No-op — sites are discovered from settings.sites, not Python files."""
    from dqe_agent.config import settings
    sites = list(settings.sites.keys())
    logger.info("Sites configured: %s", sites)
