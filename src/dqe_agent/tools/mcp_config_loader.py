"""MCP Config Loader — Load MCP servers from YAML config file."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_mcp_config(config_path: str | Path = "mcp_config.yaml") -> list[dict[str, Any]]:
    """Load MCP server configurations from YAML file.
    
    Args:
        config_path: Path to mcp_config.yaml
    
    Returns:
        List of MCP server configs ready to pass to load_mcp_tools()
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning("MCP config not found: %s", config_path)
        return []
    
    # Load .env into os.environ so ${VAR} references expand correctly.
    # pydantic-settings reads .env into model fields but does NOT set os.environ,
    # so _expand_env_vars (which uses os.getenv) would see None without this.
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(override=False)
    except ImportError:
        pass

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if not config or "servers" not in config:
            logger.warning("No servers defined in MCP config")
            return []
        
        mcp_servers = []
        for server in config["servers"]:
            # Skip disabled servers
            if not server.get("enabled", True):
                logger.info("MCP server '%s' is disabled, skipping", server.get("name", "unknown"))
                continue
            
            transport = server.get("transport", "stdio")
            server_config = server.get("config", {})
            
            # Expand environment variables in config
            server_config = _expand_env_vars(server_config)
            
            # Build config for load_mcp_tools()
            if transport == "stdio":
                mcp_servers.append({
                    "transport": "stdio",
                    "command": server_config.get("command", ""),
                    "args": server_config.get("args", []),
                    "env": server_config.get("env"),
                })
            elif transport == "sse":
                mcp_servers.append({
                    "transport": "sse",
                    "url": server_config.get("url", ""),
                    "headers": server_config.get("headers"),
                })
            
            logger.info("✅ Loaded MCP server config: %s (%s)", server.get("name"), transport)
        
        return mcp_servers
    
    except Exception as e:
        logger.error("Failed to load MCP config: %s", e)
        return []


def _expand_env_vars(config: dict | list | str | Any) -> Any:
    """Recursively expand ${VAR} in config values."""
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Expand ${VAR} patterns
        if "${" in config:
            import re
            def replace_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, f"${{{var_name}}}")  # Keep placeholder if not found
            return re.sub(r'\$\{([^}]+)\}', replace_var, config)
        return config
    else:
        return config
