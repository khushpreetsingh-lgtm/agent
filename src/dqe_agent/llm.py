"""LLM factory — route to planner (large), executor (small), vision models.

Three tiers:
  get_planner_llm()  — large/smart model (runs once per task, cost OK)
  get_executor_llm() — small/fast model (runs per step, must be cheap)
  get_vision_llm()   — vision model for screenshot-based verification
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from dqe_agent.config import settings


def _make_llm(provider: str, model: str, max_tokens: int | None = None) -> BaseChatModel:
    p = provider.lower()
    kw: dict = {}
    if max_tokens:
        kw["max_tokens"] = max_tokens

    if p == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            **kw,
        )
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=settings.openai_api_key, **kw)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, **kw)

    raise ValueError(f"Unknown provider: '{provider}'. Use azure | openai | anthropic.")


@lru_cache(maxsize=1)
def get_planner_llm() -> BaseChatModel:
    """Large model for Planner — runs once per task to generate the full plan."""
    provider = settings.planner_provider or settings.llm_provider
    return _make_llm(provider, settings.planner_model, max_tokens=8192)


@lru_cache(maxsize=1)
def get_executor_llm() -> BaseChatModel:
    """Small fast model for Executor — runs per step, must be cheap."""
    return _make_llm(settings.llm_provider, settings.executor_model)


# DOMAgent uses the same executor-tier model
get_dom_llm = get_executor_llm


@lru_cache(maxsize=1)
def get_vision_llm() -> BaseChatModel:
    """Vision model for screenshot-based verification fallback."""
    return _make_llm(settings.llm_provider, settings.vision_model)


# Backward-compat alias used by nodes.py, email_tools.py, user_selection.py
get_llm = get_executor_llm
