"""Smoke tests — verify graph construction and state flow without a real browser."""

from __future__ import annotations

import pytest

from dqe_agent.schemas.models import (
    ContactInfo,
    EmailPayload,
    HumanReview,
    OpportunityData,
    QuoteData,
    StructureData,
)


def test_opportunity_model_defaults():
    opp = OpportunityData()
    assert opp.bandwidth_unit == "Mbps"
    assert opp.exclude_ipl is False
    assert opp.contact.name == ""


def test_opportunity_model_full():
    opp = OpportunityData(
        buyer="1158 EQT Corporation",
        opportunity_id="OP-20080",
        exclude_ipl=False,
        bandwidth_amount=500,
        bandwidth_unit="Mbps",
        ip_address_assignment="/29",
        term_amount=2,
        term_unit="years",
        contact=ContactInfo(
            name="Andy Guley",
            email="aguley@eqt.com",
            phone="(412) 508-7076",
            quote_valid_till="27-03-2026 12:00",
        ),
        additional_info=(
            "New - EQT Corporation - 500 Mbps DIA @ 625 Liberty Ave, 28th Fl - 24 months"
        ),
    )
    assert opp.buyer == "1158 EQT Corporation"
    assert opp.bandwidth_amount == 500
    assert opp.contact.email == "aguley@eqt.com"


def test_structure_model():
    s = StructureData(
        country="United States",
        state="Pennsylvania",
        city="Pittsburgh",
        street="625 Liberty Ave (15222)",
        postcode="15222",
    )
    assert s.state == "Pennsylvania"


def test_quote_model_status():
    q = QuoteData()
    assert q.status == "draft"
    q.status = "finalized"
    assert q.status == "finalized"


def test_email_payload():
    e = EmailPayload(
        to=["aguley@eqt.com"],
        subject="Quote Q-12345",
        body="Dear Andy ...",
    )
    assert not e.approved
    e.approved = True
    assert e.approved


def test_human_review():
    h = HumanReview(step_name="review_info", question="Looks good?", response="yes", approved=True)
    assert h.approved


def test_graph_builds():
    """Ensure the graph can be constructed (without starting a real browser)."""
    from unittest.mock import AsyncMock, MagicMock

    from dqe_agent.graph import build_graph

    mock_browser = MagicMock()
    mock_browser.page = AsyncMock()

    graph = build_graph(mock_browser)
    # StateGraph should have all our nodes
    assert "netsuite_login" in graph.nodes
    assert "cpq_finalize" in graph.nodes
    assert "review_email" in graph.nodes
    assert "send_email" in graph.nodes
