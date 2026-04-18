"""Pydantic models that flow through the LangGraph state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# ── Contact ──────────────────────────────────────────────
class ContactInfo(BaseModel):
    name: str = Field("", description="Contact full name")
    email: str = Field("", description="Contact email address")
    phone: str = Field("", description="Contact phone number")
    quote_valid_till: str = Field("", description="Quote validity date, e.g. 27-03-2026 12:00")
    additional_info: str = ""


# ── Opportunity (extracted from NetSuite) ────────────────
class OpportunityData(BaseModel):
    buyer: str = Field("", description="Buyer name, e.g. '1158 EQT Corporation'")
    opportunity_id: str = Field("", description="Opportunity ID, e.g. 'OP-20080'")
    exclude_ipl: bool = Field(False, description="A-Exclude IPL flag")
    bandwidth_amount: float = Field(0, description="Bandwidth numeric value")
    bandwidth_unit: str = Field("Mbps", description="Mbps or Gbps")
    gateway: bool = Field(False, description="Gateway checkbox")
    ip_address_assignment: str = Field("", description="IP assignment, e.g. '/29'")
    bgp: bool = Field(False, description="BGP checkbox")
    additional_port: bool = Field(False, description="Additional Port checkbox")
    burst: bool = Field(False, description="Burst checkbox")
    additional_info: str = Field("", description="Free-text additional information")
    term_amount: int = Field(0, description="Contract term numeric value")
    term_unit: str = Field("years", description="Contract term unit")
    address: str = Field("", description="Street address parsed from Additional Info, e.g. '625 Liberty Ave (15222)'")
    contact: ContactInfo = Field(default_factory=ContactInfo)


# ── Structure / Address (extracted from NetSuite) ────────
class StructureData(BaseModel):
    country: str = Field("United States")
    state: str = Field("")
    city: str = Field("")
    street: str = Field("")
    postcode: str = Field("")


# ── Quote (built during CPQ flow) ────────────────────────
class QuoteData(BaseModel):
    quote_id: str = Field("", description="Generated quote ID from CPQ")
    price: str = Field("", description="Fetched price string")
    status: str = Field("draft", description="draft | priced | finalized | sent")
    needs_approval: bool = Field(False, description="Whether quote needs reviewer approval")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Email ────────────────────────────────────────────────
class EmailPayload(BaseModel):
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str = ""
    body: str = ""
    approved: bool = False


# ── Human Review ─────────────────────────────────────────
class HumanReview(BaseModel):
    """Carrier for human-in-the-loop decisions."""

    step_name: str = ""
    question: str = ""
    response: Optional[str] = None
    approved: bool = False
