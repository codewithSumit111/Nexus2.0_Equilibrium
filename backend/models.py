"""
NEXUS 2.0 — Pydantic models for API request/response contracts.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Models ──

class AlertSubmission(BaseModel):
    """Payload sent by the frontend to kick off SAR generation."""
    case_id: str = Field(..., description="Unique case identifier")
    alert_data: dict = Field(..., description="Raw alert / transaction data")


# ── Response Models ──

class AuditLogEntry(BaseModel):
    """Single entry in the glass-box audit trail."""
    step: str
    action: str
    data: str  # stringified for safety


class EquilibriumAuditLogEntry(BaseModel):
    """Single entry in the Equilibrium audit trail."""
    step: str
    action: str
    evidence: str


class SARResult(BaseModel):
    """Full result returned after the LangGraph pipeline completes."""
    case_id: str
    detected_typology: str = ""
    typology_analysis: str = ""
    draft_sar_masked: str = ""
    final_sar_clean: str = ""
    compliance_score: int = 0
    audit_log: list[AuditLogEntry] = []
    status: str = "pending"


class CaseSummary(BaseModel):
    """Lightweight row for the case-list endpoint."""
    id: int
    case_id: str
    detected_typology: Optional[str] = ""
    compliance_score: int = 0
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""
