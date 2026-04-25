"""
NEXUS 2.0 — LangGraph State Definition
TypedDict that flows through every node in the SAR generation pipeline.
"""

from typing import TypedDict


class SARState(TypedDict):
    """
    Shared state passed through the LangGraph workflow.
    Every key below is read/written by one or more nodes.
    """

    # ── Identity ──
    case_id: str

    # ── Raw & Masked Data ──
    raw_alert_data: dict          # Original alert payload from the frontend
    masked_data: dict             # Alert with PII replaced by placeholders
    pii_mapping: dict             # {placeholder: original_value} for unmasking

    # ── Analysis ──
    detected_typology: str        # e.g. "Elder Financial Exploitation", "Structuring"
    typology_analysis: str        # Detailed pattern analysis from the Typology Node

    # ── SAR Narrative ──
    draft_sar_masked: str         # FinCEN-compliant draft with PII still masked
    final_sar_clean: str          # Final narrative with real PII restored

    # ── Compliance ──
    compliance_score: int         # 1-10 grade from the Compliance Judge

    # ── Glass-Box Audit Trail ──
    audit_log: list               # list of {"step": str, "action": str, "data": str}
    equilibrium_audit_log: list   # list of {"step": str, "action": str, "evidence": str}
