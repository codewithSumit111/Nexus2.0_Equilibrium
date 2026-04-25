"""
NEXUS 2.0 — FastAPI Application Entry Point
SAR Narrative Generator with Glass-Box Audit Trail
"""

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import init_db, insert_case, update_case, get_case, get_all_cases
from models import AlertSubmission, SARResult, CaseSummary, EquilibriumAuditLogEntry
from sample_data import get_sample_alert
from workflow import run_pipeline

# ── Initialise app ──
app = FastAPI(
    title="NEXUS 2.0 — SAR Narrative Generator",
    description="Multi-agent LangGraph pipeline with FinCEN-compliant SAR generation and glass-box audit trail.",
    version="0.1.0",
)

# ── CORS (allow React dev server) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle ──
@app.on_event("startup")
def on_startup():
    init_db()
    print("[NEXUS] Database initialised. Server ready.")


# ──────────────────────────────────────────────
#  Health & Utility
# ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "nexus-2.0"}


@app.get("/sample-alert/{case_type}")
def sample_alert(case_type: str = "elder_exploitation"):
    """Return a sample alert payload for testing / demo."""
    try:
        data = get_sample_alert(case_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return data


# ──────────────────────────────────────────────
#  SAR Pipeline Endpoints
# ──────────────────────────────────────────────

@app.post("/api/generate-sar", response_model=SARResult)
def generate_sar(submission: AlertSubmission):
    """
    Submit an alert to the full LangGraph SAR generation pipeline.
    Runs: mask_pii → router → typology → narrative → compliance_judge → unmask
    """
    # Check for duplicate
    existing = get_case(submission.case_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Case {submission.case_id} already exists. Use GET /api/cases/{{case_id}} to retrieve it.",
        )

    # Persist initial row
    insert_case(submission.case_id, submission.alert_data)
    update_case(submission.case_id, status="processing")

    try:
        # ── Run the LangGraph pipeline ──
        result = run_pipeline(submission.case_id, submission.alert_data)

        # ── Persist all results back to the DB ──
        update_case(
            submission.case_id,
            masked_data=result["masked_data"],
            pii_mapping=result["pii_mapping"],
            detected_typology=result["detected_typology"],
            typology_analysis=result["typology_analysis"],
            draft_sar_masked=result["draft_sar_masked"],
            final_sar_clean=result["final_sar_clean"],
            compliance_score=result["compliance_score"],
            audit_log=result["audit_log"],
            status="completed",
        )

        return SARResult(
            case_id=result["case_id"],
            detected_typology=result["detected_typology"],
            typology_analysis=result["typology_analysis"],
            draft_sar_masked=result["draft_sar_masked"],
            final_sar_clean=result["final_sar_clean"],
            compliance_score=result["compliance_score"],
            audit_log=result["audit_log"],
            status="completed",
        )

    except Exception as e:
        # Log the error and mark the case as failed
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[NEXUS ERROR] Pipeline failed for {submission.case_id}: {error_msg}")
        traceback.print_exc()
        update_case(
            submission.case_id,
            status="failed",
            audit_log=[{"step": "pipeline_error", "action": "Pipeline failed", "data": error_msg}],
        )
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {error_msg}")


@app.post("/nexus/v2/generate-sar")
def generate_sar_v2(submission: AlertSubmission):
    """
    Submit an alert to the full LangGraph SAR generation pipeline (v2).
    Returns both equilibrium_draft and equilibrium_audit_log.
    """
    existing = get_case(submission.case_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Case {submission.case_id} already exists. Use GET /api/cases/{{case_id}} to retrieve it.",
        )

    insert_case(submission.case_id, submission.alert_data)
    update_case(submission.case_id, status="processing")

    try:
        result = run_pipeline(submission.case_id, submission.alert_data)

        update_case(
            submission.case_id,
            masked_data=result["masked_data"],
            pii_mapping=result["pii_mapping"],
            detected_typology=result["detected_typology"],
            typology_analysis=result["typology_analysis"],
            draft_sar_masked=result["draft_sar_masked"],
            final_sar_clean=result["final_sar_clean"],
            compliance_score=result["compliance_score"],
            audit_log=result["audit_log"],
            status="completed",
        )

        return {
            "case_id": result["case_id"],
            "equilibrium_draft": result["final_sar_clean"],
            "equilibrium_audit_log": result.get("equilibrium_audit_log", []),
            "detected_typology": result["detected_typology"],
            "compliance_score": result["compliance_score"],
            "status": "completed",
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[NEXUS ERROR] Pipeline failed for {submission.case_id}: {error_msg}")
        traceback.print_exc()
        update_case(
            submission.case_id,
            status="failed",
            audit_log=[{"step": "pipeline_error", "action": "Pipeline failed", "data": error_msg}],
        )
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {error_msg}")


@app.get("/api/cases", response_model=list[CaseSummary])
def list_cases():
    """List all SAR cases (summary view)."""
    return get_all_cases()


@app.get("/api/cases/{case_id}", response_model=SARResult)
def get_sar_case(case_id: str):
    """Retrieve full SAR result + audit trail for a case."""
    row = get_case(case_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return SARResult(
        case_id=row["case_id"],
        detected_typology=row.get("detected_typology", ""),
        typology_analysis=row.get("typology_analysis", ""),
        draft_sar_masked=row.get("draft_sar_masked", ""),
        final_sar_clean=row.get("final_sar_clean", ""),
        compliance_score=row.get("compliance_score", 0),
        audit_log=row.get("audit_log", []),
        status=row.get("status", "pending"),
    )


# ──────────────────────────────────────────────
#  Run with: uvicorn main:app --reload --port 8000
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
