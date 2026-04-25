"""
agents/agent1_ingestion.py — Case Intake & Structuring
=======================================================
Thin LangGraph wrapper around the existing ML pipeline.

What this does:
  1. Resolves input source from state:
       a) S3 path  — downloads all 4 files from s3_bucket/s3_prefix/ to a temp dir
       b) Local fallback — infers input_dir from transactions_csv (dev/testing only)
  2. Calls pipeline.run_pipeline(input_dir) which returns the classification result.
  3. Parses that result into SARAgentState fields.
  4. Sets the sar_worthy flag — if False, the graph exits here and no LLM is called.

S3 is the ONLY external storage interaction in the entire pipeline.
"""

import logging
import os
import tempfile
from datetime import datetime, timezone

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ML_DIR = _PROJECT_ROOT / "ml" / "hack-o-hire"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from state import SARAgentState
from privacy_guard import NexusPrivacyGuard

# ── Import your existing ML pipeline ──────────────────────────────────
# pipeline.py lives at ml/hack-o-hire/pipeline.py
# It already handles: CSV parsing, feature extraction, classification,
# structured_case assembly.
# The only contract this agent needs: pipeline.run_pipeline(paths) → dict
from pipeline import run_pipeline as run_ml_pipeline

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Agent node                                                          #
# ------------------------------------------------------------------ #

def agent1_ingestion(state: SARAgentState) -> SARAgentState:
    """
    LangGraph node for Agent 1.

    Call 1 inputs (set by backend before graph.invoke):
        case_id    : str — e.g. "CASE-2026-007"
        s3_bucket  : str — S3 bucket containing the 4 input files
        s3_prefix  : str — S3 key prefix (folder) inside the bucket

    Dev/testing fallback (when s3_bucket is absent):
        transactions_csv : str — local path; input_dir is inferred from its parent

    Sets on return:
        sar_worthy, confidence_score, typology, risk_score, error_log
    Note: structured_case is NOT populated here — it arrives on Call 2 from backend.
    """
    case_id: str = state.get("case_id", f"CASE-UNKNOWN-{_ts()}")
    logger.info(f"Agent 1 — starting ingestion for {case_id}")

    errors: list[str] = []
    _tmp_dir = None  # track temp dir for cleanup

    # ── 1. Resolve input_dir ───────────────────────────────────────
    s3_bucket = state.get("s3_bucket", "")
    s3_prefix = state.get("s3_prefix", "")

    if s3_bucket:
        # Production path: download all 4 files from S3 into a temp dir
        input_dir, _tmp_dir, s3_errors = _fetch_from_s3(s3_bucket, s3_prefix, case_id)
        errors.extend(s3_errors)
        if not input_dir:
            return _error_fallback(state, errors)
    else:
        # Dev/testing fallback: infer input_dir from transactions_csv local path
        txn_csv = state.get("transactions_csv", "")
        if txn_csv:
            input_dir = str(Path(txn_csv).parent)
            logger.info(f"Agent 1 [{case_id}] — using local fallback: {input_dir}")
        else:
            msg = (
                f"[{_ts()}] Agent1 [{case_id}]: "
                "No s3_bucket or transactions_csv in state. Cannot run ML pipeline."
            )
            logger.error(msg)
            errors.append(msg)
            return _error_fallback(state, errors)

    if not input_dir:
        return _error_fallback(state, errors)

    # ── 2. Call your existing ML pipeline ─────────────────────────
    try:
        result: dict = run_ml_pipeline(
            input_dir=input_dir,
            output_csv=Path(input_dir) / "data_aggregated.csv" if input_dir else None
        )
    except Exception as e:
        msg = f"[{_ts()}] Agent1 [{case_id}]: ML pipeline raised an exception — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        result = {
            "sar_worthy":       True,
            "confidence_score": 0.5,
            "typology":         "Unknown",
            "risk_score":       0.5,
        }
    finally:
        # Clean up temp dir if we created one for S3 downloads
        if _tmp_dir:
            import shutil
            shutil.rmtree(_tmp_dir, ignore_errors=True)

    # ── 3. Parse pipeline output ───────────────────────────────────
    sar_worthy       = bool(result.get("sar_worthy", True))
    confidence_score = float(result.get("confidence_score", 0.5))
    typology         = str(result.get("typology", "Unknown"))
    risk_score       = float(result.get("risk_score", 0.5))

    logger.info(
        f"Agent 1 — {case_id} | "
        f"SAR-worthy: {sar_worthy} (conf: {confidence_score:.2f}) | "
        f"Typology: {typology} | Risk: {risk_score:.2f}"
    )

    if not sar_worthy:
        logger.info(
            f"Agent 1 — {case_id} is NON-SAR. "
            "Pipeline exits here — no LLM will be called."
        )

    # ── 4. Apply PII masking before data hits LLM ─────────────────
    guard = NexusPrivacyGuard()
    masked_errors = []
    pii_mapping = {}
    for error in errors:
        masked_error, mapping = guard.mask_pii(error)
        masked_errors.append(masked_error)
        pii_mapping.update(mapping)

    # ── 5. Write to state ──────────────────────────────────────────
    # structured_case is NOT set here — it arrives on Call 2 from backend.
    return {
        **state,
        "sar_worthy":       sar_worthy,
        "confidence_score": round(confidence_score, 4),
        "typology":         typology,
        "risk_score":       round(risk_score, 4),
        "error_log":        errors,
        "masked_error_log": masked_errors,
        "pii_mapping":      pii_mapping,
        # Initialise all downstream fields so LangGraph
        # never hits a KeyError on a conditional bypass path.
        "structured_case":       state.get("structured_case", {}),
        "plan":                  state.get("plan", {}),
        "requires_enrichment":   state.get("requires_enrichment", False),
        "triggered_rules":       state.get("triggered_rules", []),
        "quantified_indicators": state.get("quantified_indicators", {}),
        "cognitive_event_flow":  state.get("cognitive_event_flow", {}),
        "enrichment_data":       state.get("enrichment_data", {}),
        "sar_draft":             state.get("sar_draft", ""),
        "sar_context_tree":      state.get("sar_context_tree", {}),
        "reasoning_traces":      state.get("reasoning_traces", []),
        "compliance_passed":     state.get("compliance_passed", False),
        "compliance_issues":     state.get("compliance_issues", []),
        "quality_score":         state.get("quality_score", 0.0),
        "revision_count":        state.get("revision_count", 0),
    }


# ------------------------------------------------------------------ #
# Conditional edge — read by LangGraph graph wiring in pipeline.py   #
# ------------------------------------------------------------------ #

def route_after_ingestion(state: SARAgentState) -> str:
    """
    Called by LangGraph after agent1_ingestion completes.

    Returns:
        "planner"  — SAR-worthy case, continue to Agent 2.
        "end"      — Non-SAR case, stop here, no LLM called.

    Wire this in pipeline.py:
        graph.add_conditional_edges(
            "agent1_ingestion",
            route_after_ingestion,
            {"planner": "agent2_planner", "end": END},
        )
    """
    return "planner" if state.get("sar_worthy", False) else "end"


# ------------------------------------------------------------------ #
# S3 fetch helper                                                     #
# ------------------------------------------------------------------ #

def _fetch_from_s3(
    bucket: str, prefix: str, case_id: str
) -> tuple[str, str, list[str]]:
    """
    Download all 4 input files from S3 into a temp directory.
    Discovers files by filename — subfolder names don't matter.
    The backend can organise files however they want inside the prefix.
    """
    errors: list[str] = []
    try:
        import boto3
    except ImportError:
        msg = f"[{_ts()}] Agent1 [{case_id}]: boto3 not installed — cannot fetch from S3."
        logger.error(msg)
        return "", "", [msg]

    tmp_dir = tempfile.mkdtemp(prefix=f"sar_{case_id}_")

    from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    session = boto3.Session(
        aws_access_key_id     = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name           = AWS_REGION,
    )
    s3 = session.client("s3")
    prefix = prefix.rstrip("/")

    # Target filenames — matched by filename regardless of subfolder
    _TARGET_FILENAMES = {
        "account_transactions.csv",
        "transaction_alerts.csv",
        "customer_kyc.json",
        "case_management.json",
    }

    # List all objects under the prefix
    try:
        paginator = s3.get_paginator("list_objects_v2")
        all_keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                all_keys.append(obj["Key"])
    except Exception as e:
        msg = f"[{_ts()}] Agent1 [{case_id}]: Failed to list s3://{bucket}/{prefix} — {e}"
        logger.error(msg)
        return "", tmp_dir, [msg]

    if not all_keys:
        msg = f"[{_ts()}] Agent1 [{case_id}]: No objects found at s3://{bucket}/{prefix}"
        logger.error(msg)
        return "", tmp_dir, [msg]

    logger.info(f"Agent 1 [{case_id}] — found {len(all_keys)} objects under prefix")

    # Map filename → S3 key (last match wins if duplicates)
    filename_to_key: dict[str, str] = {}
    for key in all_keys:
        fname = key.split("/")[-1]
        if fname in _TARGET_FILENAMES:
            filename_to_key[fname] = key
            logger.info(f"Agent 1 [{case_id}] — matched {fname} → {key}")

    # Download each matched file
    for fname, key in filename_to_key.items():
        local_path = os.path.join(tmp_dir, fname)
        try:
            s3.download_file(bucket, key, local_path)
            logger.info(f"Agent 1 [{case_id}] — downloaded s3://{bucket}/{key}")
        except Exception as e:
            msg = f"[{_ts()}] Agent1 [{case_id}]: Failed to download s3://{bucket}/{key} — {e}"
            logger.error(msg)
            errors.append(msg)

    # Warn about any missing target files
    for fname in _TARGET_FILENAMES:
        if fname not in filename_to_key:
            msg = f"[{_ts()}] Agent1 [{case_id}]: File not found in S3 prefix: {fname}"
            logger.warning(msg)
            errors.append(msg)

    downloaded = [f for f in _TARGET_FILENAMES if os.path.exists(os.path.join(tmp_dir, f))]
    if not downloaded:
        logger.error(f"Agent 1 [{case_id}]: No files downloaded from S3 — all 4 failed")
        return "", tmp_dir, errors

    return tmp_dir, tmp_dir, errors


def _error_fallback(state: SARAgentState, errors: list[str]) -> SARAgentState:
    """S3 or pipeline failure — mark as not sar_worthy so the pipeline stops cleanly."""
    # Apply PII masking to errors even in fallback case
    guard = NexusPrivacyGuard()
    masked_errors = []
    pii_mapping = {}
    for error in errors:
        masked_error, mapping = guard.mask_pii(error)
        masked_errors.append(masked_error)
        pii_mapping.update(mapping)

    return {
        **state,
        "sar_worthy":       False,
        "confidence_score": 0.0,
        "typology":         "Unknown",
        "risk_score":       0.0,
        "error_log":        errors,
        "masked_error_log": masked_errors,
        "pii_mapping":      pii_mapping,
    }


# ------------------------------------------------------------------ #
# Helper                                                              #
# ------------------------------------------------------------------ #

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()