"""
lambda_handler.py
=================
AWS Lambda entry point for the masking_agent (AI-Privacy Guard Layer).

Two-call backend flow
---------------------
  Call 1  Backend → Agent 1 (separate Lambda)
          Agent 1 outputs: sar_worthy, confidence_score, typology, risk_score
          Backend receives AgentState.

  [Between calls]
          Backend fetches structured_case from RDS → populates AgentState.
          Backend calls THIS Lambda  action="mask".
          If sar_worthy is False → 400 (pipeline must have exited at Agent 1).
          Returns masked AgentState.

  Call 2  Backend → agentic graph  (Agents 2 → 3 → 4 → 5 → 6)
          All agents see tokens, not real PII.

  [After Agent 6]
          Backend calls THIS Lambda  action="unmask".
          Returns SAR narrative with real PII restored.

  [After approval]
          Backend calls  action="delete_map"  to purge RDS rows.

AgentState fields validated on action="mask"
--------------------------------------------
  Call-1 inputs  (written before Call 1, never changed):
    case_id           str   non-empty  e.g. "CASE-2026-007"
    s3_bucket         str   empty string allowed (dev fallback)
    s3_prefix         str
    transactions_csv  str   empty string in production

  Call-1 outputs (written by Agent 1):
    sar_worthy        bool  MUST be True — False → 400
    confidence_score  float 0.0–1.0
    typology          str   one of 6 known values (case-sensitive)
    risk_score        float 0.0–1.0

  Call-2 input (populated by backend from RDS):
    structured_case   dict
      customer        dict  (risk_rating, customer_type, nationality,
                             pep_flag, previous_sar_count)
      accounts        list  (account_id, account_type, account_age_days,
                             international_txn)
      transactions    list  (txn_id, txn_date, txn_type, amount, currency,
                             channel, counterparty_name, counterparty_account,
                             counterparty_country, is_high_value, velocity_score)

Environment variables
---------------------
  DB_HOST      RDS endpoint
  DB_PORT      default 5432
  DB_NAME      database name
  DB_USER      database user
  DB_PASSWORD  use AWS Secrets Manager in production
  DB_SSLMODE   default "require"
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg2

from masking_agent import (
    delete_mask_map,
    ensure_table,
    get_connection,
    get_mask_map,
    mask_structured_case,
    unmask_sar_narrative,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Known typologies from Agent 1 classifier — case-sensitive
VALID_TYPOLOGIES: frozenset[str] = frozenset({
    "Structuring",
    "Rapid Movement of Funds",
    "Funnel Account",
    "TBML",
    "Shell Company",
    "Round Tripping",
})

# Module-level cached connection — reused on warm Lambda invocations
_conn: psycopg2.extensions.connection | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_config() -> dict[str, Any]:
    return {
        "host":     os.environ["DB_HOST"],
        "port":     int(os.environ.get("DB_PORT", 5432)),
        "dbname":   os.environ["DB_NAME"],
        "user":     os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "sslmode":  os.environ.get("DB_SSLMODE", "require"),
    }


def _get_conn() -> psycopg2.extensions.connection:
    """Return a live connection; reconnect if the cached one is stale."""
    global _conn
    try:
        if _conn is None or _conn.closed:
            raise psycopg2.OperationalError("no connection")
        with _conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        _conn = get_connection(_db_config())
        ensure_table(_conn)
        logger.info("DB connection (re)established")
    return _conn


def _ok(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }


def _err(status: int, message: str) -> dict[str, Any]:
    logger.error("masking_agent [%d]: %s", status, message)
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }


def _parse_event(event: dict[str, Any]) -> dict[str, Any]:
    """Unwrap API Gateway proxy body string if present."""
    if "body" in event and isinstance(event["body"], str):
        return json.loads(event["body"])
    return event


# ---------------------------------------------------------------------------
# AgentState validation
# ---------------------------------------------------------------------------

def _validate_state(state: dict[str, Any]) -> str | None:
    """
    Validate the full AgentState against the contract from the state definition.
    Returns None if valid, or an error-message string if not.

    Checks (in order):
      case_id           non-empty str
      s3_bucket         str  (empty string = dev fallback, allowed)
      s3_prefix         str
      transactions_csv  str
      sar_worthy        bool — False triggers the pipeline-exit gate
      confidence_score  float 0.0–1.0
      typology          one of VALID_TYPOLOGIES (case-sensitive)
      risk_score        float 0.0–1.0
      structured_case   non-empty dict
        .customer       dict
        .accounts       list
        .transactions   list
    """
    # ── Call-1 inputs ────────────────────────────────────────────────────────
    case_id = state.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        return "state.case_id must be a non-empty string"

    for field in ("s3_bucket", "s3_prefix", "transactions_csv"):
        val = state.get(field)
        if not isinstance(val, str):
            return (
                f"state.{field} must be a string "
                f"(got {type(val).__name__!r})"
            )

    # ── sar_worthy gate ───────────────────────────────────────────────────────
    sar_worthy = state.get("sar_worthy")
    if not isinstance(sar_worthy, bool):
        return (
            f"state.sar_worthy must be bool "
            f"(got {type(sar_worthy).__name__!r}). "
            "Agent 1 must complete before the masking agent is called."
        )
    if sar_worthy is False:
        return (
            "state.sar_worthy is False — the pipeline should have exited "
            "after Agent 1. Masking is only valid for SAR-worthy cases."
        )

    # ── Call-1 outputs ────────────────────────────────────────────────────────
    confidence_score = state.get("confidence_score")
    if not isinstance(confidence_score, (int, float)) or not (0.0 <= float(confidence_score) <= 1.0):
        return (
            f"state.confidence_score must be a float in [0.0, 1.0] "
            f"(got {confidence_score!r})"
        )

    typology = state.get("typology")
    if typology not in VALID_TYPOLOGIES:
        return (
            f"state.typology {typology!r} is not recognised. "
            f"Valid values: {sorted(VALID_TYPOLOGIES)}"
        )

    risk_score = state.get("risk_score")
    if not isinstance(risk_score, (int, float)) or not (0.0 <= float(risk_score) <= 1.0):
        return (
            f"state.risk_score must be a float in [0.0, 1.0] "
            f"(got {risk_score!r})"
        )

    # ── structured_case schema ────────────────────────────────────────────────
    sc = state.get("structured_case")
    if not isinstance(sc, dict) or not sc:
        return "state.structured_case must be a non-empty dict"

    if not isinstance(sc.get("customer"), dict):
        return (
            f"state.structured_case.customer must be a dict "
            f"(got {type(sc.get('customer')).__name__!r})"
        )

    if not isinstance(sc.get("accounts"), list):
        return (
            f"state.structured_case.accounts must be a list "
            f"(got {type(sc.get('accounts')).__name__!r})"
        )

    if not isinstance(sc.get("transactions"), list):
        return (
            f"state.structured_case.transactions must be a list "
            f"(got {type(sc.get('transactions')).__name__!r})"
        )

    return None  # valid


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Masking agent Lambda entry point.

    Supported actions
    -----------------
    mask        Validate full AgentState (including Agent 1 outputs and
                sar_worthy gate), mask structured_case PII, return updated state.

    unmask      Restore all tokens → real PII in the SAR narrative.
                Single bulk DB read, all replacements done in Python.

    get_map     Return full token ↔ real_value audit map for a case.
                Attach to the Audit Package before SAR approval.

    delete_map  Purge token rows after SAR is approved and archived.
    """
    body   = _parse_event(event)
    action = body.get("action", "").strip().lower()

    if action not in ("mask", "unmask", "get_map", "delete_map"):
        return _err(400, "action must be: mask | unmask | get_map | delete_map")

    try:
        conn = _get_conn()

        # ── MASK ──────────────────────────────────────────────────────────────
        if action == "mask":
            state = body.get("state")
            if not isinstance(state, dict):
                return _err(400, "'state' must be a dict")

            err = _validate_state(state)
            if err:
                return _err(400, err)

            case_id: str = state["case_id"].strip()
            masked_sc = mask_structured_case(state["structured_case"], case_id, conn)

            # All state fields pass through — only structured_case is replaced
            masked_state: dict[str, Any] = {**state, "structured_case": masked_sc}
            token_count = len(get_mask_map(case_id, conn))

            logger.info(
                "action=mask  case=%s  typology=%s  sar_worthy=%s  "
                "confidence=%.3f  risk=%.3f  tokens=%d",
                case_id, state["typology"], state["sar_worthy"],
                state["confidence_score"], state["risk_score"], token_count,
            )
            return _ok({"state": masked_state, "token_count": token_count})

        # ── UNMASK ────────────────────────────────────────────────────────────
        elif action == "unmask":
            case_id = str(body.get("case_id", "")).strip()
            if not case_id:
                return _err(400, "'case_id' is required for action=unmask")
            narrative = body.get("narrative")
            if narrative is None:
                return _err(400, "'narrative' (str or dict) is required for action=unmask")

            restored = unmask_sar_narrative(narrative, case_id, conn)
            logger.info("action=unmask  case=%s", case_id)
            return _ok({"narrative": restored})

        # ── GET MAP ───────────────────────────────────────────────────────────
        elif action == "get_map":
            case_id = str(body.get("case_id", "")).strip()
            if not case_id:
                return _err(400, "'case_id' is required for action=get_map")
            mask_map = get_mask_map(case_id, conn)
            logger.info("action=get_map  case=%s  entries=%d", case_id, len(mask_map))
            return _ok({"case_id": case_id, "mask_map": mask_map})

        # ── DELETE MAP ────────────────────────────────────────────────────────
        elif action == "delete_map":
            case_id = str(body.get("case_id", "")).strip()
            if not case_id:
                return _err(400, "'case_id' is required for action=delete_map")
            deleted = delete_mask_map(case_id, conn)
            logger.info("action=delete_map  case=%s  rows=%d", case_id, deleted)
            return _ok({"case_id": case_id, "rows_deleted": deleted})

    except psycopg2.Error as db_err:
        logger.exception("DB error in masking_agent")
        return _err(500, f"Database error: {db_err}")
    except Exception as exc:
        logger.exception("Unexpected error in masking_agent")
        return _err(500, str(exc))
