"""
masking_agent.py  (v2 — FIXED)
================================
AI-Privacy Guard Layer — SAR Narrative Generator (Team Baymax).

Role
----
Sits between the backend and the LLM pipeline.  Masks PII in
AgentState.structured_case before any LLM agent sees it, and
restores real values in the final SAR narrative after Agent 6.

Pipeline position
-----------------
  [Backend — Call 1]
    Agent 1 runs independently → writes sar_worthy, confidence_score,
    typology, risk_score to AgentState → returns to backend.

  [Backend — between calls]
    Backend fetches structured_case from RDS, populates AgentState.
    Backend calls  mask_structured_case()  →  masked state.

  [Backend — Call 2]
    Agents 2 → 3 → 4 → 5 → 6  all receive masked state.
    LLM (Bedrock / Claude) never sees real PII.

  [Backend — after Agent 6]
    Backend calls  unmask_sar_narrative()  on final SAR narrative.
    unmask() raises UnmaskIncompleteError if any token remains unreplaced.
    Real PII restored.  Narrative goes to Analyst Review UI.

  [Backend — after approval]
    delete_mask_map()  purges RDS rows for this case.

═══════════════════════════════════════════════════════════════
WHAT IS MASKED (v2 — corrected)
═══════════════════════════════════════════════════════════════

structured_case.customer  (single dict)
────────────────────────────────────────
  full_name           → <NAME_XXXXXXXX>
  dob_or_incorp_date  → <DOB_XXXXXXXX>

structured_case.accounts  (list of dicts)
──────────────────────────────────────────
  account_number      → <ACCOUNT_XXXXXXXX>

structured_case.transactions  (list of dicts)
───────────────────────────────────────────────
  counterparty_name    → <NAME_XXXXXXXX>
  counterparty_account → <ACCOUNT_XXXXXXXX>

NOT masked (intentionally):
──────────────────────────
  customer.declared_income    — needed by Agent 3 deviation_ratio feature.
  customer.nationality        — ISO-2 code; not personally identifiable alone.
  customer.occupation         — category label; not PII by itself.
  customer.risk_rating        — derived label, not source PII.
  account.avg_monthly_balance — needed by Agent 3 aggregation logic.
  txn.counterparty_country    — ISO-2 code; FATF logic in Agent 3 depends on it.

  All IDs (customer_id, account_id, txn_id) — internal surrogate keys;
  not PII under DPDPA/GDPR because they carry no intrinsic meaning.

═══════════════════════════════════════════════════════════════
HOW MASKING WORKS (token model)
═══════════════════════════════════════════════════════════════

1. For each (case_id, raw_value) pair:
   - Check DB for existing token.  If found → reuse it.
   - If not found → generate <ENTITY_8HEXCHARS> and INSERT.
   Same raw value always gets the same token within a case.
   "Rajesh Kumar" as counterparty_name and customer.full_name
   both get the SAME token.

2. The token is written back into the structured_case dict in-place.
   The caller receives a deep-copied, masked dict.  Original is untouched.

3. All LLM agents receive the masked dict.  The narrative they produce
   will contain tokens like <NAME_4A3F1B2C>.

4. unmask_sar_narrative():
   - Fetches all tokens for the case in ONE query.
   - String-replaces every token in the narrative.
   - Validates that NO tokens remain (raises UnmaskIncompleteError if any do).
   This validation is the guarantee: a token in the final SAR means
   something went wrong, and it never reaches the analyst silently.

═══════════════════════════════════════════════════════════════
WHAT IS NOT IN THIS FILE (out of scope for masking agent)
═══════════════════════════════════════════════════════════════

mask_text() for free-text unstructured strings is removed in v2.
The original regex NAME pattern matched "Suspicious Activity Report",
"Rapid Movement", "Reserve Bank" etc. — catastrophic false positives
that would corrupt the narrative.

If alert.description or other free-text fields need masking, the
correct approach is to pass those fields through mask_structured_case()
using the same structured registry, NOT a name-regex sweep.

═══════════════════════════════════════════════════════════════
BUGS FIXED vs v1
═══════════════════════════════════════════════════════════════

FIX-1  customer.full_name was NOT masked. It is the single most
       sensitive PII field and went directly into LLM prompts.
       Now: CUSTOMER_PII_FIELDS includes full_name → NAME.

FIX-2  customer.dob_or_incorp_date was NOT masked.
       Date of birth is explicit PII under DPDPA 2023 and GDPR.
       Now: CUSTOMER_PII_FIELDS includes dob_or_incorp_date → DOB.

FIX-3  account.account_number was NOT masked.
       10-digit bank account number is direct financial PII.
       Now: ACCOUNT_PII_FIELDS includes account_number → ACCOUNT.

FIX-4  mask_text() regex NAME pattern matched "Suspicious Activity
       Report", "Reserve Bank of India", "Rapid Movement", "Anti
       Money Laundering". Confirmed via test. These false positives
       would have corrupted the SAR narrative with masked tokens.
       Fix: mask_text() removed. Free-text goes through structured
       registry, not regex.

FIX-5  unmask_sar_narrative() had no validation. A missing DB row
       or accidental delete_mask_map() call before unmask would
       silently return a SAR narrative containing <NAME_XXXXXXXX>
       tokens that the analyst would never notice.
       Fix: UnmaskIncompleteError raised if any token remains after
       replacement. Backend catches and alerts.

FIX-6  mask_structured_case() now covers all three sub-dicts:
       customer, accounts[], transactions[]. v1 only did transactions.
"""

from __future__ import annotations

import copy
import logging
import re
import uuid
from typing import Any

import psycopg2
import psycopg2.errors
import psycopg2.extras

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ═══════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════

class UnmaskIncompleteError(RuntimeError):
    """
    Raised by unmask_sar_narrative() when one or more tokens remain
    in the narrative after replacement.

    This means a PII token was generated during masking but its
    DB row was missing at unmask time (deleted prematurely, DB error,
    etc.).  The narrative MUST NOT be shown to the analyst in this state.

    Attributes
    ----------
    remaining_tokens : list[str]  — the token strings that were not replaced.
    """
    def __init__(self, remaining_tokens: list[str]):
        self.remaining_tokens = remaining_tokens
        super().__init__(
            f"Unmask incomplete — {len(remaining_tokens)} token(s) "
            f"still present in narrative: {remaining_tokens}. "
            f"Do not surface this narrative to the analyst."
        )


# ═══════════════════════════════════════════════════════════════════
# PII FIELD REGISTRY
# ═══════════════════════════════════════════════════════════════════
#
# These are the ONLY fields that get tokenised.
# Derived directly from the actual data schema (customers.csv,
# accounts.csv, transactions.csv) cross-referenced against the
# AgentState.structured_case contract in state.py.
#
# Format: { field_name: entity_type_prefix }
# entity_type_prefix becomes the first segment of the token:
#   NAME    → <NAME_4A3F1B2C>
#   DOB     → <DOB_9D7E2A0F>
#   ACCOUNT → <ACCOUNT_1C3F5E7A>
# ═══════════════════════════════════════════════════════════════════

# structured_case["customer"]  (single dict)
CUSTOMER_PII_FIELDS: dict[str, str] = {
    "full_name":          "NAME",   # person name or company name
    "dob_or_incorp_date": "DOB",    # date of birth / incorporation date
}

# structured_case["accounts"]  (list of dicts)
ACCOUNT_PII_FIELDS: dict[str, str] = {
    "account_number": "ACCOUNT",   # 10-digit bank account number
}

# structured_case["transactions"]  (list of dicts)
TRANSACTION_PII_FIELDS: dict[str, str] = {
    "counterparty_name":    "NAME",    # external party name
    "counterparty_account": "ACCOUNT", # external party account number
}


# ═══════════════════════════════════════════════════════════════════
# TOKEN PATTERN — used by unmask validation to detect unreplaced tokens
# ═══════════════════════════════════════════════════════════════════

_TOKEN_PATTERN = re.compile(r"<(?:NAME|DOB|ACCOUNT)_[0-9A-F]{8}>")


# ═══════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_connection(db_config: dict[str, Any]) -> psycopg2.extensions.connection:
    """Open and return a new psycopg2 connection from a config dict."""
    conn_params = {
        "host":     db_config["host"],
        "port":     int(db_config.get("port", 5432)),
        "dbname":   db_config["dbname"],
        "user":     db_config["user"],
        "password": db_config["password"],
        "sslmode":  db_config.get("sslmode", "require"),
    }
    # Add sslrootcert if provided in config
    if "sslrootcert" in db_config:
        conn_params["sslrootcert"] = db_config["sslrootcert"]
    return psycopg2.connect(**conn_params)


def ensure_table(conn: psycopg2.extensions.connection) -> None:
    """Create pii_mask_map + indexes if they do not exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pii_mask_map (
                id          SERIAL       PRIMARY KEY,
                case_id     TEXT         NOT NULL,
                token       TEXT         NOT NULL UNIQUE,
                entity_type TEXT         NOT NULL,
                real_value  TEXT         NOT NULL,
                created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                CONSTRAINT  uq_case_value UNIQUE (case_id, real_value)
            );
            CREATE INDEX IF NOT EXISTS idx_pii_case_id ON pii_mask_map (case_id);
            CREATE INDEX IF NOT EXISTS idx_pii_token   ON pii_mask_map (token);
        """)
        conn.commit()


# ═══════════════════════════════════════════════════════════════════
# TOKEN MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def _make_token(entity_type: str) -> str:
    suffix = uuid.uuid4().hex[:8].upper()
    return f"<{entity_type}_{suffix}>"


def _get_or_create_token(
    conn: psycopg2.extensions.connection,
    case_id: str,
    entity_type: str,
    real_value: str,
) -> str:
    """
    Return the existing token for (case_id, real_value), or create one.

    Key invariant: same (case_id, real_value) → same token, always.
    This means customer.full_name = "Rajesh Kumar" and
    transactions[n].counterparty_name = "Rajesh Kumar" get the SAME
    token, so unmask restores both consistently in one pass.

    Uses two separate cursor contexts (SELECT then INSERT) so no cursor
    is reused across operations.  ON CONFLICT DO NOTHING handles concurrent
    Lambda invocations.  Retries up to 5× on token UNIQUE collision (rare).
    """
    # SELECT first — avoids write amplification on hot paths
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            "SELECT token FROM pii_mask_map WHERE case_id = %s AND real_value = %s",
            (case_id, real_value),
        )
        row = cur.fetchone()
        if row:
            return row["token"]

    # INSERT with token-collision retry (token column has UNIQUE constraint)
    for attempt in range(5):
        token = _make_token(entity_type)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pii_mask_map (case_id, token, entity_type, real_value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (case_id, real_value) DO NOTHING
                    RETURNING token
                    """,
                    (case_id, token, entity_type, real_value),
                )
                returned = cur.fetchone()
                conn.commit()
                if returned:
                    logger.debug(
                        "TOKEN CREATE  [%s]  '%s' → '%s'  case=%s",
                        entity_type, real_value, token, case_id,
                    )
                    return token

            # Concurrent writer won the INSERT race — fetch their token
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT token FROM pii_mask_map WHERE case_id = %s AND real_value = %s",
                    (case_id, real_value),
                )
                row = cur.fetchone()
                if row:
                    return row["token"]

        except psycopg2.errors.UniqueViolation:
            # token column collision (very rare) — retry with new UUID suffix
            conn.rollback()

    raise RuntimeError(
        f"Could not create unique token after 5 attempts  "
        f"(case_id={case_id!r}, entity_type={entity_type!r})"
    )


def _mask_field(
    value: Any,
    entity_type: str,
    case_id: str,
    conn: psycopg2.extensions.connection,
) -> Any:
    """
    Tokenise a single scalar field value.

    Non-strings and empty/whitespace-only values are returned unchanged.
    Integers (e.g. account_number stored as int) are converted to str
    before tokenising so DB lookup and replacement are consistent.
    """
    if value is None:
        return value

    # Normalise integer account numbers to str so token lookup is consistent
    if isinstance(value, (int, float)):
        value = str(int(value))

    if not isinstance(value, str) or not value.strip():
        return value

    return _get_or_create_token(conn, case_id, entity_type, value)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC: mask_structured_case
# ═══════════════════════════════════════════════════════════════════

def mask_structured_case(
    structured_case: dict[str, Any],
    case_id: str,
    conn: psycopg2.extensions.connection,
) -> dict[str, Any]:
    """
    Mask all PII in AgentState.structured_case before the LLM pipeline runs.

    Covers three sub-dicts per the state schema:
      structured_case["customer"]     → customer.full_name, .dob_or_incorp_date
      structured_case["accounts"]     → each account.account_number
      structured_case["transactions"] → each txn.counterparty_name, .counterparty_account

    Deep-copies the input so the caller's dict is never mutated.
    The returned dict is safe to pass to any LLM agent.

    Args:
        structured_case : AgentState.structured_case populated by the backend.
        case_id         : SAR case ID — partition key in RDS pii_mask_map.
        conn            : Open psycopg2 connection.

    Returns:
        New dict with all PII fields replaced by deterministic tokens.
        All non-PII fields are byte-identical to the input.
    """
    sc = copy.deepcopy(structured_case)

    # ── customer (single dict) ────────────────────────────────────────
    customer: dict[str, Any] = sc.get("customer", {})
    for field, entity_type in CUSTOMER_PII_FIELDS.items():
        if field in customer:
            original = customer[field]
            masked   = _mask_field(original, entity_type, case_id, conn)
            customer[field] = masked
            if masked != original:
                logger.info(
                    "MASK  customer.%s  [%s]  case=%s",
                    field, entity_type, case_id,
                )
    sc["customer"] = customer

    # ── accounts (list of dicts) ──────────────────────────────────────
    accounts: list[dict[str, Any]] = sc.get("accounts", [])
    for idx, acct in enumerate(accounts):
        for field, entity_type in ACCOUNT_PII_FIELDS.items():
            if field in acct:
                original = acct[field]
                masked   = _mask_field(original, entity_type, case_id, conn)
                acct[field] = masked
                if masked != original:
                    logger.info(
                        "MASK  accounts[%d].%s  [%s]  case=%s",
                        idx, field, entity_type, case_id,
                    )
    sc["accounts"] = accounts

    # ── transactions (list of dicts) ─────────────────────────────────
    transactions: list[dict[str, Any]] = sc.get("transactions", [])
    for idx, txn in enumerate(transactions):
        for field, entity_type in TRANSACTION_PII_FIELDS.items():
            if field in txn:
                original = txn[field]
                masked   = _mask_field(original, entity_type, case_id, conn)
                txn[field] = masked
                if masked != original:
                    logger.info(
                        "MASK  transactions[%d].%s  [%s]  case=%s",
                        idx, field, entity_type, case_id,
                    )
    sc["transactions"] = transactions

    return sc


# ═══════════════════════════════════════════════════════════════════
# PUBLIC: unmask_sar_narrative
# ═══════════════════════════════════════════════════════════════════

def unmask_sar_narrative(
    narrative: str | dict[str, Any],
    case_id: str,
    conn: psycopg2.extensions.connection,
) -> str | dict[str, Any]:
    """
    Restore all PII tokens in the final SAR output to real values.

    Fetches the complete token → real_value map in ONE query, then
    replaces all tokens in Python — zero per-token DB round-trips.

    After replacement, scans the result for any remaining tokens using
    the _TOKEN_PATTERN regex.  If any are found, raises UnmaskIncompleteError.
    The caller (backend) MUST catch this and NOT surface the narrative
    to the analyst UI.

    Args:
        narrative : SAR output from Agent 6 — str or dict.
                    Dicts are recursed deeply (handles nested structures).
        case_id   : SAR case ID.
        conn      : Open psycopg2 connection.

    Returns:
        Same type as input with all tokens replaced by real values.

    Raises:
        UnmaskIncompleteError : if any token remains after replacement.
    """
    # Fetch ALL tokens for this case in one query
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            "SELECT token, real_value FROM pii_mask_map WHERE case_id = %s",
            (case_id,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning(
            "unmask_sar_narrative: no mappings found for case_id=%s — "
            "either masking was never run or map was already deleted.",
            case_id,
        )
        # Still run the validation scan below to catch any stray tokens
        token_map: dict[str, str] = {}
    else:
        token_map = {r["token"]: r["real_value"] for r in rows}
        logger.info("UNMASK  %d token(s) loaded  case=%s", len(token_map), case_id)

    # ── replacement ──────────────────────────────────────────────────
    def _replace(text: str) -> str:
        for token, real_value in token_map.items():
            if token in text:
                text = text.replace(token, real_value)
                logger.info("UNMASK  '%s' → '%s'  case=%s", token, real_value, case_id)
        return text

    def _recurse(obj: Any) -> Any:
        if isinstance(obj, str):
            return _replace(obj)
        if isinstance(obj, dict):
            return {k: _recurse(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recurse(item) for item in obj]
        return obj

    result = _recurse(narrative)

    # ── completeness validation ───────────────────────────────────────
    # Serialize result back to a single string for scanning
    if isinstance(result, str):
        scan_target = result
    else:
        import json
        scan_target = json.dumps(result, default=str)

    remaining = _TOKEN_PATTERN.findall(scan_target)
    if remaining:
        logger.error(
            "UNMASK INCOMPLETE  %d token(s) remain  case=%s  tokens=%s",
            len(remaining), case_id, remaining,
        )
        raise UnmaskIncompleteError(remaining)

    return result


# ═══════════════════════════════════════════════════════════════════
# AUDIT HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_mask_map(
    case_id: str,
    conn: psycopg2.extensions.connection,
) -> list[dict[str, str]]:
    """
    Return all token ↔ real_value entries for a case (for audit trail).
    Ordered by creation time.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            SELECT token, entity_type, real_value, created_at
            FROM   pii_mask_map
            WHERE  case_id = %s
            ORDER  BY created_at
            """,
            (case_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def delete_mask_map(
    case_id: str,
    conn: psycopg2.extensions.connection,
) -> int:
    """
    Purge all token mappings for a case AFTER the approved SAR is
    archived and unmask has already been confirmed successful.

    Returns the number of rows deleted.

    IMPORTANT: Call this only AFTER unmask_sar_narrative() has returned
    without raising UnmaskIncompleteError.  Deleting before unmask
    will cause UnmaskIncompleteError on the unmask call.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM pii_mask_map WHERE case_id = %s RETURNING id",
            (case_id,),
        )
        deleted = cur.rowcount
        conn.commit()
    logger.info("delete_mask_map: %d row(s) deleted  case=%s", deleted, case_id)
    return deleted