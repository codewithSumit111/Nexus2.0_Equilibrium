"""
agents/agent2_planner.py — Planning & Orchestration Agent
==========================================================
Stage 2 of the 7-agent SAR pipeline.

What this agent does:
  1. Reads the typology, risk scores, and structured case data from state.
  2. Sends a carefully constructed prompt to the LLM (Groq in dev,
     Bedrock in prod — swapped via config.get_llm()).
  3. Parses the LLM's JSON response into a routing plan.
  4. Writes plan and requires_enrichment to state.

The plan tells every downstream agent:
  - which analysis modules are active (agent 3 reads this)
  - whether external enrichment is needed (graph edge reads this)
  - case priority (HIGH / MEDIUM / LOW)
  - a one-sentence rationale for the audit trail

If the LLM call fails or returns malformed JSON, the agent falls back
to a deterministic rule-based plan so the pipeline never stops dead.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from state import SARAgentState
from utils.json_parser import safe_parse_json

logger = logging.getLogger(__name__)

# ── Load prompt from file (edit the .txt, not this file) ──────────────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.txt"

def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Planner prompt not found at {_PROMPT_PATH}. "
            "Create prompts/planner.txt before running Agent 2."
        )
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


# ── Valid module names (guards against LLM hallucinating module names) ─
VALID_MODULES = {
    "structuring_analysis",
    "velocity_analysis",
    "network_analysis",
    "international_analysis",
    "kyc_escalation",
    "account_behaviour_analysis",
}

VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}


# ------------------------------------------------------------------ #
# Agent node                                                          #
# ------------------------------------------------------------------ #

def agent2_planner(state: SARAgentState) -> SARAgentState:
    """
    LangGraph node for Agent 2.

    Reads from state:
        case_id, typology, confidence_score, risk_score, structured_case

    Writes to state:
        plan, requires_enrichment, error_log
    """
    case_id: str = state.get("case_id", "UNKNOWN")
    logger.info(f"Agent 2 — building routing plan for {case_id}")

    errors: list[str] = []

    # ── 1. Extract relevant fields from structured_case ────────────
    structured_case = state.get("structured_case", {})
    customer     = structured_case.get("customer", {})
    accounts     = structured_case.get("accounts", [])
    transactions = structured_case.get("transactions", [])
    summary      = structured_case.get("summary", {})

    primary_account = accounts[0] if accounts else {}

    # Build compact summaries — we don't want to dump the full raw
    # data into the prompt (token cost + PII risk).
    customer_summary = {
        "customer_type":      customer.get("customer_type", "individual"),
        "risk_rating":        customer.get("risk_rating", "LOW"),
        "nationality":        customer.get("nationality", "IN"),
        "pep_flag":           bool(customer.get("pep_flag", False)),
        "previous_sar_count": int(customer.get("previous_sar_count", 0) or 0),
    }

    account_summary = {
        "account_type":      primary_account.get("account_type", "savings"),
        "account_age_days":  primary_account.get("account_age_days", 0),
        "international_txn": bool(primary_account.get("international_txn", False)),
    }

    # Compute basic transaction stats if not already in summary
    txn_amounts = [float(t.get("amount", 0)) for t in transactions if t.get("amount")]
    txn_count   = summary.get("transaction_count", len(transactions))
    total_amt   = summary.get("total_txn_amount", sum(txn_amounts))
    avg_amt     = total_amt / max(txn_count, 1)

    debits  = [float(t.get("amount", 0)) for t in transactions if t.get("txn_type") == "DEBIT"]
    credits = [float(t.get("amount", 0)) for t in transactions if t.get("txn_type") == "CREDIT"]
    fund_exit_ratio = sum(debits) / max(sum(credits), 1e-6)
    fund_exit_ratio = min(round(fund_exit_ratio, 4), 1.5)

    velocities  = [float(t.get("velocity_score", 0)) for t in transactions]
    burst_score = sum(1 for v in velocities if v > 0.75) / max(txn_count, 1)

    transaction_summary = {
        "txn_count":        txn_count,
        "total_txn_amount": round(total_amt, 2),
        "avg_txn_amount":   round(avg_amt, 2),
        "fund_exit_ratio":  fund_exit_ratio,
        "burst_score":      round(burst_score, 4),
    }

    # ── 2. Build the user message for the LLM ─────────────────────
    user_content = json.dumps({
        "case_id":             case_id,
        "typology":            state.get("typology", "Unknown"),
        "confidence_score":    state.get("confidence_score", 0.5),
        "risk_score":          state.get("risk_score", 0.5),
        "customer_summary":    customer_summary,
        "account_summary":     account_summary,
        "transaction_summary": transaction_summary,
        "alert_count":         summary.get("alert_count", 0),
    }, indent=2)

    # ── 3. Call the LLM ────────────────────────────────────────────
    try:
        system_prompt = _load_prompt()
        llm = get_llm(temperature=0.0)   # 0.0 = deterministic routing decisions

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        logger.debug(f"Agent 2 [{case_id}] — sending to LLM")
        response = llm.invoke(messages)
        raw_output: str = response.content

        logger.debug(f"Agent 2 [{case_id}] — raw LLM output: {raw_output[:300]}")

    except Exception as e:
        msg = f"[{_ts()}] Agent2 [{case_id}]: LLM call failed — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        # Fall back to deterministic plan
        plan = _fallback_plan(state)
        return _write_state(state, plan, errors)

    # ── 4. Parse JSON output ───────────────────────────────────────
    plan, parse_errors = _parse_plan(raw_output, case_id)
    errors.extend(parse_errors)

    # ── 5. Validate and sanitise ───────────────────────────────────
    plan = _validate_plan(plan, state, case_id)

    logger.info(
        f"Agent 2 — {case_id} | Priority: {plan['priority']} | "
        f"Modules: {plan['active_modules']} | "
        f"Enrichment: {plan['requires_enrichment']}"
    )

    return _write_state(state, plan, errors)


# ------------------------------------------------------------------ #
# Conditional edge — read by LangGraph                               #
# ------------------------------------------------------------------ #

def route_after_planner(state: SARAgentState) -> str:
    """
    Called by LangGraph after agent2_planner.

    Returns:
        "enrichment" — case needs external data (Agent 4 runs first).
        "typology"   — no enrichment needed, go straight to Agent 3.

    Wire this in pipeline.py:
        graph.add_conditional_edges(
            "agent2_planner",
            route_after_planner,
            {
                "enrichment": "agent4_enrichment",
                "typology":   "agent3_typology",
            },
        )
    """
    return "enrichment" if state.get("requires_enrichment", False) else "typology"


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

def _parse_plan(raw_output: str, case_id: str) -> tuple[dict, list[str]]:
    """
    Parse the LLM's JSON output into a plan dict.
    Uses safe_parse_json from utils to handle markdown fences and retry.
    Returns (plan_dict, list_of_error_strings).
    """
    errors: list[str] = []

    parsed = safe_parse_json(raw_output)

    if parsed is None:
        msg = (
            f"[{_ts()}] Agent2 [{case_id}]: "
            f"Could not parse LLM output as JSON. "
            f"Raw output (first 200 chars): {raw_output[:200]}"
        )
        logger.error(msg)
        errors.append(msg)
        return {}, errors

    return parsed, errors


def _validate_plan(plan: dict, state: SARAgentState, case_id: str) -> dict:
    """
    Validate the parsed plan and fill in any missing or invalid fields.
    Ensures the pipeline never routes to a non-existent module.
    """
    # active_modules: must all be in VALID_MODULES, always include baseline
    raw_modules = plan.get("active_modules", [])
    if isinstance(raw_modules, list):
        valid = [m for m in raw_modules if m in VALID_MODULES]
        invalid = [m for m in raw_modules if m not in VALID_MODULES]
        if invalid:
            logger.warning(
                f"Agent2 [{case_id}]: LLM returned unknown modules {invalid}, ignoring."
            )
    else:
        valid = []

    # Always include baseline analysis
    if "account_behaviour_analysis" not in valid:
        valid.insert(0, "account_behaviour_analysis")

    # priority
    priority = str(plan.get("priority", "MEDIUM")).upper()
    if priority not in VALID_PRIORITIES:
        logger.warning(
            f"Agent2 [{case_id}]: Invalid priority '{priority}', defaulting to MEDIUM."
        )
        priority = "MEDIUM"

    # requires_enrichment
    requires_enrichment = bool(plan.get("requires_enrichment", False))

    # rationale
    rationale = str(plan.get("rationale", "Routing plan generated by Agent 2."))

    return {
        "active_modules":      valid,
        "priority":            priority,
        "requires_enrichment": requires_enrichment,
        "rationale":           rationale,
    }


def _fallback_plan(state: SARAgentState) -> dict:
    """
    Deterministic fallback plan when the LLM call fails entirely.
    Uses typology and risk score from state to make a reasonable routing decision
    without any LLM involvement — pure Python, always works.
    """
    typology    = state.get("typology", "Unknown")
    risk_score  = state.get("risk_score", 0.5)
    conf_score  = state.get("confidence_score", 0.5)

    structured_case = state.get("structured_case", {})
    customer = structured_case.get("customer", {})
    accounts = structured_case.get("accounts", [{}])
    primary_account = accounts[0] if accounts else {}

    pep_flag           = bool(customer.get("pep_flag", False))
    previous_sar_count = int(customer.get("previous_sar_count", 0) or 0)
    international_txn  = bool(primary_account.get("international_txn", False))

    # ── Module selection ───────────────────────────────────────────
    modules = ["account_behaviour_analysis"]

    typology_module_map = {
        "Structuring":             "structuring_analysis",
        "Rapid Movement of Funds": "velocity_analysis",
        "Funnel Account":          "velocity_analysis",
        "TBML":                    "international_analysis",
        "Shell Company":           "network_analysis",
        "Round Tripping":          "network_analysis",
    }
    if typology in typology_module_map:
        modules.append(typology_module_map[typology])

    if international_txn or typology in ("TBML", "Shell Company", "Round Tripping"):
        if "international_analysis" not in modules:
            modules.append("international_analysis")

    if pep_flag or previous_sar_count > 0 or _risk_rating_to_float(customer.get("risk_rating", "LOW")) > 0.70:
        modules.append("kyc_escalation")

    # ── Priority ───────────────────────────────────────────────────
    if risk_score >= 0.75 or conf_score >= 0.90 or pep_flag or previous_sar_count > 0:
        priority = "HIGH"
    elif risk_score >= 0.50 or conf_score >= 0.70:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # ── Enrichment ─────────────────────────────────────────────────
    needs_enrichment = (
        typology in ("TBML", "Shell Company", "Round Tripping")
        or international_txn
        or pep_flag
    )

    return {
        "active_modules":      list(dict.fromkeys(modules)),  # deduplicate, preserve order
        "priority":            priority,
        "requires_enrichment": needs_enrichment,
        "rationale":           (
            f"Fallback plan (LLM unavailable). "
            f"Typology: {typology}, Risk: {risk_score:.2f}, "
            f"Priority set to {priority}."
        ),
    }


def _write_state(
    state: SARAgentState,
    plan: dict,
    errors: list[str],
) -> SARAgentState:
    """Merge plan and errors into state and return."""
    if not plan:
        plan = _fallback_plan(state)

    return {
        **state,
        "plan":              plan,
        "requires_enrichment": plan.get("requires_enrichment", False),
        "error_log":         errors,
    }


def _risk_rating_to_float(rating: str) -> float:
    """Map string risk_rating to a float for numeric comparisons."""
    return {"LOW": 0.25, "MEDIUM": 0.55, "HIGH": 0.90}.get(str(rating).upper(), 0.25)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()