"""
agents/agent5_narrative.py — SAR Narrative Generator
=====================================================
Stage 5 of the SAR pipeline. Produces the final SAR narrative draft.

Implementation is split into three stages:

  Stage 1 — fetch_case_context(state)
      Pure read from LangGraph state. No transformation.
      Unpacks everything the upstream agents have prepared:
        - Case identifiers and ML scores          (Agent 1)
        - Routing plan and priority               (Agent 2)
        - Triggered rules and quantified stats    (Agent 3)
        - Cognitive event flow                    (Agent 3 LLM)
        - External enrichment                     (Agent 4)
        - Compliance feedback on revision loops   (Agent 6)
      From structured_case: customer KYC + account details only.
      Transactions are NOT pulled here — Agent 3 already distilled
      them into triggered_rules and quantified_indicators.

  Stage 2 — rag_retrieval(case_context)
      Three-layer retrieval:
        a) Typology docs  — what/why for this fraud pattern
        b) Guidelines     — PMLA, RBI KYC, FinCEN analyst obligations
        c) SAR examples   — tone and structure reference cases

  Stage 3 — compile_and_generate(case_context, rag_context)
      Builds a 10-section context tree, constructs the LLM prompt,
      calls the LLM, parses sar_narrative + reasoning_traces from
      the JSON response, and writes both to state.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from state import SARAgentState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CASE CONTEXT FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_case_context(state: SARAgentState) -> dict[str, Any]:
    """
    Pure read from LangGraph state. Zero transformation — just unpacking.

    Pulls everything upstream agents have prepared and organises it into
    a single flat context dict that Stages 2 and 3 can consume without
    touching the state object again.

    Returns
    -------
    dict with keys grouped by source:

      [identifiers]
        case_id, typology, confidence_score, risk_score, sar_worthy

      [plan]
        priority, active_modules, requires_enrichment, plan_rationale

      [customer]          — from structured_case["customer"]
        customer_type, risk_rating, nationality, pep_flag,
        previous_sar_count

      [accounts]          — from structured_case["accounts"]
        accounts           (full list, passed through as-is)
        primary_account    (first account dict, convenience shortcut)

      [agent3_evidence]
        triggered_rules, quantified_indicators

      [cognitive_flow]    — from Agent 3 LLM output
        event_sequence, fund_flow_summary, key_anomalies,
        analyst_hypothesis, risk_summary

      [enrichment]        — from Agent 4 (empty dict if skipped)
        sanctions_hits, adverse_news, regulatory_flags

      [revision_context]  — populated on revision loops from Agent 6
        revision_count, compliance_issues
    """
    case_id = state.get("case_id", "UNKNOWN")
    logger.info(f"Agent 5 — fetching case context for {case_id}")

    # ── structured_case sub-keys ───────────────────────────────────────────────
    structured_case = state.get("structured_case", {})
    customer        = structured_case.get("customer", {})
    accounts        = structured_case.get("accounts", [])

    # ── plan ──────────────────────────────────────────────────────────────────
    plan = state.get("plan", {})

    # ── cognitive event flow ──────────────────────────────────────────────────
    cef = state.get("cognitive_event_flow", {})

    # ── enrichment ────────────────────────────────────────────────────────────
    enrichment = state.get("enrichment_data", {})

    context: dict[str, Any] = {

        # ── identifiers ───────────────────────────────────────────────────────
        "case_id":          case_id,
        "typology":         state.get("typology", "Unknown"),
        "confidence_score": state.get("confidence_score", 0.0),
        "risk_score":       state.get("risk_score", 0.0),
        "sar_worthy":       state.get("sar_worthy", True),

        # ── routing plan ──────────────────────────────────────────────────────
        "priority":             plan.get("priority", "MEDIUM"),
        "active_modules":       plan.get("active_modules", []),
        "requires_enrichment":  plan.get("requires_enrichment", False),
        "plan_rationale":       plan.get("rationale", ""),

        # ── customer KYC ──────────────────────────────────────────────────────
        "customer_type":      customer.get("customer_type", "individual"),
        "risk_rating":        customer.get("risk_rating", "UNKNOWN"),
        "nationality":        customer.get("nationality", ""),
        "pep_flag":           bool(customer.get("pep_flag", False)),
        "previous_sar_count": int(customer.get("previous_sar_count", 0) or 0),

        # ── accounts ──────────────────────────────────────────────────────────
        "accounts":        accounts,
        "primary_account": accounts[0] if accounts else {},

        # ── agent 3 rule evidence ─────────────────────────────────────────────
        "triggered_rules":       state.get("triggered_rules", []),
        "quantified_indicators": state.get("quantified_indicators", {}),

        # ── agent 3 cognitive event flow ──────────────────────────────────────
        "event_sequence":     cef.get("event_sequence", []),
        "fund_flow_summary":  cef.get("fund_flow_summary", ""),
        "key_anomalies":      cef.get("key_anomalies", []),
        "analyst_hypothesis": cef.get("analyst_hypothesis", ""),
        "risk_summary":       cef.get("risk_summary", {}),

        # ── agent 4 enrichment ────────────────────────────────────────────────
        "sanctions_hits":   enrichment.get("sanctions_hits", []),
        "adverse_news":     enrichment.get("adverse_news", []),
        "regulatory_flags": enrichment.get("regulatory_flags", []),

        # ── revision loop context (from Agent 6) ──────────────────────────────
        "revision_count":    state.get("revision_count", 0),
        "compliance_issues": state.get("compliance_issues", []),

        # ── analyst instructions (optional, from /analyse API call) ───────────
        "analyst_prompt":    state.get("analyst_prompt", ""),
    }

    logger.info(
        f"Agent 5 [{case_id}] — context fetched | "
        f"typology: {context['typology']} | "
        f"priority: {context['priority']} | "
        f"rules fired: {len(context['triggered_rules'])} | "
        f"revision: {context['revision_count']}"
    )

    return context


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RAG RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

# Maps Agent 1 typology labels to the exact PDF filenames in rag/docs/typology/
# so Layer A can do a targeted filename-anchored query instead of pure semantic.
_TYPOLOGY_FILE_MAP = {
    "structuring":             ["structuring.pdf", "structuring2.pdf"],
    "rapid movement of funds": ["funnel+rapid.pdf"],
    "funnel account":          ["funnel accs.pdf", "funnel+rapid.pdf"],
    "tbml":                    ["tbml.pdf", "Trade-based-ML-062006.pdf"],
    "shell company":           ["shellcompanies.pdf", "layering.pdf"],
    "round tripping":          ["roundtrip.pdf"],
    "layering":                ["layering.pdf"],
}

# Standard guidelines always retrieved regardless of case specifics
_STANDARD_GUIDELINE_FILES = [
    "PMLA_2005.pdf",
    "PMLA_Rules.pdf",
    "narrative.pdf",
    "fincenguide.pdf",
]

# Case-specific guideline triggers
_KYC_GUIDELINE_FILES      = ["kyc.pdf", "rbi kyc.pdf"]
_PEP_GUIDELINE_FILE       = "pep.pdf"
_INTL_GUIDELINE_FILES     = [
    "Guidance-Correspondent-Banking-Services.pdf",
    "Guidance-RBA-money-value-transfer-services.pdf.coredownload.pdf",
    "A2003-15.pdf",
]


def rag_retrieval(case_context: dict[str, Any]) -> dict[str, Any]:
    """
    Three-layer RAG retrieval scoped to the correct doc folders.

    Layer A — Typology
        Query: typology label + key anomalies from cognitive flow.
        Folder: 'typology'
        Anchored first to the known PDF(s) for this typology,
        then a semantic sweep for any additional relevant chunks.

    Layer B — Guidelines (two sub-queries)
        B1 Standard: PMLA, FinCEN narrative guide — always fetched.
        B2 Case-specific: KYC docs if HIGH risk; PEP doc if pep_flag;
           international guidance if FATF exposure or enrichment present.
        Folder: 'guideline how to'

    Layer C — SAR Examples
        Query built from typology + customer_type + pep_flag + enrichment.
        Folder: 'sar case examples'
        Gives the LLM tone and structural guardrails from real filed SARs.

    Returns
    -------
    dict with keys:
        typology_chunks    : list[str]
        standard_guideline_chunks  : list[str]
        casespec_guideline_chunks  : list[str]
        sar_example_chunks : list[str]
    """
    from rag.retriever import retrieve_from_folder

    typology     = case_context.get("typology", "").lower().strip()
    customer_type = case_context.get("customer_type", "individual")
    pep_flag     = case_context.get("pep_flag", False)
    nationality  = case_context.get("nationality", "IN")
    has_fatf     = case_context.get("quantified_indicators", {}).get("has_fatf_exposure", False)
    has_enrichment = bool(
        case_context.get("sanctions_hits") or
        case_context.get("adverse_news") or
        case_context.get("regulatory_flags")
    )
    key_anomalies    = case_context.get("key_anomalies", [])
    analyst_hypothesis = case_context.get("analyst_hypothesis", "")
    case_id          = case_context.get("case_id", "UNKNOWN")

    logger.info(f"Agent 5 [{case_id}] — starting RAG retrieval")

    # ── Layer A: Typology ─────────────────────────────────────────────────────
    # Build a rich query from the typology label + cognitive flow signals
    typology_query = _build_typology_query(typology, key_anomalies, analyst_hypothesis)
    typology_chunks = retrieve_from_folder("typology", typology_query, top_k=2)

    logger.info(f"Agent 5 [{case_id}] — Layer A: {len(typology_chunks)} typology chunks")
    for i, chunk in enumerate(typology_chunks):
        logger.info(f"  [A-{i+1}] {chunk[:200]}")

    # ── Layer B1: Standard guidelines ─────────────────────────────────────────
    standard_query = (
        f"SAR filing obligations analyst duties {typology} "
        f"suspicious activity reporting legal requirements India"
    )
    standard_chunks = retrieve_from_folder("guideline how to", standard_query, top_k=2)

    logger.info(f"Agent 5 [{case_id}] — Layer B1: {len(standard_chunks)} standard guideline chunks")
    for i, chunk in enumerate(standard_chunks):
        logger.info(f"  [B1-{i+1}] {chunk[:200]}")

    # ── Layer B2: Case-specific guidelines ────────────────────────────────────
    casespec_chunks: list[str] = []

    if case_context.get("risk_rating") in ("HIGH", "MEDIUM") or customer_type == "corporate":
        kyc_query = f"KYC due diligence {customer_type} customer risk {typology}"
        casespec_chunks += retrieve_from_folder("guideline how to", kyc_query, top_k=2)

    if pep_flag:
        pep_query = "politically exposed person PEP enhanced due diligence SAR obligations"
        casespec_chunks += retrieve_from_folder("guideline how to", pep_query, top_k=2)

    if has_fatf or has_enrichment or nationality != "IN":
        intl_query = (
            f"international wire transfer correspondent banking FATF "
            f"cross-border suspicious transaction {typology}"
        )
        casespec_chunks += retrieve_from_folder("guideline how to", intl_query, top_k=2)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in casespec_chunks:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    casespec_chunks = deduped

    logger.info(f"Agent 5 [{case_id}] — Layer B2: {len(casespec_chunks)} case-specific guideline chunks")
    for i, chunk in enumerate(casespec_chunks):
        logger.info(f"  [B2-{i+1}] {chunk[:200]}")

    # ── Layer C: SAR examples ─────────────────────────────────────────────────
    example_query = _build_example_query(
        typology, customer_type, pep_flag, has_enrichment, has_fatf
    )
    sar_example_chunks = retrieve_from_folder("sar case examples", example_query, top_k=2)

    logger.info(f"Agent 5 [{case_id}] — Layer C: {len(sar_example_chunks)} SAR example chunks")
    for i, chunk in enumerate(sar_example_chunks):
        logger.info(f"  [C-{i+1}] {chunk[:200]}")

    return {
        "typology_chunks":          typology_chunks,
        "standard_guideline_chunks": standard_chunks,
        "casespec_guideline_chunks": casespec_chunks,
        "sar_example_chunks":        sar_example_chunks,
    }

def _build_typology_query(
    typology: str,
    key_anomalies: list[str],
    analyst_hypothesis: str,
) -> str:
    """
    Builds a rich semantic query for the typology folder.
    Leads with the typology name (highest signal), then appends
    anomaly signals from the cognitive flow for better chunk recall.
    """
    parts = [f"{typology} money laundering typology indicators red flags"]
    if key_anomalies:
        # Take up to 2 anomalies to keep the query focused
        parts.append(" ".join(key_anomalies[:2]))
    if analyst_hypothesis:
        # First sentence only — enough context without overwhelming the embedding
        first_sentence = analyst_hypothesis.split(".")[0]
        parts.append(first_sentence)
    return " ".join(parts)


def _build_example_query(
    typology: str,
    customer_type: str,
    pep_flag: bool,
    has_enrichment: bool,
    has_fatf: bool,
) -> str:
    """
    Builds the SAR examples query from case-level signals.
    The more signals present, the more targeted the example retrieval.
    """
    parts = [f"SAR narrative {typology} suspicious activity report example"]
    if customer_type == "corporate":
        parts.append("corporate entity business account")
    if pep_flag:
        parts.append("politically exposed person PEP")
    if has_fatf:
        parts.append("international wire FATF high-risk jurisdiction")
    if has_enrichment:
        parts.append("sanctions adverse media regulatory flag")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — COMPILE AND GENERATE
# ═══════════════════════════════════════════════════════════════════════════════

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm

_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "agent5_narrative.txt"


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Agent 5 prompt not found at {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def compile_and_generate(
    case_context: dict[str, Any],
    rag_context:  dict[str, Any],
) -> dict[str, Any]:
    """
    Stage 3 — builds the full context tree, constructs the LLM prompt,
    calls the LLM, parses the response, and returns state updates.

    The context tree passed to the LLM is structured in sections so the
    model can clearly distinguish case facts from retrieved knowledge:

      Section 1 — Case identifiers and scores
      Section 2 — Customer KYC and account summary
      Section 3 — Triggered rules (with rule IDs)
      Section 4 — Quantified indicators
      Section 5 — Cognitive event flow (Agent 3 LLM output)
      Section 6 — Enrichment data (Agent 4, may be empty)
      Section 7 — RAG: typology reference
      Section 8 — RAG: legal guidelines (standard + case-specific)
      Section 9 — RAG: SAR example excerpts
      Section 10 — Revision instructions (only on revision loops)

    Returns state-compatible dict:
        sar_draft        : str
        reasoning_traces : list[dict]
        error_log        : list[str]
    """
    case_id = case_context.get("case_id", "UNKNOWN")
    errors: list[str] = []

    # ── Build context tree ────────────────────────────────────────────────────
    context_tree = _build_context_tree(case_context, rag_context)

    # ── Construct prompt ──────────────────────────────────────────────────────
    try:
        system_prompt = _load_prompt()
    except FileNotFoundError as e:
        errors.append(f"[{_ts()}] Agent5 [{case_id}]: {e}")
        return {"sar_draft": "", "reasoning_traces": [], "error_log": errors}

    user_content = json.dumps(context_tree, indent=2, default=str)

    # ── Call LLM ──────────────────────────────────────────────────────────────
    try:
        llm = get_llm(temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        logger.info(f"Agent 5 [{case_id}] — calling LLM for SAR narrative")
        response = llm.invoke(messages)
        raw: str = response.content.strip()
        logger.info(f"Agent 5 [{case_id}] — raw LLM response:\n{raw}")
    except Exception as e:
        msg = f"[{_ts()}] Agent5 [{case_id}]: LLM call failed — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {"sar_draft": "", "reasoning_traces": [], "error_log": errors}

    # ── Parse response ────────────────────────────────────────────────────────
    sar_draft, reasoning_traces, parse_errors = _parse_llm_response(raw, case_id)
    errors.extend(parse_errors)

    logger.info(
        f"Agent 5 [{case_id}] — narrative generated | "
        f"{len(sar_draft.split())} words | "
        f"{len(reasoning_traces)} audit trace entries"
    )

    return {
        "sar_draft":        sar_draft,
        "sar_context_tree": context_tree,
        "reasoning_traces": reasoning_traces,
        "error_log":        errors,
    }


def _build_context_tree(
    case_context: dict[str, Any],
    rag_context:  dict[str, Any],
) -> dict[str, Any]:
    """
    Assembles the full structured context passed to the LLM as the user message.
    Keeps case facts and retrieved knowledge clearly separated.
    """
    qi = case_context.get("quantified_indicators", {})

    tree: dict[str, Any] = {}

    # ── Section 1: Identifiers ────────────────────────────────────────────────
    tree["case"] = {
        "case_id":          case_context["case_id"],
        "typology":         case_context["typology"],
        "priority":         case_context["priority"],
        "confidence_score": case_context["confidence_score"],
        "risk_score":       case_context["risk_score"],
    }

    # ── Section 2: Subject ────────────────────────────────────────────────────
    tree["subject"] = {
        "customer_type":      case_context["customer_type"],
        "risk_rating":        case_context["risk_rating"],
        "nationality":        case_context["nationality"],
        "pep_flag":           case_context["pep_flag"],
        "previous_sar_count": case_context["previous_sar_count"],
        "primary_account":    case_context["primary_account"],
        "all_accounts":       case_context["accounts"],
    }

    # ── Section 3: Triggered rules ────────────────────────────────────────────
    # Group by severity so the LLM leads with HIGH severity claims
    high_rules   = [r for r in case_context["triggered_rules"] if r.get("severity") == "HIGH"]
    medium_rules = [r for r in case_context["triggered_rules"] if r.get("severity") == "MEDIUM"]
    tree["triggered_rules"] = {
        "HIGH":   high_rules,
        "MEDIUM": medium_rules,
        "total_fired": len(case_context["triggered_rules"]),
    }

    # ── Section 4: Quantified indicators ─────────────────────────────────────
    tree["quantified_indicators"] = qi

    # ── Section 5: Cognitive event flow ──────────────────────────────────────
    tree["cognitive_event_flow"] = {
        "event_sequence":     case_context["event_sequence"],
        "fund_flow_summary":  case_context["fund_flow_summary"],
        "key_anomalies":      case_context["key_anomalies"],
        "analyst_hypothesis": case_context["analyst_hypothesis"],
        "risk_summary":       case_context["risk_summary"],
    }

    # ── Section 6: Enrichment ─────────────────────────────────────────────────
    tree["enrichment"] = {
        "sanctions_hits":   case_context["sanctions_hits"],
        "adverse_news":     case_context["adverse_news"],
        "regulatory_flags": case_context["regulatory_flags"],
        "enrichment_present": bool(
            case_context["sanctions_hits"] or
            case_context["adverse_news"] or
            case_context["regulatory_flags"]
        ),
    }

    # ── Section 7: RAG — typology reference ──────────────────────────────────
    tree["rag_typology_reference"] = rag_context.get("typology_chunks", [])

    # ── Section 8: RAG — legal guidelines ────────────────────────────────────
    tree["rag_guidelines"] = {
        "standard":      rag_context.get("standard_guideline_chunks", []),
        "case_specific": rag_context.get("casespec_guideline_chunks", []),
    }

    # ── Section 9: RAG — SAR examples ────────────────────────────────────────
    tree["rag_sar_examples"] = rag_context.get("sar_example_chunks", [])

    # ── Section 10: Revision instructions (only on loops) ────────────────────
    if case_context["revision_count"] > 0:
        tree["revision"] = {
            "revision_number":  case_context["revision_count"],
            "compliance_issues": case_context["compliance_issues"],
            "instruction": (
                "This is a revision. Address ONLY the compliance_issues listed above. "
                "Do not rewrite sections that were not flagged."
            ),
        }

    # ── Section 11: Analyst instructions (optional, from API call) ───────────
    analyst_prompt = case_context.get("analyst_prompt", "")
    if analyst_prompt:
        tree["analyst_instructions"] = {
            "prompt": analyst_prompt,
            "instruction": (
                "The analyst has provided specific guidance for this SAR. "
                "Incorporate these instructions into the narrative where relevant. "
                "Do not contradict the factual evidence — analyst guidance supplements, not overrides."
            ),
        }

    return tree


def _parse_llm_response(
    raw: str,
    case_id: str,
) -> tuple[str, list[dict], list[str]]:
    """
    Parse the LLM JSON response into (sar_draft, reasoning_traces, errors).

    New schema: LLM returns { sar_sections: {...}, reasoning_traces: [...] }
    sar_draft stored as JSON string of sar_sections for downstream consumers.
    Falls back gracefully on parse failure.
    """
    errors: list[str] = []

    # Strip markdown fences and control characters before attempting JSON parse
    text = raw
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()

    # Strip ALL control characters regardless of fences — must happen before json.loads
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        # One more attempt — try extracting the first { ... } block
        import re as _re
        match = _re.search(r'\{.*\}', text, _re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

        if parsed is None:
            msg = f"[{_ts()}] Agent5 [{case_id}]: LLM returned non-JSON — {e}. Storing raw output."
            logger.error(msg)
            errors.append(msg)
            return raw, [], errors

    # Support both new schema (sar_sections) and old flat schema (sar_narrative)
    if "sar_sections" in parsed:
        sar_sections = parsed["sar_sections"]
        sar_draft = json.dumps(sar_sections, indent=2, ensure_ascii=False)
    elif "sar_narrative" in parsed:
        # prompt used sar_narrative key — treat identically
        sar_sections = parsed["sar_narrative"]
        sar_draft = json.dumps(sar_sections, indent=2, ensure_ascii=False)
    else:
        msg = f"[{_ts()}] Agent5 [{case_id}]: LLM response missing 'sar_sections' key."
        logger.warning(msg)
        errors.append(msg)
        sar_draft = text

    raw_traces = parsed.get("reasoning_traces", [])
    if not isinstance(raw_traces, list):
        raw_traces = []

    # Validate and normalise each trace entry
    # Accepts both the new schema (narrative_claim/rule_id/source_data_field/severity)
    # and the prompt's schema (claim/source_type/source_value/confidence)
    reasoning_traces: list[dict] = []
    for i, trace in enumerate(raw_traces):
        if not isinstance(trace, dict):
            continue

        trace_id = str(trace.get("trace_id", f"RT-{i+1:03d}"))
        if not trace_id.startswith("RT-"):
            trace_id = f"RT-{i+1:03d}"

        # claim field — accept either key name
        claim = str(trace.get("narrative_claim", trace.get("claim", ""))).strip()
        if not claim:
            continue

        # rule_id — accept rule_id, source_ref, or source_type
        rule_id = str(trace.get("rule_id", trace.get("source_ref", trace.get("source_type", "UNKNOWN")))).strip()

        # regulatory_ref
        regulatory_ref = str(trace.get("regulatory_ref", "N/A")).strip()

        # source data — accept source_data_field, evidence, or source_value
        source_data = str(
            trace.get("source_data_field", trace.get("evidence", trace.get("source_value", "")))
        ).strip()

        # severity — accept severity or map confidence string
        raw_severity = trace.get("severity", trace.get("confidence", "MEDIUM"))
        severity = str(raw_severity).upper()
        if severity not in ("HIGH", "MEDIUM", "LOW", "INFO"):
            severity = "MEDIUM"

        reasoning_traces.append({
            "trace_id":          trace_id,
            "narrative_claim":   claim,
            "rule_id":           rule_id,
            "regulatory_ref":    regulatory_ref,
            "source_data_field": source_data,
            "severity":          severity,
            "confidence":        1.0,
        })

    logger.info(
        f"Agent 5 [{case_id}] — parsed {len(reasoning_traces)} valid trace entries "
        f"(dropped {len(raw_traces) - len(reasoning_traces)} malformed)"
    )

    return sar_draft, reasoning_traces, errors


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def agent5_narrative(state: SARAgentState) -> dict[str, Any]:
    """
    LangGraph node for Agent 5.

    Reads from state:
        case_id, typology, confidence_score, risk_score,
        structured_case (customer + accounts),
        plan, triggered_rules, quantified_indicators,
        cognitive_event_flow, enrichment_data,
        revision_count, compliance_issues

    Writes to state:
        sar_draft, reasoning_traces, error_log
    """
    errors: list[str] = []

    # Stage 1 — always runs
    case_context = fetch_case_context(state)

    # Stage 2 — RAG retrieval
    try:
        rag_context = rag_retrieval(case_context)
    except Exception as e:
        msg = f"[{_ts()}] Agent5 [{case_context['case_id']}]: RAG retrieval failed — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        rag_context = {
            "typology_chunks":           [],
            "standard_guideline_chunks": [],
            "casespec_guideline_chunks": [],
            "sar_example_chunks":        [],
        }

    # Stage 3 — compile context tree, call LLM, parse response
    try:
        result = compile_and_generate(case_context, rag_context)
        result["error_log"] = errors + result.get("error_log", [])
        return result
    except Exception as e:
        msg = f"[{_ts()}] Agent5 [{case_context['case_id']}]: compile_and_generate failed — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {
            "sar_draft":        "",
            "sar_context_tree": {},
            "reasoning_traces": [],
            "error_log":        errors,
        }


# ── helper ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()
