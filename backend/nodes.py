"""
NEXUS 2.0 — LangGraph Node Functions
Each function takes SARState, performs its task, appends to audit_log, and returns
the updated state keys.

Pipeline:  mask_pii → router → typology → narrative → compliance_judge → unmask
"""

import json
from graph_state import SARState
from privacy_guard import PrivacyGuard
from bedrock_llm import invoke_claude


# ──────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _append_audit(state: SARState, step: str, action: str, data: str) -> list:
    """Return a new audit_log list with the entry appended (immutable update)."""
    return state["audit_log"] + [{"step": step, "action": action, "data": data}]


def _safe_json(obj, max_len: int = 2000) -> str:
    """Serialize obj to JSON string, truncated for audit readability."""
    raw = json.dumps(obj, default=str)
    if len(raw) > max_len:
        return raw[:max_len] + "... [truncated]"
    return raw


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 1 — PII MASKING (no LLM call)
# ──────────────────────────────────────────────────────────────────────────────

def mask_pii_node(state: SARState) -> dict:
    """
    Pre-processing step: mask PII in the raw alert before it reaches any LLM.
    """
    guard = PrivacyGuard()
    masked_data, pii_mapping = guard.mask_alert(state["raw_alert_data"])

    audit = _append_audit(
        state,
        step="mask_pii",
        action="PII masked using PrivacyGuard",
        data=f"Masked {len(pii_mapping)} PII entities: {list(pii_mapping.keys())}",
    )

    equilibrium_audit = state.get("equilibrium_audit_log", []) + [{
        "step": "mask_pii",
        "action": "PII masked using PrivacyGuard",
        "evidence": f"Masked {len(pii_mapping)} PII entities: {list(pii_mapping.keys())}"
    }]

    return {
        "masked_data": masked_data,
        "pii_mapping": pii_mapping,
        "audit_log": audit,
        "equilibrium_audit_log": equilibrium_audit,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 2 — ROUTER (LLM classifies the alert type)
# ──────────────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are an expert AML/BSA compliance analyst at a major financial institution.
Your task is to classify the given alert data into EXACTLY ONE of the following typologies:

- Elder Financial Exploitation
- Structuring / Smurfing
- Money Mule Activity
- Romance Scam / Social Engineering
- Sanctions Evasion
- Cryptocurrency Layering / Mixing
- Trade-Based Money Laundering
- Terrorist Financing
- Fraud (General)
- KYC / Due Diligence Failure
- Other Suspicious Activity

Respond with ONLY the typology name, nothing else. No explanation, no punctuation beyond what's in the name."""


def router_node(state: SARState) -> dict:
    """
    Classify the alert into a standard AML typology.
    """
    user_msg = (
        "Classify the following alert data into one typology.\n\n"
        f"Alert reason: {state['masked_data'].get('alert_reason', 'N/A')}\n"
        f"Priority: {state['masked_data'].get('priority', 'N/A')}\n"
        f"Risk score: {state['masked_data'].get('initial_risk_score', 'N/A')}\n"
        f"Total suspicious amount: ${state['masked_data'].get('total_suspicious_amount', 'N/A')}\n\n"
        f"Transaction summary:\n{_safe_json(state['masked_data'].get('transactions', []))}\n\n"
        f"Analyst notes:\n{_safe_json(state['masked_data'].get('analyst_notes', []))}"
    )

    typology = invoke_claude(ROUTER_SYSTEM, user_msg, temperature=0.1)
    typology = typology.strip()

    audit = _append_audit(
        state,
        step="router",
        action=f"Alert classified as: {typology}",
        data=f"System prompt: {ROUTER_SYSTEM[:200]}... | Input keys: alert_reason, transactions, analyst_notes",
    )

    equilibrium_audit = state.get("equilibrium_audit_log", []) + [{
        "step": "router",
        "action": f"Alert classified as: {typology}",
        "evidence": f"Alert reason: {state['masked_data'].get('alert_reason', 'N/A')}, Risk score: {state['masked_data'].get('initial_risk_score', 'N/A')}"
    }]

    return {
        "detected_typology": typology,
        "audit_log": audit,
        "equilibrium_audit_log": equilibrium_audit,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 3 — TYPOLOGY ANALYSIS (LLM deep-dives on the pattern)
# ──────────────────────────────────────────────────────────────────────────────

TYPOLOGY_SYSTEM = """You are a senior AML investigator. You have been given alert data that has been classified as a specific typology.

Analyze the transaction patterns, subject behavior, risk flags, and any communications. Produce a structured analysis with these sections:

1. **Pattern Summary**: 2-3 sentence overview of the suspicious pattern.
2. **Key Indicators**: Bullet list of specific red flags with supporting data points.
3. **Transaction Flow**: Describe the flow of funds step by step.
4. **Risk Assessment**: Severity (Critical/High/Medium/Low) and confidence level.
5. **Regulatory Triggers**: Which BSA/AML regulations or FinCEN advisories are relevant.

Be precise. Reference specific transaction IDs, amounts, and dates from the data."""


def typology_node(state: SARState) -> dict:
    """
    Perform deep-dive analysis on the detected typology.
    """
    user_msg = (
        f"Detected typology: {state['detected_typology']}\n\n"
        f"Full masked alert data:\n{_safe_json(state['masked_data'], max_len=6000)}"
    )

    analysis = invoke_claude(TYPOLOGY_SYSTEM, user_msg, temperature=0.3)

    audit = _append_audit(
        state,
        step="typology_analysis",
        action=f"Deep analysis completed for typology: {state['detected_typology']}",
        data=f"Analysis length: {len(analysis)} chars | System prompt: typology analysis template",
    )

    equilibrium_audit = state.get("equilibrium_audit_log", []) + [{
        "step": "typology_analysis",
        "action": f"Deep analysis completed for typology: {state['detected_typology']}",
        "evidence": f"Analysis length: {len(analysis)} chars | Typology: {state['detected_typology']}"
    }]

    return {
        "typology_analysis": analysis,
        "audit_log": audit,
        "equilibrium_audit_log": equilibrium_audit,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 4 — SAR NARRATIVE DRAFTING (LLM writes the FinCEN SAR)
# ──────────────────────────────────────────────────────────────────────────────

NARRATIVE_SYSTEM = """You are an expert SAR (Suspicious Activity Report) narrative writer for FinCEN filings.

Write a complete SAR narrative following FinCEN guidelines. The narrative MUST include:

1. **Subject Information**: Who is involved (use the placeholder identifiers provided — do NOT invent real names).
2. **Suspicious Activity Description**: What happened, when, where, and how.
3. **Transaction Details**: Specific amounts, dates, methods, and flow of funds.
4. **Why It Is Suspicious**: Connect the activity to the identified typology and explain why it deviates from expected behavior.
5. **Supporting Information**: Any communications, teller observations, or other corroborating data.

FORMAT REQUIREMENTS:
- Write in clear, professional prose (not bullet points).
- Use past tense.
- Reference specific transaction IDs and amounts.
- Keep the narrative between 400-800 words.
- Do NOT include any real PII — only use the masked placeholders provided in the data.

IMPORTANT: Use ONLY the placeholder identifiers (e.g., [PERSON_1], [ACCOUNT_1]) from the masked data. Never invent or guess real names or account numbers."""


def narrative_node(state: SARState) -> dict:
    """
    Draft the FinCEN-compliant SAR narrative using masked data.
    """
    user_msg = (
        f"Typology: {state['detected_typology']}\n\n"
        f"Typology Analysis:\n{state['typology_analysis']}\n\n"
        f"Masked Alert Data:\n{_safe_json(state['masked_data'], max_len=6000)}"
    )

    draft = invoke_claude(NARRATIVE_SYSTEM, user_msg, temperature=0.4)

    audit = _append_audit(
        state,
        step="narrative_drafting",
        action="FinCEN SAR narrative drafted (masked)",
        data=f"Draft length: {len(draft)} chars | Used masked placeholders only",
    )

    equilibrium_audit = state.get("equilibrium_audit_log", []) + [{
        "step": "narrative_drafting",
        "action": "FinCEN SAR narrative drafted (masked)",
        "evidence": f"Draft length: {len(draft)} chars | Typology: {state['detected_typology']} | Used masked placeholders only"
    }]

    return {
        "draft_sar_masked": draft,
        "audit_log": audit,
        "equilibrium_audit_log": equilibrium_audit,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 5 — COMPLIANCE JUDGE (LLM grades the draft)
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a senior BSA/AML Compliance Officer reviewing a SAR narrative draft for FinCEN filing quality.

Score the narrative on a scale of 1-10 based on these criteria:
- **Completeness** (2 pts): Are all required elements present (who, what, when, where, why, how)?
- **Accuracy** (2 pts): Do the facts match the underlying alert data?
- **Regulatory Compliance** (2 pts): Does it follow FinCEN narrative guidelines?
- **Clarity** (2 pts): Is it clearly written and well-organized?
- **Actionability** (2 pts): Would law enforcement find this useful?

Respond in this EXACT format (no deviation):
SCORE: <number>
FEEDBACK: <2-4 sentences of constructive feedback>

Example:
SCORE: 8
FEEDBACK: The narrative covers all required elements and provides clear transaction flow. Consider adding more detail about the timeline of account changes. The regulatory triggers section is strong."""


def compliance_judge_node(state: SARState) -> dict:
    """
    Grade the SAR draft and provide automated feedback.
    """
    user_msg = (
        f"Typology: {state['detected_typology']}\n\n"
        f"SAR Narrative Draft:\n{state['draft_sar_masked']}\n\n"
        f"Original Alert Data (masked):\n{_safe_json(state['masked_data'], max_len=4000)}"
    )

    response = invoke_claude(JUDGE_SYSTEM, user_msg, temperature=0.2)

    # Parse the structured response
    score = 0
    feedback = response
    for line in response.strip().splitlines():
        if line.strip().upper().startswith("SCORE:"):
            try:
                score = int(line.split(":")[1].strip())
                score = max(1, min(10, score))  # clamp 1-10
            except (ValueError, IndexError):
                score = 5  # default if parsing fails

    audit = _append_audit(
        state,
        step="compliance_judge",
        action=f"SAR graded: {score}/10",
        data=f"Full judge response: {response[:500]}",
    )

    equilibrium_audit = state.get("equilibrium_audit_log", []) + [{
        "step": "compliance_judge",
        "action": f"SAR graded: {score}/10",
        "evidence": f"Full judge response: {response[:500]}"
    }]

    return {
        "compliance_score": score,
        "audit_log": audit,
        "equilibrium_audit_log": equilibrium_audit,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 6 — PII UNMASKING (no LLM call)
# ──────────────────────────────────────────────────────────────────────────────

def unmask_node(state: SARState) -> dict:
    """
    Post-processing step: replace placeholders with real PII in the final SAR.
    """
    guard = PrivacyGuard()
    final_clean = guard.unmask_text(state["draft_sar_masked"], state["pii_mapping"])

    audit = _append_audit(
        state,
        step="unmask_pii",
        action="PII restored in final SAR narrative",
        data=f"Replaced {len(state['pii_mapping'])} placeholders with original values",
    )

    equilibrium_audit = state.get("equilibrium_audit_log", []) + [{
        "step": "unmask_pii",
        "action": "PII restored in final SAR narrative",
        "evidence": f"Replaced {len(state['pii_mapping'])} placeholders with original values"
    }]

    return {
        "final_sar_clean": final_clean,
        "audit_log": audit,
        "equilibrium_audit_log": equilibrium_audit,
    }
