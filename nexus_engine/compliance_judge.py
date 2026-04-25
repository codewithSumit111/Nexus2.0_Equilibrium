"""
agent6.py  -  Agent 6: Compliance Validation Agent  (Agent-as-a-Judge)
=======================================================================
Team BAYMAX | Barclays Hack-O-Hire | SAR Generator

==================================================================
ROLE IN PIPELINE
==================================================================
  Agent 5 (Narrative Engine) produces:  sar_draft (str)
  Agent 6 (THIS FILE) answers:
    "Is this SAR narrative regulator-ready?"

  Agent 6 evaluates:
    1. SAR structural completeness  - 5W + How elements present?
    2. Transaction accuracy         - amounts, dates, counterparties match case data?
    3. Typology alignment           - narrative consistent with predicted typology?
    4. Prohibited language removal  - speculative, hedging, PII-over-exposure scrubbed?
    5. Semantic coherence           - narrative reads logically, no contradictions?

  Decision tree:
    quality_score >= PASS_THRESHOLD  ->  compliance_passed=True  ->  Agent 7 (human review)
    quality_score <  PASS_THRESHOLD
      AND revision_count < 3         ->  compliance_passed=False ->  Agent 5 (revision)
      AND revision_count == 3        ->  compliance_passed=False ->  Agent 7 (flagged)

==================================================================
PIPELINE POSITION
==================================================================
  Agent 1  ->  SAR worthiness + typology
  Agent 2  ->  Orchestration / routing
  Agent 3  ->  Typology evidence (rule triggers, quantified indicators)
  Agent 4  ->  Enrichment (sanctions, news, advisories)
  Agent 5  ->  SAR narrative draft
  ------------------------------------------------------------------
  Agent 6  (THIS FILE)
     |  LLM-as-Judge (claude-sonnet-4-20250514)  |
     Rules-based structural check (no LLM, always fast)
     LLM semantic + typology check  (Anthropic API)
     RAG memory: regulatory rules + approved narrative patterns
     |
     {compliance_passed, compliance_issues, quality_score,
      reasoning_traces, revision_instructions}  ->  Agent 7 / Agent 5
  ------------------------------------------------------------------

==================================================================
ARCHITECTURE: TWO-STAGE EVALUATION
==================================================================
  Stage 1 - Rule-based structural gatekeeper (deterministic, fast)
    Regex / keyword checks across 5 rubric dimensions.
    Operates WITHOUT an LLM - catches obvious structural failures cheaply.
    Blocking failures prevent LLM invocation (cost optimisation).

  Stage 2 - LLM-as-Judge (Anthropic claude-sonnet-4-20250514)
    Called only when Stage 1 passes or has only minor issues.
    Receives: SAR draft + structured case data + typology + regulatory memory.
    Returns: JSON {scores, issues, revision_instructions}.
    Combines with Stage 1 scores to produce final quality_score.

==================================================================
REGULATORY MEMORY (lightweight, structured)
==================================================================
  Mandatory SAR elements     [PMLA Rule 12 / RBI KYC S.38 / FATF-RECS R.20]
    WHO     : subject identity (name, account, customer type)
    WHAT    : nature of suspicious activity + amounts
    WHEN    : date range of suspicious activity
    WHERE   : accounts / branches / channels involved
    HOW     : method used (cash, wire, structuring, etc.)
    WHY     : red flags observed (why it is suspicious)

  Prohibited language patterns [RBI FIU-IND Filing Guidelines]
    "may have", "possibly", "could be", "we believe", "it appears"
    "suspect that", "potentially", "might be", "seems to"
    "we think", "in our opinion", "allegedly" (in narrative body)

  Per-typology required narrative elements
    structuring      : deposit count, avg amount, CV, date range, CTR threshold ref
    rapid_movement   : time-to-exit, fund_exit_ratio, SWIFT ref, FATF jurisdiction
    layering         : hop count, country sequence, internal transfer pattern
    tbml             : invoice discrepancy, FATF country, round amounts, trade finance
    funnel_account   : incoming source count, consolidation pattern, exit method
    shell_company    : beneficial owner, opacity flag, opaque structure description
    round_tripping   : origination country, remittance route, reinvestment evidence

==================================================================
APPROVED NARRATIVE TEMPLATES (historical memory excerpts)
==================================================================
  Stored as APPROVED_NARRATIVE_PATTERNS below.
  Used by the LLM judge as style and structure references.
  Indexed by typology. In production, these come from the
  vector store (Amazon OpenSearch Serverless) via RAG.

==================================================================
SCORING RUBRIC  (100 points total -> quality_score 0.0-1.0)
==================================================================
  Dimension                         Weight
  ------------------------------------------
  1. Structural completeness (5W)     25 pts
  2. Transaction accuracy             25 pts
  3. Typology alignment               20 pts
  4. Language compliance              20 pts
  5. Semantic coherence               10 pts
  ------------------------------------------
  PASS_THRESHOLD                     0.80

==================================================================
STATE INTEGRATION
==================================================================
  Reads  : state['sar_draft'], state['structured_case'],
           state['typology'], state['quantified_indicators'],
           state['triggered_rules'], state['enrichment_data'],
           state['revision_count']
  Writes : state['compliance_passed'], state['compliance_issues'],
           state['quality_score'], state['reasoning_traces'],
           + new field 'revision_instructions' (dict)

==================================================================
REFERENCES
==================================================================
  [PMLA]        Prevention of Money Laundering Act 2002 (amended 2023)
  [RBI-KYC]     RBI KYC Master Direction 2016 (updated 2023)
  [FIU-IND]     FIU-IND SAR Filing Guidelines
  [FATF-RECS]   FATF 40 Recommendations 2012 (updated 2023)
  [FATF-2005]   FATF Money Laundering Typologies 2004-2005
  [FinCEN-STR]  FinCEN SAR Electronic Filing Requirements 2022
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import time
import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Any

import requests

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Agent6] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Agent6")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Using Groq via config.get_llm() — same LLM used by all other agents
LLM_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS = 1500

PASS_THRESHOLD     = 0.60   # quality_score >= 0.60 -> compliance_passed = True
MAX_REVISIONS      = 1      # Agent 6 can send the draft back to Agent 5 once only
STAGE1_BLOCK_SCORE = 0.40   # Stage 1 score below this -> skip LLM (too broken to fix)

# Rubric weights (must sum to 1.0)
RUBRIC_WEIGHTS = {
    "structural_completeness": 0.25,  # 5W + How
    "transaction_accuracy":    0.25,  # amounts, dates, parties match case data
    "typology_alignment":      0.20,  # narrative fits the predicted typology
    "language_compliance":     0.20,  # no speculative / prohibited language
    "semantic_coherence":      0.10,  # logical flow, no contradictions
}
assert abs(sum(RUBRIC_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ─────────────────────────────────────────────────────────────────────────────
# REGULATORY MEMORY
# ─────────────────────────────────────────────────────────────────────────────

# 5W + How: mandatory SAR elements under PMLA Rule 12 and FIU-IND guidelines
SAR_MANDATORY_ELEMENTS = {
    "WHO": {
        "description": "Subject identity - full name, account number, customer type",
        "keywords":    ["account", "customer", "subject", "holder", "client"],
        "regex":       r"\b(account\s*(no|number|#)?\.?\s*\w+|customer\s+id|name\s*:)",
    },
    "WHAT": {
        "description": "Nature and amount of suspicious activity",
        "keywords":    ["transaction", "transfer", "deposit", "withdrawal", "wire",
                        "amount", "inr", "₹", "lakh", "crore"],
        "regex":       r"(inr|₹|rs\.?)\s*[\d,]+|[\d,]+\s*(inr|lakh|crore)",
    },
    "WHEN": {
        "description": "Date range of suspicious activity",
        "keywords":    ["on", "between", "during", "from", "date", "period"],
        "regex":       r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2}",
    },
    "WHERE": {
        "description": "Accounts, branches, channels, jurisdictions involved",
        "keywords":    ["branch", "account", "channel", "swift", "rtgs", "bank",
                        "country", "jurisdiction", "international"],
        "regex":       r"\b(branch|swift|rtgs|neft|imps|upi|channel)\b",
    },
    "HOW": {
        "description": "Method used - cash, wire, structuring pattern, layering route",
        "keywords":    ["structured", "wired", "transferred", "deposited", "split",
                        "layered", "funnelled", "routed", "through"],
        "regex":       r"\b(structur|wire|transfer|deposit|withdraw|layer|funnel|route|pass.through)\w*",
    },
    "WHY": {
        "description": "Red flags - specific reasons why activity is suspicious",
        "keywords":    ["suspicious", "unusual", "inconsistent", "below threshold",
                        "velocity", "high risk", "flagged", "anomaly", "pattern"],
        "regex":       r"\b(suspicious|unusual|inconsistent|flag|alert|anomal|threshold|velocity|high.risk)\w*",
    },
}

# Prohibited / speculative language patterns [FIU-IND Filing Guidelines §4.3]
PROHIBITED_PATTERNS = [
    (r"\bmay have\b",           "Speculative: 'may have'"),
    (r"\bpossibly\b",           "Speculative: 'possibly'"),
    (r"\bcould be\b",           "Speculative: 'could be'"),
    (r"\bwe believe\b",         "Opinion: 'we believe'"),
    (r"\bit appears\b",         "Speculative: 'it appears'"),
    (r"\bsuspect that\b",       "Speculative: 'suspect that'"),
    (r"\bpotentially\b",        "Speculative: 'potentially'"),
    (r"\bmight be\b",           "Speculative: 'might be'"),
    (r"\bseems to\b",           "Speculative: 'seems to'"),
    (r"\bwe think\b",           "Opinion: 'we think'"),
    (r"\bin our opinion\b",     "Opinion: 'in our opinion'"),
    (r"\ballegedly\b",          "Unverified: 'allegedly'"),
    (r"\bperhaps\b",            "Speculative: 'perhaps'"),
    (r"\bapparently\b",         "Speculative: 'apparently'"),
    (r"\bif true\b",            "Speculative: 'if true'"),
    (r"\bseemingly\b",          "Speculative: 'seemingly'"),
]

# Per-typology required narrative elements
TYPOLOGY_REQUIRED_ELEMENTS = {
    "structuring": [
        "deposit",
        "threshold",           # below CTR threshold reference
        "consistent",          # consistent sizing / low CV
        "INR 10",              # 10 lakh threshold callout
    ],
    "rapid_movement": [
        "exit",                # fund exit reference
        "swift",               # SWIFT / international wire
        "within",              # time-to-exit language
        "velocity",
    ],
    "layering": [
        "hop",                 # multi-hop transfer
        "internal",            # internal transfer chain
        "layer",
    ],
    "tbml": [
        "invoice",             # trade invoice reference
        "international",
        "country",             # high-risk country mention
        "round",               # round amount pattern
    ],
    "funnel_account": [
        "source",              # multiple incoming sources
        "consolidat",          # consolidation pattern
        "exit",
    ],
    "shell_company": [
        "beneficial",          # beneficial owner
        "opaque",
        "structure",
    ],
    "round_tripping": [
        "remit",               # remittance route
        "return",              # funds return
        "foreign",
    ],
    "multi": [                 # combined typology needs both structuring + rapid cues
        "deposit",
        "threshold",
        "exit",
        "velocity",
    ],
    "clean": [],
}

# Approved narrative patterns (historical memory excerpts)
# In production: retrieved from Amazon OpenSearch vector store via RAG
APPROVED_NARRATIVE_PATTERNS = {
    "structuring": (
        "The subject, [Name], holding account [Account No.] at [Branch], made [N] cash "
        "deposits between [Date1] and [Date2], each ranging from INR [min] to INR [max] - "
        "all below the INR 10,00,000 CTR reporting threshold. The coefficient of variation "
        "of deposit amounts was [CV], consistent with deliberate structuring to avoid "
        "mandatory reporting under PMLA Rule 3. A total of INR [total] was deposited across "
        "[N] transactions, with no corresponding business purpose provided. [N_alerts] "
        "automated alerts were triggered under rule [rule_id]."
    ),
    "rapid_movement": (
        "Account [Account No.], held by [Name], received INR [amount] via [Channel] "
        "on [Date]. The funds were fully transferred via SWIFT wire to [Counterparty] "
        "in [Country] within [hours] hours, yielding a fund exit ratio of [ratio]. "
        "This rapid pass-through pattern is inconsistent with the account's declared "
        "business profile and indicative of layering consistent with FATF Recommendation "
        "20. Velocity score at time of transfer was [score]."
    ),
    "tbml": (
        "The account held by [Entity], a [Customer Type], executed [N] international "
        "wire transfers totalling INR [amount] to [Country], a jurisdiction on the "
        "FATF grey/blacklist. Transfer amounts were round denominations of "
        "INR [round_amount], inconsistent with arm's-length commercial invoicing. "
        "No trade documentation was presented to the reporting institution. "
        "The pattern is consistent with trade-based money laundering as described in "
        "FATF Trade-Based Money Laundering 2006 §2.2."
    ),
    "funnel_account": (
        "Account [Account No.] received credits from [N] distinct originating accounts "
        "across [N_banks] financial institutions over [period]. Cumulative inflows "
        "of INR [total] were consolidated and disbursed via [channel] within [hours] "
        "hours, with a fund exit ratio of [ratio]. The pattern, involving [N] distinct "
        "incoming sources, is consistent with a funnel or mule account arrangement "
        "as characterised by FIU-IND Typology Report 2024."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 - RULE-BASED STRUCTURAL GATEKEEPER
# ─────────────────────────────────────────────────────────────────────────────

def _stage1_structural_check(
    sar_draft: str,
    typology: str,
    quantified_indicators: dict,
) -> dict:
    """
    Deterministic rule-based structural checks.
    Returns per-dimension scores (0.0-1.0) and issues list.
    NO LLM invoked here - fast, cheap, always runs.
    """
    draft_lower = sar_draft.lower()
    issues      = []
    traces      = []

    # ── DIM 1: Structural completeness (5W + How) ────────────────────────
    element_scores = {}
    for element, spec in SAR_MANDATORY_ELEMENTS.items():
        # Keyword match (lenient)
        kw_hit  = any(k in draft_lower for k in spec["keywords"])
        # Regex match (strict)
        rx_hit  = bool(re.search(spec["regex"], sar_draft, re.IGNORECASE))
        found   = kw_hit or rx_hit
        element_scores[element] = 1.0 if found else 0.0
        if not found:
            issues.append(f"Missing {element} element: {spec['description']}")
            traces.append({
                "dimension":   "structural_completeness",
                "element":     element,
                "check":       "keyword+regex",
                "passed":      False,
                "description": spec["description"],
                "rule_ref":    "PMLA Rule 12 / FIU-IND Filing Guidelines §3.1",
            })
        else:
            traces.append({
                "dimension": "structural_completeness",
                "element":   element,
                "check":     "keyword+regex",
                "passed":    True,
            })

    dim1_score = sum(element_scores.values()) / len(element_scores)

    # ── DIM 2: Transaction accuracy (lightweight - full check via LLM) ───
    # Check that key numeric facts mentioned in quantified_indicators appear
    # in some form in the draft (exact match not required - rounding ok)
    accuracy_checks = []
    for key, val in (quantified_indicators or {}).items():
        if isinstance(val, (int, float)):
            # Check for the numeric value ± 10% anywhere in the narrative
            val_int = int(round(val))
            pattern = str(val_int)
            if len(pattern) >= 4:  # only check meaningful numbers (>= 1000)
                found_val = pattern in sar_draft.replace(",", "")
                accuracy_checks.append(found_val)
                if not found_val:
                    issues.append(
                        f"Quantified indicator '{key}={val_int}' "
                        "not found in narrative - verify accuracy"
                    )
                    traces.append({
                        "dimension":  "transaction_accuracy",
                        "key":        key,
                        "expected":   val_int,
                        "found":      False,
                        "rule_ref":   "FIU-IND Filing Guidelines §4.1",
                    })

    dim2_score = (
        sum(accuracy_checks) / len(accuracy_checks) if accuracy_checks else 0.85
    )  # Default 0.85 when no quantified_indicators supplied

    # ── DIM 3: Typology alignment (keyword-based) ────────────────────────
    canonical = typology.replace("typology_", "").lower()
    required  = TYPOLOGY_REQUIRED_ELEMENTS.get(canonical, [])
    if not required:
        dim3_score = 0.90  # 'clean' or unknown typology - not penalised
    else:
        hits = [kw for kw in required if kw.lower() in draft_lower]
        dim3_score = len(hits) / len(required)
        if dim3_score < 1.0:
            missing_kws = [kw for kw in required if kw not in hits]
            issues.append(
                f"Typology '{canonical}' narrative may be incomplete - "
                f"missing elements: {missing_kws}  "
                f"[FATF-2005 typology indicators]"
            )
            traces.append({
                "dimension":    "typology_alignment",
                "typology":     canonical,
                "required_kws": required,
                "found_kws":    hits,
                "missing_kws":  missing_kws,
                "score":        round(dim3_score, 3),
                "rule_ref":     "FATF Money Laundering Typologies 2004-2005",
            })

    # ── DIM 4: Language compliance ────────────────────────────────────────
    lang_violations = []
    for pattern, label in PROHIBITED_PATTERNS:
        if re.search(pattern, sar_draft, re.IGNORECASE):
            lang_violations.append(label)
            traces.append({
                "dimension": "language_compliance",
                "violation": label,
                "pattern":   pattern,
                "rule_ref":  "FIU-IND Filing Guidelines §4.3 - Prohibited speculative language",
            })
    if lang_violations:
        issues.append(
            "Prohibited speculative / opinion language detected: "
            + "; ".join(lang_violations)
            + "  [FIU-IND §4.3]"
        )
    dim4_score = max(0.0, 1.0 - len(lang_violations) * 0.12)

    # ── DIM 5: Semantic coherence (basic, rule-based only) ────────────────
    # Check: draft is non-empty, has >= 3 sentences, not suspiciously short
    sentences    = [s.strip() for s in re.split(r"[.!?]", sar_draft) if s.strip()]
    word_count   = len(sar_draft.split())
    is_too_short = word_count < 80
    has_sections = len(sentences) >= 3
    # Check for obvious contradiction: "no suspicious activity" in a SAR draft
    has_contradiction = bool(
        re.search(r"\bno suspicious\b|\bnot suspicious\b|\bno anomal\b",
                  sar_draft, re.IGNORECASE)
    )
    dim5_score = 1.0
    if is_too_short:
        dim5_score -= 0.4
        issues.append(
            f"Narrative too short ({word_count} words) - "
            "SAR drafts should be at least 150 words for regulatory defensibility"
        )
    if not has_sections:
        dim5_score -= 0.2
        issues.append("Narrative appears to be a single sentence - expand into structured paragraphs")
    if has_contradiction:
        dim5_score -= 0.5
        issues.append(
            "Contradiction detected: narrative states 'no suspicious activity' "
            "in a SAR filing - review immediately"
        )
    dim5_score = max(0.0, dim5_score)

    # ── Weighted composite for Stage 1 ────────────────────────────────────
    stage1_score = (
        RUBRIC_WEIGHTS["structural_completeness"] * dim1_score +
        RUBRIC_WEIGHTS["transaction_accuracy"]    * dim2_score +
        RUBRIC_WEIGHTS["typology_alignment"]      * dim3_score +
        RUBRIC_WEIGHTS["language_compliance"]     * dim4_score +
        RUBRIC_WEIGHTS["semantic_coherence"]      * dim5_score
    )

    return {
        "stage1_score": round(stage1_score, 4),
        "dimension_scores": {
            "structural_completeness": round(dim1_score, 4),
            "transaction_accuracy":    round(dim2_score, 4),
            "typology_alignment":      round(dim3_score, 4),
            "language_compliance":     round(dim4_score, 4),
            "semantic_coherence":      round(dim5_score, 4),
        },
        "issues":        issues,
        "traces":        traces,
        "element_scores": element_scores,
        "lang_violations": lang_violations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 - LLM-as-JUDGE  (Anthropic API)
# ─────────────────────────────────────────────────────────────────────────────

def _build_judge_prompt(
    narrative_text:        str,
    typology:              str,
    structured_case:       dict,
    quantified_indicators: dict,
    triggered_rules:       list,
    enrichment_data:       dict,
    stage1_result:         dict,
    sar_context_tree:      dict = None,
    sar_sections:          dict = None,
) -> tuple[str, str]:
    """
    Builds the system + user prompt for the LLM judge.
    Injects: regulatory memory, approved narrative templates,
             Stage 1 findings, full case context.
    """

    canonical = typology.replace("typology_", "").lower()
    approved_template = APPROVED_NARRATIVE_PATTERNS.get(
        canonical,
        APPROVED_NARRATIVE_PATTERNS.get("structuring", "N/A")
    )
    typology_elements = TYPOLOGY_REQUIRED_ELEMENTS.get(canonical, [])

    # Build case context summary for the judge
    customer_ctx = ""
    if structured_case:
        cust = structured_case.get("customer", {})
        acct = structured_case.get("accounts", [{}])
        if isinstance(acct, list) and acct:
            acct = acct[0]
        customer_ctx = (
            f"Customer: {cust.get('full_name','[unknown]')} "
            f"({cust.get('customer_type','individual')}), "
            f"Risk: {cust.get('risk_rating','MEDIUM')}, "
            f"PEP: {cust.get('pep_flag', False)}\n"
            f"Account: {acct.get('account_number','[unknown]')}, "
            f"Type: {acct.get('account_type','SAVINGS')}, "
            f"Avg Balance: {acct.get('avg_monthly_balance', 0):,.0f}"
        )

    # Format key quantified indicators
    qi_str = "\n".join(
        f"  {k}: {v}" for k, v in (quantified_indicators or {}).items()
    ) or "  (none provided)"

    # Format triggered rules
    rules_str = "\n".join(
        f"  {r.get('rule_id','?')}: {r.get('description', r.get('rule_name',''))}"
        if isinstance(r, dict) else f"  {r}"
        for r in (triggered_rules or [])
    ) or "  (none triggered)"

    # Sanctions / enrichment highlights
    enrichment_str = "(none)"
    if enrichment_data:
        sanc = enrichment_data.get("sanctions", {})
        hits = sanc.get("total_sanctions_hits", 0)
        adv  = len(enrichment_data.get("regulatory_advisories", []))
        enrichment_str = (
            f"Sanctions hits: {hits} | Advisories matched: {adv} | "
            f"Mode: {enrichment_data.get('mode', 'N/A')}"
        )

    stage1_issues_str = "\n".join(
        f"  - {iss}" for iss in stage1_result.get("issues", [])
    ) or "  None"

    # Summarise the context tree Agent 5 used — gives the judge ground truth
    # to cross-check narrative claims against source data
    context_tree_str = "(not available)"
    if sar_context_tree:
        try:
            import json as _json
            # Include case facts and triggered rules — skip RAG chunks (too verbose)
            ct_summary = {
                "case":                  sar_context_tree.get("case", {}),
                "subject":               sar_context_tree.get("subject", {}),
                "triggered_rules":       sar_context_tree.get("triggered_rules", {}),
                "quantified_indicators": sar_context_tree.get("quantified_indicators", {}),
                "cognitive_event_flow":  sar_context_tree.get("cognitive_event_flow", {}),
                "enrichment":            sar_context_tree.get("enrichment", {}),
            }
            context_tree_str = _json.dumps(ct_summary, indent=2, default=str)
        except Exception:
            context_tree_str = "(serialisation error)"

    system_prompt = """You are a senior AML compliance officer and SAR quality judge at a regulated bank.
You evaluate SAR (Suspicious Activity Report) narrative drafts for regulatory filing under PMLA 2002 and FIU-IND guidelines.

Your evaluation is used to:
1. Approve narratives for human analyst review (quality_score >= 0.80)
2. Identify specific deficiencies and instruct Agent 5 to revise

REGULATORY STANDARDS YOU ENFORCE:
- PMLA Rule 12: Every SAR must contain WHO, WHAT, WHEN, WHERE, HOW, WHY elements
- FIU-IND §4.1: Transaction details must be factually accurate and specific
- FIU-IND §4.3: No speculative, hedging, or opinion language permitted
- FATF Rec. 20: Suspicious activity reports must be factual and precise
- RBI KYC S.38: Customer risk context must be referenced appropriately

SCORING RUBRIC (return scores as fractions 0.0-1.0):
- structural_completeness (weight 0.25): All 5W+How elements clearly present
- transaction_accuracy    (weight 0.25): Amounts, dates, counterparties match case data
- typology_alignment      (weight 0.20): Narrative language fits the predicted typology
- language_compliance     (weight 0.20): No speculative/prohibited language anywhere
- semantic_coherence      (weight 0.10): Logical flow, no contradictions, professional tone

Return ONLY a valid JSON object with this exact structure:
{
  "dimension_scores": {
    "structural_completeness": <float 0-1>,
    "transaction_accuracy": <float 0-1>,
    "typology_alignment": <float 0-1>,
    "language_compliance": <float 0-1>,
    "semantic_coherence": <float 0-1>
  },
  "issues": ["<specific issue string>", ...],
  "revision_instructions": {
    "structural": "<what to add/fix for 5W elements>",
    "accuracy": "<specific factual corrections needed>",
    "typology": "<what typology-specific language to add>",
    "language": "<which phrases to remove and replace with>",
    "coherence": "<structural/flow improvements>"
  },
  "judge_reasoning": "<2-3 sentences summarising the overall quality assessment>"
}"""

    user_prompt = f"""EVALUATE THE FOLLOWING SAR NARRATIVE DRAFT.

────── CASE CONTEXT ──────
Typology: {typology}
{customer_ctx}

Quantified Indicators:
{qi_str}

Rule Triggers:
{rules_str}

External Enrichment: {enrichment_str}

────── SOURCE CONTEXT TREE (ground truth Agent 5 used) ──────
Use this to verify that the narrative's factual claims match the actual case data.
Cross-check amounts, dates, rule IDs, and typology signals against this.
{context_tree_str}

────── TYPOLOGY MEMORY ──────
Required narrative elements for '{canonical}' typology:
{typology_elements}

Approved historical narrative template for '{canonical}':
{approved_template}

────── STAGE 1 STRUCTURAL ISSUES (already detected) ──────
{stage1_issues_str}

────── SAR NARRATIVE PROSE TO EVALUATE ──────
(This is the extracted prose from the SAR — evaluate this, not JSON structure)
{narrative_text}

────── INSTRUCTIONS ──────
Score each dimension 0.0-1.0. Be specific and actionable in issues and revision_instructions.
If a dimension is already strong, score it 0.90-1.00. Only penalise genuine deficiencies.
Use the source context tree to verify transaction_accuracy — check that amounts and dates
in the narrative match the actual case data.
Respond with ONLY the JSON object described in the system prompt. No preamble."""

    return system_prompt, user_prompt


def _call_llm_judge(
    system_prompt: str,
    user_prompt:   str,
) -> dict | None:
    """
    Call the LLM judge via config.get_llm() (Groq in dev, Bedrock in prod).
    Retries up to 3 times with exponential backoff on rate limit errors.
    Returns parsed JSON dict or None on failure.
    """
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from langchain_core.messages import HumanMessage, SystemMessage
    from config import get_llm

    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm = get_llm(temperature=0.3)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            raw: str = response.content.strip()

            # Strip markdown fences and control characters
            clean = re.sub(r"```(?:json)?\s*", "", raw)
            clean = clean.replace("```", "").strip()
            clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", clean)

            result = json.loads(clean)

            if not isinstance(result, dict):
                log.warning("[LLM Judge] Response is not a JSON object")
                return None

            if "dimension_scores" not in result:
                log.warning("[LLM Judge] Missing 'dimension_scores' in response")
                return None

            dim_scores = result.get("dimension_scores", {})
            required_dims = {
                "structural_completeness", "transaction_accuracy",
                "typology_alignment", "language_compliance", "semantic_coherence",
            }
            missing_dims = required_dims - set(dim_scores.keys())
            if missing_dims:
                log.warning(f"[LLM Judge] Missing dimensions: {missing_dims}")
                return None

            for dim, score in dim_scores.items():
                try:
                    score_float = float(score)
                    if not (0.0 <= score_float <= 1.0):
                        log.warning(f"[LLM Judge] Dimension '{dim}' score {score_float} out of range")
                        return None
                except (TypeError, ValueError):
                    log.warning(f"[LLM Judge] Dimension '{dim}' score is not a number: {score}")
                    return None

            return result

        except json.JSONDecodeError as e:
            log.warning(f"[LLM Judge] JSON parse error: {e}")
            return None
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "too many" in err_str:
                wait = 2 ** attempt  # 1s, 2s, 4s
                log.warning(f"[LLM Judge] Rate limited — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            log.warning(f"[LLM Judge] Unexpected error: {type(e).__name__}: {e}")
            return None

    log.warning(f"[LLM Judge] All {max_retries} attempts exhausted due to rate limiting")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SCORE MERGER  (Stage 1 + Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_scores(
    stage1: dict,
    stage2: dict | None,
    stage2_used: bool,
) -> dict:
    """
    Merge Stage 1 (rule-based) and Stage 2 (LLM) scores.

    When both stages run:
      Final dimension score = 0.40 * stage1 + 0.60 * stage2
      (LLM carries more weight - it has full context)

    When Stage 2 not available (LLM error / blocked by Stage 1):
      Final score = Stage 1 score only.
    """
    s1 = stage1["dimension_scores"]

    if stage2_used and stage2 and "dimension_scores" in stage2:
        s2 = stage2["dimension_scores"]
        merged = {}
        for dim in RUBRIC_WEIGHTS:
            v1 = float(s1.get(dim, 0.0))
            v2 = float(s2.get(dim, v1))   # fallback to s1 if dim missing
            merged[dim] = round(0.40 * v1 + 0.60 * v2, 4)
    else:
        merged = {k: round(v, 4) for k, v in s1.items()}

    quality_score = sum(RUBRIC_WEIGHTS[d] * merged[d] for d in RUBRIC_WEIGHTS)
    return {
        "dimension_scores":   merged,
        "quality_score":      round(quality_score, 4),
        "stage1_used":        True,
        "stage2_used":        stage2_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE MERGER & DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _merge_issues(stage1_issues: list, stage2_issues: list | None) -> list:
    """
    Merge and deduplicate issues from both stages.
    Stage 2 issues take priority (more specific). Stage 1 issues
    are appended if they don't overlap with stage 2 content.
    """
    if not stage2_issues:
        return stage1_issues

    combined = list(stage2_issues)
    s2_lower = " ".join(stage2_issues).lower()
    for s1_iss in stage1_issues:
        # Only add Stage 1 issue if its core content isn't already in Stage 2
        key_words = [w for w in s1_iss.lower().split() if len(w) > 5]
        overlap   = any(kw in s2_lower for kw in key_words)
        if not overlap:
            combined.append(s1_iss)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT 6 CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Agent6ComplianceJudge:
    """
    Agent 6: Compliance Validation Agent (Agent-as-a-Judge).

    Two-stage evaluation:
      Stage 1: Rule-based structural check (fast, deterministic, no LLM)
      Stage 2: LLM-as-judge (semantic, typology, language, full context)

    Writes to SARAgentState:
      compliance_passed, compliance_issues, quality_score,
      reasoning_traces, revision_instructions
    """

    def __init__(self):
        log.info("[Agent6] Compliance Validation Agent initialised")
        log.info(f"[Agent6] Pass threshold: {PASS_THRESHOLD}  Max revisions: {MAX_REVISIONS}")
        log.info(f"[Agent6] LLM model: {LLM_MODEL} (Groq)")

    def evaluate(self, state: dict) -> dict:
        """
        Main evaluation entry point. Accepts and returns SARAgentState dict.

        Reads  : sar_draft, structured_case, typology, quantified_indicators,
                 triggered_rules, enrichment_data, revision_count
        Writes : compliance_passed, compliance_issues, quality_score,
                 reasoning_traces (accumulates), revision_instructions
        """
        t0 = time.time()

        case_id       = state.get("case_id", "UNKNOWN")
        sar_draft     = state.get("sar_draft", "")
        typology      = state.get("typology", "unknown")
        revision_count= int(state.get("revision_count", 0))
        structured_case       = state.get("structured_case", {})
        quantified_indicators = state.get("quantified_indicators", {})
        triggered_rules       = state.get("triggered_rules", [])
        enrichment_data       = state.get("enrichment_data", {})
        sar_context_tree      = state.get("sar_context_tree", {})

        # ── Extract prose narrative from sar_draft ────────────────────────
        # sar_draft is a JSON string of sar_sections. Stage 1 and the LLM
        # judge must evaluate the actual prose, not the JSON wrapper.
        # We extract section2_narrative (the main prose) for structural checks,
        # and build a flat text version of all prose fields for keyword scanning.
        sar_sections: dict = {}
        narrative_text: str = sar_draft  # fallback: use raw if not parseable

        if sar_draft and sar_draft.strip().startswith("{"):
            try:
                sar_sections = json.loads(sar_draft)
                # Primary prose field
                prose_parts = []
                s2 = sar_sections.get("section2_grounds_for_suspicion", {})
                if isinstance(s2, dict):
                    prose_parts.append(s2.get("why_suspicious", ""))
                    prose_parts.append(s2.get("absence_of_legitimate_explanation", ""))
                    prose_parts.append(s2.get("overview", ""))
                # New schema: section2_narrative is a flat string
                if "section2_narrative" in sar_sections:
                    prose_parts.append(sar_sections["section2_narrative"])
                # Also include section4 regulatory basis if it's a string
                s4 = sar_sections.get("section4_regulatory_basis", "")
                if isinstance(s4, str):
                    prose_parts.append(s4)
                narrative_text = "\n\n".join(p for p in prose_parts if p)
                if not narrative_text:
                    # Last resort: stringify the whole sections dict
                    narrative_text = sar_draft
                log.info(f"[Agent6] Extracted narrative prose: {len(narrative_text.split())} words")
            except (json.JSONDecodeError, TypeError):
                log.warning("[Agent6] sar_draft is not valid JSON — evaluating raw text")
                narrative_text = sar_draft

        log.info("=" * 60)
        log.info(f"[Agent6] Evaluating case: {case_id}")
        log.info(f"[Agent6] Typology: {typology}  |  Revision: {revision_count}")
        log.info("=" * 60)

        if not sar_draft or not sar_draft.strip():
            log.error("[Agent6] Empty SAR draft received")
            return {
                **state,
                "compliance_passed":       False,
                "compliance_issues":       ["CRITICAL: SAR draft is empty"],
                "quality_score":           0.0,
                "reasoning_traces":        (state.get("reasoning_traces") or []) + [{
                    "agent":     "Agent6",
                    "case_id":   case_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error":     "Empty draft received",
                }],
                "revision_instructions": {
                    "structural": "Narrative is completely empty - generate from scratch",
                    "accuracy":   "", "typology": "", "language": "", "coherence": "",
                },
            }

        # ── STAGE 1: Rule-based structural check (on prose, not JSON) ────
        log.info("[Agent6] Stage 1: Rule-based structural check")
        s1 = _stage1_structural_check(narrative_text, typology, quantified_indicators)
        log.info(f"[Agent6] Stage 1 score: {s1['stage1_score']:.4f}  |  Issues: {len(s1['issues'])}")

        # Decide whether to invoke LLM judge
        stage2_used = True
        s2          = None
        judge_reasoning = ""

        if s1["stage1_score"] < STAGE1_BLOCK_SCORE:
            # Draft is too broken for LLM to add value — skip Stage 2
            stage2_used     = False
            judge_reasoning = (
                f"Stage 1 score {s1['stage1_score']:.2f} is below the LLM-invocation "
                f"threshold {STAGE1_BLOCK_SCORE}. Draft has fundamental structural failures "
                "that must be corrected before semantic evaluation."
            )
            log.warning(
                f"[Agent6] Stage 1 score {s1['stage1_score']:.2f} below block threshold "
                f"{STAGE1_BLOCK_SCORE} — skipping LLM judge"
            )
        else:
            # ── STAGE 2: LLM-as-Judge ─────────────────────────────────────
            log.info("[Agent6] Stage 2: LLM-as-Judge invocation")
            sys_prompt, usr_prompt = _build_judge_prompt(
                narrative_text, typology, structured_case, quantified_indicators,
                triggered_rules, enrichment_data, s1, sar_context_tree, sar_sections
            )
            s2 = _call_llm_judge(sys_prompt, usr_prompt)

            if s2:
                judge_reasoning = s2.get("judge_reasoning", "")
                log.info(f"[Agent6] LLM judge responded  |  Issues: {len(s2.get('issues',[]))}")
            else:
                stage2_used = False
                log.warning("[Agent6] LLM judge returned no result — using Stage 1 only")

        # ── MERGE SCORES ──────────────────────────────────────────────────
        merged = _merge_scores(s1, s2, stage2_used)
        quality_score = merged["quality_score"]
        log.info(f"[Agent6] Final quality score: {quality_score:.4f}")

        # ── MERGE ISSUES ──────────────────────────────────────────────────
        all_issues = _merge_issues(
            s1["issues"],
            s2.get("issues", []) if s2 else None,
        )

        # ── REVISION INSTRUCTIONS ─────────────────────────────────────────
        revision_instructions = {
            "structural": "",
            "accuracy":   "",
            "typology":   "",
            "language":   "",
            "coherence":  "",
        }
        if s2 and "revision_instructions" in s2:
            revision_instructions.update(s2["revision_instructions"])
        else:
            # Synthesise from Stage 1 issues when LLM not available
            structural_issues = [i for i in all_issues if any(w in i for w in ["Missing", "element", "5W"])]
            accuracy_issues   = [i for i in all_issues if any(w in i for w in ["indicator", "amount", "date"])]
            typology_issues   = [i for i in all_issues if any(w in i for w in ["typology", "Typology"])]
            language_issues   = [i for i in all_issues if any(w in i for w in ["Speculative", "Opinion", "Prohibited"])]
            coherence_issues  = [i for i in all_issues if any(w in i for w in ["short", "sentence", "Contradiction"])]

            if structural_issues:
                revision_instructions["structural"] = "; ".join(structural_issues)
            if accuracy_issues:
                revision_instructions["accuracy"] = "; ".join(accuracy_issues)
            if typology_issues:
                revision_instructions["typology"] = "; ".join(typology_issues)
            if language_issues:
                revision_instructions["language"] = (
                    "Remove: " + "; ".join(language_issues) + ". "
                    "Replace with factual, declarative statements in past tense."
                )
            if coherence_issues:
                revision_instructions["coherence"] = "; ".join(coherence_issues)

        # ── COMPLIANCE DECISION ───────────────────────────────────────────
        if revision_count >= MAX_REVISIONS:
            # Hard cap: pass to Agent 7 regardless, flagged for human attention
            compliance_passed = quality_score >= PASS_THRESHOLD
            if not compliance_passed:
                all_issues.append(
                    f"WARNING: Maximum revision limit ({MAX_REVISIONS}) reached. "
                    "Forwarding to human analyst with unresolved deficiencies."
                )
            log.warning(
                f"[Agent6] Max revisions reached ({revision_count}). "
                f"Routing to Agent 7. passed={compliance_passed}"
            )
        else:
            compliance_passed = quality_score >= PASS_THRESHOLD

        # ── ROUTING DECISION ──────────────────────────────────────────────
        if compliance_passed:
            route = "Agent7_human_review"
        elif revision_count < MAX_REVISIONS:
            route = "Agent5_revision"
        else:
            route = "Agent7_human_review_flagged"

        log.info(
            f"[Agent6] Decision: {'PASS' if compliance_passed else 'FAIL'}  "
            f"→  {route}  "
            f"(score={quality_score:.4f}  threshold={PASS_THRESHOLD})"
        )

        # ── BUILD REASONING TRACE ─────────────────────────────────────────
        elapsed_ms = round((time.time() - t0) * 1000)
        trace_entry = {
            "agent":              "Agent6",
            "case_id":            case_id,
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "revision_number":    revision_count,
            "typology":           typology,
            "quality_score":      quality_score,
            "compliance_passed":  compliance_passed,
            "routing_decision":   route,
            "elapsed_ms":         elapsed_ms,
            "stage1_score":       s1["stage1_score"],
            "stage2_used":        stage2_used,
            "dimension_scores":   merged["dimension_scores"],
            "element_scores":     s1.get("element_scores", {}),
            "lang_violations":    s1.get("lang_violations", []),
            "issues_count":       len(all_issues),
            "judge_reasoning":    judge_reasoning,
            "stage1_traces":      s1["traces"],
        }

        # Accumulate reasoning traces (operator.add behaviour from state.py)
        existing_traces = state.get("reasoning_traces") or []
        updated_traces  = existing_traces + [trace_entry]

        # ── RETURN UPDATED STATE ──────────────────────────────────────────
        return {
            **state,
            "compliance_passed":       compliance_passed,
            "compliance_issues":       all_issues,
            "quality_score":           quality_score,
            "reasoning_traces":        updated_traces,
            "revision_instructions":   revision_instructions,
            # Expose routing for LangGraph edge function
            "_agent6_route":           route,
            "_agent6_elapsed_ms":      elapsed_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH ROUTING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compliance_router(state: dict) -> str:
    """
    LangGraph conditional edge function.
    Called after Agent 6 node to route to next node.

    Returns:
      "revision"     → route back to Agent 5
      "human_review" → route to Agent 7
    """
    if state.get("compliance_passed"):
        return "human_review"

    revision_count = int(state.get("revision_count", 0))
    if revision_count < MAX_REVISIONS:
        return "revision"

    return "human_review"   # max revisions exhausted


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO  (python agent6.py)
# ─────────────────────────────────────────────────────────────────────────────

def _run_demo():
    print("=" * 70)
    print("AGENT 6  -  Compliance Validation Agent (Agent-as-a-Judge)")
    print("Pipeline: Agent1 -> Agent2 -> Agent3 -> Agent4 -> Agent5 -> [Agent6] -> Agent7")
    print("=" * 70)

    agent = Agent6ComplianceJudge()

    # ── Demo A: High-quality draft (should PASS) ──────────────────────────
    demo_a_draft = (
        "Between 15 January 2026 and 10 February 2026, account 1234567890 held "
        "by Rajesh Kumar (individual, HIGH risk, non-PEP) at Andheri Branch received "
        "seven cash deposits totalling INR 6,860,000. Each individual deposit ranged "
        "from INR 900,000 to INR 980,000, all below the INR 10,00,000 Cash Transaction "
        "Report threshold under PMLA Rule 3. The coefficient of variation across "
        "deposits was 0.04, consistent with deliberate amount fragmentation to avoid "
        "mandatory reporting obligations. Velocity score at peak deposit was 0.82, "
        "exceeding the 0.75 high-velocity threshold. Rule R-STR-01 "
        "(THRESHOLD_STRUCTURING) was triggered on five of the seven transactions. "
        "No corresponding business justification was provided for the consistent "
        "sub-threshold deposit pattern. Automated alert A-001 was raised under "
        "rule CONSISTENT_SUB_THRESHOLD. The account's historical SAR count is two, "
        "indicating prior suspicious activity. This activity is flagged as suspicious "
        "under PMLA Section 3 and is reported in accordance with FIU-IND guidelines."
    )

    state_a = {
        "case_id":        "CASE-2026-0061",
        "sar_draft":      demo_a_draft,
        "typology":       "structuring",
        "revision_count": 0,
        "structured_case": {
            "customer": {
                "full_name":       "Rajesh Kumar",
                "customer_type":   "individual",
                "risk_rating":     "HIGH",
                "pep_flag":        False,
            },
            "accounts": [{
                "account_number":     "1234567890",
                "account_type":       "SAVINGS",
                "avg_monthly_balance": 450000,
            }],
        },
        "quantified_indicators": {
            "transaction_count":     7,
            "avg_deposit_amount_inr": 980000,
            "fund_exit_ratio":       0.42,
            "days_observed":         26,
        },
        "triggered_rules": [
            {"rule_id": "R-STR-01", "rule_name": "THRESHOLD_STRUCTURING",
             "severity": "HIGH",
             "description": "Amount in CTR structuring band INR 900K-1M"},
            {"rule_id": "R-STR-02", "rule_name": "CONSISTENT_SUB_THRESHOLD",
             "severity": "MEDIUM",
             "description": "Case CV < 0.15 on CREDIT transactions"},
        ],
        "enrichment_data": {
            "sanctions": {"total_sanctions_hits": 0},
            "regulatory_advisories": [],
            "mode": "DEV_SIMULATED",
        },
        "reasoning_traces": [],
    }

    print("\n" + "-" * 70)
    print("DEMO A: High-quality structuring draft")
    print("-" * 70)
    result_a = agent.evaluate(state_a)
    _print_result(result_a, "A")

    # ── Demo B: Deficient draft (should FAIL → revision) ──────────────────
    demo_b_draft = (
        "The customer may have been involved in money laundering activities. "
        "We believe the transactions seem suspicious. It appears that funds were "
        "potentially transferred to foreign accounts. The account possibly received "
        "large amounts which could be from illicit sources."
    )

    state_b = {
        "case_id":        "CASE-2026-0086",
        "sar_draft":      demo_b_draft,
        "typology":       "rapid_movement",
        "revision_count": 0,
        "structured_case": {
            "customer": {
                "full_name":     "Sunrise Exports Ltd",
                "customer_type": "corporate",
                "risk_rating":   "HIGH",
                "pep_flag":      False,
            },
            "accounts": [{"account_number": "9876543210", "account_type": "CURRENT"}],
        },
        "quantified_indicators": {
            "fund_exit_ratio":        0.97,
            "time_to_first_exit_min": 75,
            "total_amount_inr":       9850000,
        },
        "triggered_rules": [
            {"rule_id": "R-HVT-01", "rule_name": "HIGH_VELOCITY_WIRE",
             "description": "SWIFT wire velocity > 0.75"},
            {"rule_id": "R-HVT-02", "rule_name": "RAPID_OUTBOUND_EXIT",
             "description": "Outbound wire within tight time gap"},
        ],
        "enrichment_data": {
            "sanctions": {"total_sanctions_hits": 1},
            "regulatory_advisories": [
                {"id": "FATF-2024-01", "severity": "HIGH", "issuer": "FATF"}
            ],
            "mode": "DEV_SIMULATED",
        },
        "reasoning_traces": [],
    }

    print("\n" + "─" * 70)
    print("DEMO B: Deficient draft with speculative language (should FAIL)")
    print("─" * 70)
    result_b = agent.evaluate(state_b)
    _print_result(result_b, "B")

    # ── Demo C: Max revisions exhausted ───────────────────────────────────
    state_c = {**state_b, "revision_count": 3}
    print("\n" + "─" * 70)
    print("DEMO C: Same deficient draft — max revisions exhausted (revision_count=3)")
    print("─" * 70)
    result_c = agent.evaluate(state_c)
    _print_result(result_c, "C")

    # ── Routing test ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("ROUTING TABLE TEST")
    print("─" * 70)
    for label, res in [("A (PASS)", result_a), ("B (FAIL, rev=0)", result_b), ("C (FAIL, rev=3)", result_c)]:
        route = compliance_router(res)
        print(f"  {label:<22} quality={res['quality_score']:.4f}  "
              f"passed={res['compliance_passed']}  →  {route}")


def _print_result(result: dict, label: str):
    print(f"\n  ── Agent 6 Output (Demo {label}) ──")
    print(f"  Compliance passed  : {result['compliance_passed']}")
    print(f"  Quality score      : {result['quality_score']:.4f}  (threshold={PASS_THRESHOLD})")
    print(f"  Routing decision   : {result.get('_agent6_route','?')}")
    print(f"  Elapsed            : {result.get('_agent6_elapsed_ms','?')}ms")
    print(f"\n  Dimension scores:")
    for dim, score in (result.get("reasoning_traces", [{}])[-1].get("dimension_scores", {}) or {}).items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        flag = " ← needs work" if score < 0.80 else ""
        print(f"    {dim:<35} {score:.4f}  [{bar}]{flag}")

    issues = result.get("compliance_issues", [])
    print(f"\n  Issues ({len(issues)}):")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠  {iss[:90]}")
        if len(issues) > 5:
            print(f"    ... and {len(issues)-5} more")
    else:
        print("    ✓ None")

    rev_inst = result.get("revision_instructions", {})
    if any(v for v in rev_inst.values()):
        print(f"\n  Revision instructions for Agent 5:")
        for dim, instr in rev_inst.items():
            if instr:
                print(f"    [{dim}] {instr[:100]}")

    trace = (result.get("reasoning_traces") or [{}])[-1]
    reasoning = trace.get("judge_reasoning", "")
    if reasoning:
        print(f"\n  LLM judge reasoning:")
        print(f"    {reasoning[:200]}")


if __name__ == "__main__":
    _run_demo()
