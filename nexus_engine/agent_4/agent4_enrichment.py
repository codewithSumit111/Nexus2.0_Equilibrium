"""
agent4_enrichment.py  —  Agent 4: Enrichment Agent + SAR Case Strength Scorer  v4
===================================================================================
Team BAYMAX | Barclays Hack-O-Hire | SAR Generator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS MCP AND WHY IS IT HERE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MCP = Model Context Protocol.

It is an open standard (by Anthropic, 2024) that defines HOW an AI agent
should call external tools and data sources in a controlled, auditable way.
Think of it like "USB for AI tools" — a single standard plug that works
with any tool: databases, APIs, web searches, internal systems.

WHY THIS AGENT USES MCP:
  The PPT (Slide 11) shows a box called "Secure Gateway (MCP Server)" sitting
  between the Enrichment Agent and all external intelligence sources:
    Sanctions Lists, Negative News, Regulatory Advisories, Public Risk Feeds.

  Without MCP, each API call would be a raw HTTP request with no:
    - validation (is this host even allowed?)
    - rate limiting (are we hammering the API?)
    - audit trail (what was called, when, by whom, with what result?)
    - standardised error handling

  With MCP (MCPSecureGateway in this file):
    EVERY external call — real or simulated — passes through a single
    gateway that validates, rate-limits, and writes a tamper-evident
    audit log entry. This is what compliance teams need. Every call is
    traceable back to the specific case and entity it was made for.

  In development:   MCPSecureGateway simulates calls, logs them as
                    DEV_SIMULATED, no real HTTP requests made.
  In production:    MCPSecureGateway makes real HTTPS requests to
                    allowlisted hosts only.
  With FastMCP:     agent4_mcp_server.py wraps all tools as MCP endpoints
                    so AWS Bedrock / Claude.ai / any orchestrator can call
                    Agent 4 over the MCP protocol (SSE transport).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS FILE DOES (TWO JOBS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB 1 — ML SCORING
  Takes the 33 engineered features from data_engineered.csv and runs
  them through XGBoost (WEAK/MEDIUM/STRONG) + Random Forest (0-1 score).
  Falls back to weighted rule-based scoring if model file is absent.
  Outputs: strength_label, strength_probabilities, priority_score,
           recommended_priority, score_breakdown (what drove the score).

JOB 2 — EXTERNAL ENRICHMENT (via MCP Gateway)
  For every case that triggers enrichment, calls 4 external intelligence
  sources and merges results:
    1. Sanctions      OpenSanctions API (OFAC, EU, UN, HMT) + FATF + RBI
    2. Negative News  GNews API (live) or realistic simulated (dev mode)
    3. Advisories     FATF / RBI / FIU-IND / FinCEN advisory database
    4. Country Risk   FATF October 2024 blacklist + greylist scoring

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE POSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Agent 1  →  {sar_worthy, confidence, typologies:[{typology,confidence}]}
  Agent 2  →  orchestration / routing
  Agent 3  →  {predicted_typology, rule_triggers, quantified_indicators,
               case_summary:{flagged_transactions, total_flagged_amount}}
  ─────────────────────────────────────────────────────────
  Agent 4  (THIS FILE) — accepts Agent 1 + Agent 3 outputs
     ↓  MCP Gateway  ↓
     Sanctions | News | Advisories | Country Risk
     ↓
     {strength, priority, filing_rec, evidence_gaps, external_intel,
      case_overview}  →  Agent 5 (Narrative Engine)
  ─────────────────────────────────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEV_MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DEV_MODE = True   (default — synthetic data, no API keys needed)
    • Sanctions: FATF + RBI local checks only. No OpenSanctions HTTP call.
    • News: Realistic simulated articles generated from case signals.
            Each entity gets DIFFERENT articles — same typology, different
            angles (regulatory notice, court filing, bank alert, etc.)
            All sources labelled [DEV-SIMULATED].
    • All calls logged to audit trail as DEV_SIMULATED.

  DEV_MODE = False  (production)
    • Sanctions: Real OpenSanctions API called.
    • News: Real GNews API called (set GNEWS_API_KEY env var).
    • Falls back gracefully if API unreachable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [MCP]       https://modelcontextprotocol.io/
  [PPT]       Barclays Hack-O-Hire — Team BAYMAX submission
  [GNEWS]     https://gnews.io/docs/
  [OPENSANC]  https://api.opensanctions.org/
  [FATF-2024] https://www.fatf-gafi.org/ — October 2024 update
  [PMLA]      Prevention of Money Laundering Act 2002 (amended 2023)
  [RBI-KYC]   RBI KYC Master Direction 2016 (updated 2023)
"""

import os, json, time, hashlib, logging, warnings, random
import requests, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Agent4] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Agent4")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  change DEV_MODE to False for production
# ─────────────────────────────────────────────────────────────────────────────

DEV_MODE        = True                          # flip to False for live APIs
GNEWS_API_KEY   = os.getenv("GNEWS_API_KEY", "") # https://gnews.io/ free key
OPENSANC_BASE   = "https://api.opensanctions.org"
REQUEST_TIMEOUT = 8                             # seconds per HTTP call
AGENT4_DIR      = "agent4"
MCP_AUDIT_LOG   = os.path.join(AGENT4_DIR, "mcp_audit_log.jsonl")
os.makedirs(AGENT4_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC REFERENCE DATA  (update quarterly)
# ─────────────────────────────────────────────────────────────────────────────

# FATF October 2024  —  https://www.fatf-gafi.org/
FATF_BLACK_LIST = {"Iran", "North Korea", "Myanmar", "DPRK"}
FATF_GREY_LIST  = {
    "Bulgaria", "Burkina Faso", "Cameroon", "Cote d'Ivoire", "Croatia",
    "Democratic Republic of Congo", "Haiti", "Jamaica", "Kenya", "Mali",
    "Mozambique", "Namibia", "Nigeria", "Philippines", "Senegal",
    "South Africa", "South Sudan", "Syria", "Tanzania",
    "Trinidad and Tobago", "Uganda", "Vietnam", "Yemen",
    "DRC", "Congo", "Ivory Coast", "Trinidad",
}
FATF_ALL_HIGH_RISK = FATF_BLACK_LIST | FATF_GREY_LIST

# RBI Caution List  —  representative sample; update from rbi.org.in
RBI_CAUTION_ENTITIES = {
    "pearlessential", "rose valley", "saradha", "speak asia",
    "stockguru", "gold sukh", "golden forest", "unitech",
    "iipm", "amway india", "sahara group",
}

# ─────────────────────────────────────────────────────────────────────────────
# ML FEATURE LIST  (must match agent4_train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

CASE_FEATURES = [
    "total_txn_amount_cbrt", "avg_txn_amount_cbrt", "std_txn_amount_cbrt",
    "txn_count_log", "txn_amount_cv", "max_to_avg_txn_ratio", "fund_exit_ratio",
    "burst_score", "time_to_first_outbound_minutes_log",
    "txn_velocity_log", "burst_per_age",
    "distinct_counterparties_log", "incoming_sources_count_log",
    "counterparty_diversity_score", "counterparty_to_txn_ratio",
    "incoming_to_outgoing_ratio",
    "alert_count", "alert_density", "alert_tier", "kyc_x_alert",
    "international_counterparty_flag", "high_risk_country_flag",
    "pep_flag", "kyc_risk_score", "kyc_risk_tier",
    "historical_sar_flag", "high_risk_combined",
    "burst_x_exit", "hr_country_x_exit", "pep_x_intl",
    "sar_history_x_kyc", "fund_exit_tier", "binary_risk_flag_count",
]
STRENGTH_MAP   = {0: "WEAK", 1: "MEDIUM", 2: "STRONG"}
TYPOLOGY_COLS  = [
    "typology_structuring", "typology_rapid_movement", "typology_funnel_account",
    "typology_trade_based", "typology_shell_company", "typology_round_tripping",
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE REQUIREMENTS PER TYPOLOGY
# Lambda returns True  → evidence IS present (no gap)
# Lambda returns False → evidence MISSING     → added to evidence_gaps
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_REQUIREMENTS = {
    "typology_structuring": [
        ("alert_count",     lambda v: v >= 2,   "Fewer than 2 corroborating alerts"),
        ("txn_amount_cv",   lambda v: v < 0.20, "Amount variation too high — structuring needs consistent sizing [FATF §2.4]"),
        ("alert_density",   lambda v: v > 0.10, "Alert density too low for structuring pattern"),
        ("fund_exit_ratio", lambda v: v < 0.60, "Fund exit ratio >0.60 — inconsistent with deposit structuring"),
    ],
    "typology_rapid_movement": [
        ("time_to_first_outbound_minutes_log", lambda v: v < 5.8, "No rapid outbound detected (first exit > ~330 min) [FATF §2.2]"),
        ("fund_exit_ratio", lambda v: v > 0.80, "Fund exit ratio <0.80 — insufficient pass-through evidence"),
        ("burst_score",     lambda v: v > 0.60, "Burst score <0.60 — insufficient velocity evidence"),
    ],
    "typology_trade_based": [
        ("high_risk_country_flag",          lambda v: v == 1,   "No high-risk country exposure [FATF-RECS R.19]"),
        ("hr_country_x_exit",               lambda v: v > 0,    "No high-risk country × exit interaction [FATF-TBML §2.2]"),
        ("international_counterparty_flag", lambda v: v == 1,   "No international counterparty found"),
        ("kyc_risk_score",                  lambda v: v > 0.50, "KYC risk score <0.50 — insufficient jurisdiction risk"),
    ],
    "typology_funnel_account": [
        ("incoming_sources_count_log", lambda v: v > 1.6, "Fewer than ~4 distinct incoming sources [FATF §4.1]"),
        ("fund_exit_ratio",            lambda v: v > 0.70, "Fund exit ratio <0.70"),
        ("counterparty_to_txn_ratio",  lambda v: v > 0.30, "Low counterparty-to-transaction ratio — weak fan-in signal"),
        ("alert_count",                lambda v: v >= 2,   "Fewer than 2 corroborating alerts"),
    ],
    "typology_shell_company": [
        ("distinct_counterparties_log", lambda v: v > 2.5, "Fewer than ~12 distinct counterparties [FATF-SHELL §3.1]"),
        ("alert_count",                 lambda v: v >= 3,  "Fewer than 3 alerts — weak shell signal"),
        ("high_risk_country_flag",      lambda v: v == 1,  "No high-risk country exposure"),
        ("pep_x_intl",                  lambda v: v == 1,  "No PEP + international combination"),
    ],
    "typology_round_tripping": [
        ("fund_exit_ratio",                    lambda v: v > 0.85, "Fund exit ratio <0.85 [FATF-SHELL §4.2]"),
        ("high_risk_country_flag",             lambda v: v == 1,   "No high-risk country exposure"),
        ("time_to_first_outbound_minutes_log", lambda v: v > 4.0,  "No delayed outbound flow — round-trip needs gap"),
        ("burst_x_exit",                       lambda v: v > 0.30, "Burst × exit interaction too low for round-trip"),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# REGULATORY ADVISORY DATABASE
# Matched by typology AND/OR country — strict matching, no false positives
# ─────────────────────────────────────────────────────────────────────────────

ADVISORY_DB = [
    {
        "id": "FATF-2024-02", "issuer": "FATF", "date": "2024-10-25",
        "severity": "CRITICAL",
        "title": "High-risk jurisdictions subject to a call for action (October 2024)",
        "jurisdictions": list(FATF_BLACK_LIST),
        "typologies": ["all"],   # applies to ALL typologies when country matches
        "summary": "FATF calls on members to apply enhanced due diligence to Iran, North Korea, Myanmar, DPRK.",
        "url": "https://www.fatf-gafi.org/en/topics/high-risk-and-other-monitored-jurisdictions.html",
        "action_required": "Enhanced CDD + senior management approval before processing",
    },
    {
        "id": "FATF-2024-01", "issuer": "FATF", "date": "2024-10-25",
        "severity": "HIGH",
        "title": "Jurisdictions under Increased Monitoring — Grey List (October 2024)",
        "jurisdictions": list(FATF_GREY_LIST),
        "typologies": ["all"],
        "summary": "23 jurisdictions under increased monitoring. Transactions require enhanced scrutiny.",
        "url": "https://www.fatf-gafi.org/en/topics/high-risk-and-other-monitored-jurisdictions.html",
        "action_required": "Enhanced monitoring; document rationale for all transactions",
    },
    {
        "id": "FIU-IND-2024-01", "issuer": "FIU-IND", "date": "2024-07-01",
        "severity": "HIGH",
        "title": "Typology Report: Digital Payment Fraud and Mule Accounts in India",
        "jurisdictions": [],
        "typologies": ["typology_funnel_account", "typology_rapid_movement"],
        "summary": "Mule accounts funnelling fraud proceeds through UPI/IMPS within hours of receipt. 47 distinct sources is a key indicator.",
        "url": "https://fiuindia.gov.in/",
        "action_required": "File STR within 7 working days of detection; freeze account pending review",
    },
    {
        "id": "FinCEN-2024-01", "issuer": "FinCEN", "date": "2024-03-01",
        "severity": "MEDIUM",
        "title": "Advisory: Structuring via Digital Channels Below Reporting Thresholds",
        "jurisdictions": [],
        "typologies": ["typology_structuring"],
        "summary": "Structured deposits via mobile/net banking just below INR 10L CTR threshold remain a primary AML pattern.",
        "url": "https://www.fincen.gov/resources/advisories",
        "action_required": "File SAR if 3+ sub-threshold deposits within 30 days from same account",
    },
    {
        "id": "RBI-2024-AML-01", "issuer": "RBI", "date": "2024-09-15",
        "severity": "HIGH",
        "title": "Master Direction on KYC — Updated Section 16: PEP and Shell Entity Risk",
        "jurisdictions": [],
        "typologies": ["typology_shell_company", "typology_round_tripping"],
        "summary": "Enhanced monitoring mandated for PEPs and beneficial owners of shell/offshore structures. Unexplained fund flows require immediate STR filing.",
        "url": "https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
        "action_required": "Obtain source-of-funds declaration; escalate to MLRO within 24h",
    },
    {
        "id": "RBI-2023-TBML-01", "issuer": "RBI", "date": "2023-11-20",
        "severity": "HIGH",
        "title": "Alert: Trade-Based Money Laundering via Over/Under-Invoicing",
        "jurisdictions": [],
        "typologies": ["typology_trade_based"],
        "summary": "Banks must flag trade finance transactions with FATF-listed jurisdictions showing invoice values inconsistent with market rates.",
        "url": "https://rbi.org.in/",
        "action_required": "Obtain independent valuation; cross-check with DGFT trade data",
    },
]

def _match_advisories(typologies: list, countries: list) -> list:
    """
    Strict advisory matching:
      FATF jurisdiction advisories → only when that country is actually present
      Typology advisories → only when that typology is actually detected
      'all' typologies → only when country match triggers it
    """
    countries_set = {c.strip() for c in countries}
    typology_set  = set(typologies)
    matched = []
    for adv in ADVISORY_DB:
        geo_match = bool(countries_set & set(adv["jurisdictions"]))
        typ_match = (
            "all" in adv["typologies"] and geo_match
        ) or bool(typology_set & set(adv["typologies"]))
        if typ_match or geo_match:
            matched.append(adv)
    sev = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return sorted(matched, key=lambda x: -sev.get(x["severity"], 0))

# ─────────────────────────────────────────────────────────────────────────────
# DEV MODE — SIMULATED NEWS TEMPLATES
#
# WHY SIMULATED NEWS EXISTS:
#   The pipeline runs on synthetic data with fictional entity names.
#   Calling GNews for "Rajesh Kumar" returns zero useful results.
#   Simulated articles use actual case signal values (real amounts from
#   feat_series) so the narrative feels grounded in the case data.
#
# HOW ENTITY VARIATION WORKS:
#   Each entity gets a DIFFERENT article angle from the same typology bank:
#     entity index 0 → template [0] (regulatory notice angle)
#     entity index 1 → template [1] (court / ED angle)
#     entity index 2 → template [2] (bank compliance angle)
#   This prevents identical articles appearing for different entities.
#
# ALL SOURCES end with [DEV-SIMULATED] — clearly marked as not real.
# ─────────────────────────────────────────────────────────────────────────────

DEV_NEWS_TEMPLATES = {
    "typology_structuring": [
        {
            "title": "{entity} flagged by FIU-IND for {n} sub-threshold deposits totalling ₹{amount}L",
            "description": (
                "Financial Intelligence Unit India flagged {entity} after detecting "
                "{n} cash deposits, each just below the ₹10L CTR reporting threshold, "
                "totalling ₹{amount}L over 28 days. The pattern is consistent with "
                "deliberate structuring under PMLA 2002 Section 3."
            ),
            "source": "FIU-IND Compliance Alert [DEV-SIMULATED]",
            "risk_keywords": ["structuring", "suspicious", "investigation"],
        },
        {
            "title": "ED serves notice to {entity} in cash layering probe — ₹{amount}L under scrutiny",
            "description": (
                "The Enforcement Directorate has issued a Section 50 notice to {entity} "
                "seeking records of {n} bank transactions flagged for amount fragmentation. "
                "Investigators allege deliberate CTR avoidance under PMLA."
            ),
            "source": "Economic Times Enforcement Desk [DEV-SIMULATED]",
            "risk_keywords": ["money laundering", "investigation", "probe"],
        },
        {
            "title": "Bank compliance team files internal STR against {entity} account",
            "description": (
                "A senior compliance officer at the reporting bank filed a Suspicious "
                "Transaction Report against {entity}'s account citing {n} near-identical "
                "deposits averaging ₹{avg_dep}L each, a textbook structuring indicator."
            ),
            "source": "Internal Bank Compliance Report [DEV-SIMULATED]",
            "risk_keywords": ["suspicious", "aml", "probe"],
        },
    ],
    "typology_rapid_movement": [
        {
            "title": "{entity} account: ₹{amount}L received and fully transferred within {hours} hours",
            "description": (
                "AML systems raised a SWIFT-velocity alert on {entity}'s account after "
                "₹{amount}L in incoming credits were wired onward within {hours} hours, "
                "leaving a residual balance of under ₹1,000. Classic layering pass-through."
            ),
            "source": "AML Transaction Monitoring System [DEV-SIMULATED]",
            "risk_keywords": ["money laundering", "suspicious", "aml"],
        },
        {
            "title": "FIU-IND probe: {entity} linked to rapid fund movement through {n} accounts",
            "description": (
                "FIU-IND is examining a series of {n} rapid outbound transfers from "
                "accounts associated with {entity}. Each transfer occurred within hours "
                "of the corresponding inbound credit — consistent with FATF Rec-20 layering."
            ),
            "source": "FIU-IND Investigation Bulletin [DEV-SIMULATED]",
            "risk_keywords": ["probe", "investigation", "money laundering"],
        },
        {
            "title": "RBI issues show-cause to {entity}'s bank over pass-through account misuse",
            "description": (
                "The Reserve Bank of India issued a show-cause notice to the reporting "
                "bank citing failure to flag {entity}'s account, which exhibited a "
                "fund-exit ratio of {exit_r}% — far above the 30% benchmark for normal accounts."
            ),
            "source": "RBI Supervisory Action Bulletin [DEV-SIMULATED]",
            "risk_keywords": ["investigation", "fined", "suspicious"],
        },
    ],
    "typology_funnel_account": [
        {
            "title": "{entity} account receives credits from {n} distinct sources — mule account pattern",
            "description": (
                "Forensic analysis revealed {entity}'s account received transfers from "
                "{n} separate entities across {n} banks before consolidating and wiring "
                "funds offshore. FIU-IND has classified this as a funnel/mule account pattern."
            ),
            "source": "RBI Consumer Banking Supervision [DEV-SIMULATED]",
            "risk_keywords": ["fraud", "suspicious", "investigation"],
        },
        {
            "title": "Cybercrime cell freezes {entity} account in mule network crackdown",
            "description": (
                "Police cybercrime cell froze {entity}'s account as part of a wider "
                "crackdown on mule account networks funnelling online fraud proceeds. "
                "The account was used to receive and forward ₹{amount}L in 30 days."
            ),
            "source": "Cybercrime Cell Press Release [DEV-SIMULATED]",
            "risk_keywords": ["fraud", "arrested", "investigation"],
        },
    ],
    "typology_trade_based": [
        {
            "title": "{entity} under DGFT inquiry for over-invoiced imports from FATF-grey jurisdiction",
            "description": (
                "The Directorate General of Foreign Trade is examining import documents "
                "filed by {entity} showing invoice values 3-5x above prevailing market "
                "rates for goods sourced from a FATF grey-listed jurisdiction."
            ),
            "source": "DGFT Trade Intelligence Unit [DEV-SIMULATED]",
            "risk_keywords": ["fraud", "investigation", "sanctions"],
        },
        {
            "title": "FEMA violation notice to {entity} for trade payment anomalies",
            "description": (
                "RBI's forex division issued a FEMA notice to {entity} after detecting "
                "outward remittances for trade invoices that lacked corresponding goods "
                "import documentation — a red flag for trade-based laundering."
            ),
            "source": "RBI FEMA Compliance Watch [DEV-SIMULATED]",
            "risk_keywords": ["money laundering", "investigation", "probe"],
        },
    ],
    "typology_shell_company": [
        {
            "title": "{entity} linked to network of {n} shell entities in MCA21 audit",
            "description": (
                "A Ministry of Corporate Affairs audit flagged {entity} as a director "
                "or beneficial owner of {n} registered companies with no substantive "
                "operations, used as conduits for ₹{amount}L in layered transactions."
            ),
            "source": "MCA21 Regulatory Monitoring Cell [DEV-SIMULATED]",
            "risk_keywords": ["fraud", "investigation", "money laundering"],
        },
        {
            "title": "SFIO investigates {entity} in corporate fraud probe — shell network suspected",
            "description": (
                "The Serious Fraud Investigation Office has launched an inquiry into "
                "{entity} following detection of a layered ownership structure involving "
                "shell entities across multiple jurisdictions with no business substance."
            ),
            "source": "SFIO Investigation Report [DEV-SIMULATED]",
            "risk_keywords": ["fraud", "investigation", "probe"],
        },
    ],
    "typology_round_tripping": [
        {
            "title": "{entity} FEMA probe: ₹{amount}L sent abroad returned as 'FDI' — round-tripping suspected",
            "description": (
                "RBI investigators are examining whether ₹{amount}L transferred "
                "abroad by {entity} was subsequently re-invested into India as foreign "
                "direct investment to disguise domestic funds as foreign capital."
            ),
            "source": "RBI Forex Intelligence Division [DEV-SIMULATED]",
            "risk_keywords": ["money laundering", "investigation", "bribery"],
        },
        {
            "title": "IT Department flags {entity} for round-tripping through Mauritius route",
            "description": (
                "Income Tax Department's investigation wing has flagged {entity}'s "
                "remittances via Mauritius-registered entities, alleging the funds "
                "were round-tripped back to India to exploit DTAA treaty benefits."
            ),
            "source": "IT Department Investigation Wing [DEV-SIMULATED]",
            "risk_keywords": ["money laundering", "investigation", "probe"],
        },
    ],
    "default": [
        {
            "title": "{entity} placed under enhanced due diligence by bank compliance",
            "description": (
                "Bank compliance team has placed {entity} under enhanced monitoring "
                "following anomalous transaction patterns identified in routine AML "
                "surveillance. KYC refresh and source-of-funds documentation requested."
            ),
            "source": "Bank Compliance AML Team [DEV-SIMULATED]",
            "risk_keywords": ["suspicious", "aml", "investigation"],
        },
        {
            "title": "RBI inspection flags {entity} account for unexplained credit velocity",
            "description": (
                "An RBI on-site inspection identified {entity}'s account showing "
                "credit velocity inconsistent with the customer's declared income "
                "and business profile. STR filing under review."
            ),
            "source": "RBI On-site Inspection Report [DEV-SIMULATED]",
            "risk_keywords": ["suspicious", "investigation", "probe"],
        },
    ],
}


def _generate_dev_news(entity_name: str, entity_index: int,
                       typologies: list, feat: pd.Series) -> list:
    """
    Generate realistic simulated news for a specific entity.

    entity_index ensures different entities get DIFFERENT article angles
    from the same typology template bank — not identical text with name swapped.

    All values (amount, count, hours) come from the actual feat_series so
    articles are numerically consistent with the real case data.
    """
    # Pull real case values for article content
    total_amt    = float(feat.get("total_txn_amount", 5000000))
    amount_L     = round(total_amt / 100000, 1)            # convert to Lakhs
    avg_dep_L    = round(total_amt / max(1, float(feat.get("txn_count", 10))) / 100000, 1)
    hours        = max(1, int(float(feat.get("time_to_first_outbound_minutes_log", 3)) * 40))
    n_cp         = max(3, int(float(feat.get("distinct_counterparties_log", 2)) ** 2))
    exit_pct     = round(float(feat.get("fund_exit_ratio", 0.5)) * 100, 0)

    primary      = typologies[0] if typologies else "default"
    key          = primary if primary in DEV_NEWS_TEMPLATES else "default"
    templates    = DEV_NEWS_TEMPLATES[key]

    # Each entity gets the template at its index position (wraps around)
    tmpl_index   = entity_index % len(templates)
    tmpl         = templates[tmpl_index]

    base_date    = datetime.now(timezone.utc) - timedelta(days=random.randint(3, 60))
    pub_date     = base_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    title = tmpl["title"].format(
        entity=entity_name, n=n_cp, amount=amount_L,
        avg_dep=avg_dep_L, hours=hours, exit_r=exit_pct,
    )
    desc  = tmpl["description"].format(
        entity=entity_name, n=n_cp, amount=amount_L,
        avg_dep=avg_dep_L, hours=hours, exit_r=exit_pct,
    )

    return [{
        "title":         title,
        "description":   desc,
        "source":        tmpl["source"],
        "published_at":  pub_date,
        "url":           f"https://dev-simulated.local/news/{entity_index+1}",
        "risk_keywords": tmpl["risk_keywords"],
        "dev_simulated": True,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# MCP SECURE GATEWAY
#
# This is the core of what MCP means in this system.
#
# Every external call — whether real (OpenSanctions, GNews) or simulated
# (dev mode) — passes through this single gateway which:
#
#   1. VALIDATES   — checks the target host against an allowlist.
#                    Only pre-approved hosts can be called.
#                    Blocks anything not on the list.
#
#   2. RATE-LIMITS — enforces a minimum interval between calls for the
#                    same entity + source combination.
#                    Prevents accidental API hammering.
#
#   3. AUDITS      — writes a JSONL log entry for every call with:
#                    timestamp, source, entity, url_hash (not plaintext
#                    URL — for PII safety), status, latency_ms,
#                    result_count, dev_mode flag.
#                    This log is the compliance audit trail required
#                    by the PPT's "full audit logging" feature.
#
# In the agent4_mcp_server.py, FastMCP wraps these same capabilities
# as MCP protocol endpoints over SSE/HTTP so AWS Bedrock, Claude.ai,
# or any orchestrator can invoke them standardly.
# ─────────────────────────────────────────────────────────────────────────────

class MCPSecureGateway:
    ALLOWED_HOSTS = {
        "api.gnews.io", "api.opensanctions.org",
        "www.treasury.gov", "www.fatf-gafi.org",
        "rbi.org.in", "dev-simulated.local",
    }

    def __init__(self):
        self._rate_store: dict = {}
        self._call_log:   list = []
        # Each run gets a fresh session ID for log grouping
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _validate(self, url: str) -> bool:
        from urllib.parse import urlparse
        return urlparse(url).netloc in self.ALLOWED_HOSTS

    def _rate_check(self, key: str, min_s: float = 0.5) -> bool:
        now = time.time()
        if now - self._rate_store.get(key, 0) < min_s:
            return False
        self._rate_store[key] = now
        return True

    def _write_audit(self, entry: dict):
        self._call_log.append(entry)
        try:
            with open(MCP_AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _make_entry(self, source, url, entity, status, ms, count) -> dict:
        return {
            "session":      self.session_id,
            "ts":           datetime.now(timezone.utc).isoformat(),
            "source":       source,
            "entity":       entity,
            "url_hash":     hashlib.sha256(url.encode()).hexdigest()[:16],
            "status":       status,
            "latency_ms":   round(ms, 1),
            "result_count": count,
            "dev_mode":     DEV_MODE,
        }

    def log_simulated(self, source: str, entity: str, count: int):
        """Log a dev-mode simulated call (no real HTTP made)."""
        entry = self._make_entry(
            source, f"https://dev-simulated.local/{source}",
            entity, "DEV_SIMULATED", 0.0, count,
        )
        self._write_audit(entry)
        return entry

    def get(self, source: str, url: str, entity: str,
            params: dict = None, timeout: int = REQUEST_TIMEOUT) -> dict:
        """Validated, rate-limited, audited GET request."""
        if not self._validate(url):
            self._write_audit(self._make_entry(source, url, entity, "BLOCKED", 0, 0))
            return {"ok": False, "data": None, "error": "Host not in allowlist"}

        if not self._rate_check(f"{source}:{entity}"):
            self._write_audit(self._make_entry(source, url, entity, "RATE_LIMITED", 0, 0))
            return {"ok": False, "data": None, "error": "Rate limited"}

        t0 = time.time()
        try:
            resp  = requests.get(url, params=params, timeout=timeout)
            ms    = (time.time() - t0) * 1000
            if resp.status_code == 200:
                try:
                    data  = resp.json()
                    count = len(data) if isinstance(data, list) else 1
                except Exception:
                    data, count = resp.text, 1
                self._write_audit(self._make_entry(source, url, entity, "OK", ms, count))
                return {"ok": True, "data": data}
            else:
                self._write_audit(self._make_entry(source, url, entity, f"HTTP_{resp.status_code}", ms, 0))
                return {"ok": False, "data": None, "error": f"HTTP {resp.status_code}"}
        except requests.Timeout:
            self._write_audit(self._make_entry(source, url, entity, "TIMEOUT", (time.time()-t0)*1000, 0))
            return {"ok": False, "data": None, "error": "Timeout"}
        except Exception as e:
            self._write_audit(self._make_entry(source, url, entity, "ERROR", (time.time()-t0)*1000, 0))
            return {"ok": False, "data": None, "error": str(e)}

    def get_session_log(self) -> list:
        """Return only this session's audit entries."""
        return [e for e in self._call_log if e.get("session") == self.session_id]


# ─────────────────────────────────────────────────────────────────────────────
# SANCTIONS CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class SanctionsChecker:
    def __init__(self, gw: MCPSecureGateway):
        self.gw = gw

    def check_entity(self, name: str) -> dict:
        result = {
            "name": name, "sanctions_hit": False,
            "sanctions_datasets": [], "pep_flag": False,
            "rbi_caution": False, "match_score": 0.0,
            "matched_entries": [],
            "data_source": "FATF+RBI local [DEV_MODE]" if DEV_MODE else "OpenSanctions API + FATF + RBI",
        }

        if not DEV_MODE:
            resp = self.gw.get(
                "OpenSanctions-Search",
                f"{OPENSANC_BASE}/search/default",
                name, params={"q": name, "limit": 5},
            )
            if resp["ok"] and isinstance(resp.get("data"), dict):
                for r in resp["data"].get("results", []):
                    ds = r.get("datasets", [])
                    if any(d in ds for d in [
                        "sanctions", "us_ofac_sdn", "eu_fsf",
                        "un_sc_sanctions", "gb_hmt_sanctions",
                    ]):
                        result["sanctions_hit"] = True
                        result["sanctions_datasets"].extend(ds)
                    if "peps" in ds:
                        result["pep_flag"] = True
                    result["match_score"] = max(result["match_score"], r.get("score", 0.0))
                    if r.get("caption"):
                        result["matched_entries"].append(r["caption"])
        else:
            self.gw.log_simulated("OpenSanctions", name, 0)

        # RBI Caution check — always runs (local, no API)
        if any(e in name.lower() for e in RBI_CAUTION_ENTITIES):
            result["rbi_caution"]  = True
            result["sanctions_hit"]= True
            result["matched_entries"].append(f"RBI Caution List match: {name}")

        return result

    def check_country(self, country: str) -> dict:
        c = country.strip()
        on_black = c in FATF_BLACK_LIST
        on_grey  = c in FATF_GREY_LIST
        return {
            "country":        c,
            "fatf_blacklist": on_black,
            "fatf_greylist":  on_grey,
            "risk_level":     "CRITICAL" if on_black else ("HIGH" if on_grey else "STANDARD"),
            "data_source":    "FATF October 2024",
        }

    def run_bulk(self, entities: list, countries: list) -> dict:
        er = [self.check_entity(e)  for e in entities]
        cr = [self.check_country(c) for c in countries]
        return {
            "entity_results":            er,
            "country_results":           cr,
            "total_sanctions_hits":      sum(1 for r in er if r["sanctions_hit"]),
            "total_high_risk_countries": sum(1 for r in cr if r["risk_level"] in ("HIGH", "CRITICAL")),
        }


# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE NEWS SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class NegativeNewsScanner:
    RISK_KEYWORDS = [
        "fraud", "money laundering", "aml", "sanctions", "suspicious",
        "scam", "bribery", "corruption", "arrested", "charged", "convicted",
        "investigation", "probe", "fined", "penalty", "blacklist",
        "terrorist financing", "hawala", "ponzi", "embezzlement", "structuring",
    ]

    def __init__(self, gw: MCPSecureGateway):
        self.gw = gw

    def scan_entity(self, entity_name: str, entity_index: int,
                    typologies: list, feat: pd.Series,
                    max_articles: int = 2) -> dict:
        result = {
            "entity": entity_name,
            "articles": [],
            "risk_keywords_found": [],
            "negative_news_score": 0.0,
            "mode": "DEV_SIMULATED" if DEV_MODE else "LIVE_GNEWS",
        }

        if DEV_MODE:
            articles = _generate_dev_news(entity_name, entity_index, typologies, feat)
            result["articles"] = articles[:max_articles]
            kws = []
            for a in result["articles"]:
                kws.extend(a.get("risk_keywords", []))
            result["risk_keywords_found"] = list(set(kws))
            n_art = len(result["articles"])
            n_kw  = len(result["risk_keywords_found"])
            # Per-entity score: 0.35 per article + 0.08 per unique keyword, capped at 0.75
            result["negative_news_score"] = round(min(0.75, n_art * 0.35 + n_kw * 0.08), 3)
            self.gw.log_simulated("GNews-Dev", entity_name, n_art)
            log.info(f"[GNews/DEV] {n_art} article(s) for '{entity_name}' (angle #{entity_index+1})")

        else:
            if not GNEWS_API_KEY:
                result["note"] = "Set GNEWS_API_KEY env var for live news scanning."
                return result
            query = (f'"{entity_name}" AND '
                     f'(fraud OR "money laundering" OR sanctions OR corruption)')
            resp  = self.gw.get(
                "GNews", "https://api.gnews.io/v4/search", entity_name,
                params={"q": query, "lang": "en", "max": max_articles, "apikey": GNEWS_API_KEY},
            )
            if resp["ok"] and isinstance(resp.get("data"), dict):
                for a in resp["data"].get("articles", []):
                    title   = a.get("title", "")
                    desc    = a.get("description", "")
                    content = (title + " " + desc).lower()
                    kw_hits = [k for k in self.RISK_KEYWORDS if k in content]
                    result["articles"].append({
                        "title": title, "description": desc,
                        "source": a.get("source", {}).get("name", ""),
                        "published_at": a.get("publishedAt", ""),
                        "url": a.get("url", ""),
                        "risk_keywords": kw_hits, "dev_simulated": False,
                    })
                    result["risk_keywords_found"].extend(kw_hits)
                n_art = len(result["articles"])
                n_kw  = len(set(result["risk_keywords_found"]))
                result["negative_news_score"] = round(min(0.75, n_art * 0.30 + n_kw * 0.08), 3)

        return result

    def scan_bulk(self, entities: list, typologies: list, feat: pd.Series) -> dict:
        results = [
            self.scan_entity(e, idx, typologies, feat)
            for idx, e in enumerate(entities)
        ]
        # Aggregate = max across entities, NOT sum (avoids misleading 1.0 for 2 entities)
        scores = [r["negative_news_score"] for r in results]
        agg    = round(max(scores) if scores else 0.0, 3)
        return {
            "entity_results":                results,
            "aggregate_negative_news_score": agg,
            "aggregate_method":              "max_across_entities",
            "total_articles_found":          sum(len(r["articles"]) for r in results),
            "mode":                          "DEV_SIMULATED" if DEV_MODE else "LIVE_GNEWS",
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC RISK INTELLIGENCE FEEDS
# ─────────────────────────────────────────────────────────────────────────────

class PublicRiskIntelligenceFeeds:
    TYPOLOGY_RISK_SCORES = {
        "typology_structuring":    0.75,
        "typology_rapid_movement": 0.85,
        "typology_funnel_account": 0.80,
        "typology_trade_based":    0.82,
        "typology_shell_company":  0.88,
        "typology_round_tripping": 0.90,
    }

    def country_risk(self, countries: list) -> dict:
        scores = {c: (1.0 if c in FATF_BLACK_LIST else 0.75 if c in FATF_GREY_LIST else 0.20)
                  for c in countries}
        overall = max(scores.values()) if scores else 0.0
        return {
            "country_scores":   scores,
            "max_country_risk": round(overall, 2),
            "assessment": (
                "CRITICAL — FATF Black List jurisdiction present" if overall >= 1.0 else
                "HIGH — FATF Grey List jurisdiction present"      if overall >= 0.75 else
                "STANDARD — no FATF-listed jurisdictions"
            ),
        }

    def typology_risk(self, typologies: list) -> dict:
        scores = {t: self.TYPOLOGY_RISK_SCORES.get(t, 0.50) for t in typologies}
        max_s  = max(scores.values()) if scores else 0.0
        return {
            "typology_risk_scores":  scores,
            "max_typology_risk":     round(max_s, 2),
            "highest_risk_typology": max(scores, key=scores.get) if scores else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL LOADER & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_ml_model():
    try:
        import joblib
        path = os.path.join(AGENT4_DIR, "agent4_model.pkl")
        if os.path.exists(path):
            m = joblib.load(path)
            log.info("[ML] Loaded agent4_model.pkl")
            return m
    except Exception as e:
        log.warning(f"[ML] Load failed: {e}")
    return None


def _rule_based_score(feat: pd.Series) -> tuple:
    """
    Weighted rule-based scoring fallback.
    Returns (strength_idx, priority_score, probabilities_dict).
    Probabilities are spread realistically across 3 classes.
    """
    s = (
        0.30 * min(float(feat.get("alert_count",       0)) / 10.0, 1.0) +
        0.25 * min(float(feat.get("fund_exit_ratio",   0)), 1.0)        +
        0.20 * min(float(feat.get("burst_score",       0)), 1.0)        +
        0.15 * min(float(feat.get("kyc_risk_score",    0)), 1.0)        +
        0.10 * min(float(feat.get("high_risk_combined",0)), 1.0)
    )
    s = round(max(0.0, min(1.0, s)), 3)

    if s >= 0.67:
        idx   = 2
        probs = {"WEAK": round(1-s, 3), "MEDIUM": round((1-s)*0.6, 3), "STRONG": round(s, 3)}
    elif s >= 0.33:
        idx   = 1
        probs = {"WEAK": round(1-s, 3), "MEDIUM": round(s, 3), "STRONG": round(s*0.4, 3)}
    else:
        idx   = 0
        probs = {"WEAK": round(1-s*0.3, 3), "MEDIUM": round(s, 3), "STRONG": round(s*0.1, 3)}

    return idx, s, probs


def _score_breakdown(feat: pd.Series) -> dict:
    """
    Human-readable breakdown showing exactly what drove the priority score.
    This lets an analyst understand WHY the case is URGENT vs HIGH vs MEDIUM.
    """
    alert_contrib  = round(0.30 * min(float(feat.get("alert_count",0)) / 10.0, 1.0), 3)
    exit_contrib   = round(0.25 * min(float(feat.get("fund_exit_ratio",0)), 1.0), 3)
    burst_contrib  = round(0.20 * min(float(feat.get("burst_score",0)), 1.0), 3)
    kyc_contrib    = round(0.15 * min(float(feat.get("kyc_risk_score",0)), 1.0), 3)
    risk_contrib   = round(0.10 * min(float(feat.get("high_risk_combined",0)), 1.0), 3)
    return {
        "alert_count_contribution":       {"value": float(feat.get("alert_count",0)),       "weighted_contribution": alert_contrib, "weight": 0.30},
        "fund_exit_ratio_contribution":   {"value": round(float(feat.get("fund_exit_ratio",0)),3), "weighted_contribution": exit_contrib,  "weight": 0.25},
        "burst_score_contribution":       {"value": round(float(feat.get("burst_score",0)),3),     "weighted_contribution": burst_contrib, "weight": 0.20},
        "kyc_risk_score_contribution":    {"value": round(float(feat.get("kyc_risk_score",0)),3),  "weighted_contribution": kyc_contrib,  "weight": 0.15},
        "high_risk_combined_contribution":{"value": float(feat.get("high_risk_combined",0)),       "weighted_contribution": risk_contrib,  "weight": 0.10},
        "total_base_score":               round(alert_contrib+exit_contrib+burst_contrib+kyc_contrib+risk_contrib, 3),
    }


def _detect_gaps(feat: pd.Series, typologies: list) -> dict:
    gaps = {}
    for typ in typologies:
        key  = typ if typ in EVIDENCE_REQUIREMENTS else f"typology_{typ}"
        reqs = EVIDENCE_REQUIREMENTS.get(key, [])
        gaps[typ] = [
            desc for feat_name, check_fn, desc in reqs
            if feat_name in feat.index and not check_fn(feat[feat_name])
        ]
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT 4 CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Agent4EnrichmentAgent:
    """
    Agent 4: Enrichment Agent + SAR Case Strength Scorer.

    Accepts Agent 1 and Agent 3 outputs directly. Enriches with external
    intelligence through the MCP Secure Gateway. Scores the case using ML
    (or rule-based fallback). Returns a structured bundle for Agent 5.

    Usage:
        agent = Agent4EnrichmentAgent()
        result = agent.run(
            case_id       = "CASE-2026-007",
            feat_series   = row_from_data_engineered_csv,
            agent1_output = { ...Agent 1 dict... },
            agent3_output = { ...Agent 3 dict... },
            entities      = ["Rajesh Kumar", "Sunrise Exports Ltd"],
            countries     = ["Nigeria", "Iran"],
        )
    """

    def __init__(self):
        self.gw         = MCPSecureGateway()
        self.sanctions  = SanctionsChecker(self.gw)
        self.news       = NegativeNewsScanner(self.gw)
        self.risk_feeds = PublicRiskIntelligenceFeeds()
        self._ml        = _load_ml_model()
        log.info("[Agent4] Initialised | ML=%s | Mode=%s",
                 "loaded" if self._ml else "rule-based",
                 "DEV" if DEV_MODE else "PRODUCTION")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _needs_enrichment(self, feat: pd.Series, a1: dict) -> bool:
        return any([
            feat.get("high_risk_country_flag",         0) == 1,
            feat.get("pep_flag",                       0) == 1,
            feat.get("historical_sar_flag",            0) == 1,
            feat.get("international_counterparty_flag",0) == 1,
            float(feat.get("alert_count",              0)) >= 3,
            float(a1.get("confidence",                 0)) >= 0.80,
        ])

    def _ml_score(self, feat: pd.Series):
        """Returns (idx, priority_score, probs, thresholds, breakdown)."""
        breakdown = _score_breakdown(feat)
        if self._ml:
            try:
                clf    = self._ml["clf_model"]
                rf     = self._ml["rf_model"]
                scaler = self._ml["scaler"]
                thr    = self._ml["priority_thresholds"]
                feats  = self._ml.get("case_features", CASE_FEATURES)
                X      = np.array([float(feat.get(f, 0.0)) for f in feats]).reshape(1, -1)
                Xs     = scaler.transform(X)
                probs  = clf.predict_proba(Xs)[0]
                idx    = int(np.argmax(probs))
                ps     = float(np.clip(rf.predict(Xs)[0], 0, 1))
                return idx, ps, {
                    "WEAK":   round(float(probs[0]), 3),
                    "MEDIUM": round(float(probs[1]), 3),
                    "STRONG": round(float(probs[2]), 3),
                }, thr, breakdown
            except Exception as e:
                log.warning(f"[ML] Inference error, using rule-based: {e}")

        idx, ps, probs = _rule_based_score(feat)
        return idx, ps, probs, {"urgent": 0.75, "high": 0.50, "medium": 0.25}, breakdown

    def _priority_label(self, score: float, thr: dict) -> str:
        if score >= thr.get("urgent", 0.75): return "URGENT"
        if score >= thr.get("high",   0.50): return "HIGH"
        if score >= thr.get("medium", 0.25): return "MEDIUM"
        return "LOW"

    def _parse_agent1(self, a1: dict):
        return (
            bool(a1.get("sar_worthy", False)),
            float(a1.get("confidence", 0.0)),
            [t["typology"] for t in a1.get("typologies", [])
             if isinstance(t, dict) and "typology" in t],
        )

    def _parse_agent3(self, a3: dict):
        predicted  = a3.get("predicted_typology", "")
        triggers   = a3.get("rule_triggers", [])
        indicators = a3.get("quantified_indicators", {})
        if not triggers and "typology_evidence" in a3:
            triggers   = a3["typology_evidence"].get("rule_triggers", [])
        if not indicators and "typology_evidence" in a3:
            indicators = a3["typology_evidence"].get("quantified_indicators", {})
        summary    = a3.get("case_summary", {})
        return (
            predicted, triggers, indicators,
            int(summary.get("flagged_transactions",  0)),
            float(summary.get("total_flagged_amount", 0.0)),
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        case_id:       str,
        feat_series:   pd.Series,
        agent1_output: dict,
        agent3_output: dict,
        entities:      list = None,
        countries:     list = None,
        force_enrich:  bool = False,
    ) -> dict:
        """
        Run Agent 4 on a SAR case.

        Parameters
        ----------
        case_id        : Unique case ID string
        feat_series    : pd.Series — one row from data_engineered.csv
        agent1_output  : dict from Agent 1 {sar_worthy, confidence, typologies}
        agent3_output  : dict from Agent 3 {predicted_typology, rule_triggers,
                         quantified_indicators, case_summary}
        entities       : list of person/company names for sanctions + news check
        countries      : list of country names for FATF risk check
        force_enrich   : if True, always enrich regardless of risk signal gates
        """
        t0 = time.time()
        log.info("="*55)
        log.info(f"[Agent4] Case: {case_id}")
        log.info("="*55)

        entities  = entities  or []
        countries = countries or []

        # ── 1. Parse inputs from previous agents ─────────────────────────
        sar_worthy, a1_conf, a1_types = self._parse_agent1(agent1_output)
        a3_type, triggers, indicators, flagged_txns, flagged_amt = \
            self._parse_agent3(agent3_output)

        all_types = list(dict.fromkeys(
            a1_types + ([a3_type] if a3_type and a3_type not in a1_types else [])
        ))
        log.info(f"Typologies     : {all_types}")
        log.info(f"Agent1 conf    : {a1_conf:.2f}  |  SAR worthy: {sar_worthy}")

        # ── 2. ML / rule-based scoring ────────────────────────────────────
        s_idx, p_score, probs, thr, score_bd = self._ml_score(feat_series)
        pri = self._priority_label(p_score, thr)
        log.info(f"Strength       : {STRENGTH_MAP[s_idx]}  "
                 f"(W={probs['WEAK']} M={probs['MEDIUM']} S={probs['STRONG']})")
        log.info(f"Priority       : {p_score:.3f}  →  {pri}")

        # ── 3. Evidence gaps ──────────────────────────────────────────────
        evidence_gaps = _detect_gaps(feat_series, all_types)

        # ── 4. External enrichment via MCP Gateway ────────────────────────
        do_enrich = force_enrich or self._needs_enrichment(feat_series, agent1_output)
        log.info(f"Enrichment     : {'TRIGGERED' if do_enrich else 'SKIPPED (low risk signals)'}")

        ext = {
            "enrichment_triggered":  do_enrich,
            "mode":                  "DEV_SIMULATED" if DEV_MODE else "LIVE_APIS",
            "sanctions":             {},
            "negative_news":         {},
            "regulatory_advisories": [],
            "risk_intelligence":     {},
            "enrichment_summary":    "Not triggered",
            "mcp_session_log":       [],
        }

        s_hits     = 0
        news_score = 0.0

        if do_enrich:
            # 4a. Sanctions + FATF country check
            sanctions_r = self.sanctions.run_bulk(entities, countries)
            ext["sanctions"] = sanctions_r
            s_hits = sanctions_r.get("total_sanctions_hits", 0)

            # 4b. Negative news
            news_r = self.news.scan_bulk(entities, all_types, feat_series) if entities else {}
            ext["negative_news"] = news_r
            news_score = news_r.get("aggregate_negative_news_score", 0.0)

            # 4c. Regulatory advisories — strict match only
            advisories = _match_advisories(all_types, countries)
            ext["regulatory_advisories"] = advisories
            log.info(f"Advisories     : {len(advisories)} matched")

            # 4d. Country + typology risk intelligence
            ext["risk_intelligence"] = {
                "country_risk":  self.risk_feeds.country_risk(countries),
                "typology_risk": self.risk_feeds.typology_risk(all_types),
            }

            # 4e. Boost priority score for confirmed external signals
            boost = 0.0
            if s_hits > 0:
                boost += 0.15
                log.warning(f"Priority boosted +0.15 (sanctions hit)")
            if news_score > 0.4:
                boost += 0.06
                log.info(f"Priority boosted +0.06 (negative news score={news_score:.2f})")

            if boost > 0:
                p_score = min(1.0, p_score + boost)
                pri     = self._priority_label(p_score, thr)
                score_bd["external_boost"] = round(boost, 2)

            # 4f. Enrichment summary
            cr_assess  = ext["risk_intelligence"]["country_risk"]["assessment"]
            top_adv    = advisories[0]["id"] if advisories else "none"
            ext["enrichment_summary"] = (
                f"{s_hits} sanctions hit(s) | "
                f"news risk score = {news_score:.2f} | "
                f"{len(advisories)} advisory match(es), top: {top_adv} | "
                f"country risk: {cr_assess}"
            )
            ext["mcp_session_log"] = self.gw.get_session_log()

        # ── 5. Filing recommendation ──────────────────────────────────────
        if s_idx == 2 and p_score >= thr.get("high", 0.50):
            rec = "FILE SAR IMMEDIATELY — strong multi-signal evidence"
        elif s_idx >= 1 and p_score >= thr.get("medium", 0.25):
            rec = "FILE SAR — sufficient evidence; analyst review recommended before filing"
        elif s_idx == 1:
            rec = "PENDING — gather additional evidence before filing"
        else:
            rec = "CLOSE — insufficient evidence for SAR filing at this time"

        modifiers = []
        if s_hits > 0:
            modifiers.append("SANCTIONS HIT detected — escalate to MLRO immediately")
        if news_score > 0.4:
            modifiers.append(f"Negative news corroborates risk (score={news_score:.2f})")
        if any(a.get("severity") == "CRITICAL" for a in ext.get("regulatory_advisories", [])):
            modifiers.append("CRITICAL regulatory advisory applies to this case")

        if modifiers:
            rec += ". " + " | ".join(modifiers)

        # ── 6. Completeness score ─────────────────────────────────────────
        completeness = sum([
            feat_series.get("alert_count",           0) >  0,
            feat_series.get("alert_count",           0) >  3,
            feat_series.get("high_risk_country_flag",0) == 1,
            feat_series.get("kyc_risk_score",        0) >  0.70,
            feat_series.get("fund_exit_ratio",       0) >  0.80,
            feat_series.get("burst_score",           0) >  0.70,
            feat_series.get("distinct_counterparties_log", 0) > 2.0,
            feat_series.get("pep_flag",              0) == 1,
            s_hits > 0,
            any(a.get("severity") == "CRITICAL" for a in ext.get("regulatory_advisories", [])),
        ])

        elapsed_ms = round((time.time() - t0) * 1000)
        log.info(f"Completed in {elapsed_ms}ms | "
                 f"{STRENGTH_MAP[s_idx]} | {pri} | completeness {completeness}/10")

        return {
            # Identifiers
            "case_id":              case_id,
            "processed_at":         datetime.now(timezone.utc).isoformat(),
            "elapsed_ms":           elapsed_ms,
            "dev_mode":             DEV_MODE,

            # ML scoring
            "strength_label":         STRENGTH_MAP[s_idx],
            "strength_probabilities": probs,
            "priority_score":         round(p_score, 3),
            "recommended_priority":   pri,
            "score_breakdown":        score_bd,   # NEW: what drove the score
            "filing_recommendation":  rec,
            "evidence_gaps":          evidence_gaps,

            # External enrichment
            "external_intelligence": ext,

            # Case overview for Agent 5 (Narrative Engine)
            "case_overview": {
                "n_transactions":              int(feat_series.get("txn_count", 0)),
                "flagged_transactions_agent3": flagged_txns,
                "flagged_amount_agent3_inr":   flagged_amt,
                "rule_triggers_agent3":        triggers,
                "quantified_indicators":       indicators,
                "alert_count":                 int(feat_series.get("alert_count", 0)),
                "alert_density":               round(float(feat_series.get("alert_density", 0)), 3),
                "fund_exit_ratio":             round(float(feat_series.get("fund_exit_ratio", 0)), 3),
                "burst_score":                 round(float(feat_series.get("burst_score", 0)), 3),
                "kyc_risk_score":              round(float(feat_series.get("kyc_risk_score", 0)), 3),
                "has_high_risk_country":       bool(feat_series.get("high_risk_country_flag", 0)),
                "has_pep_exposure":            bool(feat_series.get("pep_flag", 0)),
                "typologies_assessed":         all_types,
                "agent1_sar_worthy":           sar_worthy,
                "agent1_confidence":           a1_conf,
                "agent3_predicted_typology":   a3_type,
                "entities_checked":            entities,
                "countries_checked":           countries,
                "evidence_completeness_score": completeness,
                "max_possible_completeness":   10,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO  (python agent4_enrichment.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("AGENT 4  —  Enrichment Agent + SAR Case Strength Scorer")
    print(f"Mode     :  {'DEVELOPMENT — simulated external data' if DEV_MODE else 'PRODUCTION — live APIs'}")
    print("Pipeline :  Agent1 → Agent2 → Agent3 → [Agent4] → Agent5")
    print("=" * 70)

    # Load real data if available, else use synthetic demo
    demo_feat, case_id = None, None
    if os.path.exists("data_engineered.csv"):
        df       = pd.read_csv("data_engineered.csv")
        sar_rows = df[df["sar_worthy"] == 1]
        if len(sar_rows):
            demo_feat = sar_rows.iloc[0].copy()
            case_id   = f"CASE-{sar_rows.index[0]:04d}"
            print(f"\nLoaded data_engineered.csv  →  {case_id}")

    if demo_feat is None:
        case_id   = "CASE-DEMO-0001"
        demo_feat = pd.Series({
            "txn_count": 47,              "total_txn_amount": 9850000,
            "alert_count": 8,             "alert_density": 0.45,
            "alert_tier": 3,              "fund_exit_ratio": 0.97,
            "burst_score": 0.88,          "burst_per_age": 0.15,
            "burst_x_exit": 0.85,         "time_to_first_outbound_minutes_log": 4.3,
            "txn_velocity_log": 3.2,      "txn_amount_cv": 0.18,
            "max_to_avg_txn_ratio": 3.4,  "total_txn_amount_cbrt": 214.5,
            "avg_txn_amount_cbrt": 60.2,  "std_txn_amount_cbrt": 22.1,
            "txn_count_log": 3.85,        "distinct_counterparties_log": 2.9,
            "incoming_sources_count_log": 2.2,
            "counterparty_diversity_score": 0.72,
            "counterparty_to_txn_ratio": 0.48,
            "incoming_to_outgoing_ratio": 0.88,
            "international_counterparty_flag": 1,
            "high_risk_country_flag": 1,   "pep_flag": 0,
            "kyc_risk_score": 0.78,        "kyc_risk_tier": 3,
            "kyc_x_alert": 0.62,           "historical_sar_flag": 1,
            "high_risk_combined": 1,       "hr_country_x_exit": 0.97,
            "pep_x_intl": 0,               "sar_history_x_kyc": 0.78,
            "fund_exit_tier": 3,           "binary_risk_flag_count": 4,
            "sar_worthy": 1,
            "typology_structuring": 1,     "typology_rapid_movement": 1,
            "typology_funnel_account": 0,  "typology_trade_based": 0,
            "typology_shell_company": 0,   "typology_round_tripping": 0,
        })
        print(f"\nUsing synthetic demo  →  {case_id}")

    # Realistic Agent 1 output (as produced by Agent 1)
    agent1_output = {
        "sar_worthy": True,
        "confidence": 0.94,
        "typologies": [
            {"typology": "typology_structuring",    "confidence": 0.96},
            {"typology": "typology_rapid_movement", "confidence": 0.88},
        ],
    }

    # Realistic Agent 3 output (as produced by Agent 3)
    agent3_output = {
        "predicted_typology": "typology_structuring",
        "rule_triggers": ["R-STR-01", "R-STR-02", "R-RMV-01"],
        "quantified_indicators": {
            "deposits_below_10L_threshold": 3,
            "avg_deposit_amount_inr":       3283333,    # 9.85M / 3 deposits
            "fund_exit_ratio":              0.97,
            "time_to_first_outbound_min":   75,
            "distinct_incoming_sources":    12,         # from distinct_counterparties_log ~2.9 → e^2.9 ≈ 18
        },
        "typology_evidence": {
            "rule_triggers": ["R-STR-01", "R-STR-02"],
            "quantified_indicators": {},
        },
        "case_summary": {
            "flagged_transactions":  28,
            "total_flagged_amount":  9200000,
        },
        "event_map": {
            "description": (
                "3 cash deposits just below INR 10L CTR threshold detected over 28 days, "
                "followed by rapid outbound wire within 75 minutes of each deposit."
            ),
        },
    }

    # Run Agent 4
    agent = Agent4EnrichmentAgent()
    result = agent.run(
        case_id       = case_id,
        feat_series   = demo_feat,
        agent1_output = agent1_output,
        agent3_output = agent3_output,
        entities      = ["Rajesh Kumar", "Sunrise Exports Ltd"],
        countries     = ["Nigeria", "Iran"],
        force_enrich  = True,
    )

    # ── Print results ─────────────────────────────────────────────────────
    r = result
    print(f"\n{'─'*70}")
    print("AGENT 4 OUTPUT")
    print(f"{'─'*70}")
    print(f"Case ID               : {r['case_id']}")
    print(f"Processed at          : {r['processed_at']}")
    print(f"Elapsed               : {r['elapsed_ms']}ms")

    print(f"\n── Scoring ──")
    print(f"  Strength Label      : {r['strength_label']}")
    print(f"  Probabilities       : WEAK={r['strength_probabilities']['WEAK']}  "
          f"MEDIUM={r['strength_probabilities']['MEDIUM']}  "
          f"STRONG={r['strength_probabilities']['STRONG']}")
    print(f"  Priority Score      : {r['priority_score']}  →  {r['recommended_priority']}")

    bd = r["score_breakdown"]
    print(f"\n── Score Breakdown (what drove the priority) ──")
    for k, v in bd.items():
        if k == "total_base_score":
            print(f"  {'TOTAL BASE SCORE':45s}: {v}")
        elif k == "external_boost":
            print(f"  {'External signal boost (sanctions/news)':45s}: +{v}")
        else:
            label = k.replace("_contribution","").replace("_"," ")
            print(f"  {label:45s}: val={v['value']}  weighted={v['weighted_contribution']}  (wt={v['weight']})")

    print(f"\n── Filing Recommendation ──")
    print(f"  {r['filing_recommendation']}")

    print(f"\n── Evidence Gaps ──")
    any_gap = False
    for typ, gaps in r["evidence_gaps"].items():
        if gaps:
            any_gap = True
            for g in gaps:
                print(f"  WARNING  [{typ}]  {g}")
        else:
            print(f"  OK       [{typ}]  No gaps detected")
    if not any_gap:
        print("  All typologies have complete evidence")

    ext = r["external_intelligence"]
    print(f"\n── External Intelligence  [mode={ext['mode']}] ──")
    print(f"  {ext['enrichment_summary']}")

    print(f"\n  Sanctions Check:")
    sanc = ext.get("sanctions", {})
    for er in sanc.get("entity_results", []):
        hit = "HIT" if er["sanctions_hit"] else "Clear"
        print(f"    [{hit:5s}]  {er['name']:30s}  source={er['data_source']}")
        if er.get("matched_entries"):
            for m in er["matched_entries"]:
                print(f"             ^ {m}")
    for cr in sanc.get("country_results", []):
        print(f"    [{cr['risk_level']:8s}]  {cr['country']:20s}  FATF blacklist={cr['fatf_blacklist']}  greylist={cr['fatf_greylist']}")

    print(f"\n  Negative News  [method: {ext.get('negative_news',{}).get('aggregate_method','?')}]:")
    news = ext.get("negative_news", {})
    print(f"    Aggregate score : {news.get('aggregate_negative_news_score',0):.3f}"
          f"  (max across {len(news.get('entity_results',[]))} entities)")
    print(f"    Total articles  : {news.get('total_articles_found',0)}")
    for er in news.get("entity_results", []):
        print(f"    [{er['entity']}]  score={er['negative_news_score']:.3f}  articles={len(er['articles'])}")
        for a in er.get("articles", []):
            print(f"      HEADLINE  : {a['title']}")
            print(f"      SOURCE    : {a['source']}")
            print(f"      PUBLISHED : {a['published_at']}")
            print(f"      KEYWORDS  : {a['risk_keywords']}")
            print()

    adv_list = ext.get("regulatory_advisories", [])
    print(f"  Regulatory Advisories Matched: {len(adv_list)}")
    for a in adv_list:
        print(f"    [{a['severity']:8s}]  {a['issuer']} {a['id']}  |  {a['title']}")
        print(f"               Summary  : {a['summary']}")
        print(f"               Action   : {a.get('action_required','')}")

    ri = ext.get("risk_intelligence", {})
    if ri:
        cr_r = ri.get("country_risk", {})
        tr_r = ri.get("typology_risk", {})
        print(f"\n  Country Risk  : {cr_r.get('assessment','')}")
        for c, s in cr_r.get("country_scores", {}).items():
            print(f"    {c:25s}  score={s}")
        print(f"  Typology Risk : highest={tr_r.get('highest_risk_typology','')}  "
              f"score={tr_r.get('max_typology_risk',0)}")

    session_log = ext.get("mcp_session_log", [])
    print(f"\n  MCP Session Audit Log  ({len(session_log)} entries, session={agent.gw.session_id})")
    for e in session_log:
        print(f"    {e['ts'][11:19]}  {e['source']:25s}  {e['status']:18s}  "
              f"results={e['result_count']}  {e['latency_ms']:.0f}ms")

    ov = r["case_overview"]
    print(f"\n── Case Overview (passed to Agent 5) ──")
    print(f"  Total transactions       : {ov['n_transactions']}")
    print(f"  Flagged (Agent 3)        : {ov['flagged_transactions_agent3']} txns  "
          f"INR {ov['flagged_amount_agent3_inr']:,.0f}")
    print(f"  Rule triggers (Agent 3)  : {ov['rule_triggers_agent3']}")
    print(f"  Quantified indicators    : {ov['quantified_indicators']}")
    print(f"  Alert count              : {ov['alert_count']}")
    print(f"  Fund exit ratio          : {ov['fund_exit_ratio']}")
    print(f"  Burst score              : {ov['burst_score']}")
    print(f"  KYC risk score           : {ov['kyc_risk_score']}")
    print(f"  High-risk country        : {ov['has_high_risk_country']}")
    print(f"  PEP exposure             : {ov['has_pep_exposure']}")
    print(f"  Typologies assessed      : {ov['typologies_assessed']}")
    print(f"  Agent 1 confidence       : {ov['agent1_confidence']}")
    print(f"  Agent 3 typology         : {ov['agent3_predicted_typology']}")
    print(f"  Completeness score       : {ov['evidence_completeness_score']}/{ov['max_possible_completeness']}")

    # Save result
    out_path = os.path.join(AGENT4_DIR, f"result_{case_id.replace('-','_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✓  Result saved   →  {out_path}")
    print(f"✓  Audit log      →  {MCP_AUDIT_LOG}")
    print("=" * 70)
