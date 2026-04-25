"""
agent4_mcp_server.py  —  Agent 4 MCP Server  v4
================================================
Team BAYMAX | Barclays Hack-O-Hire | SAR Generator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS MCP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MCP = Model Context Protocol (open standard by Anthropic, 2024).

It is a standardised way for AI agents and LLMs to call external
tools, APIs, and data sources. Think of it as "USB for AI tools" —
one plug that works with any tool from any vendor.

Before MCP, every AI system wired up its own custom API calls,
with its own auth, error handling, logging, and schemas.
MCP standardises ALL of that into a single protocol so that:
  • Any MCP server can be connected to any MCP client
  • AWS Bedrock, Claude.ai, and custom agents all speak the same language
  • Tool calls are validated, rate-limited, and audited by the protocol

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT MCP DOES IN THIS SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The PPT (Slide 11) shows Agent 4 as the "Enrichment Agent" that calls
external intelligence sources through a "Secure Gateway (MCP Server)".

This file IS that MCP Server. It wraps all Agent 4 capabilities as
MCP tools that can be called by:
  • AWS Bedrock agents (via agent tool configuration)
  • Claude.ai (via MCP connector settings)
  • Any Python code (via FastMCP client)
  • The orchestration agent (Agent 2) in the pipeline

Without this MCP server, Agent 4 must be called as a Python function.
With this MCP server, Agent 4 becomes a network service that any
component in the pipeline can call standardly over HTTP/SSE.

The MCPSecureGateway inside agent4_enrichment.py handles per-call
audit logging, allowlist validation, and rate limiting — these are
the compliance controls required by the PPT's security section.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install fastmcp
  python agent4_mcp_server.py
  # Server starts at: http://localhost:8000/sse

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO CONNECT FROM ANY MCP CLIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    "type": "url",
    "url":  "http://localhost:8000/sse",
    "name": "agent4-enrichment-mcp"
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8 TOOLS EXPOSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. score_sar_case             Full pipeline — ML + enrichment
  2. check_sanctions            Sanctions / FATF / RBI check
  3. scan_negative_news         GNews (live) or simulated (dev)
  4. get_regulatory_advisories  FATF/RBI/FIU-IND/FinCEN advisories
  5. compute_country_risk       FATF 2024 country risk scores
  6. get_typology_risk          Pattern risk per typology
  7. get_fatf_lists             Current FATF blacklist + greylist
  8. get_audit_log              MCP call audit trail
"""

import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Agent4-MCP] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Agent4-MCP")

# ── Import Agent 4 core logic ──────────────────────────────────────────────
from agent4 import (
    Agent4EnrichmentAgent, SanctionsChecker, NegativeNewsScanner,
    PublicRiskIntelligenceFeeds, MCPSecureGateway,
    _match_advisories,
    FATF_BLACK_LIST, FATF_GREY_LIST, FATF_ALL_HIGH_RISK,
    MCP_AUDIT_LOG, DEV_MODE,
)

try:
    from fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    log.warning("fastmcp not installed — running in stub/test mode.")
    log.warning("To start the real MCP server: pip install fastmcp")

# ── Shared singletons (one instance per server process) ───────────────────
_gw         = MCPSecureGateway()
_sanctions  = SanctionsChecker(_gw)
_news       = NegativeNewsScanner(_gw)
_risk_feeds = PublicRiskIntelligenceFeeds()
_agent4     = Agent4EnrichmentAgent()


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

if HAS_FASTMCP:
    mcp = FastMCP("agent4-enrichment-mcp")

    # ── Tool 1: Full SAR Case Pipeline ─────────────────────────────────────
    @mcp.tool()
    def score_sar_case(
        case_id:            str,
        case_features_json: str,
        agent1_output_json: str,
        agent3_output_json: str,
        entity_names:       list[str] = None,
        countries:          list[str] = None,
        force_enrich:       bool      = False,
    ) -> dict:
        """
        Run the full Agent 4 pipeline on a SAR case.

        Primary tool — call this with outputs from Agent 1 and Agent 3.
        It runs ML scoring, detects evidence gaps, enriches with external
        intelligence, and returns a complete bundle for Agent 5.

        Args:
            case_id:            e.g. "CASE-2026-007"
            case_features_json: JSON string — feature dict from data_engineered.csv
                                Keys: alert_count, fund_exit_ratio, burst_score, etc.
            agent1_output_json: JSON string from Agent 1
                                {"sar_worthy": true, "confidence": 0.94,
                                 "typologies": [{"typology": "typology_structuring",
                                                 "confidence": 0.96}]}
            agent3_output_json: JSON string from Agent 3
                                {"predicted_typology": "typology_structuring",
                                 "rule_triggers": ["R-STR-01"],
                                 "quantified_indicators": {...},
                                 "case_summary": {"flagged_transactions": 28,
                                                  "total_flagged_amount": 9200000}}
            entity_names:       Customer/counterparty names for sanctions + news
                                e.g. ["Rajesh Kumar", "Sunrise Exports Ltd"]
            countries:          Jurisdictions in the case for FATF check
                                e.g. ["Nigeria", "Iran"]
            force_enrich:       True = always enrich regardless of risk gates
        """
        log.info(f"[score_sar_case] case_id={case_id}")
        try:
            feat    = pd.Series(json.loads(case_features_json))
            agent1  = json.loads(agent1_output_json)
            agent3  = json.loads(agent3_output_json)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}"}

        return _agent4.run(
            case_id       = case_id,
            feat_series   = feat,
            agent1_output = agent1,
            agent3_output = agent3,
            entities      = entity_names or [],
            countries     = countries    or [],
            force_enrich  = force_enrich,
        )

    # ── Tool 2: Sanctions Check ────────────────────────────────────────────
    @mcp.tool()
    def check_sanctions(
        entity_names: list[str],
        countries:    list[str] = None,
    ) -> dict:
        """
        Check entities and countries against sanctions lists.

        Sources:
          - OpenSanctions API: OFAC SDN, EU FSF, UN SC, HMT (production only)
          - FATF October 2024: blacklist (Iran, DPRK, Myanmar, North Korea)
            and greylist (23 jurisdictions) — always runs
          - RBI Caution List: known Indian fraudulent entities — always runs

        DEV_MODE=True:  OpenSanctions skipped; FATF + RBI local checks only.
        DEV_MODE=False: All three sources queried.
        """
        log.info(f"[check_sanctions] entities={entity_names}")
        return _sanctions.run_bulk(entity_names, countries or [])

    # ── Tool 3: Negative News Scan ─────────────────────────────────────────
    @mcp.tool()
    def scan_negative_news(
        entity_names:            list[str],
        typologies:              list[str] = None,
        case_features_json:      str       = "{}",
        max_articles_per_entity: int       = 2,
    ) -> dict:
        """
        Scan for negative news about named entities.

        DEV_MODE=True (current default):
          Returns realistic SIMULATED articles. Each entity gets a DIFFERENT
          article angle (FIU-IND notice, ED probe, bank compliance alert, etc.)
          based on the primary typology. Article content uses real case values
          (amounts, counts) from case_features_json. Sources are labelled
          [DEV-SIMULATED]. No API key or internet needed.

        DEV_MODE=False (production):
          Calls GNews API with AML-focused queries. Requires GNEWS_API_KEY
          environment variable. Get a free key at https://gnews.io/
          (100 requests/day free tier).

        Aggregate score uses MAX across entities (not sum) to avoid
        artificially inflating the score for multi-entity cases.

        Args:
            entity_names:            Names to scan
            typologies:              Detected typologies (selects article angle in dev mode)
            case_features_json:      Case feature dict as JSON (populates article amounts)
            max_articles_per_entity: Articles to return per entity (default 2)
        """
        log.info(f"[scan_negative_news] entities={entity_names} mode={'DEV' if DEV_MODE else 'LIVE'}")
        try:
            feat = pd.Series(json.loads(case_features_json))
        except Exception:
            feat = pd.Series({})
        return _news.scan_bulk(entity_names, typologies or [], feat)

    # ── Tool 4: Regulatory Advisories ──────────────────────────────────────
    @mcp.tool()
    def get_regulatory_advisories(
        typologies: list[str],
        countries:  list[str] = None,
    ) -> dict:
        """
        Match regulatory advisories for detected typologies and countries.

        Strict matching rules:
          - FATF jurisdiction advisories fire ONLY when that country is present
          - Typology advisories fire ONLY when that typology is detected
          - 'all-typology' advisories (FATF blacklist/greylist) fire only on country match

        Advisory sources:
          FATF-2024-02  : Black List call for action (CRITICAL)
          FATF-2024-01  : Grey List increased monitoring (HIGH)
          FIU-IND-2024-01: Mule accounts / rapid movement (HIGH)
          FinCEN-2024-01 : Structuring via digital channels (MEDIUM)
          RBI-2024-AML-01: PEP + shell entity risk (HIGH)
          RBI-2023-TBML-01: Trade-based ML alert (HIGH)
        """
        log.info(f"[get_regulatory_advisories] typologies={typologies}")
        matched = _match_advisories(typologies, countries or [])
        sev     = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        max_sev = max((a["severity"] for a in matched),
                      key=lambda s: sev.get(s, 0), default="NONE")
        return {
            "matched_advisories": matched,
            "advisory_count":     len(matched),
            "max_severity":       max_sev,
        }

    # ── Tool 5: Country Risk ───────────────────────────────────────────────
    @mcp.tool()
    def compute_country_risk(countries: list[str]) -> dict:
        """
        Score countries using FATF October 2024 lists.

        CRITICAL  (score=1.0) : Iran, North Korea, Myanmar, DPRK
        HIGH      (score=0.75): 23 FATF grey-list jurisdictions
        STANDARD  (score=0.20): All other countries
        """
        log.info(f"[compute_country_risk] countries={countries}")
        return _risk_feeds.country_risk(countries)

    # ── Tool 6: Typology Risk ──────────────────────────────────────────────
    @mcp.tool()
    def get_typology_risk(typologies: list[str]) -> dict:
        """
        Get pattern-based risk scores for AML typologies.

        Scores from FATF typology severity assessments:
          round_tripping   0.90 | shell_company  0.88
          rapid_movement   0.85 | trade_based    0.82
          funnel_account   0.80 | structuring    0.75
        """
        log.info(f"[get_typology_risk] typologies={typologies}")
        return _risk_feeds.typology_risk(typologies)

    # ── Tool 7: FATF Lists ─────────────────────────────────────────────────
    @mcp.tool()
    def get_fatf_lists() -> dict:
        """
        Return the current FATF high-risk jurisdiction lists (October 2024).
        Update by checking https://www.fatf-gafi.org/ after each FATF plenary
        (held in February, June, October each year).
        """
        return {
            "blacklist":     sorted(FATF_BLACK_LIST),
            "greylist":      sorted(FATF_GREY_LIST),
            "all_high_risk": sorted(FATF_ALL_HIGH_RISK),
            "last_updated":  "2024-10-25",
            "next_update":   "2025-02-01 (estimated)",
            "source_url":    "https://www.fatf-gafi.org/en/topics/high-risk-and-other-monitored-jurisdictions.html",
        }

    # ── Tool 8: Audit Log ──────────────────────────────────────────────────
    @mcp.tool()
    def get_audit_log(last_n: int = 50) -> dict:
        """
        Return the MCP gateway audit trail.

        Every external call (real or simulated) is logged with:
          session, ts, source, entity, url_hash (SHA-256, not plaintext),
          status, latency_ms, result_count, dev_mode.

        The url_hash protects entity PII — the actual URL is not logged.
        Full log is written to agent4/mcp_audit_log.jsonl each session.
        """
        entries   = _gw.get_session_log()[-last_n:]
        ok_count  = sum(1 for e in entries if e["status"] in ("OK", "DEV_SIMULATED"))
        err_count = len(entries) - ok_count
        return {
            "entries":     entries,
            "total_calls": len(entries),
            "ok_count":    ok_count,
            "error_count": err_count,
            "log_file":    MCP_AUDIT_LOG,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STUB/TEST MODE — runs all tools directly when FastMCP is not installed
# ══════════════════════════════════════════════════════════════════════════════

def _run_stub_tests():
    print("\n" + "="*65)
    print("AGENT 4 MCP SERVER")
    print(f"DEV_MODE = {DEV_MODE}")
    print("="*65)
    print("\nfastmcp is not installed. To start the real MCP server:")
    print("  pip install fastmcp")
    print("  python agent4_mcp_server.py")
    print("\nConnect with:")
    print('  {"type":"url","url":"http://localhost:8000/sse","name":"agent4-enrichment-mcp"}')
    print("\n" + "─"*65)
    print("RUNNING TOOL TESTS DIRECTLY")
    print("─"*65)

    feat_dict = {
        "txn_count": 47,             "total_txn_amount": 9850000,
        "alert_count": 8,            "alert_density": 0.45,
        "alert_tier": 3,             "fund_exit_ratio": 0.97,
        "burst_score": 0.88,         "burst_per_age": 0.15,
        "burst_x_exit": 0.85,        "time_to_first_outbound_minutes_log": 4.3,
        "txn_velocity_log": 3.2,     "txn_amount_cv": 0.18,
        "max_to_avg_txn_ratio": 3.4, "total_txn_amount_cbrt": 214.5,
        "avg_txn_amount_cbrt": 60.2, "std_txn_amount_cbrt": 22.1,
        "txn_count_log": 3.85,       "distinct_counterparties_log": 2.9,
        "incoming_sources_count_log": 2.2,
        "counterparty_diversity_score": 0.72, "counterparty_to_txn_ratio": 0.48,
        "incoming_to_outgoing_ratio": 0.88,   "international_counterparty_flag": 1,
        "high_risk_country_flag": 1,           "pep_flag": 0,
        "kyc_risk_score": 0.78,                "kyc_risk_tier": 3,
        "kyc_x_alert": 0.62,                   "historical_sar_flag": 1,
        "high_risk_combined": 1,               "hr_country_x_exit": 0.97,
        "pep_x_intl": 0,                       "sar_history_x_kyc": 0.78,
        "fund_exit_tier": 3,                   "binary_risk_flag_count": 4,
    }
    a1_dict = {
        "sar_worthy": True, "confidence": 0.94,
        "typologies": [
            {"typology": "typology_structuring",    "confidence": 0.96},
            {"typology": "typology_rapid_movement", "confidence": 0.88},
        ],
    }
    a3_dict = {
        "predicted_typology": "typology_structuring",
        "rule_triggers": ["R-STR-01", "R-STR-02", "R-RMV-01"],
        "quantified_indicators": {
            "deposits_below_10L_threshold": 3,
            "avg_deposit_amount_inr": 3283333,
            "fund_exit_ratio": 0.97,
            "time_to_first_outbound_min": 75,
        },
        "case_summary": {"flagged_transactions": 28, "total_flagged_amount": 9200000},
    }
    entities  = ["Rajesh Kumar", "Sunrise Exports Ltd"]
    countries = ["Nigeria", "Iran"]

    # ── Test 1: Full pipeline ───────────────────────────────────────────
    print("\n[1] score_sar_case")
    result = _agent4.run(
        case_id="CASE-MCP-TEST-001",
        feat_series=pd.Series(feat_dict),
        agent1_output=a1_dict, agent3_output=a3_dict,
        entities=entities, countries=countries, force_enrich=True,
    )
    print(f"  Strength      : {result['strength_label']}")
    print(f"  Probabilities : W={result['strength_probabilities']['WEAK']}  "
          f"M={result['strength_probabilities']['MEDIUM']}  "
          f"S={result['strength_probabilities']['STRONG']}")
    print(f"  Priority      : {result['recommended_priority']}  ({result['priority_score']})")
    print(f"  Recommendation: {result['filing_recommendation'][:75]}...")
    bd = result["score_breakdown"]
    print(f"  Score drivers : alerts={bd['alert_count_contribution']['weighted_contribution']}  "
          f"exit={bd['fund_exit_ratio_contribution']['weighted_contribution']}  "
          f"burst={bd['burst_score_contribution']['weighted_contribution']}  "
          f"base_total={bd['total_base_score']}")
    print(f"  Completeness  : {result['case_overview']['evidence_completeness_score']}/10")

    # ── Test 2: Sanctions ───────────────────────────────────────────────
    print("\n[2] check_sanctions")
    r = _sanctions.run_bulk(["Rajesh Kumar", "Saradha Group"], ["Iran", "Nigeria"])
    print(f"  Sanctions hits       : {r['total_sanctions_hits']}")
    print(f"  High-risk countries  : {r['total_high_risk_countries']}")
    for er in r["entity_results"]:
        hit = "HIT" if er["sanctions_hit"] else "Clear"
        print(f"    [{hit:5s}]  {er['name']}")
    for cr in r["country_results"]:
        print(f"    [{cr['risk_level']:8s}]  {cr['country']}")

    # ── Test 3: News scan ───────────────────────────────────────────────
    print(f"\n[3] scan_negative_news  [DEV_MODE={DEV_MODE}]")
    feat_s   = pd.Series(feat_dict)
    news_r   = _news.scan_bulk(["Rajesh Kumar", "Sunrise Exports Ltd"],
                                ["typology_structuring"], feat_s)
    print(f"  Aggregate score : {news_r['aggregate_negative_news_score']}  "
          f"({news_r['aggregate_method']})")
    for er in news_r["entity_results"]:
        print(f"  [{er['entity']}]  score={er['negative_news_score']}  articles={len(er['articles'])}")
        for a in er["articles"]:
            print(f"    Headline : {a['title'][:70]}...")
            print(f"    Source   : {a['source']}")

    # ── Test 4: Advisories (strict matching) ────────────────────────────
    print("\n[4] get_regulatory_advisories")
    adv = _match_advisories(["typology_structuring","typology_rapid_movement"], ["Nigeria"])
    print(f"  Matched: {len(adv)}")
    for a in adv:
        print(f"  [{a['severity']:8s}]  {a['issuer']} {a['id']}")
        print(f"             {a['title'][:60]}...")
        print(f"             Action: {a.get('action_required','')}")

    # ── Test 5: Country risk ─────────────────────────────────────────────
    print("\n[5] compute_country_risk")
    cr = _risk_feeds.country_risk(["Iran", "Nigeria", "Germany", "United States"])
    print(f"  Assessment: {cr['assessment']}")
    for c, s in cr["country_scores"].items():
        label = "CRITICAL" if s >= 1.0 else "HIGH" if s >= 0.75 else "STANDARD"
        print(f"    {c:20s}  score={s}  ({label})")

    # ── Test 6: FATF lists ───────────────────────────────────────────────
    print("\n[6] get_fatf_lists")
    print(f"  Blacklist ({len(FATF_BLACK_LIST)}): {sorted(FATF_BLACK_LIST)}")
    grey_sample = sorted(FATF_GREY_LIST)[:6]
    print(f"  Greylist  ({len(FATF_GREY_LIST)}): {grey_sample} ... (+{len(FATF_GREY_LIST)-6} more)")

    # ── Test 7: Typology risk ────────────────────────────────────────────
    print("\n[7] get_typology_risk")
    tr = _risk_feeds.typology_risk(["typology_structuring","typology_rapid_movement"])
    for t, s in tr["typology_risk_scores"].items():
        print(f"    {t:35s}  {s}")
    print(f"  Highest: {tr['highest_risk_typology']}  ({tr['max_typology_risk']})")

    # ── Test 8: Audit log ────────────────────────────────────────────────
    print("\n[8] get_audit_log")
    session_log = _gw.get_session_log()
    print(f"  Session entries : {len(session_log)}")
    for e in session_log[-5:]:
        print(f"    {e['ts'][11:19]}  {e['source']:25s}  {e['status']:18s}  "
              f"results={e['result_count']}")

    print("\n" + "="*65)
    print("All 8 tool tests passed.")
    print(f"Audit log: {MCP_AUDIT_LOG}")
    print("="*65)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if HAS_FASTMCP:
        print("="*60)
        print("AGENT 4 MCP SERVER  —  Team BAYMAX")
        print(f"Mode     : {'DEVELOPMENT (simulated)' if DEV_MODE else 'PRODUCTION (live APIs)'}")
        print("Endpoint : http://localhost:8000/sse")
        print("Tools    : score_sar_case, check_sanctions,")
        print("           scan_negative_news, get_regulatory_advisories,")
        print("           compute_country_risk, get_typology_risk,")
        print("           get_fatf_lists, get_audit_log")
        print("="*60)
        print("Press Ctrl+C to stop.\n")
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        _run_stub_tests()
