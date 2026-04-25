"""
agent3_typology.py  —  Agent 3: Typology-Level Evidence Analyser
=================================================================

ACTIVE IMPLEMENTATION
---------------------
  Rule-based evidence extractor driven by agent state.
  Typology is sourced from state["typology"] (Agent 1 output via pipeline.py).
  Transaction data is sourced from state["structured_case"]["transactions"].
  Customer KYC is sourced from state["structured_case"]["customer"].

  Entry point: run_agent3(state: SARAgentState) -> dict
  Updates state fields: triggered_rules, quantified_indicators

COMMENTED-OUT SECTION (below active code)
------------------------------------------
  The original ML training pipeline (Steps 1–8) is preserved below
  for future reference. It includes:
    - 31-feature transaction-level engineering
    - MI feature selection
    - 70/15/15 case-level stratified split
    - Optuna 30-trial hyperparameter tuning
    - XGBoost multi:softprob classifier (6 typology classes)
    - analyse_case() inference function with embedded ML classification
  Do not delete — may be reintegrated if per-transaction ML classification
  is needed independently of Agent 1's case-level typology.

RULE IDs
--------
  R-STR-01  THRESHOLD_STRUCTURING        Amount in 900K-1M structuring band
  R-STR-02  CONSISTENT_SUB_THRESHOLD     Case CV < 0.15 on CREDIT txns
  R-HVT-01  HIGH_VELOCITY_WIRE           SWIFT wire with velocity > 0.75
  R-HVT-02  RAPID_OUTBOUND_EXIT          Outbound wire within tight time gap
  R-LYR-01  MULTI_HOP_TRANSFER           Internal or RTGS multi-hop
  R-IWT-01  INTL_WIRE_TRANSFER           Cross-border outbound wire
  R-TBM-01  ROUND_AMOUNT_INTL            Round 100K amount to foreign country
  R-IWT-02  FATF_JURISDICTION_WIRE       Wire to/from FATF risk score > 0.50

REFERENCES
----------
  [FATF-2005]  FATF Money Laundering Typologies 2004-2005
  [FATF-TBML]  FATF Trade-Based Money Laundering 2006
  [FATF-RECS]  FATF 40 Recommendations 2012 (updated 2023)
  [PMLA]       Prevention of Money Laundering Act 2002 (amended 2023)
  [RBI-KYC]    RBI KYC Master Direction 2016 (updated 2023)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── LangGraph / LLM imports ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "agent3_cognitive_flow.txt"

def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Agent 3 prompt not found at {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()
from typing import Any

# ── FATF LOOKUP TABLES ────────────────────────────────────────────────────────

FATF_BLACK = {'IR', 'KP', 'MM', 'YE'}
FATF_GREY  = {'AE', 'PK', 'NG', 'VN', 'ET', 'SD', 'SY'}
FATF_SCORE = {
    'IN': 0.10, 'GB': 0.10, 'US': 0.10, 'SG': 0.15, 'DE': 0.10,
    'AE': 0.45, 'PK': 0.70, 'NG': 0.70, 'VN': 0.55, 'ET': 0.65,
    'IR': 0.95, 'KP': 1.00, 'MM': 0.90, 'YE': 0.85, 'SD': 0.75, 'SY': 0.80,
}
RISK_ENC = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}

# ── TYPOLOGY NORMALISATION MAP ────────────────────────────────────────────────
# Maps Agent 1 typology strings (title-case, spaced) to rule-engine keys

_TYPOLOGY_NORM = {
    'structuring':           'structuring',
    'rapid movement':        'rapid_movement',
    'rapid movement of funds': 'rapid_movement',
    'rapid_movement':        'rapid_movement',
    'funnel account':        'layering',
    'funnel_account':        'layering',
    'layering':              'layering',
    'tbml':                  'tbml',
    'trade based':           'tbml',
    'trade_based':           'tbml',
    'shell company':         'layering',
    'shell_company':         'layering',
    'round tripping':        'tbml',
    'round_tripping':        'tbml',
    'multi':                 'multi',
    'clean':                 'clean',
    'unknown':               'clean',
}


# ── RISK FLAG DEFINITIONS ─────────────────────────────────────────────────────

RISK_FLAGS: dict[str, list[dict]] = {
    'structuring': [
        {
            'rule_id':   'R-STR-01',
            'rule_name': 'THRESHOLD_STRUCTURING',
            'severity':  'HIGH',
            'check':     lambda r: r['is_below_threshold'] == 1,
            'desc':      'Amount INR {amount:.0f} in structuring band 900K-1M, '
                         'just below INR 10L CTR threshold  [PMLA Rule 3]',
        },
        {
            'rule_id':   'R-STR-02',
            'rule_name': 'CONSISTENT_SUB_THRESHOLD',
            'severity':  'MEDIUM',
            'check':     lambda r: r['case_cv'] < 0.15 and r['is_credit'] == 1,
            'desc':      'Case amount CV {case_cv:.3f} < 0.15: '
                         'highly consistent sizing pattern across deposits  [FATF-2005 §2.4]',
        },
    ],
    'rapid_movement': [
        {
            'rule_id':   'R-HVT-01',
            'rule_name': 'HIGH_VELOCITY_WIRE',
            'severity':  'HIGH',
            'check':     lambda r: r['velocity_score'] > 0.75 and r['is_swift'] == 1,
            'desc':      'SWIFT wire velocity {velocity_score:.3f} > 0.75: '
                         'rapid international movement  [FATF-2005 §2.2]',
        },
        {
            'rule_id':   'R-HVT-02',
            'rule_name': 'RAPID_OUTBOUND_EXIT',
            'severity':  'HIGH',
            'check':     lambda r: r['is_wire_out'] == 1 and r['log_time_gap_hrs'] < 3.5,
            'desc':      'Outbound wire {time_gap_hrs:.1f}h after prior txn: '
                         'rapid fund exit pattern  [FATF-2005 §4.1]',
        },
    ],
    'layering': [
        {
            'rule_id':   'R-LYR-01',
            'rule_name': 'MULTI_HOP_TRANSFER',
            'severity':  'MEDIUM',
            'check':     lambda r: r['is_internal'] == 1 or (r['is_wire_out'] == 1 and r['is_rtgs'] == 1),
            'desc':      'Internal/RTGS transfer: multi-hop layering movement  [FATF-2005 §3.1]',
        },
        {
            'rule_id':   'R-IWT-01',
            'rule_name': 'INTL_WIRE_TRANSFER',
            'severity':  'HIGH',
            'check':     lambda r: r['wire_out_intl'] == 1,
            'desc':      'International wire INR {amount:.0f} to {country}: '
                         'cross-border layering  [FATF-RECS R.16]',
        },
    ],
    'tbml': [
        {
            'rule_id':   'R-TBM-01',
            'rule_name': 'ROUND_AMOUNT_INTL',
            'severity':  'HIGH',
            'check':     lambda r: r['is_round_100k'] == 1 and r['is_cross_border'] == 1,
            'desc':      'Round INR {amount:.0f} to {country}: '
                         'TBML over/under-invoicing pattern  [FATF-TBML §2.2]',
        },
        {
            'rule_id':   'R-IWT-02',
            'rule_name': 'FATF_JURISDICTION_WIRE',
            'severity':  'HIGH',
            'check':     lambda r: r['is_cross_border'] == 1 and r['fatf_score'] > 0.50,
            'desc':      'Wire to/from {country} (FATF risk {fatf_score:.2f}): '
                         'high-risk jurisdiction  [FATF-RECS R.19]',
        },
    ],
    'clean': [],
    'multi': [],  # triggers both structuring + rapid_movement rules
}

THRESHOLD_CHECKS: list[dict] = [
    {
        'id':    'CTR_STRUCTURING',
        'label': 'PMLA Rule 3: structuring band INR 900K-999K',
        'check': lambda r: r['is_below_threshold'] == 1,
    },
    {
        'id':    'CTR_BREACH',
        'label': 'PMLA Rule 3: CTR threshold breached (>= INR 10L)',
        'check': lambda r: r['amount'] >= 1_000_000,
    },
    {
        'id':    'FATF_BLACK_WIRE',
        'label': 'FATF-RECS R.19: wire to/from FATF blacklisted country',
        'check': lambda r: r['is_fatf_black'] == 1 and (r['is_wire_out'] == 1 or r['is_wire_in'] == 1),
    },
    {
        'id':    'FATF_GREY_RAPID',
        'label': 'FATF-RECS R.19: high-velocity txn with grey-list counterparty',
        'check': lambda r: r['is_fatf_grey'] == 1 and r['velocity_score'] > 0.75,
    },
    {
        'id':    'SWIFT_RAPID',
        'label': 'FATF-2005 §2.2: SWIFT wire within rapid timeframe (< 3.5h gap)',
        'check': lambda r: r['is_swift'] == 1 and r['log_time_gap_hrs'] < 3.5,
    },
    {
        'id':    'ROUND_INTL',
        'label': 'FATF-TBML §2.2: round-number international transaction',
        'check': lambda r: r['is_round_100k'] == 1 and r['is_cross_border'] == 1,
    },
    {
        'id':    'HIGH_VAL_RAPID',
        'label': 'High-value transaction with elevated velocity score',
        'check': lambda r: r['is_high_value_int'] == 1 and r['velocity_score'] > 0.75,
    },
]


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _normalise_typology(raw: str) -> str:
    """Map Agent 1 typology string to rule-engine key."""
    return _TYPOLOGY_NORM.get(raw.lower().strip(), 'clean')


def _build_txn_context(
    txn: dict,
    c_mean: float,
    c_std: float,
    c_tot: float,
    c_n: int,
    c_vel: float,
    c_cv: float,
    tgap: float,
    customer: dict,
) -> dict:
    """
    Build the flat context dict for a single transaction.
    Used by both THRESHOLD_CHECKS and RISK_FLAGS lambdas.

    Parameters
    ----------
    txn      : one transaction dict from structured_case["transactions"]
    c_*      : case-level aggregate stats computed from all transactions
    tgap     : hours since previous transaction (99.0 for first txn)
    customer : structured_case["customer"] dict
    """
    amount  = float(txn.get('amount', 0))
    country = str(txn.get('counterparty_country', 'IN'))
    txn_type = str(txn.get('txn_type', ''))
    channel  = str(txn.get('channel', ''))
    vs       = float(np.clip(float(txn.get('velocity_score', 0)), 0, 1))

    return {
        # ── amount signals ────────────────────────────────────────────────
        'amt_case_zscore':    float((amount - c_mean) / (c_std + 1)),
        'amt_pct_of_case':    float(amount / (c_tot + 1)),
        'amt_vs_case_mean':   float(amount / (c_mean + 1)),
        'is_below_threshold': int(900_000 <= amount < 1_000_000),
        'amt_near_10L':       float(np.clip(abs(amount - 950_000) / 50_001, 0, 1)),
        'is_round_100k':      int(amount % 100_000 < 1_000),
        'is_round_50k':       int(amount % 50_000 < 500),
        'is_high_value_int':  int(bool(txn.get('is_high_value', False))),
        # ── jurisdiction signals ──────────────────────────────────────────
        'is_fatf_black':      int(country in FATF_BLACK),
        'is_fatf_grey':       int(country in FATF_GREY),
        'country_fatf_score': float(FATF_SCORE.get(country, 0.20)),
        'is_cross_border':    int(country != 'IN'),
        # ── transaction type / channel ────────────────────────────────────
        'is_wire_out':  int(txn_type == 'WIRE_OUT'),
        'is_wire_in':   int(txn_type == 'WIRE_IN'),
        'is_internal':  int(txn_type == 'INTERNAL_TRANSFER'),
        'is_credit':    int(txn_type == 'CREDIT'),
        'is_swift':     int(channel == 'SWIFT'),
        'is_rtgs':      int(channel == 'RTGS'),
        'is_branch':    int(channel == 'BRANCH'),
        'wire_out_intl':   int(txn_type == 'WIRE_OUT' and country != 'IN'),
        'wire_fatf_black': int(txn_type == 'WIRE_OUT' and country in FATF_BLACK),
        # ── temporal ─────────────────────────────────────────────────────
        'log_time_gap_hrs': float(np.log1p(tgap)),
        # ── velocity ─────────────────────────────────────────────────────
        'velocity_score':   vs,
        'vel_vs_case_mean': vs / (c_vel + 0.001),
        'is_high_vel':      int(vs > 0.75),
        # ── case context ─────────────────────────────────────────────────
        'case_cv':    float(c_cv),
        'log_case_n': float(np.log1p(c_n)),
        # ── KYC ──────────────────────────────────────────────────────────
        'risk_enc':      RISK_ENC.get(str(customer.get('risk_rating', 'LOW')).upper(), 0),
        'prev_sar_flag': int(int(customer.get('previous_sar_count', 0)) > 0),
        # ── raw values for rule descriptions ─────────────────────────────
        'amount':      amount,
        'country':     country,
        'fatf_score':  float(FATF_SCORE.get(country, 0.20)),
        'time_gap_hrs': float(tgap),
    }


def _compute_case_stats(transactions: list[dict]) -> tuple[float, float, float, int, float, float]:
    """Compute case-level aggregate stats from transaction list."""
    amounts = [float(t.get('amount', 0)) for t in transactions]
    velocities = [float(np.clip(float(t.get('velocity_score', 0)), 0, 1)) for t in transactions]
    c_mean = float(np.mean(amounts)) if amounts else 0.0
    c_std  = float(np.std(amounts))  if len(amounts) > 1 else 0.0
    c_tot  = float(np.sum(amounts))
    c_n    = len(amounts)
    c_vel  = float(np.mean(velocities)) if velocities else 0.0
    c_cv   = c_std / (c_mean + 1)
    return c_mean, c_std, c_tot, c_n, c_vel, c_cv


def _time_gap_hrs(current_dt: Any, previous_dt: Any) -> float:
    """Hours between two datetime-like values. Returns 99.0 if no previous."""
    if previous_dt is None:
        return 99.0
    try:
        cur = pd.Timestamp(current_dt)
        prv = pd.Timestamp(previous_dt)
        return max(0.0, (cur - prv).total_seconds() / 3600)
    except Exception:
        return 99.0


# ── CORE EVIDENCE EXTRACTOR ───────────────────────────────────────────────────

def extract_evidence(
    case_id:      str,
    typology:     str,
    transactions: list[dict],
    customer:     dict,
) -> dict:
    """
    Rule-based evidence extractor.

    Parameters
    ----------
    case_id      : from state["case_id"]
    typology     : from state["typology"]  (Agent 1 output)
    transactions : from state["structured_case"]["transactions"]
    customer     : from state["structured_case"]["customer"]

    Returns
    -------
    dict with keys:
      case_id, typology, transactions (per-txn evidence),
      typology_evidence (bundle), case_summary,
      triggered_rules (flat list for state),
      quantified_indicators (flat dict for state)
    """
    if not transactions:
        return {
            'case_id': case_id,
            'error': 'No transactions provided',
            'triggered_rules': [],
            'quantified_indicators': {},
        }

    norm_typology = _normalise_typology(typology)

    # ── case-level stats ──────────────────────────────────────────────────────
    c_mean, c_std, c_tot, c_n, c_vel, c_cv = _compute_case_stats(transactions)

    # sort by date for time-gap calculation
    sorted_txns = sorted(
        transactions,
        key=lambda t: pd.Timestamp(t.get('txn_date', '1970-01-01'))
    )

    # ── per-transaction analysis ──────────────────────────────────────────────
    txns_out = []
    buckets: dict[str, list] = {}
    prev_dt = None

    for txn in sorted_txns:
        tgap = _time_gap_hrs(txn.get('txn_date'), prev_dt)
        prev_dt = txn.get('txn_date')

        ctx = _build_txn_context(txn, c_mean, c_std, c_tot, c_n, c_vel, c_cv, tgap, customer)

        # anomaly indicators (human-readable subset)
        anomaly_indicators = {
            'amt_case_zscore':     round(ctx['amt_case_zscore'], 3),
            'velocity_score':      round(ctx['velocity_score'], 3),
            'below_threshold_flag':ctx['is_below_threshold'],
            'round_amt_deviation': round(ctx['amt_near_10L'], 4),
            'fatf_risk_score':     round(ctx['country_fatf_score'], 3),
            'time_gap_hrs':        round(tgap, 2),
            'amt_pct_of_case':     round(ctx['amt_pct_of_case'], 4),
            'vel_vs_case_mean':    round(ctx['vel_vs_case_mean'], 3),
        }

        # threshold breaches — always run all checks regardless of typology
        threshold_breaches = [
            tc['label'] for tc in THRESHOLD_CHECKS
            if _safe_check(tc['check'], ctx)
        ]

        # risk flags — routed by typology from agent state
        typs_to_check = (
            ['structuring', 'rapid_movement'] if norm_typology == 'multi'
            else [norm_typology]
        )
        risk_flags = []
        for typ in typs_to_check:
            for fd in RISK_FLAGS.get(typ, []):
                if _safe_check(fd['check'], ctx):
                    risk_flags.append({
                        'rule_id':    fd['rule_id'],
                        'rule_name':  fd['rule_name'],
                        'severity':   fd['severity'],
                        'triggered':  True,
                        'description': _format_desc(fd['desc'], ctx),
                    })

        txn_result = {
            'txn_id':               txn.get('txn_id', ''),
            'txn_date':             str(pd.Timestamp(txn.get('txn_date', '')).date()),
            'amount':               round(ctx['amount'], 2),
            'currency':             txn.get('currency', 'INR'),
            'txn_type':             txn.get('txn_type', ''),
            'channel':              txn.get('channel', ''),
            'counterparty_name':    txn.get('counterparty_name', ''),
            'counterparty_country': ctx['country'],
            'typology':             norm_typology,
            'anomaly_indicators':   anomaly_indicators,
            'threshold_breaches':   threshold_breaches,
            'risk_flags':           risk_flags,
        }
        txns_out.append(txn_result)
        buckets.setdefault(norm_typology, []).append(txn_result)

    # ── typology evidence bundle ──────────────────────────────────────────────
    typology_evidence = _build_typology_evidence(buckets, c_cv, FATF_BLACK, FATF_GREY)

    # ── case summary ──────────────────────────────────────────────────────────
    flagged   = [tx for tx in txns_out if tx['risk_flags'] or tx['threshold_breaches']]
    all_rules = sorted(set(f['rule_id'] for tx in txns_out for f in tx['risk_flags']))

    case_summary = {
        'total_transactions':   len(txns_out),
        'flagged_transactions': len(flagged),
        'pct_flagged':          round(len(flagged) / max(len(txns_out), 1) * 100, 1),
        'primary_typology':     norm_typology,
        'total_flagged_amount': round(sum(tx['amount'] for tx in flagged), 2),
        'has_fatf_exposure':    any(
            tx['counterparty_country'] in FATF_BLACK | FATF_GREY for tx in txns_out
        ),
        'has_regulatory_breach': len(flagged) > 0,
        'rules_triggered':      all_rules,
        'typologies_detected':  list(buckets.keys()),
    }

    # ── state-compatible flat outputs ─────────────────────────────────────────
    triggered_rules = [
        f for tx in txns_out for f in tx['risk_flags']
    ]

    quantified_indicators = _build_quantified_indicators(
        txns_out, c_mean, c_std, c_tot, c_n, c_vel, c_cv, norm_typology
    )

    return {
        'case_id':              case_id,
        'typology':             norm_typology,
        'transactions':         txns_out,
        'typology_evidence':    typology_evidence,
        'case_summary':         case_summary,
        'triggered_rules':      triggered_rules,
        'quantified_indicators': quantified_indicators,
    }


# ── EVIDENCE BUNDLE BUILDER ───────────────────────────────────────────────────

def _build_typology_evidence(
    buckets: dict,
    c_cv: float,
    fatf_black: set,
    fatf_grey: set,
) -> dict:
    """Build per-typology evidence summary from bucketed transaction results."""
    evidence = {}
    for typ, txns in buckets.items():
        all_rules    = list(set(f['rule_id'] for tx in txns for f in tx['risk_flags']))
        all_breaches = list(set(b for tx in txns for b in tx['threshold_breaches']))
        fatf_exp     = any(tx['counterparty_country'] in fatf_black | fatf_grey for tx in txns)

        if typ == 'structuring':
            bc  = sum(1 for tx in txns if tx['anomaly_indicators']['below_threshold_flag'])
            key_metrics = {
                'n_below_threshold':   bc,
                'pct_below_threshold': round(bc / max(len(txns), 1), 3),
                'avg_amount':          round(float(np.mean([tx['amount'] for tx in txns])), 2),
                'case_cv':             round(c_cv, 4),
            }
        elif typ == 'rapid_movement':
            hv = [tx for tx in txns if tx['anomaly_indicators']['velocity_score'] > 0.75]
            key_metrics = {
                'n_high_velocity':  len(hv),
                'avg_velocity':     round(float(np.mean([tx['anomaly_indicators']['velocity_score'] for tx in txns])), 3),
                'avg_time_gap_hrs': round(float(np.mean([tx['anomaly_indicators']['time_gap_hrs'] for tx in txns])), 2),
                'pct_swift':        round(sum(1 for tx in txns if tx['channel'] == 'SWIFT') / max(len(txns), 1), 3),
            }
        elif typ == 'layering':
            key_metrics = {
                'n_internal':   sum(1 for tx in txns if tx['txn_type'] == 'INTERNAL_TRANSFER'),
                'n_countries':  len(set(tx['counterparty_country'] for tx in txns)),
                'pct_outbound': round(sum(1 for tx in txns if tx['txn_type'] in ['WIRE_OUT', 'DEBIT']) / max(len(txns), 1), 3),
            }
        elif typ == 'tbml':
            rc = sum(1 for tx in txns if tx['anomaly_indicators']['round_amt_deviation'] < 0.01)
            key_metrics = {
                'n_round_amounts': rc,
                'pct_round':       round(rc / max(len(txns), 1), 3),
                'avg_fatf_score':  round(float(np.mean([tx['anomaly_indicators']['fatf_risk_score'] for tx in txns])), 3),
                'countries':       sorted(set(tx['counterparty_country'] for tx in txns if tx['counterparty_country'] != 'IN')),
            }
        else:
            key_metrics = {}

        evidence[typ] = {
            'n_transactions':    len(txns),
            'total_amount':      round(sum(tx['amount'] for tx in txns), 2),
            'txn_ids':           [tx['txn_id'] for tx in txns],
            'threshold_breaches': all_breaches,
            'rule_ids_triggered': all_rules,
            'has_fatf_exposure':  fatf_exp,
            'key_metrics':        key_metrics,
        }
    return evidence


def _build_quantified_indicators(
    txns_out: list,
    c_mean: float,
    c_std: float,
    c_tot: float,
    c_n: int,
    c_vel: float,
    c_cv: float,
    typology: str,
) -> dict:
    """
    Flat dict of quantified stats for state["quantified_indicators"].
    Agent 5 injects these directly into the SAR narrative prompt.
    """
    amounts    = [tx['amount'] for tx in txns_out]
    velocities = [tx['anomaly_indicators']['velocity_score'] for tx in txns_out]
    flagged    = [tx for tx in txns_out if tx['risk_flags'] or tx['threshold_breaches']]

    return {
        'transaction_count':      c_n,
        'total_amount':           round(c_tot, 2),
        'avg_txn_amount':         round(c_mean, 2),
        'std_txn_amount':         round(c_std, 2),
        'max_txn_amount':         round(max(amounts), 2) if amounts else 0.0,
        'min_txn_amount':         round(min(amounts), 2) if amounts else 0.0,
        'case_cv':                round(c_cv, 4),
        'avg_velocity_score':     round(float(np.mean(velocities)), 3) if velocities else 0.0,
        'flagged_txn_count':      len(flagged),
        'flagged_amount':         round(sum(tx['amount'] for tx in flagged), 2),
        'pct_flagged':            round(len(flagged) / max(c_n, 1) * 100, 1),
        'has_fatf_exposure':      any(tx['counterparty_country'] in FATF_BLACK | FATF_GREY for tx in txns_out),
        'has_regulatory_breach':  len(flagged) > 0,
        'primary_typology':       typology,
        'unique_rules_triggered': len(set(f['rule_id'] for tx in txns_out for f in tx['risk_flags'])),
    }


def _safe_check(fn, ctx: dict) -> bool:
    """Run a rule lambda safely, returning False on any exception."""
    try:
        return bool(fn(ctx))
    except Exception:
        return False


def _format_desc(template: str, ctx: dict) -> str:
    """Format a rule description template with context values."""
    try:
        return template.format(
            amount=ctx['amount'],
            country=ctx['country'],
            velocity_score=ctx['velocity_score'],
            case_cv=ctx['case_cv'],
            time_gap_hrs=ctx['time_gap_hrs'],
            fatf_score=ctx['fatf_score'],
        )
    except Exception:
        return template


# ── LANGGRAPH AGENT ENTRY POINT ───────────────────────────────────────────────

def run_agent3(state: dict) -> dict:
    """
    LangGraph node function for Agent 3.

    Reads from state:
      state["case_id"]
      state["typology"]                              — from Agent 1
      state["structured_case"]["transactions"]       — list of txn dicts
      state["structured_case"]["customer"]           — KYC dict

    Writes to state:
      triggered_rules        — list[dict], accumulates via operator.add
      quantified_indicators  — dict[str, Any]
      cognitive_event_flow   — dict[str, Any]  (LLM output)
      error_log              — list[str]
    """
    case_id      = state.get('case_id', 'UNKNOWN')
    typology     = state.get('typology', 'unknown')
    structured   = state.get('structured_case', {})
    transactions = structured.get('transactions', [])
    customer     = structured.get('customer', {})
    errors: list[str] = []

    # ── Step 1: Rule-based evidence extraction ────────────────────
    result = extract_evidence(
        case_id=case_id,
        typology=typology,
        transactions=transactions,
        customer=customer,
    )

    triggered_rules       = result.get('triggered_rules', [])
    quantified_indicators = result.get('quantified_indicators', {})

    # Deduplicate triggered_rules — same rule on same transaction is a duplicate.
    # operator.add in state accumulates across revision loops so we deduplicate
    # here before returning to prevent the list doubling on each revision.
    seen: set[tuple] = set()
    deduped_rules: list[dict] = []
    for rule in triggered_rules:
        key = (rule.get('rule_id', ''), rule.get('description', ''))
        if key not in seen:
            seen.add(key)
            deduped_rules.append(rule)
    triggered_rules = deduped_rules

    # ── Step 2: LLM cognitive event flow ─────────────────────────
    cognitive_event_flow = _build_cognitive_event_flow(
        case_id=case_id,
        typology=typology,
        transactions=transactions,
        customer=customer,
        triggered_rules=triggered_rules,
        quantified_indicators=quantified_indicators,
        errors=errors,
    )

    return {
        'triggered_rules':       triggered_rules,
        'quantified_indicators': quantified_indicators,
        'cognitive_event_flow':  cognitive_event_flow,
        'error_log':             errors,
    }


def _build_cognitive_event_flow(
    case_id: str,
    typology: str,
    transactions: list[dict],
    customer: dict,
    triggered_rules: list[dict],
    quantified_indicators: dict,
    errors: list[str],
) -> dict:
    """
    Calls the LLM with rule evidence + transaction data to produce a
    structured cognitive event flow — the analyst's mental model of the case.
    Falls back to an empty skeleton on any failure so the pipeline never stops.
    """
    # Build a compact transaction list for the prompt (avoid token bloat)
    txn_summary = [
        {
            "txn_id":              t.get("txn_id"),
            "date":                t.get("txn_date"),
            "type":                t.get("txn_type"),
            "amount":              t.get("amount"),
            "currency":            t.get("currency"),
            "channel":             t.get("channel"),
            "counterparty_country": t.get("counterparty_country"),
            "is_high_value":       t.get("is_high_value"),
            "velocity_score":      t.get("velocity_score"),
        }
        for t in transactions
    ]

    customer_summary = {
        "risk_rating":        customer.get("risk_rating", "UNKNOWN"),
        "pep_flag":           customer.get("pep_flag", False),
        "previous_sar_count": customer.get("previous_sar_count", 0),
        "nationality":        customer.get("nationality", ""),
        "customer_type":      customer.get("customer_type", "individual"),
    }

    user_content = json.dumps({
        "case_id":              case_id,
        "typology":             typology,
        "quantified_indicators": quantified_indicators,
        "triggered_rules":      triggered_rules,
        "transactions":         txn_summary,
        "customer":             customer_summary,
    }, indent=2, default=str)

    try:
        system_prompt = _load_prompt()
        llm = get_llm(temperature=0.2)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        logger.info(f"Agent 3 [{case_id}] — calling LLM for cognitive event flow")
        response = llm.invoke(messages)
        raw: str = response.content.strip()

        # Strip markdown fences if the LLM wraps output anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        cognitive_flow: dict = json.loads(raw)
        logger.info(f"Agent 3 [{case_id}] — cognitive event flow built successfully")
        return cognitive_flow

    except json.JSONDecodeError as e:
        msg = f"[{_ts()}] Agent3 [{case_id}]: LLM returned non-JSON for cognitive flow — {e}"
        logger.error(msg)
        errors.append(msg)
    except Exception as e:
        msg = f"[{_ts()}] Agent3 [{case_id}]: LLM call failed for cognitive flow — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # Fallback skeleton — downstream agents handle empty gracefully
    return {
        "event_sequence":    [],
        "fund_flow_summary": "",
        "key_anomalies":     [],
        "analyst_hypothesis": "",
        "risk_summary": {
            "primary_concern":    "",
            "supporting_evidence": [],
            "confidence":         "LOW",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# COMMENTED OUT — ORIGINAL ML TRAINING PIPELINE
# Preserved for future reference. Do not delete.
# To reactivate: uncomment and ensure transactions.csv, customers.csv,
# accounts.csv, alerts.csv are available in the working directory.
# ═════════════════════════════════════════════════════════════════════════════

# import joblib
# import os
# from sklearn.model_selection   import train_test_split
# from sklearn.preprocessing     import StandardScaler, LabelEncoder
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.metrics           import (f1_score, classification_report,
#                                        balanced_accuracy_score, confusion_matrix)
# import xgboost as xgb
# import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)
#
# np.random.seed(42)
# os.makedirs('agent3', exist_ok=True)
#
# # ── LOAD ──────────────────────────────────────────────────────────────────────
# print("=" * 65)
# print("AGENT 3: Typology-Level Evidence Analyser")
# print("=" * 65)
#
# txn  = pd.read_csv('transactions.csv', parse_dates=['txn_date'])
# cust = pd.read_csv('customers.csv')
# acct = pd.read_csv('accounts.csv')
# alrt = pd.read_csv('alerts.csv')
#
# print(f"\nLoaded: {len(txn):,} txns | {txn['case_id'].nunique()} cases | {len(alrt)} alerts")
# print(f"Alert rules: {sorted(alrt['rule_name'].unique())}")
#
# case_nums = {c: int(c.split('-')[-1]) for c in txn['case_id'].unique()}
#
# def _typology(case_id: str) -> str:
#     n = case_nums[case_id]
#     if n <= 60:  return 'clean'
#     if n <= 85:  return 'structuring'
#     if n <= 105: return 'rapid_movement'
#     if n <= 120: return 'layering'
#     if n <= 135: return 'tbml'
#     if n <= 145: return 'multi'
#     return 'clean'
#
# txn['true_typology'] = txn['case_id'].map(_typology)
#
# # ── STEP 1: FEATURE ENGINEERING (31 features, 7 groups) ──────────────────────
# txn = txn.sort_values(['case_id', 'txn_date']).reset_index(drop=True)
# acct_cust = acct.merge(cust, on='customer_id', how='left')
# txn = txn.merge(
#     acct_cust[['account_id','risk_rating','pep_flag','declared_income','previous_sar_count']],
#     on='account_id', how='left'
# )
# cs = txn.groupby('case_id').agg(
#     c_mean=('amount','mean'), c_std=('amount','std'),
#     c_tot=('amount','sum'),   c_n=('txn_id','count'),
#     c_vel=('velocity_score','mean'),
# ).reset_index()
# cs['c_std'] = cs['c_std'].fillna(0)
# txn = txn.merge(cs, on='case_id', how='left')
# txn['time_gap_hrs'] = (
#     txn.groupby('case_id')['txn_date'].diff().dt.total_seconds().div(3600).fillna(99)
# )
# vs = txn['velocity_score'].clip(0, 1)
# txn['log_amount']         = np.log1p(txn['amount'])
# txn['amt_vs_case_mean']   = txn['amount'] / (txn['c_mean'] + 1)
# txn['amt_case_zscore']    = (txn['amount'] - txn['c_mean']) / (txn['c_std'] + 1)
# txn['amt_pct_of_case']    = txn['amount'] / (txn['c_tot'] + 1)
# txn['is_fatf_black']      = txn['counterparty_country'].isin(FATF_BLACK).astype(int)
# txn['is_fatf_grey']       = txn['counterparty_country'].isin(FATF_GREY).astype(int)
# txn['country_fatf_score'] = txn['counterparty_country'].map(FATF_SCORE).fillna(0.20)
# txn['is_cross_border']    = (txn['counterparty_country'] != 'IN').astype(int)
# txn['is_wire_out']        = (txn['txn_type'] == 'WIRE_OUT').astype(int)
# txn['is_wire_in']         = (txn['txn_type'] == 'WIRE_IN').astype(int)
# txn['is_internal']        = (txn['txn_type'] == 'INTERNAL_TRANSFER').astype(int)
# txn['is_credit']          = (txn['txn_type'] == 'CREDIT').astype(int)
# txn['is_swift']           = (txn['channel'] == 'SWIFT').astype(int)
# txn['is_rtgs']            = (txn['channel'] == 'RTGS').astype(int)
# txn['is_branch']          = (txn['channel'] == 'BRANCH').astype(int)
# txn['wire_out_intl']      = (txn['is_wire_out'] & txn['is_cross_border']).astype(int)
# txn['wire_fatf_black']    = (txn['is_wire_out'] & txn['is_fatf_black']).astype(int)
# txn['log_time_gap_hrs']   = np.log1p(txn['time_gap_hrs'])
# txn['is_offhours']        = ((txn['txn_date'].dt.hour < 9) | (txn['txn_date'].dt.hour > 18)).astype(int)
# txn['velocity_score']     = vs
# txn['vel_vs_case_mean']   = vs / (txn['c_vel'] + 0.001)
# txn['is_high_vel']        = (vs > 0.75).astype(int)
# txn['is_below_threshold'] = ((txn['amount'] >= 900_000) & (txn['amount'] < 1_000_000)).astype(int)
# txn['amt_near_10L']       = np.clip(np.abs(txn['amount'] - 950_000) / 50_001, 0, 1)
# txn['is_round_100k']      = (txn['amount'] % 100_000 < 1_000).astype(int)
# txn['is_round_50k']       = (txn['amount'] % 50_000  <   500).astype(int)
# txn['is_high_value_int']  = txn['is_high_value'].astype(int)
# txn['case_cv']            = txn['c_std'] / (txn['c_mean'] + 1)
# txn['log_case_n']         = np.log1p(txn['c_n'])
# txn['risk_enc']           = txn['risk_rating'].map(RISK_ENC).fillna(0).astype(int)
# txn['prev_sar_flag']      = (txn['previous_sar_count'] > 0).astype(int)
#
# ALL_FEATURES = [
#     'log_amount','amt_vs_case_mean','amt_case_zscore','amt_pct_of_case',
#     'is_fatf_black','is_fatf_grey','country_fatf_score','is_cross_border',
#     'is_wire_out','is_wire_in','is_internal','is_credit',
#     'is_swift','is_rtgs','is_branch','wire_out_intl','wire_fatf_black',
#     'log_time_gap_hrs','is_offhours',
#     'velocity_score','vel_vs_case_mean','is_high_vel',
#     'is_below_threshold','amt_near_10L','is_round_100k','is_round_50k','is_high_value_int',
#     'case_cv','log_case_n','risk_enc','prev_sar_flag',
# ]
#
# # ── STEP 2: MI FEATURE SELECTION ─────────────────────────────────────────────
# MI_THRESHOLD = 0.05
# le    = LabelEncoder()
# y_all = le.fit_transform(txn['true_typology'])
# mi    = mutual_info_classif(txn[ALL_FEATURES].fillna(0).values, y_all, random_state=42)
# mi_df = pd.DataFrame({'feature':ALL_FEATURES,'mi_score':mi}).sort_values('mi_score',ascending=False)
# selected     = mi_df[mi_df['mi_score'] >= MI_THRESHOLD]['feature'].tolist()
# dropped      = mi_df[mi_df['mi_score'] <  MI_THRESHOLD]['feature'].tolist()
# LABEL_CLASSES = list(le.classes_)
# mi_df.to_csv('agent3/agent3_mi_scores.csv', index=False)
#
# # ── STEP 3: CASE-LEVEL STRATIFIED SPLIT (70/15/15) ───────────────────────────
# ctdf = txn[['case_id','true_typology']].drop_duplicates('case_id').copy()
# ctdf['enc'] = le.transform(ctdf['true_typology'])
# c_tv, c_te  = train_test_split(ctdf['case_id'].values, test_size=0.15,
#                                 random_state=42, stratify=ctdf['enc'].values)
# y_tv = ctdf.set_index('case_id').loc[c_tv, 'enc'].values
# c_tr, c_vl  = train_test_split(c_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)
# tr_m = txn['case_id'].isin(c_tr)
# va_m = txn['case_id'].isin(c_vl)
# te_m = txn['case_id'].isin(c_te)
# X_tr = txn.loc[tr_m, selected].fillna(0).values
# X_va = txn.loc[va_m, selected].fillna(0).values
# X_te = txn.loc[te_m, selected].fillna(0).values
# y_tr = le.transform(txn.loc[tr_m, 'true_typology'])
# y_va = le.transform(txn.loc[va_m, 'true_typology'])
# y_te = le.transform(txn.loc[te_m, 'true_typology'])
# sc       = StandardScaler()
# X_trs    = sc.fit_transform(X_tr)
# X_vas    = sc.transform(X_va)
# X_tes    = sc.transform(X_te)
#
# # ── STEP 4: OPTUNA HYPERPARAMETER TUNING (30 trials) ─────────────────────────
# N_TRIALS = 30
# def objective(trial: optuna.Trial) -> float:
#     params = dict(
#         n_estimators=trial.suggest_int('n_estimators', 50, 500),
#         max_depth=trial.suggest_int('max_depth', 2, 7),
#         learning_rate=trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
#         subsample=trial.suggest_float('subsample', 0.50, 1.00),
#         colsample_bytree=trial.suggest_float('colsample_bytree', 0.40, 1.00),
#         min_child_weight=trial.suggest_int('min_child_weight', 1, 15),
#         gamma=trial.suggest_float('gamma', 0.0, 3.0),
#         reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
#         reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
#         objective='multi:softprob', num_class=len(LABEL_CLASSES),
#         eval_metric='mlogloss', random_state=42, verbosity=0,
#         early_stopping_rounds=15,
#     )
#     clf = xgb.XGBClassifier(**params)
#     clf.fit(X_trs, y_tr, eval_set=[(X_vas, y_va)], verbose=False)
#     return float(f1_score(y_va, clf.predict(X_vas), average='macro', zero_division=0))
# study = optuna.create_study(direction='maximize',
#     sampler=optuna.samplers.TPESampler(seed=42),
#     pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
# study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
# best_params = study.best_params
# best_val_f1 = study.best_value
#
# # ── STEP 5: FINAL MODEL TRAINING ─────────────────────────────────────────────
# final_params = {**best_params, 'objective':'multi:softprob',
#     'num_class':len(LABEL_CLASSES), 'eval_metric':'mlogloss',
#     'random_state':42, 'verbosity':0, 'early_stopping_rounds':20}
# model = xgb.XGBClassifier(**final_params)
# model.fit(X_trs, y_tr, eval_set=[(X_vas, y_va)], verbose=False)
# y_pred  = model.predict(X_tes)
# macro_f1    = f1_score(y_te, y_pred, average='macro',    zero_division=0)
# weighted_f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
# bal_acc     = balanced_accuracy_score(y_te, y_pred)
#
# # ── STEP 6: FEATURE IMPORTANCE ───────────────────────────────────────────────
# fi_df = pd.DataFrame({'feature':selected,'gain':model.feature_importances_}).sort_values(
#     'gain', ascending=False).reset_index(drop=True)
# fi_df.to_csv('agent3/agent3_feature_importance.csv', index=False)
#
# # ── SAVE ARTEFACTS ────────────────────────────────────────────────────────────
# joblib.dump({
#     'model': model, 'scaler': sc, 'label_encoder': le,
#     'selected_features': selected, 'all_features': ALL_FEATURES,
#     'label_classes': LABEL_CLASSES, 'fatf_black': list(FATF_BLACK),
#     'fatf_grey': list(FATF_GREY), 'fatf_score': FATF_SCORE,
#     'mi_threshold': MI_THRESHOLD, 'optuna_best_params': best_params,
#     'risk_flags_meta': {
#         k: [{'rule_id':f['rule_id'],'rule_name':f['rule_name'],'severity':f['severity']}
#             for f in v] for k, v in RISK_FLAGS.items()
#     },
# }, 'agent3/agent3_model.pkl')
