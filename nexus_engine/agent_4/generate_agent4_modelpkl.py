"""
agent4_train.py  —  Agent 4: SAR Case Strength Scorer  (v2)
============================================================
Role in pipeline:
  Agent 3 produces: suspicious transactions + typology evidence bundle
  Agent 4 answers:  "how strong is this SAR case overall?"
                    → strength_label:        WEAK / MEDIUM / STRONG
                    → priority_score:        0-1 continuous
                    → recommended_priority:  URGENT / HIGH / MEDIUM / LOW
                    → evidence_gaps:         what is missing per typology
                    → filing_recommendation: text decision string

  Agent 4 also reserves a slot for contextual intelligence from external
  sources (sanctions lists, negative news, regulatory advisories, risk
  intelligence feeds). MCP / API connections are not wired in this
  training script; the field external_intelligence in the output dict
  is reserved for that layer.

No LLM, no Bedrock, no AWS. Pure ML.

────────────────────────────────────────────────────────────────
DATA SOURCE — WHY data_engineered.csv INSTEAD OF RAW CSVs
────────────────────────────────────────────────────────────────
  The previous version built its 40-feature case matrix by joining
  transactions.csv, customers.csv, accounts.csv and alerts.csv at
  training time. data_engineered.csv replaces all of that:

  ✓ Already aggregated to case level — no groupby joins needed.
  ✓ Contains 50 clean features (20 raw + 5 cbrt + 6 log1p + 19 engineered)
    that directly cover or improve 26 of the 40 original Agent4 features.
  ✓ Engineered interaction terms (burst_x_exit, kyc_x_alert, hr_country_x_exit
    etc.) carry richer signal than the raw Agent4 features they replace.
  ✓ Validated: zero nulls, zero infs, no composite leak features,
    no near-duplicate near-duplicate engineered features.
  ✓ Ships 7 label columns (sar_worthy + 6 typology flags) for multi-label
    target construction.

  Features dropped vs original Agent4 (not available in engineered CSV):
    max_alert_risk_score, avg_alert_confidence, n_distinct_rules,
    n_alert_types, has_watchlist_hit, has_velocity_spike,
    has_pattern_match, has_peer_deviation   → need alerts.csv
    n_fatf_black, n_fatf_grey, max_fatf_country_score,
    wire_to_fatf_black                      → need counterparty countries
    n_below_10L_threshold, n_round_intl_txns → need per-txn amounts
    n_high_velocity_txns, max_velocity_score → need per-txn velocity
    income_deviation_ratio                  → need declared_income
    customer_type_encoded, risk_trend_encoded → need customers.csv

  These 14 concepts are partially captured by the engineered
  high_risk_country_flag, burst_score, burst_x_exit, hr_country_x_exit
  and alert_density features already in the CSV.

────────────────────────────────────────────────────────────────
FEATURE SET  (33 selected from 50 available)
────────────────────────────────────────────────────────────────
  Group A — Transaction volume & shape  (7)
    total_txn_amount_cbrt, avg_txn_amount_cbrt, std_txn_amount_cbrt,
    txn_count_log, txn_amount_cv, max_to_avg_txn_ratio, fund_exit_ratio

  Group B — Speed & urgency  (4)
    burst_score, time_to_first_outbound_minutes_log,
    txn_velocity_log, burst_per_age

  Group C — Counterparty network  (5)
    distinct_counterparties_log, incoming_sources_count_log,
    counterparty_diversity_score, counterparty_to_txn_ratio,
    incoming_to_outgoing_ratio

  Group D — Alert signals  (4)
    alert_count, alert_density, alert_tier, kyc_x_alert

  Group E — Jurisdiction & risk flags  (7)
    international_counterparty_flag, high_risk_country_flag,
    pep_flag, kyc_risk_score, kyc_risk_tier,
    historical_sar_flag, high_risk_combined

  Group F — Interaction & tier features  (6)
    burst_x_exit, hr_country_x_exit, pep_x_intl,
    sar_history_x_kyc, fund_exit_tier, binary_risk_flag_count

────────────────────────────────────────────────────────────────
ARCHITECTURE
────────────────────────────────────────────────────────────────
  Step 1  Load data_engineered.csv; select 33 features + labels
  Step 2  Realistic 3-class strength labels via composite score +
          feature noise (σ=0.10) + 12% label flip noise
  Step 3A XGBoost multi-class classifier (WEAK / MEDIUM / STRONG)
  Step 3B Random Forest regressor (priority score 0-1)
          with data-driven URGENT / HIGH / MEDIUM / LOW thresholds
  Step 4  Feature importance (XGBoost gain + RF MDI combined)
  Step 5  Evidence gap detector (rule-based, per typology)
  Step 6  Inference function + demo

Output files: agent4/agent4_model.pkl
              agent4/agent4_metrics.json
              agent4/agent4_feature_importance.csv

────────────────────────────────────────────────────────────────
FIXES CARRIED FORWARD FROM v1
────────────────────────────────────────────────────────────────
  FIX-1  use_label_encoder removed (deprecated XGBoost param)
  FIX-2  Agent1 typology extraction uses correct 'typologies' plural key
  FIX-3  Realistic labels via composite + noise (not single-feature tercile)
  FIX-4  Data-driven priority thresholds (not hardcoded 0.80/0.60/0.40)
  FIX-5  Per-fold scaler in CV loop (no main scaler pollution)

References:
  [XGBoost]    Chen & Guestrin (2016) KDD — arXiv:1603.02754
  [RF]         Breiman (2001) Machine Learning 45(1):5-32
  [PMLA]       Prevention of Money Laundering Act 2002 (amended 2023)
  [RBI-KYC]    RBI KYC Master Direction 2016 (updated 2023)
  [FATF-2005]  FATF Money Laundering Typologies 2004-2005
  [FATF-RECS]  FATF 40 Recommendations 2012 (updated 2023)
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (f1_score, classification_report,
                                     confusion_matrix, mean_absolute_error)
import xgboost as xgb

np.random.seed(42)
os.makedirs('agent4', exist_ok=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
LABEL_COLS = [
    'sar_worthy', 'typology_structuring', 'typology_rapid_movement',
    'typology_funnel_account', 'typology_trade_based',
    'typology_shell_company', 'typology_round_tripping',
]
TYPOLOGY_COLS = LABEL_COLS[1:]

# 33 features selected from data_engineered.csv (see module docstring)
CASE_FEATURES = [
    # A — Transaction volume & shape
    'total_txn_amount_cbrt', 'avg_txn_amount_cbrt', 'std_txn_amount_cbrt',
    'txn_count_log', 'txn_amount_cv', 'max_to_avg_txn_ratio', 'fund_exit_ratio',
    # B — Speed & urgency
    'burst_score', 'time_to_first_outbound_minutes_log',
    'txn_velocity_log', 'burst_per_age',
    # C — Counterparty network
    'distinct_counterparties_log', 'incoming_sources_count_log',
    'counterparty_diversity_score', 'counterparty_to_txn_ratio',
    'incoming_to_outgoing_ratio',
    # D — Alert signals
    'alert_count', 'alert_density', 'alert_tier', 'kyc_x_alert',
    # E — Jurisdiction & risk flags
    'international_counterparty_flag', 'high_risk_country_flag',
    'pep_flag', 'kyc_risk_score', 'kyc_risk_tier',
    'historical_sar_flag', 'high_risk_combined',
    # F — Interaction & tier features
    'burst_x_exit', 'hr_country_x_exit', 'pep_x_intl',
    'sar_history_x_kyc', 'fund_exit_tier', 'binary_risk_flag_count',
]
assert len(CASE_FEATURES) == 33

STRENGTH_MAP  = {0: 'WEAK', 1: 'MEDIUM', 2: 'STRONG'}
LABELS_MAP    = STRENGTH_MAP  # alias for print statements

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("AGENT 4: SAR Case Strength Scorer  (v2 — data_engineered.csv)")
print("=" * 60)

df = pd.read_csv('data_engineered.csv')
print(f"\nLoaded : {len(df):,} rows × {df.shape[1]} cols")
print(f"SAR rate: {df['sar_worthy'].mean():.3f}  "
      f"({df['sar_worthy'].sum():,} SAR / {(df['sar_worthy']==0).sum():,} non-SAR)")

# Validate all selected features are present
missing_feats = [f for f in CASE_FEATURES if f not in df.columns]
if missing_feats:
    raise ValueError(f"Features missing from data_engineered.csv: {missing_feats}")

# Validate all label columns are present
missing_labels = [l for l in LABEL_COLS if l not in df.columns]
if missing_labels:
    raise ValueError(f"Label columns missing: {missing_labels}")

print(f"\nFeature groups (33 total):")
print(f"  A Transaction volume & shape : 7  (cbrt amounts, txn_count_log, cv, fund_exit)")
print(f"  B Speed & urgency             : 4  (burst, time_to_first_outbound, velocity)")
print(f"  C Counterparty network        : 5  (distinct_cp, sources, diversity, ratios)")
print(f"  D Alert signals               : 4  (count, density, tier, kyc interaction)")
print(f"  E Jurisdiction & risk flags   : 7  (intl, high-risk, PEP, KYC, SAR history)")
print(f"  F Interaction & tier features : 6  (burst×exit, hr×exit, PEP×intl, tiers)")

# ── TYPOLOGY DISTRIBUTION ─────────────────────────────────────────────────────
print(f"\nTypology distribution (SAR cases only):")
sar_df = df[df['sar_worthy'] == 1]
for col in TYPOLOGY_COLS:
    n = sar_df[col].sum()
    print(f"  {col:<35}: {n:>4}  ({n/len(sar_df)*100:.1f}%)")
multi = (sar_df[TYPOLOGY_COLS].sum(axis=1) > 1).sum()
print(f"  Multi-typology cases: {multi} ({multi/len(sar_df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1: Feature matrix from data_engineered.csv")
print("="*60)

cf = df[CASE_FEATURES + LABEL_COLS].copy().reset_index(drop=True)
X_raw = cf[CASE_FEATURES].values.astype(float)

print(f"  Feature matrix: {X_raw.shape[0]:,} rows × {X_raw.shape[1]} cols")
print(f"  Null check : {cf[CASE_FEATURES].isnull().sum().sum()} nulls")
print(f"  Inf check  : {np.isinf(X_raw).sum()} infs")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: REALISTIC 3-CLASS STRENGTH LABELS
#
# Composite score across 5 feature groups with Gaussian noise + label flip.
# See module docstring FIX-3 for rationale.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 2: Generating realistic 3-class strength labels")
print("="*60)

rng = np.random.default_rng(42)

def _norm(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)

noise_scale = 0.10
alert_noisy   = (_norm(cf['alert_count'])
                 + rng.normal(0, noise_scale, len(cf))).clip(0, 1)
exit_noisy    = (_norm(cf['fund_exit_ratio'])
                 + rng.normal(0, noise_scale, len(cf))).clip(0, 1)
burst_noisy   = (_norm(cf['burst_score'])
                 + rng.normal(0, noise_scale, len(cf))).clip(0, 1)
kyc_noisy     = (_norm(cf['kyc_risk_score'])
                 + rng.normal(0, noise_scale, len(cf))).clip(0, 1)
risk_noisy    = (_norm(cf['high_risk_combined'].astype(float))
                 + rng.normal(0, noise_scale, len(cf))).clip(0, 1)

composite = (
    0.30 * alert_noisy  +
    0.25 * exit_noisy   +
    0.20 * burst_noisy  +
    0.15 * kyc_noisy    +
    0.10 * risk_noisy
)

q33 = float(np.percentile(composite, 33.33))
q66 = float(np.percentile(composite, 66.67))
print(f"  Composite score terciles: q33={q33:.3f}  q66={q66:.3f}")

labels_clean = np.where(composite > q66, 2,
               np.where(composite > q33, 1, 0))

LABEL_NOISE_RATE = 0.12
n_flip   = int(LABEL_NOISE_RATE * len(cf))
flip_idx = rng.choice(len(cf), size=n_flip, replace=False)
labels_noisy = labels_clean.copy()
for idx in flip_idx:
    current    = labels_noisy[idx]
    candidates = [c for c in [current - 1, current + 1] if 0 <= c <= 2]
    labels_noisy[idx] = rng.choice(candidates)

cf['strength_label'] = labels_noisy.astype(int)

print(f"\n  Strength label distribution:")
for s in [0, 1, 2]:
    n = (cf['strength_label'] == s).sum()
    print(f"    {LABELS_MAP[s]:6s} ({s}): {n:4d}  ({n/len(cf)*100:.1f}%)")

# Continuous priority score (0-1) for RF regression
# 5-signal weighted formula; more diverse than original 4-signal version
cf['priority_score'] = (
    0.28 * cf['alert_count'].clip(0, 18) / 18 +        # normalise to [0,1]
    0.22 * cf['fund_exit_ratio'].clip(0, 1)  +
    0.20 * cf['burst_score'].clip(0, 1)      +
    0.18 * cf['kyc_risk_score'].clip(0, 1)   +
    0.12 * cf['high_risk_combined'].astype(float)
).clip(0, 1).round(4)

print(f"\n  Priority score stats:")
print(f"    mean={cf['priority_score'].mean():.3f}  "
      f"std={cf['priority_score'].std():.3f}  "
      f"min={cf['priority_score'].min():.3f}  "
      f"max={cf['priority_score'].max():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3A: XGBoost MULTI-CLASS CLASSIFIER  (WEAK / MEDIUM / STRONG)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3A: XGBoost classifier  (WEAK / MEDIUM / STRONG)")
print("="*60)

X     = cf[CASE_FEATURES].values.astype(float)
y_cls = cf['strength_label'].values
y_reg = cf['priority_score'].values

X_tv, X_test, y_tv_cls, y_test_cls, y_tv_reg, y_test_reg = train_test_split(
    X, y_cls, y_reg, test_size=0.15, random_state=42, stratify=y_cls)

X_train, X_val, y_train_cls, y_val_cls, y_train_reg, y_val_reg = train_test_split(
    X_tv, y_tv_cls, y_tv_reg,
    test_size=0.15 / 0.85, random_state=42, stratify=y_tv_cls)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"  Class distribution (train):")
for s in [0, 1, 2]:
    n = (y_train_cls == s).sum()
    print(f"    {LABELS_MAP[s]:6s}: {n:,}")

# FIX-1: use_label_encoder removed
clf_model = xgb.XGBClassifier(
    n_estimators          = 500,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.80,
    colsample_bytree      = 0.70,
    min_child_weight      = 3,
    gamma                 = 0.1,
    reg_alpha             = 0.1,
    reg_lambda            = 1.5,
    objective             = 'multi:softprob',
    num_class             = 3,
    eval_metric           = 'mlogloss',
    early_stopping_rounds = 30,
    random_state          = 42,
    verbosity             = 0,
)
clf_model.fit(X_train_s, y_train_cls,
              eval_set=[(X_val_s, y_val_cls)], verbose=False)

y_pred_cls = clf_model.predict(X_test_s)
f1_macro   = f1_score(y_test_cls, y_pred_cls, average='macro',    zero_division=0)
f1_weight  = f1_score(y_test_cls, y_pred_cls, average='weighted', zero_division=0)

print(f"\n  Results:")
print(f"    Macro F1   : {f1_macro:.4f}")
print(f"    Weighted F1: {f1_weight:.4f}")
print(f"    Best round : {clf_model.best_iteration + 1}")
print(f"\n  Classification report:")
print(classification_report(y_test_cls, y_pred_cls,
      target_names=['WEAK', 'MEDIUM', 'STRONG'], zero_division=0))

cm = confusion_matrix(y_test_cls, y_pred_cls)
print(f"  Confusion matrix (rows=actual, cols=predicted):")
print(f"               WEAK   MED   STR")
for i, lbl in enumerate(['WEAK', 'MEDIUM', 'STRONG']):
    print(f"  Actual {lbl:<6}: {cm[i]}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3B: RANDOM FOREST REGRESSOR  (priority score 0-1)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3B: Random Forest regressor  (priority score 0-1)")
print("="*60)

rf_model = RandomForestRegressor(
    n_estimators     = 300,
    max_depth        = 6,
    min_samples_leaf = 3,
    max_features     = 'sqrt',
    oob_score        = True,
    random_state     = 42,
    n_jobs           = -1,
)
rf_model.fit(X_train_s, y_train_reg)

y_pred_reg = np.clip(rf_model.predict(X_test_s), 0, 1)
mae        = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"  Test MAE  : {mae:.4f}")
print(f"  OOB  R²   : {rf_model.oob_score_:.4f}")
print(f"  Pred range: [{y_pred_reg.min():.3f}, {y_pred_reg.max():.3f}]")

# 5-fold CV — FIX-5: dedicated per-fold scaler keeps main scaler clean
skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_maes = []
for fold_train, fold_val in skf.split(X, y_cls):
    fs  = StandardScaler()
    Xtr = fs.fit_transform(X[fold_train])
    Xvl = fs.transform(X[fold_val])
    rf_cv = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   min_samples_leaf=3, random_state=42)
    rf_cv.fit(Xtr, y_reg[fold_train])
    cv_maes.append(mean_absolute_error(y_reg[fold_val],
                   np.clip(rf_cv.predict(Xvl), 0, 1)))

print(f"  5-fold CV MAE: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

# FIX-4: data-driven priority thresholds from training predictions
train_preds_reg = np.clip(rf_model.predict(X_train_s), 0, 1)
PRI_URGENT = float(np.percentile(train_preds_reg, 75))
PRI_HIGH   = float(np.percentile(train_preds_reg, 50))
PRI_MEDIUM = float(np.percentile(train_preds_reg, 25))
print(f"\n  Priority thresholds (training percentiles):")
print(f"    URGENT  >= {PRI_URGENT:.3f}  (top 25%)")
print(f"    HIGH    >= {PRI_HIGH:.3f}  (top 50%)")
print(f"    MEDIUM  >= {PRI_MEDIUM:.3f}  (top 75%)")
print(f"    LOW      < {PRI_MEDIUM:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 4: Feature importance  (XGBoost gain + RF MDI, 60/40)")
print("="*60)

fi = pd.DataFrame({
    'feature':  CASE_FEATURES,
    'xgb_gain': clf_model.feature_importances_,
    'rf_mdi':   rf_model.feature_importances_,
})
fi['combined'] = 0.60 * fi['xgb_gain'] + 0.40 * fi['rf_mdi']
fi = fi.sort_values('combined', ascending=False).reset_index(drop=True)

print(f"\n  {'Rank':<5} {'Feature':<40} {'XGB':>8} {'RF':>8} {'Comb':>8}  Bar")
print(f"  {'-'*75}")
for rank, row in fi.iterrows():
    bar = '█' * int(row['combined'] * 200)
    print(f"  {rank+1:<5} {row['feature']:<40} {row['xgb_gain']:>8.4f} "
          f"{row['rf_mdi']:>8.4f} {row['combined']:>8.4f}  {bar}")

fi.to_csv('agent4/agent4_feature_importance.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: EVIDENCE GAP DETECTOR  (rule-based, per typology)
#
# Checks whether the case has the expected evidence elements for each
# detected typology. Thresholds are mapped to the engineered feature names.
# Where the original Agent4 used raw-CSV features that are absent from
# data_engineered.csv, equivalent engineered features are substituted:
#
#   Original                → Engineered substitute
#   n_alerts                → alert_count
#   max_alert_risk_score    → alert_density (alerts/30-days, positively correlated)
#   n_distinct_counterparties → distinct_counterparties_log (log-scaled)
#   log_time_to_first_outbound → time_to_first_outbound_minutes_log
#   max_velocity_score      → burst_score (same rapid-movement concept)
#   n_high_velocity_txns    → burst_tier (ordinal encoding of velocity)
#   coeff_variation         → txn_amount_cv
#   n_incoming_sources      → incoming_sources_count_log
#   fund_exit_ratio         → fund_exit_ratio (same)
#   n_round_intl_txns       → hr_country_x_exit > 0 (high-risk × exit proxy)
#   n_fatf_black            → high_risk_country_flag (binary FATF proxy)
#   max_fatf_country_score  → high_risk_country_flag + kyc_risk_score combined
# ══════════════════════════════════════════════════════════════════════════════

EVIDENCE_REQUIREMENTS = {
    'typology_structuring': [
        ('alert_count',     lambda v: v >= 2,   "Fewer than 2 corroborating alerts"),
        ('txn_amount_cv',   lambda v: v < 0.20, "Amount variation too high — structuring needs consistent sizing  [FATF-2005 §2.4]"),
        ('alert_density',   lambda v: v > 0.10, "Alert density too low for structuring pattern"),
        ('fund_exit_ratio', lambda v: v < 0.60, "High exit ratio inconsistent with deposit structuring"),
    ],
    'typology_rapid_movement': [
        ('time_to_first_outbound_minutes_log', lambda v: v < 5.8, "No rapid outbound detected (exit time > ~330 min)  [FATF-2005 §2.2]"),
        ('fund_exit_ratio',  lambda v: v > 0.80, "Fund exit ratio < 0.80 — insufficient pass-through evidence"),
        ('burst_score',      lambda v: v > 0.60, "Burst score < 0.60 — insufficient velocity evidence"),
        ('burst_tier',       lambda v: v >= 2,   "Burst tier < 2 — fewer than expected high-velocity transactions"),
    ],
    'typology_trade_based': [
        ('high_risk_country_flag', lambda v: v == 1, "No high-risk country exposure  [FATF-RECS R.19]"),
        ('hr_country_x_exit',      lambda v: v > 0,  "No high-risk country × exit interaction  [FATF-TBML §2.2]"),
        ('international_counterparty_flag', lambda v: v == 1, "No international counterparty"),
        ('kyc_risk_score',         lambda v: v > 0.50, "KYC risk score < 0.50 — insufficient jurisdiction risk"),
    ],
    'typology_funnel_account': [
        ('incoming_sources_count_log', lambda v: v > 1.6, "Fewer than ~4 distinct incoming sources  [FATF-2005 §4.1]"),
        ('fund_exit_ratio',            lambda v: v > 0.70, "Fund exit ratio < 0.70"),
        ('counterparty_to_txn_ratio',  lambda v: v > 0.30, "Low counterparty-to-transaction ratio — insufficient fan-in"),
        ('alert_count',                lambda v: v >= 2,   "Fewer than 2 corroborating alerts"),
    ],
    'typology_shell_company': [
        ('distinct_counterparties_log', lambda v: v > 2.5, "Fewer than ~12 distinct counterparties  [FATF-SHELL §3.1]"),
        ('alert_count',                 lambda v: v >= 3,  "Fewer than 3 alerts"),
        ('high_risk_country_flag',      lambda v: v == 1,  "No high-risk country exposure"),
        ('pep_x_intl',                  lambda v: v == 1,  "No PEP + international combination"),
    ],
    'typology_round_tripping': [
        ('fund_exit_ratio',  lambda v: v > 0.85, "Fund exit ratio < 0.85  [FATF-SHELL §4.2]"),
        ('high_risk_country_flag', lambda v: v == 1, "No high-risk country exposure"),
        ('time_to_first_outbound_minutes_log', lambda v: v > 4.0, "No delayed outbound flow detected"),
        ('burst_x_exit',     lambda v: v > 0.30, "burst × exit interaction too low — insufficient round-trip signal"),
    ],
    'multi': [
        ('alert_count',      lambda v: v >= 2,   "Fewer than 2 corroborating alerts"),
        ('fund_exit_ratio',  lambda v: v > 0.70, "Fund exit ratio < 0.70"),
        ('burst_score',      lambda v: v > 0.50, "Burst score < 0.50"),
        ('txn_amount_cv',    lambda v: v < 0.30, "Amount variation too high for combined typology"),
    ],
}


def detect_gaps(feat_series, typologies):
    """
    Return {typology: [gap_description, ...]} for each detected typology.

    Parameters
    ----------
    feat_series : pd.Series  indexed by feature name
    typologies  : list[str]  typology column names, e.g. ['typology_structuring']
                             OR short names like 'structuring' (prefix auto-added)
    """
    gaps = {}
    for typ in typologies:
        # Accept both 'structuring' and 'typology_structuring' as keys
        key = typ if typ in EVIDENCE_REQUIREMENTS else f'typology_{typ}'
        reqs = EVIDENCE_REQUIREMENTS.get(key, [])
        missing = []
        for feat, check_fn, description in reqs:
            if feat in feat_series.index:
                if not check_fn(feat_series[feat]):
                    missing.append(description)
        gaps[typ] = missing
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: INFERENCE FUNCTION + DEMO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 6: Inference function + demo")
print("="*60)

# Column rename map: data_engineered.csv name → canonical name used here
# (inference function accepts either name via the feat_series lookup)
FEAT_ALIASES = {
    # Engineered CSV name        : Agent4 concept
    'alert_count':               'n_alerts',
    'txn_count':                 'n_transactions',      # raw; log version used in model
    'txn_amount_cv':             'coeff_variation',
    'time_to_first_outbound_minutes_log': 'log_time_to_first_outbound',
    'distinct_counterparties':   'n_distinct_counterparties',  # raw; log used in model
    'incoming_sources_count':    'n_incoming_sources',         # raw; log used in model
    'burst_score':               'max_velocity_score_proxy',
    'kyc_risk_score':            'customer_risk_score',
    'historical_sar_flag':       'previous_sar_flag',
}


def score_case(
    case_id,
    feat_series,           # pd.Series: one row from data_engineered.csv
    agent1_output,         # dict: {sar_worthy, confidence, typologies:[{typology,confidence}]}
    agent3_output,         # dict: {predicted_typology, typology_evidence, case_summary}
    clf_model   = clf_model,
    rf_model    = rf_model,
    scaler      = scaler,
    feat_names  = CASE_FEATURES,
    pri_urgent  = PRI_URGENT,
    pri_high    = PRI_HIGH,
    pri_medium  = PRI_MEDIUM,
    evidence_requirements = EVIDENCE_REQUIREMENTS,
):
    """
    Agent 4 inference.

    Parameters
    ----------
    case_id      : str
    feat_series  : pd.Series — one row from data_engineered.csv (or equivalent)
                   Must contain all columns in CASE_FEATURES.
    agent1_output: dict — {sar_worthy, confidence,
                            typologies:[{typology:str, confidence:float}]}
    agent3_output: dict — {predicted_typology, typology_evidence, case_summary}

    Returns
    -------
    dict:
      strength_label, strength_probabilities, priority_score,
      recommended_priority, filing_recommendation, evidence_gaps,
      case_overview, external_intelligence (reserved for MCP layer)
    """
    X_inf = scaler.transform(
        np.array([feat_series[f] for f in feat_names], dtype=float).reshape(1, -1)
    )

    strength_probs = clf_model.predict_proba(X_inf)[0]
    strength_idx   = int(np.argmax(strength_probs))
    priority_s     = float(np.clip(rf_model.predict(X_inf)[0], 0, 1))

    # FIX-4: data-driven thresholds
    if   priority_s >= pri_urgent: pri_label = 'URGENT'
    elif priority_s >= pri_high:   pri_label = 'HIGH'
    elif priority_s >= pri_medium: pri_label = 'MEDIUM'
    else:                          pri_label = 'LOW'

    if strength_idx == 2 and priority_s >= pri_high:
        rec = "FILE SAR IMMEDIATELY — strong multi-signal evidence, high priority"
    elif strength_idx >= 1 and priority_s >= pri_medium:
        rec = "FILE SAR — sufficient evidence; analyst review recommended before filing"
    elif strength_idx == 1:
        rec = "PENDING — gather additional evidence before filing"
    else:
        rec = "CLOSE — insufficient evidence for SAR filing at this time"

    # FIX-2: correct Agent1 schema parsing (plural 'typologies' list of dicts)
    a1_typologies = [t['typology'] for t in agent1_output.get('typologies', [])]
    a3_typology   = agent3_output.get('predicted_typology', '')
    all_typologies = list(dict.fromkeys(
        a1_typologies + ([a3_typology] if a3_typology else [])
    ))
    gaps = detect_gaps(feat_series, all_typologies)

    return {
        'case_id':             case_id,
        'strength_label':      STRENGTH_MAP[strength_idx],
        'strength_probabilities': {
            'WEAK':   round(float(strength_probs[0]), 3),
            'MEDIUM': round(float(strength_probs[1]), 3),
            'STRONG': round(float(strength_probs[2]), 3),
        },
        'priority_score':        round(priority_s, 3),
        'recommended_priority':  pri_label,
        'filing_recommendation': rec,
        'evidence_gaps':         gaps,
        # Reserved for MCP / sanctions / news / advisory layer
        'external_intelligence': None,
        'case_overview': {
            'n_transactions':               int(feat_series.get('txn_count', 0)),
            'n_suspicious_txns_agent3':     agent3_output.get(
                                                'case_summary', {}).get(
                                                'flagged_transactions', 0),
            'alert_count':                  int(feat_series.get('alert_count', 0)),
            'alert_density':                round(float(feat_series.get('alert_density', 0)), 3),
            'fund_exit_ratio':              round(float(feat_series.get('fund_exit_ratio', 0)), 3),
            'burst_score':                  round(float(feat_series.get('burst_score', 0)), 3),
            'has_high_risk_country':        bool(feat_series.get('high_risk_country_flag', 0)),
            'has_pep_exposure':             bool(feat_series.get('pep_flag', 0)),
            'kyc_risk_score':               round(float(feat_series.get('kyc_risk_score', 0)), 3),
            'typologies_assessed':          all_typologies,
            'evidence_completeness_score':  sum([
                feat_series.get('alert_count', 0) > 0,
                feat_series.get('alert_count', 0) > 3,
                feat_series.get('high_risk_country_flag', 0) == 1,
                feat_series.get('kyc_risk_score', 0) > 0.70,
                feat_series.get('fund_exit_ratio', 0) > 0.80,
                feat_series.get('burst_score', 0) > 0.70,
                feat_series.get('distinct_counterparties_log', 0) > 2.0,
                feat_series.get('pep_flag', 0) == 1,
            ]),
        },
    }


# Demo — use first SAR case from dataset
demo_row = df[df['sar_worthy'] == 1].iloc[0]
demo_id  = f"CASE-DEMO-{demo_row.name}"

# FIX-2: Real Agent1 schema (plural typologies list of dicts)
# Derive typology from the label columns in the demo row
demo_typologies = [col for col in TYPOLOGY_COLS if demo_row[col] == 1]
agent1_mock = {
    'sar_worthy':  True,
    'confidence':  0.87,
    'typologies':  [{'typology': t, 'confidence': 0.82} for t in demo_typologies],
}
# Real Agent3 schema
agent3_mock = {
    'predicted_typology': demo_typologies[0] if demo_typologies else 'unknown',
    'case_summary': {
        'flagged_transactions':   int(demo_row['txn_count'] * 0.6),
        'total_flagged_amount':   float(demo_row['total_txn_amount'] * 0.6),
    },
}

result = score_case(demo_id, demo_row, agent1_mock, agent3_mock)

print(f"\n  Demo — {demo_id}")
print(f"  True typologies: {demo_typologies}")
print(f"  Strength        : {result['strength_label']}")
print(f"  Probabilities   : {result['strength_probabilities']}")
print(f"  Priority score  : {result['priority_score']}")
print(f"  Priority label  : {result['recommended_priority']}")
print(f"  Recommendation  : {result['filing_recommendation']}")
print(f"  Typologies assessed: {result['case_overview']['typologies_assessed']}")
print(f"  Evidence gaps:")
for typ, gaps in result['evidence_gaps'].items():
    if gaps:
        for g in gaps:
            print(f"    [{typ}] {g}")
    else:
        print(f"    [{typ}] no gaps detected")
print(f"  Completeness score: {result['case_overview']['evidence_completeness_score']}/8")


# ── SAVE ──────────────────────────────────────────────────────────────────────
joblib.dump({
    'clf_model':   clf_model,
    'rf_model':    rf_model,
    'scaler':      scaler,
    'case_features':  CASE_FEATURES,
    'strength_map':   STRENGTH_MAP,
    'typology_cols':  TYPOLOGY_COLS,
    'priority_thresholds': {
        'urgent': PRI_URGENT, 'high': PRI_HIGH, 'medium': PRI_MEDIUM,
    },
    'evidence_requirements': {
        k: [(f, None, d) for f, _, d in v]
        for k, v in EVIDENCE_REQUIREMENTS.items()
    },
    'data_source':    'data_engineered.csv',
    'feat_aliases':   FEAT_ALIASES,
}, 'agent4/agent4_model.pkl')

metrics = {
    'data_source': 'data_engineered.csv',
    'n_rows': len(df),
    'n_features_available': 50,
    'n_features_used': len(CASE_FEATURES),
    'classifier': {
        'f1_macro':    round(f1_macro, 4),
        'f1_weighted': round(f1_weight, 4),
        'best_round':  int(clf_model.best_iteration + 1),
        'n_classes':   3,
        'class_names': ['WEAK', 'MEDIUM', 'STRONG'],
        'label_generation': {
            'method':           'composite_score_with_noise',
            'noise_sigma':       0.10,
            'label_noise_rate':  0.12,
            'composite_weights': {
                'alert_count': 0.30, 'fund_exit_ratio': 0.25,
                'burst_score': 0.20, 'kyc_risk_score':  0.15,
                'high_risk_combined': 0.10,
            },
        },
    },
    'regressor': {
        'mae':         round(mae, 4),
        'oob_r2':      round(rf_model.oob_score_, 4),
        'cv_mae_mean': round(float(np.mean(cv_maes)), 4),
        'cv_mae_std':  round(float(np.std(cv_maes)), 4),
        'priority_thresholds': {
            'urgent':  round(PRI_URGENT, 4),
            'high':    round(PRI_HIGH,   4),
            'medium':  round(PRI_MEDIUM, 4),
        },
    },
    'splits': {
        'train': len(X_train), 'val': len(X_val), 'test': len(X_test),
    },
    'label_distribution': {
        STRENGTH_MAP[s]: int((cf['strength_label'] == s).sum())
        for s in [0, 1, 2]
    },
    'features': CASE_FEATURES,
}
with open('agent4/agent4_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*60}")
print("SAVED")
print(f"  agent4/agent4_model.pkl")
print(f"  agent4/agent4_metrics.json")
print(f"  agent4/agent4_feature_importance.csv")
print(f"\n  Classifier : macro F1={f1_macro:.3f}  weighted F1={f1_weight:.3f}")
print(f"  Regressor  : test MAE={mae:.3f}  OOB R²={rf_model.oob_score_:.3f}")
print(f"  Thresholds : URGENT>={PRI_URGENT:.3f}  HIGH>={PRI_HIGH:.3f}  MEDIUM>={PRI_MEDIUM:.3f}")
print("="*60)
