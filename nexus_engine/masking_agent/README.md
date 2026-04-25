# masking_agent

> **AI-Privacy Guard Layer** — SAR Narrative Generator · Team Baymax · Barclays Hack-O-Hire

---

## Pipeline position

```
[Backend — Call 1]
  └─► Agent 1  (separate Lambda)
        writes: sar_worthy, confidence_score, typology, risk_score
        → returns AgentState to backend

[Backend — between calls]
  └─► fetches structured_case from RDS → populates AgentState
  └─► calls masking_agent  action="mask"   ◄── THIS MODULE
        • validates full AgentState (all Agent 1 fields + schema)
        • enforces sar_worthy gate  (False → 400)
        • masks transactions[].counterparty_name / counterparty_account
        • stores token ↔ real_value in RDS pii_mask_map
        • returns full updated AgentState (all other fields unchanged)

[Backend — Call 2]
  └─► Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6
        LLM (Amazon Bedrock) never sees real PII

[Backend — after Agent 6]
  └─► calls masking_agent  action="unmask"  ◄── THIS MODULE
        • single bulk DB read for all tokens
        • restores real PII in SAR narrative
        → Analyst Review UI

[Backend — after SAR approval]
  └─► calls masking_agent  action="delete_map"
        • purges pii_mask_map rows for the case
```

---

## What gets masked

Per the **AgentState structured_case schema** (state definition document):

| Sub-dict | Field | Masked? | Notes |
|---|---|---|---|
| `customer` | `risk_rating` | ✗ | `str` "LOW"\|"MEDIUM"\|"HIGH" — operational |
| `customer` | `customer_type` | ✗ | `str` — operational |
| `customer` | `nationality` | ✗ | `str` ISO-2 — operational |
| `customer` | `pep_flag` | ✗ | `bool` — operational |
| `customer` | `previous_sar_count` | ✗ | `int` — operational |
| `accounts[]` | `account_id` | ✗ | internal system ID |
| `accounts[]` | `account_type` | ✗ | operational |
| `accounts[]` | `account_age_days` | ✗ | operational |
| `accounts[]` | `international_txn` | ✗ | operational |
| `transactions[]` | `txn_id` | ✗ | internal system ID |
| `transactions[]` | `txn_date` | ✗ | ISO-8601 — operational |
| `transactions[]` | `txn_type` | ✗ | WIRE_OUT\|CREDIT etc. — Agent 3 needs this |
| `transactions[]` | `amount` | ✗ | operational |
| `transactions[]` | `currency` | ✗ | ISO-4217 — operational |
| `transactions[]` | `channel` | ✗ | SWIFT\|NEFT etc. — Agent 3 needs this |
| `transactions[]` | **`counterparty_name`** | **✓ PII** | `→ <NAME_XXXXXXXX>` |
| `transactions[]` | **`counterparty_account`** | **✓ PII** | `→ <ACCOUNT_XXXXXXXX>` |
| `transactions[]` | `counterparty_country` | ✗ | ISO-2 — Agent 3 FATF logic needs this |
| `transactions[]` | `is_high_value` | ✗ | operational |
| `transactions[]` | `velocity_score` | ✗ | operational |

**Only `counterparty_name` and `counterparty_account` inside `transactions[]` are masked.**

---

## Token format

```
<ENTITY_TYPE_XXXXXXXX>
```

| Raw value | Token |
|---|---|
| `Alice Brown` | `<NAME_4A3F1B2C>` |
| `CPTY-ACC-001` | `<ACCOUNT_9D7E2A0F>` |

- Same `(case_id, raw_value)` → same token always (deterministic within a case)
- Globally unique across all cases (DB `UNIQUE` constraint)

---

## Files

```
masking_agent/
├── masking_agent.py            ← Core logic
├── lambda_handler.py           ← Lambda entry point + AgentState validation
├── requirements.txt
├── migrations/
│   └── 001_create_pii_mask_map.sql
└── tests/
    └── test_masking_agent.py   ← 87 tests, no real DB needed
```

---

## RDS setup

```bash
psql -h $DB_HOST -U $DB_USER -d $DB_NAME \
  -f migrations/001_create_pii_mask_map.sql
```

---

## Lambda environment variables

| Variable | Description |
|---|---|
| `DB_HOST` | RDS endpoint |
| `DB_PORT` | default `5432` |
| `DB_NAME` | database name |
| `DB_USER` | database user |
| `DB_PASSWORD` | use AWS Secrets Manager in production |
| `DB_SSLMODE` | default `require` |

---

## Lambda API

### `mask`

```json
{
  "action": "mask",
  "state": {
    "case_id":          "CASE-2026-007",
    "s3_bucket":        "my-sar-bucket",
    "s3_prefix":        "cases/CASE-2026-007/inputs",
    "transactions_csv": "",
    "sar_worthy":       true,
    "confidence_score": 0.91,
    "typology":         "Structuring",
    "risk_score":       0.78,
    "structured_case": {
      "customer": {
        "risk_rating": "HIGH", "customer_type": "individual",
        "nationality": "IN", "pep_flag": false, "previous_sar_count": 2
      },
      "accounts": [
        {"account_id": "ACC-001", "account_type": "current",
         "account_age_days": 365, "international_txn": true}
      ],
      "transactions": [
        {"txn_id": "TXN-001", "txn_date": "2026-01-15T10:00:00",
         "txn_type": "WIRE_OUT", "amount": 500000.0, "currency": "INR",
         "channel": "SWIFT",
         "counterparty_name": "Alice Brown",
         "counterparty_account": "CPTY-ACC-001",
         "counterparty_country": "AE",
         "is_high_value": true, "velocity_score": 0.85}
      ]
    }
  }
}
```

Response:
```json
{
  "state": {
    "case_id": "CASE-2026-007",
    "s3_bucket": "my-sar-bucket",
    "s3_prefix": "cases/CASE-2026-007/inputs",
    "transactions_csv": "",
    "sar_worthy": true,
    "confidence_score": 0.91,
    "typology": "Structuring",
    "risk_score": 0.78,
    "structured_case": {
      "customer": { "risk_rating": "HIGH", "customer_type": "individual",
                    "nationality": "IN", "pep_flag": false,
                    "previous_sar_count": 2 },
      "accounts": [ {"account_id": "ACC-001", "account_type": "current",
                     "account_age_days": 365, "international_txn": true} ],
      "transactions": [
        {"txn_id": "TXN-001", "txn_date": "2026-01-15T10:00:00",
         "txn_type": "WIRE_OUT", "amount": 500000.0, "currency": "INR",
         "channel": "SWIFT",
         "counterparty_name": "<NAME_4A3F1B2C>",
         "counterparty_account": "<ACCOUNT_9D7E2A0F>",
         "counterparty_country": "AE",
         "is_high_value": true, "velocity_score": 0.85}
      ]
    }
  },
  "token_count": 2
}
```

### `unmask`

```json
{
  "action":    "unmask",
  "case_id":   "CASE-2026-007",
  "narrative": "Funds sent to <NAME_4A3F1B2C> via <ACCOUNT_9D7E2A0F>."
}
```

Response:
```json
{ "narrative": "Funds sent to Alice Brown via CPTY-ACC-001." }
```

### `get_map` (audit trail)

```json
{ "action": "get_map", "case_id": "CASE-2026-007" }
```

### `delete_map` (post-approval cleanup)

```json
{ "action": "delete_map", "case_id": "CASE-2026-007" }
```

---

## Running tests

```bash
pip install psycopg2-binary pytest
pytest tests/test_masking_agent.py -v
# 87 tests, no real RDS connection needed
```
