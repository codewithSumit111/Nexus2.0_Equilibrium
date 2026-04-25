"""
tests/test_masking_agent.py
===========================
Unit tests for masking_agent.py and lambda_handler._validate_state.

All fixtures match the EXACT AgentState structured_case schema
from the state definition document:

  customer   : risk_rating, customer_type, nationality, pep_flag,
               previous_sar_count  — NO PII fields
  accounts[] : account_id, account_type, account_age_days,
               international_txn   — NO PII fields
  transactions[]: txn_id, txn_date, txn_type, amount, currency,
                  channel, counterparty_name*, counterparty_account*,
                  counterparty_country, is_high_value, velocity_score
                  (* = PII, masked)

Run:
    pip install pytest psycopg2-binary
    pytest tests/test_masking_agent.py -v
"""

from __future__ import annotations

import os
import re
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal psycopg2 stub — no real driver needed
# ---------------------------------------------------------------------------

def _build_psycopg2_stub() -> types.ModuleType:
    stub = types.ModuleType("psycopg2")

    extras = types.ModuleType("psycopg2.extras")
    class DictCursor: pass
    extras.DictCursor = DictCursor
    stub.extras = extras

    extensions = types.ModuleType("psycopg2.extensions")
    class connection: pass
    extensions.connection = connection
    stub.extensions = extensions

    errors_mod = types.ModuleType("psycopg2.errors")
    class UniqueViolation(Exception): pass
    errors_mod.UniqueViolation = UniqueViolation
    stub.errors = errors_mod

    class OperationalError(Exception): pass
    stub.OperationalError = OperationalError
    stub.Error = Exception

    return stub


_stub = _build_psycopg2_stub()
sys.modules["psycopg2"]            = _stub
sys.modules["psycopg2.extras"]     = _stub.extras
sys.modules["psycopg2.extensions"] = _stub.extensions
sys.modules["psycopg2.errors"]     = _stub.errors

from masking_agent import (          # noqa: E402
    ACCOUNT_PII_FIELDS,
    CUSTOMER_PII_FIELDS,
    PII_PATTERNS,
    TRANSACTION_PII_FIELDS,
    _make_token,
    delete_mask_map,
    get_mask_map,
    mask_structured_case,
    mask_text,
    unmask_sar_narrative,
)


# ---------------------------------------------------------------------------
# FakeDB — in-memory pii_mask_map simulator
# ---------------------------------------------------------------------------

class FakeDB:
    def __init__(self) -> None:
        # (case_id, real_value) → row
        self._by_cv:  dict[tuple[str, str], dict] = {}
        # token → row
        self._by_tok: dict[str, dict] = {}

    def make_conn(self) -> MagicMock:
        conn = MagicMock()
        conn.closed   = False
        conn.cursor.side_effect = self._cursor_factory
        conn.commit   = MagicMock()
        conn.rollback = MagicMock()
        return conn

    def rows_for_case(self, case_id: str) -> list[dict]:
        return [r for r in self._by_cv.values() if r["case_id"] == case_id]

    def _cursor_factory(self, cursor_factory=None) -> MagicMock:
        ctx = MagicMock()
        ctx.__enter__ = lambda s: s
        ctx.__exit__  = MagicMock(return_value=False)
        ctx._result: list[dict] = []
        ctx.rowcount = 0
        ctx.execute  = MagicMock(
            side_effect=lambda sql, params=None: self._execute(ctx, sql, params or ())
        )
        ctx.fetchone = MagicMock(side_effect=lambda: ctx._result[0] if ctx._result else None)
        ctx.fetchall = MagicMock(side_effect=lambda: list(ctx._result))
        return ctx

    def _execute(self, ctx: MagicMock, sql: str, params: tuple) -> None:
        s = " ".join(sql.split()).upper()

        # SELECT token WHERE case_id AND real_value
        if ("SELECT TOKEN FROM PII_MASK_MAP WHERE CASE_ID" in s
                and "REAL_VALUE" in s):
            case_id, real_value = params
            row = self._by_cv.get((case_id, real_value))
            ctx._result = [row] if row else []

        # SELECT token, real_value WHERE case_id  (unmask bulk)
        elif "SELECT TOKEN, REAL_VALUE FROM PII_MASK_MAP WHERE CASE_ID" in s:
            ctx._result = self.rows_for_case(params[0])

        # SELECT token, entity_type, real_value, created_at  (get_mask_map)
        elif "SELECT TOKEN, ENTITY_TYPE, REAL_VALUE, CREATED_AT" in s:
            ctx._result = sorted(
                self.rows_for_case(params[0]),
                key=lambda r: r.get("created_at", "")
            )

        # INSERT ... ON CONFLICT DO NOTHING RETURNING token
        elif "INSERT INTO PII_MASK_MAP" in s and "ON CONFLICT" in s:
            case_id, token, entity_type, real_value = params
            key = (case_id, real_value)
            if key in self._by_cv:          # already exists → DO NOTHING
                ctx._result  = []
                ctx.rowcount = 0
                return
            if token in self._by_tok:       # token collision → UniqueViolation
                raise _stub.errors.UniqueViolation("token collision")
            row = {"case_id": case_id, "token": token,
                   "entity_type": entity_type, "real_value": real_value,
                   "created_at": "2026-01-01T00:00:00+00:00"}
            self._by_cv[key]    = row
            self._by_tok[token] = row
            ctx._result  = [row]
            ctx.rowcount = 1

        # DELETE WHERE case_id RETURNING id
        elif "DELETE FROM PII_MASK_MAP WHERE CASE_ID" in s:
            case_id = params[0]
            keys = [k for k, v in self._by_cv.items() if v["case_id"] == case_id]
            for k in keys:
                row = self._by_cv.pop(k)
                self._by_tok.pop(row["token"], None)
            ctx.rowcount = len(keys)

        elif s.strip() == "SELECT 1":
            ctx._result = [{"?column?": 1}]


# ---------------------------------------------------------------------------
# Canonical structured_case fixture
# Matches the exact AgentState schema from the state definition document.
# txn_type values: CREDIT|DEBIT|WIRE_OUT|WIRE_IN|INTERNAL_TRANSFER
# channel values:  NEFT|RTGS|IMPS|CASH|SWIFT|BRANCH
# ---------------------------------------------------------------------------

SAMPLE_SC: dict[str, Any] = {
    "customer": {
        "risk_rating":        "HIGH",       # str "LOW"|"MEDIUM"|"HIGH"
        "customer_type":      "individual", # str "individual"|"corporate"
        "nationality":        "IN",         # str ISO-2
        "pep_flag":           False,        # bool
        "previous_sar_count": 2,            # int
    },
    "accounts": [
        {
            "account_id":        "ACC-001",  # str — internal system ID
            "account_type":      "current",  # str "savings"|"current"|"wallet"
            "account_age_days":  365,        # int
            "international_txn": True,       # bool
        },
    ],
    "transactions": [
        {
            "txn_id":               "TXN-001",           # str — NOT PII
            "txn_date":             "2026-01-15T10:00:00",
            "txn_type":             "WIRE_OUT",
            "amount":               500000.0,
            "currency":             "INR",
            "channel":              "SWIFT",
            "counterparty_name":    "Alice Brown",        # PII ← masked
            "counterparty_account": "CPTY-ACC-001",       # PII ← masked
            "counterparty_country": "AE",                 # ISO-2 — NOT PII
            "is_high_value":        True,
            "velocity_score":       0.85,
        },
        {
            "txn_id":               "TXN-002",
            "txn_date":             "2026-01-16T14:30:00",
            "txn_type":             "CREDIT",
            "amount":               250000.0,
            "currency":             "INR",
            "channel":              "NEFT",
            "counterparty_name":    "Bob Jones",           # PII ← masked
            "counterparty_account": "CPTY-ACC-002",        # PII ← masked
            "counterparty_country": "US",
            "is_high_value":        False,
            "velocity_score":       0.40,
        },
    ],
}


def _mask(db: FakeDB, sc=None, case_id="CASE-001"):
    import copy
    return mask_structured_case(
        copy.deepcopy(sc or SAMPLE_SC), case_id, db.make_conn()
    )


# ===========================================================================
# TestPIIFieldRegistries
# ===========================================================================

class TestPIIFieldRegistries(unittest.TestCase):

    def test_customer_pii_fields_is_empty(self):
        """customer has no PII per AgentState schema."""
        self.assertEqual(CUSTOMER_PII_FIELDS, {})

    def test_account_pii_fields_is_empty(self):
        """accounts has no PII per AgentState schema."""
        self.assertEqual(ACCOUNT_PII_FIELDS, {})

    def test_transaction_pii_has_exactly_two_fields(self):
        self.assertEqual(len(TRANSACTION_PII_FIELDS), 2)

    def test_counterparty_name_is_NAME(self):
        self.assertEqual(TRANSACTION_PII_FIELDS["counterparty_name"], "NAME")

    def test_counterparty_account_is_ACCOUNT(self):
        self.assertEqual(TRANSACTION_PII_FIELDS["counterparty_account"], "ACCOUNT")

    def test_txn_id_not_pii(self):
        self.assertNotIn("txn_id", TRANSACTION_PII_FIELDS)

    def test_account_id_not_pii(self):
        self.assertNotIn("account_id", ACCOUNT_PII_FIELDS)

    def test_counterparty_country_not_pii(self):
        """counterparty_country is ISO-2 — must not be masked (Agent 3 needs it)."""
        self.assertNotIn("counterparty_country", TRANSACTION_PII_FIELDS)


# ===========================================================================
# TestMaskTransactionPII
# ===========================================================================

class TestMaskTransactionPII(unittest.TestCase):

    def setUp(self):
        self.db = FakeDB()

    def test_counterparty_name_replaced_by_token(self):
        result = _mask(self.db)
        self.assertNotEqual(result["transactions"][0]["counterparty_name"], "Alice Brown")
        self.assertRegex(result["transactions"][0]["counterparty_name"], r"^<NAME_[0-9A-F]{8}>$")

    def test_counterparty_account_replaced_by_token(self):
        result = _mask(self.db)
        self.assertNotEqual(result["transactions"][0]["counterparty_account"], "CPTY-ACC-001")
        self.assertRegex(result["transactions"][0]["counterparty_account"], r"^<ACCOUNT_[0-9A-F]{8}>$")

    def test_all_transactions_both_pii_fields_masked(self):
        result = _mask(self.db)
        for i, txn in enumerate(result["transactions"]):
            with self.subTest(txn=i):
                self.assertRegex(txn["counterparty_name"],    r"^<NAME_[0-9A-F]{8}>$")
                self.assertRegex(txn["counterparty_account"], r"^<ACCOUNT_[0-9A-F]{8}>$")

    def test_different_counterparties_different_name_tokens(self):
        result = _mask(self.db)
        t0 = result["transactions"][0]["counterparty_name"]
        t1 = result["transactions"][1]["counterparty_name"]
        self.assertNotEqual(t0, t1)   # "Alice Brown" ≠ "Bob Jones"

    def test_same_counterparty_name_same_token_within_case(self):
        import copy
        sc = copy.deepcopy(SAMPLE_SC)
        sc["transactions"][1]["counterparty_name"] = "Alice Brown"
        conn = self.db.make_conn()
        result = mask_structured_case(sc, "CASE-DET", conn)
        self.assertEqual(
            result["transactions"][0]["counterparty_name"],
            result["transactions"][1]["counterparty_name"],
        )

    def test_same_value_different_cases_different_tokens(self):
        import copy
        db2 = FakeDB()
        r1 = mask_structured_case(copy.deepcopy(SAMPLE_SC), "CASE-A", self.db.make_conn())
        r2 = mask_structured_case(copy.deepcopy(SAMPLE_SC), "CASE-B", db2.make_conn())
        self.assertNotEqual(
            r1["transactions"][0]["counterparty_name"],
            r2["transactions"][0]["counterparty_name"],
        )


# ===========================================================================
# TestCustomerFieldsUntouched
# ===========================================================================

class TestCustomerFieldsUntouched(unittest.TestCase):
    """Every customer field must be byte-for-byte identical after masking."""

    def setUp(self):
        self.result = _mask(FakeDB())

    def test_risk_rating(self):
        self.assertEqual(self.result["customer"]["risk_rating"], "HIGH")

    def test_customer_type(self):
        self.assertEqual(self.result["customer"]["customer_type"], "individual")

    def test_nationality(self):
        self.assertEqual(self.result["customer"]["nationality"], "IN")

    def test_pep_flag(self):
        self.assertIs(self.result["customer"]["pep_flag"], False)

    def test_previous_sar_count(self):
        self.assertEqual(self.result["customer"]["previous_sar_count"], 2)

    def test_no_extra_keys_added(self):
        self.assertEqual(
            set(self.result["customer"].keys()),
            set(SAMPLE_SC["customer"].keys()),
        )


# ===========================================================================
# TestAccountFieldsUntouched
# ===========================================================================

class TestAccountFieldsUntouched(unittest.TestCase):
    """Every account field must be byte-for-byte identical after masking."""

    def setUp(self):
        self.result = _mask(FakeDB())

    def test_account_id_unchanged(self):
        """account_id is an internal system ID — must NOT be tokenised."""
        self.assertEqual(self.result["accounts"][0]["account_id"], "ACC-001")

    def test_account_type_unchanged(self):
        self.assertEqual(self.result["accounts"][0]["account_type"], "current")

    def test_account_age_days_unchanged(self):
        self.assertEqual(self.result["accounts"][0]["account_age_days"], 365)

    def test_international_txn_unchanged(self):
        self.assertIs(self.result["accounts"][0]["international_txn"], True)

    def test_accounts_list_length_unchanged(self):
        self.assertEqual(len(self.result["accounts"]), 1)


# ===========================================================================
# TestTransactionOperationalFieldsUntouched
# ===========================================================================

class TestTransactionOperationalFieldsUntouched(unittest.TestCase):
    """
    All transaction fields EXCEPT counterparty_name / counterparty_account
    must be byte-for-byte identical after masking.
    Especially: txn_id, txn_type (WIRE_OUT, CREDIT etc.), channel (SWIFT,
    NEFT etc.), counterparty_country — Agent 3 depends on these.
    """

    def setUp(self):
        self.result = _mask(FakeDB())

    def test_txn_id_txn0(self):
        self.assertEqual(self.result["transactions"][0]["txn_id"], "TXN-001")

    def test_txn_id_txn1(self):
        self.assertEqual(self.result["transactions"][1]["txn_id"], "TXN-002")

    def test_txn_date(self):
        self.assertEqual(self.result["transactions"][0]["txn_date"],
                         "2026-01-15T10:00:00")

    def test_txn_type_wire_out(self):
        self.assertEqual(self.result["transactions"][0]["txn_type"], "WIRE_OUT")

    def test_txn_type_credit(self):
        self.assertEqual(self.result["transactions"][1]["txn_type"], "CREDIT")

    def test_amount(self):
        self.assertAlmostEqual(self.result["transactions"][0]["amount"], 500000.0)

    def test_currency(self):
        self.assertEqual(self.result["transactions"][0]["currency"], "INR")

    def test_channel_swift(self):
        self.assertEqual(self.result["transactions"][0]["channel"], "SWIFT")

    def test_channel_neft(self):
        self.assertEqual(self.result["transactions"][1]["channel"], "NEFT")

    def test_counterparty_country_ae(self):
        """counterparty_country MUST NOT be masked — Agent 3 FATF logic needs it."""
        self.assertEqual(self.result["transactions"][0]["counterparty_country"], "AE")

    def test_counterparty_country_us(self):
        self.assertEqual(self.result["transactions"][1]["counterparty_country"], "US")

    def test_is_high_value(self):
        self.assertIs(self.result["transactions"][0]["is_high_value"], True)

    def test_velocity_score(self):
        self.assertAlmostEqual(self.result["transactions"][0]["velocity_score"], 0.85)


# ===========================================================================
# TestMaskInputSafety
# ===========================================================================

class TestMaskInputSafety(unittest.TestCase):

    def test_original_dict_not_mutated(self):
        import copy
        sc = copy.deepcopy(SAMPLE_SC)
        original_name = sc["transactions"][0]["counterparty_name"]
        mask_structured_case(sc, "CASE-MUT", FakeDB().make_conn())
        self.assertEqual(sc["transactions"][0]["counterparty_name"], original_name)

    def test_empty_string_counterparty_not_masked(self):
        import copy
        sc = copy.deepcopy(SAMPLE_SC)
        sc["transactions"][0]["counterparty_name"] = ""
        result = mask_structured_case(sc, "CASE-EMPTY-STR", FakeDB().make_conn())
        self.assertEqual(result["transactions"][0]["counterparty_name"], "")

    def test_empty_transactions_list_ok(self):
        sc = {
            "customer": {"risk_rating": "LOW", "customer_type": "individual",
                         "nationality": "IN", "pep_flag": False,
                         "previous_sar_count": 0},
            "accounts":     [],
            "transactions": [],
        }
        result = mask_structured_case(sc, "CASE-EMPTY-TXN", FakeDB().make_conn())
        self.assertEqual(result["transactions"], [])

    def test_txn_missing_counterparty_fields_no_crash(self):
        """Transactions without counterparty fields must not raise."""
        sc = {
            "customer":     {"risk_rating": "LOW", "customer_type": "individual",
                             "nationality": "IN", "pep_flag": False,
                             "previous_sar_count": 0},
            "accounts":     [],
            "transactions": [
                {"txn_id": "TXN-X", "txn_date": "2026-01-01",
                 "txn_type": "CREDIT", "amount": 1000.0,
                 "currency": "INR", "channel": "NEFT",
                 "counterparty_country": "IN",
                 "is_high_value": False, "velocity_score": 0.1},
            ],
        }
        result = mask_structured_case(sc, "CASE-NO-CPTY", FakeDB().make_conn())
        self.assertEqual(result["transactions"][0]["txn_id"], "TXN-X")

    def test_missing_transactions_key_no_crash(self):
        sc = {"customer": {"risk_rating": "LOW"}, "accounts": []}
        result = mask_structured_case(sc, "CASE-NO-TXN-KEY", FakeDB().make_conn())
        self.assertEqual(result.get("transactions", []), [])


# ===========================================================================
# TestUnmaskSarNarrative
# ===========================================================================

class TestUnmaskSarNarrative(unittest.TestCase):

    def _setup_masked(self, case_id="CASE-UN"):
        import copy
        db = FakeDB()
        conn = db.make_conn()
        masked = mask_structured_case(copy.deepcopy(SAMPLE_SC), case_id, conn)
        return masked, conn

    def test_roundtrip_string(self):
        masked, conn = self._setup_masked("CASE-RT")
        name_tok = masked["transactions"][0]["counterparty_name"]
        acc_tok  = masked["transactions"][0]["counterparty_account"]

        narrative = f"Funds sent to {name_tok} via {acc_tok}."
        restored  = unmask_sar_narrative(narrative, "CASE-RT", conn)

        self.assertIn("Alice Brown",  restored)
        self.assertIn("CPTY-ACC-001", restored)
        self.assertNotIn(name_tok, restored)
        self.assertNotIn(acc_tok,  restored)

    def test_roundtrip_dict(self):
        masked, conn = self._setup_masked("CASE-DICT")
        name_tok = masked["transactions"][0]["counterparty_name"]

        nar_dict = {
            "subject":   "SAR Report",
            "narrative": f"The subject transferred to {name_tok}.",
            "items":     [f"Counterparty: {name_tok}", "Amount: 500000"],
        }
        restored = unmask_sar_narrative(nar_dict, "CASE-DICT", conn)
        self.assertIn("Alice Brown", restored["narrative"])
        self.assertIn("Alice Brown", restored["items"][0])
        self.assertEqual(restored["items"][1], "Amount: 500000")
        self.assertNotIn(name_tok, restored["narrative"])

    def test_all_four_tokens_restored(self):
        masked, conn = self._setup_masked("CASE-ALL4")
        narrative = " ".join([
            masked["transactions"][0]["counterparty_name"],
            masked["transactions"][0]["counterparty_account"],
            masked["transactions"][1]["counterparty_name"],
            masked["transactions"][1]["counterparty_account"],
        ])
        restored = unmask_sar_narrative(narrative, "CASE-ALL4", conn)
        self.assertIn("Alice Brown",   restored)
        self.assertIn("CPTY-ACC-001",  restored)
        self.assertIn("Bob Jones",     restored)
        self.assertIn("CPTY-ACC-002",  restored)

    def test_no_mappings_returns_original(self):
        conn = FakeDB().make_conn()
        text = "No tokens here."
        self.assertEqual(unmask_sar_narrative(text, "MISSING-CASE", conn), text)

    def test_tokens_from_case_x_not_unmasked_in_case_y(self):
        import copy
        db   = FakeDB()
        conn = db.make_conn()
        masked = mask_structured_case(copy.deepcopy(SAMPLE_SC), "CASE-X", conn)
        tok = masked["transactions"][0]["counterparty_name"]
        # Unmask with different case_id — token must survive unchanged
        result = unmask_sar_narrative(tok, "CASE-Y", conn)
        self.assertEqual(result, tok)


# ===========================================================================
# TestMaskText (free-text regex path)
# ===========================================================================

class TestMaskText(unittest.TestCase):

    def setUp(self):
        self.db   = FakeDB()
        self.conn = self.db.make_conn()

    def test_masks_email(self):
        result = mask_text("Report to alice@bank.com please.", "CASE-TXT", self.conn)
        self.assertNotIn("alice@bank.com", result)
        self.assertRegex(result, r"<EMAIL_[0-9A-F]{8}>")

    def test_masks_ssn(self):
        result = mask_text("SSN: 123-45-6789 on file.", "CASE-TXT", self.conn)
        self.assertNotIn("123-45-6789", result)
        self.assertRegex(result, r"<SSN_[0-9A-F]{8}>")

    def test_masks_name(self):
        result = mask_text("Subject is John Doe.", "CASE-TXT", self.conn)
        self.assertNotIn("John Doe", result)

    def test_no_pii_unchanged(self):
        text = "WIRE_OUT via SWIFT channel for INR 500000 on 2026-01-15."
        self.assertEqual(mask_text(text, "CASE-NOPII", self.conn), text)

    def test_empty_string_unchanged(self):
        self.assertEqual(mask_text("", "CASE-EMPTY", self.conn), "")

    def test_determinism(self):
        t1 = mask_text("Contact bob@test.com.", "CASE-DET", self.conn)
        t2 = mask_text("Contact bob@test.com.", "CASE-DET", self.conn)
        tok1 = re.search(r"<EMAIL_[0-9A-F]{8}>", t1).group()
        tok2 = re.search(r"<EMAIL_[0-9A-F]{8}>", t2).group()
        self.assertEqual(tok1, tok2)


# ===========================================================================
# TestAuditHelpers
# ===========================================================================

class TestAuditHelpers(unittest.TestCase):

    def setUp(self):
        self.db   = FakeDB()
        self.conn = self.db.make_conn()

    def test_get_mask_map_returns_four_entries(self):
        """2 transactions × 2 PII fields = 4 tokens."""
        import copy
        mask_structured_case(copy.deepcopy(SAMPLE_SC), "CASE-AUDIT", self.conn)
        entries = get_mask_map("CASE-AUDIT", self.conn)
        self.assertEqual(len(entries), 4)
        for e in entries:
            self.assertIn("token",       e)
            self.assertIn("real_value",  e)
            self.assertIn("entity_type", e)

    def test_get_mask_map_empty_unknown_case(self):
        self.assertEqual(get_mask_map("UNKNOWN", self.conn), [])

    def test_delete_mask_map_removes_all(self):
        import copy
        mask_structured_case(copy.deepcopy(SAMPLE_SC), "CASE-DEL", self.conn)
        before = get_mask_map("CASE-DEL", self.conn)
        self.assertGreater(len(before), 0)

        deleted = delete_mask_map("CASE-DEL", self.conn)
        self.assertEqual(deleted, len(before))
        self.assertEqual(get_mask_map("CASE-DEL", self.conn), [])

    def test_delete_nonexistent_returns_zero(self):
        self.assertEqual(delete_mask_map("GHOST", self.conn), 0)


# ===========================================================================
# TestPIIPatterns (regex)
# ===========================================================================

class TestPIIPatterns(unittest.TestCase):

    def _match(self, label: str, text: str) -> bool:
        p = next(pat for lbl, pat in PII_PATTERNS if lbl == label)
        return bool(p.search(text))

    def test_ssn(self):          self.assertTrue(self._match("SSN",     "123-45-6789"))
    def test_email(self):        self.assertTrue(self._match("EMAIL",   "user@example.com"))
    def test_phone_intl(self):   self.assertTrue(self._match("PHONE",   "+1-800-555-0199"))
    def test_phone_local(self):  self.assertTrue(self._match("PHONE",   "020-1234-5678"))
    def test_address(self):      self.assertTrue(self._match("ADDRESS", "42 Baker Street, NY"))
    def test_name_bigram(self):  self.assertTrue(self._match("NAME",    "John Doe visited."))
    def test_name_trigram(self): self.assertTrue(self._match("NAME",    "Mary Jane Watson."))
    def test_ssn_no_fp(self):    self.assertFalse(self._match("SSN",    "no ssn here"))


# ===========================================================================
# TestMakeToken
# ===========================================================================

class TestMakeToken(unittest.TestCase):

    def test_format(self):
        tok = _make_token("NAME")
        self.assertRegex(tok, r"^<NAME_[0-9A-F]{8}>$")

    def test_uniqueness(self):
        tokens = {_make_token("ACCOUNT") for _ in range(200)}
        self.assertEqual(len(tokens), 200)


# ===========================================================================
# TestValidateState  (lambda_handler._validate_state)
# ===========================================================================

# Patch env vars before importing lambda_handler
os.environ.setdefault("DB_HOST",     "localhost")
os.environ.setdefault("DB_PORT",     "5432")
os.environ.setdefault("DB_NAME",     "sardb")
os.environ.setdefault("DB_USER",     "user")
os.environ.setdefault("DB_PASSWORD", "password")

import unittest.mock as _mock
with _mock.patch("masking_agent.get_connection", return_value=MagicMock()), \
     _mock.patch("masking_agent.ensure_table"):
    from lambda_handler import _validate_state, VALID_TYPOLOGIES  # noqa: E402


def _good_state(**overrides) -> dict:
    """Fully valid AgentState matching the state definition contract."""
    base = {
        # Call-1 inputs
        "case_id":          "CASE-2026-007",
        "s3_bucket":        "my-sar-bucket",
        "s3_prefix":        "cases/CASE-2026-007/inputs",
        "transactions_csv": "",
        # Call-1 outputs (from Agent 1)
        "sar_worthy":       True,
        "confidence_score": 0.91,
        "typology":         "Structuring",
        "risk_score":       0.78,
        # Call-2 input (from backend → RDS)
        "structured_case": {
            "customer": {
                "risk_rating":        "HIGH",
                "customer_type":      "individual",
                "nationality":        "IN",
                "pep_flag":           False,
                "previous_sar_count": 2,
            },
            "accounts": [
                {"account_id": "ACC-001", "account_type": "current",
                 "account_age_days": 365, "international_txn": True},
            ],
            "transactions": [
                {"txn_id": "TXN-001", "txn_date": "2026-01-15T10:00:00",
                 "txn_type": "WIRE_OUT", "amount": 500000.0, "currency": "INR",
                 "channel": "SWIFT", "counterparty_name": "Alice Brown",
                 "counterparty_account": "CPTY-ACC-001",
                 "counterparty_country": "AE",
                 "is_high_value": True, "velocity_score": 0.85},
            ],
        },
    }
    base.update(overrides)
    return base


class TestValidateState(unittest.TestCase):

    # ── happy path ────────────────────────────────────────────────────────────

    def test_valid_state_passes(self):
        self.assertIsNone(_validate_state(_good_state()))

    def test_all_six_typologies_valid(self):
        for t in VALID_TYPOLOGIES:
            with self.subTest(typology=t):
                self.assertIsNone(_validate_state(_good_state(typology=t)))

    def test_s3_bucket_empty_string_ok(self):
        """Empty s3_bucket = dev fallback via transactions_csv — allowed."""
        self.assertIsNone(_validate_state(_good_state(s3_bucket="")))

    def test_transactions_csv_empty_string_ok(self):
        self.assertIsNone(_validate_state(_good_state(transactions_csv="")))

    def test_empty_accounts_list_ok(self):
        sc = {**_good_state()["structured_case"], "accounts": []}
        self.assertIsNone(_validate_state(_good_state(structured_case=sc)))

    def test_empty_transactions_list_ok(self):
        sc = {**_good_state()["structured_case"], "transactions": []}
        self.assertIsNone(_validate_state(_good_state(structured_case=sc)))

    def test_confidence_score_boundary_zero(self):
        self.assertIsNone(_validate_state(_good_state(confidence_score=0.0)))

    def test_confidence_score_boundary_one(self):
        self.assertIsNone(_validate_state(_good_state(confidence_score=1.0)))

    # ── sar_worthy gate ───────────────────────────────────────────────────────

    def test_sar_worthy_false_rejected(self):
        err = _validate_state(_good_state(sar_worthy=False))
        self.assertIsNotNone(err)
        self.assertIn("sar_worthy", err)

    def test_sar_worthy_int_1_rejected(self):
        """int 1 is not bool True."""
        err = _validate_state(_good_state(sar_worthy=1))
        self.assertIsNotNone(err)
        self.assertIn("sar_worthy", err)

    def test_sar_worthy_missing_rejected(self):
        state = _good_state()
        del state["sar_worthy"]
        self.assertIsNotNone(_validate_state(state))

    # ── case_id ───────────────────────────────────────────────────────────────

    def test_empty_case_id_rejected(self):
        err = _validate_state(_good_state(case_id=""))
        self.assertIsNotNone(err)
        self.assertIn("case_id", err)

    def test_missing_case_id_rejected(self):
        state = _good_state()
        del state["case_id"]
        self.assertIsNotNone(_validate_state(state))

    # ── Call-1 input strings ──────────────────────────────────────────────────

    def test_s3_bucket_none_rejected(self):
        err = _validate_state(_good_state(s3_bucket=None))
        self.assertIsNotNone(err)
        self.assertIn("s3_bucket", err)

    def test_s3_prefix_int_rejected(self):
        err = _validate_state(_good_state(s3_prefix=123))
        self.assertIsNotNone(err)
        self.assertIn("s3_prefix", err)

    def test_transactions_csv_bool_rejected(self):
        err = _validate_state(_good_state(transactions_csv=False))
        self.assertIsNotNone(err)
        self.assertIn("transactions_csv", err)

    # ── confidence_score ──────────────────────────────────────────────────────

    def test_confidence_score_above_1_rejected(self):
        err = _validate_state(_good_state(confidence_score=1.01))
        self.assertIsNotNone(err)
        self.assertIn("confidence_score", err)

    def test_confidence_score_negative_rejected(self):
        self.assertIsNotNone(_validate_state(_good_state(confidence_score=-0.01)))

    # ── typology ──────────────────────────────────────────────────────────────

    def test_unknown_typology_rejected(self):
        err = _validate_state(_good_state(typology="Ponzi Scheme"))
        self.assertIsNotNone(err)
        self.assertIn("typology", err)

    def test_typology_wrong_case_rejected(self):
        """Typology matching is case-sensitive."""
        self.assertIsNotNone(_validate_state(_good_state(typology="structuring")))

    # ── risk_score ────────────────────────────────────────────────────────────

    def test_risk_score_above_1_rejected(self):
        err = _validate_state(_good_state(risk_score=1.5))
        self.assertIsNotNone(err)
        self.assertIn("risk_score", err)

    # ── structured_case ───────────────────────────────────────────────────────

    def test_structured_case_missing_rejected(self):
        state = _good_state()
        del state["structured_case"]
        err = _validate_state(state)
        self.assertIsNotNone(err)
        self.assertIn("structured_case", err)

    def test_structured_case_empty_dict_rejected(self):
        self.assertIsNotNone(_validate_state(_good_state(structured_case={})))

    def test_customer_not_dict_rejected(self):
        sc = {**_good_state()["structured_case"], "customer": "bad"}
        err = _validate_state(_good_state(structured_case=sc))
        self.assertIsNotNone(err)
        self.assertIn("customer", err)

    def test_accounts_not_list_rejected(self):
        sc = {**_good_state()["structured_case"], "accounts": {}}
        err = _validate_state(_good_state(structured_case=sc))
        self.assertIsNotNone(err)
        self.assertIn("accounts", err)

    def test_transactions_not_list_rejected(self):
        sc = {**_good_state()["structured_case"], "transactions": "bad"}
        err = _validate_state(_good_state(structured_case=sc))
        self.assertIsNotNone(err)
        self.assertIn("transactions", err)


if __name__ == "__main__":
    unittest.main()
