-- =============================================================================
-- Migration 001 — create pii_mask_map
-- Run once against your Amazon RDS (PostgreSQL) instance.
-- All statements are idempotent — safe to re-run.
-- =============================================================================

CREATE TABLE IF NOT EXISTS pii_mask_map (
    id          SERIAL       PRIMARY KEY,

    -- SAR case identifier — matches AgentState.case_id
    -- Partition key: all reads/deletes filter by this column.
    case_id     TEXT         NOT NULL,

    -- Opaque token shown to the LLM, e.g.  <NAME_4A3F1B2C>
    -- Globally unique across ALL cases.
    token       TEXT         NOT NULL UNIQUE,

    -- PII category label: NAME | ACCOUNT
    -- (NAME  = counterparty_name,  ACCOUNT = counterparty_account)
    entity_type TEXT         NOT NULL,

    -- Original PII value — NEVER sent to the LLM.
    real_value  TEXT         NOT NULL,

    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Same (case_id, real_value) always maps to the same token.
    CONSTRAINT uq_case_value UNIQUE (case_id, real_value)
);

-- Fast case-level lookup (mask / unmask / get_map / delete_map)
CREATE INDEX IF NOT EXISTS idx_pii_case_id ON pii_mask_map (case_id);

-- Fast token lookup (unmask individual tokens if ever needed)
CREATE INDEX IF NOT EXISTS idx_pii_token   ON pii_mask_map (token);

COMMENT ON TABLE  pii_mask_map IS
  'AI-Privacy Guard — token ↔ real PII map per SAR case. '
  'Only counterparty_name and counterparty_account from '
  'AgentState.structured_case.transactions[] are stored here.';

COMMENT ON COLUMN pii_mask_map.case_id     IS 'AgentState.case_id, e.g. CASE-2026-007';
COMMENT ON COLUMN pii_mask_map.token       IS 'Opaque token shown to LLM, e.g. <NAME_4A3F1B2C>';
COMMENT ON COLUMN pii_mask_map.entity_type IS 'NAME or ACCOUNT';
COMMENT ON COLUMN pii_mask_map.real_value  IS 'Original PII — never sent to LLM';
