-- Étape 4 follow-up — promote the SQL-computed ``overhead`` (= latency_ms
-- - feature_assembly_ms - inference_ms) to a first-class stored column
-- ``plumbing_ms``. Computed once in the API at log time so the dashboard
-- (and any external query) reads a self-describing column instead of
-- re-deriving the formula in every SELECT.
--
-- Nullable: legacy rows and error rows leave it NULL.
-- Idempotent: ADD COLUMN IF NOT EXISTS.

ALTER TABLE predictions_log
    ADD COLUMN IF NOT EXISTS plumbing_ms INTEGER;

ALTER TABLE predictions_log_test
    ADD COLUMN IF NOT EXISTS plumbing_ms INTEGER;
