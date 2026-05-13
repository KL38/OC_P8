-- Étape 4 follow-up — promote `latency_ms` from INTEGER to DOUBLE PRECISION.
--
-- The Python handler measures wall-clock via ``time.perf_counter()`` which
-- has sub-millisecond resolution. Storing it as an integer truncated the
-- decimal part (e.g. 12.7 ms → 12), and on fast request paths that
-- truncation could exceed the genuine overhead between the timed
-- sub-steps, yielding spurious negative values for the computed
-- ``overhead = latency_ms - feature_assembly_ms - inference_ms`` in the
-- dashboard.
--
-- Postgres auto-casts existing integer values to DOUBLE PRECISION; legacy
-- rows keep their value but lose nothing (they were ints already).
--
-- Run once via Supabase SQL editor. Idempotent — ALTER COLUMN TYPE is a
-- no-op if the column already matches.

ALTER TABLE predictions_log
    ALTER COLUMN latency_ms TYPE DOUBLE PRECISION;

ALTER TABLE predictions_log_test
    ALTER COLUMN latency_ms TYPE DOUBLE PRECISION;
