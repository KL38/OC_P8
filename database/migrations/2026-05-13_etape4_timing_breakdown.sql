-- Étape 4 — fine-grained latency breakdown of /predict
--
-- Brings both production and test tables to the final étape-4 schema.
-- Idempotent: ALTER COLUMN to the existing type is a no-op, ADD COLUMN
-- IF NOT EXISTS guards new columns. Safe to re-run.
--
-- Final state (timing-related columns):
--   latency_ms          INTEGER NOT NULL  -- handler wall-clock (existing)
--   feature_assembly_ms INTEGER NULL      -- assemble() time
--   inference_ms        INTEGER NULL      -- predict_proba wall-clock
--   inference_cpu_ms    INTEGER NULL      -- predict_proba CPU time
--   plumbing_ms         INTEGER NULL      -- latency - assembly - inference
--   db_log_ms           INTEGER NULL      -- INSERT wall-clock (self-measured)
--
-- All timings are INTEGER milliseconds (the API rounds in Python via
-- round(), not int(), to avoid a systematic downward bias).

ALTER TABLE predictions_log
    ALTER COLUMN latency_ms          TYPE INTEGER USING ROUND(latency_ms)::INTEGER,
    ADD COLUMN  IF NOT EXISTS feature_assembly_ms INTEGER,
    ADD COLUMN  IF NOT EXISTS inference_ms        INTEGER,
    ADD COLUMN  IF NOT EXISTS inference_cpu_ms    INTEGER,
    ADD COLUMN  IF NOT EXISTS plumbing_ms         INTEGER,
    ADD COLUMN  IF NOT EXISTS db_log_ms           INTEGER;

-- If those timing columns were previously created as DOUBLE PRECISION (an
-- earlier draft of this migration), normalise them to INTEGER. The USING
-- clause rounds on the way in. If they're already INTEGER, this is a no-op.
ALTER TABLE predictions_log
    ALTER COLUMN feature_assembly_ms TYPE INTEGER USING ROUND(feature_assembly_ms)::INTEGER,
    ALTER COLUMN inference_ms        TYPE INTEGER USING ROUND(inference_ms)::INTEGER,
    ALTER COLUMN inference_cpu_ms    TYPE INTEGER USING ROUND(inference_cpu_ms)::INTEGER;

ALTER TABLE predictions_log_test
    ALTER COLUMN latency_ms          TYPE INTEGER USING ROUND(latency_ms)::INTEGER,
    ADD COLUMN  IF NOT EXISTS feature_assembly_ms INTEGER,
    ADD COLUMN  IF NOT EXISTS inference_ms        INTEGER,
    ADD COLUMN  IF NOT EXISTS inference_cpu_ms    INTEGER,
    ADD COLUMN  IF NOT EXISTS plumbing_ms         INTEGER,
    ADD COLUMN  IF NOT EXISTS db_log_ms           INTEGER;

ALTER TABLE predictions_log_test
    ALTER COLUMN feature_assembly_ms TYPE INTEGER USING ROUND(feature_assembly_ms)::INTEGER,
    ALTER COLUMN inference_ms        TYPE INTEGER USING ROUND(inference_ms)::INTEGER,
    ALTER COLUMN inference_cpu_ms    TYPE INTEGER USING ROUND(inference_cpu_ms)::INTEGER;
