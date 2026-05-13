-- Étape 4 — Fine-grained timing breakdown of /predict.
--
-- Adds three nullable columns capturing the wall-clock cost of feature
-- assembly and model inference, plus the CPU time spent in inference. The
-- existing `latency_ms` stays end-to-end; the new columns let us isolate the
-- actual bottleneck before targeted optimization.
--
-- Run once via the Supabase SQL editor (or psql against the pooler URL).
-- All columns are NULLABLE so legacy rows remain valid and error rows
-- (status_code != 200) can leave them NULL.
--
-- Idempotent: re-running is a no-op thanks to IF NOT EXISTS.

ALTER TABLE predictions_log
    ADD COLUMN IF NOT EXISTS feature_assembly_ms DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS inference_ms        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS inference_cpu_ms    DOUBLE PRECISION;

ALTER TABLE predictions_log_test
    ADD COLUMN IF NOT EXISTS feature_assembly_ms DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS inference_ms        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS inference_cpu_ms    DOUBLE PRECISION;
