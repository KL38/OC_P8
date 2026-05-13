-- Étape 4 follow-up — consolidate all latency measurements to INTEGER
-- milliseconds and add a dedicated ``db_log_ms`` column for the INSERT
-- wall-clock cost.
--
-- Rationale (see plan / report):
-- * The previous DOUBLE PRECISION choice was overkill — we display in ms,
--   we communicate in ms, sub-ms precision was misleading.
-- * The negative ``overhead`` values surfaced on the dashboard were caused
--   by Python-side ``int()`` truncation; the fix is to switch to ``round()``
--   at insert time (already done in api/main.py + api/logger.py) and to
--   actually measure the INSERT itself, which was previously invisible.
-- * ``db_log_ms`` is filled by a follow-up UPDATE in api.logger.log_prediction;
--   nullable so legacy rows and error rows stay valid.
--
-- Postgres rounds DOUBLE PRECISION → INTEGER half-to-even on cast, which is
-- consistent with Python's ``round()``. Idempotent (TYPE INTEGER on an
-- already-INTEGER column is a no-op; ADD COLUMN IF NOT EXISTS guards the
-- new column).

ALTER TABLE predictions_log
    ALTER COLUMN latency_ms          TYPE INTEGER USING ROUND(latency_ms)::INTEGER,
    ALTER COLUMN feature_assembly_ms TYPE INTEGER USING ROUND(feature_assembly_ms)::INTEGER,
    ALTER COLUMN inference_ms        TYPE INTEGER USING ROUND(inference_ms)::INTEGER,
    ALTER COLUMN inference_cpu_ms    TYPE INTEGER USING ROUND(inference_cpu_ms)::INTEGER,
    ADD COLUMN  IF NOT EXISTS db_log_ms INTEGER;

ALTER TABLE predictions_log_test
    ALTER COLUMN latency_ms          TYPE INTEGER USING ROUND(latency_ms)::INTEGER,
    ALTER COLUMN feature_assembly_ms TYPE INTEGER USING ROUND(feature_assembly_ms)::INTEGER,
    ALTER COLUMN inference_ms        TYPE INTEGER USING ROUND(inference_ms)::INTEGER,
    ALTER COLUMN inference_cpu_ms    TYPE INTEGER USING ROUND(inference_cpu_ms)::INTEGER,
    ADD COLUMN  IF NOT EXISTS db_log_ms INTEGER;
