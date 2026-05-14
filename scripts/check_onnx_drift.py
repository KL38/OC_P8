"""Quick numerical drift check: LightGBM joblib vs ONNX Runtime.

Compares positive-class probabilities on N rows of the reference dataset and
reports decision flips at the production threshold (read from model_info.json,
falling back to 0.5 for the rule-of-thumb sanity check).

Run:
    uv run python scripts/check_onnx_drift.py --n 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBLIB_PATH = ROOT / "models" / "model.joblib"
DEFAULT_ONNX_PATH = ROOT / "models" / "model.onnx"
DEFAULT_FEATURE_NAMES_PATH = ROOT / "models" / "feature_names.json"
DEFAULT_MODEL_INFO_PATH = ROOT / "models" / "model_info.json"
DEFAULT_REFERENCE_PATH = ROOT / "data" / "reference_dataset.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Number of rows")
    parser.add_argument("--joblib", type=Path, default=DEFAULT_JOBLIB_PATH)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument(
        "--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES_PATH
    )
    parser.add_argument("--model-info", type=Path, default=DEFAULT_MODEL_INFO_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_names = json.loads(args.feature_names.read_text())
    info = json.loads(args.model_info.read_text())
    threshold = float(info.get("metrics", {}).get("best_threshold_mean", 0.5))
    print(f"Production threshold (from model_info.json): {threshold}")

    # --- Sample N rows ---
    reference = pd.read_parquet(args.reference)
    n = min(args.n, len(reference))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(reference), size=n, replace=False)
    X_df = reference.iloc[idx][feature_names].reset_index(drop=True)
    X_np = X_df.to_numpy(dtype=np.float32)

    # --- LightGBM native (batch) ---
    print(f"Running LightGBM on {n} rows (batch)...")
    model = joblib.load(args.joblib)
    raw = model.get_raw_model() if hasattr(model, "get_raw_model") else model
    probas_lgbm = raw.predict_proba(X_df)[:, 1]

    # --- ONNX Runtime (batch) ---
    print(f"Running ONNX on {n} rows (batch)...")
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    proba_output = session.get_outputs()[1].name
    probas_onnx = session.run([proba_output], {input_name: X_np})[0][:, 1]

    # --- Drift on probabilities ---
    abs_delta = np.abs(probas_lgbm - probas_onnx)
    print()
    print("=== Numerical drift (probability) ===")
    print(f"max  |delta|    = {abs_delta.max():.2e}")
    print(f"mean |delta|    = {abs_delta.mean():.2e}")
    print(f"p95  |delta|    = {np.percentile(abs_delta, 95):.2e}")
    print(f"p99  |delta|    = {np.percentile(abs_delta, 99):.2e}")
    print(f"# rows > 1e-5   = {(abs_delta > 1e-5).sum()} / {n}")
    print(f"# rows > 1e-3   = {(abs_delta > 1e-3).sum()} / {n}")

    # --- Decision flips at production threshold ---
    preds_lgbm = (probas_lgbm >= threshold).astype(int)
    preds_onnx = (probas_onnx >= threshold).astype(int)
    flips_mask = preds_lgbm != preds_onnx
    n_flips = int(flips_mask.sum())
    print()
    print(f"=== Decision flips at threshold {threshold} ===")
    print(f"# flips         = {n_flips} / {n} ({100 * n_flips / n:.3f}%)")
    print(f"# REFUSED→GRANT = {int(((preds_lgbm == 1) & (preds_onnx == 0)).sum())}")
    print(f"# GRANT→REFUSED = {int(((preds_lgbm == 0) & (preds_onnx == 1)).sum())}")

    # --- Borderline band around threshold ---
    band = 0.005
    near_threshold = (probas_lgbm > threshold - band) & (probas_lgbm < threshold + band)
    print()
    print(
        f"=== Borderline band [{threshold - band:.3f}, {threshold + band:.3f}] ==="
    )
    print(f"# rows in band  = {int(near_threshold.sum())} / {n}")
    if near_threshold.any():
        flips_in_band = int((flips_mask & near_threshold).sum())
        print(f"# flips in band = {flips_in_band} (= what users actually feel)")

    # --- Sanity check at simple 0.5 (the request's default) ---
    preds_lgbm_05 = (probas_lgbm > 0.5).astype(int)
    preds_onnx_05 = (probas_onnx > 0.5).astype(int)
    print()
    print(f"=== Reference @ 0.5 (textbook sanity) ===")
    print(f"# flips @ 0.5   = {int((preds_lgbm_05 != preds_onnx_05).sum())} / {n}")


if __name__ == "__main__":
    main()
