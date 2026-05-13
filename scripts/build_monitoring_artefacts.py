"""Build monitoring artefacts that the dashboard reads at runtime.

Produces two JSON files in ``models/``:

- ``proba_reference.json``: histogram of ``probability_default`` for the
  reference (training) dataset, so the dashboard can overlay it on prod
  predictions to surface OUTPUT drift (the most direct early-warning
  signal for a credit scoring model).

- ``feature_importance.json``: SHAP-based feature importance (mean
  absolute SHAP value across a sample of reference rows), top-K features
  ranked. Used by the dashboard for two things: (1) a "critical features"
  panel showing drift status of the most influential inputs, and (2)
  weighting the global drift score by importance.

Re-run only when retraining the model or rebuilding the reference dataset.

Usage:
    uv run python scripts/build_monitoring_artefacts.py
    uv run python scripts/build_monitoring_artefacts.py --shap-sample 2000 --top-k 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("scripts.build_monitoring_artefacts")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_MODEL = Path("models/model.joblib")
DEFAULT_REFERENCE = Path("data/reference_dataset.parquet")
DEFAULT_FEATURE_NAMES = Path("models/feature_names.json")
# Write directly under dashboard/static/ so the deploy script bundles them
# into the monitoring HF Space without extra plumbing.
DEFAULT_PROBA_OUT = Path("dashboard/static/proba_reference.json")
DEFAULT_IMPORTANCE_OUT = Path("dashboard/static/feature_importance.json")
DEFAULT_SHAP_SAMPLE = 1000
DEFAULT_TOP_K = 20
DEFAULT_BINS = 40
DEFAULT_PROBA_SAMPLES = 5000


def _unwrap_model(raw):
    """Return the underlying sklearn / LightGBM estimator from a MLflow PyFunc."""
    return raw.get_raw_model() if hasattr(raw, "get_raw_model") else raw


def build_proba_reference(
    model,
    reference: pd.DataFrame,
    feature_names: list[str],
    bins: int,
    max_samples: int,
    seed: int,
) -> dict:
    """Run the model on the reference dataset, return a snapshot dict.

    Saves both a histogram (for fast plotting) AND a subsampled list of raw
    probabilities (so the dashboard can run a K-S test against prod probas).

    Output schema (consumed by the dashboard):
        {"n": int, "mean": float, "median": float, "p95": float,
         "bins": [b0, b1, ..., bN], "counts": [c1, c2, ..., cN],
         "values": [p1, p2, ... pK]}    # subsample, K ~= max_samples
    """
    logger.info("Predicting probabilities on %d reference rows ...", len(reference))
    X = reference[feature_names]
    proba = model.predict_proba(X)[:, 1]
    counts, edges = np.histogram(proba, bins=bins, range=(0.0, 1.0))

    if len(proba) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(proba), size=max_samples, replace=False)
        sampled = proba[idx]
    else:
        sampled = proba

    snapshot = {
        "n": int(len(proba)),
        "mean": float(proba.mean()),
        "median": float(np.median(proba)),
        "p95": float(np.quantile(proba, 0.95)),
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "values": [float(p) for p in sampled.tolist()],
    }
    logger.info(
        "proba_reference: n=%d mean=%.3f median=%.3f p95=%.3f (subsampled %d values for K-S)",
        snapshot["n"], snapshot["mean"], snapshot["median"], snapshot["p95"],
        len(sampled),
    )
    return snapshot


def build_feature_importance(
    model,
    reference: pd.DataFrame,
    feature_names: list[str],
    sample_size: int,
    top_k: int,
    seed: int,
) -> dict:
    """Compute SHAP-based feature importance, return top-K dict.

    Output schema:
        {"method": "SHAP mean(|value|)", "sample_size": int,
         "top": [{"feature": "...", "importance": float, "rank": int}, ...]}
    """
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "shap not installed. It is declared in pyproject.toml dependencies — "
            "run `uv sync` to install it."
        ) from exc

    n = min(sample_size, len(reference))
    sample = reference.sample(n=n, random_state=seed)
    X = sample[feature_names]
    logger.info("Computing SHAP values on %d sampled rows ...", n)

    explainer = shap.TreeExplainer(model)
    # For LightGBM binary classifier, shap_values can be a list of two arrays
    # (one per class). We take the positive class which corresponds to
    # probability_default.
    raw_shap = explainer.shap_values(X)
    if isinstance(raw_shap, list):
        sv = raw_shap[1]
    else:
        sv = raw_shap
    if sv.ndim == 3:
        # newer SHAP returns (n_samples, n_features, n_classes) for binary
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    ordered = sorted(
        zip(feature_names, mean_abs, strict=True),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_k]

    snapshot = {
        "method": "SHAP mean(|value|)",
        "sample_size": n,
        "top": [
            {"feature": str(name), "importance": float(imp), "rank": i + 1}
            for i, (name, imp) in enumerate(ordered)
        ],
    }
    logger.info(
        "feature_importance top-5: %s",
        ", ".join(f"{x['feature']}={x['importance']:.4f}" for x in snapshot["top"][:5]),
    )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--proba-out", type=Path, default=DEFAULT_PROBA_OUT)
    parser.add_argument("--importance-out", type=Path, default=DEFAULT_IMPORTANCE_OUT)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument(
        "--proba-samples",
        type=int,
        default=DEFAULT_PROBA_SAMPLES,
        help="Reference probabilities subsampled in JSON for K-S (default %(default)s).",
    )
    parser.add_argument("--shap-sample", type=int, default=DEFAULT_SHAP_SAMPLE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP computation (only refresh proba_reference.json).",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found.")
    if not args.reference.exists():
        raise SystemExit(
            f"{args.reference} not found. Run scripts/build_reference_dataset.py first."
        )

    logger.info("Loading model from %s ...", args.model)
    raw = joblib.load(args.model)
    model = _unwrap_model(raw)

    logger.info("Loading reference dataset from %s ...", args.reference)
    reference = pd.read_parquet(args.reference)
    feature_names = json.loads(args.feature_names.read_text())
    missing = [c for c in feature_names if c not in reference.columns]
    if missing:
        raise SystemExit(
            f"Reference dataset is missing {len(missing)} expected features; "
            "rebuild it with scripts/build_reference_dataset.py."
        )

    proba_snapshot = build_proba_reference(
        model, reference, feature_names, args.bins, args.proba_samples, args.seed
    )
    args.proba_out.parent.mkdir(parents=True, exist_ok=True)
    args.proba_out.write_text(json.dumps(proba_snapshot, indent=2))
    logger.info("Wrote %s (%.1f KB)", args.proba_out, args.proba_out.stat().st_size / 1024)

    if not args.skip_shap:
        importance_snapshot = build_feature_importance(
            model, reference, feature_names, args.shap_sample, args.top_k, args.seed
        )
        args.importance_out.write_text(json.dumps(importance_snapshot, indent=2))
        logger.info(
            "Wrote %s (%.1f KB)",
            args.importance_out, args.importance_out.stat().st_size / 1024,
        )
    else:
        logger.info("Skipped SHAP (--skip-shap)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
