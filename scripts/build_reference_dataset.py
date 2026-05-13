"""Build the drift reference dataset.

Re-creates the EXACT feature space the model sees at inference time by
running the offline ``app_train_clean()`` pipeline (factorize + one-hot +
5 derived ratios) and joining it with the precomputed feature store
aggregations. The output is then reindexed on ``models/feature_names.json``
so columns match production 1-to-1.

Result: 10 000 stratified rows × (TARGET + 768 features), frozen baseline
for Evidently drift detection across the full input space — not just the
agg subset.

Re-run only when retraining the model or refreshing the feature store.

Usage:
    uv run python scripts/build_reference_dataset.py
    uv run python scripts/build_reference_dataset.py --upload    # push to HF
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow `uv run python scripts/build_reference_dataset.py` to import project
# modules without requiring the project to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering.orchestrator import app_train_clean  # noqa: E402

logger = logging.getLogger("scripts.build_reference_dataset")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_FEATURE_STORE = Path("data/features_store.parquet")
DEFAULT_APP_TRAIN = Path("data/application_train.csv")
DEFAULT_OUTPUT = Path("data/reference_dataset.parquet")
DEFAULT_FEATURE_NAMES = Path("models/feature_names.json")
DEFAULT_HF_REPO = os.getenv("OC_P8_HF_DATASET_REPO_ID", "KLEB38/oc-p8-features")


def build(
    feature_store_path: Path,
    app_train_path: Path,
    feature_names_path: Path,
    output_path: Path,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    if not feature_store_path.exists():
        raise SystemExit(
            f"{feature_store_path} not found. Run scripts/build_feature_store.py first."
        )
    if not app_train_path.exists():
        raise SystemExit(
            f"{app_train_path} not found. Drop the original Kaggle "
            "application_train.csv there (gitignored) or pass --app-train."
        )
    if not feature_names_path.exists():
        raise SystemExit(
            f"{feature_names_path} not found. It is committed to the repo, "
            "check your checkout."
        )

    # 1. Run the offline app_train pipeline (same code that trained the model).
    #    Produces ~150 columns: numerics, binary mappings, one-hot, 5 ratios.
    logger.info("Running app_train_clean() on %s", app_train_path)
    df_app = app_train_clean(app_train_path.parent)
    logger.info("app_train_clean output: %d rows × %d columns", *df_app.shape)
    df_app = df_app.set_index("SK_ID_CURR")

    # 2. Load the precomputed aggregate feature store (~600 cols, BURO_/PREV_/
    #    INSTAL_/POS_/CC_/ACTIVE_/CLOSED_).
    logger.info("Loading feature store from %s", feature_store_path)
    feature_store = pd.read_parquet(feature_store_path)
    if feature_store.index.name != "SK_ID_CURR" and "SK_ID_CURR" in feature_store.columns:
        feature_store = feature_store.set_index("SK_ID_CURR")
    logger.info("Feature store: %d rows × %d columns", *feature_store.shape)

    # 3. Inner join: only training clients (those with TARGET) survive.
    joined = df_app.join(feature_store, how="inner")
    logger.info(
        "Joined: %d clients with TARGET (dropped %d app_train clients without "
        "a feature store entry)",
        len(joined),
        len(df_app) - len(joined),
    )

    # 4. Reindex to match the model's expected column order (768 features).
    #    Missing columns become all-NaN; that's fine — Evidently can drop them
    #    if needed, and it surfaces the schema gap up front rather than silently.
    feature_names = json.loads(feature_names_path.read_text())
    missing = [c for c in feature_names if c not in joined.columns]
    extra = [c for c in joined.columns if c not in feature_names and c != "TARGET"]
    if missing:
        logger.warning(
            "%d feature(s) from feature_names.json absent from joined data (filled "
            "with NaN). Sample: %s%s",
            len(missing), ", ".join(missing[:5]), " ..." if len(missing) > 5 else "",
        )
    if extra:
        logger.info(
            "%d column(s) in joined data not in feature_names.json (dropped). "
            "Sample: %s%s",
            len(extra), ", ".join(extra[:5]), " ..." if len(extra) > 5 else "",
        )
    aligned = joined.reindex(columns=["TARGET"] + feature_names)
    logger.info(
        "Reindexed: %d rows × %d columns (1 TARGET + %d features)",
        *aligned.shape, len(feature_names),
    )

    # 5. Stratified sample to preserve the ~8% default rate of the training set.
    n_samples = min(n_samples, len(aligned))
    sampled, _ = train_test_split(
        aligned,
        train_size=n_samples,
        stratify=aligned["TARGET"],
        random_state=seed,
    )
    logger.info(
        "Stratified sample: %d rows, default_rate=%.3f",
        len(sampled),
        sampled["TARGET"].mean(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_path)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved reference dataset to %s (%.1f MB)", output_path, size_mb)
    return sampled


def upload(output_path: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed; skipping upload")
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set; skipping upload")
        return

    HfApi(token=token).upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="reference_dataset.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Uploaded to %s/reference_dataset.parquet", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--app-train", type=Path, default=DEFAULT_APP_TRAIN)
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true", help="Push to HF Dataset")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    args = parser.parse_args()

    build(
        args.feature_store,
        args.app_train,
        args.feature_names,
        args.output,
        args.samples,
        args.seed,
    )
    if args.upload:
        upload(args.output, args.repo_id)


if __name__ == "__main__":
    main()
