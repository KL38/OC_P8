"""Build the drift reference dataset.

Sample 10 000 rows from the training feature store, stratified on the
TARGET label, and pair them with the label itself. The result is a frozen
baseline that supports:
- Input drift (Evidently DataDriftPreset) on the 523 aggregated features
- Target drift (if needed later) since TARGET is preserved
- Concept drift (when production ground_truth becomes available)

Keeping the TARGET in the reference doesn't cost anything storage-wise and
preserves the option to extend monitoring without rebuilding.

Re-run only when retraining the model or refreshing the feature store.

Usage:
    uv run python scripts/build_reference_dataset.py
    uv run python scripts/build_reference_dataset.py --upload    # push to HF
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("scripts.build_reference_dataset")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_FEATURE_STORE = Path("data/features_store.parquet")
DEFAULT_APP_TRAIN = Path("data/application_train.csv")
DEFAULT_OUTPUT = Path("data/reference_dataset.parquet")
DEFAULT_HF_REPO = os.getenv("OC_P8_HF_DATASET_REPO_ID", "KLEB38/oc-p8-features")


def build(
    feature_store_path: Path,
    app_train_path: Path,
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

    logger.info("Loading feature store from %s", feature_store_path)
    feature_store = pd.read_parquet(feature_store_path)
    logger.info("Feature store: %d rows × %d columns", *feature_store.shape)

    logger.info("Loading TARGET from %s", app_train_path)
    targets = pd.read_csv(app_train_path, usecols=["SK_ID_CURR", "TARGET"]).set_index(
        "SK_ID_CURR"
    )

    # Inner join: only train clients (with TARGET available) survive.
    joined = feature_store.join(targets, how="inner")
    logger.info(
        "Joined: %d clients with TARGET (drop %d test rows)",
        len(joined),
        len(feature_store) - len(joined),
    )

    # Stratified sample to preserve the ~8% default rate of the training set.
    n_samples = min(n_samples, len(joined))
    sampled, _ = train_test_split(
        joined,
        train_size=n_samples,
        stratify=joined["TARGET"],
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true", help="Push to HF Dataset")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    args = parser.parse_args()

    build(args.feature_store, args.app_train, args.output, args.samples, args.seed)
    if args.upload:
        upload(args.output, args.repo_id)


if __name__ == "__main__":
    main()
