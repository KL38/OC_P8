"""Offline script: build the runtime feature store and metadata artefacts.

Outputs:
- data/features_store.parquet      : ~600 aggregated columns indexed by SK_ID_CURR
                                     (bureau / prev / POS / CC / install only —
                                     application_train columns are reconstructed
                                     at inference time from the JSON input).
- models/feature_names.json        : ordered list of the 768 columns the model expects.
- models/app_train_columns.json    : metadata for the 122 raw app_train columns
                                     (dtype, min, max, sample) — used to draft the
                                     Pydantic schema and validate inference inputs.
- models/app_train_categories.json : exhaustive list of categorical values seen in
                                     training, per column. Crucial for one-hot
                                     consistency at inference time.

Run once before starting the API:
    uv run python scripts/build_feature_store.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# allow importing feature_engineering from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_engineering.aggregations import (  # noqa: E402
    bureau_and_balance,
    credit_card_balance,
    installments_payments,
    pos_cash,
    previous_applications,
)
from feature_engineering.orchestrator import merge_files  # noqa: E402

DATA_DIR = Path("C:/Users/Kevin/projects/OC_P6/data")
OUT_PARQUET = ROOT / "data" / "features_store.parquet"
OUT_FEATURE_NAMES = ROOT / "models" / "feature_names.json"
OUT_APP_TRAIN_COLS = ROOT / "models" / "app_train_columns.json"
OUT_APP_TRAIN_CATEGORIES = ROOT / "models" / "app_train_categories.json"
OUT_BINARY_MAPPINGS = ROOT / "models" / "app_train_binary_mappings.json"

# Same binary columns the notebook factorizes — order matters.
BINARY_COLUMNS = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]


def _build_app_train_metadata(data_dir: Path) -> tuple[dict, dict, dict]:
    """Read the raw application_train.csv and return:
    - column_meta: {col: {dtype, min, max, n_missing, sample}}
    - categories: {col: [unique values]} for object-dtype columns (excluding the
      binary ones, since those are factorized rather than one-hot encoded)
    - binary_mappings: {col: {value: code}} for the 3 binary columns, captured
      with the EXACT same first-occurrence order pd.factorize() saw at training.
    """
    raw = pd.read_csv(data_dir / "application_train.csv")
    raw = raw[raw["CODE_GENDER"] != "XNA"].reset_index(drop=True)

    binary_mappings: dict[str, dict[str, int]] = {}
    for col in BINARY_COLUMNS:
        codes, uniques = pd.factorize(raw[col])
        binary_mappings[col] = {str(v): int(i) for i, v in enumerate(uniques)}

    column_meta: dict[str, dict] = {}
    categories: dict[str, list] = {}

    for col in raw.columns:
        dtype = str(raw[col].dtype)
        n_missing = int(raw[col].isna().sum())
        meta: dict = {"dtype": dtype, "n_missing": n_missing}

        if raw[col].dtype == "object":
            uniques_sorted = sorted(raw[col].dropna().unique().tolist())
            meta["categories"] = uniques_sorted
            if col not in BINARY_COLUMNS:
                categories[col] = uniques_sorted
        else:
            non_null = raw[col].dropna()
            if len(non_null) > 0:
                meta["min"] = float(non_null.min())
                meta["max"] = float(non_null.max())
                meta["sample"] = float(non_null.iloc[0])
        column_meta[col] = meta

    return column_meta, categories, binary_mappings


def _build_aggregate_only_store(data_dir: Path) -> pd.DataFrame:
    """Build the parquet that contains ONLY the auxiliary aggregates.

    application_train columns are deliberately excluded — they are always
    reconstructed at inference time from the JSON input.
    """
    bureau = bureau_and_balance(data_dir)
    print(f"  bureau aggregates: {bureau.shape}")

    prev = previous_applications(data_dir)
    print(f"  previous_application aggregates: {prev.shape}")

    pos = pos_cash(data_dir)
    print(f"  POS_CASH aggregates: {pos.shape}")

    ins = installments_payments(data_dir)
    print(f"  installments aggregates: {ins.shape}")

    cc = credit_card_balance(data_dir)
    print(f"  credit_card aggregates: {cc.shape}")

    # Outer-join all aggregates on SK_ID_CURR. Any client absent from a given
    # auxiliary file will simply have NaN there — preserves the training signal.
    out = bureau.join(prev, how="outer").join(pos, how="outer")
    out = out.join(ins, how="outer").join(cc, how="outer")
    return out


def _extract_feature_names_from_full_pipeline(data_dir: Path) -> list[str]:
    """Run the full merge_files() once to recover the exact 770-column order
    the model was trained on. We keep only the 768 feature columns
    (excluding SK_ID_CURR and TARGET)."""
    print("Building full training dataframe to capture feature_names...")
    df = merge_files(data_dir)
    feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]
    return feature_cols


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_FEATURE_NAMES.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1 — extracting raw application_train metadata")
    column_meta, categories, binary_mappings = _build_app_train_metadata(DATA_DIR)
    print(f"  {len(column_meta)} columns analysed, {len(categories)} multi-cat")
    print(f"  binary mappings: {binary_mappings}")

    print("\nSTEP 2 — building aggregate-only feature store")
    agg_store = _build_aggregate_only_store(DATA_DIR)
    print(f"  Combined aggregates: {agg_store.shape}")

    print("\nSTEP 3 — extracting feature_names from full training pipeline")
    feature_names = _extract_feature_names_from_full_pipeline(DATA_DIR)
    print(f"  {len(feature_names)} feature columns identified")

    print("\nSTEP 4 — persisting artefacts")
    agg_store.to_parquet(OUT_PARQUET, compression="snappy")
    size_mb = OUT_PARQUET.stat().st_size / 1e6
    print(f"  ✅ {OUT_PARQUET} ({size_mb:.1f} MB)")

    OUT_FEATURE_NAMES.write_text(json.dumps(feature_names, indent=2))
    print(f"  ✅ {OUT_FEATURE_NAMES}")

    OUT_APP_TRAIN_COLS.write_text(json.dumps(column_meta, indent=2, default=str))
    print(f"  ✅ {OUT_APP_TRAIN_COLS}")

    OUT_APP_TRAIN_CATEGORIES.write_text(json.dumps(categories, indent=2))
    print(f"  ✅ {OUT_APP_TRAIN_CATEGORIES}")

    OUT_BINARY_MAPPINGS.write_text(json.dumps(binary_mappings, indent=2))
    print(f"  ✅ {OUT_BINARY_MAPPINGS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
