"""Offline script: build the 'no-history' template for unknown clients.

For each aggregate column (bureau / prev / POS / CC / install), decide:
- COUNT-style columns → 0
- All other aggregates  → NaN

This template is loaded once at API startup and used as the agg_part for
unknown SK_ID_CURR (Case 2 in the inference flow).

Run:
    uv run python scripts/build_no_history_template.py

Depends on:
    models/feature_names.json  (produced by build_feature_store.py)
    data/features_store.parquet (column reference)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "models" / "feature_names.json"
FEATURE_STORE_PATH = ROOT / "data" / "features_store.parquet"
OUT_TEMPLATE = ROOT / "models" / "no_history_template.json"

COUNT_PATTERN = re.compile(r"_COUNT$|_COUNT_|COUNT_$|^COUNT_")


def main() -> None:
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    store = pd.read_parquet(FEATURE_STORE_PATH)

    aggregate_cols = [c for c in store.columns]
    print(f"Inspecting {len(aggregate_cols)} aggregate columns...")

    template: dict[str, float | int] = {}
    count_cols: list[str] = []
    nan_cols: list[str] = []

    for col in aggregate_cols:
        if COUNT_PATTERN.search(col):
            template[col] = 0
            count_cols.append(col)
        else:
            template[col] = None  # serialised to JSON null → np.nan at load
            nan_cols.append(col)

    print(f"  count-style (→ 0): {len(count_cols)}")
    print(f"  others (→ NaN)   : {len(nan_cols)}")
    print("\nSample count columns:", count_cols[:10])
    print("Sample NaN columns:  ", nan_cols[:5])

    OUT_TEMPLATE.write_text(json.dumps(template, indent=2))
    print(f"\n✅ {OUT_TEMPLATE} ({len(template)} entries)")

    # Sanity check: every aggregate column from the store is also in feature_names
    missing = [c for c in aggregate_cols if c not in feature_names]
    if missing:
        print(f"⚠️  {len(missing)} columns in parquet but absent from feature_names")
        print(f"   first 5: {missing[:5]}")


if __name__ == "__main__":
    main()
