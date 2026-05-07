"""Single-row transformation of raw application_train inputs.

Reproduces the engineering applied by feature_engineering.orchestrator
:: app_train_clean() but designed for one row at inference time.

The crucial detail: pd.get_dummies() on a single row only emits columns
for values actually present, so we feed it pd.Categorical(values,
categories=KNOWN) to guarantee that every category seen during training
yields a column — even when the value is absent from this particular request.

KNOWN_CATEGORIES is loaded from models/app_train_categories.json.
BINARY_MAPPINGS is loaded from models/app_train_binary_mappings.json
(captures the actual pd.factorize() codes used at training).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BINARY_COLUMNS = ("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY")
DAYS_EMPLOYED_SENTINEL = 365243


def load_categories(path: Path) -> dict[str, list[str]]:
    """Load the {column: [training categories]} map for multi-valued cats."""
    return json.loads(path.read_text())


def load_binary_mappings(path: Path) -> dict[str, dict[str, int]]:
    """Load the {column: {value: code}} factorize mapping captured at training."""
    return json.loads(path.read_text())


def transform_app_train_inputs(
    raw: dict[str, Any],
    known_categories: dict[str, list[str]],
    binary_mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Convert a raw JSON payload (dict) to a one-row DataFrame matching the
    training-time output of app_train_clean(), excluding TARGET.

    Output is missing the 5 derived ratios — pipe through
    api.ratios.apply_derived_ratios() afterward.
    """
    df = pd.DataFrame([raw])

    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(binary_mappings[col]).astype("Int64")

    multi_cat_cols = [
        c for c in known_categories if c not in BINARY_COLUMNS and c in df.columns
    ]
    for col in multi_cat_cols:
        df[col] = pd.Categorical(df[col], categories=known_categories[col])

    df = pd.get_dummies(df, columns=multi_cat_cols, dummy_na=False)

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_SENTINEL, np.nan
        )

    return df
