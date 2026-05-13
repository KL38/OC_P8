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

    Implementation note (étape 4 optimisation): the legacy version applied
    every transform (None→NaN, sentinel→NaN, factorize, one-hot) to a 1-row
    pandas DataFrame, which is pandas' worst-case workload — full overhead
    per column without amortisation. cProfile + line_profiler showed 16 ms
    per call dominated by ``pd.get_dummies`` (37%), ``pd.Categorical`` loop
    (29%), and the initial ``pd.DataFrame`` (19%). The new version does all
    transforms on a plain Python dict and builds the DataFrame ONCE at the
    end. Same outputs (column names match what ``pd.get_dummies`` would
    have emitted), ~5-7× faster on a single row.
    """
    multi_cat_set = {c for c in known_categories if c not in BINARY_COLUMNS}

    out: dict[str, Any] = {}

    for key, value in raw.items():
        # JSON null → np.nan so numeric columns keep float dtype and reach
        # LightGBM as its native missing-value signal (rather than object
        # dtype None, which the booster cannot consume).
        if value is None:
            value = np.nan

        # DAYS_EMPLOYED uses 365243 as a "not employed" sentinel at training
        # time; match the same NaN substitution.
        if key == "DAYS_EMPLOYED" and value == DAYS_EMPLOYED_SENTINEL:
            value = np.nan

        # Binary columns — factorize using the exact codes captured at
        # training time. Unknown / NaN values stay NaN.
        if key in BINARY_COLUMNS:
            mapping = binary_mappings[key]
            out[key] = mapping[value] if isinstance(value, str) and value in mapping else np.nan
            continue

        # Multi-valued categoricals — emit one 0/1 column per known category
        # (drop_first=False, dummy_na=False). Unknown / NaN values produce
        # all-zero dummies, matching pd.get_dummies semantics.
        if key in multi_cat_set:
            for category in known_categories[key]:
                out[f"{key}_{category}"] = 1 if value == category else 0
            continue

        # Numeric / pass-through columns.
        out[key] = value

    # Defensive: ensure every expected one-hot column exists even if the
    # source key was absent from the payload (shouldn't happen under
    # Pydantic validation, but cheap to guard against silent data loss).
    for key in multi_cat_set:
        if key not in raw:
            for category in known_categories[key]:
                out.setdefault(f"{key}_{category}", 0)

    df = pd.DataFrame([out])

    # Match the training-time dtype for the 3 binary columns. Other columns
    # keep whatever dtype pandas inferred from the dict — same behaviour as
    # the legacy implementation.
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    return df
