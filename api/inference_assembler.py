"""Build the 768-column feature row consumed by the model.

The flow:
  1. Take the raw JSON inputs and apply app_train transforms (factorize +
     one-hot using the training categories) and the 5 derived ratios.
  2. Look up the auxiliary aggregates either from the parquet feature store
     (Case 1: known SK_ID_CURR) or from the no-history template (Case 2).
  3. Concatenate horizontally and reindex to feature_names so the column
     order matches what the model was trained on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.inputs_transform import (
    load_binary_mappings,
    load_categories,
    transform_app_train_inputs,
)
from api.ratios import apply_derived_ratios


@dataclass(frozen=True)
class InferenceArtefacts:
    """Read-only container for everything needed to assemble a feature row."""

    feature_names: list[str]
    known_categories: dict[str, list[str]]
    binary_mappings: dict[str, dict[str, int]]
    no_history_template: dict[str, float | int | None]
    feature_store: pd.DataFrame  # indexed by SK_ID_CURR

    @classmethod
    def load(
        cls,
        feature_names_path: Path,
        categories_path: Path,
        binary_mappings_path: Path,
        no_history_template_path: Path,
        feature_store_path: Path,
    ) -> "InferenceArtefacts":
        feature_names = json.loads(feature_names_path.read_text())
        known_categories = load_categories(categories_path)
        binary_mappings = load_binary_mappings(binary_mappings_path)
        # JSON null → Python None → np.nan, so LightGBM sees the same NaN signal
        # it learned during training rather than an object-dtype None.
        raw_template = json.loads(no_history_template_path.read_text())
        no_history_template = {
            k: (np.nan if v is None else v) for k, v in raw_template.items()
        }

        feature_store = pd.read_parquet(feature_store_path)
        if feature_store.index.name != "SK_ID_CURR":
            # Parquet stores SK_ID_CURR either as index or column; normalise.
            if "SK_ID_CURR" in feature_store.columns:
                feature_store = feature_store.set_index("SK_ID_CURR")

        return cls(
            feature_names=feature_names,
            known_categories=known_categories,
            binary_mappings=binary_mappings,
            no_history_template=no_history_template,
            feature_store=feature_store,
        )


def assemble(
    raw_inputs: dict[str, Any],
    sk_id_curr: int,
    artefacts: InferenceArtefacts,
) -> tuple[pd.DataFrame, bool]:
    """Build a 1×768 DataFrame ready for model.predict().

    Returns (features, client_known).
    """
    # 1. App_train portion — always reconstructed from the JSON payload.
    app_part = transform_app_train_inputs(
        raw_inputs,
        known_categories=artefacts.known_categories,
        binary_mappings=artefacts.binary_mappings,
    )
    app_part = apply_derived_ratios(app_part)

    # 2. Aggregate portion — lookup or template.
    if sk_id_curr in artefacts.feature_store.index:
        agg_part = artefacts.feature_store.loc[[sk_id_curr]].copy()
        agg_part.reset_index(drop=True, inplace=True)
        client_known = True
    else:
        agg_part = pd.DataFrame([artefacts.no_history_template])
        client_known = False

    # 3. Combine and align to the model's expected column order.
    combined = pd.concat(
        [app_part.reset_index(drop=True), agg_part.reset_index(drop=True)],
        axis=1,
    )

    # Some columns may appear in both halves (e.g. duplicated SK_ID_CURR if
    # not stripped). Keep the first occurrence.
    combined = combined.loc[:, ~combined.columns.duplicated()]

    aligned = combined.reindex(columns=artefacts.feature_names)
    aligned = aligned.replace([np.inf, -np.inf], np.nan)

    return aligned, client_known
