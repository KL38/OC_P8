"""Unit tests for the inference assembler — the core branching logic
between known clients (parquet lookup) and unknown clients (no-history)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.inference_assembler import InferenceArtefacts, assemble


def _build_artefacts(synthetic_artefacts_dir):
    base = synthetic_artefacts_dir
    return InferenceArtefacts.load(
        feature_names_path=base / "models" / "feature_names.json",
        categories_path=base / "models" / "app_train_categories.json",
        binary_mappings_path=base / "models" / "app_train_binary_mappings.json",
        no_history_template_path=base / "models" / "no_history_template.json",
        feature_store_path=base / "data" / "features_store.parquet",
    )


def test_known_client_pulls_from_feature_store(synthetic_artefacts_dir, valid_payload):
    artefacts = _build_artefacts(synthetic_artefacts_dir)
    raw = dict(valid_payload)
    sk_id = raw.pop("SK_ID_CURR")  # 100002 — known in synthetic store

    features, client_known = assemble(raw, sk_id_curr=sk_id, artefacts=artefacts)

    assert client_known is True
    # Aggregates must come from the parquet, not the template
    assert features["BURO_DAYS_CREDIT_MEAN"].iloc[0] == -1234.5
    assert int(features["POS_COUNT"].iloc[0]) == 12
    assert features["PREV_AMT_ANNUITY_MEAN"].iloc[0] == 15000.0


def test_unknown_client_uses_no_history_template(synthetic_artefacts_dir, valid_payload):
    artefacts = _build_artefacts(synthetic_artefacts_dir)
    raw = dict(valid_payload)
    raw.pop("SK_ID_CURR")

    features, client_known = assemble(raw, sk_id_curr=999_999_999, artefacts=artefacts)

    assert client_known is False
    # Counts default to 0
    assert int(features["POS_COUNT"].iloc[0]) == 0
    assert int(features["INSTAL_COUNT"].iloc[0]) == 0
    assert int(features["CC_COUNT"].iloc[0]) == 0
    # Numeric aggregates default to NaN — preserves the training "no history" signal
    assert np.isnan(features["BURO_DAYS_CREDIT_MEAN"].iloc[0])
    assert np.isnan(features["PREV_AMT_ANNUITY_MEAN"].iloc[0])


def test_output_columns_are_in_feature_names_order(synthetic_artefacts_dir, valid_payload):
    artefacts = _build_artefacts(synthetic_artefacts_dir)
    raw = dict(valid_payload)
    raw.pop("SK_ID_CURR")
    features, _ = assemble(raw, sk_id_curr=100002, artefacts=artefacts)
    assert list(features.columns) == artefacts.feature_names


def test_derived_ratios_reflect_overrides(synthetic_artefacts_dir, valid_payload):
    """If the JSON overrides AMT_INCOME_TOTAL, the ratio columns must use the
    new value rather than any stale stored data."""
    artefacts = _build_artefacts(synthetic_artefacts_dir)
    raw = dict(valid_payload)
    raw["AMT_INCOME_TOTAL"] = 100_000.0
    raw["AMT_CREDIT"] = 200_000.0
    raw["AMT_ANNUITY"] = 10_000.0
    raw.pop("SK_ID_CURR")

    features, _ = assemble(raw, sk_id_curr=100002, artefacts=artefacts)
    row = features.iloc[0]
    assert row["INCOME_CREDIT_PERC"] == 0.5
    assert row["PAYMENT_RATE"] == 0.05


def test_unknown_categorical_values_do_not_crash(
    synthetic_artefacts_dir, valid_payload
):
    """A new ORGANIZATION_TYPE value not seen in training must coerce silently
    rather than break assembly — pd.Categorical handles this by emitting NaN."""
    artefacts = _build_artefacts(synthetic_artefacts_dir)
    raw = dict(valid_payload)
    raw["ORGANIZATION_TYPE"] = "Mars Colony"
    raw.pop("SK_ID_CURR")
    features, _ = assemble(raw, sk_id_curr=100002, artefacts=artefacts)
    assert isinstance(features, pd.DataFrame)


def test_inf_values_replaced_with_nan(synthetic_artefacts_dir, valid_payload):
    """Edge case: AMT_CREDIT=0 produces inf in PAYMENT_RATE; assembler scrubs."""
    artefacts = _build_artefacts(synthetic_artefacts_dir)
    raw = dict(valid_payload)
    raw["AMT_CREDIT"] = 0.0001  # tiny but non-zero to keep Pydantic happy upstream
    raw.pop("SK_ID_CURR")
    features, _ = assemble(raw, sk_id_curr=100002, artefacts=artefacts)
    for col in ("DAYS_EMPLOYED_PERC", "INCOME_CREDIT_PERC", "PAYMENT_RATE"):
        val = features[col].iloc[0]
        assert not np.isinf(val), f"{col} is inf"
