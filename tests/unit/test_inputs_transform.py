"""Unit tests for the runtime app_train transformation.

The critical property under test is the one-hot pitfall fix: a single-row
input must produce one column for EVERY known training category, not just
the one currently present.
"""

from __future__ import annotations

import numpy as np

from api.inputs_transform import transform_app_train_inputs

CATS = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "NAME_INCOME_TYPE": ["Pensioner", "Working", "State servant"],
    "ORGANIZATION_TYPE": ["Bank", "School", "XNA"],
}
BINARY = {
    "CODE_GENDER": {"M": 0, "F": 1},
    "FLAG_OWN_CAR": {"N": 0, "Y": 1},
    "FLAG_OWN_REALTY": {"Y": 0, "N": 1},
}


def test_binary_columns_factorized_with_training_codes():
    raw = {"CODE_GENDER": "F", "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "N"}
    out = transform_app_train_inputs(raw, CATS, BINARY)
    assert int(out["CODE_GENDER"].iloc[0]) == 1
    assert int(out["FLAG_OWN_CAR"].iloc[0]) == 1
    assert int(out["FLAG_OWN_REALTY"].iloc[0]) == 1


def test_one_hot_emits_all_known_categories_even_with_one_row():
    """The whole point of pd.Categorical(categories=KNOWN): every training
    category becomes a column regardless of what the row contains."""
    raw = {"NAME_CONTRACT_TYPE": "Cash loans"}
    out = transform_app_train_inputs(raw, CATS, BINARY)
    assert "NAME_CONTRACT_TYPE_Cash loans" in out.columns
    assert "NAME_CONTRACT_TYPE_Revolving loans" in out.columns
    assert int(out["NAME_CONTRACT_TYPE_Cash loans"].iloc[0]) == 1
    assert int(out["NAME_CONTRACT_TYPE_Revolving loans"].iloc[0]) == 0


def test_unknown_category_value_becomes_zero_everywhere():
    """If the JSON sends a value never seen in training, it's coerced to NaN
    by pd.Categorical, then get_dummies emits all-zero columns."""
    raw = {"ORGANIZATION_TYPE": "Mars Colony"}
    out = transform_app_train_inputs(raw, CATS, BINARY)
    assert int(out["ORGANIZATION_TYPE_Bank"].iloc[0]) == 0
    assert int(out["ORGANIZATION_TYPE_School"].iloc[0]) == 0
    assert int(out["ORGANIZATION_TYPE_XNA"].iloc[0]) == 0


def test_days_employed_sentinel_replaced_with_nan():
    raw = {"DAYS_EMPLOYED": 365243}
    out = transform_app_train_inputs(raw, CATS, BINARY)
    assert np.isnan(out["DAYS_EMPLOYED"].iloc[0])


def test_days_employed_normal_value_preserved():
    raw = {"DAYS_EMPLOYED": -1500}
    out = transform_app_train_inputs(raw, CATS, BINARY)
    assert int(out["DAYS_EMPLOYED"].iloc[0]) == -1500


def test_output_is_single_row():
    raw = {"NAME_CONTRACT_TYPE": "Cash loans"}
    out = transform_app_train_inputs(raw, CATS, BINARY)
    assert len(out) == 1
