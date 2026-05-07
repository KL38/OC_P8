"""Unit tests for the 5 derived-ratio formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.ratios import apply_derived_ratios


def _row(**overrides) -> pd.DataFrame:
    base = {
        "DAYS_EMPLOYED": -637.0,
        "DAYS_BIRTH": -9461.0,
        "AMT_INCOME_TOTAL": 200000.0,
        "AMT_CREDIT": 400000.0,
        "AMT_ANNUITY": 25000.0,
        "CNT_FAM_MEMBERS": 2.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_all_five_ratios_present():
    out = apply_derived_ratios(_row())
    expected = {
        "DAYS_EMPLOYED_PERC",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
    }
    assert expected.issubset(out.columns)


def test_known_values():
    out = apply_derived_ratios(_row())
    row = out.iloc[0]
    assert row["DAYS_EMPLOYED_PERC"] == pytest.approx(-637.0 / -9461.0)
    assert row["INCOME_CREDIT_PERC"] == pytest.approx(200000 / 400000)
    assert row["INCOME_PER_PERSON"] == pytest.approx(200000 / 2)
    assert row["ANNUITY_INCOME_PERC"] == pytest.approx(25000 / 200000)
    assert row["PAYMENT_RATE"] == pytest.approx(25000 / 400000)


def test_division_by_zero_returns_nan():
    out = apply_derived_ratios(_row(AMT_CREDIT=0))
    row = out.iloc[0]
    assert pd.isna(row["INCOME_CREDIT_PERC"])
    assert pd.isna(row["PAYMENT_RATE"])


def test_nan_propagates():
    out = apply_derived_ratios(_row(DAYS_EMPLOYED=np.nan))
    assert pd.isna(out.iloc[0]["DAYS_EMPLOYED_PERC"])


def test_input_dataframe_not_mutated():
    df = _row()
    snapshot = df.copy()
    _ = apply_derived_ratios(df)
    pd.testing.assert_frame_equal(df, snapshot)
