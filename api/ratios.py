"""Derived ratios computed from raw application_train inputs.

Mirrors the 5 ratio features added in feature_engineering/orchestrator.py
:: app_train_clean(). Re-applied at inference time after JSON inputs override
the stored row, so the ratios always reflect the user-provided values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 ratio features in-place and return the DataFrame.

    Division by zero or NaN propagates as NaN — LightGBM handles it natively
    using the same routing it learned during training.
    """
    out = df.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        out["DAYS_EMPLOYED_PERC"] = _safe_divide(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])
        out["INCOME_CREDIT_PERC"] = _safe_divide(
            out["AMT_INCOME_TOTAL"], out["AMT_CREDIT"]
        )
        out["INCOME_PER_PERSON"] = _safe_divide(
            out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"]
        )
        out["ANNUITY_INCOME_PERC"] = _safe_divide(
            out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"]
        )
        out["PAYMENT_RATE"] = _safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])

    return out


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise division returning NaN when denominator is 0 or NaN."""
    result = num / denom
    return result.replace([np.inf, -np.inf], np.nan)
