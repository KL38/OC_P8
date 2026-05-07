"""Offline orchestration of the full feature engineering pipeline.

Replicates the merge_files() function from notebooks/EDA.ipynb so that the
parquet feature store can be built once with the same logic that trained
the model.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.aggregations import (
    bureau_and_balance,
    credit_card_balance,
    installments_payments,
    one_hot_encoder,
    pos_cash,
    previous_applications,
)


def app_train_clean(
    data_dir: Path, num_rows: int | None = None, nan_as_category: bool = False
) -> pd.DataFrame:
    """Read application_train.csv and apply the same transformations as the
    training notebook (binary factorize, one-hot, ratio features)."""
    df = pd.read_csv(data_dir / "application_train.csv", nrows=num_rows)
    df = df[df["CODE_GENDER"] != "XNA"]

    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    df, _ = one_hot_encoder(df, nan_as_category)

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    return df


def merge_files(data_dir: Path, debug: bool = False) -> pd.DataFrame:
    """Build the full 770-column training dataframe by joining application_train
    with all auxiliary aggregates. Returns a DataFrame with SK_ID_CURR + TARGET
    + 768 engineered features.
    """
    num_rows = 30000 if debug else None
    df = app_train_clean(data_dir, num_rows)

    bureau = bureau_and_balance(data_dir, num_rows)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how="left", on="SK_ID_CURR")
    del bureau
    gc.collect()

    prev = previous_applications(data_dir, num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how="left", on="SK_ID_CURR")
    del prev
    gc.collect()

    pos = pos_cash(data_dir, num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how="left", on="SK_ID_CURR")
    del pos
    gc.collect()

    ins = installments_payments(data_dir, num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how="left", on="SK_ID_CURR")
    del ins
    gc.collect()

    cc = credit_card_balance(data_dir, num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how="left", on="SK_ID_CURR")
    del cc
    gc.collect()

    return df
