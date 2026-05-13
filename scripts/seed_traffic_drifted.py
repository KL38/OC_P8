"""Seed traffic with explicit income drift.

Variant of ``scripts/seed_traffic.py`` that multiplies ``AMT_INCOME_TOTAL``
by a configurable factor (default 1.3 = +30%) before each POST. Simulates
a population-level shift where the average demanding client suddenly has
significantly higher income — the kind of drift a real production
monitor must catch and alert on.

The API re-derives ratios (INCOME_CREDIT_PERC, ANNUITY_INCOME_PERC,
INCOME_PER_PERSON) server-side, so the shifted income propagates to
those derived features automatically.

Usage:
    uv run python scripts/seed_traffic_drifted.py                    # 1.3x income
    uv run python scripts/seed_traffic_drifted.py --income-factor 2.0
    uv run python scripts/seed_traffic_drifted.py --known 100 --unknown 0
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from seed_traffic import (
    APP_TRAIN_PATH,
    DEFAULT_BASE_URL,
    DEFAULT_DELAY_S,
    DEFAULT_KNOWN,
    DEFAULT_SEED,
    DEFAULT_UNKNOWN,
    UNKNOWN_ID_START,
    _row_to_payload,
    run,
)

logger = logging.getLogger("scripts.seed_traffic_drifted")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_INCOME_FACTOR = 1.3


def build_payloads(
    app_train_path: Path,
    n_known: int,
    n_unknown: int,
    income_factor: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not app_train_path.exists():
        raise SystemExit(
            f"{app_train_path} not found. Place the Kaggle application_train.csv there."
        )
    logger.info("Loading %s ...", app_train_path)
    df = pd.read_csv(app_train_path)
    df = df[df["CODE_GENDER"] != "XNA"]
    df = df[df["DAYS_BIRTH"].notna()]
    logger.info("application_train clean rows: %d", len(df))

    seed_state = rng.randint(0, 2**31 - 1)
    sample = df.sample(n=n_known + n_unknown, random_state=seed_state)

    payloads: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(sample.iterrows()):
        payload = _row_to_payload(row)
        if payload.get("AMT_INCOME_TOTAL") is not None:
            payload["AMT_INCOME_TOTAL"] = float(payload["AMT_INCOME_TOTAL"]) * income_factor
        if i >= n_known:
            payload["SK_ID_CURR"] = UNKNOWN_ID_START + (i - n_known)
        payloads.append(payload)

    rng.shuffle(payloads)
    logger.info(
        "Built %d payloads with income x%.2f (%d known + %d unknown)",
        len(payloads), income_factor, n_known, n_unknown,
    )
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--app-train-path", type=Path, default=APP_TRAIN_PATH)
    parser.add_argument("--known", type=int, default=DEFAULT_KNOWN)
    parser.add_argument("--unknown", type=int, default=DEFAULT_UNKNOWN)
    parser.add_argument(
        "--income-factor",
        type=float,
        default=DEFAULT_INCOME_FACTOR,
        help="Multiplier applied to AMT_INCOME_TOTAL (default %(default)s).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    payloads = build_payloads(
        args.app_train_path, args.known, args.unknown, args.income_factor, rng
    )
    return run(payloads, args.base_url, args.delay)


if __name__ == "__main__":
    sys.exit(main())
