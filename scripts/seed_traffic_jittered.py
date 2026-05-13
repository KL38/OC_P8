"""Seed traffic with controlled jitter on each payload.

Variant of ``scripts/seed_traffic.py`` that perturbs each sampled
application_train row before POSTing:
- numerics: x = x * (1 + uniform(-10%, +10%)), clamped to schema bounds
- categoricals: 10% chance of swapping to another value in the vocab
- binary FLAG_*: 5% chance of flipping 0 <-> 1
- FLAG_DOCUMENT_* and SK_ID_CURR: untouched

Use to compare the drift report against the "clean" realistic seed. The
jitter introduces synthetic variability that should bump the drift score
above the natural sampling baseline — a way to demonstrate that the
monitoring pipeline reacts to input perturbation.

Usage:
    uv run python scripts/seed_traffic_jittered.py
    uv run python scripts/seed_traffic_jittered.py --known 100 --unknown 0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    INT_FIELDS,
    NON_SCHEMA_FIELDS,
    UNKNOWN_ID_START,
    run,
)

logger = logging.getLogger("scripts.seed_traffic_jittered")
logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES_PATH = REPO_ROOT / "models" / "app_train_categories.json"

JITTER_PCT = 0.10
CAT_SWAP_PROB = 0.10
FLAG_FLIP_PROB = 0.05

EXPLICIT_BOUNDS: dict[str, tuple[float, float]] = {
    "REGION_POPULATION_RELATIVE": (0.0, 1.0),
    "DAYS_BIRTH": (-25550, -6570),
    "DAYS_EMPLOYED": (-25000, 365_243),
    "DAYS_REGISTRATION": (-25000.0, 0.0),
    "DAYS_ID_PUBLISH": (-10000, 0),
    "DAYS_LAST_PHONE_CHANGE": (-15000.0, 0.0),
    "OWN_CAR_AGE": (0.0, 100.0),
    "CNT_CHILDREN": (0, 20),
    "CNT_FAM_MEMBERS": (1.0, 20.0),
    "HOUR_APPR_PROCESS_START": (0, 23),
    "REGION_RATING_CLIENT": (1, 3),
    "REGION_RATING_CLIENT_W_CITY": (1, 3),
}

FROZEN_FIELDS: set[str] = {"SK_ID_CURR"} | {f"FLAG_DOCUMENT_{i}" for i in range(2, 22)}


def _bounds_for(name: str) -> tuple[float, float]:
    if name in EXPLICIT_BOUNDS:
        lo, hi = EXPLICIT_BOUNDS[name]
        return float(lo), float(hi)
    if name.startswith("EXT_SOURCE_"):
        return 0.0, 1.0
    if name.endswith(("_AVG", "_MODE", "_MEDI")):
        return 0.0, 1.0
    if name.startswith(("OBS_", "DEF_", "AMT_REQ_CREDIT_BUREAU_")):
        return 0.0, 500.0
    if name.startswith("AMT_"):
        return 1.0, float("inf")
    return float("-inf"), float("inf")


def _is_binary_flag(name: str, value: Any) -> bool:
    return (
        name.startswith("FLAG_")
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def _jitter_numeric(name: str, value: float | int, rng: random.Random) -> float | int:
    factor = 1.0 + rng.uniform(-JITTER_PCT, JITTER_PCT)
    new = value * factor
    lo, hi = _bounds_for(name)
    new = max(lo, min(hi, new))
    return int(round(new)) if isinstance(value, int) else new


def _row_to_jittered_payload(
    row: pd.Series,
    categories: dict[str, list[str]],
    rng: random.Random,
) -> dict[str, Any]:
    """Inline jitter while converting the CSV row to a payload dict."""
    payload: dict[str, Any] = {}
    for name, value in row.items():
        if name in NON_SCHEMA_FIELDS:
            continue
        # NaN → None first, then decide whether to jitter the resulting Python type.
        if isinstance(value, float) and math.isnan(value):
            payload[name] = None
            continue
        if name in FROZEN_FIELDS:
            payload[name] = value
            continue
        if name in categories:
            payload[name] = (
                rng.choice(categories[name]) if rng.random() < CAT_SWAP_PROB else value
            )
            continue
        # Pandas reads ints as float when the column has NaN. Recast first so
        # _is_binary_flag and _jitter_numeric see the right type.
        if name in INT_FIELDS:
            value = int(value)
        if _is_binary_flag(name, value):
            payload[name] = (1 - value) if rng.random() < FLAG_FLIP_PROB else value
            continue
        if isinstance(value, (int, float)):
            payload[name] = _jitter_numeric(name, value, rng)
            continue
        payload[name] = value
    return payload


def build_payloads(
    app_train_path: Path,
    n_known: int,
    n_unknown: int,
    categories: dict[str, list[str]],
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
        payload = _row_to_jittered_payload(row, categories, rng)
        if i >= n_known:
            payload["SK_ID_CURR"] = UNKNOWN_ID_START + (i - n_known)
        payloads.append(payload)

    rng.shuffle(payloads)
    logger.info(
        "Built %d jittered payloads (%d known + %d unknown)",
        len(payloads), n_known, n_unknown,
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
    parser.add_argument("--categories-path", type=Path, default=CATEGORIES_PATH)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    categories = json.loads(args.categories_path.read_text(encoding="utf-8"))
    payloads = build_payloads(
        args.app_train_path, args.known, args.unknown, categories, rng
    )
    return run(payloads, args.base_url, args.delay)


if __name__ == "__main__":
    sys.exit(main())
