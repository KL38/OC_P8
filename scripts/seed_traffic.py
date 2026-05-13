"""Seed production traffic for monitoring / drift detection.

Samples real client rows from ``data/application_train.csv`` and POSTs
them to the live API. The drift report is statistically meaningful
because the current distribution mirrors a slice of the training set
with one row per distinct client.

Default mix: 90 known clients + 10 unknown. Unknowns reuse a real
application_train row but rewrite ``SK_ID_CURR`` to a value outside the
feature store range (999100-999109), exercising the no_history path.

Usage:
    uv run python scripts/seed_traffic.py                       # 90 + 10
    uv run python scripts/seed_traffic.py --known 200 --unknown 20
    uv run python scripts/seed_traffic.py --unknown 0           # known only
    uv run python scripts/seed_traffic.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger("scripts.seed_traffic")
logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_TRAIN_PATH = REPO_ROOT / "data" / "application_train.csv"

DEFAULT_BASE_URL = "https://kleb38-oc-p8.hf.space"
DEFAULT_DELAY_S = 0.5
DEFAULT_SEED = 42
DEFAULT_KNOWN = 90
DEFAULT_UNKNOWN = 10

# SK_ID_CURR space reserved for synthetic "unknown" clients (well above the
# Kaggle Home Credit training range which tops out around 456 255).
UNKNOWN_ID_START = 999_100

# CSV columns that the Pydantic schema declares as ``int``. Pandas reads them
# as ``float64`` when the column contains any NaN, so we need to recast.
INT_FIELDS: set[str] = (
    {
        "SK_ID_CURR",
        "CNT_CHILDREN",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_ID_PUBLISH",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "HOUR_APPR_PROCESS_START",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    }
    | {f"FLAG_DOCUMENT_{i}" for i in range(2, 22)}
)

# CSV columns NOT in the Pydantic schema — must be dropped before POST.
NON_SCHEMA_FIELDS: set[str] = {"TARGET"}


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert one application_train row to a Pydantic-compatible payload.

    Steps:
    - drop columns absent from the API schema (TARGET, etc.)
    - replace pandas NaN with Python None (JSON-serialisable)
    - recast int-schema columns back to int (pandas widens them to float
      when the column has any NaN cell)
    """
    payload: dict[str, Any] = {}
    for name, value in row.items():
        if name in NON_SCHEMA_FIELDS:
            continue
        if isinstance(value, float) and math.isnan(value):
            payload[name] = None
        else:
            payload[name] = value
    for name in INT_FIELDS:
        if name in payload and payload[name] is not None:
            payload[name] = int(payload[name])
    return payload


def build_payloads(
    app_train_path: Path,
    n_known: int,
    n_unknown: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample real client rows + synthetic unknowns from application_train.csv."""
    if not app_train_path.exists():
        raise SystemExit(
            f"{app_train_path} not found. Place the Kaggle application_train.csv "
            "there (gitignored)."
        )
    logger.info("Loading %s ...", app_train_path)
    df = pd.read_csv(app_train_path)
    # Mirror the filter from feature_engineering.orchestrator.app_train_clean
    df = df[df["CODE_GENDER"] != "XNA"]
    df = df[df["DAYS_BIRTH"].notna()]
    logger.info("application_train clean rows: %d", len(df))

    seed_state = rng.randint(0, 2**31 - 1)
    sample = df.sample(n=n_known + n_unknown, random_state=seed_state)

    payloads: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(sample.iterrows()):
        payload = _row_to_payload(row)
        if i >= n_known:
            # Force the SK_ID_CURR out of the feature store range so the API
            # falls back on the no_history_template. The rest of the payload
            # remains a realistic application_train profile.
            payload["SK_ID_CURR"] = UNKNOWN_ID_START + (i - n_known)
        payloads.append(payload)

    rng.shuffle(payloads)
    logger.info(
        "Built %d payloads (%d known + %d unknown, unknown IDs %d-%d)",
        len(payloads), n_known, n_unknown,
        UNKNOWN_ID_START, UNKNOWN_ID_START + max(n_unknown - 1, 0),
    )
    return payloads


def post_one(
    client: httpx.Client, base_url: str, payload: dict[str, Any]
) -> tuple[int, float, dict[str, Any] | None]:
    started = time.perf_counter()
    try:
        resp = client.post(f"{base_url.rstrip('/')}/predict", json=payload)
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.warning("HTTP error: %s", exc)
        return 0, latency_ms, None
    latency_ms = (time.perf_counter() - started) * 1000
    body: dict[str, Any] | None
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, latency_ms, body


def run(payloads: list[dict[str, Any]], base_url: str, delay: float) -> int:
    total = len(payloads)
    logger.info("POSTing %d payloads to %s (delay=%.2fs)", total, base_url, delay)
    ok = 0
    errors = 0
    latencies: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        for i, payload in enumerate(payloads, start=1):
            status, latency_ms, body = post_one(client, base_url, payload)
            sk_id = payload.get("SK_ID_CURR")
            if status == 200 and body:
                ok += 1
                latencies.append(latency_ms)
                logger.info(
                    "[%3d/%3d] sk_id=%s known=%s status=200 latency=%4dms "
                    "proba=%.3f decision=%s",
                    i, total, sk_id, body.get("client_known"),
                    int(latency_ms), body.get("probability_default", -1.0),
                    body.get("decision", "?"),
                )
            else:
                errors += 1
                detail = body.get("detail") if isinstance(body, dict) else "no body"
                logger.warning(
                    "[%3d/%3d] sk_id=%s status=%s detail=%r",
                    i, total, sk_id, status, detail,
                )
            if delay > 0:
                time.sleep(delay)

    if latencies:
        ordered = sorted(latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    else:
        p50 = p95 = 0.0
    logger.info(
        "Done: %d ok / %d errors / %d total. Local round-trip p50=%dms p95=%dms",
        ok, errors, total, int(p50), int(p95),
    )
    return 0 if errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_S,
        help="Seconds between POSTs to avoid HF rate limits (default %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--app-train-path",
        type=Path,
        default=APP_TRAIN_PATH,
        help="application_train.csv source (default %(default)s).",
    )
    parser.add_argument(
        "--known",
        type=int,
        default=DEFAULT_KNOWN,
        help="Known clients sampled from app_train (default %(default)s).",
    )
    parser.add_argument(
        "--unknown",
        type=int,
        default=DEFAULT_UNKNOWN,
        help="Synthetic unknowns with rewritten SK_ID_CURR (default %(default)s).",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)
    payloads = build_payloads(args.app_train_path, args.known, args.unknown, rng)
    return run(payloads, args.base_url, args.delay)


if __name__ == "__main__":
    sys.exit(main())
