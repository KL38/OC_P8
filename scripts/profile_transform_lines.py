"""Line-by-line profiling of transform_app_train_inputs (étape 4).

Uses line_profiler programmatically — no @profile decorator pollution in
the source. Builds 100 realistic payloads from application_train.csv, loads
the training-time category artefacts, and runs transform_app_train_inputs
in a loop wrapped by LineProfiler.

Usage:
    uv run python scripts/profile_transform_lines.py
    uv run python scripts/profile_transform_lines.py --n 500
    uv run python scripts/profile_transform_lines.py --output profiling/lines.txt
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("scripts.profile_transform_lines")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="number of transform calls to profile (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "profiling" / "profile_transform_lines.txt",
                        help="text dump path (default: ./profiling/profile_transform_lines.txt)")
    args = parser.parse_args()

    from line_profiler import LineProfiler

    from api import settings
    from api.inputs_transform import (
        load_binary_mappings,
        load_categories,
        transform_app_train_inputs,
    )
    from scripts.seed_traffic import APP_TRAIN_PATH, build_payloads

    known_categories = load_categories(settings.APP_TRAIN_CATEGORIES_PATH)
    binary_mappings = load_binary_mappings(settings.APP_TRAIN_BINARY_MAPPINGS_PATH)

    rng = random.Random(args.seed)
    payloads = build_payloads(APP_TRAIN_PATH, n_known=args.n, n_unknown=0, rng=rng)
    # Drop SK_ID_CURR — transform_app_train_inputs expects the raw_inputs dict
    # WITHOUT the id (the handler pops it before calling assemble).
    payloads = [{k: v for k, v in p.items() if k != "SK_ID_CURR"} for p in payloads]

    def driver():
        for payload in payloads:
            transform_app_train_inputs(payload, known_categories, binary_mappings)

    lp = LineProfiler()
    lp.add_function(transform_app_train_inputs)
    wrapped = lp(driver)
    wrapped()
    # Console (live feedback) + file dump (for the étape 4 report).
    lp.print_stats(output_unit=1e-3)  # show ms instead of seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        lp.print_stats(stream=fh, output_unit=1e-3)
    logger.info("Line-profile dump saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
