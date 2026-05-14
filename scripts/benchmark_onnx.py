"""Benchmark LightGBM joblib vs ONNX Runtime inference.

Loads N real feature rows from `data/reference_dataset.parquet`, runs each
sample one at a time through both backends (simulating the per-request path),
and reports p50/p95/p99 latency plus numerical equivalence.

Run:
    uv run python scripts/benchmark_onnx.py --n 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
DEFAULT_ONNX_PATH = ROOT / "models" / "model.onnx"
DEFAULT_FEATURE_NAMES_PATH = ROOT / "models" / "feature_names.json"
DEFAULT_REFERENCE_PATH = ROOT / "data" / "reference_dataset.parquet"


def percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0..100)."""
    return float(np.percentile(samples, pct))


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f} ms"


def bench_lightgbm(model, samples_df: pd.DataFrame) -> tuple[list[float], np.ndarray]:
    """Run predict_proba one row at a time. Returns (latencies_s, probas)."""
    latencies: list[float] = []
    probas = np.empty(len(samples_df), dtype=np.float64)
    for i in range(len(samples_df)):
        row = samples_df.iloc[[i]]
        t0 = time.perf_counter()
        p = model.predict_proba(row)[0, 1]
        latencies.append(time.perf_counter() - t0)
        probas[i] = p
    return latencies, probas


def bench_onnx(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    samples_array: np.ndarray,
) -> tuple[list[float], np.ndarray]:
    """Run ONNX inference one row at a time. Returns (latencies_s, probas)."""
    latencies: list[float] = []
    probas = np.empty(len(samples_array), dtype=np.float64)
    for i in range(len(samples_array)):
        # Reshape to (1, n_features); float32 is required by the ONNX graph.
        row = samples_array[i : i + 1]
        t0 = time.perf_counter()
        out = session.run([output_name], {input_name: row})[0]
        latencies.append(time.perf_counter() - t0)
        # zipmap=False → out shape is (1, 2): [prob_class_0, prob_class_1].
        probas[i] = out[0, 1]
    return latencies, probas


def summarise(name: str, latencies: list[float]) -> dict[str, float]:
    return {
        "backend": name,
        "n": len(latencies),
        "p50_ms": percentile(latencies, 50) * 1000,
        "p95_ms": percentile(latencies, 95) * 1000,
        "p99_ms": percentile(latencies, 99) * 1000,
        "mean_ms": float(np.mean(latencies)) * 1000,
        "total_s": sum(latencies),
    }


def print_table(rows: list[dict[str, float]]) -> None:
    print(
        f"{'Backend':<14}{'n':>6}{'p50':>12}{'p95':>12}{'p99':>12}{'mean':>12}"
    )
    for r in rows:
        print(
            f"{r['backend']:<14}{r['n']:>6}"
            f"{r['p50_ms']:>10.3f}ms"
            f"{r['p95_ms']:>10.3f}ms"
            f"{r['p99_ms']:>10.3f}ms"
            f"{r['mean_ms']:>10.3f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="Number of inferences")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument(
        "--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES_PATH
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report (e.g. profiling/benchmark_onnx.json).",
    )
    args = parser.parse_args()

    feature_names = json.loads(args.feature_names.read_text())

    # Sample N rows from the reference dataset, aligned to model feature order.
    reference = pd.read_parquet(args.reference)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(reference), size=args.n + args.warmup, replace=True)
    samples_df = reference.iloc[idx][feature_names].reset_index(drop=True)
    samples_array = samples_df.to_numpy(dtype=np.float32)

    # --- Load both backends ---
    print("Loading LightGBM model...")
    model = joblib.load(args.model)
    raw_model = model.get_raw_model() if hasattr(model, "get_raw_model") else model

    print("Loading ONNX session...")
    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[1].name  # probabilities output

    # --- Warmup (excluded from stats) ---
    print(f"Warmup x{args.warmup}...")
    bench_lightgbm(raw_model, samples_df.iloc[: args.warmup])
    bench_onnx(session, input_name, output_name, samples_array[: args.warmup])

    # --- Measured runs ---
    print(f"Benchmarking LightGBM (n={args.n})...")
    lgbm_lat, lgbm_proba = bench_lightgbm(raw_model, samples_df.iloc[args.warmup :])

    print(f"Benchmarking ONNX     (n={args.n})...")
    onnx_lat, onnx_proba = bench_onnx(
        session, input_name, output_name, samples_array[args.warmup :]
    )

    # --- Numerical equivalence ---
    diff = np.abs(lgbm_proba - onnx_proba)
    print()
    print("=== Numerical equivalence ===")
    print(f"max  |delta_proba| = {diff.max():.2e}")
    print(f"mean |delta_proba| = {diff.mean():.2e}")
    print(f"# rows with |delta| > 1e-5: {(diff > 1e-5).sum()} / {len(diff)}")

    # --- Latency summary ---
    print()
    print("=== Latency (single-row predict_proba) ===")
    rows = [
        summarise("LightGBM", lgbm_lat),
        summarise("ONNX-Runtime", onnx_lat),
    ]
    print_table(rows)

    speedup = median(lgbm_lat) / median(onnx_lat) if median(onnx_lat) > 0 else float("inf")
    gain_pct = (1 - median(onnx_lat) / median(lgbm_lat)) * 100
    print()
    print(f"Speed-up (p50): x{speedup:.2f}")
    print(f"Gain (p50): {gain_pct:.1f}%")

    if args.out is not None:
        report = {
            "config": {
                "n": args.n,
                "warmup": args.warmup,
                "seed": args.seed,
                "model_joblib": str(args.model),
                "model_onnx": str(args.onnx),
                "reference": str(args.reference),
            },
            "equivalence": {
                "max_abs_delta": float(diff.max()),
                "mean_abs_delta": float(diff.mean()),
                "rows_above_1e-5": int((diff > 1e-5).sum()),
                "n_compared": int(len(diff)),
            },
            "latency": rows,
            "speedup_p50": speedup,
            "gain_p50_pct": gain_pct,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print()
        print(f"Wrote report → {args.out}")


if __name__ == "__main__":
    main()
