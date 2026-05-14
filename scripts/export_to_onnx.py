"""Convert the LightGBM credit-scoring model to ONNX format.

Reads `models/model.joblib`, unwraps the MLflow PyFunc layer if present,
converts the underlying LightGBM model to ONNX via `onnxmltools`, and writes
`models/model.onnx`.

Run:
    uv run python scripts/export_to_onnx.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
DEFAULT_FEATURE_NAMES_PATH = ROOT / "models" / "feature_names.json"
DEFAULT_OUT_PATH = ROOT / "models" / "model.onnx"


def export(
    model_path: Path,
    feature_names_path: Path,
    out_path: Path,
    target_opset: int = 13,
) -> Path:
    """Convert LightGBM joblib → ONNX file. Returns the output path."""
    model = joblib.load(model_path)
    raw_model = model.get_raw_model() if hasattr(model, "get_raw_model") else model

    feature_names = json.loads(feature_names_path.read_text())
    n_features = len(feature_names)

    # ONNX requires a fixed-shape input declaration. None = dynamic batch dim.
    initial_types = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_lightgbm(
        raw_model,
        initial_types=initial_types,
        target_opset=target_opset,
        zipmap=False,  # Output raw probability array instead of list-of-dicts.
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES_PATH
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    out = export(
        model_path=args.model,
        feature_names_path=args.feature_names,
        out_path=args.out,
        target_opset=args.opset,
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
