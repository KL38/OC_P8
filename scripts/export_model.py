"""
Exports the champion credit scoring model from OC_P6 to a local joblib file
for embedding in the OC_P8 deployment image.

Approach: bypass MLflow Server entirely. Load the model directly from the 
filesystem using its physical path. Use MlflowClient only to fetch run-level 
metadata (metrics, params) for traceability.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

# ============================================================
# Configuration
# ============================================================
P6_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "lgbm_credit_scoring"
MODEL_VERSION = "2"

# Direct filesystem path to the model artifacts
# Adjust this if your model_id differs (check the "Source" output below)
P6_MODEL_PATH = Path(
    "C:/Users/Kevin/projects/OC_P6/mlartifacts/3/models/"
    "m-dbe6fc95d74c49a8bb4775a6b7516fcd/artifacts"
)

OUTPUT_DIR = Path("models")
OUTPUT_MODEL = OUTPUT_DIR / "model.joblib"
OUTPUT_INFO = OUTPUT_DIR / "model_info.json"

# ============================================================
# Setup
# ============================================================
mlflow.set_tracking_uri(P6_MLFLOW_TRACKING_URI)
client = MlflowClient()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Identify the model version (for metadata only)
# ============================================================
mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)

print("=" * 60)
print("Model identified for export:")
print(f"  Name      : {mv.name}")
print(f"  Version   : {mv.version}")
print(f"  Run ID    : {mv.run_id}")
print(f"  Source    : {mv.source}")

if mv.run_id is None:
    raise RuntimeError(f"Model version {MODEL_NAME}/{MODEL_VERSION} has no associated run_id")

run = client.get_run(mv.run_id)
print("\n  Run metrics:")
for k, v in run.data.metrics.items():
    print(f"    {k}: {v:.4f}")

# ============================================================
# 2. Load the model — directly from filesystem (no HTTP, no Registry resolution)
# ============================================================
if not P6_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifacts not found at: {P6_MODEL_PATH}\n"
        f"Verify the path matches the 'Source' shown above."
    )

print(f"\n  Loading from filesystem: {P6_MODEL_PATH}")
model = mlflow.pyfunc.load_model(str(P6_MODEL_PATH))

print("\n  Model signature:")
sig = model.metadata.signature
if sig is None:
    print("    [WARN] No signature logged - inputs will not be validated at inference!")
    n_features = None
    feature_names = None
else:
    inputs = sig.inputs.to_dict()
    n_features = len(inputs)
    feature_names = [f["name"] for f in inputs]
    print(f"    Expected input features: {n_features}")
    print(f"    First 5 feature names: {feature_names[:5]}")

# ============================================================
# 3. Save the model locally
# ============================================================
joblib.dump(model, OUTPUT_MODEL)
size_mb = OUTPUT_MODEL.stat().st_size / 1e6
print(f"\n[OK] Model saved to: {OUTPUT_MODEL} ({size_mb:.2f} MB)")

# ============================================================
# 4. Save provenance metadata
# ============================================================
info = {
    "model_name": mv.name,
    "version": mv.version,
    "model_source": mv.source,
    "run_id": mv.run_id,
    "physical_path": str(P6_MODEL_PATH),
    "exported_at": datetime.now().isoformat(),
    "metrics": run.data.metrics,
    "params": run.data.params,
    "n_features_expected": n_features,
    "feature_names": feature_names,
}
with open(OUTPUT_INFO, "w") as f:
    json.dump(info, f, indent=2, default=str)
print(f"[OK] Metadata saved to: {OUTPUT_INFO}")

print("=" * 60)