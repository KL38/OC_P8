"""API runtime configuration. Paths and constants resolved at import time."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(os.getenv("OC_P8_MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_INFO_PATH = Path(
    os.getenv("OC_P8_MODEL_INFO_PATH", ROOT / "models" / "model_info.json")
)
FEATURE_NAMES_PATH = Path(
    os.getenv("OC_P8_FEATURE_NAMES_PATH", ROOT / "models" / "feature_names.json")
)
APP_TRAIN_CATEGORIES_PATH = Path(
    os.getenv(
        "OC_P8_APP_TRAIN_CATEGORIES_PATH",
        ROOT / "models" / "app_train_categories.json",
    )
)
APP_TRAIN_BINARY_MAPPINGS_PATH = Path(
    os.getenv(
        "OC_P8_APP_TRAIN_BINARY_MAPPINGS_PATH",
        ROOT / "models" / "app_train_binary_mappings.json",
    )
)
NO_HISTORY_TEMPLATE_PATH = Path(
    os.getenv("OC_P8_NO_HISTORY_TEMPLATE_PATH", ROOT / "models" / "no_history_template.json")
)
FEATURE_STORE_PATH = Path(
    os.getenv("OC_P8_FEATURE_STORE_PATH", ROOT / "data" / "features_store.parquet")
)

# Default fallback if model_info.json does not expose the optimised threshold.
DEFAULT_THRESHOLD = 0.33
