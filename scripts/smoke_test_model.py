"""
Quick smoke test: reload the exported model and run a dummy prediction.
"""
import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/model.joblib")
print(f"✅ Model reloaded: {type(model)}")

# Dummy prediction with random features
# Adjust n_features if your model expects a specific count
n_features = 768
dummy = pd.DataFrame(np.random.randn(2, n_features), columns=[f"f_{i}" for i in range(n_features)])

try:
    pred = model.predict(dummy)
    print(f"✅ Prediction OK. Shape: {pred.shape}, sample values: {pred[:2]}")
except Exception as e:
    print(f"❌ Prediction failed: {type(e).__name__}: {e}")