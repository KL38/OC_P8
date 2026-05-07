import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///C:/Users/Kevin/projects/OC_P6/mlruns")
client = MlflowClient()

# Liste tous les modèles enregistrés
models = client.search_registered_models()
print(f"Nombre de modèles enregistrés : {len(models)}")
for m in models:
    print(f"  - {m.name}")
    for v in m.latest_versions:
        print(f"      version {v.version}, stage: {v.current_stage}, run_id: {v.run_id}")