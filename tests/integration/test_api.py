"""Integration tests that boot the FastAPI app via TestClient.

The app is configured to use synthetic artefacts via env vars, so these
tests run in seconds and don't need the real 5 GB of training data.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(patched_settings):  # patched_settings sets the env vars first
    """Build a TestClient against a freshly-loaded app.

    Re-import api.main so it picks up the patched api.settings module.
    """
    import api.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["model_version"] == "test-1"


def test_swagger_docs_available(client):
    """The brief explicitly requires Swagger documentation."""
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_schema(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/predict" in schema["paths"]


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["threshold"] == 0.33
    assert payload["version"] == "test-1"
    assert payload["n_features_expected"] > 0


def test_predict_known_client(client, valid_payload):
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["sk_id_curr"] == valid_payload["SK_ID_CURR"]
    assert payload["client_known"] is True
    assert payload["decision"] in {"GRANTED", "REFUSED"}
    assert 0.0 <= payload["probability_default"] <= 1.0
    assert payload["threshold"] == 0.33
    assert payload["model_version"] == "test-1"


def test_predict_unknown_client(client, valid_payload):
    valid_payload["SK_ID_CURR"] = 999_999_999  # absent from synthetic feature store
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["client_known"] is False


def test_predict_rejects_negative_income(client, valid_payload):
    valid_payload["AMT_INCOME_TOTAL"] = -1.0
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_required_field(client, valid_payload):
    del valid_payload["DAYS_BIRTH"]
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 422


def test_predict_rejects_extra_field(client, valid_payload):
    valid_payload["__injection__"] = "evil"
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 422


def test_predict_rejects_unknown_contract_type(client, valid_payload):
    valid_payload["NAME_CONTRACT_TYPE"] = "Bitcoin loans"
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 422
