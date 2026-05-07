"""FastAPI application entrypoint.

Loads the model and feature artefacts once at startup (lifespan), then
serves them to every request without ever reloading — per the brief's
critical guideline.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api import settings
from api.inference_assembler import InferenceArtefacts, assemble
from api.predictor import CreditScoringPredictor
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy artefacts once and attach them to app.state."""
    logger.info("Loading model from %s", settings.MODEL_PATH)
    app.state.predictor = CreditScoringPredictor.load(
        model_path=settings.MODEL_PATH,
        model_info_path=settings.MODEL_INFO_PATH,
        default_threshold=settings.DEFAULT_THRESHOLD,
    )
    logger.info(
        "Predictor ready: version=%s threshold=%.4f",
        app.state.predictor.model_version,
        app.state.predictor.threshold,
    )

    logger.info("Loading inference artefacts...")
    app.state.artefacts = InferenceArtefacts.load(
        feature_names_path=settings.FEATURE_NAMES_PATH,
        categories_path=settings.APP_TRAIN_CATEGORIES_PATH,
        binary_mappings_path=settings.APP_TRAIN_BINARY_MAPPINGS_PATH,
        no_history_template_path=settings.NO_HISTORY_TEMPLATE_PATH,
        feature_store_path=settings.FEATURE_STORE_PATH,
    )
    logger.info(
        "Artefacts ready: %d feature_names, feature_store=%d clients",
        len(app.state.artefacts.feature_names),
        len(app.state.artefacts.feature_store),
    )

    # Cache model_info for the /model/info route.
    app.state.model_info = json.loads(settings.MODEL_INFO_PATH.read_text())

    yield


app = FastAPI(
    title="Credit Scoring API",
    description=(
        "Real-time credit default prediction for Prêt à Dépenser. "
        "Wraps a LightGBM model with business threshold 10*FN + FP."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """JSON error envelope for unexpected failures, structured for log shipping."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "type": exc.__class__.__name__},
    )



@app.get("/") # La page d'accueil 
async def read_root():
    return {"message": "Welcome to the CREDIT DEFAULT predictor API for Prêt à Dépenser"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", model_version=request.app.state.predictor.model_version)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info(request: Request) -> ModelInfoResponse:
    info = request.app.state.model_info
    return ModelInfoResponse(
        model_name=info.get("model_name", "lgbm_credit_scoring"),
        version=str(info.get("version", "unknown")),
        threshold=request.app.state.predictor.threshold,
        n_features_expected=len(request.app.state.artefacts.feature_names),
        metrics=info.get("metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    raw_inputs = payload.model_dump()
    sk_id = raw_inputs.pop("SK_ID_CURR")

    artefacts = request.app.state.artefacts
    predictor: CreditScoringPredictor = request.app.state.predictor

    try:
        features, client_known = assemble(raw_inputs, sk_id_curr=sk_id, artefacts=artefacts)
    except Exception as exc:
        logger.exception("Failed to assemble features for sk_id=%s", sk_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature assembly failed: {exc.__class__.__name__}",
        ) from exc

    proba, decision = predictor.predict(features)

    return PredictionResponse(
        sk_id_curr=sk_id,
        probability_default=proba,
        decision=decision,
        threshold=predictor.threshold,
        model_version=predictor.model_version,
        client_known=client_known,
    )
