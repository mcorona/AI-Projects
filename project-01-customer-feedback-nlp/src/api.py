"""
FastAPI inference server for the fine-tuned FinancialSentimentClassifier.

Run with:
    uvicorn src.api:app --reload

Model checkpoint path resolution (first match wins):
    1. MODEL_CKPT_PATH env var
    2. The most recently modified output/models/best-*.ckpt

Author: Manuel Corona
"""

import glob
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.sentiment_classifier import SentimentInference

logger = logging.getLogger(__name__)

_inference: Optional[SentimentInference] = None


def resolve_checkpoint_path() -> Optional[str]:
    env_path = os.environ.get("MODEL_CKPT_PATH")
    if env_path:
        return env_path
    candidates = sorted(glob.glob("output/models/best-*.ckpt"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference
    ckpt_path = resolve_checkpoint_path()
    if ckpt_path is None:
        logger.warning(
            "No checkpoint found (set MODEL_CKPT_PATH or place one at output/models/best-*.ckpt). "
            "Starting without a loaded model -- /predict and /batch will return 503."
        )
    else:
        logger.info(f"Loading checkpoint: {ckpt_path}")
        _inference = SentimentInference(ckpt_path)
        logger.info("Model loaded.")
    yield


app = FastAPI(title="Financial Sentiment Analyzer", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify.")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Texts to classify.")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]


def _require_model() -> SentimentInference:
    if _inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_CKPT_PATH and restart the server.")
    return _inference


def _to_response(detail: dict) -> SentimentResponse:
    return SentimentResponse(
        text=detail["text"],
        sentiment=detail["prediction"].lower(),
        confidence=detail["confidence"],
        probabilities={k.lower(): v for k, v in detail["probabilities"].items()},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _inference is not None}


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest) -> SentimentResponse:
    inference = _require_model()
    try:
        details = inference.predict_batch_with_details([request.text])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return _to_response(details[0])


@app.post("/batch", response_model=List[SentimentResponse])
async def batch_predict(request: BatchRequest) -> List[SentimentResponse]:
    inference = _require_model()
    try:
        details = inference.predict_batch_with_details(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return [_to_response(d) for d in details]
