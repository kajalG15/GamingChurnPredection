"""
Player Engagement Prediction API
=================================

A FastAPI service that loads the trained pipeline + label encoder and exposes
prediction endpoints for player engagement classification (Low / Medium / High).

Endpoints
---------
GET  /              — service info
GET  /health        — liveness check
GET  /metadata      — model info + test metrics
POST /predict       — predict for ONE player
POST /predict_batch — predict for many players at once

Run locally:
    uvicorn api.app:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import json
import os
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(HERE)
MODEL_PATH = os.path.join(PROJECT, "models", "engagement_pipeline.pkl")
LABEL_PATH = os.path.join(PROJECT, "models", "label_encoder.pkl")
META_PATH  = os.path.join(PROJECT, "models", "metadata.json")

# ----------------------------------------------------------------------------
# Load model + encoder + metadata at startup
# ----------------------------------------------------------------------------
for p in (MODEL_PATH, LABEL_PATH, META_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing {p}. Run the notebook first to train and save artifacts."
        )

pipeline      = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

CLASSES = list(label_encoder.classes_)  # e.g. ['High', 'Low', 'Medium']

# ----------------------------------------------------------------------------
# Pydantic schemas
# ----------------------------------------------------------------------------
class Player(BaseModel):
    """Raw player fields exactly as they appear in the Kaggle CSV."""

    Age:                       int  = Field(..., ge=10, le=100)
    Gender:                    Literal["Male", "Female"]
    Location:                  Literal["USA", "Europe", "Asia", "Other"]
    GameGenre:                 Literal["Strategy", "Sports", "Action", "RPG", "Simulation"]
    PlayTimeHours:             float = Field(..., ge=0, le=24)
    InGamePurchases:           Literal[0, 1] = Field(..., description="0 = No, 1 = Yes")
    GameDifficulty:            Literal["Easy", "Medium", "Hard"]
    SessionsPerWeek:           int = Field(..., ge=0, le=50)
    AvgSessionDurationMinutes: int = Field(..., ge=0, le=600)
    PlayerLevel:               int = Field(..., ge=1, le=200)
    AchievementsUnlocked:      int = Field(..., ge=0, le=200)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 28,
                "Gender": "Male",
                "Location": "USA",
                "GameGenre": "RPG",
                "PlayTimeHours": 18.5,
                "InGamePurchases": 1,
                "GameDifficulty": "Hard",
                "SessionsPerWeek": 14,
                "AvgSessionDurationMinutes": 95,
                "PlayerLevel": 78,
                "AchievementsUnlocked": 42,
            }
        }
    }


class PredictionResponse(BaseModel):
    engagement_level: Literal["Low", "Medium", "High"]
    confidence:       float = Field(..., ge=0, le=1)
    probabilities:    dict[str, float]
    retention_action: str


class BatchRequest(BaseModel):
    players: List[Player]


class BatchResponse(BaseModel):
    count:   int
    results: List[PredictionResponse]


# ----------------------------------------------------------------------------
# Feature engineering — MUST match the notebook exactly
# ----------------------------------------------------------------------------
# These thresholds were the 75th percentiles on the training data.
# In production, you'd persist them; for this exercise we use sensible defaults.
POWER_USER_PLAYTIME_THRESHOLD     = 18.0   # hours
POWER_USER_ACHIEVEMENTS_THRESHOLD = 38     # count


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["weekly_play_minutes"] = (
        df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"]
    )

    df["achievements_per_level"] = (
        df["AchievementsUnlocked"] / df["PlayerLevel"].replace(0, 1)
    )

    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 25, 35, 50, 100],
        labels=["Teen", "18-24", "25-34", "35-49", "50+"],
    )

    df["is_power_user"] = (
        (df["PlayTimeHours"] >= POWER_USER_PLAYTIME_THRESHOLD) &
        (df["AchievementsUnlocked"] >= POWER_USER_ACHIEVEMENTS_THRESHOLD)
    ).astype(int)

    df["session_intensity"] = (
        df["AvgSessionDurationMinutes"] * np.log1p(df["SessionsPerWeek"])
    )

    return df


def retention_recommendation(level: str, confidence: float) -> str:
    """Map prediction to a business-friendly retention action."""
    if level == "Low":
        return ("Send re-engagement push notification + offer free in-game currency. "
                "Consider lowering difficulty if available.")
    if level == "Medium":
        return ("Promote new content / events. Show personalized achievement targets to "
                "nudge them toward High engagement.")
    return ("Reward loyalty: exclusive content, early access, or VIP perks. "
            "These players are your best advocates — invite to beta tests.")


def predict_one(player_dict: dict) -> PredictionResponse:
    df = pd.DataFrame([player_dict])
    df = add_engineered_features(df)

    pred_idx = pipeline.predict(df)[0]
    proba    = pipeline.predict_proba(df)[0]
    label    = label_encoder.inverse_transform([pred_idx])[0]

    probs_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, proba)}
    confidence = float(proba.max())

    return PredictionResponse(
        engagement_level=label,
        confidence=round(confidence, 4),
        probabilities=probs_dict,
        retention_action=retention_recommendation(label, confidence),
    )


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Player Engagement Prediction API",
    description="Predict whether an online-game player is Low, Medium, or High engagement.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "Player Engagement Prediction API",
        "model":   metadata.get("model_name"),
        "classes": CLASSES,
        "docs":    "/docs",
        "endpoints": ["/health", "/metadata", "/predict", "/predict_batch"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "n_classes": len(CLASSES)}


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(player: Player):
    try:
        return predict_one(player.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.players:
        raise HTTPException(status_code=400, detail="`players` list is empty.")
    if len(req.players) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 players per request.")
    try:
        df = pd.DataFrame([p.model_dump() for p in req.players])
        df = add_engineered_features(df)

        pred_idx = pipeline.predict(df)
        probs    = pipeline.predict_proba(df)
        labels   = label_encoder.inverse_transform(pred_idx)

        results = []
        for label, p_row in zip(labels, probs):
            probs_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, p_row)}
            confidence = float(p_row.max())
            results.append(PredictionResponse(
                engagement_level=label,
                confidence=round(confidence, 4),
                probabilities=probs_dict,
                retention_action=retention_recommendation(label, confidence),
            ))
        return BatchResponse(count=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
