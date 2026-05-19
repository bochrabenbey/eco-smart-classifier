"""
API REST FastAPI — Eco-Smart Classifier
========================================
Endpoints :
  GET  /health          → santé du service
  POST /predict         → prédiction depuis features numériques
  POST /predict_text    → prédiction depuis description textuelle (NLP)
  POST /predict_multi   → prédiction multimodale (numérique + texte)
  GET  /model_info      → informations sur les modèles chargés
"""

import os
import sys
from typing import Optional

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.nlp.text_preprocessing import preprocess_text

# ── Configuration ──────────────────────────────────────────────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
APP_VERSION = "1.0.0"

app = FastAPI(
    title="Eco-Smart Classifier API",
    description="Classification de déchets et estimation de valeur de revente.",
    version=APP_VERSION,
)

# ── Chargement des modèles ──────────────────────────────────────────────────
_models = {}


def load_models():
    global _models
    try:
        _models["classifier"] = joblib.load(f"{PROCESSED_DIR}/best_classifier.pkl")
        _models["label_encoder"] = joblib.load(f"{PROCESSED_DIR}/label_encoder.pkl")
        _models["scaler"] = joblib.load(f"{PROCESSED_DIR}/scaler.pkl")
        _models["source_encoder"] = joblib.load(f"{PROCESSED_DIR}/source_encoder.pkl")
    except Exception as e:
        print(f"[WARN] Classifier non chargé : {e}")

    try:
        _models["tfidf"] = joblib.load(f"{PROCESSED_DIR}/tfidf_vectorizer.pkl")
        _models["nlp_classifier"] = joblib.load(f"{PROCESSED_DIR}/nlp_best_classifier.pkl")
    except Exception as e:
        print(f"[WARN] NLP non chargé : {e}")

    try:
        _models["multimodal_pipeline"] = joblib.load(f"{PROCESSED_DIR}/pipeline_multimodal.pkl")
    except Exception as e:
        print(f"[WARN] Pipeline multimodal non chargé : {e}")

    try:
        _models["regressor"] = joblib.load(f"{PROCESSED_DIR}/best_regressor.pkl")
    except Exception as e:
        print(f"[WARN] Régresseur non chargé : {e}")

    print(f"[API] Modèles chargés : {list(_models.keys())}")


load_models()


# ── Schémas Pydantic ──────────────────────────────────────────────────────
class PredictNumericRequest(BaseModel):
    poids: float = Field(..., ge=0, description="Poids en kg")
    volume: float = Field(..., ge=0, description="Volume en litres")
    conductivite: float = Field(..., ge=0, le=1, description="Conductivité [0,1]")
    opacite: float = Field(..., ge=0, le=1, description="Opacité [0,1]")
    rigidite: float = Field(..., ge=0, description="Rigidité")
    source: Optional[str] = Field("Unknown", description="Source (Usine_A, etc.)")


class PredictTextRequest(BaseModel):
    rapport_collecte: str = Field(..., min_length=5,
                                  description="Description textuelle du déchet")


class PredictMultimodalRequest(BaseModel):
    poids: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)
    conductivite: float = Field(..., ge=0, le=1)
    opacite: float = Field(..., ge=0, le=1)
    rigidite: float = Field(..., ge=0)
    source: Optional[str] = "Unknown"
    rapport_collecte: str = Field(..., min_length=5)


class PredictionResponse(BaseModel):
    categorie: str
    categorie_id: int
    confiance: Optional[float] = None
    prix_estime: Optional[float] = None
    pipeline: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "modeles_charges": list(_models.keys()),
    }


@app.get("/model_info")
def model_info():
    info = {}
    for name, model in _models.items():
        info[name] = type(model).__name__
    return info


@app.post("/predict", response_model=PredictionResponse)
def predict_numeric(req: PredictNumericRequest):
    """Prédiction depuis les features numériques."""
    if "classifier" not in _models:
        raise HTTPException(503, "Modèle numérique non disponible")

    le = _models["label_encoder"]
    le_src = _models["source_encoder"]

    # Encodage source
    try:
        src_enc = le_src.transform([req.source])[0]
    except ValueError:
        src_enc = 0  # Unknown

    # Feature engineering
    densite = req.poids / (req.volume + 1e-6)

    features = np.array([[
        req.poids, req.volume, req.conductivite,
        req.opacite, req.rigidite, src_enc, densite
    ]])

    pred_id = int(_models["classifier"].predict(features)[0])
    pred_cat = le.inverse_transform([pred_id])[0]

    confiance = None
    if hasattr(_models["classifier"], "predict_proba"):
        proba = _models["classifier"].predict_proba(features)[0]
        confiance = float(np.max(proba))

    prix = None
    if "regressor" in _models:
        prix = float(_models["regressor"].predict(
            np.append(features[0], pred_id).reshape(1, -1)
        )[0])

    return PredictionResponse(
        categorie=pred_cat,
        categorie_id=pred_id,
        confiance=confiance,
        prix_estime=prix,
        pipeline="numeric",
    )


@app.post("/predict_text", response_model=PredictionResponse)
def predict_text(req: PredictTextRequest):
    """Prédiction depuis la description textuelle (NLP)."""
    if "nlp_classifier" not in _models or "tfidf" not in _models:
        raise HTTPException(503, "Pipeline NLP non disponible")

    le = _models["label_encoder"]
    texte_clean = preprocess_text(req.rapport_collecte)
    X = _models["tfidf"].transform([texte_clean])

    pred_id = int(_models["nlp_classifier"].predict(X)[0])
    pred_cat = le.inverse_transform([pred_id])[0]

    return PredictionResponse(
        categorie=pred_cat,
        categorie_id=pred_id,
        pipeline="nlp",
    )


@app.post("/predict_multi", response_model=PredictionResponse)
def predict_multimodal(req: PredictMultimodalRequest):
    """Prédiction multimodale (numérique + texte)."""
    if "multimodal_pipeline" not in _models:
        raise HTTPException(503, "Pipeline multimodal non disponible")

    le = _models["label_encoder"]
    le_src = _models.get("source_encoder")

    try:
        src_enc = le_src.transform([req.source])[0] if le_src else 0
    except ValueError:
        src_enc = 0

    densite = req.poids / (req.volume + 1e-6)
    texte_clean = preprocess_text(req.rapport_collecte)

    row = {
        "Poids": req.poids, "Volume": req.volume,
        "Conductivite": req.conductivite, "Opacite": req.opacite,
        "Rigidite": req.rigidite, "Source_enc": src_enc,
        "Densite_estimee": densite,
        "Rapport_clean": texte_clean,
    }
    import pandas as pd
    X = pd.DataFrame([row])

    pred_id = int(_models["multimodal_pipeline"].predict(X)[0])
    pred_cat = le.inverse_transform([pred_id])[0]

    return PredictionResponse(
        categorie=pred_cat,
        categorie_id=pred_id,
        pipeline="multimodal",
    )


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
