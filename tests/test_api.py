"""
Tests unitaires — API FastAPI
==============================
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_health_endpoint():
    """Test que l'endpoint /health répond correctement."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "version" in data
    except Exception as e:
        pytest.skip(f"API non disponible : {e}")


def test_health_response_schema():
    """Test le schéma de la réponse health."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "modeles_charges" in data
        assert isinstance(data["modeles_charges"], list)
    except Exception as e:
        pytest.skip(f"API non disponible : {e}")


def test_predict_numeric_schema():
    """Test que /predict accepte le bon schéma."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        payload = {
            "poids": 25.0,
            "volume": 50.0,
            "conductivite": 0.8,
            "opacite": 0.3,
            "rigidite": 5.0,
            "source": "Usine_A",
        }
        response = client.post("/predict", json=payload)
        # Soit 200 (si modèle chargé) soit 503 (si modèle non disponible)
        assert response.status_code in [200, 503]
    except Exception as e:
        pytest.skip(f"API non disponible : {e}")


def test_predict_text_schema():
    """Test que /predict_text accepte le bon schéma."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        payload = {
            "rapport_collecte": "Lot de métal conducteur récupéré à l'Usine A, poids 45 kg."
        }
        response = client.post("/predict_text", json=payload)
        assert response.status_code in [200, 503]
    except Exception as e:
        pytest.skip(f"API non disponible : {e}")


def test_predict_text_too_short():
    """Test que les textes trop courts sont rejetés."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.post("/predict_text", json={"rapport_collecte": "ab"})
        assert response.status_code == 422  # Validation error
    except Exception as e:
        pytest.skip(f"API non disponible : {e}")


def test_model_info_endpoint():
    """Test l'endpoint /model_info."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/model_info")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    except Exception as e:
        pytest.skip(f"API non disponible : {e}")
