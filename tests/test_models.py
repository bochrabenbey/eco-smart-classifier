"""
Tests unitaires — Modèles ML
=============================
Tests de performance minimale et de prédiction.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MIN_ACCURACY = 0.70


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Données synthétiques multi-classes (4 classes = catégories de déchets)."""
    X, y = make_classification(
        n_samples=500, n_features=7, n_informative=5,
        n_redundant=1, n_classes=4, n_clusters_per_class=1,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── Tests performance minimale ────────────────────────────────────────────────

class TestModelPerformance:
    def test_random_forest_accuracy(self, synthetic_data):
        X_train, X_test, y_train, y_test = synthetic_data
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        assert acc >= MIN_ACCURACY, f"RF accuracy {acc:.4f} < seuil {MIN_ACCURACY}"

    def test_logistic_regression_accuracy(self, synthetic_data):
        X_train, X_test, y_train, y_test = synthetic_data
        clf = LogisticRegression(max_iter=300, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        assert acc >= MIN_ACCURACY, f"LR accuracy {acc:.4f} < seuil {MIN_ACCURACY}"

    def test_prediction_shape(self, synthetic_data):
        X_train, X_test, y_train, y_test = synthetic_data
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == y_test.shape

    def test_prediction_classes_valides(self, synthetic_data):
        X_train, X_test, y_train, y_test = synthetic_data
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        valid_classes = set(np.unique(y_train))
        pred_classes = set(np.unique(preds))
        assert pred_classes.issubset(valid_classes)

    def test_predict_proba_somme_1(self, synthetic_data):
        X_train, X_test, y_train, y_test = synthetic_data
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)


# ── Tests modèles sauvegardés ─────────────────────────────────────────────────

class TestSavedModels:
    def test_classifier_chargeable(self):
        import joblib
        model_path = "data/processed/best_classifier.pkl"
        if not os.path.exists(model_path):
            pytest.skip("Modèle pas encore entraîné")
        clf = joblib.load(model_path)
        assert hasattr(clf, "predict")

    def test_label_encoder_chargeable(self):
        import joblib
        enc_path = "data/processed/label_encoder.pkl"
        if not os.path.exists(enc_path):
            pytest.skip("Label encoder pas encore créé")
        le = joblib.load(enc_path)
        assert hasattr(le, "transform")
        assert hasattr(le, "classes_")
        assert len(le.classes_) == 4  # Papier, Plastique, Verre, Métal

    def test_pipeline_multimodal_chargeable(self):
        import joblib
        pipe_path = "data/processed/pipeline_multimodal.pkl"
        if not os.path.exists(pipe_path):
            pytest.skip("Pipeline multimodal pas encore créé")
        pipe = joblib.load(pipe_path)
        assert hasattr(pipe, "predict")
