"""
Tests unitaires — Module 1 : Preprocessing
==========================================
⚠️  Charte IA : ces tests sont écrits SANS aide IA (zone rouge)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.preprocess import (
    analyze_missing,
    compare_imputers,
    treat_outliers,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame de test minimal avec les colonnes du projet."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Poids": np.random.normal(50, 15, n),
        "Volume": np.random.normal(100, 30, n),
        "Conductivite": np.random.uniform(0, 1, n),
        "Opacite": np.random.uniform(0, 1, n),
        "Rigidite": np.random.randint(1, 10, n).astype(float),
        "Prix_Revente": np.random.uniform(0.5, 10, n),
        "Categorie": np.random.choice(["Papier", "Plastique", "Verre", "Métal"], n),
        "Source": np.random.choice(["Usine_A", "Usine_B", None], n),
        "Rapport_Collecte": [f"Déchet collecté lot {i}" for i in range(n)],
    })
    # Introduire des NaN (~10%)
    for col in ["Poids", "Volume", "Conductivite"]:
        mask = np.random.random(n) < 0.10
        df.loc[mask, col] = np.nan
    return df


@pytest.fixture
def num_cols():
    return ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]


# ── Tests schéma ─────────────────────────────────────────────────────────────

class TestDataSchema:
    def test_colonnes_attendues(self, sample_df):
        expected = {"Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
                    "Prix_Revente", "Categorie", "Source", "Rapport_Collecte"}
        assert expected.issubset(set(sample_df.columns))

    def test_types_numeriques(self, sample_df):
        for col in ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]:
            assert sample_df[col].dtype in [np.float64, np.float32, np.int64]

    def test_categories_valides(self, sample_df):
        cats = set(sample_df["Categorie"].dropna().unique())
        valid = {"Papier", "Plastique", "Verre", "Métal"}
        assert cats.issubset(valid), f"Catégories inconnues : {cats - valid}"

    def test_taille_minimale(self, sample_df):
        assert len(sample_df) >= 100

    def test_rapport_collecte_non_vide(self, sample_df):
        # La colonne texte ne doit pas être entièrement vide
        assert sample_df["Rapport_Collecte"].notna().sum() > 0


# ── Tests imputation ──────────────────────────────────────────────────────────

class TestImputation:
    def test_nan_residuels_apres_imputation(self, sample_df, num_cols):
        cols_with_nan = [c for c in num_cols if c in sample_df.columns]
        _, df_imp, _ = compare_imputers(sample_df, cols_with_nan)
        remaining = df_imp[cols_with_nan].isnull().sum().sum()
        assert remaining == 0, f"NaN résiduels après imputation : {remaining}"

    def test_valeurs_nan_avant(self, sample_df, num_cols):
        cols_with_nan = [c for c in num_cols if c in sample_df.columns]
        total_nan = sample_df[cols_with_nan].isnull().sum().sum()
        assert total_nan > 0, "Le dataset de test devrait contenir des NaN"

    def test_meilleure_methode_retournee(self, sample_df, num_cols):
        cols = [c for c in num_cols if c in sample_df.columns]
        best, _, rmse_dict = compare_imputers(sample_df, cols)
        assert best in ["Médiane", "KNN", "IterativeImputer"]
        assert all(v >= 0 for v in rmse_dict.values())

    def test_rmse_positif(self, sample_df, num_cols):
        cols = [c for c in num_cols if c in sample_df.columns]
        _, _, rmse_dict = compare_imputers(sample_df, cols)
        for method, rmse in rmse_dict.items():
            assert rmse >= 0, f"RMSE négatif pour {method}"


# ── Tests outliers ──────────────────────────────────────────────────────────

class TestOutliers:
    def test_outliers_traites(self, sample_df, num_cols):
        # Injecter un outlier extrême
        df = sample_df.copy()
        df.loc[0, "Poids"] = 99999.0
        cols = [c for c in num_cols if c in df.columns]
        df_clean = treat_outliers(df, cols)
        Q3 = df["Poids"].quantile(0.75)
        IQR = df["Poids"].quantile(0.75) - df["Poids"].quantile(0.25)
        upper = Q3 + 1.5 * IQR
        assert df_clean["Poids"].max() <= upper + 1

    def test_shape_preserve(self, sample_df, num_cols):
        cols = [c for c in num_cols if c in sample_df.columns]
        df_clean = treat_outliers(sample_df.dropna(), cols)
        assert df_clean.shape == sample_df.dropna().shape


# ── Tests analyse manquance ──────────────────────────────────────────────────

class TestMissingAnalysis:
    def test_retourne_dict(self, sample_df):
        result = analyze_missing(sample_df)
        assert isinstance(result, dict)

    def test_mecanismes_valides(self, sample_df):
        result = analyze_missing(sample_df)
        for col, mec in result.items():
            assert mec in ["MCAR", "MAR", "MNAR"], f"Mécanisme inconnu : {mec}"

# ── Tests fonction encode_and_scale ──────────────────────────────────────────

class TestEncodeAndScale:
    def test_colonnes_encodees_presentes(self, sample_df, num_cols):
        cols = [c for c in num_cols if c in sample_df.columns]
        _, df_imp, _ = compare_imputers(sample_df, cols)
        df_imp["Prix_Revente"] = df_imp.get("Prix_Revente", 1.0)
        df_imp["Source"] = df_imp.get("Source", "Usine_A").fillna("Usine_A")
        from src.preprocessing.preprocess import encode_and_scale
        df_enc, le_cat, le_src, scaler = encode_and_scale(df_imp)
        assert "Source_enc" in df_enc.columns
        assert "Densite_estimee" in df_enc.columns

    def test_label_encoder_classes(self, sample_df, num_cols):
        cols = [c for c in num_cols if c in sample_df.columns]
        _, df_imp, _ = compare_imputers(sample_df, cols)
        df_imp["Prix_Revente"] = 5.0
        df_imp["Source"] = "Usine_A"
        from src.preprocessing.preprocess import encode_and_scale
        _, le_cat, _, _ = encode_and_scale(df_imp)
        assert len(le_cat.classes_) == 4
        assert set(le_cat.classes_) == {"Métal", "Papier", "Plastique", "Verre"}
        