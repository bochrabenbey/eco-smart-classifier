"""
Tests unitaires — Module 4 : Pipeline NLP
==========================================
⚠️  Charte IA : ces tests sont écrits SANS aide IA (zone rouge)
"""

import os
import sys

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nlp.text_preprocessing import (
    extraire_contamination,
    extraire_etat,
    preprocess_text,
)


# ── Tests prétraitement textuel ───────────────────────────────────────────────

class TestPreprocessText:
    def test_retourne_chaine(self):
        result = preprocess_text("Lot de plastique collecté à l'Usine A.")
        assert isinstance(result, str)

    def test_texte_vide(self):
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
        assert preprocess_text("   ") == ""

    def test_minuscules(self):
        result = preprocess_text("LOT DE PAPIER")
        assert result == result.lower()

    def test_stopwords_supprimes(self):
        result = preprocess_text("le la les de du des")
        # Après suppression des stopwords, le résultat doit être quasi-vide
        words = result.split()
        fr_stopwords = {"le", "la", "les", "de", "du", "des"}
        assert len([w for w in words if w in fr_stopwords]) == 0

    def test_mots_domaine_supprimes(self):
        result = preprocess_text("lot collecte rapport matériau")
        words = set(result.split())
        domain_words = {"lot", "collecte", "rapport", "matériau", "materiau"}
        assert len(words.intersection(domain_words)) == 0

    def test_mots_cles_conserves(self):
        result = preprocess_text("métal conducteur rigide")
        # Ces mots ne sont pas des stopwords → doivent rester
        for mot in ["métal", "conducteur", "rigide"]:
            assert mot in result or mot.replace("é", "e") in result

    def test_ponctuation_supprimee(self):
        result = preprocess_text("plastique, rigide! conducteur?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result


# ── Tests extraction regex ────────────────────────────────────────────────────

class TestExtractionRegex:
    def test_contamination_detectee(self):
        assert extraire_contamination("Légère contamination observée") == 1
        assert extraire_contamination("Traces de rouille présentes") == 1
        assert extraire_contamination("Matériau souillé") == 1

    def test_pas_de_contamination(self):
        assert extraire_contamination("Matériau en bon état") == 0
        assert extraire_contamination("Plastique propre") == 0

    def test_contamination_none(self):
        assert extraire_contamination(None) == 0
        assert extraire_contamination(123) == 0

    def test_etat_brise(self):
        assert extraire_etat("Brisé en plusieurs morceaux") == "Brisé"
        assert extraire_etat("matériau cassé") == "Brisé"

    def test_etat_neuf(self):
        assert extraire_etat("Matériau neuf, intact") == "Neuf"

    def test_etat_moyen(self):
        assert extraire_etat("État moyen, légèrement usé") == "Moyen"

    def test_etat_inconnu(self):
        assert extraire_etat("description quelconque") == "Inconnu"
        assert extraire_etat(None) == "Inconnu"


# ── Tests vectorisation TF-IDF ────────────────────────────────────────────────

class TestVectorisation:
    def test_tfidf_shape(self):
        corpus = [
            "métal conducteur rigide lourd",
            "plastique souple léger transparent",
            "verre fragile transparent dense",
            "papier léger souple opaque",
        ]
        tfidf = TfidfVectorizer(max_features=50)
        X = tfidf.fit_transform(corpus)
        assert X.shape[0] == len(corpus)
        assert X.shape[1] <= 50

    def test_tfidf_valeurs_positives(self):
        corpus = ["métal lourd conducteur", "plastique léger souple"]
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(corpus)
        assert (X.data >= 0).all()

    def test_tfidf_fit_transform_vs_transform(self):
        train = ["métal conducteur", "plastique souple"]
        test = ["verre fragile", "papier léger"]
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(train)
        X_test = tfidf.transform(test)
        # Même nombre de colonnes
        assert X_train.shape[1] == X_test.shape[1]
