"""
Module 4 : Pipeline NLP — Extracteur de Caractéristiques
==========================================================
- Vectorisations : BoW, TF-IDF, Word2Vec, FastText
- Classifieurs  : Naive Bayes, LogisticRegression, LinearSVC, RandomForest
- Tracking MLflow
"""

import json
import os
import sys
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nlp.text_preprocessing import (
    extraire_contamination,
    extraire_etat,
    preprocess_text,
)

warnings.filterwarnings("ignore")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_DIR = params["data"]["processed_dir"]
RAW_PATH = params["data"]["raw_path"]
RANDOM_STATE = params["data"]["random_state"]

MAX_FEATURES = params["nlp"]["max_features"]
NGRAM_MIN = params["nlp"]["ngram_min"]
NGRAM_MAX = params["nlp"]["ngram_max"]
MIN_DF = params["nlp"]["min_df"]
W2V_DIM = params["nlp"]["w2v_vector_size"]
W2V_WINDOW = params["nlp"]["w2v_window"]
W2V_EPOCHS = params["nlp"]["w2v_epochs"]


def load_data():
    df_raw = pd.read_csv(RAW_PATH)
    df_tr = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    df_vl = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    df_te = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    # Nettoyage textuel
    print("[NLP] Nettoyage textuel...")
    df_raw["Rapport_clean"] = df_raw["Rapport_Collecte"].apply(preprocess_text)
    df_raw["Contamination"] = df_raw["Rapport_Collecte"].apply(extraire_contamination)
    df_raw["Etat_materiau"] = df_raw["Rapport_Collecte"].apply(extraire_etat)

    mapping = df_raw[["Rapport_Collecte", "Rapport_clean",
                       "Contamination", "Etat_materiau"]].drop_duplicates()

    for df in [df_tr, df_vl, df_te]:
        df.merge(mapping, on="Rapport_Collecte", how="left")

    df_tr = df_tr.merge(mapping, on="Rapport_Collecte", how="left")
    df_vl = df_vl.merge(mapping, on="Rapport_Collecte", how="left")
    df_te = df_te.merge(mapping, on="Rapport_Collecte", how="left")

    for df in [df_tr, df_vl, df_te]:
        df["Rapport_clean"] = df.get("Rapport_clean", pd.Series([""] * len(df))).fillna("")

    # Garder seulement les lignes labellisées
    df_tr = df_tr[df_tr["Categorie_enc"].notna()].copy()
    df_vl = df_vl[df_vl["Categorie_enc"].notna()].copy()
    df_te = df_te[df_te["Categorie_enc"].notna()].copy()

    for df in [df_tr, df_vl, df_te]:
        df["Categorie_enc"] = df["Categorie_enc"].astype(int)

    # Sauvegarder les features NLP
    df_raw[["Rapport_Collecte", "Rapport_clean", "Contamination", "Etat_materiau"]].to_csv(
        f"{PROCESSED_DIR}/nlp_features.csv", index=False
    )

    return df_tr, df_vl, df_te


def texte_vers_embedding(texte, modele, dim=100):
    """Mean pooling d'un texte vers un vecteur d'embedding."""
    mots = texte.split()
    vecs = [modele.wv[m] for m in mots if m in modele.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(dim)


def main():
    print("=" * 60)
    print("MODULE 4 — NLP")
    print("=" * 60)

    df_tr, df_vl, df_te = load_data()

    text_train = df_tr["Rapport_clean"].values
    text_val = df_vl["Rapport_clean"].values
    text_test = df_te["Rapport_clean"].values

    y_train = df_tr["Categorie_enc"].values
    y_val = df_vl["Categorie_enc"].values
    y_test = df_te["Categorie_enc"].values

    le = joblib.load(f"{PROCESSED_DIR}/label_encoder.pkl")

    # ── Vectorisations ──────────────────────────────────────────────────────
    print("\n[NLP] Vectorisations...")

    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(NGRAM_MIN, NGRAM_MAX),
                            max_features=MAX_FEATURES, min_df=MIN_DF,
                            sublinear_tf=True)
    X_tfidf_tr = tfidf.fit_transform(text_train)
    X_tfidf_vl = tfidf.transform(text_val)
    X_tfidf_te = tfidf.transform(text_test)

    # BoW
    bow = CountVectorizer(ngram_range=(1, 1), max_features=MAX_FEATURES, min_df=MIN_DF)
    X_bow_tr = bow.fit_transform(text_train)
    X_bow_vl = bow.transform(text_val)
    X_bow_te = bow.transform(text_test)

    # Word2Vec
    try:
        from gensim.models import Word2Vec
        corpus = [t.split() for t in text_train if t.strip()]
        w2v = Word2Vec(sentences=corpus, vector_size=W2V_DIM, window=W2V_WINDOW,
                       min_count=2, workers=4, epochs=W2V_EPOCHS, seed=RANDOM_STATE)
        X_w2v_tr = np.array([texte_vers_embedding(t, w2v, W2V_DIM) for t in text_train])
        X_w2v_vl = np.array([texte_vers_embedding(t, w2v, W2V_DIM) for t in text_val])
        X_w2v_te = np.array([texte_vers_embedding(t, w2v, W2V_DIM) for t in text_test])
        joblib.dump(w2v, f"{PROCESSED_DIR}/word2vec_model.pkl")
        print("  Word2Vec entraîné.")
    except ImportError:
        print("  [WARN] gensim non disponible — Word2Vec ignoré")
        X_w2v_tr = X_w2v_vl = X_w2v_te = None
        w2v = None

    # FastText
    try:
        from gensim.models import FastText
        ft = FastText(sentences=corpus, vector_size=W2V_DIM, window=W2V_WINDOW,
                      min_count=2, workers=4, epochs=W2V_EPOCHS, seed=RANDOM_STATE,
                      min_n=params["nlp"]["ft_min_n"], max_n=params["nlp"]["ft_max_n"])
        X_ft_tr = np.array([texte_vers_embedding(t, ft, W2V_DIM) for t in text_train])
        X_ft_vl = np.array([texte_vers_embedding(t, ft, W2V_DIM) for t in text_val])
        X_ft_te = np.array([texte_vers_embedding(t, ft, W2V_DIM) for t in text_test])
        joblib.dump(ft, f"{PROCESSED_DIR}/fasttext_model.pkl")
        print("  FastText entraîné.")
    except ImportError:
        print("  [WARN] gensim non disponible — FastText ignoré")
        X_ft_tr = X_ft_vl = X_ft_te = None
        ft = None

    # ── Comparaison 4×4 ─────────────────────────────────────────────────────
    vectorisations = {
        "BoW": (X_bow_tr, X_bow_vl, X_bow_te),
        "TF-IDF": (X_tfidf_tr, X_tfidf_vl, X_tfidf_te),
    }
    if X_w2v_tr is not None:
        vectorisations["Word2Vec"] = (X_w2v_tr, X_w2v_vl, X_w2v_te)
    if X_ft_tr is not None:
        vectorisations["FastText"] = (X_ft_tr, X_ft_vl, X_ft_te)

    resultats = []
    mlflow.set_experiment("eco_smart_nlp")

    for vec_name, (Xtr, Xvl, Xte) in vectorisations.items():
        classifieurs = {
            "LinearSVC": LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
            "LogReg": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE, n_jobs=-1),
        }
        if issparse(Xtr):
            classifieurs["NaiveBayes"] = MultinomialNB()

        for clf_name, clf in classifieurs.items():
            clf.fit(Xtr, y_train)
            acc_val = accuracy_score(y_val, clf.predict(Xvl))
            acc_test = accuracy_score(y_test, clf.predict(Xte))
            f1_test = f1_score(y_test, clf.predict(Xte), average="weighted")

            with mlflow.start_run(run_name=f"NLP_{vec_name}_{clf_name}"):
                mlflow.log_params({"vectorisation": vec_name, "classifieur": clf_name})
                mlflow.log_metrics({
                    "accuracy_val": acc_val,
                    "accuracy_test": acc_test,
                    "f1_test": f1_test,
                })

            resultats.append({
                "Vectorisation": vec_name, "Classifieur": clf_name,
                "Acc_Val": round(acc_val, 4), "Acc_Test": round(acc_test, 4),
                "F1_Test": round(f1_test, 4),
            })
            print(f"  {vec_name:<10} × {clf_name:<14} | Val={acc_val:.3f} | Test={acc_test:.3f} | F1={f1_test:.3f}")

    df_nlp = pd.DataFrame(resultats).sort_values("F1_Test", ascending=False)
    best = df_nlp.iloc[0]
    print(f"\n🏆 Meilleure combinaison : {best['Vectorisation']} + {best['Classifieur']} (F1={best['F1_Test']:.4f})")

    # Heatmap
    try:
        import seaborn as sns
        pivot = df_nlp.pivot_table(index="Classifieur", columns="Vectorisation", values="F1_Test")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, vmin=0.4, vmax=1.0)
        ax.set_title("F1 Score — Vectorisation × Classifieur", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{PROCESSED_DIR}/nlp_heatmap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Sauvegardes
    joblib.dump(tfidf, f"{PROCESSED_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(bow, f"{PROCESSED_DIR}/bow_vectorizer.pkl")

    # Réentraîner le meilleur classifieur
    vec_map = {
        "BoW": (X_bow_tr, X_bow_te),
        "TF-IDF": (X_tfidf_tr, X_tfidf_te),
        "Word2Vec": (X_w2v_tr, X_w2v_te) if X_w2v_tr is not None else (X_tfidf_tr, X_tfidf_te),
        "FastText": (X_ft_tr, X_ft_te) if X_ft_tr is not None else (X_tfidf_tr, X_tfidf_te),
    }
    clf_map = {
        "LinearSVC": LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
        "LogReg": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "NaiveBayes": MultinomialNB(),
    }
    Xtr_b, Xte_b = vec_map[best["Vectorisation"]]
    clf_b = clf_map[best["Classifieur"]]
    clf_b.fit(Xtr_b, y_train)
    joblib.dump(clf_b, f"{PROCESSED_DIR}/nlp_best_classifier.pkl")

    metrics = {
        "best_vectorisation": best["Vectorisation"],
        "best_classifieur": best["Classifieur"],
        "f1_test": float(best["F1_Test"]),
        "accuracy_test": float(best["Acc_Test"]),
        "n_combinations": len(resultats),
    }
    with open(f"{PROCESSED_DIR}/nlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ NLP terminé")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
