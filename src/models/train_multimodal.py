"""
Module 5 : Pipeline Multimodal — Fusion Numérique + NLP
=========================================================
- Fusion par hstack (concaténation sparse)
- ColumnTransformer sklearn (reproductible)
- Pondérations NLP / numérique
- Stacking de modèles
- Tracking MLflow + Model Registry
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
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nlp.text_preprocessing import preprocess_text

warnings.filterwarnings("ignore")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_DIR = params["data"]["processed_dir"]
RAW_PATH = params["data"]["raw_path"]
RANDOM_STATE = params["data"]["random_state"]
MAX_FEATURES = params["nlp"]["max_features"]

FEATURES_NUM = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
                "Source_enc", "Densite_estimee"]
TARGET = "Categorie_enc"


def load_data():
    df_tr = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    df_vl = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    df_te = pd.read_csv(f"{PROCESSED_DIR}/test.csv")
    df_raw = pd.read_csv(RAW_PATH)

    df_raw["Rapport_clean"] = df_raw["Rapport_Collecte"].apply(preprocess_text)
    mapping = df_raw[["Rapport_Collecte", "Rapport_clean"]].drop_duplicates()

    for df in [df_tr, df_vl, df_te]:
        df_m = df.merge(mapping, on="Rapport_Collecte", how="left")
        df["Rapport_clean"] = df_m["Rapport_clean"].fillna("").values

    for df in [df_tr, df_vl, df_te]:
        df.dropna(subset=[TARGET], inplace=True)
        df[TARGET] = df[TARGET].astype(int)
        for col in FEATURES_NUM:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    return df_tr, df_vl, df_te


def main():
    print("=" * 60)
    print("MODULE 5 — Pipeline Multimodal")
    print("=" * 60)

    df_tr, df_vl, df_te = load_data()

    feats = [f for f in FEATURES_NUM if f in df_tr.columns]

    X_num_tr = df_tr[feats].values
    X_num_vl = df_vl[feats].values
    X_num_te = df_te[feats].values

    text_tr = df_tr["Rapport_clean"].values
    text_vl = df_vl["Rapport_clean"].values
    text_te = df_te["Rapport_clean"].values

    y_tr = df_tr[TARGET].values
    y_vl = df_vl[TARGET].values
    y_te = df_te[TARGET].values

    # ── TF-IDF ──
    tfidf_mm = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES,
                               min_df=2, sublinear_tf=True)
    X_tfidf_tr = tfidf_mm.fit_transform(text_tr)
    X_tfidf_vl = tfidf_mm.transform(text_vl)
    X_tfidf_te = tfidf_mm.transform(text_te)

    # ── hstack sparse ──
    X_num_tr_sp = csr_matrix(X_num_tr)
    X_num_vl_sp = csr_matrix(X_num_vl)
    X_num_te_sp = csr_matrix(X_num_te)

    X_multi_tr = hstack([X_num_tr_sp, X_tfidf_tr])
    X_multi_vl = hstack([X_num_vl_sp, X_tfidf_vl])
    X_multi_te = hstack([X_num_te_sp, X_tfidf_te])

    print(f"  Features numériques : {X_num_tr_sp.shape[1]}")
    print(f"  Features TF-IDF     : {X_tfidf_tr.shape[1]}")
    print(f"  Features multimodal : {X_multi_tr.shape[1]}")

    mlflow.set_experiment("eco_smart_multimodal")
    resultats = []

    # ── ColumnTransformer + LinearSVC ──
    df_ct_tr = df_tr[feats + ["Rapport_clean"]].copy()
    df_ct_vl = df_vl[feats + ["Rapport_clean"]].copy()
    df_ct_te = df_te[feats + ["Rapport_clean"]].copy()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), feats),
        ("nlp", TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES,
                                min_df=2, sublinear_tf=True), "Rapport_clean"),
    ], remainder="drop")

    pipeline_mm = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipeline_mm.fit(df_ct_tr, y_tr)
    y_pred_ct = pipeline_mm.predict(df_ct_te)
    acc_ct = accuracy_score(y_te, y_pred_ct)
    f1_ct = f1_score(y_te, y_pred_ct, average="weighted")
    print(f"\n  ColumnTransformer + LinearSVC : acc={acc_ct:.4f}  f1={f1_ct:.4f}")

    with mlflow.start_run(run_name="MM_ColumnTransformer_LinearSVC"):
        mlflow.log_params({"approche": "ColumnTransformer+LinearSVC"})
        mlflow.log_metrics({"accuracy_test": acc_ct, "f1_test": f1_ct})

    resultats.append({"Approche": "ColumnTransformer + LinearSVC",
                       "Accuracy": acc_ct, "F1": f1_ct})

    # ── Pondérations ──
    for nom, facteur in [("Num:NLP=1:1", 1.0), ("Num:NLP=2:1", 2.0), ("Num:NLP=1:2", 0.5)]:
        Xtr_p = hstack([X_num_tr_sp * facteur, X_tfidf_tr])
        Xte_p = hstack([X_num_te_sp * facteur, X_tfidf_te])
        clf_p = LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
        clf_p.fit(Xtr_p, y_tr)
        y_p = clf_p.predict(Xte_p)
        acc_p = accuracy_score(y_te, y_p)
        f1_p = f1_score(y_te, y_p, average="weighted")
        print(f"  Pondération {nom:<15} : acc={acc_p:.4f}  f1={f1_p:.4f}")
        resultats.append({"Approche": f"hstack ({nom})", "Accuracy": acc_p, "F1": f1_p})

    # ── Stacking ──
    print("\n  Stacking (SVC + LR + RF → LR méta)...")
    estimators_stack = [
        ("svc", LinearSVC(max_iter=2000, random_state=RANDOM_STATE)),
        ("lr", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    stacking_clf = StackingClassifier(
        estimators=estimators_stack,
        final_estimator=LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
        cv=3, n_jobs=-1,
    )
    X_dense_tr = X_multi_tr.toarray()
    X_dense_te = X_multi_te.toarray()
    stacking_clf.fit(X_dense_tr, y_tr)
    y_stack = stacking_clf.predict(X_dense_te)
    acc_stack = accuracy_score(y_te, y_stack)
    f1_stack = f1_score(y_te, y_stack, average="weighted")
    print(f"  Stacking : acc={acc_stack:.4f}  f1={f1_stack:.4f}")
    resultats.append({"Approche": "Stacking (SVC+LR+RF)", "Accuracy": acc_stack, "F1": f1_stack})

    # ── Comparaison ──
    # Modèle numérique seul
    try:
        clf_num = joblib.load(f"{PROCESSED_DIR}/best_classifier.pkl")
        y_num = clf_num.predict(X_num_te)
        acc_num = accuracy_score(y_te, y_num)
        f1_num = f1_score(y_te, y_num, average="weighted")
        resultats.append({"Approche": "Numérique seul (Module 2)",
                           "Accuracy": acc_num, "F1": f1_num})
    except Exception:
        f1_num = 0.0

    # Modèle NLP seul
    try:
        tfidf_nlp = joblib.load(f"{PROCESSED_DIR}/tfidf_vectorizer.pkl")
        clf_nlp = joblib.load(f"{PROCESSED_DIR}/nlp_best_classifier.pkl")
        X_nlp_te = tfidf_nlp.transform(text_te)
        y_nlp = clf_nlp.predict(X_nlp_te)
        acc_nlp = accuracy_score(y_te, y_nlp)
        f1_nlp = f1_score(y_te, y_nlp, average="weighted")
        resultats.append({"Approche": "NLP seul (Module 4)",
                           "Accuracy": acc_nlp, "F1": f1_nlp})
    except Exception:
        pass

    df_comp = pd.DataFrame(resultats).sort_values("F1", ascending=False).reset_index(drop=True)
    meilleur = df_comp.iloc[0]
    print(f"\n🏆 Meilleure approche : {meilleur['Approche']} (F1={meilleur['F1']:.4f})")

    # Enregistrer au Model Registry
    with mlflow.start_run(run_name="BestMultimodal_Registry"):
        mlflow.log_params({"approche": "ColumnTransformer+LinearSVC", "module": "Module5"})
        mlflow.log_metrics({"f1_test": f1_ct, "accuracy_test": acc_ct})
        mlflow.sklearn.log_model(
        pipeline_mm,
        artifact_path="multimodal_pipeline",
        registered_model_name="EcoSmart_Multimodal",
    )

    # Sauvegarde
    joblib.dump(tfidf_mm, f"{PROCESSED_DIR}/tfidf_multimodal.pkl")
    joblib.dump(pipeline_mm, f"{PROCESSED_DIR}/pipeline_multimodal.pkl")
    joblib.dump(stacking_clf, f"{PROCESSED_DIR}/stacking_classifier.pkl")
    df_comp.to_csv(f"{PROCESSED_DIR}/multimodal_results.csv", index=False)

    metrics = {
        "best_approche": str(meilleur["Approche"]),
        "f1_test": round(float(meilleur["F1"]), 4),
        "accuracy_test": round(float(meilleur["Accuracy"]), 4),
        "gain_vs_numerique": round(float(meilleur["F1"]) - float(f1_num), 4),
        "n_approaches": len(resultats),
    }
    with open(f"{PROCESSED_DIR}/multimodal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Multimodal terminé")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
