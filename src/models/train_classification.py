"""
Module 2 : Classification supervisée
=====================================
- Modèles : LogisticRegression, RandomForest, GradientBoosting, SVC
- Tuning : GridSearchCV / Optuna
- SHAP values pour la sélection de features
- Tracking MLflow
"""

import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_DIR = params["data"]["processed_dir"]
RANDOM_STATE = params["data"]["random_state"]
CV_FOLDS = params["models"]["classification"]["cv_folds"]
MIN_ACC = params["models"]["classification"]["min_accuracy"]

FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
            "Source_enc", "Densite_estimee"]
TARGET = "Categorie_enc"


def load_splits():
    train = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    val = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    test = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    # Garder uniquement les lignes labellisées
    train = train[train[TARGET].notna()].copy()
    val = val[val[TARGET].notna()].copy()
    test = test[test[TARGET].notna()].copy()

    for df in [train, val, test]:
        df[TARGET] = df[TARGET].astype(int)

    # Remplir NaN résiduels dans les features
    for df in [train, val, test]:
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    return train, val, test


def get_models_and_grids():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5]},
        ),
        "SVC": (
            SVC(random_state=RANDOM_STATE, probability=True),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        ),
    }


def run_shap_analysis(model, X_train, feature_names, model_name):
    """Analyse SHAP pour l'interprétabilité du modèle."""
    try:
        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model) if "Forest" in model_name or "Boosting" in model_name \
                else shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train[:500])
        if isinstance(shap_values, list):
            importance = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "importance": importance})
        shap_df = shap_df.sort_values("importance", ascending=False)
        shap_df.to_csv(f"{PROCESSED_DIR}/shap_{model_name}.csv", index=False)
        print(f"  [SHAP] Top features :")
        for _, row in shap_df.head(5).iterrows():
            print(f"    {row['feature']:<25} {row['importance']:.4f}")
    except Exception as e:
        print(f"  [SHAP] Impossible pour {model_name}: {e}")


def main():
    print("=" * 60)
    print("MODULE 2 — Classification")
    print("=" * 60)

    train, val, test = load_splits()

    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_val = val[FEATURES].values
    y_val = val[TARGET].values
    X_test = test[FEATURES].values
    y_test = test[TARGET].values

    le = joblib.load(f"{PROCESSED_DIR}/label_encoder.pkl")
    class_names = le.classes_

    mlflow.set_experiment("eco_smart_classification")
    results = []

    models_grids = get_models_and_grids()

    for model_name, (model, param_grid) in models_grids.items():
        print(f"\n[TRAIN] {model_name}")
        with mlflow.start_run(run_name=f"CLF_{model_name}"):
            # GridSearchCV
            gs = GridSearchCV(
                model, param_grid,
                cv=CV_FOLDS, scoring="f1_weighted",
                n_jobs=-1, verbose=0
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_

            y_pred_val = best.predict(X_val)
            y_pred_test = best.predict(X_test)

            acc_val = accuracy_score(y_val, y_pred_val)
            acc_test = accuracy_score(y_test, y_pred_test)
            f1_val = f1_score(y_val, y_pred_val, average="weighted")
            f1_test = f1_score(y_test, y_pred_test, average="weighted")

            print(f"  Best params : {gs.best_params_}")
            print(f"  Val  : acc={acc_val:.4f}  f1={f1_val:.4f}")
            print(f"  Test : acc={acc_test:.4f}  f1={f1_test:.4f}")

            # MLflow
            mlflow.log_params({"model": model_name, **gs.best_params_})
            mlflow.log_metrics({
                "accuracy_val": acc_val, "f1_val": f1_val,
                "accuracy_test": acc_test, "f1_test": f1_test,
            })
            mlflow.sklearn.log_model(best, model_name)

            results.append({
                "model": model_name,
                "best_params": str(gs.best_params_),
                "acc_val": round(acc_val, 4),
                "f1_val": round(f1_val, 4),
                "acc_test": round(acc_test, 4),
                "f1_test": round(f1_test, 4),
            })

            # SHAP sur le modèle adapté
            run_shap_analysis(best, X_train, FEATURES, model_name)

    # Meilleur modèle
    df_results = pd.DataFrame(results).sort_values("f1_test", ascending=False)
    best_row = df_results.iloc[0]
    print(f"\n🏆 Meilleur modèle : {best_row['model']} (F1={best_row['f1_test']:.4f})")

    # Réentraîner le meilleur modèle sur train+val
    best_model_name = best_row["model"]
    best_model_obj = get_models_and_grids()[best_model_name][0]
    gs_final = GridSearchCV(
        best_model_obj,
        get_models_and_grids()[best_model_name][1],
        cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1
    )
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    gs_final.fit(X_trainval, y_trainval)
    final_model = gs_final.best_estimator_

    # Rapport de classification
    y_pred_final = final_model.predict(X_test)
    print("\n" + classification_report(y_test, y_pred_final, target_names=class_names))

    assert accuracy_score(y_test, y_pred_final) >= MIN_ACC, \
        f"Accuracy ({accuracy_score(y_test, y_pred_final):.4f}) < seuil minimum ({MIN_ACC})"

    # Enregistrement au Model Registry
    with mlflow.start_run(run_name="BestClassifier_Registry"):
        mlflow.log_params({"model": best_model_name, "final_train": "train+val"})
        mlflow.log_metrics({
            "accuracy_test": accuracy_score(y_test, y_pred_final),
            "f1_test": f1_score(y_test, y_pred_final, average="weighted"),
        })
        mlflow.sklearn.log_model(
        final_model,
        artifact_path="best_classifier",
        registered_model_name="EcoSmart_Classifier",
            )
    # Sauvegarde
    joblib.dump(final_model, f"{PROCESSED_DIR}/best_classifier.pkl")
    df_results.to_csv(f"{PROCESSED_DIR}/classification_results.csv", index=False)

    metrics = {
        "best_model": best_model_name,
        "accuracy_test": round(float(accuracy_score(y_test, y_pred_final)), 4),
        "f1_test": round(float(f1_score(y_test, y_pred_final, average="weighted")), 4),
        "n_models_compared": len(results),
    }
    with open(f"{PROCESSED_DIR}/classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Classification terminée — modèle sauvegardé")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
