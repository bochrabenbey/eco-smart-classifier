"""
Module 2 : Régression supervisée — Prédiction de Prix_Revente
==============================================================
"""

import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_DIR = params["data"]["processed_dir"]
RANDOM_STATE = params["data"]["random_state"]
CV_FOLDS = params["models"]["regression"]["cv_folds"]

FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
            "Source_enc", "Categorie_enc", "Densite_estimee"]
TARGET = "Prix_Revente"


def load_splits():
    train = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    test = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    for df in [train, test]:
        df.dropna(subset=[TARGET, "Categorie_enc"], inplace=True)
        df["Categorie_enc"] = df["Categorie_enc"].astype(int)
        for col in FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    return train, test


def main():
    print("=" * 60)
    print("MODULE 2 — Régression (Prix_Revente)")
    print("=" * 60)

    train, test = load_splits()

    feats = [f for f in FEATURES if f in train.columns]
    X_train = train[feats].values
    y_train = train[TARGET].values
    X_test = test[feats].values
    y_test = test[TARGET].values

    mlflow.set_experiment("eco_smart_regression")
    results = []

    models_grids = {
        "Ridge": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [None, 10]},
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        ),
    }

    for model_name, (model, param_grid) in models_grids.items():
        print(f"\n[TRAIN] {model_name}")
        with mlflow.start_run(run_name=f"REG_{model_name}"):
            gs = GridSearchCV(model, param_grid, cv=CV_FOLDS,
                              scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_

            y_pred = best.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"  Best params : {gs.best_params_}")
            print(f"  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

            mlflow.log_params({"model": model_name, **gs.best_params_})
            mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse})
            mlflow.sklearn.log_model(best, model_name)

            results.append({
                "model": model_name,
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
            })

    df_results = pd.DataFrame(results).sort_values("r2", ascending=False)
    best_row = df_results.iloc[0]
    print(f"\n🏆 Meilleur régresseur : {best_row['model']} (R²={best_row['r2']:.4f})")

    # Réentraîner le meilleur
    best_model_name = best_row["model"]
    gs_final = GridSearchCV(
        models_grids[best_model_name][0],
        models_grids[best_model_name][1],
        cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    gs_final.fit(X_train, y_train)
    final_model = gs_final.best_estimator_

    joblib.dump(final_model, f"{PROCESSED_DIR}/best_regressor.pkl")
    df_results.to_csv(f"{PROCESSED_DIR}/regression_results.csv", index=False)

    final_pred = final_model.predict(X_test)
    metrics = {
        "best_model": best_model_name,
        "r2": round(float(r2_score(y_test, final_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, final_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, final_pred))), 4),
    }
    with open(f"{PROCESSED_DIR}/regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Régression terminée")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
