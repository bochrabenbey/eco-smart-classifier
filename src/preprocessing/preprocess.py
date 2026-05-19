"""
Module 1 : Exploration, Nettoyage et Analyse des données
=========================================================
- Gestion des valeurs manquantes (Médiane / KNN / IterativeImputer)
- Traitement des outliers (IQR)
- Feature engineering : normalisation / standardisation
- Encodage des variables catégorielles
- Split stratifié train/val/test (70/15/15)
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Chargement des paramètres ──────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data"]["raw_path"]
PROCESSED_DIR = params["data"]["processed_dir"]
RANDOM_STATE = params["data"]["random_state"]
TEST_SIZE = params["data"]["test_size"]
VAL_SIZE = params["data"]["val_size"]

NUMERIC_COLS = params["preprocessing"]["numeric_cols"]
CAT_COLS = params["preprocessing"]["cat_cols"]
KNN_K = params["preprocessing"]["knn_neighbors"]
ITER_MAX = params["preprocessing"]["iterative_max_iter"]
OUTLIER_FACTOR = params["preprocessing"]["outlier_factor"]

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[LOAD] {len(df)} lignes × {len(df.columns)} colonnes")
    print(f"[LOAD] Colonnes : {df.columns.tolist()}")
    print(f"[LOAD] NaN totaux : {df.isnull().sum().sum()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSE DES MÉCANISMES DE MANQUANCE
# ══════════════════════════════════════════════════════════════════════════════
def analyze_missing(df: pd.DataFrame) -> dict:
    """
    Analyse le type de manquance pour chaque colonne.
    Retourne un dict {col: type} où type ∈ {MCAR, MAR, MNAR}.
    """
    results = {}
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]

    for col in num_cols:
        if df[col].isna().sum() == 0:
            continue
        missing_mask = df[col].isna()
        # Corrélation avec les autres colonnes → si corrélée : MAR
        corr_max = 0
        for other in num_cols:
            if other == col or df[other].isna().all():
                continue
            c = abs(df[other].corr(missing_mask.astype(float)))
            if not np.isnan(c):
                corr_max = max(corr_max, c)
        if corr_max > 0.3:
            results[col] = "MAR"
        else:
            results[col] = "MCAR"

    # Categorie : 514 labels manquants → supposés MNAR (censure intentionnelle)
    if "Categorie" in df.columns and df["Categorie"].isna().sum() > 0:
        results["Categorie"] = "MNAR"

    print("[MISSING] Mécanismes détectés :")
    for k, v in results.items():
        pct = df[k].isna().mean() * 100
        print(f"  {k:<20} {v}  ({pct:.1f}% manquant)")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. IMPUTATION — COMPARAISON DES MÉTHODES
# ══════════════════════════════════════════════════════════════════════════════
def compare_imputers(df: pd.DataFrame, num_cols: list) -> tuple:
    """
    Compare Médiane / KNN / IterativeImputer via simulation MCAR 10%.
    Retourne (meilleure_méthode, df_imputé, rmse_dict).
    """
    df_original = df.copy()

    # -- Imputation Médiane --
    df_median = df_original.copy()
    imp_med = SimpleImputer(strategy="median")
    df_median[num_cols] = imp_med.fit_transform(df_median[num_cols])

    # -- Imputation KNN --
    df_knn = df_original.copy()
    imp_knn = KNNImputer(n_neighbors=KNN_K)
    df_knn[num_cols] = imp_knn.fit_transform(df_knn[num_cols])

    # -- Imputation IterativeImputer (MICE) --
    df_iter = df_original.copy()
    imp_iter = IterativeImputer(max_iter=ITER_MAX, random_state=RANDOM_STATE)
    df_iter[num_cols] = imp_iter.fit_transform(df_iter[num_cols])

    # -- Évaluation MCAR 10% --
    np.random.seed(RANDOM_STATE)
    df_ref = df_median.copy()
    mask = np.random.random(df_ref[num_cols].shape) < 0.10
    df_masked = df_ref.copy()
    df_masked[num_cols] = df_ref[num_cols].mask(
        pd.DataFrame(mask, columns=num_cols)
    )

    rmse = {}
    for name, imp in [
        ("Médiane", SimpleImputer(strategy="median")),
        ("KNN", KNNImputer(n_neighbors=KNN_K)),
        ("IterativeImputer", IterativeImputer(max_iter=ITER_MAX, random_state=RANDOM_STATE)),
    ]:
        imputed = imp.fit_transform(df_masked[num_cols])
        r = np.sqrt(mean_squared_error(df_ref[num_cols].values[mask], imputed[mask]))
        rmse[name] = round(float(r), 4)

    best = min(rmse, key=rmse.get)
    print(f"\n[IMPUTATION] RMSE par méthode :")
    for k, v in rmse.items():
        flag = "✅" if k == best else "  "
        print(f"  {flag} {k:<20} RMSE = {v:.4f}")
    print(f"  → Méthode retenue : {best}")

    df_map = {"Médiane": df_median, "KNN": df_knn, "IterativeImputer": df_iter}
    return best, df_map[best], rmse


# ══════════════════════════════════════════════════════════════════════════════
# 4. OUTLIERS — IQR
# ══════════════════════════════════════════════════════════════════════════════
def treat_outliers(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Cap les outliers au seuil IQR × facteur."""
    df = df.copy()
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - OUTLIER_FACTOR * IQR
        high = Q3 + OUTLIER_FACTOR * IQR
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        df[col] = df[col].clip(low, high)
        if n_out > 0:
            print(f"  [OUTLIERS] {col}: {n_out} outliers cappés [{low:.2f}, {high:.2f}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENCODAGE & NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
def encode_and_scale(df: pd.DataFrame) -> tuple:
    """
    Encode Source + Categorie, standardise les features numériques.
    Retourne (df_encoded, label_encoder, source_encoder, scaler).
    """
    df = df.copy()

    # Encodage Source (LabelEncoder)
    le_source = LabelEncoder()
    df["Source"] = df["Source"].fillna("Unknown")
    df["Source_enc"] = le_source.fit_transform(df["Source"])

    # Encodage Categorie (LabelEncoder — uniquement sur les lignes labellisées)
    le_cat = LabelEncoder()
    mask_labeled = df["Categorie"].notna()
    le_cat.fit(df.loc[mask_labeled, "Categorie"])
    df.loc[mask_labeled, "Categorie_enc"] = le_cat.transform(
        df.loc[mask_labeled, "Categorie"]
    )
    df["Categorie_enc"] = df["Categorie_enc"].astype("Int64")

    # Feature engineering : ratio Poids/Volume
    df["Densite_estimee"] = df["Poids"] / (df["Volume"].replace(0, np.nan) + 1e-6)
    df["Densite_estimee"] = df["Densite_estimee"].fillna(df["Densite_estimee"].median())

    # StandardScaler sur les features numériques (excl. cibles)
    SCALE_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Densite_estimee"]
    scaler = StandardScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])

    print(f"[ENCODE] Classes Categorie : {le_cat.classes_.tolist()}")
    print(f"[ENCODE] Classes Source    : {le_source.classes_.tolist()}")

    return df, le_cat, le_source, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 6. SPLIT TRAIN / VAL / TEST (stratifié)
# ══════════════════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame) -> tuple:
    """
    Split stratifié 70/15/15 sur les lignes labellisées uniquement.
    Les lignes non-labellisées (Categorie_enc = NaN) vont dans train.
    """
    df_labeled = df[df["Categorie_enc"].notna()].copy()
    df_unlabeled = df[df["Categorie_enc"].isna()].copy()

    X = df_labeled.drop(columns=["Categorie"], errors="ignore")
    y = df_labeled["Categorie_enc"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    rel_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - rel_val,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # Ajouter les non-labellisés au train
    train_df = pd.concat([X_train, df_unlabeled], ignore_index=True)
    val_df = X_val.copy()
    test_df = X_test.copy()

    print(f"\n[SPLIT] Train   : {len(train_df)} lignes (dont {len(df_unlabeled)} non-labellisés)")
    print(f"[SPLIT] Val     : {len(val_df)} lignes")
    print(f"[SPLIT] Test    : {len(test_df)} lignes")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("MODULE 1 — Preprocessing")
    print("=" * 60)

    # 1. Chargement
    df = load_data(RAW_PATH)

    # 2. Analyse manquance
    analyze_missing(df)

    # 3. Colonnes numériques (excl. Prix_Revente = cible régression)
    NUM_COLS_IMP = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]

    # 4. Imputation
    best_method, df_imputed, rmse_dict = compare_imputers(df, NUM_COLS_IMP)

    # Imputer les colonnes catégorielles
    for col in ["Source"]:
        if df_imputed[col].isna().any():
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])

    # Imputer Prix_Revente séparément (Médiane)
    df_imputed["Prix_Revente"] = df_imputed["Prix_Revente"].fillna(
        df_imputed["Prix_Revente"].median()
    )

    # 5. Outliers
    print("\n[OUTLIERS]")
    df_clean = treat_outliers(df_imputed, NUM_COLS_IMP + ["Prix_Revente"])

    # 6. Encodage & normalisation
    print("\n[ENCODE & SCALE]")
    df_final, le_cat, le_source, scaler = encode_and_scale(df_clean)

    # 7. Split
    train_df, val_df, test_df = split_data(df_final)

    # 8. Sauvegarde
    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)
    joblib.dump(le_cat, f"{PROCESSED_DIR}/label_encoder.pkl")
    joblib.dump(le_source, f"{PROCESSED_DIR}/source_encoder.pkl")
    joblib.dump(scaler, f"{PROCESSED_DIR}/scaler.pkl")

    metrics = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "imputation_method": best_method,
        "imputation_rmse": rmse_dict,
        "nan_remaining": int(df_final.isnull().sum().sum()),
        "n_classes": int(len(le_cat.classes_)),
    }
    with open(f"{PROCESSED_DIR}/preprocessing_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Preprocessing terminé — fichiers sauvegardés dans {PROCESSED_DIR}/")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
