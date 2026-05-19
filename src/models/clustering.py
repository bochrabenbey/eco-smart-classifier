"""
Module 3 : Clustering non-supervisé
=====================================
- K-Means avec méthode du coude (Elbow)
- PCA 2D pour visualisation
- Métriques : Silhouette, ARI, NMI
"""

import json
import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_DIR = params["data"]["processed_dir"]
RANDOM_STATE = params["data"]["random_state"]
K_MIN = params["clustering"]["k_min"]
K_MAX = params["clustering"]["k_max"]

FEATURES_CLU = ["Poids", "Volume", "Conductivite", "Opacite",
                "Rigidite", "Densite_estimee", "Source_enc"]


def load_all_data():
    train = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    test = pd.read_csv(f"{PROCESSED_DIR}/test.csv")
    df_all = pd.concat([train, test], ignore_index=True)
    return df_all


def main():
    print("=" * 60)
    print("MODULE 3 — Clustering")
    print("=" * 60)

    df_all = load_all_data()

    feats = [f for f in FEATURES_CLU if f in df_all.columns]
    X_clu = df_all[feats].fillna(df_all[feats].median()).values

    # ── Méthode du coude ──
    inertias = []
    sil_scores = []
    K_range = range(K_MIN, K_MAX + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_clu)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_clu, labels)
        sil_scores.append(sil)
        print(f"  K={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    # Choix automatique du K optimal (max silhouette)
    K_OPTIMAL = K_range[np.argmax(sil_scores)]
    print(f"\n→ K optimal (max silhouette) : {K_OPTIMAL}")

    # Courbe coude
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(K_range), inertias, "bo-")
    axes[0].set_title("Méthode du coude (Elbow)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertie (WCSS)")
    axes[0].axvline(x=K_OPTIMAL, color="red", linestyle="--", label=f"K={K_OPTIMAL}")
    axes[0].legend()

    axes[1].plot(list(K_range), sil_scores, "go-")
    axes[1].set_title("Score de silhouette")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].axvline(x=K_OPTIMAL, color="red", linestyle="--", label=f"K={K_OPTIMAL}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/elbow_curve.png", dpi=150)
    plt.close()
    print(f"  Courbe coude sauvegardée.")

    # ── K-Means final ──
    kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clu)
    df_all["Cluster"] = cluster_labels

    # ── PCA 2D ──
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_clu)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    var_total = var1 + var2
    print(f"  Variance expliquée PCA 2D : {var_total:.1f}%")

    # ── Visualisation clusters vs catégories ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    couleurs = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    ax = axes[0]
    for i in range(K_OPTIMAL):
        mask = cluster_labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=couleurs[i % len(couleurs)], label=f"Cluster {i}",
                   alpha=0.5, s=15, edgecolors="none")
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c="black", marker="X", s=250, zorder=5, label="Centroïdes")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("Clusters K-Means (PCA 2D)", fontweight="bold")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    couleurs_cat = {"Métal": "#8E44AD", "Papier": "#1ABC9C",
                    "Plastique": "#E67E22", "Verre": "#2980B9",
                    "Non labellisé": "#95A5A6"}
    categories = df_all["Categorie"].fillna("Non labellisé") if "Categorie" in df_all.columns \
        else pd.Series(["Non labellisé"] * len(df_all))
    for cat, couleur in couleurs_cat.items():
        m = categories == cat
        if m.any():
            ax2.scatter(X_pca[m, 0], X_pca[m, 1], c=couleur, label=cat,
                        alpha=0.5, s=15, edgecolors="none")
    ax2.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax2.set_title("Catégories réelles (PCA 2D)", fontweight="bold")
    ax2.legend(fontsize=8)

    plt.suptitle("K-Means vs Catégories réelles dans l'espace PCA",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/pca_clusters.png", dpi=150)
    plt.close()

    # ── Métriques ──
    sil = silhouette_score(X_clu, cluster_labels)

    ari, nmi = 0.0, 0.0
    if "Categorie_enc" in df_all.columns:
        le = joblib.load(f"{PROCESSED_DIR}/label_encoder.pkl")
        df_lab = df_all[df_all["Categorie_enc"].notna()].copy()
        if len(df_lab) > 0:
            y_true_enc = df_lab["Categorie_enc"].astype(int).values
            y_pred_clu = df_lab["Cluster"].values
            ari = adjusted_rand_score(y_true_enc, y_pred_clu)
            nmi = normalized_mutual_info_score(y_true_enc, y_pred_clu)

    print(f"\n  Silhouette Score : {sil:.4f}")
    print(f"  ARI              : {ari:.4f}")
    print(f"  NMI              : {nmi:.4f}")

    # ── Sauvegarde ──
    joblib.dump(kmeans, f"{PROCESSED_DIR}/kmeans_model.pkl")
    joblib.dump(pca, f"{PROCESSED_DIR}/pca_model.pkl")

    df_pca_export = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": cluster_labels,
        "Categorie": df_all.get("Categorie", pd.Series(["?"] * len(df_all))).fillna("Non labellisé").values,
    })
    df_pca_export.to_csv(f"{PROCESSED_DIR}/pca_clusters.csv", index=False)

    metrics = {
        "k_optimal": int(K_OPTIMAL),
        "silhouette": round(float(sil), 4),
        "ari": round(float(ari), 4),
        "nmi": round(float(nmi), 4),
        "pca_variance_2d": round(float(var_total), 2),
    }
    with open(f"{PROCESSED_DIR}/clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import mlflow
    mlflow.set_experiment("eco_smart_clustering")
    with mlflow.start_run(run_name="KMeans_K3_PCA"):
        mlflow.log_params({
            "k_optimal": int(K_OPTIMAL),
            "k_range": f"{K_MIN}-{K_MAX}",
            "features": str(FEATURES_CLU),
        })
        mlflow.log_metrics({
            "silhouette": float(sil),
            "ari":        float(ari),
            "nmi":        float(nmi),
            "pca_variance_2d": float(var_total),
        })

    print(f"\n✅ Clustering terminé")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
