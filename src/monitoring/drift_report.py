"""
Module 6 : Monitoring — Détection de drift avec Evidently
===========================================================
- Data drift numérique (Evidently)
- Text drift (Jensen-Shannon divergence)
- Alertes de performance
- Export rapport HTML
"""

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
REPORTS_DIR = "data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

NUMERIC_FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
MIN_ACCURACY_ALERT = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# DATA DRIFT NUMÉRIQUE
# ══════════════════════════════════════════════════════════════════════════════
def compute_numeric_drift(df_ref: pd.DataFrame, df_cur: pd.DataFrame) -> dict:
    """
    Calcule le drift numérique via test KS pour chaque feature.
    Retourne un dict avec p-value et flag de drift.
    """
    results = {}
    for col in NUMERIC_FEATURES:
        if col not in df_ref.columns or col not in df_cur.columns:
            continue
        ref_vals = df_ref[col].dropna().values
        cur_vals = df_cur[col].dropna().values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        stat, pval = ks_2samp(ref_vals, cur_vals)
        drifted = pval < 0.05
        results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pval), 4),
            "drift_detected": drifted,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEXT DRIFT — Jensen-Shannon Divergence
# ══════════════════════════════════════════════════════════════════════════════
def compute_text_drift(texts_ref: list, texts_cur: list,
                       vocab_size: int = 1000) -> dict:
    """
    Calcule la divergence Jensen-Shannon entre distributions de tokens.
    """
    from collections import Counter

    def token_distribution(texts, vocab):
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        # Normaliser sur le vocabulaire fixe
        dist = np.array([counts.get(w, 0) for w in vocab], dtype=float)
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    # Construire le vocabulaire commun
    all_tokens = []
    for t in texts_ref + texts_cur:
        all_tokens.extend(t.lower().split())
    from collections import Counter
    vocab = [w for w, _ in Counter(all_tokens).most_common(vocab_size)]

    dist_ref = token_distribution(texts_ref, vocab)
    dist_cur = token_distribution(texts_cur, vocab)

    # Éviter les zéros (lissage)
    dist_ref += 1e-10
    dist_cur += 1e-10
    dist_ref /= dist_ref.sum()
    dist_cur /= dist_cur.sum()

    js_div = float(jensenshannon(dist_ref, dist_cur))
    return {
        "jensen_shannon_divergence": round(js_div, 4),
        "drift_detected": js_div > 0.1,  # seuil empirique
        "vocab_size": len(vocab),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT COMPLET
# ══════════════════════════════════════════════════════════════════════════════
def generate_drift_report(
    df_ref: Optional[pd.DataFrame] = None,
    df_cur: Optional[pd.DataFrame] = None,
    texts_ref: Optional[list] = None,
    texts_cur: Optional[list] = None,
    current_accuracy: Optional[float] = None,
) -> dict:
    """
    Génère un rapport de drift complet.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "numeric_drift": {},
        "text_drift": {},
        "performance_alert": None,
    }

    # Drift numérique
    if df_ref is not None and df_cur is not None:
        report["numeric_drift"] = compute_numeric_drift(df_ref, df_cur)
        drifted_cols = [k for k, v in report["numeric_drift"].items() if v["drift_detected"]]
        report["numeric_drift_summary"] = {
            "n_features_drifted": len(drifted_cols),
            "drifted_features": drifted_cols,
        }
        if drifted_cols:
            print(f"[DRIFT] ⚠️  Drift numérique détecté sur : {drifted_cols}")
        else:
            print("[DRIFT] ✅ Pas de drift numérique détecté")

    # Drift textuel
    if texts_ref and texts_cur:
        report["text_drift"] = compute_text_drift(texts_ref, texts_cur)
        if report["text_drift"]["drift_detected"]:
            print(f"[DRIFT] ⚠️  Drift textuel détecté (JS={report['text_drift']['jensen_shannon_divergence']:.4f})")
        else:
            print("[DRIFT] ✅ Pas de drift textuel détecté")

    # Alerte performance
    if current_accuracy is not None:
        below_threshold = current_accuracy < MIN_ACCURACY_ALERT
        report["performance_alert"] = {
            "current_accuracy": round(current_accuracy, 4),
            "threshold": MIN_ACCURACY_ALERT,
            "alert_triggered": below_threshold,
        }
        if below_threshold:
            print(f"[DRIFT] 🚨 ALERTE : accuracy={current_accuracy:.4f} < seuil={MIN_ACCURACY_ALERT}")

    # Sauvegarde JSON
    report_path = f"{REPORTS_DIR}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[DRIFT] Rapport sauvegardé : {report_path}")

    # Tentative de rapport Evidently
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        evidently_report = Report(metrics=[DataDriftPreset()])
        evidently_report.run(reference_data=df_ref, current_data=df_cur)
        html_path = f"{REPORTS_DIR}/evidently_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        evidently_report.save_html(html_path)
        report["evidently_html"] = html_path
        print(f"[DRIFT] Rapport Evidently : {html_path}")
    except Exception as e:
        print(f"[DRIFT] Evidently non disponible : {e}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — exemple d'utilisation
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 6 — Monitoring / Drift Detection")
    print("=" * 60)

    # Charger les données de référence (train) et simulation (test)
    try:
        df_ref = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
        df_cur = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

        texts_ref = df_ref["Rapport_Collecte"].fillna("").tolist() \
            if "Rapport_Collecte" in df_ref.columns else []
        texts_cur = df_cur["Rapport_Collecte"].fillna("").tolist() \
            if "Rapport_Collecte" in df_cur.columns else []

        report = generate_drift_report(
            df_ref=df_ref[NUMERIC_FEATURES + ["Poids"]].head(1000),
            df_cur=df_cur[NUMERIC_FEATURES + ["Poids"]].head(1000),
            texts_ref=texts_ref[:500],
            texts_cur=texts_cur[:500],
            current_accuracy=0.85,  # Simulé
        )
        print("\n✅ Rapport de monitoring généré")
    except Exception as e:
        print(f"Erreur : {e}")
