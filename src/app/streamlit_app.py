"""
Application Web — Eco-Smart Classifier v2.1
=============================================
Dark theme · Comparaison modèles · Historique · Scaler corrigé
"""

import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")

PALETTE = {
    "Métal":         {"color": "#60A5FA", "bg": "#1E3A5F", "icon": "🔩"},
    "Papier":        {"color": "#34D399", "bg": "#0D3D2E", "icon": "📄"},
    "Plastique":     {"color": "#F97316", "bg": "#3D1F0D", "icon": "♻️"},
    "Verre":         {"color": "#A78BFA", "bg": "#2D1F5E", "icon": "🫙"},
    "Non labellisé": {"color": "#6B7280", "bg": "#1F2937", "icon": "❓"},
}

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
  --bg-base:#0A0E1A;--bg-card:#111827;--bg-el:#1C2333;
  --border:#1F2D45;--accent:#22D3EE;--text:#F1F5F9;--muted:#64748B;
}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg-base)!important;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0D1220!important;border-right:1px solid var(--border)}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;display:none}
h1,h2,h3{font-family:'Space Mono',monospace!important;color:var(--text)!important}
p,li,label{font-family:'DM Sans',sans-serif!important;color:var(--text)!important}
.stMarkdown p{color:#CBD5E1!important}
[data-testid="metric-container"]{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:16px!important}
[data-testid="stMetricValue"]{font-family:'Space Mono',monospace!important;color:var(--accent)!important;font-size:1.8rem!important}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.75rem!important}
.stButton>button{background:linear-gradient(135deg,#0EA5E9,#22D3EE)!important;color:#0A0E1A!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-weight:700!important;font-size:.8rem!important;padding:10px 24px!important;transition:all .2s ease!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(34,211,238,.35)!important}
.stButton>button p,.stButton>button span,.stButton>button div{color:#0A0E1A!important}
div[data-testid="column"] .stButton>button{background:#1C2333!important;color:#F1F5F9!important;border:1px solid #334155!important;padding:6px 14px!important}
div[data-testid="column"] .stButton>button p,div[data-testid="column"] .stButton>button span,div[data-testid="column"] .stButton>button div{color:#F1F5F9!important}
div[data-testid="column"] .stButton>button:hover{background:#22D3EE22!important;border-color:#22D3EE!important;color:#22D3EE!important;transform:none!important;box-shadow:none!important}
[data-testid="stDownloadButton"] button{background:#1C2333!important;color:#F1F5F9!important;border:1px solid #334155!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;padding:8px 16px!important}
[data-testid="stDownloadButton"] button p,[data-testid="stDownloadButton"] button span,[data-testid="stDownloadButton"] button div{color:#F1F5F9!important}
[data-testid="stDownloadButton"] button:hover{background:#22D3EE22!important;border-color:#22D3EE!important;color:#22D3EE!important;transform:none!important;box-shadow:none!important}
[data-baseweb="select"]>div{background:var(--bg-el)!important;border-color:var(--border)!important}
[data-baseweb="select"] [data-testid="stMarkdownContainer"],[class*="singleValue"],[class*="placeholder"]{color:var(--text)!important}
[data-baseweb="popover"] li{background:var(--bg-el)!important;color:var(--text)!important}
[data-baseweb="popover"] li:hover{background:#22D3EE22!important}
textarea,input{background:var(--bg-el)!important;color:var(--text)!important;border-color:var(--border)!important}
[data-baseweb="tab-list"]{background:var(--bg-card)!important;border-radius:10px;padding:4px;gap:2px}
[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.78rem!important}
[aria-selected="true"][data-baseweb="tab"]{background:var(--accent)!important;color:#0A0E1A!important}
[data-testid="stDataFrame"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px}
[data-testid="stExpander"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg-base)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.eco-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:20px 24px;margin:8px 0}
.pred-card{border-radius:14px;padding:28px;text-align:center;margin:12px 0}
.pred-label{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;opacity:.7;margin-bottom:8px}
.pred-value{font-family:'Space Mono',monospace;font-size:2.2rem;font-weight:700;margin:4px 0}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-family:'Space Mono',monospace;font-size:.65rem;font-weight:700}
.section-header{font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border)}
@keyframes border-pulse{0%,100%{border-color:var(--border)}50%{border-color:var(--accent)}}
.live-card{animation:border-pulse 2s ease infinite}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    try:
        return pd.read_csv("data/raw/dataset_ProjetML_2026.csv")
    except Exception:
        return None


@st.cache_data
def load_pca():
    try:
        return pd.read_csv(f"{PROCESSED_DIR}/pca_clusters.csv")
    except Exception:
        return None


@st.cache_data
def load_metrics():
    m = {}
    for n in ["classification", "regression", "nlp", "multimodal", "clustering", "preprocessing"]:
        try:
            with open(f"{PROCESSED_DIR}/{n}_metrics.json") as f:
                m[n] = json.load(f)
        except Exception:
            pass
    return m


@st.cache_data
def load_results():
    r = {}
    for n in ["classification_results", "regression_results", "multimodal_results"]:
        try:
            r[n] = pd.read_csv(f"{PROCESSED_DIR}/{n}.csv")
        except Exception:
            pass
    return r


@st.cache_resource
def load_models():
    m = {}
    for name, path in [
        ("classifier",          f"{PROCESSED_DIR}/best_classifier.pkl"),
        ("label_encoder",       f"{PROCESSED_DIR}/label_encoder.pkl"),
        ("scaler",              f"{PROCESSED_DIR}/scaler.pkl"),
        ("source_encoder",      f"{PROCESSED_DIR}/source_encoder.pkl"),
        ("tfidf",               f"{PROCESSED_DIR}/tfidf_vectorizer.pkl"),
        ("nlp_classifier",      f"{PROCESSED_DIR}/nlp_best_classifier.pkl"),
        ("pipeline_multimodal", f"{PROCESSED_DIR}/pipeline_multimodal.pkl"),
        ("regressor",           f"{PROCESSED_DIR}/best_regressor.pkl"),
    ]:
        try:
            m[name] = joblib.load(path)
        except Exception:
            pass
    return m


# ── Helpers ───────────────────────────────────────────────────────────────────
def cat_badge(cat):
    p = PALETTE.get(cat, PALETTE["Non labellisé"])
    return (
        f"<span class='badge' style='background:{p['bg']};"
        f"color:{p['color']};border:1px solid {p['color']}44'>"
        f"{p['icon']} {cat}</span>"
    )


def confidence_bar(val, color):
    pct = int(val * 100)
    return (
        f"<div style='margin:4px 0'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
        f"<span style='font-family:Space Mono,monospace;font-size:.7rem;color:#94A3B8'>Confiance</span>"
        f"<span style='font-family:Space Mono,monospace;font-size:.7rem;color:{color}'>{pct}%</span></div>"
        f"<div style='background:#1F2937;border-radius:4px;height:6px;overflow:hidden'>"
        f"<div style='width:{pct}%;height:100%;background:linear-gradient(90deg,{color}88,{color});border-radius:4px'>"
        f"</div></div></div>"
    )


def add_history(mode, inputs, cat, conf=None, prix=None):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": mode,
        "inputs": inputs,
        "categorie": cat,
        "confiance": conf,
        "prix": prix,
    })
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]


def predict_numeric(models, poids, volume, conductivite, opacite, rigidite, src_enc, densite):
    """
    Prédiction numérique avec scaler correct.
    Scaler : [Poids, Volume, Conductivite, Opacite, Rigidite, Densite_estimee]  (6 features)
    Classifier : [Poids_s, Volume_s, Conductivite_s, Opacite_s, Rigidite_s, Source_enc, Densite_s]  (7 features)
    """
    feats_num = np.array([[poids, volume, conductivite, opacite, rigidite, densite]])

    if "scaler" in models:
        scaled = models["scaler"].transform(feats_num)[0]
        p_s, v_s, c_s, o_s, r_s, d_s = scaled
    else:
        p_s, v_s, c_s, o_s, r_s, d_s = poids, volume, conductivite, opacite, rigidite, densite

    # Source_enc inséré à la position 5 (index 5), Densite en dernier
    feats = np.array([[p_s, v_s, c_s, o_s, r_s, src_enc, d_s]])
    return feats


# ── Sidebar ───────────────────────────────────────────────────────────────────
models  = load_models()
metrics = load_metrics()

with st.sidebar:
    st.markdown(
        "<div style='padding:20px 0 12px'>"
        "<div style='font-family:Space Mono,monospace;color:#22D3EE;font-size:1.4rem;font-weight:700'>♻ ECO-SMART</div>"
        "<div style='color:#64748B;font-size:.75rem;margin-top:-4px'>Waste Classification System v2.1</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    nav = st.radio("", [
        "📊  Dashboard",
        "🎯  Prédiction",
        "🤖  Assistant NLP",
        "📈  Comparaison Modèles",
        "🕐  Historique",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("<div class='section-header'>Pipeline Status</div>", unsafe_allow_html=True)
    for label, key in [
        ("Preprocessing",  "preprocessing"),
        ("Classification", "classification"),
        ("Régression",     "regression"),
        ("NLP",            "nlp"),
        ("Multimodal",     "multimodal"),
        ("Clustering",     "clustering"),
    ]:
        done  = key in metrics
        color = "#34D399" if done else "#4B5563"
        icon  = "✅" if done else "⏳"
        st.markdown(
            f"<div style='font-family:Space Mono,monospace;font-size:.7rem;color:{color};padding:3px 0'>"
            f"{icon} {label}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    n = len(st.session_state.history)
    st.markdown(
        f"<div style='font-family:Space Mono,monospace;font-size:.7rem;color:#64748B'>"
        f"🕐 {n} prédiction{'s' if n != 1 else ''} en session</div>",
        unsafe_allow_html=True,
    )
    if n and st.button("🗑️ Vider historique", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if nav == "📊  Dashboard":
    st.markdown("# 📊 Dashboard")
    st.markdown(
        "<div style='color:#64748B;margin-bottom:24px'>Vue d'ensemble du dataset et performances du pipeline.</div>",
        unsafe_allow_html=True,
    )

    df = load_raw()

    if df is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Échantillons",  f"{len(df):,}")
        c2.metric("Features num.", "6")
        c3.metric("Classes",       "4")
        c4.metric("NaN Catégorie", df["Categorie"].isna().sum())
        c5.metric("Sources",       df["Source"].nunique())

    st.markdown("<div class='section-header' style='margin-top:24px'>Performance des modèles</div>", unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)

    def perf_card(col, label, val, color="#22D3EE"):
        with col:
            v = f"{val:.3f}" if isinstance(val, float) else str(val)
            st.markdown(
                f"<div class='eco-card'>"
                f"<div style='font-family:Space Mono,monospace;font-size:.6rem;letter-spacing:.15em;color:#64748B;text-transform:uppercase'>{label}</div>"
                f"<div style='font-family:Space Mono,monospace;font-size:2rem;color:{color};margin:8px 0'>{v}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    perf_card(pc1, "Classification Acc",  metrics.get("classification", {}).get("accuracy_test", "—"), "#22D3EE")
    perf_card(pc2, "NLP F1-Score",        metrics.get("nlp", {}).get("f1_test", "—"),                  "#6EE7B7")
    perf_card(pc3, "Multimodal Accuracy", metrics.get("multimodal", {}).get("accuracy_test", "—"),     "#A78BFA")
    perf_card(pc4, "Régression R²",       metrics.get("regression", {}).get("r2", "—"),                "#F97316")

    if df is not None:
        try:
            import plotly.graph_objects as go

            cfg = {"displayModeBar": False}
            bg  = "rgba(0,0,0,0)"

            ca, cb = st.columns(2)
            with ca:
                st.markdown("<div class='section-header'>Distribution des catégories</div>", unsafe_allow_html=True)
                vc = df["Categorie"].value_counts().reset_index()
                vc.columns = ["cat", "n"]
                fig = go.Figure(go.Bar(
                    x=vc["cat"], y=vc["n"],
                    marker_color=[PALETTE.get(c, PALETTE["Non labellisé"])["color"] for c in vc["cat"]],
                    marker_line_width=0,
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                    font=dict(family="Space Mono", color="#94A3B8"),
                    margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=False,
                    yaxis=dict(gridcolor="#1F2D45"), xaxis=dict(gridcolor=bg),
                )
                st.plotly_chart(fig, use_container_width=True, config=cfg)

            with cb:
                st.markdown("<div class='section-header'>Valeurs manquantes (%)</div>", unsafe_allow_html=True)
                nan_pct = (df.isnull().sum() / len(df) * 100).round(1)
                nan_df  = nan_pct[nan_pct > 0].sort_values(ascending=True)
                fig2 = go.Figure(go.Bar(
                    x=nan_df.values, y=nan_df.index, orientation="h",
                    marker=dict(
                        color=nan_df.values,
                        colorscale=[[0, "#1E3A5F"], [1, "#60A5FA"]],
                        line_width=0,
                    ),
                    text=[f"{v:.1f}%" for v in nan_df.values],
                    textposition="inside",
                    textfont=dict(family="Space Mono", size=10, color="#F1F5F9"),
                ))
                fig2.update_layout(
                    template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                    font=dict(family="Space Mono", color="#94A3B8"),
                    margin=dict(t=10, b=10, l=10, r=10), height=280,
                    xaxis=dict(gridcolor="#1F2D45"), yaxis=dict(gridcolor=bg),
                )
                st.plotly_chart(fig2, use_container_width=True, config=cfg)

            st.markdown("<div class='section-header'>Corrélations features numériques</div>", unsafe_allow_html=True)
            ncols = [c for c in ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"] if c in df.columns]
            corr  = df[ncols].corr()
            fig3  = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="Blues", zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
                textfont=dict(family="Space Mono", size=9),
            ))
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                font=dict(family="Space Mono", color="#94A3B8"),
                margin=dict(t=10, b=10, l=10, r=10), height=320,
            )
            st.plotly_chart(fig3, use_container_width=True, config=cfg)

            df_pca = load_pca()
            if df_pca is not None:
                st.markdown("<div class='section-header'>Clusters K-Means — PCA 2D</div>", unsafe_allow_html=True)
                colors_pca = [
                    PALETTE.get(c, PALETTE["Non labellisé"])["color"]
                    for c in df_pca.get("Categorie", pd.Series(["Non labellisé"] * len(df_pca)))
                ]
                fig4 = go.Figure(go.Scatter(
                    x=df_pca["PC1"], y=df_pca["PC2"], mode="markers",
                    marker=dict(color=colors_pca, size=5, opacity=0.6, line_width=0),
                    text=df_pca.get("Categorie", "") + " · Cluster " + df_pca["Cluster"].astype(str),
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig4.update_layout(
                    template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                    font=dict(family="Space Mono", color="#94A3B8"),
                    margin=dict(t=10, b=10, l=10, r=10), height=380,
                    xaxis=dict(gridcolor="#1F2D45", title="PC1"),
                    yaxis=dict(gridcolor="#1F2D45", title="PC2"),
                )
                st.plotly_chart(fig4, use_container_width=True, config=cfg)

        except ImportError:
            st.info("Installe plotly : `pip install plotly`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRÉDICTION MANUELLE
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🎯  Prédiction":
    st.markdown("# 🎯 Prédiction Manuelle")
    st.markdown(
        "<div style='color:#64748B;margin-bottom:24px'>Ajustez les curseurs — la prédiction se met à jour en temps réel.</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    # ── Inputs (colonne gauche) ───────────────────────────────────────────────
    with col_l:
        st.markdown("<div class='section-header'>Paramètres du déchet</div>", unsafe_allow_html=True)
        poids        = st.slider("⚖️  Poids (kg)",     0.0, 200.0, 50.0,  0.5)
        volume       = st.slider("📦  Volume (litres)", 0.0, 500.0, 100.0, 1.0)
        conductivite = st.slider("⚡  Conductivité",    0.0, 1.0,   0.5,   0.01)
        opacite      = st.slider("🌫️  Opacité",        0.0, 1.0,   0.5,   0.01)
        rigidite     = st.slider("🔧  Rigidité",       1.0, 10.0,  5.0,   0.1)
        source       = st.selectbox("🏭  Source", ["Usine_A", "Usine_B", "Centre_Tri", "Unknown"])
        densite      = poids / (volume + 1e-6)

        st.markdown(
            f"<div class='eco-card' style='margin-top:12px'>"
            f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#64748B;letter-spacing:.1em'>DENSITÉ ESTIMÉE</div>"
            f"<div style='font-family:Space Mono,monospace;font-size:1.4rem;color:#22D3EE;margin-top:4px'>"
            f"{densite:.3f} <span style='font-size:.8rem;color:#64748B'>kg/L</span></div></div>",
            unsafe_allow_html=True,
        )

    # ── Prédiction (colonne droite) ───────────────────────────────────────────
    with col_r:
        st.markdown("<div class='section-header'>Résultat en direct</div>", unsafe_allow_html=True)

        pred_cat   = None
        confiance  = None
        prix       = None
        proba_dict = None

        if "classifier" in models and "label_encoder" in models:
            le  = models["label_encoder"]
            src = models.get("source_encoder")

            # Encodage source
            try:
                src_enc = int(src.transform([source])[0]) if src else 0
            except ValueError:
                src_enc = 0

            # Prédiction avec scaler correct
            feats = predict_numeric(models, poids, volume, conductivite, opacite, rigidite, src_enc, densite)

            clf     = models["classifier"]
            pred_id = int(clf.predict(feats)[0])
            pred_cat = le.inverse_transform([pred_id])[0]

            if hasattr(clf, "predict_proba"):
                proba      = clf.predict_proba(feats)[0]
                confiance  = float(np.max(proba))
                proba_dict = dict(zip(le.classes_, proba))

            if "regressor" in models:
                try:
                    prix = float(
                        models["regressor"].predict(
                            np.append(feats[0], pred_id).reshape(1, -1)
                        )[0]
                    )
                except Exception:
                    prix = None
        else:
            pred_cat = "Non disponible"

        # ── Carte résultat ────────────────────────────────────────────────────
        p      = PALETTE.get(pred_cat, PALETTE["Non labellisé"])
        notice = "Modèles non chargés — lancez dvc repro" if pred_cat == "Non disponible" else "Pipeline numérique"

        st.markdown(
            f"<div class='pred-card live-card' style='background:{p['bg']};border:1px solid {p['color']}44'>"
            f"<div class='pred-label' style='color:{p['color']}'>Catégorie prédite</div>"
            f"<div class='pred-value' style='color:{p['color']}'>{p['icon']} {pred_cat}</div>"
            f"<div style='font-family:Space Mono,monospace;font-size:.7rem;color:#94A3B8;margin-top:8px'>{notice}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if confiance is not None:
            st.markdown(confidence_bar(confiance, p["color"]), unsafe_allow_html=True)

        if prix is not None:
            st.markdown(
                f"<div class='eco-card'>"
                f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#64748B;letter-spacing:.1em'>PRIX DE REVENTE ESTIMÉ</div>"
                f"<div style='font-family:Space Mono,monospace;font-size:1.8rem;color:#F97316;margin-top:6px'>"
                f"{prix:.2f} <span style='font-size:.9rem;color:#64748B'>DT/kg</span></div></div>",
                unsafe_allow_html=True,
            )

        # ── Graphe probabilités ───────────────────────────────────────────────
        if proba_dict:
            try:
                import plotly.graph_objects as go

                le_obj = models["label_encoder"]

                # Construire proba_named selon le type des clés
                proba_named = {}
                for k, v in proba_dict.items():
                    if isinstance(k, str):
                        # Clé déjà un nom de classe
                        proba_named[k] = v
                    else:
                        # Clé entière → inverse_transform
                        proba_named[le_obj.inverse_transform([int(k)])[0]] = v

                sp = sorted(proba_named.items(), key=lambda x: x[1])
                fig_p = go.Figure(go.Bar(
                    x=[v for _, v in sp],
                    y=[k for k, _ in sp],
                    orientation="h",
                    marker_color=[PALETTE.get(k, PALETTE["Non labellisé"])["color"] for k, _ in sp],
                    marker_line_width=0,
                    text=[f"{v * 100:.1f}%" for _, v in sp],
                    textposition="inside",
                    textfont=dict(family="Space Mono", size=9, color="#F1F5F9"),
                ))
                fig_p.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Space Mono", color="#94A3B8"),
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=180,
                    showlegend=False,
                    xaxis=dict(range=[0, 1], gridcolor="#1F2D45"),
                )
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass

        # ── Bouton historique ─────────────────────────────────────────────────
        if pred_cat and pred_cat != "Non disponible":
            if st.button("💾 Sauvegarder dans l'historique", use_container_width=True):
                add_history(
                    "Numérique",
                    {"poids": poids, "volume": volume, "conductivite": conductivite,
                     "opacite": opacite, "rigidite": rigidite, "source": source},
                    pred_cat, confiance, prix,
                )
                st.success("✅ Ajouté à l'historique !")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASSISTANT NLP
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🤖  Assistant NLP":
    st.markdown("# 🤖 Assistant NLP")
    st.markdown(
        "<div style='color:#64748B;margin-bottom:24px'>Décrivez le déchet en français. Le pipeline analyse et classe votre texte.</div>",
        unsafe_allow_html=True,
    )

    EXEMPLES = {
        "🔩 Métal":     "Lot de ferraille collecté à l'Usine B. Matériau lourd et conducteur, aspect métallique brillant. Légère oxydation en surface détectée.",
        "📄 Papier":    "Cartons et feuilles collectés au Centre de Tri. Matériau souple et léger, non conducteur. Quelques traces d'humidité mais état correct.",
        "♻️ Plastique": "Déchets plastiques variés de l'Usine A. Emballages légers, souples, non conducteurs. Mélange de PET et HDPE, bonne condition générale.",
        "🫙 Verre":     "Bris de verre et contenants provenant de la déchetterie. Matériau dense et transparent, rigide. Quelques fragments brisés, légère contamination.",
    }

    MOTS_CLES = {
        "Métal":     ["métal", "métallique", "ferraille", "conducteur", "acier",
                      "aluminium", "oxydation", "rouille", "fer", "cuivre", "zinc", "lourd"],
        "Verre":     ["verre", "transparent", "bris", "fragile", "vitre", "bouteille", "cristal"],
        "Plastique": ["plastique", "pet", "hdpe", "emballage", "souple", "polymère", "pvc"],
        "Papier":    ["papier", "carton", "feuille", "journal", "cellulose", "imprimé", "cartonné"],
    }

    def regle_lexicale(texte):
        scores = {cat: 0 for cat in MOTS_CLES}
        mots   = texte.lower().split()
        for cat, kw_list in MOTS_CLES.items():
            for mot in mots:
                for kw in kw_list:
                    if kw in mot:
                        scores[cat] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='section-header'>Description textuelle</div>", unsafe_allow_html=True)
        ex_cols = st.columns(4)
        for i, (label, texte) in enumerate(EXEMPLES.items()):
            if ex_cols[i].button(label, use_container_width=True, key=f"ex_{i}"):
                st.session_state["nlp_input"] = texte

        rapport = st.text_area(
            "",
            value=st.session_state.get("nlp_input", ""),
            height=160,
            placeholder="Décrivez le type, l'état, la provenance du déchet...",
            label_visibility="collapsed",
        )

        c1b, c2b = st.columns(2)
        run_nlp   = c1b.button("🔍 Analyser", use_container_width=True)
        run_clear = c2b.button("🗑️ Effacer",  use_container_width=True)

        if run_clear:
            st.session_state["nlp_input"] = ""
            st.rerun()

    with col2:
        st.markdown("<div class='section-header'>Analyse</div>", unsafe_allow_html=True)

        if rapport and run_nlp:
            try:
                from src.nlp.text_preprocessing import (
                    extraire_contamination,
                    extraire_etat,
                    preprocess_text,
                )

                with st.spinner("Traitement NLP..."):
                    time.sleep(0.3)
                    texte_clean   = preprocess_text(rapport)
                    contamination = extraire_contamination(rapport)
                    etat          = extraire_etat(rapport)

                nb  = len(rapport.split())
                na  = len(texte_clean.split())
                red = int((1 - na / max(nb, 1)) * 100)

                st.markdown(
                    f"<div class='eco-card'>"
                    f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#64748B;letter-spacing:.1em;margin-bottom:10px'>STATS PRÉTRAITEMENT</div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px'>"
                    f"<div style='text-align:center'><div style='font-family:Space Mono,monospace;font-size:1.2rem;color:#22D3EE'>{nb}</div><div style='font-size:.65rem;color:#64748B'>mots bruts</div></div>"
                    f"<div style='text-align:center'><div style='font-family:Space Mono,monospace;font-size:1.2rem;color:#6EE7B7'>{na}</div><div style='font-size:.65rem;color:#64748B'>après NLP</div></div>"
                    f"<div style='text-align:center'><div style='font-family:Space Mono,monospace;font-size:1.2rem;color:#F97316'>-{red}%</div><div style='font-size:.65rem;color:#64748B'>réduction</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                cc = "#F97316" if contamination else "#34D399"
                st.markdown(
                    f"<div class='eco-card' style='margin-top:8px'>"
                    f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#64748B;letter-spacing:.1em;margin-bottom:8px'>EXTRACTION REGEX</div>"
                    f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
                    f"<span class='badge' style='background:{'#3D1F0D' if contamination else '#0D3D2E'};color:{cc};border:1px solid {cc}44'>{'⚠️ Contaminé' if contamination else '✅ Propre'}</span>"
                    f"<span class='badge' style='background:#1E3A5F;color:#60A5FA;border:1px solid #60A5FA44'>🔍 {etat}</span>"
                    f"</div>"
                    f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:.7rem;color:#475569;word-break:break-word'>{texte_clean or '— vide —'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Prédiction NLP ────────────────────────────────────────────
                pred_cat    = None
                source_pred = ""

                if "tfidf" in models and "nlp_classifier" in models and "label_encoder" in models:
                    X           = models["tfidf"].transform([texte_clean])
                    pred_id     = int(models["nlp_classifier"].predict(X)[0])
                    pred_modele = models["label_encoder"].inverse_transform([pred_id])[0]
                else:
                    pred_modele = None

                pred_lexical = regle_lexicale(texte_clean)

                if pred_lexical and pred_modele and pred_lexical != pred_modele:
                    pred_cat    = pred_lexical
                    source_pred = "Règle lexicale"
                elif pred_modele:
                    pred_cat    = pred_modele
                    source_pred = "Modèle NLP"
                elif pred_lexical:
                    pred_cat    = pred_lexical
                    source_pred = "Règle lexicale"

                if pred_cat is not None:
                    p2 = PALETTE.get(pred_cat, PALETTE["Non labellisé"])
                    st.markdown(
                        f"<div class='pred-card' style='background:{p2['bg']};border:1px solid {p2['color']}44;margin-top:12px'>"
                        f"<div class='pred-label' style='color:{p2['color']}'>Classification NLP</div>"
                        f"<div class='pred-value' style='color:{p2['color']}'>{p2['icon']} {pred_cat}</div>"
                        f"<div style='font-family:Space Mono,monospace;font-size:.65rem;color:#64748B;margin-top:6px'>{source_pred}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("💾 Sauvegarder", use_container_width=True, key="save_nlp"):
                        add_history("NLP", {"rapport": rapport[:80] + "..."}, pred_cat)
                        st.success("✅ Ajouté !")
                else:
                    st.info("Description insuffisante pour classifier.")

            except ImportError as e:
                st.error(f"Erreur import : {e}")

        elif not rapport:
            st.markdown(
                "<div style='text-align:center;padding:40px 20px;color:#334155'>"
                "<div style='font-size:2.5rem;margin-bottom:12px;opacity:.3'>✍️</div>"
                "<div style='font-family:Space Mono,monospace;font-size:.75rem'>Tapez une description<br>ou choisissez un exemple</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARAISON MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📈  Comparaison Modèles":
    st.markdown("# 📈 Comparaison des Modèles")
    st.markdown(
        "<div style='color:#64748B;margin-bottom:24px'>Performance comparative de tous les modèles entraînés.</div>",
        unsafe_allow_html=True,
    )

    results = load_results()
    subtab  = st.radio("", ["Classification", "NLP", "Multimodal", "Régression"],
                       horizontal=True, label_visibility="collapsed")

    bg  = "rgba(0,0,0,0)"
    cfg = {"displayModeBar": False}

    try:
        import plotly.graph_objects as go

        def bar_multi(df, x_col, y_cols, colors, height=350):
            fig = go.Figure()
            for yc, col in zip(y_cols, colors):
                if yc in df.columns:
                    fig.add_trace(go.Bar(
                        name=yc, x=df[x_col], y=df[yc],
                        marker_color=col, marker_line_width=0,
                        text=df[yc].round(3), textposition="outside",
                        textfont=dict(family="Space Mono", size=9),
                    ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                font=dict(family="Space Mono", color="#94A3B8"),
                margin=dict(t=10, b=10, l=10, r=10), height=height,
                barmode="group",
                yaxis=dict(gridcolor="#1F2D45", range=[0, 1.1]),
                xaxis=dict(gridcolor=bg),
                legend=dict(bgcolor=bg, font=dict(size=9)),
            )
            return fig

        if subtab == "Classification":
            df_clf = results.get("classification_results")
            if df_clf is not None:
                st.markdown("<div class='section-header'>Tous les modèles comparés</div>", unsafe_allow_html=True)
                fig = bar_multi(df_clf, "model",
                                ["acc_val", "acc_test", "f1_val", "f1_test"],
                                ["#22D3EE", "#60A5FA", "#6EE7B7", "#34D399"])
                st.plotly_chart(fig, use_container_width=True, config=cfg)
                best = df_clf.sort_values("f1_test", ascending=False).iloc[0]
                st.markdown(
                    f"<div class='eco-card' style='border-color:#22D3EE44'>"
                    f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#64748B;letter-spacing:.1em'>🏆 MEILLEUR MODÈLE</div>"
                    f"<div style='font-family:Space Mono,monospace;font-size:1.3rem;color:#22D3EE;margin:6px 0'>{best['model']}</div>"
                    f"<div style='display:flex;gap:20px;margin-top:8px'>"
                    f"<div><div style='font-size:.65rem;color:#64748B'>Accuracy Test</div><div style='font-family:Space Mono,monospace;color:#6EE7B7'>{best['acc_test']:.4f}</div></div>"
                    f"<div><div style='font-size:.65rem;color:#64748B'>F1 Test</div><div style='font-family:Space Mono,monospace;color:#6EE7B7'>{best['f1_test']:.4f}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(df_clf, use_container_width=True)
            else:
                st.info("Lancez `python src/models/train_classification.py`")

        elif subtab == "NLP":
            nlp_m = metrics.get("nlp", {})
            if nlp_m:
                c1, c2, c3 = st.columns(3)
                c1.metric("Meilleure Vectorisation", nlp_m.get("best_vectorisation", "—"))
                c2.metric("Meilleur Classifieur",    nlp_m.get("best_classifieur", "—"))
                c3.metric("F1 Test",                 f"{nlp_m.get('f1_test', 0):.4f}")
                img = f"{PROCESSED_DIR}/nlp_heatmap.png"
                if os.path.exists(img):
                    st.markdown("<div class='section-header' style='margin-top:20px'>Heatmap vectorisation × classifieur</div>", unsafe_allow_html=True)
                    st.image(img, use_column_width=True)
                else:
                    st.info("Heatmap disponible après `python src/nlp/train_nlp.py`")
            else:
                st.info("Lancez `python src/nlp/train_nlp.py`")

        elif subtab == "Multimodal":
            df_mm = results.get("multimodal_results")
            if df_mm is not None:
                fig_mm = bar_multi(df_mm, "Approche", ["Accuracy", "F1"], ["#22D3EE", "#6EE7B7"])
                st.plotly_chart(fig_mm, use_container_width=True, config=cfg)
                gain = metrics.get("multimodal", {}).get("gain_vs_numerique")
                if gain is not None:
                    col = "#34D399" if gain > 0 else "#F87171"
                    st.markdown(
                        f"<div class='eco-card' style='border-color:{col}44'>"
                        f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#64748B;letter-spacing:.1em'>GAIN VS NUMÉRIQUE SEUL</div>"
                        f"<div style='font-family:Space Mono,monospace;font-size:1.5rem;color:{col};margin-top:6px'>"
                        f"{'+' if gain > 0 else ''}{gain:.4f} F1</div></div>",
                        unsafe_allow_html=True,
                    )
                st.dataframe(df_mm, use_container_width=True)
            else:
                st.info("Lancez `python src/models/train_multimodal.py`")

        elif subtab == "Régression":
            df_reg = results.get("regression_results")
            if df_reg is not None:
                fig_r = go.Figure()
                for mc, col in [("r2", "#22D3EE"), ("mae", "#F97316"), ("rmse", "#A78BFA")]:
                    if mc in df_reg.columns:
                        fig_r.add_trace(go.Bar(
                            name=mc.upper(), x=df_reg["model"], y=df_reg[mc],
                            marker_color=col, marker_line_width=0,
                        ))
                fig_r.update_layout(
                    template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                    font=dict(family="Space Mono", color="#94A3B8"),
                    margin=dict(t=10, b=10, l=10, r=10), height=320,
                    barmode="group",
                    yaxis=dict(gridcolor="#1F2D45"),
                    xaxis=dict(gridcolor=bg),
                )
                st.plotly_chart(fig_r, use_container_width=True, config=cfg)
                st.dataframe(df_reg, use_container_width=True)
            else:
                st.info("Lancez `python src/models/train_regression.py`")

    except ImportError:
        st.warning("Installe plotly : `pip install plotly`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🕐  Historique":
    st.markdown("# 🕐 Historique des Prédictions")

    hist = st.session_state.history
    st.markdown(
        f"<div style='color:#64748B;margin-bottom:24px'>{len(hist)} prédiction(s) enregistrée(s) en session.</div>",
        unsafe_allow_html=True,
    )

    if not hist:
        st.markdown(
            "<div style='text-align:center;padding:80px 20px'>"
            "<div style='font-size:3rem;margin-bottom:16px;opacity:.2'>🕐</div>"
            "<div style='font-family:Space Mono,monospace;font-size:.85rem;color:#334155'>Aucune prédiction enregistrée.</div>"
            "<div style='font-size:.75rem;color:#1E293B;margin-top:8px'>Faites une prédiction dans les onglets Prédiction ou Assistant NLP.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        hist_df = pd.DataFrame(hist)
        cats    = hist_df["categorie"].value_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total",         len(hist_df))
        c2.metric("Catégorie top", cats.index[0] if len(cats) else "—")
        c3.metric("Modes",         hist_df["mode"].nunique())

        try:
            import plotly.graph_objects as go

            fig_pie = go.Figure(go.Pie(
                labels=cats.index,
                values=cats.values,
                hole=0.5,
                marker_colors=[PALETTE.get(c, PALETTE["Non labellisé"])["color"] for c in cats.index],
                textinfo="label+percent",
                textfont=dict(family="Space Mono", size=10),
            ))
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        except ImportError:
            pass

        st.markdown("<div class='section-header'>Détail</div>", unsafe_allow_html=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter CSV", csv, "historique_predictions.csv", "text/csv")

        for i, entry in enumerate(hist):
            p        = PALETTE.get(entry["categorie"], PALETTE["Non labellisé"])
            conf_str = f"· {entry['confiance'] * 100:.0f}% conf." if entry.get("confiance") else ""
            prix_str = f"· {entry['prix']:.2f} DT/kg"             if entry.get("prix")      else ""

            col_card, col_del = st.columns([10, 1])
            with col_card:
                st.markdown(
                    f"<div class='eco-card' style='border-left:3px solid {p['color']}'>"
                    f"<span style='font-family:Space Mono,monospace;font-size:.65rem;color:#64748B'>"
                    f"#{len(hist) - i:03d} · {entry['time']} · {entry['mode']}</span>"
                    f"<div style='margin-top:4px'>{cat_badge(entry['categorie'])} "
                    f"<span style='font-family:Space Mono,monospace;font-size:.7rem;color:#475569'>{conf_str} {prix_str}</span></div>"
                    f"<div style='margin-top:8px;font-size:.72rem;color:#334155;font-family:Space Mono,monospace'>"
                    f"{str(entry['inputs'])[:130]}</div></div>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{i}", help="Supprimer", type="primary"):
                    st.session_state.history.pop(i)
                    st.rerun()
