# 🌿 Eco-Smart Classifier

Workflow complet de Machine Learning pour la classification de déchets et l'estimation de leur valeur de revente.

## ⚡ Lancer le pipeline en 3 commandes

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Reproduire tout le pipeline DVC
dvc repro

# 3. Lancer l'API FastAPI
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🏗️ Structure du projet

```
eco_smart_classifier/
├── data/
│   ├── raw/                    # Dataset original
│   └── processed/              # Données nettoyées, splits, modèles sauvegardés
├── notebooks/
│   └── ecoSmart.ipynb          # Notebook principal (exploration + modules 1–5)
├── src/
│   ├── preprocessing/          # Module 1 : EDA, nettoyage, feature engineering
│   ├── models/                 # Module 2 : Classification + Régression
│   ├── nlp/                    # Module 4 : Pipeline NLP
│   ├── api/                    # Module 6 : API FastAPI
│   └── monitoring/             # Module 6 : Monitoring Evidently
├── tests/                      # Tests pytest
├── .github/workflows/          # CI/CD GitHub Actions
├── dvc.yaml                    # Pipeline DVC (DAG)
├── params.yaml                 # Paramètres du pipeline
├── Dockerfile                  # Image Docker
├── docker-compose.yml          # Composition services
├── requirements.txt
├── PROMPTS.md                  # Journal des interactions IA
└── README.md
```

## 📦 Modules

| Module | Description |
|--------|-------------|
| 1 | EDA, nettoyage, imputation (Médiane / KNN / MICE), outliers, feature engineering |
| 2 | Classification (Catégorie) + Régression (Prix_Revente), SHAP, GridSearchCV |
| 3 | Clustering K-Means, méthode du coude, PCA 2D |
| 4 | NLP : BoW / TF-IDF / Word2Vec / FastText × 4 classifieurs |
| 5 | Pipeline multimodal : fusion numérique + NLP (hstack + ColumnTransformer + Stacking) |
| 6 | MLOps : DVC, MLflow (≥5 expériences), CI/CD, Docker, FastAPI, Evidently |

## 🔬 MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Ouvrir http://localhost:5000 pour visualiser les expériences.

## 🐳 Docker

```bash
docker build -t eco-smart-classifier .
docker run -p 8000:8000 eco-smart-classifier
```

Ou avec docker-compose :

```bash
docker-compose up --build
```

## ✅ Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Couverture minimale requise : **70%**.

## 📊 Application Web

L'application Streamlit est accessible après déploiement :

- **Dashboard Data** : Visualisation du dataset et des clusters PCA
- **Prédiction Manuelle** : Curseurs interactifs pour prédire en temps réel
- **Assistant NLP** : Saisie de description textuelle → prédiction de catégorie

```bash
streamlit run src/app/streamlit_app.py
```

## 📝 Charte IA

Voir [PROMPTS.md](PROMPTS.md) pour le journal complet des interactions IA et la justification des choix.
