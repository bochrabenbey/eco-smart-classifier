# PROMPTS.md — Journal des Interactions IA
## Projet : Eco-Smart Classifier | Charte IA respectée

---

## 🔴 ZONES INTERDITES (IA non utilisée)

### Tests unitaires
Tous les fichiers `tests/test_*.py` ont été écrits manuellement sans aide IA.
- `test_preprocessing.py` : logique métier EDA/imputation rédigée à la main
- `test_nlp.py` : cas bords regex, tokenisation, vectorisation — 100% manuel
- `test_models.py` : seuil performance, schéma prédictions — manuel

### Fonctions EDA de base
`analyze_missing()`, `treat_outliers()` — premières versions sans IA.

### Prétraitement NLP
`src/nlp/text_preprocessing.py` — tokenisation, regex extraction, stopwords domaine : rédigé sans IA.

---

## 🟠 ZONES STRUCTURATION SEULEMENT

### Configuration DVC (dvc.yaml)
**Prompt utilisé** :
> "Génère la structure d'un dvc.yaml avec 6 étapes pour un pipeline ML : preprocess → train_classification → train_regression → clustering → train_nlp → train_multimodal. Chaque étape a ses deps, params et outs."

**Critique** : La structure proposée était correcte mais manquait le champ `metrics` avec `cache: false`. Correction manuelle ajoutée.

### Configuration MLflow
**Prompt utilisé** :
> "Comment enregistrer un modèle au MLflow Model Registry en Python avec mlflow.sklearn.log_model et registered_model_name ?"

**Critique** : La réponse était générique. Adaptation au contexte multi-expériences du projet.

### Débogage GridSearchCV + LabelEncoder
**Prompt utilisé** :
> "J'ai une erreur ValueError: y contains previously unseen labels lors du transform. Comment gérer les labels inconnus avec sklearn LabelEncoder ?"

**Critique** : Solution proposée (try/except) adoptée, meilleure que le fallback à 0 suggéré initialement.

---

## 🟢 ZONES LIBRES

### Dockerfile
**Prompt** :
> "Crée un Dockerfile optimisé pour une API FastAPI Python 3.11 avec scikit-learn, nltk et uvicorn. Multi-stage non requis. Ajoute un HEALTHCHECK."

**Résultat** : Adopté tel quel, minor edit pour ajouter le téléchargement NLTK.

### docker-compose.yml
**Prompt** :
> "docker-compose.yml avec 3 services : FastAPI (port 8000), MLflow UI (port 5000), Streamlit (port 8501). Réseau commun."

**Résultat** : Adopté, ajout du volume mlruns partagé.

### CI/CD GitHub Actions
**Prompt** :
> "GitHub Actions workflow : lint (black, flake8, isort), pytest avec coverage ≥ 70%, build Docker. Déclenché sur push main et PR."

**Résultat** : Adopté avec ajout du cache pip.

### API FastAPI — Endpoint /predict_multi
**Prompt** :
> "Écris un endpoint FastAPI POST /predict_multi qui accepte à la fois des features numériques et un texte, et retourne la prédiction d'un pipeline sklearn ColumnTransformer."

**Critique** : Gestion d'erreur 503 ajoutée manuellement. Pydantic v2 incompatibilité corrigée.

### Dashboard Streamlit — Dark Theme CSS
**Prompt** :
> "CSS Streamlit dark theme professionnel avec Space Mono pour les titres, DM Sans pour le body. Variables CSS custom pour couleurs. Cards avec borders subtiles."

**Critique** : Les `!important` excessifs ont été réduits. Compatibilité Streamlit 1.33 vérifiée.

### Monitoring Evidently
**Prompt** :
> "Comment utiliser evidently.Report avec DataDriftPreset pour comparer deux DataFrames pandas et exporter en HTML ?"

**Résultat** : Adopté + ajout fallback sans Evidently (Jensen-Shannon manuel).

### Optimisation Optuna (alternative GridSearchCV)
**Prompt** :
> "Implémente une fonction Optuna pour optimiser les hyperparamètres d'un RandomForestClassifier avec 50 trials, objectif f1_weighted."

**Critique** : Non intégré au pipeline principal (risque de reproductibilité DVC). Conservé comme référence.

---

## 📊 Bilan

| Zone | Fichiers concernés | IA utilisée |
|------|-------------------|-------------|
| 🔴 Rouge | tests/, text_preprocessing.py | ❌ Non |
| 🟠 Orange | dvc.yaml, mlflow config | ✅ Structure seule |
| 🟢 Vert | Dockerfile, CI/CD, API, Streamlit CSS | ✅ Libre |

**Justification des choix face à l'IA** : Chaque suggestion IA a été évaluée et souvent modifiée. Les cas critiques (tests, NLP de base) sont restés 100% manuels conformément à la charte. L'IA a accéléré la génération de code boilerplate (Dockerfile, YAML) sans remplacer le raisonnement métier.
