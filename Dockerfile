FROM python:3.11-slim

LABEL maintainer="Eco-Smart Classifier"
LABEL description="API de classification de déchets"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROCESSED_DIR=/app/data/processed \
    PORT=8000

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    nltk.download('punkt_tab', quiet=True)"

# Copier le code source
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY params.yaml .

# Créer les répertoires nécessaires
RUN mkdir -p data/reports logs

# Exposer le port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Commande de démarrage
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
