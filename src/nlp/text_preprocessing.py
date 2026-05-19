"""
Prétraitement textuel NLP — Module 4
======================================
Tokenisation, suppression stopwords français + domaine, lemmatisation.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Téléchargement silencieux des ressources NLTK
for resource in ["punkt", "stopwords", "wordnet", "punkt_tab", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ── Stopwords ──────────────────────────────────────────────────────────────
STOPWORDS_FR = set(stopwords.words("french"))

STOPWORDS_DOMAINE = {
    "collecte", "rapport", "materiau", "matériau",
    "dechet", "déchet", "lot", "site", "usine",
    "provenance", "source", "type", "objet",
    "collecté", "collectés", "collectée",
    "issu", "issus", "présente", "présent",
    "environ", "estimation", "estimé", "estimée",
    "poids", "volume", "masse", "quantite", "quantité",
    "kg", "litre", "cm", "mm", "non", "renseigné",
}

STOPWORDS_TOUS = STOPWORDS_FR | STOPWORDS_DOMAINE

lemmatizer = WordNetLemmatizer()


def preprocess_text(texte: str) -> str:
    """
    Nettoie un rapport de collecte pour le NLP.

    Pipeline :
    1. Minuscules
    2. Normalisation des unités (45.8 kg → 45.8kg)
    3. Suppression ponctuation
    4. Tokenisation (NLTK french)
    5. Suppression stopwords (français + domaine)
    6. Filtrage mots courts (< 3 chars, sauf chiffres)
    7. Lemmatisation (WordNet)

    Args:
        texte: texte brut

    Returns:
        texte nettoyé (chaîne)

    Example:
        >>> preprocess_text("Le matériau collecté est un métal conducteur.")
        'métal conducteur'
    """
    if not isinstance(texte, str) or texte.strip() == "":
        return ""

    texte = texte.lower()

    # Normaliser les unités de mesure collées aux chiffres
    texte = re.sub(r"(\d+[\.,]?\d*)\s*(kg|g|cm|mm|m²|l|ml|%)", r"\1\2", texte)

    # Supprimer la ponctuation
    texte = re.sub(r"[^\w\s]", " ", texte)

    # Tokeniser
    try:
        tokens = word_tokenize(texte, language="french")
    except LookupError:
        tokens = texte.split()

    # Filtrer stopwords + mots trop courts
    tokens = [
        t for t in tokens
        if t not in STOPWORDS_TOUS
        and (len(t) > 2 or t.isdigit())
    ]

    # Lemmatiser
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def extraire_contamination(texte: str) -> int:
    """Retourne 1 si une contamination est mentionnée, 0 sinon."""
    if not isinstance(texte, str):
        return 0
    mots = r"contaminat|humidit|traces?|souillé|pollué|rouille|corrodé|oxydé"
    return 1 if re.search(mots, texte, re.IGNORECASE) else 0


def extraire_etat(texte: str) -> str:
    """Retourne l'état du matériau : Neuf / Moyen / Brisé / Inconnu."""
    if not isinstance(texte, str):
        return "Inconnu"
    t = texte.lower()
    if re.search(r"bris[ée]|cassé|fracturé|endommagé|déchiré|fissur", t):
        return "Brisé"
    elif re.search(r"neuf|nouveau|intact|parfait.état|excellent", t):
        return "Neuf"
    elif re.search(r"usé|moyen|acceptable|correct|légèrement|partielle", t):
        return "Moyen"
    return "Inconnu"


def extraire_source_texte(texte: str) -> str:
    """Extrait le nom de l'usine mentionné dans le texte."""
    if not isinstance(texte, str):
        return ""
    patterns = [
        r"usine\s+([A-Z][a-zA-Z\-]+)",
        r"site\s+([A-Z][a-zA-Z\-]+)",
        r"provenance\s*[:]\s*([A-Z][a-zA-Z\-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, texte)
        if m:
            return m.group(1)
    return ""
