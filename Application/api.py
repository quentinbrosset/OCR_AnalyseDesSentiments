# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:43:33 2024

@author: qbrosset
"""
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from pathlib import Path
import joblib
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import os

# Configuration du logger pour Application Insights
logger = logging.getLogger(__name__)
# On ne met le handler que si la clé est présente (pour éviter erreurs en local sans clé)
if "APPLICATIONINSIGHTS_CONNECTION_STRING" in os.environ:
    logger.addHandler(AzureLogHandler(connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]))
else:
    # Fallback pour voir les logs en local
    handler = logging.StreamHandler()
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)

# Charger le modèle sauvegardé
model_path = Path(__file__).resolve().parent / 'best_model.joblib'
model = joblib.load(model_path)

# Initialiser l'application FastAPI
app = FastAPI()

# Définir la structure des données d'entrée
class TweetInput(BaseModel):
    tweet: str  # Le texte du tweet à analyser

class Feedback(BaseModel):
    tweet: str
    prediction: str
    commentaire: str = None


""" Preprocessing du tweet """

# Liste des émoticônes et sigles à remplacer
emoticons = {
    ":)": "smile", ":-)": "smile", ":D": "laugh", ":-D": "laugh", ";D": "wink_smile", ";)": "wink", ":P": "playful", ";P": "playful_wink",
    ":-P": "playful", "XD": "laugh_hard", "xD": "laugh_hard", "=)": "happy_face", ":]": "happy_face", ":-]": "happy_face", ":3": "cute_smile", 
    "<3": "heart", "❤️": "heart", "😊": "blush", "☺️": "smile_blush", "😄": "smile_big", "😁": "grin", "😆": "laugh_out_loud", "😂": "tears_of_joy",
    ":(": "sad", ":-(": "sad", ":'(": "crying", ":'-(": "crying", ":/": "disappointed", ":-/": "disappointed", ":|": "neutral", ":-|": "neutral",
    ">:(": "angry", ">:-(": "angry", "D:": "shocked", "DX": "distressed", "D8": "distressed", "D;": "distressed", "D=": "horrified", ">:O": "surprised",
    ">:0": "surprised", "😔": "pensive", "😢": "crying", "😭": "sob", "😡": "angry", "😠": "annoyed", "😞": "disappointed", "😟": "worried",
    "😒": "unamused", ":|": "neutral", ":-|": "neutral", ":o": "surprised", ":-o": "surprised", ":O": "surprised_big", ":-O": "surprised_big",
    "o.O": "confused", "O.o": "confused", "-_-": "unimpressed", "-.-": "bored", ">_>": "skeptical", "<_<": "skeptical", "😐": "neutral_face",
    "😑": "expressionless", "🤔": "thinking", "😶": "speechless", "🙄": "eyeroll", ":*": "kiss", ":-*": "kiss", ";*": "wink_kiss",
    "😋": "savoring_food", "😜": "playful_face", "😝": "playful_tongue", "🤪": "zany_face", "😎": "cool", "😇": "innocent", "🥰": "affection",
    "🤗": "hug", "😏": "smirk", "🙃": "upside_down_face", "😴": "sleepy", "😌": "relieved", "🤤": "drooling"
}

# Liste des stopwords en anglais
stopW = nltk.corpus.stopwords.words("english")
ponctuation = set(string.punctuation)
mots_nuages = ["going", "got", "go", "twitter", "lol", "quot", "amp", "can't", "today", "gonna", "ca", "n't", "gon", "na", "think", "s"]
stopW.extend(ponctuation)
stopW.extend(mots_nuages)

# Fonction pour retirer les stopwords
def sup_stopwords(word_list, stopwords):
    return [word for word in word_list if word not in stopwords]

# Fonction pour vérifier si une chaîne est un nombre
def is_number(s):
    try:
        float(s)  # Si la conversion en float fonctionne, c'est un nombre
        return True
    except ValueError:
        return False

# Fonction pour retirer les nombres de listes
def sup_nombres(word_list):
    return [word for word in word_list if not (isinstance(word, (int, float)) or is_number(word))]

# Chargement du modèle USE
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet, emoticons_dict):
    # Remplacement des émoticônes par leur signification
    for emoticon, meaning in emoticons_dict.items():
        tweet = re.sub(re.escape(emoticon), meaning, tweet)
    
    # Normalisation
    tweet = tweet.lower()
    
    # Suppression des urls et mentions
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@\s?\w+", "", tweet)
    
    # Tokenisation
    tokens = nltk.word_tokenize(tweet)
    
    # Suppression des stopwords
    tokens = sup_stopwords(tokens, stopW)
    tokens = sup_nombres(tokens)
    
    # Lemmatisation
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def feature_USE_fct(sentences, b_size):
    batch_size = b_size
    features_list = []

    for step in range(0, len(sentences), batch_size):
        batch_sentences = sentences[step:step + batch_size]
        if not batch_sentences:
            break
        # Génération des features pour le batch
        feats = embed(batch_sentences)  # Utilisation de USE sur le batch
        features_list.append(feats)

    # Concaténation finale de toutes les features
    features = np.vstack(features_list) if features_list else np.array([])

    return features

# Paramètres USE
batch_size = 10

@app.post("/predict/")
def predict_sentiment(data: TweetInput):
    # Prétraitement du tweet (adapté selon le modèle utilisé)
    processed_tweet = preprocess_tweet(data.tweet, emoticons)
    
    # Extraction des features avec USE
    features = feature_USE_fct(processed_tweet, batch_size)
    
    # Si aucune feature n'est générée, lever une exception
    if features.size == 0:
        raise ValueError("No features generated from the input.")
    
    # Faire une prédiction avec le modèle
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    # Déterminer le sentiment en fonction de la prédiction
    sentiment = "Positif" if prediction[0] == 1 else "Négatif"
    
    # Probabilité associée à la classe prédite
    confiance = round(prediction_proba[0, prediction[0]] * 100, 2)
    confiance = confiance.astype(str)
    
    # Retourner le sentiment
    return {"tweet": data.tweet, "sentiment": sentiment, "confiance": f"{confiance}%"}
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:43:33 2024

@author: qbrosset
"""
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from pathlib import Path
import joblib

# Charger le modèle sauvegardé
model_path = Path(__file__).resolve().parent / 'best_model.joblib'
model = joblib.load(model_path)

# Initialiser l'application FastAPI
app = FastAPI()

# Définir la structure des données d'entrée
class TweetInput(BaseModel):
    tweet: str  # Le texte du tweet à analyser

""" Preprocessing du tweet """

# Liste des émoticônes et sigles à remplacer
emoticons = {
    ":)": "smile", ":-)": "smile", ":D": "laugh", ":-D": "laugh", ";D": "wink_smile", ";)": "wink", ":P": "playful", ";P": "playful_wink",
    ":-P": "playful", "XD": "laugh_hard", "xD": "laugh_hard", "=)": "happy_face", ":]": "happy_face", ":-]": "happy_face", ":3": "cute_smile", 
    "<3": "heart", "❤️": "heart", "😊": "blush", "☺️": "smile_blush", "😄": "smile_big", "😁": "grin", "😆": "laugh_out_loud", "😂": "tears_of_joy",
    ":(": "sad", ":-(": "sad", ":'(": "crying", ":'-(": "crying", ":/": "disappointed", ":-/": "disappointed", ":|": "neutral", ":-|": "neutral",
    ">:(": "angry", ">:-(": "angry", "D:": "shocked", "DX": "distressed", "D8": "distressed", "D;": "distressed", "D=": "horrified", ">:O": "surprised",
    ">:0": "surprised", "😔": "pensive", "😢": "crying", "😭": "sob", "😡": "angry", "😠": "annoyed", "😞": "disappointed", "😟": "worried",
    "😒": "unamused", ":|": "neutral", ":-|": "neutral", ":o": "surprised", ":-o": "surprised", ":O": "surprised_big", ":-O": "surprised_big",
    "o.O": "confused", "O.o": "confused", "-_-": "unimpressed", "-.-": "bored", ">_>": "skeptical", "<_<": "skeptical", "😐": "neutral_face",
    "😑": "expressionless", "🤔": "thinking", "😶": "speechless", "🙄": "eyeroll", ":*": "kiss", ":-*": "kiss", ";*": "wink_kiss",
    "😋": "savoring_food", "😜": "playful_face", "😝": "playful_tongue", "🤪": "zany_face", "😎": "cool", "😇": "innocent", "🥰": "affection",
    "🤗": "hug", "😏": "smirk", "🙃": "upside_down_face", "😴": "sleepy", "😌": "relieved", "🤤": "drooling"
}

# Liste des stopwords en anglais
stopW = nltk.corpus.stopwords.words("english")
ponctuation = set(string.punctuation)
mots_nuages = ["going", "got", "go", "twitter", "lol", "quot", "amp", "can't", "today", "gonna", "ca", "n't", "gon", "na", "think", "s"]
stopW.extend(ponctuation)
stopW.extend(mots_nuages)

# Fonction pour retirer les stopwords
def sup_stopwords(word_list, stopwords):
    return [word for word in word_list if word not in stopwords]

# Fonction pour vérifier si une chaîne est un nombre
def is_number(s):
    try:
        float(s)  # Si la conversion en float fonctionne, c'est un nombre
        return True
    except ValueError:
        return False

# Fonction pour retirer les nombres de listes
def sup_nombres(word_list):
    return [word for word in word_list if not (isinstance(word, (int, float)) or is_number(word))]

# Chargement du modèle USE
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet, emoticons_dict):
    # Remplacement des émoticônes par leur signification
    for emoticon, meaning in emoticons_dict.items():
        tweet = re.sub(re.escape(emoticon), meaning, tweet)
    
    # Normalisation
    tweet = tweet.lower()
    
    # Suppression des urls et mentions
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@\s?\w+", "", tweet)
    
    # Tokenisation
    tokens = nltk.word_tokenize(tweet)
    
    # Suppression des stopwords
    tokens = sup_stopwords(tokens, stopW)
    tokens = sup_nombres(tokens)
    
    # Lemmatisation
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def feature_USE_fct(sentences, b_size):
    batch_size = b_size
    features_list = []

    for step in range(0, len(sentences), batch_size):
        batch_sentences = sentences[step:step + batch_size]
        if not batch_sentences:
            break
        # Génération des features pour le batch
        feats = embed(batch_sentences)  # Utilisation de USE sur le batch
        features_list.append(feats)

    # Concaténation finale de toutes les features
    features = np.vstack(features_list) if features_list else np.array([])

    return features

# Paramètres USE
batch_size = 10

@app.post("/predict/")
def predict_sentiment(data: TweetInput):
    # Prétraitement du tweet (adapté selon le modèle utilisé)
    processed_tweet = preprocess_tweet(data.tweet, emoticons)
    
    # Extraction des features avec USE
    features = feature_USE_fct(processed_tweet, batch_size)
    
    # Si aucune feature n'est générée, lever une exception
    if features.size == 0:
        raise ValueError("No features generated from the input.")
    
    # Faire une prédiction avec le modèle
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    # Déterminer le sentiment en fonction de la prédiction
    sentiment = "Positif" if prediction[0] == 1 else "Négatif"
    
    # Probabilité associée à la classe prédite
    confiance = round(prediction_proba[0, prediction[0]] * 100, 2)
    confiance = confiance.astype(str)
    
    # Retourner le sentiment
    return {"tweet": data.tweet, "sentiment": sentiment, "confiance": f"{confiance}%"}

@app.post("/feedback/")
def log_feedback(feedback: Feedback):
    # Log spécifique pour App Insights
    properties = {
        "custom_dimensions": {
            "tweet": feedback.tweet,
            "prediction": feedback.prediction,
            "type": "IncorrectPrediction" 
        }
    }
    logger.warning("Feedback Utilisateur : Prédiction incorrecte signalée", extra=properties)
    return {"status": "logged"}
