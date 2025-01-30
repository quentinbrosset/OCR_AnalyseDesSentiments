import sys
from pathlib import Path

# Dossier parent du projet
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from Application.api import app  # Importer l'application FastAPI

# Initialisation du test
client = TestClient(app)

def test_predict_positive_sentiment():
    # Exemple d'un tweet positif
    tweet = {"tweet": "I love this product!"}
    response = client.post("/predict/", json=tweet)
    
    # Vérification du code de statut HTTP
    assert response.status_code == 200
    
    # Vérification du format de la réponse
    assert "sentiment" in response.json()
    assert "confiance" in response.json()
    
    # Vérification du sentiment prédit
    assert response.json()["sentiment"] == "Positif"

def test_predict_negative_sentiment():
    # Exemple d'un tweet négatif
    tweet = {"tweet": "I hate this product!"}
    response = client.post("/predict/", json=tweet)
    
    # Vérification du code de statut HTTP
    assert response.status_code == 200
    
    # Vérification du format de la réponse
    assert "sentiment" in response.json()
    assert "confiance" in response.json()
    
    # Vérification du sentiment prédit
    assert response.json()["sentiment"] == "Négatif"

def test_invalid_input():
    # Exemple d'une requête avec un champ manquant
    tweet = {}  # Champ tweet manquant
    response = client.post("/predict/", json=tweet)
    
    # Vérification du code de statut HTTP
    assert response.status_code == 422  # Erreur de validation

