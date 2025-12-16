# OCR_AnalyseDesSentiments

Ce projet vise à mettre en œuvre une chaîne de traitement complète pour l'analyse de sentiments sur des tweets. Il intègre un modèle de Deep Learning pour prédire si un tweet est positif ou négatif, exposé via une API et consommable via une interface utilisateur.

## Objectif du Projet

L'objectif est de fournir un outil capable d'analyser le sentiment de textes courts (tweets) en utilisant des techniques de Traitement Automatique du Langage Naturel (NLP). Le projet comprend l'entraînement du modèle, son déploiement via une API REST, et une interface de démonstration.

## Structure du Projet (Découpage des dossiers)

Le projet est organisé selon la structure suivante :

*   **Application/** : Contient le code source de l'application déployée.
    *   `api.py` : L'API FastAPI qui expose le modèle de prédiction.
    *   `app.py` : L'interface utilisateur Streamlit pour interagir avec l'API.
    *   `best_model.joblib` : Le modèle de classification entraîné.
    *   `requirements.txt` : Liste des dépendances spécifiques à l'application.
*   **Notebooks/** : Contient les notebooks Jupyter utilisés pour l'analyse exploratoire, le pré-traitement des données et l'entraînement des modèles.
*   **TestsUnitaires/** : Contient les tests unitaires pour valider la robustesse du code (notamment de l'API).
*   **requirements.txt** (Racine) : Fichier listant l'ensemble des packages Python utilisés dans le projet pour faciliter l'installation de l'environnement complet.

## Installation et Dépendances

Les packages nécessaires au fonctionnement du projet sont listés dans les fichiers `requirements.txt`.
Pour installer l'environnement complet, vous pouvez utiliser la commande :

```bash
pip install -r requirements.txt
```

## Utilisation

Le projet s'articule autour de deux composants principaux situés dans le dossier `Application/` :

1.  **API (api.py)** : Service web gérant les prédictions.
2.  **Interface (app.py)** : Dashboard interactif pour tester le modèle.