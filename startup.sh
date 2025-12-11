#!/bin/bash
echo "Installation des dépendances..."
cd /home/site/wwwroot

# Créer et activer un environnement virtuel si nécessaire
if [ ! -d "env" ]; then
    python -m venv env
fi
source env/bin/activate

# Installer les dépendances de PROD uniquement
# Le flag --no-cache-dir peut aider à éviter les timeouts si l'espace disque est juste
pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires
python -m nltk.downloader punkt stopwords wordnet punkt_tab

echo "Démarrage de l'API..."
cd Application

# Démarrer l'API comme processus PRINCIPAL (pas de & à la fin)
# Azure attend que le container réponde sur le port 8000
python -m uvicorn api:app --host 0.0.0.0 --port 8000