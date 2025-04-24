#!/bin/bash
echo "Installation des dépendances..."
cd /home/site/wwwroot

# Créer et activer un environnement virtuel si nécessaire
if [ ! -d "env" ]; then
    python -m venv env
fi
source env/bin/activate

# Installer les dépendances
pip install -r requirements_dev.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab

echo "Démarrage des applications..."
cd Application
# Démarrer l'API en arrière-plan
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &
# Démarrer Streamlit comme processus principal
python -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0