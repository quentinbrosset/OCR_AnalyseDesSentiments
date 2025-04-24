#!/bin/bash
echo "Installation des dépendances..."
export PATH="$HOME/python/bin:$PATH"

# Utiliser le Python d'Azure App Service
/home/site/wwwroot/env/bin/pip install -r requirements_dev.txt
/home/site/wwwroot/env/bin/python -m nltk.downloader punkt stopwords wordnet punkt_tab

echo "Démarrage des applications..."
cd Application
/home/site/wwwroot/env/bin/uvicorn api:app --host 0.0.0.0 --port $PORT &
/home/site/wwwroot/env/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0