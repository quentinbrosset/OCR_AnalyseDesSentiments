#!/bin/bash
echo "Installation des dépendances..."
pip install -r requirements_dev.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab

echo "Démarrage des applications..."
cd Application
uvicorn api:app --host 0.0.0.0 --port $PORT &
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
