import streamlit as st
import httpx
import json
import os

# Endpoint de l'API FastAPI
# Si une variable d'environnement API_URL est définie, l'utiliser, sinon utiliser localhost
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
API_ENDPOINT = f"{API_URL}/predict/"


def get_sentiment(tweet):
    # Préparer les données pour l'API
    data = {"tweet": tweet}
    try:
        # Faire une requête POST à l'API pour obtenir le sentiment
        response = httpx.post(API_ENDPOINT, json=data)
        response.raise_for_status()
        result = response.json()
        return result['sentiment'], result["confiance"]
    except httpx.RequestError as e:
        st.error(f"Une erreur s'est produite : {e}")
        return None, None

def main():
    st.title("Prédiction du Sentiment d'un Tweet")
    
    # Afficher l'URL de l'API en mode debug (vous pourrez retirer ceci plus tard)
    st.sidebar.write(f"API endpoint: {API_ENDPOINT}")
    
    # Demander à l'utilisateur de rentrer un tweet
    tweet = st.text_area("Entrez votre tweet ici :")
    
    if st.button("Prévoir le sentiment"):
        # Supprimer les espaces en début et en fin de chaîne
        cleaned_tweet = tweet.strip()
        # Vérifier que le tweet contient au moins deux caractères non-espaces
        if len(cleaned_tweet.replace(" ", "")) < 2:
            st.warning("Veuillez entrer un tweet valide.")
        else:
            with st.spinner("Analyse en cours..."):
                sentiment, confiance = get_sentiment(cleaned_tweet)
                if sentiment:
                    st.write(f"Le sentiment prédictif est : **{sentiment}**")
                    st.write(f"L'indice de confiance est de : {confiance}")

if __name__ == "__main__":
    main()
