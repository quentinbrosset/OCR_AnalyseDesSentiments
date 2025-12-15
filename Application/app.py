import streamlit as st
import httpx
import json

# Endpoint de l'API FastAPI
if "API_ENDPOINT" in st.secrets:
    API_ENDPOINT = st.secrets["API_ENDPOINT"]
else:
    API_ENDPOINT = "http://127.0.0.1:8000/predict/"

def get_sentiment(tweet):
    # Préparer les données pour l'API
    data = {"tweet": tweet}
    try:
        # Faire une requête POST à l'API pour obtenir le sentiment
        response = httpx.post(API_ENDPOINT, json=data, timeout=60.0)
        response.raise_for_status()
        result = response.json()
        return result['sentiment'], result["confiance"]
    except Exception as e:
        st.error(f"Erreur lors de la requête vers l'API ({API_ENDPOINT}) : {e}")
        return None

def main():
    st.title("Prédiction du Sentiment d'un Tweet")

    # Demander à l'utilisateur de rentrer un tweet
    tweet = st.text_area("Entrez votre tweet ici :")

    if st.button("Prévoir le sentiment"):
        # Supprimer les espaces en début et en fin de chaîne
        cleaned_tweet = tweet.strip()
        # Vérifier que le tweet contient au moins deux caractères non-espaces
        if len(cleaned_tweet.replace(" ", "")) < 2:
            st.warning("Veuillez entrer un tweet valide.")
        else:
            result = get_sentiment(cleaned_tweet)
            if result:
                sentiment, confiance = result
                st.write(f"Le sentiment prédictif est : **{sentiment}**")
                st.write(f"L'indice de confiance est de : {confiance}")

if __name__ == "__main__":
    main()
