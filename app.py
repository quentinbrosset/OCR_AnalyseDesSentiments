import streamlit as st
import httpx
import json

# Endpoint de l'API FastAPI
API_ENDPOINT = "http://localhost:8000/predict/"

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
        return None

def main():
    st.title("Prédiction du Sentiment d'un Tweet")

    # Demander à l'utilisateur de rentrer un tweet
    tweet = st.text_area("Entrez votre tweet ici :")

    if st.button("Prévoir le sentiment"):
        if tweet:
            sentiment, confiance = get_sentiment(tweet)
            if sentiment:
                st.write(f"Le sentiment prédictif est : **{sentiment}**")
                st.write(f"L'indice de confiance est de : {confiance}")
        else:
            st.warning("Veuillez entrer un tweet pour prédire son sentiment.")

if __name__ == "__main__":
    main()
