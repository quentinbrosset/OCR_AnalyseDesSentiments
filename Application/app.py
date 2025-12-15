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
        response = httpx.post(API_ENDPOINT, json=data, timeout=120.0)
        response.raise_for_status()
        result = response.json()
        return result['sentiment'], result["confiance"]
    except Exception as e:
        st.error(f"Erreur lors de la requête vers l'API ({API_ENDPOINT}) : {e}")
        return None

def send_feedback(tweet, prediction):
    feedback_endpoint = API_ENDPOINT.replace("/predict/", "/feedback/")
    data = {
        "tweet": tweet,
        "prediction": prediction,
        "commentaire": "Signalé par utilisateur Streamlit"
    }
    try:
        response = httpx.post(feedback_endpoint, json=data, timeout=5.0)
        response.raise_for_status() # Lève une erreur si 404, 500, etc.
        return True
    except Exception as e:
        st.error(f"Impossible d'envoyer le feedback : {e}")
        return False

def main():
    st.title("Prédiction du Sentiment d'un Tweet")

    # Initialisation du session_state
    if "result" not in st.session_state:
        st.session_state.result = None
    if "tweet_analyzed" not in st.session_state:
        st.session_state.tweet_analyzed = ""

    # Demander à l'utilisateur de rentrer un tweet
    tweet = st.text_area("Entrez votre tweet ici :")

    if st.button("Prévoir le sentiment"):
        cleaned_tweet = tweet.strip()
        if len(cleaned_tweet.replace(" ", "")) < 2:
            st.warning("Veuillez entrer un tweet valide.")
        else:
            result = get_sentiment(cleaned_tweet)
            if result:
                # Stocker le résultat et le tweet dans la session
                st.session_state.result = result
                st.session_state.tweet_analyzed = cleaned_tweet

    # Affichage du résultat s'il existe en session
    if st.session_state.result:
        sentiment, confiance = st.session_state.result
        st.write(f"Le sentiment prédictif est : **{sentiment}**")
        st.write(f"L'indice de confiance est de : {confiance}")
        
        # Zone de Feedback
        st.markdown("---")
        st.write("Le résultat vous semble incorrect ?")
        if st.button("👎 Signaler une erreur"):
            if send_feedback(st.session_state.tweet_analyzed, sentiment):
                st.success("Merci ! L'erreur a été signalée à l'équipe technique.")
            else:
                st.error("Erreur lors de l'envoi du signalement.")

if __name__ == "__main__":
    main()
