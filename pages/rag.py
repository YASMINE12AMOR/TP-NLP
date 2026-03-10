import os
from pathlib import Path
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

# Configuration de la page
st.set_page_config(
    page_title="🔍 RAG - Recherche Augmentée par Génération",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Données d'exemple améliorées ---
documents = [
    {
        "title": "Chatbot Éducatif avec RASA",
        "description": "Un chatbot basé sur RASA pour répondre aux questions des étudiants en master sur leurs cours. Idéal pour les projets NLP. Utilise des intentions et entités pour comprendre les questions des étudiants.",
        "tags": ["NLP", "Éducation", "Python", "Chatbot"],
        "lien": "https://rasa.com/docs/rasa/"
    },
    {
        "title": "Détection de Fraudes avec PyTorch",
        "description": "Modèle de détection de fraudes bancaires utilisant PyTorch et des techniques de Data Science. Utilise des réseaux de neurones profonds pour analyser les transactions suspectes.",
        "tags": ["Data Science", "Sécurité", "PyTorch", "Fraude"],
        "lien": "https://pytorch.org/tutorials/"
    },
    {
        "title": "Système de Recommandation pour les MOOCs",
        "description": "Un système de recommandation de cours en ligne basé sur les préférences des étudiants. Utilise des algorithmes de filtrage collaboratif et de contenu pour personnaliser les recommandations.",
        "tags": ["Data Science", "Recommandation", "Scikit-learn", "Éducation"],
        "lien": "https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble"
    },
    {
        "title": "Monitoring de la Qualité de l'Air avec Raspberry Pi",
        "description": "Projet IoT pour surveiller la qualité de l'air dans une salle de classe. Utilise des capteurs de CO2, de particules fines et de température, avec transmission des données via MQTT et stockage dans une base de données InfluxDB.",
        "tags": ["IoT", "Raspberry Pi", "MQTT", "Environnement"],
        "lien": "https://projects.raspberrypi.org/en/projects/air-quality"
    },
    {
        "title": "Certification Décentralisée avec Blockchain",
        "description": "Plateforme de certification des diplômes utilisant la blockchain Ethereum et des smart contracts en Solidity. Permet de vérifier l'authenticité des diplômes de manière sécurisée et transparente.",
        "tags": ["Blockchain", "Solidity", "Ethereum", "Éducation"],
        "lien": "https://docs.soliditylang.org/"
    },
    {
        "title": "Analyse des Sentiments sur les Réseaux Sociaux",
        "description": "Projet d'analyse des sentiments sur Twitter ou Facebook utilisant des modèles de NLP comme BERT ou RoBERTa. Permet de comprendre les opinions des utilisateurs sur un sujet donné.",
        "tags": ["NLP", "Réseaux Sociaux", "BERT", "Analyse de Sentiments"],
        "lien": "https://huggingface.co/transformers/"
    },
    {
        "title": "Reconnaissance d'Images Médicales avec TensorFlow",
        "description": "Modèle de reconnaissance d'images médicales utilisant TensorFlow et des réseaux de neurones convolutifs (CNN). Idéal pour détecter des anomalies dans des radiographies ou des IRM.",
        "tags": ["Deep Learning", "TensorFlow", "Santé", "CNN"],
        "lien": "https://www.tensorflow.org/tutorials"
    },
    {
        "title": "Automatisation Domotique avec Home Assistant",
        "description": "Projet de domotique utilisant Home Assistant pour automatiser les tâches ménagères. Intègre des capteurs IoT, des actionneurs et des scénarios personnalisés.",
        "tags": ["IoT", "Domotique", "Home Assistant", "Automatisation"],
        "lien": "https://www.home-assistant.io/"
    },
    {
        "title": "Traduction Automatique avec Transformers",
        "description": "Système de traduction automatique utilisant des modèles de transformers comme MarianMT ou T5. Permet de traduire des textes entre plusieurs langues avec une haute précision.",
        "tags": ["NLP", "Traduction", "Transformers", "MarianMT"],
        "lien": "https://huggingface.co/transformers/model_doc/marian.html"
    },
    {
        "title": "Prédiction des Ventes avec Prophet",
        "description": "Utilisation de Facebook Prophet pour prédire les ventes futures d'une entreprise. Idéal pour les projets de business intelligence et d'analyse prédictive.",
        "tags": ["Data Science", "Prédiction", "Prophet", "Business Intelligence"],
        "lien": "https://facebook.github.io/prophet/"
    }
]

# Charger les variables d'environnement
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "mistral-small-latest"

@st.cache_resource
def get_client():
    if not api_key:
        return None
    return Mistral(api_key=api_key)

# --- Initialisation du modèle d'embeddings ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Indexation des documents avec FAISS ---
@st.cache_resource
def build_faiss_index():
    model = load_embedding_model()
    descriptions = [doc["description"] for doc in documents]
    embeddings = model.encode(descriptions, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

# --- Fonction de recherche ---
def retrieve_docs(query, index, model, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# --- Génération de la réponse ---
def generate_response(query, retrieved_docs):
    context = "\n\n".join(
        [
            f"Titre: {doc['title']}\nDescription: {doc['description']}\nTags: {', '.join(doc['tags'])}\nLien: {doc['lien']}"
            for doc in retrieved_docs
        ]
    )
    prompt = f"""
Tu es un assistant qui recommande des idées de projets. Base-toi uniquement sur les documents ci-dessous. Si l'information n'apparaît pas dans le contexte, dis-le clairement.
Question utilisateur: {query}
Contexte: {context}
Réponds en français avec:
1) Recommandation principale
2) Pourquoi ce choix
3) 2 alternatives
4) Liens utiles
"""
    client = get_client()
    if client is None:
        return (
            "MISTRAL_API_KEY absente. Je ne peux pas interroger le LLM, "
            "mais voici les documents les plus proches de ta requête."
        )
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# --- Interface Streamlit ---
def main():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
            padding: 10px;
        }
        .stTextArea>div>div>textarea {
            border-radius: 12px;
            padding: 10px;
        }
        .stExpander>div>div>div {
            border-radius: 12px;
            border: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🔍 RAG - Recherche Augmentée par Génération")
    st.markdown("""
    Bienvenue dans l'outil de recherche augmentée par génération. Posez une question ou décrivez un projet,
    et je vous fournirai des recommandations basées sur une base de données de projets variés.
    """)

    index, model = build_faiss_index()

    query = st.text_input(
        "Que recherchez-vous ?",
        value="Ex: Je veux un projet NLP pour analyser les sentiments sur les réseaux sociaux",
        placeholder="Ex: projet IoT pour surveiller la qualité de l'air"
    )

    if st.button("Rechercher"):
        if query:
            with st.spinner("Recherche en cours..."):
                retrieved_docs = retrieve_docs(query, index, model)
                response = generate_response(query, retrieved_docs)

            st.markdown("### 🎯 Réponse")
            st.markdown(response)

            st.write("---")
            st.markdown("### 📚 Documents Utilisés")
            for doc in retrieved_docs:
                with st.expander(f"📄 {doc['title']}"):
                    st.markdown(
                        f"""
                        **Description**: {doc['description']}
                        **Tags**: {', '.join(doc['tags'])}
                        [Lien]({doc['lien']})
                        """
                    )
        else:
            st.warning("Veuillez entrer une requête avant de lancer la recherche.")

    st.write("---")
    st.markdown("### 📂 Tous les Projets Disponibles")
    for doc in documents:
        with st.expander(f"📄 {doc['title']}"):
            st.markdown(
                f"""
                **Description**: {doc['description']}
                **Tags**: {', '.join(doc['tags'])}
                [Lien]({doc['lien']})
                """
            )

if __name__ == "__main__":
    main()
