import streamlit as st
from transformers import CamembertTokenizer, CamembertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Charger le modèle et le tokeniseur Camembert
@st.cache_resource
def load_model():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    return tokenizer, model

tokenizer, model = load_model()

# Fonction pour obtenir l'embedding d'un mot
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Fonction pour calculer la similarité entre deux mots
def calculate_similarity(word1, word2):
    embedding1 = get_embedding(word1)
    embedding2 = get_embedding(word2)
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Interface Streamlit
st.title("🎲 Cemantik - Mode 2 Joueurs")
st.markdown("""
**Règles :**
- **Joueur 1** : Entre un mot cible (il sera masqué après validation).
- **Joueur 2** : Propose des mots pour deviner le mot cible.
- Le Joueur 2 a **5 tentatives** pour proposer des mots.
""")

# Initialisation des variables de session
if "mot_cible" not in st.session_state:
    st.session_state.mot_cible = None
if "tentatives_restantes" not in st.session_state:
    st.session_state.tentatives_restantes = 5
if "historique" not in st.session_state:
    st.session_state.historique = []
if "jeu_termine" not in st.session_state:
    st.session_state.jeu_termine = False
if "dernier_mot_propose" not in st.session_state:
    st.session_state.dernier_mot_propose = ""

# Étape 1 : Joueur 1 entre le mot cible
if not st.session_state.mot_cible and not st.session_state.jeu_termine:
    mot_cible = st.text_input("**Joueur 1** : Entrez le mot cible (puis validez)", key="input_mot_cible", type="password")
    if st.button("Valider le mot cible") and mot_cible:
        st.session_state.mot_cible = mot_cible.lower()
        st.session_state.tentatives_restantes = 5
        st.session_state.historique = []
        st.session_state.jeu_termine = False
        st.session_state.dernier_mot_propose = ""
        st.success("Mot cible enregistré et masqué. Au tour du Joueur 2 !")

# Étape 2 : Joueur 2 propose des mots
if st.session_state.mot_cible and not st.session_state.jeu_termine:
    st.subheader(f"Tentatives restantes : {st.session_state.tentatives_restantes}")

    # Utiliser une clé fixe pour le champ de saisie du mot proposé
    st.session_state.dernier_mot_propose = st.text_input(
        "**Joueur 2** : Proposez un mot",
        value=st.session_state.dernier_mot_propose,
        key="input_proposition"
    )

    if st.button("Proposer ce mot") and st.session_state.dernier_mot_propose:
        mot_propose = st.session_state.dernier_mot_propose.lower()
        similarity = calculate_similarity(st.session_state.mot_cible, mot_propose)
        score = similarity * 100
        st.session_state.historique.append((mot_propose, score))

        # Afficher le résultat de la tentative
        st.write(f"Similarité avec le mot cible : **{score:.2f}/100**")

        # Vérifier si le mot est exactement le bon
        if mot_propose == st.session_state.mot_cible:
            st.balloons()
            st.success(f"🎉 Bravo ! Vous avez trouvé le mot cible : **{st.session_state.mot_cible}** !")
            st.session_state.jeu_termine = True

        st.session_state.tentatives_restantes -= 1

        # Afficher l'historique
        st.subheader("Historique des tentatives :")
        for mot, score in st.session_state.historique:
            st.write(f"- **{mot}** : {score:.2f}/100")

        # Fin du jeu si plus de tentatives
        if st.session_state.tentatives_restantes <= 0:
            st.error(f"😢 Game Over ! Le mot cible était : **{st.session_state.mot_cible}**.")
            st.session_state.jeu_termine = True

# Bouton pour rejouer
if st.session_state.jeu_termine:
    if st.button("Rejouer"):
        # Réinitialiser les variables de session
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
