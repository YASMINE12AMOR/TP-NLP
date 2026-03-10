import streamlit as st
import re

st.title("Pre-traitement des Donnees en NLP")

try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    from gensim.models import Word2Vec
except ModuleNotFoundError as e:
    st.error(
        f"Dependance manquante: {e.name}. Installe les dependances avec: pip install -r pages/requirements.txt"
    )
    st.stop()

# Telecharger les ressources NLTK (si non deja presentes)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Charger le modele spaCy pour la lemmatisation
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    st.error(
        "Le modele spaCy 'fr_core_news_sm' n'est pas installe. Installe-le avec : python -m spacy download fr_core_news_sm"
    )
    st.stop()

# Data
text = """
<p>Cette commune de 500 ames fait partie des petites cites de caractere. Situee dans le parc regional du Perche, elle presente un ordonnancement orthogonal typique des cites des 18e et 19e siecles.</p>
"""
text = st.text_area("Entrez votre texte brut :", text, height=300)

# Etape 1 : Nettoyage
st.subheader("1. Nettoyage du texte")
if st.button("Nettoyer le texte"):
    # TODO : Supprimer les balises html / la ponctuation / patterns recurrents hors vocabulaire / Uniformiser la casse
    cleanText = "?"

    st.write("**Texte nettoye :**", cleanText)
    st.session_state.cleanText = cleanText
else:
    st.write("Cliquez sur le bouton pour nettoyer le texte.")

st.markdown("---")

# Etape 2 : Tokenisation
st.subheader("2. Tokenisation")
if st.button("Tokeniser le texte"):
    if "cleanText" in st.session_state:

        # TODO tokeniser en mot et en phrases le texte nettoye
        tokenizedWords = "?"
        tokenizedSentences = "?"

        st.write("**Mots tokenises :**", tokenizedWords)
        st.write("**Phrases tokenisees :**", tokenizedSentences)
        st.session_state.tokenizedWords = tokenizedWords
    else:
        st.warning("Nettoyez d'abord le texte !")


# Etape 3 : Stemming et Lemmatisation
st.subheader("3. Stemming et Lemmatisation")
if st.button("Appliquer le Stemming"):
    if "tokenizedWords" in st.session_state:
        # TODO : Utiliser le stemmer

        stemmer = PorterStemmer()
        stemmedWords = "?"
        st.write("**Mots apres Stemming :**", stemmedWords)
        st.session_state.stemmedWords = stemmedWords
    else:
        st.warning("Tokenisez d'abord le texte !")

if st.button("Appliquer la Lemmatisation"):
    if "cleanText" in st.session_state:

        # TODO : Utiliser le lemmatizer
        lemmatized = nlp(st.session_state.cleanText)
        lemmatizedWords = "?"
        st.write("**Mots apres Lemmatisation :**", lemmatizedWords)
        st.session_state.lemmatizedWords = lemmatizedWords
    else:
        st.warning("Nettoyez d'abord le texte !")

st.markdown("---")

# Etape 4 : Suppression des Stop Words
st.subheader("4. Suppression des Stop Words")
if st.button("Supprimer les Stop Words"):
    if "lemmatizedWords" in st.session_state:

        # TODO : Generer les stop words francais et nettoyer le texte
        stopWords = "?"
        filteredWords = "?"
        st.write("**Mots apres suppression des Stop Words :**", filteredWords)
        st.session_state.filteredWords = filteredWords
    else:
        st.warning("Appliquez d'abord la lemmatisation !")

st.markdown("---")
