import streamlit as st
import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.title("Bag of Words & TF-IDF")

DOCUMENT = st.text_area(
    "Entrez des phrases.",
    "J'aime le NLP.\nLe NLP est passionnant.\nLe NLP et l'IA sont liés.",
)

if DOCUMENT:

    st.subheader("Bag of Word")
    # TODO : Séparer les phrases
    docs = DOCUMENT.split("\n")

    # TODO : Utiliser le countVectorizer() pour compter les occurences (Voir doc sklearn)
    vectorizer = CountVectorizer()
    vector = "?"

    # TODO : Récupérer les mots plutôt que les index pour la lisibilité
    columns = "?"  # mot 0 / mot 1 ...
    indexes = "?"  # phrase 0 / phrase 1 ...

    # TODO : Affichage (à décommenter)
    # df = pandas.DataFrame(vector.toarray(), columns=columns, index=indexes)
    # st.subheader("Matrice Bag of Words")
    # st.dataframe(df, width=800)

    st.subheader("TF")

    # TODO : Utiliser TfidfVectorizer() (voir doc sklearn) pour générer uniquement TF
    # TODO : Utiliser TfidfVectorizer() (voir doc sklearn) pour générer TF-IDF et récupérer le vecteur IDF et afficher TF-IDF

    tf_vectorizer = TfidfVectorizer(use_idf=False, norm="l1", smooth_idf=False)
    idf_vectorizer = TfidfVectorizer(use_idf=True, norm="l1", smooth_idf=False)

    tfVector = "?"
    idfVector = "?"
    columns = "?"
    indexes = "?"

    # TODO : Affichage (à décommenter)
    # tf_df = pandas.DataFrame(tfVector.toarray(), columns=columns, index=indexes)
    # idf_df = pandas.DataFrame(idfVector.toarray(), columns=columns, index=indexes)
    # st.subheader("Matrice Bag of Words")
    # st.dataframe(df, width=800)
