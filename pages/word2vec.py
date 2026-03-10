import streamlit as st
import pandas
import plotly.express as px
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import random
import numpy as np


st.set_page_config(page_title="TP NLP & IA Générative", layout="wide")
st.title("Word 2 Vec et distance cosinus")

mots = ["chat", "chien", "souris", "ballon", "table", "fromage", "arbre"]
actions = ["mange", "court", "dort", "saute", "joue", "chasse", "cache"]
adjectifs = ["petit", "grand", "rapide", "fort", "agile"]

docs = []
for _ in range(10000):  # 1000 phrases
    phrase = f"{random.choice(mots)} {random.choice(actions)} {random.choice(mots)} {random.choice(adjectifs)}"
    docs.append(phrase)


def vectors_to_df(model):
    words = model.wv.index_to_key
    vectors = [model.wv[word] for word in words]
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)
    df = pandas.DataFrame(vectors_3d, columns=["x", "y", "z"])
    df["word"] = words
    return df


if docs:
    tokenizedDocs = [phrase.split() for phrase in docs]

    vector_size = 50
    window = 3
    min_count = 1
    sg = 0
    epochs = 10
    sentences = tokenizedDocs
    cbow = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=1,
    )

    vector_size = 50
    window = 3
    min_count = 1
    sg = 1
    epochs = 10
    sentences = tokenizedDocs
    skipgram = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=1,
    )

    df_cbow = vectors_to_df(cbow)
    df_skip = vectors_to_df(skipgram)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CBOW - Vecteurs")
        fig_cbow = px.scatter_3d(
            df_cbow, x="x", y="y", z="z", text="word", width=400, height=400
        )
        st.plotly_chart(fig_cbow)
        st.markdown("### Similarite cosinus entre des mots")
        vocab_cbow = cbow.wv.index_to_key
        mot1_cbow = st.selectbox("Mot 1 (CBOW)", vocab_cbow, key="mot1_cbow")
        mot2_cbow = st.selectbox("Mot 2 (CBOW)", vocab_cbow, key="mot2_cbow", index=1)
        vec1_cbow = cbow.wv.get_vector(mot1_cbow, norm=True)
        vec2_cbow = cbow.wv.get_vector(mot2_cbow, norm=True)
        sim_cbow = float(np.dot(vec1_cbow, vec2_cbow))
        st.write(f"Similarite cosinus: **{sim_cbow:.4f}**")
    with col2:
        st.subheader("Skip-Gram - Vecteurs")
        fig_skip = px.scatter_3d(
            df_skip, x="x", y="y", z="z", text="word", width=400, height=400
        )
        st.plotly_chart(fig_skip)
        st.markdown("### Similarite cosinus entre des mots")
        vocab_skip = skipgram.wv.index_to_key
        mot1_skip = st.selectbox("Mot 1 (Skip-Gram)", vocab_skip, key="mot1_skip")
        mot2_skip = st.selectbox(
            "Mot 2 (Skip-Gram)", vocab_skip, key="mot2_skip", index=1
        )
        vec1_skip = skipgram.wv.get_vector(mot1_skip, norm=True)
        vec2_skip = skipgram.wv.get_vector(mot2_skip, norm=True)
        sim_skip = float(np.dot(vec1_skip, vec2_skip))
        st.write(f"Similarite cosinus: **{sim_skip:.4f}**")
