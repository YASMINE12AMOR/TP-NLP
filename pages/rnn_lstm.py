import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from io import StringIO


def format_probabilities(probs, vocab, top_k=None):
    df = pd.DataFrame({
        "Mot": vocab,
        "Probabilite": probs,
    }).sort_values("Probabilite", ascending=False)
    if top_k is not None:
        df = df.head(top_k)
    df["Probabilite"] = df["Probabilite"].map(lambda x: round(float(x), 4))
    return df.reset_index(drop=True)

# ------------------------
# Configuration de la page
# ------------------------
st.set_page_config(
    page_title="RNN & LSTM - Prédiction de mots",
    page_icon="🤖",
    layout="wide"
)

# ------------------------
# Titre et description
# ------------------------
st.title("🧠 RNN & LSTM pour la prédiction de mots")
st.markdown("""
Cette application entraîne un modèle **RNN** ou **LSTM** pour prédire le mot suivant dans une phrase.
- **Choisissez** le type de modèle (RNN/LSTM).
- **Entraînez** le modèle sur une phrase.
- **Visualisez** les états cachés et les prédictions.
""")

# ------------------------
# Saisie de la phrase par l'utilisateur
# ------------------------
default_sentence = "Le chat mange la souris"
sentence = st.text_input("Entrez une phrase (ex: 'Le chat mange la souris'):", value=default_sentence)

# ------------------------
# Prétraitement des données
# ------------------------
words = sentence.split()
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# Affichage du vocabulaire
st.subheader("📚 Vocabulaire")
st.write(f"**Mots uniques:** {vocab}")
st.write(f"**Taille du vocabulaire:** {len(vocab)}")

# Préparation des données d'entraînement
X = torch.tensor([[word2idx[w] for w in words[:-1]]])  # Tous les mots sauf le dernier
y = torch.tensor([word2idx[w] for w in words[1:]])    # Tous les mots sauf le premier

# ------------------------
# Hyperparamètres (réglables par l'utilisateur)
# ------------------------
st.sidebar.header("⚙️ Paramètres du modèle")
embedding_dim = st.sidebar.slider("Dimension des embeddings", 4, 32, 8)
hidden_dim = st.sidebar.slider("Dimension cachée", 8, 64, 16)
epochs = st.sidebar.slider("Nombre d'époques", 50, 500, 300)
lr = st.sidebar.select_slider("Taux d'apprentissage", options=[0.1, 0.01, 0.001, 0.0001], value=0.01)

# ------------------------
# Définition du modèle RNN/LSTM
# ------------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, rnn_type='RNN'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return out, hidden

# ------------------------
# Sélection du modèle
# ------------------------
rnn_type = st.sidebar.selectbox("Type de modèle", ["RNN", "LSTM"])

# ------------------------
# Initialisation du modèle
# ------------------------
vocab_size = len(vocab)
model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, rnn_type)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ------------------------
# Bouton pour lancer l'entraînement
# ------------------------
if st.sidebar.button("🚀 Entraîner le modèle"):
    with st.spinner("Entraînement en cours..."):
        for epoch in range(epochs):
            optimizer.zero_grad()
            out, hidden = model(X)
            logits = model.fc(out[0])
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Affichage de la perte toutes les 50 époques
            if (epoch + 1) % 50 == 0:
                st.sidebar.write(f"Époque {epoch + 1}/{epochs} - Perte: {loss.item():.4f}")

    st.sidebar.success("✅ Entraînement terminé!")

# ------------------------
# Visualisation des prédictions
# ------------------------
st.subheader("🔮 Prédictions du modèle")

with torch.no_grad():
    out, hidden = model(X)
    for t in range(out.shape[1]):
        st.markdown(f"---")
        st.markdown(f"**Mot d'entrée:** {words[t]}")

        # État caché (h_t)
        h_t = out[0, t].numpy()
        st.write("**État caché (h_t):**", np.round(h_t, 3))

        # Cellule de mémoire (c_t) pour LSTM
        if rnn_type == 'LSTM':
            c_t = hidden[1][0, :, 0].numpy()  # batch=0, premier vecteur
            st.write("**Cellule de mémoire (c_t):**", np.round(c_t, 3))

        # Prédictions
        logits = model.fc(out[0, t])
        probs = F.softmax(logits, dim=-1).numpy()
        pred_word = idx2word[np.argmax(probs)]

        # Affichage des probabilités
        df_probs = format_probabilities(probs, vocab, top_k=min(5, len(vocab)))
        st.dataframe(df_probs, use_container_width=True)

        # Mot prédit
        st.write(f"**Mot prédit le plus probable:** **{pred_word}**")

        # Mot réel (pour comparaison)
        if t < len(words) - 1:
            st.write(f"**Mot réel suivant:** {words[t + 1]}")

# ------------------------
# Test avec une nouvelle phrase
# ------------------------
st.subheader("🧪 Tester avec une nouvelle phrase")
test_sentence = st.text_input("Entrez une nouvelle phrase pour tester le modèle:", value="Le chien aboie")
if st.button("Tester"):
    test_words = test_sentence.split()
    test_X = torch.tensor([[word2idx.get(w, 0) for w in test_words[:-1]]])  # 0 pour les mots inconnus

    with torch.no_grad():
        out, _ = model(test_X)
        logits = model.fc(out[0, -1])  # Prédiction pour le dernier mot
        probs = F.softmax(logits, dim=-1).numpy()
        pred_word = idx2word[np.argmax(probs)]

        st.write(f"**Phrase test:** {' '.join(test_words)}")
        st.write(f"**Mot prédit:** {pred_word}")
        st.write("**Probabilités:**")
        test_df = format_probabilities(probs, vocab, top_k=min(5, len(vocab)))
        st.dataframe(test_df, use_container_width=True)
