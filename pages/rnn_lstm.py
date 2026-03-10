# streamlit_rnn_lstm_secret.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

st.title("RNN & LSTM")

# ------------------------
# Phrase artificielle
# ------------------------
sentence = "Le chat mange la souris"
words = sentence.split()
vocab = list(set(words))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

st.subheader("Vocabulaire")
st.write(vocab)

X = torch.tensor([[word2idx[w] for w in words[:-1]]])
y = torch.tensor([word2idx[w] for w in words[1:]])

# ------------------------
# Hyperparamètres
# ------------------------
embedding_dim = 8
hidden_dim = 16
vocab_size = len(vocab)
epochs = 300
lr = 0.01

# ------------------------
# Modèle RNN/LSTM
# ------------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, rnn_type='RNN'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type=='RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type=='LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return out, hidden

# ------------------------
# Choix du modèle
# ------------------------
rnn_type = st.selectbox("Choisir le modèle", ["RNN", "LSTM"])
model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, rnn_type)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ------------------------
# Entraînement rapide
# ------------------------
for epoch in range(epochs):
    optimizer.zero_grad()
    out, hidden = model(X)
    logits = model.fc(out[0])
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

# ------------------------
# Visualisation étape par étape
# ------------------------
with torch.no_grad():
    out, hidden = model(X)
    for t in range(out.shape[1]):
        h_t = out[0,t].numpy()
        if rnn_type=='LSTM':
            c_t = hidden[1][0,:,0].numpy()  # batch=0, premier vecteur
        logits = model.fc(out[0,t])
        probs = F.softmax(logits, dim=-1).numpy()
        pred_word = vocab[np.argmax(probs)]
        
        st.markdown(f"**Mot {t+1}: {words[t]}**")
        st.write("h_t :", np.round(h_t, 3))
        if rnn_type=='LSTM':
            st.write("c_t :", np.round(c_t, 3))
        
        df_probs = pd.DataFrame([probs], columns=vocab)
        st.dataframe(df_probs.style.background_gradient(cmap='Blues'))
        st.write(f"Mot prédit le plus probable : **{pred_word}**")
