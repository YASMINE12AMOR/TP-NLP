# TP-NLP

Projet de traitement automatique du langage naturel avec plusieurs mini-applications Streamlit autour des embeddings, de la génération et de la recherche sémantique.

## Modules principaux

### `cemantik`
[`pages/cemantik.py`](d:/projet%20NLP/pages/cemantik.py) est une interface orientée sémantique. Le module sert à explorer la proximité de mots ou de concepts à partir de représentations vectorielles, pour montrer comment des embeddings capturent le sens au-delà des mots exacts.

### `code_names`
[`pages/code_names.py`](d:/projet%20NLP/pages/code_names.py) est une version manuelle du jeu Codenames. Un maître du jeu humain donne un indice et un nombre, puis les joueurs devinent les mots sur la grille. Le module gère les scores, les tours, les mots neutres, l’assassin et la victoire des équipes.

### `AI_code_names`
[`pages/AI_code_names.py`](d:/projet%20NLP/pages/AI_code_names.py) est une version enrichie de Codenames où l’IA joue le rôle du maître du jeu. L’application génère automatiquement un indice, peut le régénérer, et propose aussi une aide basée sur les embeddings et le clustering pour suggérer les mots les plus proches de l’indice.

### `rag`
[`pages/rag.py`](d:/projet%20NLP/pages/rag.py) est dédié au principe de Retrieval-Augmented Generation. Le module combine une phase de recherche d’informations pertinentes avec une phase de génération de réponse, afin de montrer comment enrichir un modèle avec du contexte récupéré depuis une base documentaire ou des textes.

### `encoder`
[`pages/encoder_bert.py`](d:/projet%20NLP/pages/encoder_bert.py) illustre la partie encodeur d’une architecture Transformer, ici autour de BERT. Ce module sert à comprendre comment un texte est transformé en représentations contextuelles utiles pour la classification, la similarité ou l’analyse sémantique.

### `decoder`
[`pages/decoder_gpt.py`](d:/projet%20NLP/pages/decoder_gpt.py) illustre la partie décodeur, dans l’esprit GPT. Le module met l’accent sur la génération auto-régressive de texte, c’est-à-dire la prédiction du prochain mot à partir du contexte précédent.

## Autres fichiers utiles

- [`pages/preprocessing.py`](d:/projet%20NLP/pages/preprocessing.py) : nettoyage et préparation des textes.
- [`pages/bow_tfidf.py`](d:/projet%20NLP/pages/bow_tfidf.py) : représentation classique des textes avec Bag of Words et TF-IDF.
- [`pages/word2vec.py`](d:/projet%20NLP/pages/word2vec.py) : apprentissage ou exploration d’embeddings de mots.
- [`pages/rnn_lstm.py`](d:/projet%20NLP/pages/rnn_lstm.py) : modèles séquentiels de type RNN/LSTM.
- [`pages/app.py`](d:/projet%20NLP/pages/app.py) : point d’entrée de navigation Streamlit.

## Lancement

Pour lancer l’application Streamlit :

```bash
streamlit run pages/app.py
```
