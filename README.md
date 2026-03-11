# TP-NLP

Projet Streamlit de demonstration autour du NLP, des embeddings, des modeles generatifs et de la recherche semantique.

## Lancement

Installer les dependances :

```bash
pip install -r pages/requirements.txt
```

Lancer l'application :

```bash
streamlit run pages/app.py
```

## Logique generale du projet

Le fichier [`pages/app.py`](/d:/projet%20NLP/pages/app.py) est le point d'entree. Il construit la navigation Streamlit et affiche les pages actives du projet :

- `code_names.py`
- `AI_code_names.py`
- `cemantik.py`
- `encoder_bert.py`
- `decoder_gpt.py`
- `rag.py`

Certaines autres pages sont presentes dans le dossier `pages/` mais sont commentees dans la navigation. Elles servent surtout d'exemples pedagogiques ou de prototypes.

## Logique de chaque fichier

### Navigation

#### [`pages/app.py`](/d:/projet%20NLP/pages/app.py)

Centralise la navigation Streamlit avec `st.navigation()`.
Chaque entree pointe vers une mini-application NLP ou IA.

### Jeux et interfaces interactives

#### [`pages/code_names.py`](/d:/projet%20NLP/pages/code_names.py)

Version classique de Codenames en francais.

Logique :

- genere une grille aleatoire de 25 mots
- assigne les roles `red`, `blue`, `neutral`, `black`
- gere les tours des equipes
- gere les indices saisis par le maitre du jeu
- calcule les scores et les conditions de victoire
- affiche deux vues : joueurs et maitre du jeu

Modeles utilises :

- aucun modele IA
- logique basee uniquement sur `random` et `st.session_state`

#### [`pages/AI_code_names.py`](/d:/projet%20NLP/pages/AI_code_names.py)

Version de Codenames ou l'IA joue le role du maitre du jeu.

Logique :

- cree une grille de mots et des equipes comme dans la version classique
- charge un modele d'embeddings pour representer les mots
- genere automatiquement un indice semantique pour l'equipe courante
- estime les risques lies aux mots adverses, neutres ou assassin
- propose une assistance supplementaire par clustering des mots proches de l'indice
- permet de regenerer un indice automatiquement

Modeles et methodes utilises :

- `SentenceTransformer("all-MiniLM-L6-v2")`
- similarite cosinus sur embeddings
- `AgglomerativeClustering` pour grouper les mots proches
- heuristiques de selection d'indice semantique

#### [`pages/cemantik.py`](/d:/projet%20NLP/pages/cemantik.py)

Jeu de devinette semantique en deux joueurs.

Logique :

- le joueur 1 saisit un mot secret
- le joueur 2 propose jusqu'a 5 mots
- chaque proposition est encodee en vecteur
- l'application calcule un score de similarite entre le mot cible et le mot propose
- l'historique des essais sert de feedback progressif

Modeles utilises :

- `CamembertTokenizer`
- `CamembertModel`
- modele charge : `camembert-base`
- similarite cosinus via `sklearn.metrics.pairwise.cosine_similarity`

### Pages de demonstration de modeles

#### [`pages/encoder_bert.py`](/d:/projet%20NLP/pages/encoder_bert.py)

Interface de classification comptable inspiree d'un encodeur de type BERT.

Logique :

- contient une base de termes comptables structuree en `actif` et `passif`
- tente d'abord une correspondance exacte
- si aucun terme exact n'est trouve, calcule une similarite semantique avec les exemples de la base
- retourne la categorie la plus proche si le score depasse un seuil de `0.65`

Modeles utilises :

- `CamembertTokenizer`
- `CamembertModel`
- modele charge : `camembert-base`
- similarite cosinus sur embeddings de texte

#### [`pages/decoder_gpt.py`](/d:/projet%20NLP/pages/decoder_gpt.py)

Interface assistant de generation de texte et de code.

Logique :

- charge la cle API Mistral depuis `.env`
- propose trois usages :
  - chat / questions-reponses
  - resume de texte
  - generation de code
- construit un prompt selon la tache choisie
- envoie la requete a l'API Mistral
- affiche la reponse textuelle ou le code genere

Modeles utilises :

- API `mistralai`
- modele configure : `mistral-tiny`

#### [`pages/rag.py`](/d:/projet%20NLP/pages/rag.py)

Demonstration complete d'un pipeline RAG pour recommander des projets academiques.

Logique :

- stocke un petit catalogue de projets avec titre, description, tags, niveau et lien
- encode les descriptions en embeddings
- construit un index vectoriel FAISS
- retrouve les projets les plus proches d'une requete utilisateur
- construit un contexte a partir des documents retrouves
- demande a Mistral de generer une recommandation en s'appuyant sur ce contexte

Modeles et composants utilises :

- `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`
- `faiss.IndexFlatL2`
- API `mistralai`
- modele configure : `mistral-small-latest`

### Pages pedagogiques supplementaires

#### [`pages/preprocessing.py`](/d:/projet%20NLP/pages/preprocessing.py)

Page de pretraitement NLP.

Logique :

- prend un texte brut
- prevoyait les etapes suivantes :
  - nettoyage
  - tokenisation
  - stemming
  - lemmatisation
  - suppression des stop words
- plusieurs parties sont encore marquees `TODO`, donc la page est surtout un support de TP

Modeles et bibliotheques utilises :

- `nltk`
- `spaCy`
- modele spaCy : `fr_core_news_sm`
- `PorterStemmer`

#### [`pages/bow_tfidf.py`](/d:/projet%20NLP/pages/bow_tfidf.py)

Page d'introduction aux representations classiques de texte.

Logique :

- separe plusieurs phrases
- doit construire une matrice Bag of Words
- doit ensuite calculer TF puis TF-IDF
- plusieurs blocs restent en `TODO`, donc c'est un squelette de TP

Modeles et methodes utilises :

- `CountVectorizer`
- `TfidfVectorizer`

#### [`pages/word2vec.py`](/d:/projet%20NLP/pages/word2vec.py)

Demonstration d'embeddings appris sur un corpus synthetique.

Logique :

- genere des phrases aleatoires a partir d'un petit vocabulaire
- entraine deux modeles `Word2Vec`
- projette les vecteurs en 3D avec PCA
- affiche la position des mots et leur similarite cosinus

Modeles utilises :

- `gensim.models.Word2Vec`
- mode `CBOW` avec `sg=0`
- mode `Skip-Gram` avec `sg=1`
- `PCA` pour la visualisation

#### [`pages/rnn_lstm.py`](/d:/projet%20NLP/pages/rnn_lstm.py)

Page pedagogique de prediction du mot suivant.

Logique :

- construit un vocabulaire a partir d'une phrase entree par l'utilisateur
- entraine un petit reseau sur cette phrase
- permet de choisir entre `RNN` et `LSTM`
- affiche les etats caches, les probabilites et la prediction du mot suivant
- permet un test sur une nouvelle phrase

Modeles utilises :

- `torch.nn.RNN`
- `torch.nn.LSTM`
- couche `Embedding`
- couche `Linear`

### Scripts et utilitaires

#### [`pages/image_generation.py`](/d:/projet%20NLP/pages/image_generation.py)

Script de generation d'image a partir d'un prompt.

Logique :

- construit un prompt texte enrichi par un contexte de style
- charge un pipeline Stable Diffusion XL
- genere une image sur CPU
- sauvegarde le resultat dans `sdxl_cpu_result.png`

Modele utilise :

- `StableDiffusionXLPipeline`
- modele charge : `stabilityai/stable-diffusion-xl-base-1.0`

#### [`pages/test.py`](/d:/projet%20NLP/pages/test.py)

Petit script de verification de l'environnement PyTorch.

Logique :

- detecte si CUDA est disponible
- cree un tenseur sur le device selectionne
- affiche le resultat dans le terminal

#### [`pages/requirements.txt`](/d:/projet%20NLP/pages/requirements.txt)

Liste des dependances Python du projet :

- Streamlit pour les interfaces
- Transformers, sentence-transformers, PyTorch pour les modeles NLP
- FAISS pour la recherche vectorielle
- spaCy, NLTK, gensim, scikit-learn pour les traitements NLP classiques
- diffusers pour la generation d'image

## Resume des modeles utilises

| Fichier | Modele / methode principale | Usage |
| --- | --- | --- |
| `cemantik.py` | `camembert-base` | similarite semantique entre mots |
| `encoder_bert.py` | `camembert-base` | classification comptable par similarite |
| `decoder_gpt.py` | `mistral-tiny` | chat, resume, generation de code |
| `rag.py` | `paraphrase-multilingual-MiniLM-L12-v2` + `mistral-small-latest` + FAISS | retrieval + generation |
| `AI_code_names.py` | `all-MiniLM-L6-v2` | generation d'indices et aide semantique |
| `word2vec.py` | `Word2Vec` CBOW / Skip-Gram | apprentissage d'embeddings |
| `rnn_lstm.py` | `RNN` / `LSTM` | prediction du mot suivant |
| `preprocessing.py` | `fr_core_news_sm` + NLTK | pretraitement linguistique |
| `image_generation.py` | `Stable Diffusion XL` | generation d'image |

## Images des interfaces

Vous pouvez ajouter un dossier comme `docs/images/` a la racine du projet, puis deposer une capture de chaque interface.

Exemple d'arborescence :

```text
docs/
  images/
    app.png
    codenames.png
    ai_codenames.png
    cemantik.png
    encoder_bert.png
    decoder_gpt.png
    rag.png
```

Ensuite, vous pouvez inserer les images directement dans ce `README`.

### Captures recommandees

#### Navigation principale

![Navigation principale](docs/images/app.png)

#### Codenames

![Interface Codenames](docs/images/codenames.png)

#### Codenames avec IA

![Interface Codenames IA](docs/images/ai_codenames.png)

#### Cemantik

![Interface Cemantik](docs/images/cemantik.png)

#### Encoder BERT

![Interface Encoder BERT](docs/images/encoder_bert.png)

#### Decoder GPT

![Interface Decoder GPT](docs/images/decoder_gpt.png)

#### RAG

![Interface RAG](docs/images/rag.png)

## Remarques

- `decoder_gpt.py` et `rag.py` necessitent une variable `MISTRAL_API_KEY` dans le fichier `.env`.
- `preprocessing.py` depend aussi du modele spaCy `fr_core_news_sm`.
- Certaines pages sont des supports de TP et contiennent encore des `TODO`.
