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

#### Codenames

![Interface Codenames](docs/images/code_names.png)

#### Codenames avec IA

![Interface Codenames IA](docs/images/play_code_names_ai.png)

#### Cemantik

![Interface Cemantik](docs/images/cemantik.png)

#### Encoder BERT

![Interface Encoder BERT](docs/images/encoder.png)

#### Decoder GPT

![Interface Decoder GPT](docs/images/decoder.png)

#### RAG

![Interface RAG](docs/images/rag.png)
