# import os
# import re
import streamlit as st
from pathlib import Path
# import plotly.express as px
# import pandas
# import numpy as np
# import spacy
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# import random
# from scipy.spatial.distance import cosine
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import PCA
# from transformers import pipeline, Mistral3ForConditionalGeneration, MistralCommonBackend, AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from mistralai import Mistral
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import faiss


BASE_DIR = Path(__file__).resolve().parent

pages = {
  "NLP" : [
    st.Page(str(BASE_DIR / "preprocessing.py"), title="Preprocessing"),
    st.Page(str(BASE_DIR / "bow_tfidf.py"), title="Bag of Word & TF-IDF"),
    st.Page(str(BASE_DIR / "word2vec.py"), title="Word2vec"),
    st.Page(str(BASE_DIR / "rnn_lstm.py"), title="RNN LSTM"),
    st.Page(str(BASE_DIR / "encoder_bert.py"), title="Encoder BERT"),
    st.Page(str(BASE_DIR / "decoder_gpt.py"), title="Decoder GPT"),
    st.Page(str(BASE_DIR / "rag.py"), title="RAG"),
    st.Page(str(BASE_DIR / "cemantik.py"), title="cemantik"),
    st.Page(str(BASE_DIR / "code_names.py"), title="code names"),
    st.Page(str(BASE_DIR / "AI_code_names.py"), title="Play code names with AI"),
    st.Page(str(BASE_DIR / "image_generation.py"), title="generate image with ai")
  ]
}
pg = st.navigation(pages)
pg.run()
