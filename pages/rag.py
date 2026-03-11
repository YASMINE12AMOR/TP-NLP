import os
from pathlib import Path
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduRAG · Recherche Intelligente de Projets",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Google Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Source+Sans+3:wght@300;400;600&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ---------- CSS Variables ---------- */
    :root {
        --cream:      #faf7f2;
        --parchment:  #f0ead8;
        --ink:        #1a1209;
        --sepia:      #5c4827;
        --amber:      #c87941;
        --gold:       #e8a84c;
        --moss:       #3d6b52;
        --sky:        #2d6a8f;
        --rule:       #c8b99a;
        --shadow:     rgba(28, 18, 5, 0.10);
        --radius:     10px;
    }

    /* ---------- Base ---------- */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--cream);
        color: var(--ink);
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 300;
    }

    /* Subtle paper texture via repeating gradient */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 27px,
                rgba(200,185,154,0.18) 27px,
                rgba(200,185,154,0.18) 28px
            );
        pointer-events: none;
        z-index: 0;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #2a1e0e 0%, #1a120a 100%);
        border-right: 3px solid var(--amber);
    }
    [data-testid="stSidebar"] * {
        color: var(--parchment) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Playfair Display', serif;
        color: var(--gold) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--amber) !important;
        opacity: 0.4;
    }

    /* ---------- Header ---------- */
    .edu-header {
        text-align: center;
        padding: 2.8rem 1rem 1.6rem;
        border-bottom: 2px solid var(--rule);
        margin-bottom: 2rem;
        position: relative;
    }
    .edu-header::after {
        content: "✦";
        position: absolute;
        bottom: -0.75rem;
        left: 50%;
        transform: translateX(-50%);
        background: var(--cream);
        padding: 0 .75rem;
        color: var(--amber);
        font-size: 1.1rem;
    }
    .edu-header .badge {
        display: inline-block;
        background: var(--amber);
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
        font-size: .72rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        padding: .28rem .9rem;
        border-radius: 99px;
        margin-bottom: .9rem;
    }
    .edu-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.9rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 .4rem;
        line-height: 1.15;
    }
    .edu-header p {
        color: var(--sepia);
        font-size: 1.05rem;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.7;
    }

    /* ---------- Search box ---------- */
    .stTextInput > div > div > input {
        background: #fff;
        border: 1.5px solid var(--rule) !important;
        border-radius: var(--radius) !important;
        padding: .75rem 1rem !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 1rem !important;
        color: var(--ink) !important;
        box-shadow: 0 2px 8px var(--shadow) !important;
        transition: border-color .2s, box-shadow .2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 3px rgba(200,121,65,.15) !important;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--amber) 0%, #b5672f 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: .65rem 2rem !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: .04em !important;
        box-shadow: 0 3px 12px rgba(200,121,65,.35) !important;
        transition: transform .15s, box-shadow .15s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(200,121,65,.45) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ---------- Answer card ---------- */
    .answer-card {
        background: #fff;
        border-left: 4px solid var(--amber);
        border-radius: var(--radius);
        padding: 1.5rem 1.8rem;
        margin: 1rem 0 1.8rem;
        box-shadow: 0 4px 20px var(--shadow);
        font-size: 1rem;
        line-height: 1.75;
        color: var(--ink);
    }
    .answer-card h3 { font-family: 'Playfair Display', serif; color: var(--sepia); }

    /* ---------- Section headings ---------- */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.45rem;
        color: var(--sepia);
        border-bottom: 1px solid var(--rule);
        padding-bottom: .4rem;
        margin: 1.6rem 0 1rem;
    }
    .section-title span {
        margin-right: .5rem;
    }

    /* ---------- Expanders / Project cards ---------- */
    [data-testid="stExpander"] {
        background: #fff !important;
        border: 1px solid var(--rule) !important;
        border-radius: var(--radius) !important;
        margin-bottom: .6rem !important;
        box-shadow: 0 2px 8px var(--shadow) !important;
        overflow: hidden;
        transition: box-shadow .2s;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 16px var(--shadow) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        font-size: .97rem !important;
        color: var(--ink) !important;
        padding: .7rem 1rem !important;
        background: var(--parchment) !important;
    }

    /* ---------- Tag pills ---------- */
    .tag-pill {
        display: inline-block;
        background: var(--parchment);
        color: var(--sepia);
        border: 1px solid var(--rule);
        border-radius: 99px;
        font-family: 'JetBrains Mono', monospace;
        font-size: .72rem;
        padding: .18rem .7rem;
        margin: .15rem .2rem .15rem 0;
        font-weight: 600;
    }
    .tag-pill.nlp    { background: #dff0e8; border-color: #8dc4a8; color: var(--moss); }
    .tag-pill.data   { background: #dde8f5; border-color: #8aaad1; color: var(--sky); }
    .tag-pill.iot    { background: #fdefd8; border-color: #e8b87a; color: #8a5a1f; }
    .tag-pill.dl     { background: #f2ddf2; border-color: #c49ac4; color: #6b2e6b; }
    .tag-pill.block  { background: #ddf5f2; border-color: #7ac4bc; color: #1e6e68; }

    /* ---------- Stats bar ---------- */
    .stats-bar {
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin: 1rem 0 2rem;
    }
    .stat-item {
        background: #fff;
        border: 1px solid var(--rule);
        border-radius: var(--radius);
        padding: .8rem 1.2rem;
        flex: 1;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 2px 6px var(--shadow);
    }
    .stat-item .stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: var(--amber);
        display: block;
    }
    .stat-item .stat-lbl {
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--sepia);
    }

    /* ---------- Divider ---------- */
    hr { border-color: var(--rule) !important; }

    /* ---------- Spinner ---------- */
    .stSpinner > div { border-top-color: var(--amber) !important; }

    /* ---------- Warning / info ---------- */
    .stAlert { border-radius: var(--radius) !important; }

    /* ---------- Links ---------- */
    a { color: var(--sky) !important; font-weight: 600; text-decoration: none; }
    a:hover { color: var(--amber) !important; text-decoration: underline; }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--parchment); }
    ::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 99px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sample documents ──────────────────────────────────────────────────────────
documents = [
    {
        "title": "Chatbot Éducatif avec RASA",
        "description": "Un chatbot basé sur RASA pour répondre aux questions des étudiants en master sur leurs cours. Idéal pour les projets NLP. Utilise des intentions et entités pour comprendre les questions des étudiants.",
        "tags": ["NLP", "Éducation", "Python", "Chatbot"],
        "lien": "https://rasa.com/docs/rasa/",
        "niveau": "Avancé",
        "domaine": "NLP",
    },
    {
        "title": "Détection de Fraudes avec PyTorch",
        "description": "Modèle de détection de fraudes bancaires utilisant PyTorch et des techniques de Data Science. Utilise des réseaux de neurones profonds pour analyser les transactions suspectes.",
        "tags": ["Data Science", "Sécurité", "PyTorch", "Fraude"],
        "lien": "https://pytorch.org/tutorials/",
        "niveau": "Avancé",
        "domaine": "Data Science",
    },
    {
        "title": "Système de Recommandation pour les MOOCs",
        "description": "Un système de recommandation de cours en ligne basé sur les préférences des étudiants. Utilise des algorithmes de filtrage collaboratif et de contenu pour personnaliser les recommandations.",
        "tags": ["Data Science", "Recommandation", "Scikit-learn", "Éducation"],
        "lien": "https://scikit-learn.org/",
        "niveau": "Intermédiaire",
        "domaine": "Data Science",
    },
    {
        "title": "Monitoring de la Qualité de l'Air avec Raspberry Pi",
        "description": "Projet IoT pour surveiller la qualité de l'air dans une salle de classe. Utilise des capteurs de CO2, de particules fines et de température, avec transmission des données via MQTT et stockage dans une base de données InfluxDB.",
        "tags": ["IoT", "Raspberry Pi", "MQTT", "Environnement"],
        "lien": "https://projects.raspberrypi.org/en/projects/air-quality",
        "niveau": "Intermédiaire",
        "domaine": "IoT",
    },
    {
        "title": "Certification Décentralisée avec Blockchain",
        "description": "Plateforme de certification des diplômes utilisant la blockchain Ethereum et des smart contracts en Solidity. Permet de vérifier l'authenticité des diplômes de manière sécurisée et transparente.",
        "tags": ["Blockchain", "Solidity", "Ethereum", "Éducation"],
        "lien": "https://docs.soliditylang.org/",
        "niveau": "Avancé",
        "domaine": "Blockchain",
    },
    {
        "title": "Analyse des Sentiments sur les Réseaux Sociaux",
        "description": "Projet d'analyse des sentiments sur Twitter ou Facebook utilisant des modèles de NLP comme BERT ou RoBERTa. Permet de comprendre les opinions des utilisateurs sur un sujet donné.",
        "tags": ["NLP", "Réseaux Sociaux", "BERT", "Analyse de Sentiments"],
        "lien": "https://huggingface.co/transformers/",
        "niveau": "Avancé",
        "domaine": "NLP",
    },
    {
        "title": "Reconnaissance d'Images Médicales avec TensorFlow",
        "description": "Modèle de reconnaissance d'images médicales utilisant TensorFlow et des réseaux de neurones convolutifs (CNN). Idéal pour détecter des anomalies dans des radiographies ou des IRM.",
        "tags": ["Deep Learning", "TensorFlow", "Santé", "CNN"],
        "lien": "https://www.tensorflow.org/tutorials",
        "niveau": "Avancé",
        "domaine": "Deep Learning",
    },
    {
        "title": "Automatisation Domotique avec Home Assistant",
        "description": "Projet de domotique utilisant Home Assistant pour automatiser les tâches ménagères. Intègre des capteurs IoT, des actionneurs et des scénarios personnalisés.",
        "tags": ["IoT", "Domotique", "Home Assistant", "Automatisation"],
        "lien": "https://www.home-assistant.io/",
        "niveau": "Débutant",
        "domaine": "IoT",
    },
    {
        "title": "Traduction Automatique avec Transformers",
        "description": "Système de traduction automatique utilisant des modèles de transformers comme MarianMT ou T5. Permet de traduire des textes entre plusieurs langues avec une haute précision.",
        "tags": ["NLP", "Traduction", "Transformers", "MarianMT"],
        "lien": "https://huggingface.co/transformers/",
        "niveau": "Avancé",
        "domaine": "NLP",
    },
    {
        "title": "Prédiction des Ventes avec Prophet",
        "description": "Utilisation de Facebook Prophet pour prédire les ventes futures d'une entreprise. Idéal pour les projets de business intelligence et d'analyse prédictive.",
        "tags": ["Data Science", "Prédiction", "Prophet", "Business Intelligence"],
        "lien": "https://facebook.github.io/prophet/",
        "niveau": "Intermédiaire",
        "domaine": "Data Science",
    },
]

# ── Env / API ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "mistral-small-latest"


@st.cache_resource
def get_client():
    if not api_key:
        return None
    return Mistral(api_key=api_key)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def build_faiss_index():
    model = load_embedding_model()
    descriptions = [doc["description"] for doc in documents]
    embeddings = model.encode(descriptions, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model


def retrieve_docs(query, index, model, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]


def generate_response(query, retrieved_docs):
    context = "\n\n".join(
        [
            f"Titre: {doc['title']}\nDescription: {doc['description']}\n"
            f"Tags: {', '.join(doc['tags'])}\nNiveau: {doc['niveau']}\nLien: {doc['lien']}"
            for doc in retrieved_docs
        ]
    )
    prompt = f"""
Tu es un assistant pédagogique expert qui recommande des idées de projets académiques.
Base-toi uniquement sur les documents ci-dessous. Si l'information n'apparaît pas dans le contexte, dis-le clairement.

Question étudiant: {query}

Contexte des projets disponibles:
{context}

Réponds en français avec une structure claire:
1. 🎯 **Recommandation principale** — nomme le projet et explique pourquoi il correspond à la demande
2. 📚 **Compétences développées** — liste les compétences clés que l'étudiant acquerra
3. 🔄 **2 projets alternatifs** — avec une courte justification pour chacun
4. 🔗 **Ressources pour démarrer** — liens utiles tirés des projets cités
"""
    client = get_client()
    if client is None:
        return (
            "⚠️ **MISTRAL_API_KEY manquante.** Le modèle LLM n'est pas disponible. "
            "Les documents les plus proches de votre recherche sont affichés ci-dessous."
        )
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ── Helpers ───────────────────────────────────────────────────────────────────
TAG_CLASS = {
    "NLP": "nlp", "BERT": "nlp", "Chatbot": "nlp", "Transformers": "nlp",
    "Data Science": "data", "Scikit-learn": "data", "Prophet": "data",
    "PyTorch": "data", "Prédiction": "data", "Recommandation": "data",
    "IoT": "iot", "Raspberry Pi": "iot", "MQTT": "iot", "Domotique": "iot",
    "Deep Learning": "dl", "TensorFlow": "dl", "CNN": "dl",
    "Blockchain": "block", "Ethereum": "block", "Solidity": "block",
}

NIVEAU_COLOR = {"Débutant": "#3d6b52", "Intermédiaire": "#2d6a8f", "Avancé": "#c87941"}
DOMAINE_ICON = {
    "NLP": "🗣️", "Data Science": "📊", "IoT": "🔌",
    "Deep Learning": "🧠", "Blockchain": "🔗",
}


def tag_pills(tags):
    pills = ""
    for t in tags:
        cls = TAG_CLASS.get(t, "")
        pills += f'<span class="tag-pill {cls}">{t}</span>'
    return pills


def niveau_badge(niveau):
    color = NIVEAU_COLOR.get(niveau, "#888")
    return (
        f'<span style="background:{color};color:#fff;border-radius:99px;'
        f'font-size:.72rem;padding:.2rem .7rem;font-family:\'JetBrains Mono\',monospace;'
        f'font-weight:600;letter-spacing:.05em;">{niveau}</span>'
    )


def render_project_card(doc):
    icon = DOMAINE_ICON.get(doc.get("domaine", ""), "📄")
    with st.expander(f"{icon}  {doc['title']}"):
        st.markdown(
            f"""
            {niveau_badge(doc.get('niveau',''))}
            <br><br>
            <p style="font-size:.96rem;line-height:1.7;color:#1a1209;">{doc['description']}</p>
            {tag_pills(doc['tags'])}
            <br><br>
            <a href="{doc['lien']}" target="_blank">📖 Accéder à la documentation →</a>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 .5rem;">
            <span style="font-size:2.6rem;">🎓</span>
            <h2 style="font-family:'Playfair Display',serif;margin:.3rem 0 0;">EduRAG</h2>
            <p style="font-size:.8rem;opacity:.65;letter-spacing:.08em;text-transform:uppercase;">
                Moteur de recommandation<br>pédagogique
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("#### 🔍 À propos")
    st.markdown(
        """
        <p style="font-size:.88rem;line-height:1.65;">
        Ce système utilise la <strong>Recherche Augmentée par Génération (RAG)</strong>
        pour recommander des projets académiques adaptés à votre profil.
        </p>
        <p style="font-size:.88rem;line-height:1.65;">
        <strong>Comment ça fonctionne :</strong><br>
        1️⃣ Votre question est encodée en vecteur<br>
        2️⃣ FAISS trouve les projets les plus proches<br>
        3️⃣ Mistral génère une recommandation ciblée
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("#### 🏷️ Domaines disponibles")
    domains = sorted({doc.get("domaine", "") for doc in documents if doc.get("domaine")})
    for d in domains:
        icon = DOMAINE_ICON.get(d, "📄")
        count = sum(1 for doc in documents if doc.get("domaine") == d)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:.25rem 0;font-size:.88rem;">'
            f'<span>{icon} {d}</span>'
            f'<span style="opacity:.5;">{count} projet{"s" if count>1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown(
        '<p style="font-size:.75rem;opacity:.45;text-align:center;">'
        'Modèle : mistral-small-latest<br>'
        'Embeddings : MiniLM-L12 multilingue<br>'
        'Index : FAISS (L2)</p>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        """
        <div class="edu-header">
            <span class="badge">Moteur RAG · v2.0</span>
            <h1>Trouvez votre projet idéal</h1>
            <p>Décrivez vos intérêts ou compétences et notre moteur pédagogique
            vous proposera les projets les mieux adaptés à votre parcours.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats bar
    domains_count = len({doc.get("domaine") for doc in documents})
    advanced = sum(1 for d in documents if d.get("niveau") == "Avancé")
    st.markdown(
        f"""
        <div class="stats-bar">
            <div class="stat-item">
                <span class="stat-num">{len(documents)}</span>
                <span class="stat-lbl">Projets</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">{domains_count}</span>
                <span class="stat-lbl">Domaines</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">{advanced}</span>
                <span class="stat-lbl">Niveau avancé</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">3</span>
                <span class="stat-lbl">Suggestions / requête</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    index, model = build_faiss_index()

    # Search
    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    with col1:
        query = st.text_input(
            "Décrivez votre intérêt ou votre domaine d'étude",
            placeholder="Ex : projet NLP pour analyser les sentiments sur les réseaux sociaux…",
            label_visibility="visible",
        )
    with col2:
        search_clicked = st.button("🔍 Rechercher", use_container_width=True)

    # Example queries
    st.markdown(
        '<p style="font-size:.82rem;color:#5c4827;margin-top:.2rem;">💡 Exemples : '
        '<em>détection d\'anomalies</em> · '
        '<em>projet IoT environnement</em> · '
        '<em>NLP pour l\'éducation</em> · '
        '<em>deep learning médical</em>'
        '</p>',
        unsafe_allow_html=True,
    )

    if search_clicked:
        if query and query.strip():
            with st.spinner("Analyse sémantique en cours…"):
                retrieved_docs = retrieve_docs(query, index, model)
                response = generate_response(query, retrieved_docs)

            st.markdown(
                '<div class="section-title"><span>🎯</span>Recommandation personnalisée</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="answer-card">{response}</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title"><span>📚</span>Projets sélectionnés par similarité</div>',
                unsafe_allow_html=True,
            )
            for doc in retrieved_docs:
                render_project_card(doc)
        else:
            st.warning("Veuillez saisir une requête avant de lancer la recherche.")

    # All projects
    st.markdown(
        '<div class="section-title"><span>🗂️</span>Catalogue complet des projets</div>',
        unsafe_allow_html=True,
    )

    # Filter by domain
    all_domains = ["Tous"] + sorted({doc.get("domaine", "") for doc in documents if doc.get("domaine")})
    selected_domain = st.selectbox("Filtrer par domaine", all_domains, label_visibility="collapsed")

    filtered = documents if selected_domain == "Tous" else [
        d for d in documents if d.get("domaine") == selected_domain
    ]

    cols = st.columns(2)
    for i, doc in enumerate(filtered):
        with cols[i % 2]:
            render_project_card(doc)


if __name__ == "__main__":
    main()