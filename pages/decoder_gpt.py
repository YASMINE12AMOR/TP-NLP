import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mistral GPT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens ─────────────────────────────────────────────────────────────
NAVY      = "#0D2A4E"
NAVY_DARK = "#091C35"
NAVY_MID  = "#1A3A5C"
ICE_BLUE  = "#C8DAEA"
WHITE     = "#FFFFFF"
ACCENT    = "#4A90D9"
LIGHT_BG  = "#F4F7FB"
TEXT_MUTED= "#7A8FA6"

st.markdown(f"""
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800&family=Lato:wght@300;400;700&display=swap');

/* ── Global reset ─────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Lato', sans-serif;
    background-color: {LIGHT_BG};
    color: {NAVY_DARK};
}}

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(175deg, {NAVY_DARK} 0%, {NAVY_MID} 100%);
    border-right: none;
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: {ICE_BLUE} !important;
    font-family: 'Lato', sans-serif;
    font-weight: 400;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] {{
    color: {WHITE} !important;
    font-weight: 700;
}}
section[data-testid="stSidebar"] hr {{
    border-color: {NAVY_MID} !important;
    opacity: 0.5;
}}
/* Sidebar header decoration */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {WHITE} !important;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

/* ── Hero banner ─────────────────────────────── */
.hero-banner {{
    background: linear-gradient(110deg, {NAVY_DARK} 55%, {NAVY_MID} 100%);
    border-radius: 4px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(9,28,53,0.18);
}}
.hero-banner::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: {ACCENT};
    opacity: 0.10;
    transform: rotate(45deg);
    border-radius: 4px;
}}
.hero-banner::after {{
    content: '';
    position: absolute;
    bottom: -60px; right: 100px;
    width: 160px; height: 160px;
    background: {ICE_BLUE};
    opacity: 0.07;
    transform: rotate(45deg);
    border-radius: 4px;
}}
.hero-banner .company-tag {{
    font-family: 'Lato', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {ICE_BLUE};
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.hero-banner .company-tag::before {{
    content: '';
    display: inline-block;
    width: 18px; height: 18px;
    background: {ACCENT};
    transform: rotate(45deg);
    opacity: 0.85;
    flex-shrink: 0;
}}
.hero-banner h1 {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800;
    font-size: 3rem !important;
    line-height: 1.0 !important;
    color: {WHITE} !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0 0 0.8rem 0 !important;
    padding: 0 !important;
}}
.hero-banner p {{
    color: {ICE_BLUE};
    font-size: 1.0rem;
    font-weight: 300;
    line-height: 1.6;
    max-width: 500px;
    margin: 0;
}}
.hero-geo-dots {{
    position: absolute;
    bottom: 20px; right: 220px;
    display: grid;
    grid-template-columns: repeat(5,1fr);
    gap: 5px;
}}
.hero-geo-dots span {{
    display: block;
    width: 5px; height: 5px;
    background: {ICE_BLUE};
    border-radius: 50%;
    opacity: 0.35;
}}

/* ── Diamond chevron decoration ────────────────── */
.chevrons {{
    position: absolute;
    top: 50%; right: 30px;
    transform: translateY(-50%);
    display: flex; gap: 6px;
    align-items: center;
}}
.chevron-diamond {{
    width: 0; height: 0;
    border: 12px solid transparent;
    border-left: 16px solid {ACCENT};
    opacity: 0.6;
}}

/* ── Section headers ─────────────────────────── */
.section-label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.3rem;
}}
.section-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: {NAVY_DARK};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-title::before {{
    content: '';
    display: inline-block;
    width: 10px; height: 10px;
    background: {ACCENT};
    transform: rotate(45deg);
    flex-shrink: 0;
}}

/* ── Cards ───────────────────────────────────── */
.card-desc {{
    background: {WHITE};
    border-left: 4px solid {ACCENT};
    border-radius: 0 4px 4px 0;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.5rem;
    font-size: 0.92rem;
    color: #3A506B;
    line-height: 1.7;
    box-shadow: 0 2px 12px rgba(9,28,53,0.06);
}}

/* ── Example buttons ─────────────────────────── */
.stButton > button {{
    background: {WHITE} !important;
    color: {NAVY_DARK} !important;
    border: 1.5px solid {ICE_BLUE} !important;
    border-radius: 3px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    padding: 0.55rem 1rem !important;
    transition: all 0.18s ease !important;
    text-align: left !important;
}}
.stButton > button:hover {{
    background: {NAVY_DARK} !important;
    color: {WHITE} !important;
    border-color: {NAVY_DARK} !important;
    transform: translateX(3px);
}}

/* ── Primary CTA button ──────────────────────── */
.stButton > [kind="primary"] {{
    background: {NAVY_DARK} !important;
    color: {WHITE} !important;
    border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.7rem 2rem !important;
    border-radius: 3px !important;
}}
.stButton > [kind="primary"]:hover {{
    background: {ACCENT} !important;
    transform: none;
}}

/* ── Text areas ──────────────────────────────── */
textarea, .stTextArea textarea {{
    font-family: 'Lato', sans-serif !important;
    font-size: 0.92rem !important;
    border: 1.5px solid {ICE_BLUE} !important;
    border-radius: 3px !important;
    background: {WHITE} !important;
    color: {NAVY_DARK} !important;
}}
textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(74,144,217,0.15) !important;
}}

/* ── Labels ──────────────────────────────────── */
label, .stTextArea label {{
    font-family: 'Lato', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_MUTED} !important;
}}

/* ── Result box ──────────────────────────────── */
.result-header {{
    background: {NAVY_DARK};
    color: {WHITE};
    padding: 0.6rem 1.2rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-radius: 3px 3px 0 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.result-body {{
    background: {WHITE};
    border: 1.5px solid {ICE_BLUE};
    border-top: none;
    border-radius: 0 0 3px 3px;
    padding: 1.5rem;
    box-shadow: 0 4px 16px rgba(9,28,53,0.07);
}}

/* ── Divider ─────────────────────────────────── */
hr {{
    border: none;
    border-top: 1px solid {ICE_BLUE} !important;
    margin: 1.5rem 0;
}}

/* ── Warning / error ─────────────────────────── */
.stAlert {{
    border-radius: 3px !important;
    border-left-width: 4px !important;
}}

/* ── Footer ──────────────────────────────────── */
.footer-bar {{
    background: {NAVY_DARK};
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.footer-bar .footer-contact {{
    color: {ICE_BLUE};
    font-size: 0.82rem;
    display: flex;
    gap: 1.5rem;
}}
.footer-bar .footer-contact span::before {{
    margin-right: 6px;
    opacity: 0.7;
}}
.footer-tip {{
    color: {WHITE};
    font-size: 0.8rem;
    font-family: 'Lato', sans-serif;
    font-weight: 300;
    letter-spacing: 0.02em;
}}

/* ── Spinner ─────────────────────────────────── */
.stSpinner > div {{
    border-top-color: {ACCENT} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Env & client ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "mistral-tiny"

if not api_key:
    st.error("⚠️ **MISTRAL_API_KEY introuvable** – Ajoutez-la dans le fichier `.env` à la racine du projet.")
    st.stop()

@st.cache_resource(show_spinner="Initialisation du modèle...")
def get_client():
    return Mistral(api_key=api_key)

client = get_client()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem 0;'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;'>
            <div style='width:28px;height:28px;background:#4A90D9;transform:rotate(45deg);border-radius:3px;flex-shrink:0;'></div>
            <span style='font-family:"Barlow Condensed",sans-serif;font-weight:800;font-size:1.3rem;
                         letter-spacing:0.1em;text-transform:uppercase;color:white;'>MISTRAL AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="color:#C8DAEA!important;">NAVIGATION</div>', unsafe_allow_html=True)

    task = st.radio(
        "Choisissez une tâche :",
        ["💬 Chat / Questions-Réponses", "📄 Résumé de texte", "💻 Génération de code"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='margin-top:auto;padding-top:2rem;'>
        <div style='font-family:"Lato",sans-serif;font-size:0.75rem;color:#7A8FA6;letter-spacing:0.05em;line-height:1.8;'>
            <div>🌐 www.reallygreatsite.com</div>
            <div>✉ hello@reallygreatsite.com</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
task_labels = {
    "💬 Chat / Questions-Réponses": ("ASSISTANT", "Questions & Réponses"),
    "📄 Résumé de texte":           ("ANALYSE",   "Résumé de Texte"),
    "💻 Génération de code":         ("STUDIO",    "Génération de Code"),
}
tag, title = task_labels[task]

st.markdown(f"""
<div class="hero-banner">
    <div class="company-tag">{tag}</div>
    <h1>{title}</h1>
    <p>Sélectionnez un exemple ci-dessous ou saisissez votre propre requête pour obtenir une réponse claire et structurée.</p>
    <div class="chevrons">
        <div class="chevron-diamond"></div>
        <div class="chevron-diamond" style="opacity:0.4"></div>
        <div class="chevron-diamond" style="opacity:0.2"></div>
    </div>
    <div class="hero-geo-dots">
        {'<span></span>' * 15}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Task content ──────────────────────────────────────────────────────────────
examples_by_task = {
    "💬 Chat / Questions-Réponses": {
        "description": "Posez des questions ouvertes sur des sujets variés — IA, programmation, conseils. Mistral répondra de manière claire et pédagogique.",
        "examples": [
            "Explique simplement la différence entre NLP, NLU et NLG avec un exemple pour chacun.",
            "Je débute en IA. Quels sont les prérequis à apprendre en 30 jours ? Fais un plan hebdomadaire.",
            "Donne-moi 5 idées de projet NLP niveau débutant à intermédiaire, avec une estimation de difficulté."
        ]
    },
    "📄 Résumé de texte": {
        "description": "Collez un texte long et demandez un résumé structuré. Précisez le format souhaité : points clés, résumé exécutif, etc.",
        "examples": [
            "Résume ce texte en 5 points clés, puis ajoute une conclusion en 2 lignes.",
            "Fais un résumé exécutif (max 120 mots) avec : objectif, méthode, résultats, limites.",
            "Condense ce texte pour une présentation orale de 1 minute, en mettant l'accent sur les impacts."
        ]
    },
    "💻 Génération de code": {
        "description": "Décrivez une fonctionnalité ou un algorithme en langage naturel, et Mistral génère du code Python exécutable et bien commenté.",
        "examples": [
            "Écris une fonction Python qui nettoie un texte (minuscules, ponctuation, stopwords) avec un exemple d'utilisation.",
            "Génère une classe PyTorch pour un modèle LSTM de classification de texte, avec une méthode forward et un exemple d'entraînement.",
            "Crée un script Streamlit minimal avec un champ texte, un bouton pour analyser le sentiment, et un affichage visuel du résultat."
        ]
    }
}

data = examples_by_task[task]

# Description card
st.markdown(f'<div class="card-desc">{data["description"]}</div>', unsafe_allow_html=True)

# Examples
st.markdown('<div class="section-title">Exemples rapides</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, example in enumerate(data["examples"]):
    with cols[i % 3]:
        short = example[:55] + "…" if len(example) > 55 else example
        if st.button(f"→  {short}", key=f"ex_{i}", use_container_width=True):
            if task == "💬 Chat / Questions-Réponses":
                st.session_state["user_input"] = example
            elif task == "📄 Résumé de texte":
                st.session_state["sum_instruction"] = example
            elif task == "💻 Génération de code":
                st.session_state["code_request"] = example

st.markdown("---")

# ── Input zone ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Votre requête</div>', unsafe_allow_html=True)

prompt = ""

if task == "💬 Chat / Questions-Réponses":
    user_input = st.text_area(
        "QUESTION",
        key="user_input",
        height=140,
        placeholder="Exemple : Comment démarrer un projet NLP avec Python ?"
    )
    prompt = user_input.strip() if user_input else ""

elif task == "📄 Résumé de texte":
    col1, col2 = st.columns([1, 1])
    with col1:
        sum_instruction = st.text_area(
            "INSTRUCTION (optionnel)",
            key="sum_instruction",
            height=100,
            placeholder="Ex : Résume en 3 points clés pour un public non technique."
        )
    with col2:
        text_to_summarize = st.text_area(
            "TEXTE À RÉSUMER",
            key="sum_text",
            height=200,
            placeholder="Collez ici le texte que vous souhaitez résumer..."
        )
    if text_to_summarize.strip():
        instruction = sum_instruction.strip() if sum_instruction else "Résume ce texte de manière claire et concise :"
        prompt = f"{instruction}\n\nTexte à résumer:\n{text_to_summarize.strip()}"

elif task == "💻 Génération de code":
    code_request = st.text_area(
        "DESCRIPTION DU CODE",
        key="code_request",
        height=180,
        placeholder='Ex : "Crée une fonction Python qui calcule la fréquence des mots dans un texte, en ignorant les stopwords."'
    )
    prompt = code_request.strip() if code_request else ""

# ── Submit ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if prompt:
    if st.button("ENVOYER LA REQUÊTE  ▶", type="primary", use_container_width=False):
        with st.spinner("Analyse en cours…"):
            try:
                response = client.chat.complete(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Tu es un assistant expert en NLP et en développement. "
                                "Réponds de manière claire, structurée et actionnable. "
                                "Pour le code, fournis du Python exécutable et bien commenté. "
                                "Adapte ton style à la tâche demandée."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result = response.choices[0].message.content

                st.markdown("---")
                st.markdown("""
                <div class="result-header">
                    <div style='width:8px;height:8px;background:#4A90D9;transform:rotate(45deg);'></div>
                    RÉSULTAT
                </div>
                <div class="result-body">
                """, unsafe_allow_html=True)

                if task == "💻 Génération de code":
                    st.code(result, language="python")
                else:
                    st.markdown(result)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {str(e)}")
else:
    st.warning("Veuillez saisir une requête ou sélectionner un exemple ci-dessus.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <div class="footer-tip">
        Soyez précis dans vos requêtes · Précisez le format souhaité · Décrivez clairement les fonctionnalités
    </div>
    <div class="footer-contact">
        <span>🌐 www.reallygreatsite.com</span>
        <span>✉ hello@reallygreatsite.com</span>
    </div>
</div>
""", unsafe_allow_html=True)