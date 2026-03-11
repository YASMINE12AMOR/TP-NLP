import streamlit as st
from transformers import CamembertTokenizer, CamembertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="FinClass · Classificateur Comptable",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --yellow:      #F5C400;
    --yellow-dark: #C9A000;
    --yellow-pale: #FFFBE8;
    --black:       #0D0D0D;
    --ink:         #1A1A1A;
    --ink-2:       #3A3A3A;
    --white:       #FFFFFF;
    --bg:          #F8F8F6;
    --border:      #E2E2DC;
    --muted:       #888880;
    --green:       #1A6B3A;
    --green-light: #E6F5ED;
    --blue:        #0A3566;
    --blue-light:  #E6EEF7;
    --red:         #8C1A1A;
    --red-light:   #FAE6E6;
    --radius:      3px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Barlow', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--black) !important;
    border-right: 5px solid var(--yellow) !important;
}
[data-testid="stSidebar"] * {
    color: #CCCCCC !important;
    font-family: 'Barlow', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    font-family: 'Barlow Condensed', sans-serif !important;
    color: var(--white) !important;
    font-weight: 800 !important;
    letter-spacing: .04em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stMarkdown hr {
    border-color: rgba(245,196,0,.25) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(245,196,0,.2) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #CCCCCC !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 1150px !important;
}

/* ── Hero banner ── */
.hero-banner {
    display: flex;
    align-items: stretch;
    min-height: 130px;
    background: var(--white);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    overflow: hidden;
}
.hero-left-bar {
    width: 12px;
    background: var(--yellow);
    flex-shrink: 0;
}
.hero-body {
    flex: 1;
    padding: 1.6rem 2rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    letter-spacing: .3em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .4rem;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--black);
    line-height: 1;
    text-transform: uppercase;
    letter-spacing: .03em;
    margin: 0;
}
.hero-title span { color: var(--yellow-dark); }
.hero-subtitle {
    font-size: .85rem;
    color: var(--muted);
    margin-top: .4rem;
    letter-spacing: .02em;
}
.hero-right-block {
    width: 50px;
    background: var(--yellow);
    flex-shrink: 0;
    position: relative;
}
.hero-right-block::before {
    content: "";
    position: absolute;
    right: 100%;
    top: 0; bottom: 0;
    width: 20px;
    background: var(--yellow);
    opacity: .3;
}

/* ── Stat tiles ── */
.stat-row {
    display: flex;
    gap: .75rem;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
}
.stat-tile {
    flex: 1;
    min-width: 110px;
    background: var(--white);
    border: 1px solid var(--border);
    border-top: 4px solid var(--yellow);
    border-radius: var(--radius);
    padding: 1rem 1.2rem .8rem;
    text-align: center;
    box-shadow: 0 1px 5px rgba(0,0,0,.05);
}
.stat-tile .st-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--black);
    display: block;
    line-height: 1;
}
.stat-tile .st-lbl {
    font-size: .68rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: .3rem;
    display: block;
    font-weight: 500;
}

/* ── Section label ── */
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .8rem;
    display: flex;
    align-items: center;
    gap: .6rem;
}
.section-label::before {
    content: "";
    display: inline-block;
    width: 14px; height: 3px;
    background: var(--yellow);
    border-radius: 2px;
}
.section-label::after {
    content: "";
    flex: 1; height: 1px;
    background: var(--border);
}

/* ── Search card ── */
.search-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-left: 5px solid var(--yellow);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 5px rgba(0,0,0,.05);
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--ink) !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 1rem !important;
    padding: .7rem 1rem !important;
    transition: border-color .2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--yellow) !important;
    box-shadow: 0 0 0 3px rgba(245,196,0,.18) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

/* ── Button ── */
.stButton > button {
    background: var(--yellow) !important;
    color: var(--black) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: .7rem 1.8rem !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 800 !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    box-shadow: 0 2px 10px rgba(245,196,0,.35) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    background: var(--yellow-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(245,196,0,.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result panels ── */
.result-box {
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
    border-left-width: 5px;
    border-left-style: solid;
}
.result-actif  { background: var(--green-light); border-left-color: var(--green); border: 1px solid #b0d9bf; border-left: 5px solid var(--green); }
.result-passif { background: var(--blue-light);  border-left-color: var(--blue);  border: 1px solid #aec4dc; border-left: 5px solid var(--blue); }
.result-notfound { background: var(--red-light); border: 1px solid #d9b0b0; border-left: 5px solid var(--red); }

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    padding: .25rem .8rem;
    border-radius: 2px;
    margin-bottom: .8rem;
    margin-right: .4rem;
}
.badge-actif    { background: var(--green); color: #fff; }
.badge-passif   { background: var(--blue);  color: #fff; }
.badge-yellow   { background: var(--yellow); color: var(--black); }
.badge-gray     { background: var(--ink-2);  color: #fff; }
.badge-red      { background: var(--red);    color: #fff; }

.result-term {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--black);
    text-transform: uppercase;
    letter-spacing: .04em;
    margin: .3rem 0 .6rem;
    line-height: 1;
}
.result-definition {
    font-size: .92rem;
    color: var(--ink-2);
    line-height: 1.7;
    border-top: 1px solid rgba(0,0,0,.08);
    padding-top: .7rem;
    margin-top: .5rem;
}

/* Similarity bar */
.sim-row { display: flex; align-items: center; gap: .8rem; margin-top: .9rem; }
.sim-bar-bg { flex: 1; height: 6px; background: rgba(0,0,0,.1); border-radius: 99px; overflow: hidden; }
.sim-bar-fill { height: 100%; border-radius: 99px; background: var(--yellow); }
.sim-label {
    font-family: 'DM Mono', monospace; font-size: .75rem;
    color: var(--muted); white-space: nowrap;
}
.sim-value {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1rem;
    font-weight: 800; color: var(--yellow-dark); white-space: nowrap;
}

/* ── DB catalogue ── */
.cat-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.2rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--black);
    border-bottom: 3px solid var(--yellow);
    padding-bottom: .35rem;
    margin-bottom: .9rem;
}
.db-row {
    padding: .65rem 0;
    border-bottom: 1px solid var(--border);
}
.db-row:last-child { border-bottom: none; }
.db-term {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--black);
    margin-bottom: .2rem;
}
.db-def { font-size: .84rem; color: var(--ink-2); line-height: 1.6; }
.db-chips { margin-top: .3rem; }
.ex-chip {
    display: inline-block;
    background: var(--yellow-pale);
    border: 1px solid var(--yellow);
    border-radius: 2px;
    padding: .08rem .5rem;
    margin: .1rem .15rem .1rem 0;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    color: var(--ink-2);
    font-weight: 500;
}

/* ── Misc ── */
.stAlert { border-radius: var(--radius) !important; }
.stSpinner > div { border-top-color: var(--yellow) !important; }
[data-testid="stCheckbox"] label {
    font-family: 'Barlow', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--yellow); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
comptabilite_db = {
    "actif": {
        "immobilisations corporelles": {
            "definition": "Biens physiques durables utilisés par l'entreprise (ex. : bâtiments, machines, véhicules).",
            "exemples": ["bâtiment", "machine", "véhicule", "ordinateur", "terrain", "matériel informatique"]
        },
        "immobilisations incorporelles": {
            "definition": "Actifs non physiques mais identifiables (ex. : brevets, logiciels, marques).",
            "exemples": ["brevet", "licence", "marque", "fonds de commerce", "logiciel"]
        },
        "stocks": {
            "definition": "Biens ou services détenus par l'entreprise pour la vente ou la production.",
            "exemples": ["matières premières", "produits finis", "marchandises", "stock"]
        },
        "créances": {
            "definition": "Montants dus à l'entreprise par des tiers (clients, État, etc.).",
            "exemples": ["créance client", "TVA à récupérer", "prêt accordé", "facture impayée"]
        },
        "trésorerie": {
            "definition": "Liquidités disponibles (argent en caisse, comptes bancaires).",
            "exemples": ["caisse", "compte bancaire", "placement court terme", "espèces"]
        }
    },
    "passif": {
        "capitaux propres": {
            "definition": "Ressources financières apportées par les associés ou générées par l'entreprise (bénéfices).",
            "exemples": ["capital social", "réserves", "résultat net", "bénéfices non distribués"]
        },
        "dettes financières": {
            "definition": "Emprunts ou dettes contractés par l'entreprise (ex. : emprunts bancaires, obligations).",
            "exemples": ["emprunt bancaire", "obligation", "crédit-bail", "dette financière"]
        },
        "dettes fournisseurs": {
            "definition": "Montants dus aux fournisseurs pour des biens ou services reçus mais non encore payés.",
            "exemples": ["dette fournisseur", "facture non réglée", "dette commerciale"]
        },
        "dettes fiscales et sociales": {
            "definition": "Montants dus à l'État ou aux organismes sociaux (ex. : impôts, cotisations sociales).",
            "exemples": ["TVA due", "impôt sur les sociétés", "cotisations sociales", "dette fiscale"]
        },
        "provisions": {
            "definition": "Montants mis de côté pour couvrir des risques ou charges futurs (ex. : litiges, garanties).",
            "exemples": ["provision pour litige", "provision pour garantie", "provision pour risques"]
        }
    }
}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    return tokenizer, model

tokenizer, model = load_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def classer_exact(mot, db):
    mot = mot.lower()
    for cat, termes in db.items():
        for terme, d in termes.items():
            if mot in d["exemples"] or mot == terme:
                return {"type": cat, "terme": terme, "definition": d["definition"]}
    return None

def classer_similaire(mot, db, seuil=0.65):
    emb = get_embedding(mot)
    best = []
    for cat, termes in db.items():
        for terme, d in termes.items():
            for ex in d["exemples"]:
                sim = cosine_similarity([emb], [get_embedding(ex)])[0][0]
                best.append((sim, cat, terme, d["definition"]))
    best.sort(reverse=True, key=lambda x: x[0])
    if best and best[0][0] > seuil:
        return {"type": best[0][1], "terme": best[0][2], "definition": best[0][3], "similarity": best[0][0]}
    return None

# ── Counts ────────────────────────────────────────────────────────────────────
total_actif  = sum(len(v["exemples"]) for v in comptabilite_db["actif"].values())
total_passif = sum(len(v["exemples"]) for v in comptabilite_db["passif"].values())
total_cat    = len(comptabilite_db["actif"]) + len(comptabilite_db["passif"])

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0 1rem;border-bottom:1px solid rgba(245,196,0,.25);margin-bottom:1rem;">
        <div style="font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.3em;
                    text-transform:uppercase;color:#F5C400;margin-bottom:.5rem;">Financial Tool</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.9rem;font-weight:900;
                    color:#FFFFFF;line-height:1;text-transform:uppercase;letter-spacing:.04em;">
            FIN<span style="color:#F5C400;">CLASS</span>
        </div>
        <div style="font-size:.8rem;color:#888;margin-top:.4rem;">Classificateur Comptable</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Mode d'emploi")
    st.markdown("""
    <div style="font-size:.84rem;line-height:1.75;color:#AAAAAA;">
    1. Saisissez un terme comptable.<br>
    2. Cliquez sur <strong style="color:#F5C400;">ANALYSER</strong>.<br>
    3. L'outil effectue d'abord une recherche exacte, puis une <em>analyse sémantique</em> via CamemBERT.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Structure du bilan")
    with st.expander("📈 Actif — 5 catégories"):
        for t in comptabilite_db["actif"]:
            st.markdown(f"· {t.title()}")
    with st.expander("📉 Passif — 5 catégories"):
        for t in comptabilite_db["passif"]:
            st.markdown(f"· {t.title()}")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:.7rem;color:rgba(255,255,255,.25);text-align:center;line-height:1.7;">'
        'CamemBERT-base · Seuil : 0.65<br>Bilan Comptable</p>',
        unsafe_allow_html=True,
    )

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-left-bar"></div>
    <div class="hero-body">
        <div class="hero-tag">PROJET · FINANCIAL AUTOMATION</div>
        <div class="hero-title">Classificateur de termes <span>Comptables</span></div>
        <div class="hero-subtitle">
            Analyse sémantique · CamemBERT · {total_actif + total_passif} termes indexés · {total_cat} catégories
        </div>
    </div>
    <div class="hero-right-block"></div>
</div>
""", unsafe_allow_html=True)

# ── STAT TILES ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
    <div class="stat-tile">
        <span class="st-num">{total_actif + total_passif}</span>
        <span class="st-lbl">Termes indexés</span>
    </div>
    <div class="stat-tile">
        <span class="st-num">{len(comptabilite_db["actif"])}</span>
        <span class="st-lbl">Catégories Actif</span>
    </div>
    <div class="stat-tile">
        <span class="st-num">{len(comptabilite_db["passif"])}</span>
        <span class="st-lbl">Catégories Passif</span>
    </div>
    <div class="stat-tile">
        <span class="st-num">65%</span>
        <span class="st-lbl">Seuil sémantique</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SEARCH ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Analyse d\'un terme</div>', unsafe_allow_html=True)
st.markdown('<div class="search-card">', unsafe_allow_html=True)

col_input, col_btn = st.columns([4, 1], vertical_alignment="bottom")
with col_input:
    mot_utilisateur = st.text_input(
        "Terme", placeholder='Ex : "bâtiment", "dette fournisseur", "capital social"…',
        key="input_mot", label_visibility="collapsed",
    )
with col_btn:
    analyser = st.button("Analyser", key="btn_analyser", use_container_width=True)

st.markdown(
    '<p style="font-size:.78rem;color:var(--muted);margin-top:.4rem;">'
    '💡 <em>terrain · emprunt bancaire · TVA due · résultat net · marchandises</em></p>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────────────────────────────
if analyser:
    if mot_utilisateur and mot_utilisateur.strip():
        with st.spinner("Analyse en cours…"):
            exact = classer_exact(mot_utilisateur, comptabilite_db)

        if exact:
            t = exact["type"]
            dcls = "result-actif" if t == "actif" else "result-passif"
            bcls = "badge-actif" if t == "actif" else "badge-passif"
            icon = "📈" if t == "actif" else "📉"
            st.markdown(f"""
            <div class="result-box {dcls}">
                <span class="result-badge {bcls}">{icon} {t.upper()}</span>
                <span class="result-badge badge-yellow">✓ Exacte</span>
                <div class="result-term">{exact['terme']}</div>
                <div class="result-definition">{exact['definition']}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            with st.spinner("Recherche par similarité sémantique…"):
                sim = classer_similaire(mot_utilisateur, comptabilite_db)

            if sim:
                t = sim["type"]
                dcls = "result-actif" if t == "actif" else "result-passif"
                bcls = "badge-actif" if t == "actif" else "badge-passif"
                icon = "📈" if t == "actif" else "📉"
                pct  = sim["similarity"] * 100
                st.markdown(f"""
                <div class="result-box {dcls}">
                    <span class="result-badge {bcls}">{icon} {t.upper()}</span>
                    <span class="result-badge badge-gray">≈ Similarité sémantique</span>
                    <div class="result-term">{sim['terme']}</div>
                    <div class="result-definition">{sim['definition']}</div>
                    <div class="sim-row">
                        <span class="sim-label">Score de similarité</span>
                        <div class="sim-bar-bg">
                            <div class="sim-bar-fill" style="width:{pct:.0f}%;"></div>
                        </div>
                        <span class="sim-value">{pct:.1f} %</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-notfound">
                    <span class="result-badge badge-red">✕ Non reconnu</span>
                    <div style="font-size:.95rem;color:#5c1a1a;line-height:1.7;margin-top:.5rem;">
                        Le terme <strong>« {mot_utilisateur} »</strong> n'a pas été trouvé dans la base
                        et aucune correspondance sémantique suffisante n'a été détectée (seuil : 65 %).
                        Vérifiez l'orthographe ou consultez le catalogue ci-dessous.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Veuillez saisir un terme comptable avant de lancer l'analyse.")

# ── CATALOGUE ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
show_db = st.checkbox("📂  Afficher le catalogue complet des termes", value=False)

if show_db:
    st.markdown('<div class="section-label">Référentiel comptable</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for col, (categorie, termes) in zip([col1, col2], comptabilite_db.items()):
        with col:
            icon = "📈" if categorie == "actif" else "📉"
            st.markdown(f'<div class="cat-header">{icon} {categorie.upper()}</div>', unsafe_allow_html=True)
            for terme, details in termes.items():
                chips = "".join(f'<span class="ex-chip">{e}</span>' for e in details["exemples"])
                st.markdown(f"""
                <div class="db-row">
                    <div class="db-term">{terme}</div>
                    <div class="db-def">{details['definition']}</div>
                    <div class="db-chips">{chips}</div>
                </div>
                """, unsafe_allow_html=True)