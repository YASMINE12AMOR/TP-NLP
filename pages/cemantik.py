import streamlit as st
from transformers import CamembertTokenizer, CamembertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CEMANTIK · Le Jeu des Mots",
    page_icon="🕹️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323:wght@400&display=swap');

:root {
    --bg:       #0d0b1e;
    --surface:  #1a1535;
    --panel:    #231d47;
    --border:   #3d2f7a;
    --pink:     #ff2d78;
    --yellow:   #ffd54f;
    --teal:     #00e5c8;
    --purple:   #b06aff;
    --white:    #ffffff;
    --muted:    #7a6aaa;
    --cloud:    #c084fc;
    --glow-p:   rgba(255,45,120,0.5);
    --glow-t:   rgba(0,229,200,0.4);
    --glow-y:   rgba(255,213,79,0.5);
}

/* ── Pixel font helper ─────────────────── */
* { image-rendering: pixelated; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--white) !important;
    font-family: 'VT323', monospace !important;
}

/* ── Animated scanline + pixel sky background ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background:
        /* pixel clouds */
        radial-gradient(ellipse 60px 30px at 12% 18%, rgba(192,132,252,.22) 0%, transparent 70%),
        radial-gradient(ellipse 80px 25px at 85% 14%, rgba(192,132,252,.18) 0%, transparent 70%),
        radial-gradient(ellipse 50px 20px at 60% 22%, rgba(192,132,252,.14) 0%, transparent 70%),
        /* pixel stars */
        radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 20%, rgba(255,255,255,.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 60%, rgba(255,255,255,.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 80%, rgba(255,255,255,.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 55%, rgba(255,255,255,.6) 0%, transparent 100%),
        /* sunset glow at center-top */
        radial-gradient(ellipse 300px 200px at 50% 5%, rgba(255,213,79,.12) 0%, transparent 70%),
        /* deep purple-blue sky */
        linear-gradient(180deg, #0d0b1e 0%, #1a0d3a 60%, #0d0b1e 100%);
    pointer-events: none;
    z-index: 0;
}

/* Scanlines overlay */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0,0,0,.08) 3px,
        rgba(0,0,0,.08) 4px
    );
    pointer-events: none;
    z-index: 1;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    position: relative;
    z-index: 2;
}

/* ── Pixel sun ── */
.pixel-sun {
    width: 80px; height: 80px;
    background: var(--yellow);
    border-radius: 50%;
    margin: 0 auto 0;
    box-shadow:
        0 0 0 6px #e6a817,
        0 0 30px 10px rgba(255,213,79,.5),
        0 0 60px 20px rgba(255,150,0,.3);
    position: relative;
    /* pixel dither bottom stripe */
}
.pixel-sun::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 30%;
    background: repeating-linear-gradient(
        90deg,
        #c8860a 0px, #c8860a 8px,
        var(--yellow) 8px, var(--yellow) 16px
    );
    border-radius: 0 0 50% 50%;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.hero-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 2.6rem;
    color: var(--white);
    letter-spacing: .05em;
    line-height: 1.2;
    /* pixel shadow / glitch effect */
    text-shadow:
        4px 4px 0px var(--pink),
        -2px -2px 0px var(--teal),
        0 0 30px rgba(255,45,120,.6);
    margin: 1rem 0 0.4rem;
    /* chunky pixel outline */
    -webkit-text-stroke: 0px;
}
.hero-sub {
    font-family: 'VT323', monospace;
    font-size: 1.3rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--cloud);
    margin-top: .2rem;
}
.hero-handle {
    font-family: 'VT323', monospace;
    font-size: 1.1rem;
    color: var(--muted);
    letter-spacing: .15em;
    margin-top: .1rem;
}
.hero-divider {
    width: 220px;
    height: 4px;
    margin: 1rem auto;
    background: repeating-linear-gradient(
        90deg,
        var(--pink) 0px, var(--pink) 12px,
        transparent 12px, transparent 16px,
        var(--yellow) 16px, var(--yellow) 28px,
        transparent 28px, transparent 32px,
        var(--teal) 32px, var(--teal) 44px,
        transparent 44px, transparent 48px
    );
    border: none;
    image-rendering: pixelated;
}

/* ── Pixel clouds decoration ── */
.pixel-clouds {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    position: relative;
    height: 40px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}
.pixel-cloud {
    width: 64px; height: 32px;
    position: relative;
    flex-shrink: 0;
}

/* ── Pixel ground / city strip ── */
.city-strip {
    width: 100%;
    height: 8px;
    background: repeating-linear-gradient(
        90deg,
        #ff2d78 0px, #ff2d78 8px,
        #b06aff 8px, #b06aff 16px,
        #00e5c8 16px, #00e5c8 24px,
        #ffd54f 24px, #ffd54f 32px
    );
    margin: 0.8rem 0;
    image-rendering: pixelated;
}

/* ── Game cards ── */
.game-card {
    background: var(--surface);
    border: 3px solid var(--border);
    border-radius: 0px;
    /* pixel border via box-shadow */
    box-shadow:
        0 0 0 1px var(--bg),
        4px 4px 0px var(--pink),
        0 0 24px rgba(176,106,255,.2);
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
    position: relative;
    image-rendering: pixelated;
}
.game-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: repeating-linear-gradient(
        90deg,
        var(--pink) 0px, var(--pink) 8px,
        var(--yellow) 8px, var(--yellow) 16px
    );
}

/* ── Player badges ── */
.player-badge {
    display: inline-block;
    font-family: 'Press Start 2P', monospace;
    font-size: .55rem;
    letter-spacing: .08em;
    padding: .4rem .9rem;
    border: 2px solid;
    margin-bottom: 1rem;
    image-rendering: pixelated;
}
.p1 {
    background: rgba(0,229,200,.08);
    border-color: var(--teal);
    color: var(--teal);
    box-shadow: 3px 3px 0 rgba(0,229,200,.3);
}
.p2 {
    background: rgba(255,45,120,.08);
    border-color: var(--pink);
    color: var(--pink);
    box-shadow: 3px 3px 0 rgba(255,45,120,.3);
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--panel) !important;
    border: 2px solid var(--border) !important;
    border-radius: 0px !important;
    color: var(--yellow) !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: .75rem !important;
    padding: .8rem 1rem !important;
    letter-spacing: .05em;
    box-shadow: 3px 3px 0 var(--border) !important;
    transition: border-color .15s, box-shadow .15s;
}
.stTextInput > div > div > input::placeholder {
    color: var(--muted) !important;
    font-size: .65rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    box-shadow: 3px 3px 0 var(--teal) !important;
    outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: var(--pink) !important;
    color: var(--white) !important;
    border: 2px solid #ff6baa !important;
    border-radius: 0px !important;
    padding: .75rem 1.5rem !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: .6rem !important;
    font-weight: 400 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    box-shadow: 4px 4px 0px #8b0040 !important;
    transition: transform .1s, box-shadow .1s !important;
    cursor: pointer !important;
    image-rendering: pixelated;
}
.stButton > button:hover {
    background: var(--yellow) !important;
    color: #0d0b1e !important;
    border-color: #e6a817 !important;
    box-shadow: 4px 4px 0px #8b6200 !important;
}
.stButton > button:active {
    transform: translate(3px, 3px) !important;
    box-shadow: 1px 1px 0px #8b0040 !important;
}

/* ── Attempts dots ── */
.attempts-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: .6rem;
    margin: .5rem 0 1.2rem;
}
.attempt-dot {
    width: 16px; height: 16px;
    display: inline-block;
    image-rendering: pixelated;
}
.dot-active  {
    background: var(--yellow);
    box-shadow: 0 0 8px var(--yellow), 2px 2px 0 #8b6200;
}
.dot-used    {
    background: var(--pink);
    box-shadow: 0 0 8px var(--pink), 2px 2px 0 #8b0040;
}
.dot-empty   {
    background: var(--border);
    box-shadow: 2px 2px 0 #1a1535;
}

/* ── Score bar ── */
.score-bar-bg {
    height: 12px;
    background: var(--panel);
    border: 2px solid var(--border);
    overflow: hidden;
    image-rendering: pixelated;
}
.score-bar-fill {
    height: 100%;
    transition: width .4s steps(10);
    image-rendering: pixelated;
}

/* ── History rows ── */
.hist-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .55rem .9rem;
    margin: .35rem 0;
    background: var(--panel);
    border: 2px solid var(--border);
    box-shadow: 3px 3px 0 var(--bg);
    animation: slideIn .25s steps(4) forwards;
    opacity: 0;
    transform: translateX(-8px);
}
@keyframes slideIn {
    to { opacity: 1; transform: translateX(0); }
}
.hist-word {
    font-family: 'Press Start 2P', monospace;
    font-size: .65rem;
    color: var(--white);
    letter-spacing: .06em;
}
.hist-score-chip {
    font-family: 'Press Start 2P', monospace;
    font-size: .55rem;
    padding: .3rem .7rem;
    border: 2px solid;
    image-rendering: pixelated;
}
.chip-hot  {
    background: rgba(0,229,200,.12);
    color: var(--teal);
    border-color: var(--teal);
    box-shadow: 2px 2px 0 rgba(0,229,200,.3);
}
.chip-warm {
    background: rgba(255,213,79,.12);
    color: var(--yellow);
    border-color: var(--yellow);
    box-shadow: 2px 2px 0 rgba(255,213,79,.3);
}
.chip-cold {
    background: rgba(255,45,120,.1);
    color: var(--pink);
    border-color: var(--pink);
    box-shadow: 2px 2px 0 rgba(255,45,120,.3);
}

/* ── Rules box ── */
.rules-box {
    background: var(--panel);
    border: 2px solid var(--border);
    box-shadow: 4px 4px 0 var(--border);
    padding: 1.1rem 1.4rem;
    margin: .6rem 0 1.2rem;
    font-family: 'VT323', monospace;
    font-size: 1.2rem;
    line-height: 1.8;
    color: #b8aadd;
}
.rules-box strong {
    color: var(--yellow);
    font-family: 'Press Start 2P', monospace;
    font-size: .65rem;
    letter-spacing: .05em;
}

/* ── Section headings ── */
.section-head {
    font-family: 'Press Start 2P', monospace;
    font-size: .55rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--purple);
    margin: 1.2rem 0 .6rem;
    display: flex;
    align-items: center;
    gap: .6rem;
}
.section-head::after {
    content: "";
    flex: 1;
    height: 2px;
    background: repeating-linear-gradient(
        90deg,
        var(--border) 0px, var(--border) 6px,
        transparent 6px, transparent 10px
    );
}

/* ── End screen ── */
.end-screen {
    text-align: center;
    padding: 2rem 1rem;
}
.end-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 1.6rem;
    margin-bottom: .6rem;
    line-height: 1.4;
}
.win-title  {
    color: var(--teal);
    text-shadow: 4px 4px 0 #006b5e, 0 0 20px rgba(0,229,200,.6);
}
.lose-title {
    color: var(--pink);
    text-shadow: 4px 4px 0 #8b0040, 0 0 20px rgba(255,45,120,.6);
}
.end-word {
    font-family: 'Press Start 2P', monospace;
    font-size: 1.2rem;
    letter-spacing: .15em;
    color: var(--yellow);
    text-shadow: 3px 3px 0 #8b6200, 0 0 15px rgba(255,213,79,.5);
    margin: .8rem 0 1.4rem;
}

/* ── Stat box ── */
.stat-label {
    font-family: 'Press Start 2P', monospace;
    font-size: .5rem;
    color: var(--muted);
    letter-spacing: .1em;
    margin-bottom: .2rem;
}
.stat-value-teal   { font-family: 'Press Start 2P', monospace; font-size: 1.4rem; color: var(--teal);   text-shadow: 3px 3px 0 #006b5e; }
.stat-value-gold   { font-family: 'Press Start 2P', monospace; font-size: 1rem;   color: var(--yellow); text-shadow: 2px 2px 0 #8b6200; }
.stat-value-pink   { font-family: 'Press Start 2P', monospace; font-size: 1.4rem; color: var(--pink);   text-shadow: 3px 3px 0 #8b0040; }

/* ── Labels & misc ── */
label, .stTextInput label, .stTextArea label {
    font-family: 'Press Start 2P', monospace !important;
    font-size: .55rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted) !important;
}
.stAlert {
    border-radius: 0px !important;
    font-family: 'VT323', monospace !important;
    font-size: 1.1rem !important;
    border: 2px solid !important;
}
.stSpinner > div { border-top-color: var(--yellow) !important; }

/* ── INSERT COIN blink ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.insert-coin {
    font-family: 'Press Start 2P', monospace;
    font-size: .65rem;
    color: var(--yellow);
    letter-spacing: .12em;
    text-align: center;
    animation: blink 1s steps(1) infinite;
    margin: .5rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    return tokenizer, model

tokenizer, model = load_model()

def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_similarity(word1, word2):
    e1 = get_embedding(word1)
    e2 = get_embedding(word2)
    return float(cosine_similarity([e1], [e2])[0][0])


# ── Session init ──────────────────────────────────────────────────────────────
MAX_TRIES = 5
for key, default in {
    "mot_cible": None,
    "tentatives_restantes": MAX_TRIES,
    "historique": [],
    "jeu_termine": False,
    "gagne": False,
    "dernier_mot_propose": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_to_color(score):
    if score >= 70: return "hot"
    if score >= 40: return "warm"
    return "cold"

def score_to_hint(score):
    if score >= 70: return "🔥 CHAUD !"
    if score >= 40: return "🌡 TIEDE"
    return "❄ FROID"

def bar_color(score):
    if score >= 70: return "repeating-linear-gradient(90deg, #00b89c 0px,#00b89c 8px,#00e5c8 8px,#00e5c8 16px)"
    if score >= 40: return "repeating-linear-gradient(90deg, #c8860a 0px,#c8860a 8px,#ffd54f 8px,#ffd54f 16px)"
    return "repeating-linear-gradient(90deg, #8b0040 0px,#8b0040 8px,#ff2d78 8px,#ff2d78 16px)"

def render_history():
    if not st.session_state.historique:
        return
    st.markdown('<div class="section-head">▶ HISTORIQUE</div>', unsafe_allow_html=True)
    for i, (mot, score) in enumerate(reversed(st.session_state.historique)):
        cls = score_to_color(score)
        hint = score_to_hint(score)
        delay = i * 0.04
        st.markdown(f"""
        <div class="hist-row" style="animation-delay:{delay}s;">
            <span class="hist-word">{mot.upper()}</span>
            <div style="display:flex;align-items:center;gap:.7rem;">
                <div class="score-bar-bg" style="width:100px;">
                    <div class="score-bar-fill" style="width:{score:.0f}%;background:{bar_color(score)};"></div>
                </div>
                <span class="hist-score-chip chip-{cls}">{score:.0f}/100</span>
                <span style="font-family:'VT323',monospace;font-size:1rem;color:{'#00e5c8' if cls=='hot' else '#ffd54f' if cls=='warm' else '#ff2d78'};">{hint}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def attempt_dots():
    used = MAX_TRIES - st.session_state.tentatives_restantes
    html = '<div class="attempts-row">'
    for i in range(MAX_TRIES):
        if i < used:
            html += '<span class="attempt-dot dot-used"></span>'
        elif i == used and not st.session_state.jeu_termine:
            html += '<span class="attempt-dot dot-active"></span>'
        else:
            html += '<span class="attempt-dot dot-empty"></span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── UI ────────────────────────────────────────────────────────────────────────

# Pixel sun + clouds
st.markdown("""
<div style="text-align:center;padding-top:1rem;">
    <div class="pixel-sun"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1 class="hero-title">CEMANTIK</h1>
    <p class="hero-sub">Le jeu des mots · Similarité sémantique</p>
    <p class="hero-handle">@cemantik</p>
    <hr class="hero-divider">
</div>
<div class="city-strip"></div>
""", unsafe_allow_html=True)


# ── PHASE 1 : Player 1 ───────────────────────────────────────────────────────
if not st.session_state.mot_cible and not st.session_state.jeu_termine:

    st.markdown('<div class="insert-coin">— INSERT COIN TO PLAY —</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="rules-box">
        <strong>COMMENT JOUER ?</strong><br>
        🕹 <span style="color:#e8eaf6;">JOUEUR 1</span> entre un mot secret — il sera caché immédiatement.<br>
        🎯 <span style="color:#e8eaf6;">JOUEUR 2</span> dispose de <span style="color:#ffd54f;">5 tentatives</span> pour deviner le mot en proposant des mots proches.<br>
        📊 Un score de <span style="color:#00e5c8;">similarité sémantique</span> (0–100) vous guide à chaque essai.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown('<span class="player-badge p1">[ P1 ] MOT SECRET</span>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--muted);font-family:\'VT323\',monospace;font-size:1.1rem;margin-bottom:1rem;">Entrez le mot cible. Il sera masqué dès validation.</p>', unsafe_allow_html=True)

    mot_cible_input = st.text_input(
        "MOT CIBLE",
        key="input_mot_cible",
        type="password",
        placeholder="> TAPEZ UN MOT...",
        label_visibility="collapsed",
    )

    if st.button("[ LOCK IN ] VALIDER LE MOT", key="btn_valider"):
        if mot_cible_input.strip():
            st.session_state.mot_cible = mot_cible_input.strip().lower()
            st.session_state.tentatives_restantes = MAX_TRIES
            st.session_state.historique = []
            st.session_state.jeu_termine = False
            st.session_state.gagne = False
            st.session_state.dernier_mot_propose = ""
            st.rerun()
        else:
            st.warning("Veuillez entrer un mot avant de valider.")

    st.markdown('</div>', unsafe_allow_html=True)


# ── PHASE 2 : Player 2 ───────────────────────────────────────────────────────
elif st.session_state.mot_cible and not st.session_state.jeu_termine:

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown(
            f'<div class="stat-label">TENTATIVES</div>'
            f'<div class="stat-value-teal">{st.session_state.tentatives_restantes}'
            f'<span style="font-family:\'VT323\',monospace;font-size:1rem;color:var(--muted);"> / {MAX_TRIES}</span></div>',
            unsafe_allow_html=True,
        )
    with col_r:
        best = max((s for _, s in st.session_state.historique), default=0)
        st.markdown(
            f'<div class="stat-label">BEST SCORE</div>'
            f'<div class="stat-value-pink">{best:.0f}'
            f'<span style="font-family:\'VT323\',monospace;font-size:1rem;color:var(--muted);"> / 100</span></div>',
            unsafe_allow_html=True,
        )

    attempt_dots()

    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown('<span class="player-badge p2">[ P2 ] DEVINEZ LE MOT</span>', unsafe_allow_html=True)

    mot_propose_input = st.text_input(
        "PROPOSITION",
        placeholder="> VOTRE MOT...",
        key="input_proposition",
        label_visibility="collapsed",
    )

    if st.button("[ A ] PROPOSER CE MOT", key="btn_proposer"):
        mot_propose = mot_propose_input.strip().lower()
        if mot_propose:
            with st.spinner("CALCUL EN COURS..."):
                score = calculate_similarity(st.session_state.mot_cible, mot_propose) * 100

            st.session_state.historique.append((mot_propose, score))
            st.session_state.tentatives_restantes -= 1

            if mot_propose == st.session_state.mot_cible:
                st.session_state.jeu_termine = True
                st.session_state.gagne = True
            elif st.session_state.tentatives_restantes <= 0:
                st.session_state.jeu_termine = True
                st.session_state.gagne = False

            st.rerun()
        else:
            st.warning("Entrez un mot avant de proposer.")

    st.markdown('</div>', unsafe_allow_html=True)
    render_history()


# ── PHASE 3 : End screen ──────────────────────────────────────────────────────
elif st.session_state.jeu_termine:

    if st.session_state.gagne:
        st.balloons()
        st.markdown("""
        <div class="game-card end-screen">
            <div class="end-title win-title">★ VICTOIRE ! ★</div>
            <p style="color:var(--muted);font-family:'VT323',monospace;font-size:1.2rem;">Le Joueur 2 a trouvé le mot secret !</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="game-card end-screen">
            <div class="end-title lose-title">✕ GAME OVER ✕</div>
            <p style="color:var(--muted);font-family:'VT323',monospace;font-size:1.2rem;">Le mot secret était :</p>
        """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="end-word">> {st.session_state.mot_cible.upper()} <</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.historique:
        best_mot, best_score = max(st.session_state.historique, key=lambda x: x[1])
        st.markdown(f"""
        <div style="display:flex;justify-content:center;gap:2.5rem;margin:.8rem 0 1.4rem;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div class="stat-label">TENTATIVES</div>
                <div class="stat-value-teal">{len(st.session_state.historique)}</div>
            </div>
            <div style="text-align:center;">
                <div class="stat-label">MEILLEUR MOT</div>
                <div class="stat-value-gold">{best_mot.upper()}</div>
            </div>
            <div style="text-align:center;">
                <div class="stat-label">MEILLEUR SCORE</div>
                <div class="stat-value-pink">{best_score:.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_history()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="city-strip" style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    if st.button("[ START ] REJOUER UNE PARTIE", key="btn_rejouer"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()