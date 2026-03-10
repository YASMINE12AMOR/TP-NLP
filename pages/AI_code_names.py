import streamlit as st
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Codenames × IA", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:          #eef3ff;
  --bg2:         #e2ecff;
  --surface:     #ffffff;
  --surface2:    #f5f8ff;
  --border:      #cdd9f5;
  --navy:        #1a2f6e;
  --navy-mid:    #2545a8;
  --blue:        #3b6de8;
  --blue-light:  #6b93f5;
  --blue-pale:   #dce8ff;
  --red:         #e53535;
  --red-pale:    #ffe4e4;
  --text:        #0f1f5c;
  --text-mid:    #3a4f8c;
  --text-muted:  #7a8fbf;
  --text-dim:    #aab6d8;
  --shadow:      rgba(58,100,220,.10);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    background-image:
      radial-gradient(ellipse 65% 45% at 5% 0%,   rgba(59,109,232,.12) 0%, transparent 60%),
      radial-gradient(ellipse 55% 40% at 95% 100%, rgba(229,53,53,.09)  0%, transparent 55%),
      linear-gradient(160deg,#eaf0ff 0%,#f4f0ff 50%,#fff0f0 100%) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container          { padding-top: 1.8rem !important; max-width: 1080px; }

[data-testid="stAppViewContainer"]::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image: radial-gradient(circle, rgba(59,109,232,.10) 1px, transparent 1px);
    background-size: 36px 36px;
}

/* ── Cards ── */
.card {
    border-radius:14px; padding:14px 8px; text-align:center;
    min-height:68px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    font-family:'Outfit',sans-serif; font-weight:700;
    font-size:.82rem; letter-spacing:.06em; text-transform:uppercase;
    border:2px solid; margin-bottom:6px;
    box-shadow:0 2px 12px var(--shadow); position:relative; overflow:hidden;
}
.card::after {
    content:''; position:absolute; top:0; left:0; right:0; height:40%;
    background:linear-gradient(180deg,rgba(255,255,255,.16) 0%,transparent 100%);
    pointer-events:none;
}
.card-hidden  { background:#fff;         border-color:var(--border); color:var(--navy); }
.card-red     { background:linear-gradient(145deg,#e53535,#c41e1e); border-color:#ff7575; color:#fff; box-shadow:0 4px 18px rgba(229,53,53,.26); }
.card-blue    { background:linear-gradient(145deg,#2545a8,#1a2f6e); border-color:#5c85f0; color:#dce8ff; box-shadow:0 4px 18px rgba(37,69,168,.26); }
.card-neutral { background:linear-gradient(145deg,#dce6f0,#c8d6e8); border-color:#b0c4d8; color:#4a6080; }
.card-black   { background:linear-gradient(145deg,#1a1a2e,#0d0d1a); border-color:#3a3a5c; color:#5a5a8a; }

.master-red     { background:rgba(229,53,53,.09);  border-color:#ff8080; color:#c41e1e; }
.master-blue    { background:rgba(37,69,168,.10);  border-color:#6b93f5; color:#1a2f6e; }
.master-neutral { background:#f0f4f8;              border-color:#c8d6e8; color:#607080; }
.master-black   { background:#e8e8f0;              border-color:#c0c0d8; color:#808098; }
.master-guessed { opacity:.35; }

/* ── Score pills ── */
.score-pill {
    border-radius:18px; padding:16px 20px; text-align:center;
    border:2px solid; background:#fff; margin-bottom:6px;
    box-shadow:0 4px 20px var(--shadow); position:relative; overflow:hidden;
}
.score-pill::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:var(--bar); }
.score-label { font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--text-muted); margin-bottom:5px; }
.score-value { font-size:1.5rem; font-weight:900; line-height:1.1; }
.score-sub   { font-size:.76rem; color:var(--text-dim); margin-top:3px; }

/* ── AI badge ── */
.ai-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:linear-gradient(90deg,rgba(59,109,232,.10),rgba(229,53,53,.07));
    border:1.5px solid rgba(59,109,232,.22); border-radius:999px;
    padding:4px 14px; font-family:'JetBrains Mono',monospace;
    font-size:.68rem; color:var(--blue); letter-spacing:.08em;
}
.ai-dot { width:7px; height:7px; border-radius:50%; background:var(--blue); display:inline-block; animation:blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.2;} }

/* ── Clue box ── */
.clue-box {
    background:#fff; border:2px solid var(--border); border-left:4px solid var(--blue);
    border-radius:14px; padding:14px 18px; margin-bottom:14px;
    box-shadow:0 2px 14px var(--shadow);
}
.clue-word { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:var(--navy); }

.progress-track { height:6px; background:var(--bg2); border-radius:999px; overflow:hidden; margin-top:10px; }
.progress-fill  { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--blue),var(--red)); transition:width .4s; }

/* ── Banners ── */
.banner { border-radius:12px; padding:13px 18px; font-weight:600; font-size:.93rem; text-align:center; border:1.5px solid; margin-bottom:14px; }
.banner-success { background:#f0fff8; border-color:#5de8a0; color:#0a6640; }
.banner-error   { background:var(--red-pale); border-color:#ff9090; color:#b01010; }
.banner-info    { background:var(--blue-pale); border-color:var(--blue-light); color:var(--navy-mid); }
.banner-warn    { background:#fff8e6; border-color:#ffc947; color:#7a4f00; }
.banner-black   { background:#f0f0f8; border-color:#c0c0d8; color:#505070; }
.banner-win     { background:linear-gradient(90deg,var(--blue-pale),#ffe8e8); border-color:var(--blue); color:var(--navy); font-size:1.05rem; }

/* ── Hint cards ── */
.hint-card { background:#fff; border:1.5px solid var(--border); border-radius:10px; padding:10px 8px; text-align:center; box-shadow:0 2px 8px var(--shadow); }
.hint-word  { font-size:.95rem; font-weight:700; color:var(--navy); }
.hint-score { font-size:.72rem; font-family:'JetBrains Mono',monospace; color:var(--blue); margin-top:4px; }

/* ── Streamlit overrides ── */
[data-testid="stForm"] {
    background:#fff !important; border:2px solid var(--border) !important;
    border-radius:16px !important; padding:18px !important;
    box-shadow:0 4px 20px var(--shadow) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background:var(--surface2) !important; border:2px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important;
    font-family:'Outfit',sans-serif !important; font-weight:600 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color:var(--blue) !important;
    box-shadow:0 0 0 3px rgba(59,109,232,.15) !important;
}
.stButton>button {
    background:#fff !important; border:2px solid var(--border) !important;
    border-radius:10px !important; color:var(--text-mid) !important;
    font-family:'Outfit',sans-serif !important; font-weight:700 !important;
    transition:all .18s !important;
}
.stButton>button:hover {
    border-color:var(--blue) !important; color:var(--blue) !important;
    background:var(--blue-pale) !important;
    box-shadow:0 4px 14px rgba(59,109,232,.18) !important;
    transform:translateY(-1px) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#fff !important; border-radius:14px !important;
    border:2px solid var(--border) !important; padding:4px !important;
    gap:4px !important; box-shadow:0 2px 10px var(--shadow) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    background:transparent !important; color:var(--text-muted) !important;
    border-radius:10px !important; font-family:'Outfit',sans-serif !important;
    font-weight:600 !important; font-size:.9rem !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,var(--blue-pale),#ffe8e8) !important;
    color:var(--navy) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }
label { color:var(--text-muted) !important; font-family:'Outfit',sans-serif !important; }
p,li  { color:var(--text-mid); }
h1,h2,h3 { color:var(--text) !important; }

.view-switch {
    background:rgba(255,255,255,.72);
    border:2px solid var(--border);
    border-radius:18px;
    padding:8px;
    box-shadow:0 6px 24px var(--shadow);
    margin-bottom:16px;
}
.view-chip {
    border-radius:14px;
    padding:10px 14px;
    text-align:center;
    border:2px solid transparent;
    font-family:'Outfit',sans-serif;
    font-weight:800;
    letter-spacing:.03em;
    font-size:.92rem;
}
.view-chip-sub {
    display:block;
    font-size:.72rem;
    font-weight:600;
    letter-spacing:.06em;
    color:var(--text-muted);
    margin-top:3px;
}

.spin {
    display:inline-block; width:10px; height:10px;
    border:2px solid rgba(59,109,232,.2); border-top-color:var(--blue);
    border-radius:50%; animation:spin 1.2s linear infinite;
    vertical-align:middle; margin-right:6px;
}
@keyframes spin { to{transform:rotate(360deg);} }
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
FRENCH_WORDS = [
    "chien","chat","maison","voiture","arbre","fleur","livre","ordinateur","téléphone","table",
    "chaise","soleil","lune","étoile","mer","montagne","rivière","ville","pays","Europe",
    "Afrique","Asie","Amérique","Océanie","musique","film","sport","football","tennis","natation",
    "piano","guitare","art","peinture","nourriture","pizza","pâtes","fromage","vin","café",
    "thé","fruit","pomme","banane","légume","carotte","vêtement","métier","médecin","professeur",
]
CLUE_CANDIDATES = FRENCH_WORDS + [
    "animal","nature","transport","mobilier","technologie","espace","géographie","continent",
    "culture","loisir","instrument","cuisine","boisson","dessert","végétal","océan","science",
    "éducation","travail","médecine","sportif","artistique","voyage","habitat","communication",
    "musicien","restaurant","forêt","planète","univers","lecture","bureau","école","alimentation",
]
WORD_GROUPS = {
    "animal":{"chien","chat"}, "transport":{"voiture"}, "mobilier":{"table","chaise"},
    "technologie":{"ordinateur","téléphone"}, "espace":{"soleil","lune","étoile","planète"},
    "nature":{"arbre","fleur","mer","montagne","rivière","océan","forêt"},
    "géographie":{"ville","pays","Europe","Afrique","Asie","Amérique","Océanie"},
    "culture":{"musique","film","art","peinture","livre"}, "sport":{"football","tennis","natation"},
    "instrument":{"piano","guitare"}, "cuisine":{"nourriture","pizza","pâtes","fromage"},
    "boisson":{"vin","café","thé"}, "fruit":{"fruit","pomme","banane"},
    "légume":{"légume","carotte"}, "métier":{"métier","médecin","professeur"}, "vêtement":{"vêtement"},
}

@st.cache_resource
def load_model(): return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_clue_embeddings():
    m = load_model()
    return {c: m.encode(c, normalize_embeddings=True) for c in CLUE_CANDIDATES}

def generate_grid(n=25):
    words = random.sample(FRENCH_WORDS, n)
    colors = ["red"]*9+["blue"]*8+["neutral"]*7+["black"]*1
    random.shuffle(colors)
    return list(zip(words, colors))

def switch_team(t): return "blue" if t=="red" else "red"

def compute_lamp_hint(grid, guesses, clue_word):
    if not clue_word: return []
    m = load_model(); avail = [w for w,_ in grid if w not in guesses]
    if not avail: return []
    enc = m.encode(avail, normalize_embeddings=True)
    cenc = m.encode(clue_word, normalize_embeddings=True)
    sims = np.dot(enc, cenc)
    if len(avail)==1: return [{"word":avail[0],"similarity":float(sims[0])}]
    nc = min(max(2,len(avail)//4),len(avail)-1)
    try:    cl = AgglomerativeClustering(n_clusters=nc, metric="cosine", linkage="average")
    except: cl = AgglomerativeClustering(n_clusters=nc, affinity="cosine", linkage="average")
    labels = cl.fit_predict(enc)
    best_id,best_sc = None,float("-inf")
    for cid in set(labels):
        idxs=[i for i,l in enumerate(labels) if l==cid]
        sc=float(np.mean([sims[i] for i in idxs]))
        if sc>best_sc: best_sc,best_id=sc,cid
    ranked=sorted([{"word":avail[i],"similarity":float(sims[i])} for i,l in enumerate(labels) if l==best_id],
                  key=lambda x:x["similarity"],reverse=True)
    return ranked[:3]

def find_group_clue(grid, guesses, team, excluded=None):
    excluded=set(excluded or []); tw={w for w,c in grid if c==team and w not in guesses}
    board={w for w,_ in grid}; opp=board-tw-set(guesses); best=None
    for cw,group in WORD_GROUPS.items():
        if cw in excluded or cw in board: continue
        tm,om=tw&group,opp&group
        if len(tm)<2 or om: continue
        if best is None or len(tm)>best[0]: best=(len(tm),cw)
    return (best[1],best[0]) if best else None

def generate_ai_clue(grid, guesses, team, excluded=None):
    g=find_group_clue(grid,guesses,team,excluded)
    if g: return g
    m=load_model(); ce=load_clue_embeddings(); board={w for w,_ in grid}; excluded=set(excluded or [])
    tw=[w for w,c in grid if c==team and w not in guesses]
    ow=[w for w,c in grid if c!=team and w not in guesses]
    if not tw: return ("victoire",1)
    te={w:m.encode(w,normalize_embeddings=True) for w in tw}
    oe={w:m.encode(w,normalize_embeddings=True) for w in ow}
    scored=[]; mt=min(4,len(tw))
    for cand,cemb in ce.items():
        if cand in board or cand in guesses or cand in excluded: continue
        ts=sorted([(w,float(np.dot(cemb,te[w]))) for w in tw],reverse=True,key=lambda x:x[1])
        risk=max([float(np.dot(cemb,e)) for e in oe.values()],default=0.)
        for tc in range(mt,0,-1):
            sel=ts[:tc]
            if len(sel)<tc: continue
            ss=[s for _,s in sel]; mean_,min_=float(np.mean(ss)),min(ss)
            if min_<0.16 or risk>=min_+0.03: continue
            if mean_<0.22 and tc>1: continue
            scored.append((mean_+0.08*(tc-1)-1.10*risk-0.03*abs(tc-2),cand,tc))
    if scored:
        scored.sort(reverse=True,key=lambda x:x[0])
        _,cand,tc=random.choice(scored[:min(5,len(scored))])
        return (cand,tc)
    fb=next((c for c in CLUE_CANDIDATES if c not in board and c not in guesses and c not in excluded),"association")
    return (fb,1)

# ── State ──────────────────────────────────────────────────────────────────────
def init():
    if "grid" not in st.session_state:
        st.session_state.update({
            "grid":generate_grid(),"current_team":"red","guesses":[],
            "scores":{"red":0,"blue":0},"game_over":False,"clue":("",1),
            "turn_correct_guesses":0,"lamp_hints":{"red":[],"blue":[]},"message":None,"active_view":"players",
        })
    for k,v in [("turn_correct_guesses",0),("lamp_hints",{"red":[],"blue":[]}),("message",None),("active_view","players")]:
        if k not in st.session_state: st.session_state[k]=v

init()
grid=st.session_state.grid; cur=st.session_state.current_team
guesses=st.session_state.guesses; scores=st.session_state.scores
game_over=st.session_state.game_over; clue=st.session_state.clue
tg=st.session_state.turn_correct_guesses; lamp_hints=st.session_state.lamp_hints
safe_n=max(1,min(5,int(clue[1]) if isinstance(clue[1],(int,float)) else 1))
red_left=sum(1 for w,c in grid if c=="red" and w not in guesses)
blue_left=sum(1 for w,c in grid if c=="blue" and w not in guesses)

if not game_over and not clue[0]:
    ai_clue=generate_ai_clue(grid,guesses,cur)
    st.session_state.clue=ai_clue; st.session_state.turn_correct_guesses=0
    st.session_state.lamp_hints={"red":[],"blue":[]}
    clue=ai_clue; safe_n=clue[1]; tg=0

tl="Rouge" if cur=="red" else "Bleue"
tc="#e53535" if cur=="red" else "#2545a8"
tb="#ff8080" if cur=="red" else "#6b93f5"
tp="#ffe4e4" if cur=="red" else "#dce8ff"
emap={"red":"🔴","blue":"🔵","neutral":"⚪","black":"💀"}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; margin-bottom:10px;'>
  <div style='margin-bottom:10px;'>
    <span class='ai-badge'><span class='ai-dot'></span>&nbsp;SYSTÈME IA ACTIF</span>
  </div>
  <div style='font-family:"Outfit",sans-serif; font-size:2.8rem; font-weight:900; letter-spacing:-.02em; line-height:1;'>
    <span style='color:#1a2f6e;'>CODE</span><span style='color:#e53535;'>NAMES</span>
    <span style='font-size:.95rem; color:#3b6de8; font-weight:700; vertical-align:middle;
                 background:#dce8ff; border-radius:999px; padding:3px 14px; margin-left:10px;'>× IA</span>
  </div>
  <div style='color:#7a8fbf; font-size:.87rem; margin-top:6px; letter-spacing:.02em;'>
    Le maître du jeu est une intelligence artificielle · Version française
  </div>
</div>
<div style='display:flex; align-items:center; gap:10px; margin:0 0 20px 0;'>
  <div style='flex:1; height:1.5px; background:linear-gradient(90deg,transparent,#3b6de8 60%,transparent);'></div>
  <span style='font-size:.7rem; color:#aab6d8; font-family:"JetBrains Mono",monospace;'>◆</span>
  <div style='flex:1; height:1.5px; background:linear-gradient(90deg,transparent,#e53535 60%,transparent);'></div>
</div>
""", unsafe_allow_html=True)

# ── Scores ─────────────────────────────────────────────────────────────────────
c1,c2,c3=st.columns([2,2.5,2])
with c1:
    r="box-shadow:0 0 0 4px rgba(229,53,53,.16);" if cur=="red" else ""
    st.markdown(f"""<div class='score-pill' style='border-color:#ff9090; --bar:linear-gradient(90deg,#e53535,#ff7070); {r}'>
      <div class='score-label'>🔴 Équipe Rouge</div>
      <div class='score-value' style='color:#e53535;'>{scores["red"]} pts</div>
      <div class='score-sub'>{red_left} mot(s) restant(s)</div>
    </div>""", unsafe_allow_html=True)
with c2:
    pct=int((tg/safe_n)*100) if safe_n else 0
    st.markdown(f"""<div class='score-pill' style='border-color:{tb}; box-shadow:0 4px 22px rgba(59,109,232,.12);'>
      <div class='score-label'>Tour actuel</div>
      <div class='score-value' style='color:{tc};'>Équipe {tl}</div>
      <div style='margin-top:8px;'>
        <div style='font-family:"JetBrains Mono",monospace; font-size:1.15rem; font-weight:700; color:#1a2f6e;'>
          {clue[0] if clue[0] else "…"}
          <span style='font-size:.78rem; color:#aab6d8; font-weight:400;'>&nbsp;({safe_n})</span>
        </div>
        <div class='progress-track'><div class='progress-fill' style='width:{pct}%;'></div></div>
        <div style='font-size:.7rem; color:#aab6d8; font-family:"JetBrains Mono",monospace; margin-top:3px;'>{tg}/{safe_n} devinés</div>
      </div>
    </div>""", unsafe_allow_html=True)
with c3:
    r="box-shadow:0 0 0 4px rgba(37,69,168,.16);" if cur=="blue" else ""
    st.markdown(f"""<div class='score-pill' style='border-color:#7090f0; --bar:linear-gradient(90deg,#2545a8,#6b93f5); {r}'>
      <div class='score-label'>🔵 Équipe Bleue</div>
      <div class='score-value' style='color:#2545a8;'>{scores["blue"]} pts</div>
      <div class='score-sub'>{blue_left} mot(s) restant(s)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

if st.session_state.message:
    mt,mx=st.session_state.message
    st.markdown(f"<div class='banner banner-{mt}'>{mx}</div>", unsafe_allow_html=True)

nav1, nav2 = st.columns(2)
with nav1:
    players_active = st.session_state.active_view == "players"
    st.markdown(
        f"""<div class='view-chip' style='background:{'#ffffff' if players_active else 'rgba(255,255,255,.45)'};
            border-color:{'#8db4ff' if players_active else '#d7e3ff'}; color:{'#1a2f6e' if players_active else '#6f84b8'};
            box-shadow:{'0 8px 22px rgba(59,109,232,.16)' if players_active else 'none'}; margin-bottom:8px;'>
            👥 Vue Joueurs
            <span class='view-chip-sub'>cartes cachées et prises de décision</span>
        </div>""",
        unsafe_allow_html=True,
    )
    if st.button("Vue Joueurs", key="nav_players", use_container_width=True, type="primary" if players_active else "secondary"):
        st.session_state.active_view = "players"
        st.rerun()
with nav2:
    master_active = st.session_state.active_view == "master"
    st.markdown(
        f"""<div class='view-chip' style='background:{'#ffffff' if master_active else 'rgba(255,255,255,.45)'};
            border-color:{'#ffb6c0' if master_active else '#ffd8de'}; color:{'#8f2942' if master_active else '#b07b86'};
            box-shadow:{'0 8px 22px rgba(229,53,53,.14)' if master_active else 'none'}; margin-bottom:8px;'>
            🤖 Vue Maître IA
            <span class='view-chip-sub'>couleurs visibles et indice automatique</span>
        </div>""",
        unsafe_allow_html=True,
    )
    if st.button("Vue Maître IA", key="nav_master", use_container_width=True, type="primary" if master_active else "secondary"):
        st.session_state.active_view = "master"
        st.rerun()

# ═══════════════════════ TAB JOUEURS ═══════════════════════
if st.session_state.active_view == "players":
    cols=st.columns(5, gap="small")
    for i,(word,color) in enumerate(grid):
        with cols[i%5]:
            if word in guesses:
                st.markdown(f"<div class='card card-{color}'>{emap.get(color,'')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card card-hidden'>{word}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if clue[0]:
        pct=int((tg/safe_n)*100) if safe_n else 0
        st.markdown(f"""<div class='clue-box'>
          <div style='display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px;'>
            <span style='font-size:.68rem; color:#aab6d8; font-family:"JetBrains Mono",monospace; letter-spacing:.1em;'>INDICE IA</span>
            <span class='ai-badge' style='font-size:.62rem;'><span class='spin'></span>embeddings sémantiques</span>
          </div>
          <div style='display:flex; align-items:baseline; gap:12px;'>
            <span class='clue-word'>«&nbsp;{clue[0]}&nbsp;»</span>
            <span style='font-size:.88rem; color:#7a8fbf; font-family:"JetBrains Mono",monospace;'>→ {safe_n} mot(s)</span>
          </div>
          <div class='progress-track'><div class='progress-fill' style='width:{pct}%;'></div></div>
        </div>""", unsafe_allow_html=True)

    ah=lamp_hints.get(cur,[])
    if ah:
        ch="".join([f"<div class='hint-card'><div class='hint-word'>{h['word']}</div><div class='hint-score'>sim&nbsp;{h['similarity']:.2f}</div></div>" for h in ah])
        st.markdown(f"""<div style='background:linear-gradient(135deg,{tp},{tp}88); border:2px solid {tb};
                    border-radius:16px; padding:16px; margin-bottom:14px;'>
          <div style='font-size:.72rem; font-family:"JetBrains Mono",monospace; color:{tc}; letter-spacing:.08em; margin-bottom:8px;'>
            <span class='spin'></span>CLUSTERING IA — Mots proches de «&nbsp;{clue[0]}&nbsp;»
          </div>
          <div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px;'>{ch}</div>
        </div>""", unsafe_allow_html=True)

    lc1,lc2=st.columns(2)
    with lc1:
        st.markdown(f"""<div style='background:linear-gradient(135deg,#fff1f1,#ffe4e6);
            border:2px solid #ffb4b4; border-radius:16px; padding:14px 14px 10px 14px; margin-bottom:8px;'>
            <div style='font-size:.78rem; font-family:"JetBrains Mono",monospace; color:#b91c1c; letter-spacing:.08em; margin-bottom:6px;'>
              💡 ASSISTANCE ROUGE
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔴 Clustering Rouge", key="lamp_red", use_container_width=True, disabled=not clue[0], type="primary"):
            st.session_state.lamp_hints["red"]=compute_lamp_hint(grid,guesses,clue[0]); st.rerun()
    with lc2:
        st.markdown(f"""<div style='background:linear-gradient(135deg,#eef5ff,#e0ecff);
            border:2px solid #a8c5ff; border-radius:16px; padding:14px 14px 10px 14px; margin-bottom:8px;'>
            <div style='font-size:.78rem; font-family:"JetBrains Mono",monospace; color:#1d4ed8; letter-spacing:.08em; margin-bottom:6px;'>
              💡 ASSISTANCE BLEUE
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔵 Clustering Bleue", key="lamp_blue", use_container_width=True, disabled=not clue[0]):
            st.session_state.lamp_hints["blue"]=compute_lamp_hint(grid,guesses,clue[0]); st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if not game_over:
        with st.form("guess_form", clear_on_submit=True):
            st.markdown(f"""<div style='font-size:.85rem; font-weight:700; color:{tc};
                letter-spacing:.05em; margin-bottom:8px;'>DEVINER — ÉQUIPE {tl.upper()}</div>""", unsafe_allow_html=True)
            guess_in=st.text_input("Mot:", placeholder="Tapez un mot de la grille…", label_visibility="collapsed")
            g1,g2=st.columns([3,1])
            with g1: submitted=st.form_submit_button("✅  Valider", use_container_width=True)
            with g2: pass_turn=st.form_submit_button("⏭ Passer", use_container_width=True)

        if submitted and guess_in.strip():
            g=guess_in.strip().lower(); matched=[(w,c) for w,c in grid if w.lower()==g]
            if not matched: st.session_state.message=("error",f"❌ « {guess_in} » introuvable.")
            elif matched[0][0] in guesses: st.session_state.message=("warn","⚠️ Déjà deviné.")
            else:
                w,col=matched[0]; guesses.append(w)
                if col=="black":
                    st.session_state.message=("black","💀 ASSASSIN — l'équipe adverse gagne !")
                    st.session_state.game_over=True; st.session_state.lamp_hints={"red":[],"blue":[]}
                elif col==cur:
                    st.session_state.scores[cur]+=1; st.session_state.turn_correct_guesses+=1
                    ntg=st.session_state.turn_correct_guesses
                    if ntg>=safe_n:
                        st.session_state.message=("info","✅ Nombre atteint — l'IA génère un nouvel indice.")
                        st.session_state.current_team=switch_team(cur); st.session_state.turn_correct_guesses=0
                        st.session_state.clue=("",1); st.session_state.lamp_hints={"red":[],"blue":[]}
                    else: st.session_state.message=("success",f"✅ Bonne réponse ! {ntg}/{safe_n} trouvés.")
                else:
                    if col in ("red","blue"): st.session_state.scores[col]+=1
                    lab="de l'adversaire" if col in ("red","blue") else "neutre"
                    st.session_state.message=("warn",f"⚠️ Mot {lab} — tour suivant.")
                    st.session_state.current_team=switch_team(cur); st.session_state.turn_correct_guesses=0
                    st.session_state.clue=("",1); st.session_state.lamp_hints={"red":[],"blue":[]}
            st.rerun()
        if pass_turn:
            st.session_state.message=("info","⏭️ Tour passé.")
            st.session_state.current_team=switch_team(cur); st.session_state.turn_correct_guesses=0
            st.session_state.clue=("",1); st.session_state.lamp_hints={"red":[],"blue":[]}; st.rerun()

    nr=sum(1 for w,c in grid if c=="red"  and w not in guesses)
    nb=sum(1 for w,c in grid if c=="blue" and w not in guesses)
    if nr==0 and not st.session_state.game_over:
        st.balloons(); st.session_state.message=("win","🎉 Équipe ROUGE — VICTOIRE !"); st.session_state.game_over=True; st.rerun()
    if nb==0 and not st.session_state.game_over:
        st.balloons(); st.session_state.message=("win","🎉 Équipe BLEUE — VICTOIRE !"); st.session_state.game_over=True; st.rerun()

# ═══════════════════════ TAB MAÎTRE IA ═══════════════════════
if st.session_state.active_view == "master":
    st.markdown(f"""<div style='background:linear-gradient(90deg,rgba(59,109,232,.07),rgba(229,53,53,.05));
                border:1.5px solid rgba(59,109,232,.18); border-radius:12px; padding:11px 16px;
                margin-bottom:16px; font-family:"JetBrains Mono",monospace; font-size:.72rem; color:#3b6de8; letter-spacing:.06em;'>
      <span class='spin'></span>MODULE IA · all-MiniLM-L6-v2 · CLUSTERING AGGLOMÉRATIF · BEAM SEARCH SÉMANTIQUE
    </div>""", unsafe_allow_html=True)

    mc=st.columns(5, gap="small")
    for i,(word,color) in enumerate(grid):
        with mc[i%5]:
            g=" master-guessed" if word in guesses else ""
            st.markdown(f"<div class='card master-{color}{g}'>{word}<br><span style='font-size:.65rem;opacity:.6;'>{emap.get(color,'')}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown(f"""<div style='background:#fff; border:2px solid var(--border); border-left:4px solid #3b6de8;
                border-radius:14px; padding:16px 20px; margin-bottom:14px; box-shadow:0 4px 20px var(--shadow);'>
      <div style='font-size:.7rem; font-family:"JetBrains Mono",monospace; color:#aab6d8; letter-spacing:.1em; margin-bottom:8px;'>
        INDICE GÉNÉRÉ — ÉQUIPE {tl.upper()}</div>
      <div style='display:flex; align-items:center; gap:14px; flex-wrap:wrap;'>
        <span style='font-family:"JetBrains Mono",monospace; font-size:1.5rem; font-weight:700; color:#1a2f6e;'>{clue[0] if clue[0] else "—"}</span>
        <span style='font-size:.9rem; color:#7a8fbf;'>({safe_n} mot(s))</span>
        <span class='ai-badge' style='font-size:.62rem;'><span class='ai-dot'></span>&nbsp;auto-généré</span>
      </div>
      <div style='font-size:.78rem; color:#aab6d8; margin-top:6px;'>Sélectionné par analyse vectorielle des embeddings sémantiques.</div>
    </div>""", unsafe_allow_html=True)

    if st.button("🔄  Régénérer l'indice IA"):
        nc=generate_ai_clue(grid,guesses,cur,excluded_clues={clue[0]})
        st.session_state.clue=nc; st.session_state.turn_correct_guesses=0
        st.session_state.lamp_hints={"red":[],"blue":[]}
        st.session_state.message=("info",f"🤖 Nouvel indice : «&nbsp;{nc[0]}&nbsp;» ({nc[1]})")
        st.rerun()

    st.markdown("""<div style='display:flex; gap:20px; flex-wrap:wrap; margin-top:16px; font-size:.82rem; color:#7a8fbf;'>
      <span><b style='color:#e53535;'>■</b> Rouge : 9</span>
      <span><b style='color:#2545a8;'>■</b> Bleue : 8</span>
      <span><b style='color:#b0c4d8;'>■</b> Neutre : 7</span>
      <span><b style='color:#808098;'>■</b> Assassin : 1</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
if st.button("⟳  Nouvelle Partie"):
    for k in ["grid","current_team","guesses","scores","game_over","clue","turn_correct_guesses","lamp_hints","message"]:
        st.session_state.pop(k, None)
    st.rerun()
