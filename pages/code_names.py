import streamlit as st
import random

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Codenames FR",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (light, airy, readable) ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fdf6ec 0%, #fef3f2 45%, #eff6ff 100%) !important;
    font-family: 'Nunito', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1.5rem !important; max-width: 1060px; }

.card {
    border-radius: 14px;
    padding: 14px 8px;
    text-align: center;
    min-height: 70px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 0.84rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 2px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    margin-bottom: 6px;
}
.card-hidden   { background:#ffffff; border-color:#e5e7eb; color:#1f2937; }
.card-red      { background:#ef4444; border-color:#dc2626; color:#ffffff; }
.card-blue     { background:#3b82f6; border-color:#2563eb; color:#ffffff; }
.card-neutral  { background:#9ca3af; border-color:#6b7280; color:#ffffff; }
.card-black    { background:#111827; border-color:#111827; color:#ffffff; }

.master-red     { background:#fff0f0; border-color:#f87171; color:#991b1b; }
.master-blue    { background:#eff6ff; border-color:#60a5fa; color:#1e3a8a; }
.master-neutral { background:#f9fafb; border-color:#d1d5db; color:#374151; }
.master-black   { background:#1f2937; border-color:#111827; color:#f9fafb; }
.master-guessed { opacity: .4; }

.score-pill {
    border-radius: 16px;
    padding: 14px 20px;
    background: white;
    text-align: center;
    border: 2px solid;
    margin-bottom: 6px;
}
.score-label  { font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:4px; }
.score-value  { font-size:1.5rem; font-weight:900; line-height:1; }
.score-sub    { font-size:.78rem; color:#9ca3af; margin-top:4px; }

.banner { border-radius:13px; padding:13px 18px; font-family:'Nunito',sans-serif; font-weight:700; font-size:.95rem; text-align:center; border:1.5px solid; margin-bottom:14px; }
.banner-success { background:#f0fdf4; border-color:#86efac; color:#15803d; }
.banner-error   { background:#fef2f2; border-color:#fca5a5; color:#991b1b; }
.banner-info    { background:#f0f9ff; border-color:#7dd3fc; color:#0369a1; }
.banner-warn    { background:#fffbeb; border-color:#fcd34d; color:#92400e; }
.banner-black   { background:#111827; border-color:#374151; color:#f9fafb; }
.banner-win     { background:linear-gradient(90deg,#fef9c3,#d1fae5); border-color:#6ee7b7; color:#065f46; font-size:1.1rem; }

[data-testid="stForm"] {
    background: white !important;
    border-radius: 16px !important;
    border: 2px solid #e9d5ff !important;
    padding: 16px !important;
}
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

def generate_grid(n=25):
    words = random.sample(FRENCH_WORDS, n)
    colors = ["red"]*9 + ["blue"]*8 + ["neutral"]*7 + ["black"]*1
    random.shuffle(colors)
    return list(zip(words, colors))

def switch_team():
    st.session_state.current_team = "blue" if st.session_state.current_team == "red" else "red"
    st.session_state.turn_correct_guesses = 0
    st.session_state.clue = ("", 1)

def init():
    if "grid" not in st.session_state:
        st.session_state.grid                  = generate_grid()
        st.session_state.current_team          = "red"
        st.session_state.guesses               = []
        st.session_state.scores                = {"red": 0, "blue": 0}
        st.session_state.game_over             = False
        st.session_state.clue                  = ("", 1)
        st.session_state.turn_correct_guesses  = 0
        st.session_state.message               = None
    elif "message" not in st.session_state:
        st.session_state.message = None

init()

# Shortcuts
grid         = st.session_state.grid
current_team = st.session_state.current_team
guesses      = st.session_state.guesses
scores       = st.session_state.scores
game_over    = st.session_state.game_over
clue         = st.session_state.clue
tg           = st.session_state.turn_correct_guesses
safe_count   = max(1, min(5, int(clue[1]) if isinstance(clue[1], (int, float)) else 1))
red_left     = sum(1 for w, c in grid if c == "red"  and w not in guesses)
blue_left    = sum(1 for w, c in grid if c == "blue" and w not in guesses)
team_label   = "🔴 Rouge" if current_team == "red" else "🔵 Bleue"
team_color   = "#ef4444"  if current_team == "red" else "#3b82f6"
team_border  = "#fca5a5"  if current_team == "red" else "#93c5fd"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-bottom:8px;'>
  <span style='font-family:"Playfair Display",serif; font-size:2.3rem; font-weight:900;
               background:linear-gradient(90deg,#ef4444,#f97316,#3b82f6);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🎲 Codenames
  </span><br>
  <span style='font-family:"Nunito",sans-serif; color:#9ca3af; font-size:.88rem;'>
    Jeu de déduction par équipes · Version française
  </span>
</div>
""", unsafe_allow_html=True)

# ── Scores ─────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2.4, 2])
with c1:
    ring = "box-shadow:0 0 0 6px rgba(239,68,68,.15);" if current_team=="red" else ""
    st.markdown(f"""<div class='score-pill' style='border-color:#fca5a5;{ring}'>
      <div class='score-label'>🔴 Équipe Rouge</div>
      <div class='score-value' style='color:#dc2626;'>{scores["red"]} pts</div>
      <div class='score-sub'>{red_left} mot(s) restant(s)</div>
    </div>""", unsafe_allow_html=True)

with c2:
    clue_disp = f"<b>«&nbsp;{clue[0]}&nbsp;»</b> &nbsp;·&nbsp; {tg}/{safe_count}" if clue[0] else "En attente d'un indice…"
    st.markdown(f"""<div class='score-pill' style='border-color:{team_border};'>
      <div class='score-label'>Tour actuel</div>
      <div class='score-value' style='color:{team_color};'>{team_label}</div>
      <div style='font-size:.85rem; color:#374151; margin-top:5px;'>{clue_disp}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    ring = "box-shadow:0 0 0 6px rgba(59,130,246,.15);" if current_team=="blue" else ""
    st.markdown(f"""<div class='score-pill' style='border-color:#93c5fd;{ring}'>
      <div class='score-label'>🔵 Équipe Bleue</div>
      <div class='score-value' style='color:#2563eb;'>{scores["blue"]} pts</div>
      <div class='score-sub'>{blue_left} mot(s) restant(s)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Message banner ─────────────────────────────────────────────────────────────
if st.session_state.message:
    mtype, mtext = st.session_state.message
    st.markdown(f"<div class='banner banner-{mtype}'>{mtext}</div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_players, tab_master = st.tabs(["👥  Vue Joueurs", "👁️  Vue Maître du Jeu"])

# ─── TAB PLAYERS ──────────────────────────────────────────────────────────────
with tab_players:
    cols = st.columns(5, gap="small")
    emoji_map = {"red":"🔴","blue":"🔵","neutral":"⚪","black":"💀"}
    for i, (word, color) in enumerate(grid):
        with cols[i % 5]:
            if word in guesses:
                st.markdown(f"<div class='card card-{color}'>{emoji_map.get(color,'')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card card-hidden'>{word}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if not game_over:
        with st.form("guess_form", clear_on_submit=True):
            st.markdown(f"""<div style='font-family:"Playfair Display",serif; font-size:1rem;
                font-weight:700; color:{team_color}; margin-bottom:10px;'>
                Deviner un mot — {team_label}</div>""", unsafe_allow_html=True)

            if clue[0]:
                pct = int((tg / safe_count) * 100)
                st.markdown(f"""
                <div style='background:#f9fafb; border-radius:10px; padding:10px 14px;
                            border:1px solid #e5e7eb; margin-bottom:10px;
                            font-family:"Nunito",sans-serif;'>
                  <span style='font-weight:700; color:#374151;'>Indice :</span>
                  <span style='font-size:1.1rem; font-weight:800; color:{team_color};'> «&nbsp;{clue[0]}&nbsp;»</span>
                  <span style='color:#9ca3af;'> &nbsp;·&nbsp; {safe_count} mot(s) à trouver</span>
                  <div style='margin-top:8px; height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden;'>
                    <div style='width:{pct}%; height:100%; background:{team_color}; border-radius:999px;'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("⏳ Le maître du jeu n'a pas encore donné d'indice.")

            guess_in = st.text_input("Mot :", placeholder="Tapez un mot de la grille…", label_visibility="collapsed")
            g1, g2 = st.columns([3, 1])
            with g1:
                submitted = st.form_submit_button("✅  Valider", use_container_width=True, disabled=not clue[0])
            with g2:
                pass_turn = st.form_submit_button("⏭️  Passer", use_container_width=True, disabled=not clue[0])

        # Process guess
        if submitted and guess_in.strip():
            g = guess_in.strip().lower()
            matched = [(w, c) for w, c in grid if w.lower() == g]
            if not matched:
                st.session_state.message = ("error", f"❌ « {guess_in} » n'est pas dans la grille.")
            elif matched[0][0] in guesses:
                st.session_state.message = ("warn", "⚠️ Ce mot a déjà été deviné !")
            else:
                w, col = matched[0]
                guesses.append(w)
                if col == "black":
                    st.session_state.message = ("black", "💀 ASSASSIN ! L'équipe adverse gagne automatiquement !")
                    st.session_state.game_over = True
                elif col == current_team:
                    st.session_state.scores[current_team] += 1
                    st.session_state.turn_correct_guesses += 1
                    new_tg = st.session_state.turn_correct_guesses
                    if new_tg >= safe_count:
                        st.session_state.message = ("info", f"✅ Bonne réponse ! Nombre atteint — tour suivant.")
                        switch_team()
                    else:
                        st.session_state.message = ("success", f"✅ Bonne réponse ! {new_tg}/{safe_count} trouvés.")
                else:
                    if col in ("red","blue"):
                        st.session_state.scores[col] += 1
                    label = "de l'équipe adverse" if col in ("red","blue") else "neutre"
                    st.session_state.message = ("warn", f"⚠️ Mot {label} — le tour passe à l'adversaire.")
                    switch_team()
            st.rerun()

        if pass_turn:
            st.session_state.message = ("info", "⏭️ Tour passé volontairement.")
            switch_team()
            st.rerun()

    # Victory
    nr = sum(1 for w, c in grid if c == "red"  and w not in guesses)
    nb = sum(1 for w, c in grid if c == "blue" and w not in guesses)
    if nr == 0 and not st.session_state.game_over:
        st.balloons()
        st.session_state.message = ("win", "🎉 L'équipe ROUGE a trouvé tous ses mots — VICTOIRE !")
        st.session_state.game_over = True; st.rerun()
    if nb == 0 and not st.session_state.game_over:
        st.balloons()
        st.session_state.message = ("win", "🎉 L'équipe BLEUE a trouvé tous ses mots — VICTOIRE !")
        st.session_state.game_over = True; st.rerun()

# ─── TAB MASTER ───────────────────────────────────────────────────────────────
with tab_master:
    st.markdown("""<div style='font-family:"Nunito",sans-serif; color:#6d28d9;
        font-weight:700; font-size:.88rem; margin-bottom:12px;'>
        🔐 Réservé au maître du jeu — toutes les couleurs sont visibles.
    </div>""", unsafe_allow_html=True)

    mcols = st.columns(5, gap="small")
    for i, (word, color) in enumerate(grid):
        with mcols[i % 5]:
            g_cls = " master-guessed" if word in guesses else ""
            st.markdown(f"""
            <div class='card master-{color}{g_cls}'>
              {word}<br>
              <span style='font-size:.68rem; opacity:.7;'>{emoji_map.get(color,"")}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    with st.form("master_form"):
        st.markdown(f"""<div style='font-family:"Playfair Display",serif; font-size:1rem;
            font-weight:700; color:#6d28d9; margin-bottom:10px;'>
            Donner un indice — {team_label}</div>""", unsafe_allow_html=True)
        mc1, mc2 = st.columns([3, 1])
        with mc1:
            new_word = st.text_input("Mot-indice :", value=clue[0], placeholder="Un mot qui relie plusieurs cartes…")
        with mc2:
            new_count = st.number_input("Nb (1–5) :", min_value=1, max_value=5, value=safe_count)
        if st.form_submit_button("🟣  Valider l'indice", use_container_width=True):
            if new_word.strip():
                st.session_state.clue = (new_word.strip(), int(new_count))
                st.session_state.turn_correct_guesses = 0
                st.session_state.message = ("info", f"💬 Indice donné : «&nbsp;{new_word.strip()}&nbsp;» · {int(new_count)} mot(s)")
                st.rerun()
            else:
                st.warning("⚠️ Veuillez entrer un mot-indice.")

    st.markdown("""
    <div style='display:flex; gap:20px; flex-wrap:wrap; margin-top:14px;
                font-family:"Nunito",sans-serif; font-size:.82rem; color:#6b7280;'>
      <span><b style='color:#ef4444;'>■</b> Rouge : 9 cartes</span>
      <span><b style='color:#3b82f6;'>■</b> Bleue : 8 cartes</span>
      <span><b style='color:#9ca3af;'>■</b> Neutre : 7 cartes</span>
      <span><b style='color:#111827;'>■</b> Assassin : 1 carte</span>
    </div>""", unsafe_allow_html=True)

# ── New game ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
if st.button("🔄  Nouvelle partie"):
    for k in ["grid","current_team","guesses","scores","game_over","clue","turn_correct_guesses","message"]:
        st.session_state.pop(k, None)
    st.rerun()
