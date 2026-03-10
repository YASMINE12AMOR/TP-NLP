import streamlit as st
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Configuration de la page
st.set_page_config(
    page_title="🎲 Codenames en Français",
    page_icon="🤖",
    layout="wide"
)

# Liste de mots français
FRENCH_WORDS = [
    "chien", "chat", "maison", "voiture", "arbre", "fleur", "livre", "ordinateur", "téléphone", "table",
    "chaise", "soleil", "lune", "étoile", "mer", "montagne", "rivière", "ville", "pays", "Europe",
    "Afrique", "Asie", "Amérique", "Océanie", "musique", "film", "sport", "football", "tennis", "natation",
    "piano", "guitare", "art", "peinture", "nourriture", "pizza", "pâtes", "fromage", "vin", "café",
    "thé", "fruit", "pomme", "banane", "légume", "carotte", "vêtement", "métier", "médecin", "professeur"
]

CLUE_CANDIDATES = FRENCH_WORDS + [
    "animal", "nature", "transport", "mobilier", "technologie", "espace", "géographie", "continent",
    "culture", "loisir", "instrument", "cuisine", "boisson", "dessert", "végétal", "océan", "science",
    "éducation", "travail", "médecine", "sportif", "artistique", "voyage", "habitat", "communication",
    "musicien", "restaurant", "forêt", "planète", "univers", "lecture", "bureau", "école", "alimentation"
]

WORD_GROUPS = {
    "animal": {"chien", "chat"},
    "transport": {"voiture"},
    "mobilier": {"table", "chaise"},
    "technologie": {"ordinateur", "téléphone"},
    "espace": {"soleil", "lune", "étoile", "planète"},
    "nature": {"arbre", "fleur", "mer", "montagne", "rivière", "océan", "forêt"},
    "géographie": {"ville", "pays", "Europe", "Afrique", "Asie", "Amérique", "Océanie"},
    "culture": {"musique", "film", "art", "peinture", "livre"},
    "sport": {"football", "tennis", "natation"},
    "instrument": {"piano", "guitare"},
    "cuisine": {"nourriture", "pizza", "pâtes", "fromage"},
    "boisson": {"vin", "café", "thé"},
    "fruit": {"fruit", "pomme", "banane"},
    "légume": {"légume", "carotte"},
    "métier": {"métier", "médecin", "professeur"},
    "vêtement": {"vêtement"},
}

# Charger le modèle
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_clue_embeddings():
    model = load_model()
    return {
        candidate: model.encode(candidate, normalize_embeddings=True)
        for candidate in CLUE_CANDIDATES
    }

# Générer la grille
def generate_grid(n=25):
    words = random.sample(FRENCH_WORDS, n)
    colors = ["red"]*9 + ["blue"]*8 + ["neutral"]*7 + ["black"]
    random.shuffle(colors)
    return list(zip(words, colors))

def switch_team(team):
    return "blue" if team == "red" else "red"

def compute_lamp_hint(grid, guesses, clue_word):
    if not clue_word:
        return []

    model = load_model()
    available_words = [word for word, _ in grid if word not in guesses]
    if not available_words:
        return []

    encoded_words = model.encode(available_words, normalize_embeddings=True)
    clue_embedding = model.encode(clue_word, normalize_embeddings=True)
    similarity_scores = np.dot(encoded_words, clue_embedding)

    if len(available_words) == 1:
        return [{
            "word": available_words[0],
            "similarity": float(similarity_scores[0]),
            "cluster_id": 0,
        }]

    n_clusters = min(max(2, len(available_words) // 4), len(available_words) - 1)
    try:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")

    cluster_labels = clustering.fit_predict(encoded_words)
    best_cluster_id = None
    best_cluster_score = float("-inf")

    for cluster_id in sorted(set(cluster_labels)):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_score = float(np.mean([similarity_scores[idx] for idx in cluster_indices]))
        if cluster_score > best_cluster_score:
            best_cluster_score = cluster_score
            best_cluster_id = cluster_id

    ranked_words = []
    for idx, word in enumerate(available_words):
        if cluster_labels[idx] != best_cluster_id:
            continue
        ranked_words.append({
            "word": word,
            "similarity": float(similarity_scores[idx]),
            "cluster_id": int(cluster_labels[idx]),
        })

    ranked_words.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked_words[: min(6, len(ranked_words))]

def find_group_clue(grid, guesses, current_team, excluded_clues=None):
    excluded_clues = set(excluded_clues or [])
    team_words = {word for word, color in grid if color == current_team and word not in guesses}
    board_words = {word for word, _ in grid}
    opponent_words = board_words - team_words - set(guesses)
    best_match = None

    for clue_word, grouped_words in WORD_GROUPS.items():
        if clue_word in excluded_clues or clue_word in board_words:
            continue

        team_matches = team_words & grouped_words
        opponent_matches = opponent_words & grouped_words

        if len(team_matches) < 2 or opponent_matches:
            continue

        candidate = (len(team_matches), clue_word, sorted(team_matches))
        if best_match is None or candidate[0] > best_match[0]:
            best_match = candidate

    if best_match:
        count, clue_word, _ = best_match
        return (clue_word, count)

    return None

def generate_ai_clue(grid, guesses, current_team, excluded_clues=None):
    grouped_clue = find_group_clue(grid, guesses, current_team, excluded_clues=excluded_clues)
    if grouped_clue:
        return grouped_clue

    model = load_model()
    clue_embeddings = load_clue_embeddings()
    board_words = {word for word, _ in grid}
    excluded_clues = set(excluded_clues or [])

    team_words = [word for word, color in grid if color == current_team and word not in guesses]
    opponent_words = [word for word, color in grid if color not in [current_team] and word not in guesses]

    if not team_words:
        return ("victoire", 1)

    team_embeddings = {
        word: model.encode(word, normalize_embeddings=True)
        for word in team_words
    }
    opponent_embeddings = {
        word: model.encode(word, normalize_embeddings=True)
        for word in opponent_words
    }

    scored_candidates = []
    max_targets = min(4, len(team_words))

    for candidate, candidate_embedding in clue_embeddings.items():
        if candidate in board_words or candidate in guesses or candidate in excluded_clues:
            continue

        target_scores = sorted(
            [
                (word, float(np.dot(candidate_embedding, team_embeddings[word])))
                for word in team_words
            ],
            reverse=True,
            key=lambda item: item[1],
        )
        risk_scores = [float(np.dot(candidate_embedding, emb)) for emb in opponent_embeddings.values()]
        max_risk = max(risk_scores) if risk_scores else 0.0

        for target_count in range(max_targets, 0, -1):
            selected_pairs = target_scores[:target_count]
            if len(selected_pairs) < target_count:
                continue

            selected_scores = [score for _, score in selected_pairs]
            mean_score = float(np.mean(selected_scores))
            min_score = min(selected_scores)
            score = (
                mean_score
                + 0.08 * (target_count - 1)
                - 1.10 * max_risk
                - 0.03 * abs(target_count - 2)
            )

            if min_score < 0.16:
                continue
            if max_risk >= min_score + 0.03:
                continue
            if mean_score < 0.22 and target_count > 1:
                continue

            scored_candidates.append((score, candidate, target_count))

    if scored_candidates:
        scored_candidates.sort(reverse=True, key=lambda item: item[0])
        top_candidates = scored_candidates[: min(5, len(scored_candidates))]
        _, candidate, target_count = random.choice(top_candidates)
        return (candidate, target_count)

    fallback_candidate = next(
        (
            candidate
            for candidate in CLUE_CANDIDATES
            if candidate not in board_words and candidate not in guesses and candidate not in excluded_clues
        ),
        "association",
    )
    return (fallback_candidate, 1)

def main():
    st.title("🎲 Codenames - Mode Maître du Jeu")
    st.markdown("""
    **Règles :**
    - Le **maître du jeu** voit les couleurs et entre les indices.
    - Les **joueurs** devinent les mots sans voir les couleurs.
    - L'équipe qui devine tous ses mots gagne.
    """)

    # Initialisation
    if "grid" not in st.session_state:
        st.session_state.grid = generate_grid()
        st.session_state.current_team = "red"
        st.session_state.guesses = []
        st.session_state.scores = {"red": 0, "blue": 0}
        st.session_state.game_over = False
        st.session_state.clue = ("", 1)  # (mot, nombre ≥ 1)
        st.session_state.show_master = False
        st.session_state.turn_correct_guesses = 0
        st.session_state.lamp_hints = {"red": [], "blue": []}
    if "turn_correct_guesses" not in st.session_state:
        st.session_state.turn_correct_guesses = 0
    if "lamp_hints" not in st.session_state:
        st.session_state.lamp_hints = {"red": [], "blue": []}

    grid = st.session_state.grid
    current_team = st.session_state.current_team
    guesses = st.session_state.guesses
    scores = st.session_state.scores
    game_over = st.session_state.game_over
    clue = st.session_state.clue
    turn_correct_guesses = st.session_state.turn_correct_guesses
    lamp_hints = st.session_state.lamp_hints
    safe_clue_num = max(1, min(5, int(clue[1]) if isinstance(clue[1], (int, float)) else 1))
    if safe_clue_num != clue[1]:
        st.session_state.clue = (clue[0], safe_clue_num)
        clue = st.session_state.clue
        safe_clue_num = clue[1]

    if not game_over and not clue[0]:
        generated_clue = generate_ai_clue(grid, guesses, current_team)
        st.session_state.clue = generated_clue
        st.session_state.turn_correct_guesses = 0
        st.session_state.lamp_hints = {"red": [], "blue": []}
        clue = generated_clue
        safe_clue_num = clue[1]

    # --- Vue Maître du Jeu (cachée) ---
    if st.checkbox("👁️ Afficher la vue Maître du Jeu", value=st.session_state.show_master):
        st.session_state.show_master = True
        st.subheader("🎯 Vue Maître du Jeu (COULEURS VISIBLES)")
        cols = st.columns(5)
        for i, (word, color) in enumerate(grid):
            color_map = {"red": "#FFCCCB", "blue": "#ADD8E6", "neutral": "#D3D3D3", "black": "#000000"}
            bg_color = color_map.get(color, "#FFFFFF")
            text_color = "#FFFFFF" if color == "black" else "#000000"
            with cols[i % 5]:
                st.markdown(f"""
                <div style='background-color:{bg_color}; color:{text_color}; padding:10px;
                            border-radius:5px; margin:5px; text-align:center;'>
                <b>{word}</b><br><small>{color}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:16px; margin-top:12px;">
                <div style="font-size:0.95rem; color:#6b7280; margin-bottom:6px;">Indice généré automatiquement</div>
                <div style="font-size:1.4rem; font-weight:800; color:#111827;">{clue[0]} <span style="color:#6b7280;">({safe_clue_num})</span></div>
                <div style="font-size:0.95rem; color:#6b7280; margin-top:6px;">L'IA joue le rôle du maître du jeu pour l'équipe {current_team.upper()}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Régénérer l'indice IA"):
            st.session_state.clue = generate_ai_clue(
                grid,
                guesses,
                current_team,
                excluded_clues={clue[0]},
            )
            st.session_state.turn_correct_guesses = 0
            st.session_state.lamp_hints[current_team] = []
            st.rerun()

    # --- Vue Joueurs ---
    st.subheader("👥 Vue Joueurs (COULEURS CACHÉES)")
    cols = st.columns(5)
    for i, (word, color) in enumerate(grid):
        if word in guesses:
            display = "✅" if color in ["red", "blue"] else "❌" if color == "black" else "⚪"
            bg_color = "#E0E0E0"
        else:
            display = word
            bg_color = "#FFFFFF"
        with cols[i % 5]:
            st.markdown(f"""
            <div style='background-color:{bg_color}; padding:10px;
                        border-radius:5px; margin:5px; text-align:center;'>
            <b>{display}</b>
            </div>
            """, unsafe_allow_html=True)

    # Affichage des infos
    team_colors = {
        "red": {"bg": "#fff1f1", "accent": "#d62828", "border": "#f3b1b1"},
        "blue": {"bg": "#eef5ff", "accent": "#1d4ed8", "border": "#b6cdfa"},
    }
    active_theme = team_colors[current_team]
    clue_label = clue[0] if clue[0] else "En attente"
    progress_percent = int((turn_correct_guesses / safe_clue_num) * 100) if safe_clue_num else 0

    st.markdown(
        f"""
        <div style="padding: 18px; margin: 12px 0 20px 0; border-radius: 18px;
                    background: linear-gradient(135deg, {active_theme['bg']} 0%, #ffffff 100%);
                    border: 1px solid {active_theme['border']}; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);">
            <div style="display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px;">
                <div style="background: white; border-radius: 14px; padding: 16px; border: 1px solid #e5e7eb;">
                    <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 8px;">📊 Score</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #111827;">Rouge {scores['red']} <span style="color:#9ca3af;">|</span> Bleu {scores['blue']}</div>
                </div>
                <div style="background: white; border-radius: 14px; padding: 16px; border: 1px solid {active_theme['border']};">
                    <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 8px;">🔄 Tour en cours</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: {active_theme['accent']}; text-transform: uppercase;">Équipe {current_team}</div>
                </div>
                <div style="background: white; border-radius: 14px; padding: 16px; border: 1px solid #e5e7eb;">
                    <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 8px;">🔍 Indice du maître</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #111827;">{clue_label}</div>
                    <div style="font-size: 0.95rem; color: #6b7280; margin-top: 4px;">{safe_clue_num} mot(s) à trouver</div>
                </div>
                <div style="background: white; border-radius: 14px; padding: 16px; border: 1px solid #e5e7eb;">
                    <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 8px;">🎯 Progression</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #111827;">{turn_correct_guesses}/{safe_clue_num}</div>
                    <div style="margin-top: 10px; height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
                        <div style="width: {progress_percent}%; height: 100%; background: {active_theme['accent']}; border-radius: 999px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lamp_col_red, lamp_col_blue = st.columns(2)
    with lamp_col_red:
        if st.button("💡 Lampe Rouge", use_container_width=True, disabled=not clue[0]):
            st.session_state.lamp_hints["red"] = compute_lamp_hint(grid, guesses, clue[0])
            st.rerun()
    with lamp_col_blue:
        if st.button("💡 Lampe Bleue", use_container_width=True, disabled=not clue[0]):
            st.session_state.lamp_hints["blue"] = compute_lamp_hint(grid, guesses, clue[0])
            st.rerun()

    active_lamp_hints = lamp_hints.get(current_team, [])
    if active_lamp_hints:
        hint_cards = "".join(
            [
                f"""
                <div style="background:white; border:1px solid {active_theme['border']}; border-radius:12px;
                            padding:12px; text-align:center;">
                    <div style="font-size:1.05rem; font-weight:800; color:#111827;">{item['word']}</div>
                    <div style="font-size:0.85rem; color:{active_theme['accent']}; margin-top:4px;">
                        similarité {item['similarity']:.2f}
                    </div>
                </div>
                """
                for item in active_lamp_hints
            ]
        )
        st.markdown(
            f"""
            <div style="margin: 8px 0 18px 0; padding: 16px; border-radius: 16px;
                        background: #ffffff; border: 1px solid {active_theme['border']};">
                <div style="font-size: 1rem; font-weight: 800; color: {active_theme['accent']}; margin-bottom: 10px;">
                    💡 Aide par clustering pour l'équipe {current_team.upper()}
                </div>
                <div style="font-size: 0.92rem; color: #6b7280; margin-bottom: 12px;">
                    Mots du cluster le plus proche de l'indice IA « {clue[0]} ».
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px;">
                    {hint_cards}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Formulaire pour les devinettes (avec bouton)
    if not game_over:
        with st.form("guess_form"):
            guess = st.text_input("Devinez un mot:", key="guess_input").strip().lower()
            submitted = st.form_submit_button("Valider la devinette")
            if submitted and guess:
                for word, color in grid:
                    if word.lower() == guess:
                        if word in guesses:
                            st.warning("⚠️ Déjà deviné!")
                            break
                        guesses.append(word)
                        if color == "black":
                            st.error(f"💀 ASSASSIN! Game Over.")
                            st.session_state.game_over = True
                            st.session_state.turn_correct_guesses = 0
                            st.session_state.lamp_hints = {"red": [], "blue": []}
                        elif color == current_team:
                            st.success(f"✅ +1 pour {current_team}!")
                            st.session_state.scores[current_team] += 1
                            st.session_state.turn_correct_guesses += 1
                            if st.session_state.turn_correct_guesses >= safe_clue_num:
                                st.info("Nombre de réponses atteint. Tour adverse.")
                                st.session_state.current_team = switch_team(current_team)
                                st.session_state.turn_correct_guesses = 0
                                st.session_state.clue = ("", 1)
                                st.session_state.lamp_hints = {"red": [], "blue": []}
                        else:
                            st.info(f"⚪ Mot {color}. Tour adverse.")
                            if color in ["red", "blue"]:
                                st.session_state.scores[color] += 1
                            st.session_state.current_team = switch_team(current_team)
                            st.session_state.turn_correct_guesses = 0
                            st.session_state.clue = ("", 1)
                            st.session_state.lamp_hints = {"red": [], "blue": []}
                        st.rerun()
                        break
                else:
                    st.error("❌ Mot introuvable.")

    # Vérification de victoire
    red_left = sum(1 for w, c in grid if c == "red" and w not in guesses)
    blue_left = sum(1 for w, c in grid if c == "blue" and w not in guesses)
    if red_left == 0:
        st.balloons()
        st.success("🎉 Équipe ROUGE gagne!")
        st.session_state.game_over = True
    if blue_left == 0:
        st.balloons()
        st.success("🎉 Équipe BLEUE gagne!")
        st.session_state.game_over = True

    # Bouton pour recommencer
    if st.button("🔄 Nouvelle Partie"):
        st.session_state.grid = generate_grid()
        st.session_state.current_team = "red"
        st.session_state.guesses = []
        st.session_state.scores = {"red": 0, "blue": 0}
        st.session_state.game_over = False
        st.session_state.clue = ("", 1)
        st.session_state.turn_correct_guesses = 0
        st.session_state.lamp_hints = {"red": [], "blue": []}
        st.rerun()

if __name__ == "__main__":
    main()
