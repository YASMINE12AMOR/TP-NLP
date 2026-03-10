import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral

# Configuration de la page
st.set_page_config(
    page_title="Mistral GPT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger les variables d'environnement
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "mistral-tiny"  # Mettre à jour selon le modèle disponible

if not api_key:
    st.error("⚠️ **MISTRAL_API_KEY introuvable** – Ajoutez-la dans le fichier `.env` à la racine du projet.")
    st.stop()

# Initialiser le client Mistral
@st.cache_resource(show_spinner="Chargement du modèle...")
def get_client():
    return Mistral(api_key=api_key)

client = get_client()

# Titre et description
st.title("🤖 **Mistral GPT Assistant**")
st.markdown("""
Bienvenue dans votre assistant IA polyvalent ! Sélectionnez une tâche dans le menu latéral,
puis utilisez un exemple ou saisissez votre propre requête.
""")

# Barre latérale pour la sélection des tâches
with st.sidebar:
    st.header("📌 **Tâches disponibles**")
    st.markdown("---")
    task = st.radio(
        "Choisissez une tâche :",
        ["💬 Chat / Questions-Réponses", "📄 Résumé de texte", "💻 Génération de code"],
        index=0,
        help="Sélectionnez le type d'assistance dont vous avez besoin."
    )

    st.markdown("---")
    st.subheader("✨ **Exemples**")
    st.markdown("Cliquez sur un exemple pour le charger automatiquement.")

# Définition des exemples pour chaque tâche
examples_by_task = {
    "💬 Chat / Questions-Réponses": {
        "description": """
        **Posez des questions ouvertes** sur des sujets variés (IA, programmation, conseils, etc.).
        Mistral répondra de manière claire et pédagogique.
        """,
        "examples": [
            "Explique simplement la différence entre NLP, NLU et NLG avec un exemple pour chacun.",
            "Je débute en IA. Quels sont les prérequis à apprendre en 30 jours ? Fais un plan hebdomadaire.",
            "Donne-moi 5 idées de projet NLP niveau débutant à intermédiaire, avec une estimation de difficulté."
        ]
    },
    "📄 Résumé de texte": {
        "description": """
        **Collez un texte long** et demandez un résumé structuré.
        Vous pouvez préciser le format (points clés, résumé exécutif, etc.).
        """,
        "examples": [
            "Résume ce texte en 5 points clés, puis ajoute une conclusion en 2 lignes.",
            "Fais un résumé exécutif (max 120 mots) avec : objectif, méthode, résultats, limites.",
            "Condense ce texte pour une présentation orale de 1 minute, en mettant l'accent sur les impacts."
        ]
    },
    "💻 Génération de code": {
        "description": """
        **Décrivez une fonctionnalité ou un algorithme** en langage naturel,
        et Mistral générera du code Python **exécutable et bien commenté**.
        """,
        "examples": [
            "Écris une fonction Python qui nettoie un texte (minuscules, ponctuation, stopwords) avec un exemple d'utilisation.",
            "Génère une classe PyTorch pour un modèle LSTM de classification de texte, avec une méthode `forward` claire et un exemple d'entraînement.",
            "Crée un script Streamlit minimal avec un champ de texte pour saisir du texte, un bouton pour analyser le sentiment (positif/négatif/neutre), et un affichage visuel du résultat."
        ]
    }
}

# Affichage des exemples avec des boutons élégants
st.subheader(f"🔍 **{examples_by_task[task]['description']}**")
st.markdown("---")

# Affichage des exemples sous forme de colonnes
cols = st.columns(1 if task == "💻 Génération de code" else 2)
for i, example in enumerate(examples_by_task[task]["examples"]):
    with cols[i % len(cols)]:
        if st.button(
            f"Exemple {i+1}",
            key=f"ex_{i}",
            use_container_width=True,
            help=f"Cliquez pour charger : {example[:50]}..."
        ):
            if task == "💬 Chat / Questions-Réponses":
                st.session_state["user_input"] = example
            elif task == "📄 Résumé de texte":
                st.session_state["sum_instruction"] = example
            elif task == "💻 Génération de code":
                st.session_state["code_request"] = example

# Zone de saisie adaptée à la tâche sélectionnée
st.markdown("---")
st.subheader("📝 **Votre requête**")

prompt = ""  # Initialiser prompt ici

if task == "💬 Chat / Questions-Réponses":
    user_input = st.text_area(
        "Posez votre question :",
        key="user_input",
        height=150,
        placeholder="Exemple : Comment démarrer un projet de NLP avec Python ?"
    )
    prompt = user_input.strip() if user_input else ""

elif task == "📄 Résumé de texte":
    col1, col2 = st.columns([1, 1])
    with col1:
        sum_instruction = st.text_area(
            "Instruction de résumé (optionnel) :",
            key="sum_instruction",
            height=100,
            placeholder="Exemple : Résume en 3 points clés pour un public non technique."
        )
    with col2:
        text_to_summarize = st.text_area(
            "Texte à résumer :",
            key="sum_text",
            height=200,
            placeholder="Collez ici le texte que vous souhaitez résumer..."
        )

    if text_to_summarize.strip():
        instruction = sum_instruction.strip() if sum_instruction else "Résume ce texte de manière claire et concise :"
        prompt = f"{instruction}\n\nTexte à résumer:\n{text_to_summarize.strip()}"

elif task == "💻 Génération de code":
    code_request = st.text_area(
        "Décrivez le code que vous souhaitez générer :",
        key="code_request",
        height=200,
        placeholder='Exemple : "Crée une fonction Python qui calcule la fréquence des mots dans un texte, en ignorant les stopwords."'
    )
    prompt = code_request.strip() if code_request else ""

# Bouton pour soumettre la requête
if prompt:
    if st.button(
        "🚀 **Envoyer la requête**",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Mistral réfléchit à votre demande..."):
            try:
                response = client.chat.complete(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Tu es un assistant expert en NLP et en développement. "
                                "Réponds de manière **claire, structurée et actionnable**. "
                                "Pour le code, fournis du **Python exécutable et bien commenté**. "
                                "Adapte ton style à la tâche demandée."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result = response.choices[0].message.content

                # Affichage du résultat
                st.markdown("---")
                st.subheader("🎯 **Résultat**")
                if task == "💻 Génération de code":
                    st.code(result, language="python")
                else:
                    st.markdown(result)
            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {str(e)}")
else:
    st.warning("Veuillez saisir une requête ou sélectionner un exemple.")

# Pied de page
st.markdown("---")
st.caption("""
**Conseils pour de meilleurs résultats** :
- Soyez précis dans vos requêtes.
- Pour le code, décrivez clairement la fonctionnalité souhaitée.
- Pour les résumés, précisez le format (ex: "en 3 points clés").
""")
