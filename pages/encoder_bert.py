import streamlit as st
from transformers import CamembertTokenizer, CamembertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Base de données des termes comptables
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

# Charger le modèle CamemBERT pour la similarité sémantique
@st.cache_resource
def load_model():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    return tokenizer, model

tokenizer, model = load_model()

# Fonction pour obtenir l'embedding d'un texte
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Fonction pour classer un terme par recherche exacte
def classer_terme_exact(mot, base_de_donnees):
    mot = mot.lower()
    for categorie, termes in base_de_donnees.items():
        for terme, details in termes.items():
            if mot in details["exemples"] or mot == terme:
                return {
                    "type": categorie,
                    "terme": terme,
                    "definition": details["definition"]
                }
    return None

# Fonction pour classer un terme par similarité sémantique
def classer_terme_par_similarite(mot, base_de_donnees, seuil=0.65):
    mot_embedding = get_embedding(mot)
    meilleures_similarites = []

    for categorie, termes in base_de_donnees.items():
        for terme, details in termes.items():
            for exemple in details["exemples"]:
                exemple_embedding = get_embedding(exemple)
                similarity = cosine_similarity([mot_embedding], [exemple_embedding])[0][0]
                meilleures_similarites.append((similarity, categorie, terme, details["definition"]))

    # Trier par similarité décroissante
    meilleures_similarites.sort(reverse=True, key=lambda x: x[0])

    # Retourner le meilleur match au-dessus du seuil
    if meilleures_similarites and meilleures_similarites[0][0] > seuil:
        return {
            "type": meilleures_similarites[0][1],
            "terme": meilleures_similarites[0][2],
            "definition": meilleures_similarites[0][3],
            "similarity": meilleures_similarites[0][0]
        }
    else:
        return None

# Interface Streamlit
st.title("📊 Classificateur de termes comptables")
st.markdown("""
Saisissez un terme comptable (ex. : "bâtiment", "dette fournisseur") pour savoir s'il s'agit d'un **actif** ou d'un **passif**.
""")

# Saisie utilisateur
mot_utilisateur = st.text_input("Entrez un terme comptable :", key="input_mot")

if st.button("Classer le terme"):
    if mot_utilisateur:
        # Essayer la classification exacte
        resultat_exact = classer_terme_exact(mot_utilisateur, comptabilite_db)

        if resultat_exact:
            st.success("✅ **Terme trouvé par recherche exacte**")
            st.write(f"**Type** : {resultat_exact['type'].upper()}")
            st.write(f"**Terme** : {resultat_exact['terme']}")
            st.write(f"**Définition** : {resultat_exact['definition']}")
        else:
            # Essayer la classification par similarité
            resultat_similaire = classer_terme_par_similarite(mot_utilisateur, comptabilite_db)

            if resultat_similaire:
                st.success("✅ **Terme trouvé par similarité sémantique**")
                st.write(f"**Type** : {resultat_similaire['type'].upper()}")
                st.write(f"**Terme** : {resultat_similaire['terme']}")
                st.write(f"**Définition** : {resultat_similaire['definition']}")
                st.write(f"**Similarité** : {resultat_similaire['similarity']:.2f}")
            else:
                st.error("❌ **Terme non trouvé** dans la base de données.")
    else:
        st.warning("Veuillez saisir un terme comptable.")

# Afficher la base de données (optionnel)
if st.checkbox("Afficher la base de données"):
    st.subheader("Base de données des termes comptables")
    for categorie, termes in comptabilite_db.items():
        st.write(f"**{categorie.upper()}**")
        for terme, details in termes.items():
            st.write(f"- **{terme}** : {details['definition']}")
            st.write(f"  *Exemples* : {', '.join(details['exemples'])}")
