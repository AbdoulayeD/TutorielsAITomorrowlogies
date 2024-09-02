import streamlit as st
from openai import OpenAI

# Configuration de l'API OpenAI avec votre clé API
openai_api_key = "sk-XXXXXXXXX"  # Remplacez par votre clé API OpenAI
st.title("🍳 Générateur de Recettes de Cuisine")

# Utiliser la barre latérale pour entrer les ingrédients
st.sidebar.header("📝 Entrée des Ingrédients")
ingredients = st.sidebar.text_input(
    "Entrez les ingrédients disponibles (séparés par des virgules) :",
    placeholder="ex: tomates, basilic, ail, huile d'olive"
)

# Initialisation de l'état de la session si ce n'est pas déjà fait
if "recipes" not in st.session_state:
    st.session_state["recipes"] = []

# Bouton pour générer une recette
if st.sidebar.button("Générer une recette"):
    client = OpenAI(api_key=openai_api_key)
    
    # Créer le prompt avec les ingrédients fournis
    recipe_prompt = f"Vous êtes un chef cuisinier expert. Créez une recette en utilisant les ingrédients suivants : {ingredients}. Donner un nom à ces recettes. "
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": recipe_prompt}])
    recipe = response.choices[0].message.content
    
    # Ajouter la recette générée à l'état de la session
    st.session_state["recipes"].append(recipe)
    st.markdown(recipe)  # Affiche la recette formatée en Markdown

    # Générer le fichier texte et fournir une option de téléchargement
    st.download_button(
        label="Télécharger la recette en fichier texte",
        data=recipe,
        file_name="recette.txt",
        mime="text/plain"
    )
