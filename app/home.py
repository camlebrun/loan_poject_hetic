import streamlit as st
from PIL import Image
st.set_page_config(page_title="Prédiction de Défaut de Prêt", page_icon=":bar_chart:", layout="wide")
def main():
    # Titre de la page d'accueil
    st.title("Bienvenue dans l'Application de Prédiction de Défaut de Prêt")

    # Description de l'application
    st.write("""
        Cette application permet de prédire la probabilité de défaut de prêt pour un client en fonction de ses caractéristiques.
        Utilisez les menus déroulants et les champs de saisie pour spécifier les informations du client et cliquez sur le bouton pour obtenir la prédiction.
    """)
if __name__ == "__main__":
    main()