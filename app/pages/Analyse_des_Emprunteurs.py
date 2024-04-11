import os
import pandas as pd
import streamlit as st
import plotly.express as px
from scripts.BorrowerCharacteristicsAnalyzer import BorrowerCharacteristicsAnalyzer

def analyser_caracteristiques_emprunteurs(data):
    st.title("Analyse des Caractéristiques des Emprunteurs")

    # Colonnes à analyser
    colonnes_a_analyser = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT'
    ]

    # Sélection des colonnes à analyser
    colonnes_selectionnees = st.multiselect("Sélectionnez les colonnes pour l'analyse", colonnes_a_analyser)

    if not colonnes_selectionnees:
        st.warning("Veuillez sélectionner au moins une colonne pour l'analyse.")
        return

    # Initialisation de l'analyseur avec les données de prêt
    analyseur = BorrowerCharacteristicsAnalyzer(data)

    # Afficher les meilleures et pires caractéristiques des emprunteurs
    if st.button("Afficher les Meilleures et Pires Caractéristiques des Emprunteurs"):
        meilleures_caracteristiques = analyseur.identify_best_borrower_characteristics(colonnes_selectionnees)
        pires_caracteristiques = analyseur.identify_worst_borrower_characteristics(colonnes_selectionnees)

        st.subheader("Meilleures Caractéristiques des Emprunteurs")
        st.write(pd.DataFrame(meilleures_caracteristiques))

        st.subheader("Pires Caractéristiques des Emprunteurs")
        st.write(pd.DataFrame(pires_caracteristiques))

    # Tracer les taux de défaut pour une colonne sélectionnée
    st.subheader("Tracer les Taux de Défaut")
    colonne_a_tracer = st.selectbox("Sélectionnez la colonne pour le tracé des taux de défaut", colonnes_selectionnees)

    if st.button("Tracer les Taux de Défaut"):
        try:
            fig_taux_default = analyseur.plot_default_rates(colonne_a_tracer)
            st.plotly_chart(fig_taux_default)
        except ValueError as e:
            st.error(f"Erreur lors du tracé des taux de défaut : {str(e)}")

def main():

    # Chargement des données de prêt
    chemin_fichier_csv = os.path.join(os.getcwd(), 'data', 'application_train.csv')
    donnees_emprunt = pd.read_csv(chemin_fichier_csv)

    # Affichage de l'analyse des caractéristiques des emprunteurs
    analyser_caracteristiques_emprunteurs(donnees_emprunt)

# Exécution de l'application Streamlit
if __name__ == "__main__":
    main()
