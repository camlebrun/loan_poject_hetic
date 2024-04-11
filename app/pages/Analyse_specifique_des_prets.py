import os
import pandas as pd
import streamlit as st
import plotly.express as px
from scripts.BorrowerCharacteristicsAnalyzer import BorrowerCharacteristicsAnalyzer
from scripts.LoanApprovalAnalyzer import LoanApprovalAnalyzer

# Chargement des données de prêt à partir du fichier CSV
@st.cache_resource
def load_loan_data():
    csv_file_path = os.path.join(os.getcwd(), 'data', 'application_train.csv')
    return pd.read_csv(csv_file_path)

def main():
    st.title("Analyse de l'Approbation des Prêts")

    # Chargement des données de prêt
    loan_data = load_loan_data()

    # Sélection de la colonne pour l'analyse d'approbation de prêt
    selected_loan_column = st.selectbox("Sélectionnez une variable pour l'analyse d'approbation de prêt", loan_data.columns)

    # Affichage des statistiques d'approbation de prêt
    if st.button("Afficher les Statistiques d'Approbation de Prêt"):
        try:
            # Initialisation de l'analyseur LoanApprovalAnalyzer avec les données de prêt
            analyzer = LoanApprovalAnalyzer(loan_data)
            
            # Tracé et affichage des statistiques d'approbation de prêt
            fig_loan_approval_stats = analyzer.plot_loan_approval_stats(selected_loan_column)
            st.plotly_chart(fig_loan_approval_stats)

        except ValueError as e:
            st.error(f"Erreur lors de l'affichage des statistiques : {str(e)}")

if __name__ == "__main__":
    main()
