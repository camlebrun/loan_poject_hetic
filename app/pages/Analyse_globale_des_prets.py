import os
import pandas as pd
import streamlit as st
import plotly.express as px

# Chargement des données
csv_file_path = os.path.join(os.getcwd(), 'data', 'application_train.csv')
loan_data = pd.read_csv(csv_file_path)

# Titre principal
st.title("Analyse des Prêts Immobiliers")

# Affichage de la répartition des types de contrat de prêt
contract_type_counts = loan_data['NAME_CONTRACT_TYPE'].value_counts()
fig_contract_type = px.pie(values=contract_type_counts.values,
                           names=contract_type_counts.index,
                           title="Répartition des Types de Contrat de Prêt")
st.plotly_chart(fig_contract_type)


# Histogramme pour l'âge des clients capables
st.subheader("Répartition d'âge des clients capables")
capable_days_birth = loan_data[loan_data['TARGET'] == 0]['DAYS_BIRTH'] / 365
fig_capable_clients_age = px.histogram(x=capable_days_birth,
                                       nbins=10,
                                       title="Âge des Clients Capables à la Demande de Prêt",
                                       labels={'x': 'Âge (années)', 'y': 'Nombre de Clients'},
                                       color_discrete_sequence=['green'])
fig_capable_clients_age.update_traces(hovertemplate="Âge: %{x}<br>Nombre de Clients: %{y}")
st.plotly_chart(fig_capable_clients_age)

# Histogramme pour l'âge des clients non capables
st.subheader("Répartition d'âge des clients non capables")
not_capable_days_birth = loan_data[loan_data['TARGET'] == 1]['DAYS_BIRTH'] / 365
fig_not_capable_clients_age = px.histogram(x=not_capable_days_birth,
                                           nbins=10,
                                           title="Âge des Clients Non Capables à la Demande de Prêt",
                                           labels={'x': 'Âge (années)', 'y': 'Nombre de Clients'},
                                           color_discrete_sequence=['red'])
fig_not_capable_clients_age.update_traces(hovertemplate="Âge: %{x}<br>Nombre de Clients: %{y}")
st.plotly_chart(fig_not_capable_clients_age)
