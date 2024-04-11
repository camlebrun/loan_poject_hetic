import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Chargement du modèle pré-entraîné, du scaler et des noms de colonnes
model = joblib.load('model/lgb_model.pkl')
scaler = joblib.load('model/scaler.pkl')
with open('model/column_names.pkl', 'rb') as f:
    column_names = joblib.load(f)

# Fonction pour prétraiter les données utilisateur
def preprocess_input(data):
    # Assurer la présence de toutes les colonnes attendues dans les données utilisateur
    for col in column_names:
        if col not in data.columns:
            # Si une colonne est manquante, l'ajouter avec une valeur par défaut (par exemple, NaN)
            data[col] = np.nan
    
    # Encodage des caractéristiques catégorielles
    data_encode = data.copy()
    categorical_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                         'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

    for col in categorical_cols:
        # Utilisation de l'encodage label pour les colonnes catégorielles binaires
        if data_encode[col].dtype == 'object':
            data_encode[col] = data_encode[col].astype('category').cat.codes
    
    # Remplissage des valeurs manquantes avec la moyenne (utilisation de la moyenne des données d'entraînement)
    data_encode = data_encode.fillna(data_encode.mean())

    # Mise à l'échelle des caractéristiques à l'aide du scaler pré-chargé
    data_scaled = scaler.transform(data_encode[column_names])

    return data_scaled

# Application Streamlit
def main():
    st.title('Prédiction de Défaut de Prêt')

    # Collecte des données utilisateur
    type_contrat = st.selectbox('Type de Contrat', ['Prêts personnels', 'Crédits renouvelables'])
    genre = st.selectbox('Genre', ['Homme', 'Femme'])
    possede_voiture = st.selectbox('Possède une Voiture', ['Oui', 'Non'])
    possede_immobilier = st.selectbox('Possède un Bien Immobilier', ['Oui', 'Non'])
    type_revenu = st.selectbox('Type de Revenu', ['Travailleur', 'Retraité', 'Fonctionnaire', 'Associé commercial', 'Sans emploi'])
    education = st.selectbox('Niveau d\'Éducation', ['Secondaire / spécial secondaire', 'Enseignement supérieur', 'Enseignement incomplet supérieur', 'Secondaire inférieur', 'Diplôme universitaire'])
    age = st.slider('Âge', min_value=18, max_value=100, value=30)
    revenu = st.number_input('Revenu', min_value=0, value=50000)
    montant_credit = st.number_input('Montant du Crédit', min_value=0, value=100000)
    prix_bien = st.number_input('Prix du Bien', min_value=0, value=100000)
    
    # Création du DataFrame à partir des saisies utilisateur
    donnees_utilisateur = pd.DataFrame({
        'NAME_CONTRACT_TYPE': [type_contrat],
        'CODE_GENDER': [genre],
        'FLAG_OWN_CAR': [possede_voiture],
        'FLAG_OWN_REALTY': [possede_immobilier],
        'NAME_INCOME_TYPE': [type_revenu],
        'NAME_EDUCATION_TYPE': [education],
        'DAYS_BIRTH': [-age * 365],  # Conversion de l'âge au format DAYS_BIRTH
        'AMT_INCOME_TOTAL': [revenu],
        'AMT_CREDIT': [montant_credit],
        'AMT_GOODS_PRICE': [prix_bien]
    })

    # Bouton de prédiction
    if st.button('Prédire le Défaut de Prêt'):
        # Prétraitement des données utilisateur
        donnees_utilisateur_pretraitees = preprocess_input(donnees_utilisateur)

        # Prédiction
        prediction = model.predict_proba(donnees_utilisateur_pretraitees)[:, 1]

        # Affichage du résultat de prédiction
        st.subheader('Prédiction')
        if prediction > 0.5:
            st.write('Ce client est susceptible de rembourser le prêt.')
        else:
            st.write('Ce client est susceptible de faire défaut.')


        # Affichage des détails
        st.write('Probabilité de Défaut:', prediction[0])
        st.write(f"Type de Contrat: {type_contrat}")
        st.write(f"Genre: {genre}")
        st.write(f"Possède une Voiture: {possede_voiture}")
        st.write(f"Possède un Bien Immobilier: {possede_immobilier}")
        st.write(f"Type de Revenu: {type_revenu}")
        st.write(f"Niveau d'Éducation: {education}")
        st.write(f"Âge: {age}")
        st.write(f"Revenu: {revenu}")
        st.write(f"Montant du Crédit: {montant_credit}")
        st.write(f"Prix du Bien: {prix_bien}")

if __name__ == '__main__':
    main()
