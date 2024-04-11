import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_filename = 'lgb_model.pkl'
model = joblib.load(model_filename)

# Function to convert age
def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive / 365
    return age_years

# Function to preprocess input data
def preprocess_input(data):
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].apply(convert_age)
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(convert_age)
    
    # Perform one-hot encoding for categorical columns
    columns_to_one_hot = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
    for column in columns_to_one_hot:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    
    # Label encoding for binary categorical columns
    binary_encoder = {'NAME_CONTRACT_TYPE': {'Cash loans': 0, 'Revolving loans': 1},
                      'FLAG_OWN_CAR': {'N': 0, 'Y': 1},
                      'FLAG_OWN_REALTY': {'N': 0, 'Y': 1}}
    data.replace(binary_encoder, inplace=True)
    
    # Fill missing values with mean
    data.fillna(data.mean(), inplace=True)
    
    return data

# Streamlit app
def main():
    st.title('Loan Prediction App')
    
    # Collect user inputs
    st.sidebar.header('User Input')
    contract_type = st.sidebar.radio('Contract Type', ('Cash loans', 'Revolving loans'))
    gender = st.sidebar.radio('Gender', ('M', 'F'))
    car_ownership = st.sidebar.radio('Owns a Car?', ('No', 'Yes'))
    realty_ownership = st.sidebar.radio('Owns Realty?', ('No', 'Yes'))
    income = st.sidebar.number_input('Income', min_value=0)
    credit_amount = st.sidebar.number_input('Credit Amount', min_value=0)
    goods_price = st.sidebar.number_input('Goods Price', min_value=0)
    education = st.sidebar.selectbox('Education', ('Lower secondary', 'Secondary / secondary special', 
                                                   'Incomplete higher', 'Higher education', 'Academic degree'))
    employment_days = st.sidebar.number_input('Employment Days', min_value=-100000, max_value=0)
    family_members = st.sidebar.number_input('Family Members', min_value=0)
    ext_source_1 = st.sidebar.number_input('External Source 1', min_value=0.0, max_value=1.0)
    ext_source_2 = st.sidebar.number_input('External Source 2', min_value=0.0, max_value=1.0)
    ext_source_3 = st.sidebar.number_input('External Source 3', min_value=0.0, max_value=1.0)
    
    # Create a DataFrame from user input
    data = {
        'NAME_CONTRACT_TYPE': [contract_type],
        'CODE_GENDER': [gender],
        'FLAG_OWN_CAR': [car_ownership],
        'FLAG_OWN_REALTY': [realty_ownership],
        'AMT_INCOME_TOTAL': [income],
        'AMT_CREDIT': [credit_amount],
        'AMT_GOODS_PRICE': [goods_price],
        'NAME_EDUCATION_TYPE': [education],
        'DAYS_EMPLOYED': [employment_days],
        'CNT_FAM_MEMBERS': [family_members],
        'EXT_SOURCE_1': [ext_source_1],
        'EXT_SOURCE_2': [ext_source_2],
        'EXT_SOURCE_3': [ext_source_3]
    }
    user_df = pd.DataFrame(data)
    
    # Preprocess user input
    processed_user_df = preprocess_input(user_df)
    
    # Make predictions
    prediction = model.predict(processed_user_df)
    prediction_proba = model.predict_proba(processed_user_df)
    
    # Display prediction
    if prediction[0] == 0:
        st.write('### Loan Status: Approved')
        st.write(f"Probability of Default: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.write('### Loan Status: Rejected')
        st.write(f"Probability of Default: {prediction_proba[0][1]*100:.2f}%")

# Run the app
if __name__ == '__main__':
    main()
