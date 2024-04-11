import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained model, scaler, and column names
model = joblib.load('lgb_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('column_names.pkl', 'rb') as f:
    column_names = joblib.load(f)

# Function to preprocess user input
def preprocess_input(data):
    # Ensure all expected columns are present in user input data
    for col in column_names:
        if col not in data.columns:
            # If a column is missing, add it with a default value (e.g., NaN)
            data[col] = np.nan
    
    # Encode categorical features
    data_encoded = data.copy()
    categorical_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                         'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

    for col in categorical_cols:
        # Use label encoding for binary categorical columns
        if data_encoded[col].dtype == 'object':
            data_encoded[col] = data_encoded[col].astype('category').cat.codes
    
    # Fill missing values with mean (using mean from training data)
    data_encoded = data_encoded.fillna(data_encoded.mean())

    # Scale features using pre-loaded scaler
    data_scaled = scaler.transform(data_encoded[column_names])

    return data_scaled

# Streamlit app
def main():
    st.title('Loan Default Prediction')

    # Collect user input
    contract_type = st.selectbox('Contract Type', ['Cash loans', 'Revolving loans'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    own_car = st.selectbox('Owns Car', ['Yes', 'No'])
    own_realty = st.selectbox('Owns Realty', ['Yes', 'No'])
    income_type = st.selectbox('Income Type', ['Working', 'Pensioner', 'State servant', 'Commercial associate', 'Unemployed'])
    education = st.selectbox('Education', ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    income = st.number_input('Income', min_value=0, value=50000)
    credit_amount = st.number_input('Credit Amount', min_value=0, value=100000)
    goods_price = st.number_input('Goods Price', min_value=0, value=100000)
    
    # Create DataFrame from user inputs
    user_data = pd.DataFrame({
        'NAME_CONTRACT_TYPE': [contract_type],
        'CODE_GENDER': [gender],
        'FLAG_OWN_CAR': [own_car],
        'FLAG_OWN_REALTY': [own_realty],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_EDUCATION_TYPE': [education],
        'DAYS_BIRTH': [-age * 365],  # Convert age to DAYS_BIRTH format
        'AMT_INCOME_TOTAL': [income],
        'AMT_CREDIT': [credit_amount],
        'AMT_GOODS_PRICE': [goods_price]
    })

    # Prediction button
    if st.button('Predict Loan Default'):
        # Preprocess user input
        user_data_scaled = preprocess_input(user_data)

        # Make prediction
        prediction = model.predict_proba(user_data_scaled)[:, 1]

        # Display prediction result
        st.subheader('Prediction')
        if prediction > 0.5:
            st.write('This customer is likely to default.')
        else:
            st.write('This customer is likely to repay the loan.')

if __name__ == '__main__':
    main()
