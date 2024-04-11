import streamlit as st
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import pandas as pd

# Load the saved model
model_filename = 'lgb_model.pkl'
model = joblib.load(model_filename)

# Function to convert categorical data into one-hot representation
columns_to_one_hot = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

def create_one_hot(reduced_df, columns):
    for column in columns:
        reduced_df = pd.concat([reduced_df, pd.get_dummies(reduced_df[column])], axis=1, join='inner')
        reduced_df = reduced_df.drop([column], axis=1)
    return reduced_df

# Label encoding for binary categorical columns
binary_encoder = LabelEncoder()

# Create a function to make predictions
def make_prediction(data):
    data = create_one_hot(data, columns_to_one_hot)
    data['NAME_CONTRACT_TYPE'] = binary_encoder.fit_transform(data['NAME_CONTRACT_TYPE'])
    data['FLAG_OWN_CAR'] = binary_encoder.fit_transform(data['FLAG_OWN_CAR'])
    data['FLAG_OWN_REALTY'] = binary_encoder.fit_transform(data['FLAG_OWN_REALTY'])
    data.fillna(data.mean(), inplace=True)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(1, -1))
    prediction = model.predict(data_scaled)
    return prediction

# Create a Streamlit app
st.title('Loan Prediction App')

# Create input fields for the user
st.subheader('Enter your information:')

contract_type = st.selectbox('Contract Type', ['Cash contracts', 'Revolving loans', 'Cash loans', 'Real estate loans'])
gender = st.selectbox('Gender', ['Male', 'Female'])
own_car = st.selectbox('Own Car', ['Yes', 'No'])
own_realty = st.selectbox('Own Realty', ['Yes', 'No'])
children = st.number_input('Number of Children')
income = st.number_input('Total Income')
credit = st.number_input('Credit Amount')
goods_price = st.number_input('Goods Price')
income_type = st.selectbox('Income Type', ['Working', 'Pensioner', 'Student', 'Commercial associate', 'State servant'])
education_type = st.selectbox('Education Type', ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree'])
age = st.number_input('Age')
employed_days = st.number_input('Days Employed')
family_members = st.number_input('Family Members')
ext_source_1 = st.number_input('Ext Source 1')
ext_source_2 = st.number_input('Ext Source 2')
ext_source_3 = st.number_input('Ext Source 3')

# When 'Predict' button is clicked, make a prediction and display the result
if st.button('Predict'):
    data = pd.DataFrame({
        'NAME_CONTRACT_TYPE': [contract_type],
        'CODE_GENDER': [gender],
        'FLAG_OWN_CAR': [own_car],
        'FLAG_OWN_REALTY': [own_realty],
        'CNT_CHILDREN': [children],
        'AMT_INCOME_TOTAL': [income],
        'AMT_CREDIT': [credit],
        'AMT_GOODS_PRICE': [goods_price],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_EDUCATION_TYPE': [education_type],
        'DAYS_BIRTH': [age],
        'DAYS_EMPLOYED': [employed_days],
        'CNT_FAM_MEMBERS': [family_members],
        'EXT_SOURCE_1': [ext_source_1],
        'EXT_SOURCE_2': [ext_source_2],
        'EXT_SOURCE_3': [ext_source_3]
    })

    prediction = make_prediction(data)

    if prediction[0] == 1:
        st.success('Loan is likely to be approved.')
    else:
        st.error('Loan is likely to be rejected.')
