import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

# Load dataset
df = pd.read_csv('/Users/camille/repo/Hetic/repo_M2/loan_project/data/application_train.csv')

# Function to convert age representation
def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive / 365
    return age_years

# Apply age conversion to DAYS_BIRTH and DAYS_EMPLOYED columns
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(convert_age)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(convert_age)

# Select features
used_features = [
    'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]

# Create reduced dataframe with selected columns
reduced_df = df[used_features].copy()

# Function to preprocess data
def preprocess_data(df):
    columns_to_one_hot = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

    # One-hot encode categorical columns
    for column in columns_to_one_hot:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df = df.drop([column], axis=1)

    # Label encode binary categorical columns
    binary_encoder = LabelEncoder()
    df['NAME_CONTRACT_TYPE'] = binary_encoder.fit_transform(df['NAME_CONTRACT_TYPE'])
    df['FLAG_OWN_CAR'] = binary_encoder.fit_transform(df['FLAG_OWN_CAR'])
    df['FLAG_OWN_REALTY'] = binary_encoder.fit_transform(df['FLAG_OWN_REALTY'])

    # Fill missing values with mean
    df = df.fillna(df.mean())

    return df

# Preprocess data
processed_df = preprocess_data(reduced_df)

# Split data into train and test sets
X = processed_df.drop('TARGET', axis=1)
y = processed_df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LightGBM model
model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=22)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'lgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save column names used for scaling
with open('column_names.pkl', 'wb') as f:
    joblib.dump(X.columns.tolist(), f)
