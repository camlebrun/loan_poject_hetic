import pandas as pd
import numpy as np

def fix_nulls_outliers(data):
    # Fill missing values for categorical variables with 'Data_Not_Available'
    categorical_cols = ['NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 
                        'FLAG_EMP_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 
                        'OCCUPATION_TYPE', 'NAME_TYPE_SUITE']
    data[categorical_cols] = data[categorical_cols].fillna('Data_Not_Available')

    # Fill missing values in 'CNT_FAM_MEMBERS' with the most frequent value
    data['CNT_FAM_MEMBERS'].fillna(data['CNT_FAM_MEMBERS'].mode().iloc[0], inplace=True)

    # Replace extreme value in 'DAYS_EMPLOYED' with NaN (assuming it's an outlier)
    data['DAYS_EMPLOYED'].replace(max(data['DAYS_EMPLOYED'].values), np.nan, inplace=True)

    # Replace 'XNA' in 'CODE_GENDER' with 'M' (assuming 'XNA' represents male)
    data['CODE_GENDER'].replace('XNA', 'M', inplace=True)

    # Fill missing values in 'AMT_ANNUITY' and 'AMT_GOODS_PRICE' with 0
    data['AMT_ANNUITY'].fillna(0, inplace=True)
    data['AMT_GOODS_PRICE'].fillna(0, inplace=True)

    # Replace 'Unknown' in 'NAME_FAMILY_STATUS' with 'Married'
    data['NAME_FAMILY_STATUS'].replace('Unknown', 'Married', inplace=True)

    # Fill missing values in external sources with 0
    external_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    data[external_sources] = data[external_sources].fillna(0)

    return data

data = pd.read_csv('data/application_train.csv')

cleaned_data = fix_nulls_outliers(data)

# Save the cleaned data to a new CSV file
output_file_path = 'data/application_train.csv'
cleaned_data.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to: {output_file_path}")
