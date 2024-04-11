# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import joblib

# Loading dataset
df = pd.read_csv('/Users/camille/repo/Hetic/repo_M2/loan_project/data/application_train.csv')

# Function to convert age representation
def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive / 365
    return age_years

# Applying convert_age() function to the data frame
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(convert_age)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(convert_age)

# Features to use for training the model
used_features = [
    'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]

# Create a new data frame with selected columns
reduced_df = df[used_features]

# Function to convert categorical data into one-hot representation
columns_to_one_hot = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

def create_one_hot(reduced_df, columns):
    for column in columns:
        reduced_df = pd.concat([reduced_df, pd.get_dummies(df[column])], axis=1, join='inner')
        reduced_df = reduced_df.drop([column], axis=1)
    return reduced_df

reduced_df = create_one_hot(reduced_df, columns_to_one_hot)

# Label encoding for binary categorical columns
binary_encoder = LabelEncoder()
reduced_df['NAME_CONTRACT_TYPE'] = binary_encoder.fit_transform(reduced_df['NAME_CONTRACT_TYPE'])
reduced_df['FLAG_OWN_CAR'] = binary_encoder.fit_transform(reduced_df['FLAG_OWN_CAR'])
reduced_df['FLAG_OWN_REALTY'] = binary_encoder.fit_transform(reduced_df['FLAG_OWN_REALTY'])

# Handling missing values by filling with mean
reduced_df.fillna(reduced_df.mean(), inplace=True)

# Splitting the data into train/test
X = reduced_df.drop('TARGET', axis=1)
y = reduced_df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Value normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing LightGBM classifier
model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=22)

# Training the LightGBM model without the 'verbose' argument
model.fit(X_train_scaled, y_train, eval_metric='auc', 
          eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)])

# Save the trained model to a file
model_filename = 'lgb_model.pkl'
joblib.dump(model, model_filename)
print("Model saved successfully as", model_filename)

# Predict probability scores
prob_train = model.predict_proba(X_train_scaled)
prob_test = model.predict_proba(X_test_scaled)

# Calculate ROC curve metrics
fpr_train, tpr_train, _ = roc_curve(y_train, prob_train[:, 1])
fpr_test, tpr_test, _ = roc_curve(y_test, prob_test[:, 1])

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr_train, tpr_train, label='Train')
plt.plot(fpr_test, tpr_test, label='Test')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Calculate AUC score
auc_score_train = roc_auc_score(y_train, prob_train[:, 1])
auc_score_test = roc_auc_score(y_test, prob_test[:, 1])
print("Train AUC:", auc_score_train)
print("Test AUC:", auc_score_test)

# Additional evaluation: Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Confusion matrix on train data
cm_train = confusion_matrix(y_train, model.predict(X_train_scaled))
plot_confusion_matrix(cm_train, 'Confusion Matrix (Train Data)')

# Confusion matrix on test data
cm_test = confusion_matrix(y_test, model.predict(X_test_scaled))
plot_confusion_matrix(cm_test, 'Confusion Matrix (Test Data)')
