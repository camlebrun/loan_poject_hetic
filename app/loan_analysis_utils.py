import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_stats_categorical(df, var):
    """
    Plot loan approval statistics for categorical variables.

    Parameters:
    - df (pd.DataFrame): DataFrame containing loan data.
    - var (list): List of categorical variables to analyze.

    Displays bar plots to visualize loan approval statistics for each categorical variable.
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 6 * len(var)))

    for i, feature in enumerate(var, start=1):
        plt.subplot(len(var), 2, 2 * i - 1)
        sns.set_color_codes("pastel")
        sns.barplot(x=feature, y="Number of contracts", data=df)
        plt.xticks(rotation=45)
        plt.title(f"{feature} - Number of Contracts")

        plt.subplot(len(var), 2, 2 * i)
        sns.barplot(x=feature, y='TARGET', data=df)
        plt.xticks(rotation=45)
        plt.title(f"{feature} - Percent of target with value 1 [%]")
        plt.axhline(df['TARGET'].mean() * 100, color='green')  # Average target percentage

    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """
    Preprocess loan data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing loan data.

    Performs data preprocessing steps such as handling missing values, encoding categorical features, etc.
    Returns the preprocessed DataFrame.
    """
    # Example preprocessing steps (replace with actual preprocessing logic)
    df.fillna(0, inplace=True)  # Replace missing values with 0
    df_encoded = pd.get_dummies(df)  # One-hot encode categorical features
    return df_encoded
