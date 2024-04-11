import pandas as pd
import plotly.express as px

class LoanApprovalAnalyzer:
    def __init__(self, csv_filename):
        """
        Initialize LoanApprovalAnalyzer with the path to the CSV file containing loan data.

        Parameters:
        - csv_filename (str): Path to the CSV file.
        """
        self.csv_filename = csv_filename
        self.columns_to_analyze = [
            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT'
        ]

    def load_data(self) -> pd.DataFrame:
        """
        Load the loan data from the CSV file.

        Returns:
        - pd.DataFrame: Loaded DataFrame containing loan data.
        """
        try:
            return pd.read_csv(self.csv_filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{self.csv_filename}' not found.")

    def plot_loan_approval_stats(self, data: pd.DataFrame, column: str, target_column='TARGET', top_n=10) -> px.bar:
        """
        Plot loan approval statistics for a specific column using Plotly.

        Parameters:
        - data (pd.DataFrame): DataFrame containing loan data.
        - column (str): Column to analyze loan approval statistics.
        - target_column (str): Target column indicating loan approval status.
        - top_n (int): Number of top categories to display.

        Returns:
        - px.bar: Plotly bar chart object.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")

        # Calculate loan approval statistics by grouping data
        default_rates = data.groupby(column)[target_column].mean() * 100
        default_rates = default_rates.sort_values(ascending=False).head(top_n)

        # Create a Plotly bar chart for loan approval statistics
        fig = px.bar(default_rates.reset_index(), x=column, y=target_column,
                     labels={column: column, target_column: 'Percentage of Defaults (%)'},
                     title=f'Loan Approval Statistics by {column}',
                     color_discrete_sequence=['skyblue'])

        return fig

    def plot_loan_repayment_pie(self, data: pd.DataFrame) -> px.pie:
        """
        Plot a pie chart showing loan repayment status using Plotly.

        Parameters:
        - data (pd.DataFrame): DataFrame containing loan data.

        Returns:
        - px.pie: Plotly pie chart object.
        """
        # Calculate loan repayment counts
        repayment_counts = data['TARGET'].value_counts()

        # Create a Plotly pie chart for loan repayment status
        fig = px.pie(values=repayment_counts, names=['Will Repay', 'Will Not Repay'],
                     labels={'label': 'Loan Repayment Status'}, title='Loan Repayment Status')

        return fig
