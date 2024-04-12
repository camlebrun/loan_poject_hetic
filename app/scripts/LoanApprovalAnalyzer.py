import pandas as pd
import plotly.express as px

class LoanApprovalAnalyzer:
    def __init__(self, loan_data):
        """
        Initialize LoanApprovalAnalyzer with loan data.

        Parameters:
        - loan_data (pd.DataFrame): DataFrame containing loan data.
        """
        self.loan_data = loan_data
        self.columns_to_analyze = [
            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT'
        ]

    def plot_loan_approval_stats(self, column, target_column='TARGET', top_n=10):
        """
        Plot loan approval statistics for a specific column using Plotly.

        Parameters:
        - column (str): Column to analyze loan approval statistics.
        - target_column (str): Target column indicating loan approval status.
        - top_n (int): Number of top categories to display.

        Returns:
        - px.bar: Plotly bar chart object.
        """
        if column not in self.loan_data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")

        # Calculate loan approval statistics by grouping data
        default_rates = self.loan_data.groupby(column)[target_column].mean() * 100
        default_rates = default_rates.sort_values(ascending=False).head(top_n)

        # Create a Plotly bar chart for loan approval statistics
        fig = px.bar(default_rates.reset_index(), x=column, y=target_column,
                     labels={column: column, target_column: 'Percentage of Defaults (%)'},
                     title=f'Loan Approval Statistics by {column}',
                     color_discrete_sequence=['skyblue'])

        return fig

    def plot_loan_repayment_pie(self) -> px.pie:
        """
        Plot a pie chart showing loan repayment status using Plotly.

        Returns:
        - px.pie: Plotly pie chart object.
        """
        # Calculate loan repayment counts
        repayment_counts = self.loan_data['TARGET'].value_counts()

        # Create a Plotly pie chart for loan repayment status
        fig = px.pie(values=repayment_counts, names=['Will Repay', 'Will Not Repay'],
                     labels={'label': 'Loan Repayment Status'}, title='Loan Repayment Status')

        return fig
