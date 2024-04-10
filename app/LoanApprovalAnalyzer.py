import streamlit as st
import pandas as pd
import plotly.express as px

class LoanApprovalAnalyzer:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.columns_to_analyze = [
            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT'
        ]

    @st.cache
    def load_data(self):
        return pd.read_csv(self.csv_filename)

    def plot_loan_approval_stats(self, data, column, target_column='TARGET', top_n=10):
        if column not in data.columns:
            st.error(f"Column '{column}' not found in the dataset.")
            return

        try:
            grouped_data = data.groupby(column)[target_column].agg(['sum', 'count', 'mean']).reset_index()
            grouped_data.columns = [column, 'Defaulters', 'Total', 'Defaulter Rate']
            grouped_data.sort_values(by='Total', ascending=False, inplace=True)

            fig = px.bar(grouped_data.head(top_n), x=column, y='Total',
                         title=f'Loan Approval Statistics by {column}',
                         labels={column: column, 'Total': 'Number of Loans'})

            fig.update_layout(xaxis_tickangle=-45, xaxis_title=column, yaxis_title='Number of Loans')

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred for column '{column}': {e}")

    def run(self):
        st.title("Loan Approval Analysis")
        data = self.load_data()

        selected_column = st.sidebar.selectbox("Select a column to analyze", self.columns_to_analyze)

        if st.sidebar.button("Analyze"):
            st.subheader(f"Loan Approval Statistics by {selected_column}")
            self.plot_loan_approval_stats(data, selected_column, target_column='TARGET', top_n=10)
