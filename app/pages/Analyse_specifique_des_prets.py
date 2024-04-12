import pandas as pd
import streamlit as st
import plotly.express as px
from scripts.LoanApprovalAnalyzer import LoanApprovalAnalyzer  # Adjust import based on your package structure

@st.cache
def load_loan_data():
    csv_file_path = "data/application_train.csv"
    return pd.read_csv(csv_file_path)

def main():
    st.title("Loan Approval Analysis")

    # Load loan data
    loan_data = load_loan_data()

    # Select loan column for analysis
    selected_loan_column = st.selectbox("Select a variable for loan approval analysis", loan_data.columns)

    # Display loan approval statistics
    if st.button("Show Loan Approval Statistics"):
        try:
            # Initialize LoanApprovalAnalyzer with loan data
            analyzer = LoanApprovalAnalyzer(loan_data)

            # Plot and display loan approval statistics
            fig_loan_approval_stats = analyzer.plot_loan_approval_stats(column=selected_loan_column)
            st.plotly_chart(fig_loan_approval_stats)

        except ValueError as e:
            st.error(f"Error displaying statistics: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
