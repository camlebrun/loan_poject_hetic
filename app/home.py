import streamlit as st
import pandas as pd
import plotly.express as px
from BorrowerCharacteristicsAnalyzer import BorrowerCharacteristicsAnalyzer
from LoanApprovalAnalyzer import LoanApprovalAnalyzer

def main():
    st.title("Loan Approval Back Office")

    # Sidebar for selecting analysis type
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Loan Approval Analysis", "Borrower Characteristics Analysis"])

    if analysis_type == "Loan Approval Analysis":
        st.header("Loan Approval Analysis")

        # Hardcoded path to the CSV file for Loan Approval Analysis
        loan_csv_filename = "/Users/camille/repo/Hetic/repo_M2/project_pred_mocel/cache/application_train.csv"

        # Initialize LoanApprovalAnalyzer with loan data for loan approval analysis
        loan_analyzer = LoanApprovalAnalyzer(loan_csv_filename)

        # Load loan data for loan approval analysis
        loan_data = loan_analyzer.load_data()

        # Select column for loan approval analysis
        loan_column_options = loan_data.columns.tolist()
        selected_loan_column = st.selectbox("Select a column for loan approval analysis", loan_column_options)

        # Display loan approval statistics
        if st.button("Show Loan Approval Statistics"):
            try:
                # Plot loan approval statistics
                fig = loan_analyzer.plot_loan_approval_stats(loan_data, selected_loan_column)
                st.plotly_chart(fig)
            except ValueError as e:
                st.error(str(e))

    elif analysis_type == "Borrower Characteristics Analysis":
        st.header("Borrower Characteristics Analysis")

        # Sample loan data (replace this with your actual loan data)
        data = pd.read_csv("/Users/camille/repo/Hetic/repo_M2/project_pred_mocel/cache/application_train.csv")

        # Initialize BorrowerCharacteristicsAnalyzer with loan data for borrower characteristics analysis
        analyzer = BorrowerCharacteristicsAnalyzer(data)

        # Select columns for characteristics analysis
        columns_to_analyze = st.multiselect("Select Columns for Analysis", data.columns)

        # Display best and worst borrower characteristics
        if st.button("Show Best and Worst Borrower Characteristics"):
            best_characteristics = analyzer.identify_best_borrower_characteristics(columns_to_analyze)
            worst_characteristics = analyzer.identify_worst_borrower_characteristics(columns_to_analyze)

            st.subheader("Best Borrower Characteristics")
            st.write(pd.DataFrame(best_characteristics))

            st.subheader("Worst Borrower Characteristics")
            st.write(pd.DataFrame(worst_characteristics))

        # Plot default rates for a selected column
        st.subheader("Plot Default Rates")
        column_to_plot = st.selectbox("Select Column for Default Rates Plot", columns_to_analyze)

        if st.button("Plot Default Rates"):
            try:
                fig = analyzer.plot_default_rates(column_to_plot)
                st.plotly_chart(fig)
            except ValueError as e:
                st.error(str(e))

# Run the Streamlit app
if __name__ == "__main__":
    main()
