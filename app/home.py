import os
import pandas as pd
import streamlit as st
import plotly.express as px
from BorrowerCharacteristicsAnalyzer import BorrowerCharacteristicsAnalyzer
from LoanApprovalAnalyzer import LoanApprovalAnalyzer

def loan_approval_analysis(loan_data):
    st.subheader("Loan Approval Analysis")

    # Display contract type distribution
    contract_type_counts = loan_data['NAME_CONTRACT_TYPE'].value_counts()
    fig_contract_type = px.pie(values=contract_type_counts.values,
                               names=contract_type_counts.index,
                               title="Contract Type Distribution")
    st.plotly_chart(fig_contract_type)

    # Select column for loan approval statistics
    selected_loan_column = st.selectbox("Select a column for loan approval analysis", loan_data.columns)

    # Display loan approval statistics
    if st.button("Show Loan Approval Statistics"):
        try:
            # Plot loan approval statistics
            fig_loan_approval_stats = LoanApprovalAnalyzer.plot_loan_approval_stats(loan_data, selected_loan_column)
            st.plotly_chart(fig_loan_approval_stats)
        except ValueError as e:
            st.error(str(e))

    # Histogram for all clients
    st.subheader("Age Distribution of All Clients")
    fig_all_clients_age = px.histogram(x=loan_data['DAYS_BIRTH'] / -365,
                                       nbins=20,
                                       title='Age of Clients at Application',
                                       labels={'x': 'Age (years)', 'y': 'Number of Clients'},
                                       color_discrete_sequence=['blue'])
    st.plotly_chart(fig_all_clients_age)

    # Plot age distribution for capable clients
    st.subheader("Age Distribution of Capable Clients")
    capable_days_birth = loan_data[loan_data['TARGET'] == 0]['DAYS_BIRTH'] / 365
    fig_capable_clients_age = px.histogram(x=capable_days_birth,
                                           nbins=10,
                                           title='Age of Capable Clients at Application',
                                           labels={'x': 'Age (years)', 'y': 'Number of Clients'},
                                           color_discrete_sequence=['green'])
    st.plotly_chart(fig_capable_clients_age)

    # Plot age distribution for not capable clients
    st.subheader("Age Distribution of Not Capable Clients")
    not_capable_days_birth = loan_data[loan_data['TARGET'] == 1]['DAYS_BIRTH'] / 365
    fig_not_capable_clients_age = px.histogram(x=not_capable_days_birth,
                                               nbins=10,
                                               title='Age of Not Capable Clients at Application',
                                               labels={'x': 'Age (years)', 'y': 'Number of Clients'},
                                               color_discrete_sequence=['red'])
    st.plotly_chart(fig_not_capable_clients_age)

def borrower_characteristics_analysis(data):
    st.subheader("Borrower Characteristics Analysis")

    # Initialize BorrowerCharacteristicsAnalyzer with loan data
    analyzer = BorrowerCharacteristicsAnalyzer(data)

    # Select columns for characteristics analysis
    columns_to_analyze = st.multiselect("Select Columns for Analysis", data.columns)

    if not columns_to_analyze:
        st.warning("Please select at least one column for analysis.")
        return

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
            fig_default_rates = analyzer.plot_default_rates(column_to_plot)
            st.plotly_chart(fig_default_rates)
        except ValueError as e:
            st.error(str(e))

def main():
    st.title("Loan Approval Back Office")

    # Load loan data
    csv_file_path = os.path.join(os.getcwd(), 'data', 'application_train.csv')
    loan_data = pd.read_csv(csv_file_path)

    # Create tabs for different analysis types
    tabs = ["Loan Approval Analysis", "Borrower Characteristics Analysis"]
    selected_tab = st.selectbox("Select Analysis Type", tabs)

    # Render appropriate analysis based on selected tab
    if selected_tab == "Loan Approval Analysis":
        loan_approval_analysis(loan_data)
    elif selected_tab == "Borrower Characteristics Analysis":
        borrower_characteristics_analysis(loan_data)

# Run the Streamlit app
if __name__ == "__main__":
    main()
