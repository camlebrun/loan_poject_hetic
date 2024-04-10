import streamlit as st
import pandas as pd
from LoanApprovalAnalyzer import LoanApprovalAnalyzer
from BorrowerCharacteristicsAnalyzer import BorrowerCharacteristicsAnalyzer

def main():
    st.title("Streamlit Loan Analysis")

    # Load the data
    data = pd.read_csv("data/application_train.csv")

    # Loan Approval Analyzer
    loan_app = LoanApprovalAnalyzer("data/application_train.csv")
    loan_app.run()

    # Borrower Characteristics Analyzer
    borrower_app = BorrowerCharacteristicsAnalyzer(data)

    st.sidebar.title("Borrower Characteristics Analysis")
    analysis_type = st.sidebar.selectbox("Select analysis type", ("Best Characteristics", "Worst Characteristics"))

    if analysis_type == "Best Characteristics":
        best_characteristics = borrower_app.identify_best_borrower_characteristics(loan_app.columns_to_analyze)
        st.subheader("Best Borrower Characteristics")
        for column, info in best_characteristics.items():
            st.write(f"For '{column}', '{info['best_category']}' has the best repayment rate ({info['repayment_rate']:.1f}%).")

    elif analysis_type == "Worst Characteristics":
        worst_characteristics = borrower_app.identify_worst_borrower_characteristics(loan_app.columns_to_analyze)
        st.subheader("Worst Borrower Characteristics")
        for column, info in worst_characteristics.items():
            st.write(f"For '{column}', '{info['worst_category']}' has the worst default rate ({info['default_rate']:.1f}%).")

if __name__ == "__main__":
    main()
