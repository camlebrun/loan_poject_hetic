import pandas as pd
import plotly.express as px

def load_csv(csv_filename):
    """Load a CSV file and handle exceptions."""
    try:
        return pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_filename}' not found.")
        return None
    except Exception as e:
        print(f"Error occurred while loading CSV file: {e}")
        return None

def calculate_repayment_rate(data, column, target_column='TARGET'):
    """Calculate repayment rate (percentage of non-defaulters) for a specific column."""
    repayment_rates = data.groupby(column)[target_column].mean() * 100
    return repayment_rates

def generate_observations(data, columns_to_analyze, target_column='TARGET'):
    """Generate dynamic observations based on specified columns."""
    observations = []

    for column in columns_to_analyze:
        if column not in data.columns:
            print(f"Column '{column}' not found in the dataset. Skipping...")
            continue
        
        if data[column].dtype == 'object':
            # For categorical columns
            repayment_rates = calculate_repayment_rate(data, column, target_column)
            top_category = repayment_rates.idxmax()
            top_repayment_rate = repayment_rates.max()
            observations.append(f"'{top_category}' for '{column}' has the highest repayment rate ({top_repayment_rate:.1f}%).")

    return observations

def plot_loan_approval_stats(data, column, target_column='TARGET', top_n=10):
    """Plot loan approval statistics for a specified column using Plotly Express."""
    if column not in data.columns:
        print(f"Column '{column}' not found in the dataset.")
        return
    
    try:
        # Group by the specified column and calculate aggregate statistics
        grouped_data = data.groupby(column)[target_column].agg(['sum', 'count', 'mean']).reset_index()
        grouped_data.columns = [column, 'Defaulters', 'Total', 'Defaulter Rate']
        grouped_data.sort_values(by='Total', ascending=False, inplace=True)
        
        # Create an interactive bar chart with Plotly Express
        fig = px.bar(grouped_data.head(top_n), x=column, y='Total', 
                     labels={'Total': 'Number of Loans'},
                     title=f'Loan Approval Statistics by {column}',
                     color='Defaulter Rate', color_continuous_scale='viridis')
        
        return fig
    
    except Exception as e:
        print(f"Error occurred for column '{column}': {e}")

def identify_best_borrower_characteristics(data, columns_to_analyze, target_column='TARGET'):
    """Identify characteristics associated with higher repayment rates (non-defaulters)."""
    best_characteristics = {}
    
    for column in columns_to_analyze:
        if column not in data.columns:
            print(f"Column '{column}' not found in the dataset. Skipping...")
            continue
        
        if data[column].dtype == 'object':
            # For categorical columns
            repayment_rates = calculate_repayment_rate(data, column, target_column)
            best_category = repayment_rates.idxmax()
            best_repayment_rate = repayment_rates.max()
            best_characteristics[column] = {'best_category': best_category, 'repayment_rate': best_repayment_rate}
    
    return best_characteristics

def identify_worst_borrower_characteristics(data, columns_to_analyze, target_column='TARGET'):
    """Identify characteristics associated with lower repayment rates (higher default rates)."""
    worst_characteristics = {}
    
    for column in columns_to_analyze:
        if column not in data.columns:
            print(f"Column '{column}' not found in the dataset. Skipping...")
            continue
        
        if data[column].dtype == 'object':
            # For categorical columns
            repayment_rates = calculate_repayment_rate(data, column, target_column)
            worst_category = repayment_rates.idxmin()  # Find category with the lowest repayment rate
            worst_repayment_rate = repayment_rates.min()
            worst_characteristics[column] = {'worst_category': worst_category, 'repayment_rate': worst_repayment_rate}
    
    return worst_characteristics
