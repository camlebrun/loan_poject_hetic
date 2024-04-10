import pandas as pd

class BorrowerCharacteristicsAnalyzer:
    def __init__(self, data):
        self.data = data

    def calculate_default_rate(self, column, target_column='TARGET'):
        default_rates = self.data.groupby(column)[target_column].mean() * 100
        return default_rates

    def identify_best_borrower_characteristics(self, columns_to_analyze, target_column='TARGET'):
        best_characteristics = {}

        for column in columns_to_analyze:
            if self.data[column].dtype == 'object':
                non_default_rates = 100 - self.calculate_default_rate(column, target_column)
                best_category = non_default_rates.idxmax()
                best_repayment_rate = non_default_rates.max()
                best_characteristics[column] = {'best_category': best_category, 'repayment_rate': best_repayment_rate}

        return best_characteristics

    def identify_worst_borrower_characteristics(self, columns_to_analyze, target_column='TARGET'):
        worst_characteristics = {}

        for column in columns_to_analyze:
            if self.data[column].dtype == 'object':
                default_rates = self.calculate_default_rate(column, target_column)
                worst_category = default_rates.idxmax()
                worst_default_rate = default_rates.max()
                worst_characteristics[column] = {'worst_category': worst_category, 'default_rate': worst_default_rate}

        return worst_characteristics
