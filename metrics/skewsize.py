import numpy as np
import pandas as pd
from scipy import stats

def calculate_skewsize(df=df, llms=llms, min_expected_value=5,hue='version', unique_labels_column='answer_idx_shuffled'):
    """
    Calculate the skewness of the effect sizes of the LLMs.
    """

    skewsize_results = {}
    cases_count = {}
    
    for llm in llms:
        v_list = []
        unique_labels = df[unique_labels_column].unique()
        total_cases = 0
        
        for label in unique_labels:
            df_label = df[df[unique_labels_column] == label]
            crosstab = pd.crosstab(df_label[hue], df_label[f'llm_{llm}_performance'])
            
            # Calculate expected values
            row_totals = crosstab.sum(axis=1)
            col_totals = crosstab.sum(axis=0)
            total = crosstab.sum().sum()
            expected = np.outer(row_totals, col_totals) / total
            
            # Remove rows and columns that don't meet the threshold
            mask = expected >= min_expected_value
            row_mask = mask.any(axis=1)
            col_mask = mask.any(axis=0)
            crosstab_filtered = crosstab.loc[row_mask, col_mask]
            
            if crosstab_filtered.shape[0] > 1 and crosstab_filtered.shape[1] > 1:
                chi2 = stats.chi2_contingency(crosstab_filtered)[0]
                dof = (crosstab_filtered.shape[0] - 1) * (crosstab_filtered.shape[1] - 1)
                n = crosstab_filtered.sum().sum()
                v = np.sqrt(chi2 / (n * dof)) if n * dof > 0 else 0
                v_list.append(v)
                total_cases += n
        
        v_values = np.array(v_list)
        v_values = v_values[~np.isnan(v_values)]
        
        if len(v_values) > 0:
            skewsize_results[llm] = stats.skew(v_values)
        else:
            skewsize_results[llm] = np.nan
        
        cases_count[llm] = total_cases
    
    return skewsize_results, cases_count
