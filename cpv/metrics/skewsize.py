import numpy as np
import pandas as pd
from scipy import stats


def derive_version(df: pd.DataFrame) -> pd.Series:
    """Derive a 'version' string (e.g. 'White_male') from one-hot gender columns + ethnicity.

    Used as the demographic variant label required by calculate_skewsize.
    """
    gender = np.select(
        [df["gender_male"] == 1, df["gender_female"] == 1, df["gender_none"] == 1],
        ["male", "female", "none"],
        default="unknown",
    )
    return df["ethnicity"] + "_" + pd.Series(gender, index=df.index)


def calculate_skewsize(
    df: pd.DataFrame,
    llms: list[str],
    *,
    min_expected_value: int = 5,
    hue: str = "version",
    unique_labels_column: str = "answer_idx",
) -> tuple[dict, dict]:
    """Calculate the skewness of Cramér's V effect sizes across answer labels.

    For each LLM, computes Cramér's V (chi-square effect size) between the
    demographic variant column (`hue`) and the binary performance column
    (`llm_{llm}_performance`) for each answer label, then returns the skew
    of that distribution.

    Args:
        df: DataFrame containing `hue`, `unique_labels_column`, and
            `llm_{llm}_performance` columns for each llm in `llms`.
        llms: List of LLM names. Each must have a corresponding
            `llm_{name}_performance` column (1=correct, 0=incorrect).
        min_expected_value: Minimum expected cell frequency for chi-square.
        hue: Column name for the demographic variant label (default: 'version').
        unique_labels_column: Column used to stratify by answer label
            (default: 'answer_idx').

    Returns:
        (skewsize_results, cases_count)
        skewsize_results: {llm_name: skew_of_cramers_v_distribution}
        cases_count:      {llm_name: total_cases_evaluated}
    """
    skewsize_results = {}
    cases_count = {}

    for llm in llms:
        v_list = []
        unique_labels = df[unique_labels_column].unique()
        total_cases = 0

        for label in unique_labels:
            df_label = df[df[unique_labels_column] == label]
            crosstab = pd.crosstab(df_label[hue], df_label[f"llm_{llm}_performance"])

            row_totals = crosstab.sum(axis=1)
            col_totals = crosstab.sum(axis=0)
            total = crosstab.sum().sum()
            expected = np.outer(row_totals, col_totals) / total

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

        skewsize_results[llm] = float(stats.skew(v_values)) if len(v_values) > 0 else float("nan")
        cases_count[llm] = total_cases

    return skewsize_results, cases_count
