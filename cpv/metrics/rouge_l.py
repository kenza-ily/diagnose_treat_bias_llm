from __future__ import annotations

from itertools import combinations

import pandas as pd
from rouge_score import rouge_scorer


def calculate_rouge_l(reference: str, candidate: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure


def rouge_l_across_variants(df: pd.DataFrame, response_col: str) -> pd.Series:
    """Mean pairwise ROUGE-L score across demographic variants of the same case.

    Groups by `case_id` and computes the mean ROUGE-L over all pairs of responses
    within each group.

    Returns:
        pd.Series indexed by case_id with mean pairwise ROUGE-L (NaN if < 2 responses).
    """
    def _mean_pairwise(texts: list[str]) -> float:
        pairs = list(combinations(texts, 2))
        if not pairs:
            return float("nan")
        return sum(calculate_rouge_l(a, b) for a, b in pairs) / len(pairs)

    return df.groupby("case_id")[response_col].apply(
        lambda s: _mean_pairwise(s.dropna().tolist())
    )