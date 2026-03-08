from __future__ import annotations

from itertools import combinations

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(reference: str, candidate: str) -> float:
    # Use a smoothing function to handle cases with 0 counts of n-gram overlaps
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)


def bleu_across_variants(df: pd.DataFrame, response_col: str) -> pd.Series:
    """Mean pairwise BLEU score across demographic variants of the same case.

    Groups by `case_id` and computes the mean BLEU over all pairs of responses
    within each group.

    Returns:
        pd.Series indexed by case_id with mean pairwise BLEU (NaN if < 2 responses).
    """
    def _mean_pairwise(texts: list[str]) -> float:
        pairs = list(combinations(texts, 2))
        if not pairs:
            return float("nan")
        return sum(calculate_bleu(a, b) for a, b in pairs) / len(pairs)

    return df.groupby("case_id")[response_col].apply(
        lambda s: _mean_pairwise(s.dropna().tolist())
    )
