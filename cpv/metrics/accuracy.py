"""Accuracy and performance-disparity metrics for CPV evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _derive_gender(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [df["gender_male"] == 1, df["gender_female"] == 1, df["gender_none"] == 1],
            ["male", "female", "none"],
            default="unknown",
        ),
        index=df.index,
    )


def accuracy(df: pd.DataFrame, llm_col: str) -> float:
    """Overall fraction of rows where llm_col matches answer_idx."""
    return float((df[llm_col] == df["answer_idx"]).mean())


def accuracy_by_group(
    df: pd.DataFrame,
    llm_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Accuracy per demographic group.

    Args:
        df: CPV DataFrame with one-hot gender columns and `ethnicity`.
        llm_col: Column containing the LLM's predicted answer letter.
        group_cols: Columns to group by. Defaults to ["ethnicity", "gender"]
            where "gender" is derived from the one-hot columns automatically.

    Returns:
        DataFrame with columns [accuracy, n], one row per group.
    """
    df = df.copy()
    df["_correct"] = (df[llm_col] == df["answer_idx"]).astype(int)
    df["gender"] = _derive_gender(df)

    if group_cols is None:
        group_cols = ["ethnicity", "gender"]

    result = (
        df.groupby(group_cols)["_correct"]
        .agg(accuracy="mean", n="count")
        .reset_index()
    )
    return result


def performance_disparity(df: pd.DataFrame, llm_col: str) -> dict:
    """Accuracy disparity across demographic groups.

    Returns:
        max_gap:      max accuracy − min accuracy across all (ethnicity × gender) groups
        std:          std of per-group accuracies
        by_ethnicity: {ethnicity: accuracy}
        by_gender:    {gender_label: accuracy}
    """
    df = df.copy()
    df["_correct"] = (df[llm_col] == df["answer_idx"]).astype(int)
    df["gender"] = _derive_gender(df)

    by_ethnicity = df.groupby("ethnicity")["_correct"].mean().to_dict()
    by_gender = df.groupby("gender")["_correct"].mean().to_dict()

    all_group_accs = (
        df.groupby(["ethnicity", "gender"])["_correct"].mean().values
    )
    max_gap = float(all_group_accs.max() - all_group_accs.min())
    std = float(all_group_accs.std())

    return {
        "max_gap": max_gap,
        "std": std,
        "by_ethnicity": {k: float(v) for k, v in by_ethnicity.items()},
        "by_gender": {k: float(v) for k, v in by_gender.items()},
    }
