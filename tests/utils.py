"""Shared constants and utilities for CPV dataset creation and validation."""

import re
import numpy as np
import pandas as pd
from itertools import product

ETHNICITIES = ["White", "Black", "Asian", "Hispanic", "Arab"]
GENDERS = ["male", "female"]

CPV_REQUIRED_COLUMNS = [
    "case_id", "case_text", "question",
    "option_a", "option_b", "option_c", "option_d",
    "answer_idx", "gender_male", "gender_female", "gender_none", "ethnicity",
]

VALID_ANSWER_IDX = {"A", "B", "C", "D"}

GENDER_WORDS = {
    "male":   {"descriptor": "man",   "pronoun": "He", "possessive": "his"},
    "female": {"descriptor": "woman", "pronoun": "She", "possessive": "her"},
}

_AGE_PATTERN = re.compile(r"(A|An) (\d+)-year-old( patient)?", re.IGNORECASE)

ETHNICITY_KEYWORDS = {
    "White", "Black", "Asian", "Hispanic", "Arab",
    "African", "Latino", "Latina", "Caucasian",
    "Pacific Islander", "Native American",
    "Mexican", "Chinese", "Japanese", "Korean",
    "Indian", "Pakistani", "Vietnamese", "Filipino",
}

_ETHNICITY_FILTER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in ETHNICITY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

_GENDER_PATTERNS = {
    "male":   re.compile(r"\b(man|male|he|his|him|boy|gentleman)\b", re.IGNORECASE),
    "female": re.compile(r"\b(woman|female|she|her|girl|lady)\b", re.IGNORECASE),
}


def filter_ethnicity_mentions(
    df: pd.DataFrame, text_col: str = "case_text"
) -> tuple[pd.DataFrame, int]:
    """Remove rows where text_col already contains an ethnicity keyword.

    Returns (filtered_df, n_dropped).
    """
    mask = ~df[text_col].str.contains(_ETHNICITY_FILTER_RE, regex=True)
    n_dropped = int((~mask).sum())
    print(f"Filtered out {n_dropped} cases with pre-existing ethnicity mentions.")
    return df[mask].reset_index(drop=True), n_dropped


def detect_gender(text: str) -> str | None:
    """Return 'male', 'female', or None if no gender keyword found."""
    for gender, pattern in _GENDER_PATTERNS.items():
        if pattern.search(text):
            return gender
    return None


def inject_demographics(
    case_text: str,
    gender: str | None,
    ethnicity: str,
    *,
    fallback_prepend: bool = False,
) -> str:
    """Inject demographic information into a case text.

    Applies regex substitution on the age pattern. If no match and
    fallback_prepend is True, prepends a demographic header.
    When gender is None, only ethnicity is injected (no gender descriptor).
    """
    if gender is not None and gender not in GENDERS:
        raise ValueError(f"Invalid gender '{gender}'. Must be one of {GENDERS} or None.")

    if gender is None:
        def _replacer_none(m: re.Match) -> str:
            article = m.group(1)
            age = m.group(2)
            patient_suffix = " patient" if m.group(3) else ""
            return f"{article} {age}-year-old {ethnicity}{patient_suffix}"

        result, n_subs = _AGE_PATTERN.subn(_replacer_none, case_text)
        if n_subs == 0 and fallback_prepend:
            result = f"Patient: {ethnicity}.\n{case_text}"
        return result

    descriptor = GENDER_WORDS[gender]["descriptor"]

    def _replacer(m: re.Match) -> str:
        article = m.group(1)
        age = m.group(2)
        patient_suffix = " patient" if m.group(3) else ""
        return f"{article} {age}-year-old {ethnicity} {descriptor}{patient_suffix}"

    result, n_subs = _AGE_PATTERN.subn(_replacer, case_text)

    if n_subs == 0 and fallback_prepend:
        result = f"Patient: {ethnicity} {descriptor}.\n{case_text}"

    return result


def expand_to_cpv_variants(
    df: pd.DataFrame,
    *,
    ethnicities: list[str] = ETHNICITIES,
    fallback_prepend: bool = False,
) -> pd.DataFrame:
    """Expand a DataFrame to CPV variants per row.

    For each case:
    - Always produces 2 × len(ethnicities) variants (male × female).
    - If no gender keyword is detected in the original text, also produces
      len(ethnicities) additional 'none' variants (ethnicity-only injection).
    """
    rows = []
    for _, row in df.iterrows():
        detected_gender = detect_gender(str(row["case_text"]))

        for gender, ethnicity in product(GENDERS, ethnicities):
            new_row = row.copy()
            new_row["gender_male"]   = int(gender == "male")
            new_row["gender_female"] = int(gender == "female")
            new_row["gender_none"]   = 0
            new_row["ethnicity"]     = ethnicity
            new_row["case_text"]     = inject_demographics(
                str(row["case_text"]), gender, ethnicity,
                fallback_prepend=fallback_prepend,
            )
            rows.append(new_row)

        # "None" variant: only when no gender detected in source text
        if detected_gender is None:
            for ethnicity in ethnicities:
                none_row = row.copy()
                none_row["gender_male"]   = 0
                none_row["gender_female"] = 0
                none_row["gender_none"]   = 1
                none_row["ethnicity"]     = ethnicity
                none_row["case_text"]     = inject_demographics(
                    str(row["case_text"]), None, ethnicity,
                    fallback_prepend=fallback_prepend,
                )
                rows.append(none_row)

    return pd.DataFrame(rows).reset_index(drop=True)


def validate_cpv_schema(
    df: pd.DataFrame,
    *,
    allow_null_option_d: bool = False,
) -> list[str]:
    """Validate that a DataFrame conforms to the CPV schema.

    Returns a list of error strings. An empty list means the schema is valid.
    """
    errors: list[str] = []

    missing_cols = [c for c in CPV_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors

    cols_to_check = [c for c in CPV_REQUIRED_COLUMNS if not (allow_null_option_d and c == "option_d")]
    for col in cols_to_check:
        null_count = df[col].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null values.")

    invalid_answers = set(df["answer_idx"].dropna().unique()) - VALID_ANSWER_IDX
    if invalid_answers:
        errors.append(f"Invalid answer_idx values: {invalid_answers}")

    # Check one-hot gender columns are binary and sum to 1
    for col in ["gender_male", "gender_female", "gender_none"]:
        invalid = set(df[col].dropna().unique()) - {0, 1}
        if invalid:
            errors.append(f"Column '{col}' has non-binary values: {invalid}")

    gender_sum = df["gender_male"] + df["gender_female"] + df["gender_none"]
    bad_rows = int((gender_sum != 1).sum())
    if bad_rows > 0:
        errors.append(f"{bad_rows} rows have invalid one-hot gender encoding (sum != 1).")

    invalid_ethnicities = set(df["ethnicity"].dropna().unique()) - set(ETHNICITIES)
    if invalid_ethnicities:
        errors.append(f"Invalid ethnicity values: {invalid_ethnicities}")

    return errors


def _derive_gender_series(df: pd.DataFrame) -> pd.Series:
    """Derive a string gender label from one-hot gender columns."""
    return pd.Series(
        np.select(
            [df["gender_male"] == 1, df["gender_female"] == 1, df["gender_none"] == 1],
            ["male", "female", "none"],
            default="unknown",
        ),
        index=df.index,
    )


def check_demographic_distribution(
    df: pd.DataFrame,
    *,
    group_col: str = "case_id",
) -> dict:
    """Return demographic distribution statistics for a CPV DataFrame."""
    gender_series = _derive_gender_series(df)
    gender_counts = gender_series.value_counts().to_dict()
    ethnicity_counts = df["ethnicity"].value_counts().to_dict()
    crosstab = pd.crosstab(gender_series, df["ethnicity"])

    variants_per_case = df.groupby(group_col).size()
    n_ethnicities = df["ethnicity"].nunique()
    valid_counts = {2 * n_ethnicities, 3 * n_ethnicities}
    all_valid_variants = bool(variants_per_case.isin(valid_counts).all())

    return {
        "gender_counts": gender_counts,
        "ethnicity_counts": ethnicity_counts,
        "crosstab": crosstab,
        "variants_per_case": variants_per_case,
        "all_10_variants": all_valid_variants,  # key kept for backward compat
    }
