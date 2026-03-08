"""Tests for the MedQA CPV dataset."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.utils import (
    CPV_REQUIRED_COLUMNS,
    ETHNICITIES,
    VALID_ANSWER_IDX,
    check_demographic_distribution,
    validate_cpv_schema,
)


class TestSchema:
    def test_required_columns_present(self, medqa_cpv_df: pd.DataFrame):
        for col in CPV_REQUIRED_COLUMNS:
            assert col in medqa_cpv_df.columns, f"Missing column: {col}"

    def test_validate_cpv_schema(self, medqa_cpv_df: pd.DataFrame):
        errors = validate_cpv_schema(medqa_cpv_df)
        assert errors == [], f"Schema validation errors: {errors}"

    def test_answer_idx_valid(self, medqa_cpv_df: pd.DataFrame):
        invalid = set(medqa_cpv_df["answer_idx"].unique()) - VALID_ANSWER_IDX
        assert not invalid, f"Invalid answer_idx values: {invalid}"

    def test_no_empty_strings_in_question(self, medqa_cpv_df: pd.DataFrame):
        assert not (medqa_cpv_df["question"].str.strip() == "").any()

    def test_no_empty_strings_in_options(self, medqa_cpv_df: pd.DataFrame):
        for col in ["option_a", "option_b", "option_c"]:
            assert not (medqa_cpv_df[col].str.strip() == "").any(), f"Empty strings in {col}"

    def test_option_d_never_null(self, medqa_cpv_df: pd.DataFrame):
        assert medqa_cpv_df["option_d"].notna().all(), "option_d should never be null for MedQA"

    def test_all_answer_letters_present(self, medqa_cpv_df: pd.DataFrame):
        present = set(medqa_cpv_df["answer_idx"].unique())
        assert present == VALID_ANSWER_IDX, f"Not all answer letters present: {present}"

    def test_gender_onehot_columns_binary(self, medqa_cpv_df: pd.DataFrame):
        for col in ["gender_male", "gender_female", "gender_none"]:
            assert col in medqa_cpv_df.columns, f"Missing one-hot column: {col}"
            assert set(medqa_cpv_df[col].unique()) <= {0, 1}, f"Non-binary values in {col}"

    def test_gender_onehot_sum_one(self, medqa_cpv_df: pd.DataFrame):
        gender_sum = (
            medqa_cpv_df["gender_male"]
            + medqa_cpv_df["gender_female"]
            + medqa_cpv_df["gender_none"]
        )
        assert (gender_sum == 1).all(), "Each row must have exactly one gender column set to 1"


class TestDemographics:
    def test_male_variants_present(self, medqa_cpv_df: pd.DataFrame):
        assert medqa_cpv_df["gender_male"].any(), "No male variants found"

    def test_female_variants_present(self, medqa_cpv_df: pd.DataFrame):
        assert medqa_cpv_df["gender_female"].any(), "No female variants found"

    def test_none_variants_present(self, medqa_cpv_df: pd.DataFrame):
        assert medqa_cpv_df["gender_none"].any(), "No gender-none variants found"

    def test_all_ethnicities_present(self, medqa_cpv_df: pd.DataFrame):
        present = set(medqa_cpv_df["ethnicity"].unique())
        assert set(ETHNICITIES) == present, f"Missing ethnicities: {set(ETHNICITIES) - present}"

    def test_all_valid_variants(self, medqa_cpv_df: pd.DataFrame):
        dist = check_demographic_distribution(medqa_cpv_df)
        assert dist["all_10_variants"], "Not all cases have valid demographic variant counts"

    def test_row_count_multiple_of_ethnicities(self, medqa_cpv_df: pd.DataFrame):
        n_ethnicities = medqa_cpv_df["ethnicity"].nunique()
        assert len(medqa_cpv_df) % n_ethnicities == 0, (
            f"Row count {len(medqa_cpv_df)} is not a multiple of {n_ethnicities}"
        )

    def test_ethnicity_in_case_text(self, medqa_cpv_df: pd.DataFrame):
        # MedQA uses regex-only injection (fallback_prepend=False).
        # Cases without the "A XX-year-old" pattern won't have ethnicity injected;
        # only sample from rows where the pattern was present.
        # Use the same pattern as inject_demographics so we only test rows that were actually modified
        has_age_pattern = medqa_cpv_df["case_text"].str.contains(r"(?:A|An) \d+-year-old", regex=True)
        injectable = medqa_cpv_df[has_age_pattern]
        sample = injectable.sample(min(100, len(injectable)), random_state=42)
        for _, row in sample.iterrows():
            assert row["ethnicity"] in row["case_text"], (
                f"Ethnicity '{row['ethnicity']}' not found in case_text for {row['case_id']}"
            )
