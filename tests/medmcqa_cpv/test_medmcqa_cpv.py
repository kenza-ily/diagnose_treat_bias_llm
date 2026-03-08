"""Tests for the MedMCQA CPV dataset."""

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
    def test_required_columns_present(self, medmcqa_cpv_df: pd.DataFrame):
        for col in CPV_REQUIRED_COLUMNS:
            assert col in medmcqa_cpv_df.columns, f"Missing column: {col}"

    def test_validate_cpv_schema(self, medmcqa_cpv_df: pd.DataFrame):
        errors = validate_cpv_schema(medmcqa_cpv_df)
        assert errors == [], f"Schema validation errors: {errors}"

    def test_answer_idx_valid(self, medmcqa_cpv_df: pd.DataFrame):
        invalid = set(medmcqa_cpv_df["answer_idx"].unique()) - VALID_ANSWER_IDX
        assert not invalid, f"Invalid answer_idx values: {invalid}"

    def test_no_empty_strings_in_question(self, medmcqa_cpv_df: pd.DataFrame):
        assert not (medmcqa_cpv_df["question"].str.strip() == "").any()

    def test_no_empty_strings_in_options(self, medmcqa_cpv_df: pd.DataFrame):
        for col in ["option_a", "option_b", "option_c"]:
            assert not (medmcqa_cpv_df[col].str.strip() == "").any(), f"Empty strings in {col}"

    def test_specialty_column_present_and_nonempty(self, medmcqa_cpv_df: pd.DataFrame):
        assert "specialty" in medmcqa_cpv_df.columns, "Missing 'specialty' column"
        assert not (medmcqa_cpv_df["specialty"].str.strip() == "").all(), "All specialty values are empty"

    def test_cop_to_letter_coverage(self, medmcqa_cpv_df: pd.DataFrame):
        present = set(medmcqa_cpv_df["answer_idx"].unique())
        assert present == VALID_ANSWER_IDX, f"Not all cop-to-letter conversions covered: {present}"

    def test_gender_onehot_columns_binary(self, medmcqa_cpv_df: pd.DataFrame):
        for col in ["gender_male", "gender_female", "gender_none"]:
            assert col in medmcqa_cpv_df.columns, f"Missing one-hot column: {col}"
            assert set(medmcqa_cpv_df[col].unique()) <= {0, 1}, f"Non-binary values in {col}"

    def test_gender_onehot_sum_one(self, medmcqa_cpv_df: pd.DataFrame):
        gender_sum = (
            medmcqa_cpv_df["gender_male"]
            + medmcqa_cpv_df["gender_female"]
            + medmcqa_cpv_df["gender_none"]
        )
        assert (gender_sum == 1).all(), "Each row must have exactly one gender column set to 1"


class TestDemographics:
    def test_male_variants_present(self, medmcqa_cpv_df: pd.DataFrame):
        assert medmcqa_cpv_df["gender_male"].any(), "No male variants found"

    def test_female_variants_present(self, medmcqa_cpv_df: pd.DataFrame):
        assert medmcqa_cpv_df["gender_female"].any(), "No female variants found"

    def test_none_variants_present(self, medmcqa_cpv_df: pd.DataFrame):
        assert medmcqa_cpv_df["gender_none"].any(), "No gender-none variants found"

    def test_all_ethnicities_present(self, medmcqa_cpv_df: pd.DataFrame):
        present = set(medmcqa_cpv_df["ethnicity"].unique())
        assert set(ETHNICITIES) == present

    def test_all_valid_variants(self, medmcqa_cpv_df: pd.DataFrame):
        dist = check_demographic_distribution(medmcqa_cpv_df)
        assert dist["all_10_variants"], "Not all cases have valid demographic variant counts"

    def test_row_count_multiple_of_ethnicities(self, medmcqa_cpv_df: pd.DataFrame):
        n_ethnicities = medmcqa_cpv_df["ethnicity"].nunique()
        assert len(medmcqa_cpv_df) % n_ethnicities == 0

    def test_ethnicity_in_case_text(self, medmcqa_cpv_df: pd.DataFrame):
        sample = medmcqa_cpv_df.sample(min(100, len(medmcqa_cpv_df)), random_state=42)
        for _, row in sample.iterrows():
            assert row["ethnicity"] in row["case_text"], (
                f"Ethnicity '{row['ethnicity']}' not found in case_text for {row['case_id']}"
            )
