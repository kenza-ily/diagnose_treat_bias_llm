"""Tests for the PubMedQA CPV dataset."""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.utils import (
    CPV_REQUIRED_COLUMNS,
    ETHNICITIES,
    check_demographic_distribution,
    validate_cpv_schema,
)

PUBMEDQA_VALID_ANSWER_IDX = {"A", "B", "C"}


class TestSchema:
    def test_required_columns_present(self, pubmedqa_cpv_df: pd.DataFrame):
        for col in CPV_REQUIRED_COLUMNS:
            assert col in pubmedqa_cpv_df.columns, f"Missing column: {col}"

    def test_validate_cpv_schema(self, pubmedqa_cpv_df: pd.DataFrame):
        errors = validate_cpv_schema(pubmedqa_cpv_df, allow_null_option_d=True)
        assert errors == [], f"Schema validation errors: {errors}"

    def test_option_d_always_null(self, pubmedqa_cpv_df: pd.DataFrame):
        assert pubmedqa_cpv_df["option_d"].isna().all(), "option_d should always be null for PubMedQA"

    def test_answer_idx_only_abc(self, pubmedqa_cpv_df: pd.DataFrame):
        invalid = set(pubmedqa_cpv_df["answer_idx"].unique()) - PUBMEDQA_VALID_ANSWER_IDX
        assert not invalid, f"Invalid answer_idx values: {invalid}"

    def test_option_a_is_yes(self, pubmedqa_cpv_df: pd.DataFrame):
        assert (pubmedqa_cpv_df["option_a"] == "yes").all()

    def test_option_b_is_no(self, pubmedqa_cpv_df: pd.DataFrame):
        assert (pubmedqa_cpv_df["option_b"] == "no").all()

    def test_option_c_is_maybe(self, pubmedqa_cpv_df: pd.DataFrame):
        assert (pubmedqa_cpv_df["option_c"] == "maybe").all()

    def test_explanation_mostly_nonempty(self, pubmedqa_cpv_df: pd.DataFrame):
        if "explanation" not in pubmedqa_cpv_df.columns:
            pytest.skip("No 'explanation' column")
        nonempty_rate = (pubmedqa_cpv_df["explanation"].str.strip() != "").mean()
        assert nonempty_rate >= 0.95, f"Only {nonempty_rate:.1%} of explanations are non-empty"

    def test_case_id_format(self, pubmedqa_cpv_df: pd.DataFrame):
        pattern = re.compile(r"^pubmedqa_\d+$")
        bad = pubmedqa_cpv_df["case_id"].loc[~pubmedqa_cpv_df["case_id"].str.match(pattern)]
        assert bad.empty, f"Malformed case_ids: {bad.head().tolist()}"

    def test_no_empty_strings_in_question(self, pubmedqa_cpv_df: pd.DataFrame):
        assert not (pubmedqa_cpv_df["question"].str.strip() == "").any()

    def test_gender_onehot_columns_binary(self, pubmedqa_cpv_df: pd.DataFrame):
        for col in ["gender_male", "gender_female", "gender_none"]:
            assert col in pubmedqa_cpv_df.columns, f"Missing one-hot column: {col}"
            assert set(pubmedqa_cpv_df[col].unique()) <= {0, 1}, f"Non-binary values in {col}"

    def test_gender_onehot_sum_one(self, pubmedqa_cpv_df: pd.DataFrame):
        gender_sum = (
            pubmedqa_cpv_df["gender_male"]
            + pubmedqa_cpv_df["gender_female"]
            + pubmedqa_cpv_df["gender_none"]
        )
        assert (gender_sum == 1).all(), "Each row must have exactly one gender column set to 1"


class TestDemographics:
    def test_male_variants_present(self, pubmedqa_cpv_df: pd.DataFrame):
        assert pubmedqa_cpv_df["gender_male"].any(), "No male variants found"

    def test_female_variants_present(self, pubmedqa_cpv_df: pd.DataFrame):
        assert pubmedqa_cpv_df["gender_female"].any(), "No female variants found"

    def test_none_variants_present(self, pubmedqa_cpv_df: pd.DataFrame):
        assert pubmedqa_cpv_df["gender_none"].any(), "No gender-none variants found"

    def test_all_ethnicities_present(self, pubmedqa_cpv_df: pd.DataFrame):
        present = set(pubmedqa_cpv_df["ethnicity"].unique())
        assert set(ETHNICITIES) == present

    def test_all_valid_variants(self, pubmedqa_cpv_df: pd.DataFrame):
        dist = check_demographic_distribution(pubmedqa_cpv_df)
        assert dist["all_10_variants"], "Not all cases have valid demographic variant counts"

    def test_row_count_multiple_of_ethnicities(self, pubmedqa_cpv_df: pd.DataFrame):
        n_ethnicities = pubmedqa_cpv_df["ethnicity"].nunique()
        assert len(pubmedqa_cpv_df) % n_ethnicities == 0

    def test_ethnicity_in_case_text(self, pubmedqa_cpv_df: pd.DataFrame):
        sample = pubmedqa_cpv_df.sample(min(100, len(pubmedqa_cpv_df)), random_state=42)
        for _, row in sample.iterrows():
            assert row["ethnicity"] in row["case_text"], (
                f"Ethnicity '{row['ethnicity']}' not found in case_text for {row['case_id']}"
            )
