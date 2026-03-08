"""Backward-compat shim — all symbols now live in cpv.data."""
from cpv.data import (
    ETHNICITIES,
    GENDERS,
    CPV_REQUIRED_COLUMNS,
    VALID_ANSWER_IDX,
    GENDER_WORDS,
    ETHNICITY_KEYWORDS,
    filter_ethnicity_mentions,
    detect_gender,
    inject_demographics,
    expand_to_cpv_variants,
    validate_cpv_schema,
    check_demographic_distribution,
)

__all__ = [
    "ETHNICITIES",
    "GENDERS",
    "CPV_REQUIRED_COLUMNS",
    "VALID_ANSWER_IDX",
    "GENDER_WORDS",
    "ETHNICITY_KEYWORDS",
    "filter_ethnicity_mentions",
    "detect_gender",
    "inject_demographics",
    "expand_to_cpv_variants",
    "validate_cpv_schema",
    "check_demographic_distribution",
]
