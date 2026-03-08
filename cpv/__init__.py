"""CPV — Counterfactual Patient Variations bias-detection framework."""

from cpv.data import (
    load,
    expand_to_cpv_variants,
    filter_ethnicity_mentions,
    validate_cpv_schema,
    check_demographic_distribution,
    ETHNICITIES,
    CPV_REQUIRED_COLUMNS,
)
from cpv.evaluate import evaluate, CPVResults
from cpv import metrics

__all__ = [
    # data
    "load",
    "expand_to_cpv_variants",
    "filter_ethnicity_mentions",
    "validate_cpv_schema",
    "check_demographic_distribution",
    "ETHNICITIES",
    "CPV_REQUIRED_COLUMNS",
    # evaluate
    "evaluate",
    "CPVResults",
    # submodule
    "metrics",
]
