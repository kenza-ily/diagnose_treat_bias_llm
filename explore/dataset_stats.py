"""CLI tool for inspecting CPV datasets (local parquet/csv or HuggingFace)."""

import argparse
from pathlib import Path

import pandas as pd

from cpv.data import validate_cpv_schema, check_demographic_distribution, CPV_REQUIRED_COLUMNS, load


def print_report(df: pd.DataFrame, sample_n: int = 5) -> None:
    """Print a structured report about a CPV dataset."""
    sep = "-" * 60

    print(sep)
    print("1. SHAPE")
    print(f"   Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print(sep)
    print("2. COLUMNS + DTYPES")
    for col in df.columns:
        print(f"   {col:<25} {df[col].dtype}")

    print(sep)
    print("3. NULL COUNTS (columns with > 0 nulls)")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("   No null values found.")
    else:
        for col, count in null_counts.items():
            print(f"   {col:<25} {count:,}")

    print(sep)
    print("4. SCHEMA VALIDATION")
    allow_null_d = "option_d" in df.columns and df["option_d"].isna().all()
    errors = validate_cpv_schema(df, allow_null_option_d=allow_null_d)
    if not errors:
        print("   PASS")
    else:
        for err in errors:
            print(f"   FAIL: {err}")

    print(sep)
    print("5. GENDER x ETHNICITY CROSSTAB")
    if "gender" in df.columns and "ethnicity" in df.columns:
        crosstab = pd.crosstab(df["gender"], df["ethnicity"])
        print(crosstab.to_string())
    else:
        print("   (gender/ethnicity columns missing)")

    print(sep)
    print("6. ANSWER LABEL DISTRIBUTION")
    if "answer_idx" in df.columns:
        counts = df["answer_idx"].value_counts().sort_index()
        for label, count in counts.items():
            print(f"   {label}: {count:,}")
    else:
        print("   (answer_idx column missing)")

    print(sep)
    print("7. SPECIALTY DISTRIBUTION (top 20)")
    if "specialty" in df.columns:
        top = df["specialty"].value_counts().head(20)
        for spec, count in top.items():
            print(f"   {spec:<40} {count:,}")
    else:
        print("   (no 'specialty' column)")

    print(sep)
    print(f"8. SAMPLE ROWS (n={sample_n}, required columns only)")
    present_cols = [c for c in CPV_REQUIRED_COLUMNS if c in df.columns]
    sample = df[present_cols].head(sample_n)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(sample.to_string(index=False))
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a CPV dataset (local file or HuggingFace Hub)."
    )
    parser.add_argument("source", help="Path to .parquet/.csv file, or HuggingFace dataset ID")
    parser.add_argument("--config", default=None, help="HuggingFace dataset config name")
    parser.add_argument("--split", default="train", help="HuggingFace dataset split (default: train)")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample rows to show (default: 5)")
    args = parser.parse_args()

    print(f"Loading: {args.source}")
    df = load(args.source, config=args.config, split=args.split)
    print_report(df, sample_n=args.sample)


if __name__ == "__main__":
    main()
