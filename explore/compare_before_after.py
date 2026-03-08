"""Compare original datasets vs their CPV expansions, with a summary and 10 before/after examples."""

import sys
import textwrap
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEP = "=" * 70
SEP2 = "-" * 70

DATASETS = [
    {
        "name": "MedQA",
        "original_id": "GBaker/MedQA-USMLE-4-options",
        "original_split": "test",
        "original_config": None,
        "cpv_parquet": REPO_ROOT / "tests/medqa_cpv/medqa_cpv.parquet",
        "original_text_col": "question",
        "fallback_prepend": False,
    },
    {
        "name": "MedMCQA",
        "original_id": "openlifescienceai/medmcqa",
        "original_split": "validation",
        "original_config": None,
        "cpv_parquet": REPO_ROOT / "tests/medmcqa_cpv/medmcqa_cpv.parquet",
        "original_text_col": "question",
        "fallback_prepend": True,
    },
    {
        "name": "PubMedQA",
        "original_id": "qiaojin/PubMedQA",
        "original_split": "train",
        "original_config": "pqa_labeled",
        "cpv_parquet": REPO_ROOT / "tests/pubmedqa_cpv/pubmedqa_cpv.parquet",
        "original_text_col": "question",
        "fallback_prepend": True,
    },
]


def load_original(cfg: dict) -> pd.DataFrame:
    kwargs = {"split": cfg["original_split"]}
    if cfg["original_config"]:
        kwargs["name"] = cfg["original_config"]
    ds = load_dataset(cfg["original_id"], **kwargs)
    return ds.to_pandas()


def print_summary(name: str, df_orig: pd.DataFrame, df_cpv: pd.DataFrame, text_col: str) -> None:
    print(SEP)
    print(f"  {name}")
    print(SEP)

    n_orig = len(df_orig)
    n_cpv = len(df_cpv)
    n_cases = n_cpv // 10 if n_cpv % 10 == 0 else n_cpv

    print(f"  Original rows   : {n_orig:,}")
    print(f"  CPV rows        : {n_cpv:,}  ({n_cpv // n_orig if n_orig else '?'}x expansion)")
    print(f"  Variants/case   : 10  (2 genders × 5 ethnicities)")

    print(f"\n  Original columns: {list(df_orig.columns)}")
    print(f"  CPV columns     : {list(df_cpv.columns)}")

    if "gender" in df_cpv.columns and "ethnicity" in df_cpv.columns:
        print(f"\n  Gender dist (CPV):")
        for g, c in df_cpv["gender"].value_counts().items():
            print(f"    {g}: {c:,}")
        print(f"\n  Ethnicity dist (CPV):")
        for e, c in df_cpv["ethnicity"].value_counts().items():
            print(f"    {e}: {c:,}")

    if "answer_idx" in df_cpv.columns:
        print(f"\n  Answer dist (CPV): {df_cpv['answer_idx'].value_counts().sort_index().to_dict()}")


def print_examples(name: str, df_orig: pd.DataFrame, df_cpv: pd.DataFrame, text_col: str, n: int = 10) -> None:
    print(f"\n{SEP2}")
    print(f"  {name} — {n} BEFORE / AFTER EXAMPLES")
    print(SEP2)

    orig_texts = df_orig[text_col].iloc[:n].tolist()

    for i, orig_text in enumerate(orig_texts):
        # The CPV parquet rows are in order: each original row expands to 10 consecutive CPV rows
        cpv_start = i * 10
        cpv_rows = df_cpv.iloc[cpv_start: cpv_start + 10]

        print(f"\n  [{i+1}] BEFORE")
        print(textwrap.fill(orig_text[:400], width=80, initial_indent="      ", subsequent_indent="      "))

        print(f"\n      AFTER (all 10 variants):")
        for _, row in cpv_rows.iterrows():
            label = f"{row['gender']}, {row['ethnicity']}"
            snippet = str(row["case_text"])[:300].replace("\n", " ")
            print(f"      [{label:25s}] {snippet[:120]}...")

        print(SEP2)


def main() -> None:
    for cfg in DATASETS:
        name = cfg["name"]
        print(f"\nLoading {name} original from HuggingFace...")
        df_orig = load_original(cfg)

        print(f"Loading {name} CPV from {cfg['cpv_parquet']}...")
        if not cfg["cpv_parquet"].exists():
            print(f"  SKIP: CPV parquet not found. Run tests/{name.lower()}_cpv/create.py first.")
            continue
        df_cpv = pd.read_parquet(cfg["cpv_parquet"])

        print_summary(name, df_orig, df_cpv, cfg["original_text_col"])
        print_examples(name, df_orig, df_cpv, cfg["original_text_col"])


if __name__ == "__main__":
    main()
