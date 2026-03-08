"""Create the MedMCQA CPV dataset and optionally push it to HuggingFace."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.utils import expand_to_cpv_variants, filter_ethnicity_mentions

HF_REPO = "kenza-ily/medmcqa-cpv"
LOCAL_CACHE = Path(__file__).parent / "medmcqa_cpv.parquet"

_COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

DATASET_CARD = """\
---
language:
  - en
license: mit
task_categories:
  - question-answering
  - multiple-choice
tags:
  - medical
  - bias
  - cpv
  - counterfactual
  - fairness
pretty_name: MedMCQA CPV (Counterfactual Patient Variations)
---

# MedMCQA CPV

Counterfactual Patient Variations of the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) `validation` split,
created to measure demographic bias in LLMs for clinical decision-making.

## Paper

> Benkirane et al. "How Can We Diagnose and Treat Bias in Large Language Models
> for Clinical Decision-Making?" NAACL 2025. arXiv:2410.16574

## Bias Representation: Before vs. After CPV

| Aspect | Before (original MedMCQA) | After (CPV expansion) |
|---|---|---|
| **Patient demographics** | Absent — questions typically do not describe a patient | Explicit — up to 15 variants per case: 3 genders × 5 ethnicities |
| **Gender distribution** | None | Balanced: male / female / None (no gender marker) |
| **Ethnicity distribution** | None | Balanced across configured ethnicities |
| **Ethnicity filtering** | None | Cases with pre-existing ethnicity mentions removed before expansion |
| **Cases × variants** | 1 version per question | 15 versions per question (3 genders × 5 ethnicities) |
| **Total rows** | ~4,183 (validation split) | ~62,745 (15× expansion) |

## Gender Encoding

Gender is one-hot encoded across three columns:
- `gender_male` (1/0)
- `gender_female` (1/0)
- `gender_none` (1/0 — original text had no detectable gender keyword; no gender injection applied)

## Gender Evaluation

Each dataset is evaluated at creation time using the **GenderBias** metric from the paper:
- Gender direction **g** derived via PCA on SBERT embeddings of ~600 gendered sentence pairs
- GenderBias(C) = (e · g) / |g| — projects case embedding onto gender direction
- Negative = masculine-leaning; Positive = feminine-leaning
- Results reported per gender × ethnicity in `cpv_report.md`

## Demographic Injection Method

Because MedMCQA questions rarely contain age patterns, the **fallback prepend** strategy is used:
`"Patient: {ethnicity} {descriptor}.\\n{question}"` e.g. `"Patient: Hispanic woman.\\nWhich drug..."`.
For `gender_none` variants: `"Patient: {ethnicity}.\\n{question}"`.

## CPV Schema

| Column | Type | Description |
|---|---|---|
| `case_id` | string | `medmcqa_NNNNNN` — unique case identifier |
| `case_text` | string | Question with prepended demographic header |
| `question` | string | Original MCQ question |
| `option_a`–`option_d` | string | Answer choices |
| `answer_idx` | string | Correct answer letter (A/B/C/D) |
| `gender_male` | int | 1 if male variant |
| `gender_female` | int | 1 if female variant |
| `gender_none` | int | 1 if no gender injection (original had no gender keyword) |
| `ethnicity` | string | Injected ethnicity |
| `explanation` | string | Explanation of correct answer (from `exp` field) |
| `specialty` | string | Medical subject/specialty (from `subject_name` field) |

## Source Dataset

- **Original:** [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) — Indian medical entrance exam questions
- **Split used:** `validation`
"""


def load_and_map() -> pd.DataFrame:
    """Load MedMCQA from HuggingFace and map to CPV schema."""
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "case_id": f"medmcqa_{i:06d}",
            "case_text": row["question"],
            "question": row["question"],
            "option_a": row.get("opa", ""),
            "option_b": row.get("opb", ""),
            "option_c": row.get("opc", ""),
            "option_d": row.get("opd", ""),
            "answer_idx": _COP_TO_LETTER.get(row.get("cop"), ""),
            "explanation": row.get("exp", ""),
            "specialty": row.get("subject_name", ""),
        })
    return pd.DataFrame(records)


def main(push_to_hub: bool = False, ethnicities: list[str] | None = None) -> pd.DataFrame:
    if ethnicities is None:
        from tests.utils import ETHNICITIES
        ethnicities = ETHNICITIES

    print("Loading MedMCQA from HuggingFace...")
    df_base = load_and_map()
    print(f"Loaded {len(df_base)} cases.")

    df_base, n_dropped = filter_ethnicity_mentions(df_base)
    print(f"Expanding {len(df_base)} cases to CPV variants...")

    df = expand_to_cpv_variants(df_base, ethnicities=ethnicities, fallback_prepend=True)
    print(f"Expanded to {len(df)} rows. Saving to {LOCAL_CACHE}...")

    LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_CACHE, index=False)
    print(f"Saved to {LOCAL_CACHE}")

    from cpv.report import generate_cpv_report
    report_path = LOCAL_CACHE.parent / "cpv_report.md"
    generate_cpv_report(
        df,
        dataset_name="MedMCQA CPV",
        source_id="openlifescienceai/medmcqa",
        source_split="validation",
        injection_method="fallback prepend (Patient: {ethnicity} {descriptor}.\\n{question})",
        hf_repo=HF_REPO,
        ethnicities=ethnicities,
        output_path=report_path,
        n_dropped=n_dropped,
        df_base=df_base,
    )
    print(f"Report written to {report_path}")

    if push_to_hub:
        from huggingface_hub import HfApi
        print(f"Pushing to HuggingFace: {HF_REPO}...")
        Dataset.from_pandas(df).push_to_hub(HF_REPO)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=DATASET_CARD.encode(),
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=report_path.read_bytes(),
            path_in_repo="cpv_report.md",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print("Done.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MedMCQA CPV dataset")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument(
        "--ethnicities", nargs="+",
        default=["White", "Black", "Asian", "Hispanic", "Arab"],
        help="Ethnicities to include (default: White Black Asian Hispanic Arab)",
    )
    args = parser.parse_args()
    main(push_to_hub=args.push, ethnicities=args.ethnicities)
