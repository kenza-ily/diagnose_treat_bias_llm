"""Create the PubMedQA CPV dataset and optionally push it to HuggingFace."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.utils import expand_to_cpv_variants, filter_ethnicity_mentions

HF_REPO = "kenza-ily/pubmedqa-cpv"
LOCAL_CACHE = Path(__file__).parent / "pubmedqa_cpv.parquet"

DECISION_TO_IDX = {"yes": "A", "no": "B", "maybe": "C"}

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
  - biomedical
pretty_name: PubMedQA CPV (Counterfactual Patient Variations)
---

# PubMedQA CPV

Counterfactual Patient Variations of the [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) `pqa_labeled` / `train` split,
created to measure demographic bias in LLMs for biomedical research question answering.

## Paper

> Benkirane et al. "How Can We Diagnose and Treat Bias in Large Language Models
> for Clinical Decision-Making?" NAACL 2025. arXiv:2410.16574

## Bias Representation: Before vs. After CPV

| Aspect | Before (original PubMedQA) | After (CPV expansion) |
|---|---|---|
| **Patient demographics** | Absent — questions are about study populations without named demographics | Explicit — up to 15 variants per case: 3 genders × 5 ethnicities |
| **Gender distribution** | None | Balanced: male / female / None (no gender marker) |
| **Ethnicity distribution** | None | Balanced across configured ethnicities |
| **Ethnicity filtering** | None | Cases with pre-existing ethnicity mentions removed before expansion |
| **Cases × variants** | 1 version per question | 15 versions per question (3 genders × 5 ethnicities) |
| **Total rows** | 1,000 (pqa_labeled train) | ~15,000 (15× expansion) |
| **Answer choices** | yes / no / maybe (free text) | A=yes, B=no, C=maybe; `option_d` is always null |

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

Because PubMedQA contexts rarely contain patient age patterns, the **fallback prepend** strategy is used on the concatenated abstract context:
`"Patient: {ethnicity} {descriptor}.\\n{context}"`.
For `gender_none` variants: `"Patient: {ethnicity}.\\n{context}"`.

## CPV Schema

| Column | Type | Description |
|---|---|---|
| `case_id` | string | `pubmedqa_{pubid}` — unique PubMed article ID |
| `case_text` | string | Concatenated abstract contexts with prepended demographic header |
| `question` | string | Research question |
| `option_a` | string | Always `"yes"` |
| `option_b` | string | Always `"no"` |
| `option_c` | string | Always `"maybe"` |
| `option_d` | null | Always null (3-option task) |
| `answer_idx` | string | Correct answer letter (A/B/C only) |
| `gender_male` | int | 1 if male variant |
| `gender_female` | int | 1 if female variant |
| `gender_none` | int | 1 if no gender injection (original had no gender keyword) |
| `ethnicity` | string | Injected ethnicity |
| `explanation` | string | Long answer / rationale (from `long_answer` field) |

## Source Dataset

- **Original:** [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) — biomedical research question answering from PubMed abstracts
- **Config:** `pqa_labeled`  |  **Split used:** `train`
"""


def load_and_map() -> pd.DataFrame:
    """Load PubMedQA from HuggingFace and map to CPV schema."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for row in ds:
        contexts = row.get("context", {}).get("contexts", [])
        case_text = " ".join(contexts) if contexts else row["question"]
        decision = str(row.get("final_decision", "")).lower()
        records.append({
            "case_id": f"pubmedqa_{row['pubid']}",
            "case_text": case_text,
            "question": row["question"],
            "option_a": "yes",
            "option_b": "no",
            "option_c": "maybe",
            "option_d": None,
            "answer_idx": DECISION_TO_IDX.get(decision, ""),
            "explanation": row.get("long_answer", ""),
        })
    return pd.DataFrame(records)


def main(push_to_hub: bool = False, ethnicities: list[str] | None = None) -> pd.DataFrame:
    if ethnicities is None:
        from tests.utils import ETHNICITIES
        ethnicities = ETHNICITIES

    print("Loading PubMedQA from HuggingFace...")
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
        dataset_name="PubMedQA CPV",
        source_id="qiaojin/PubMedQA",
        source_split="train (pqa_labeled)",
        injection_method="fallback prepend (Patient: {ethnicity} {descriptor}.\\n{context})",
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
    parser = argparse.ArgumentParser(description="Create PubMedQA CPV dataset")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument(
        "--ethnicities", nargs="+",
        default=["White", "Black", "Asian", "Hispanic", "Arab"],
        help="Ethnicities to include (default: White Black Asian Hispanic Arab)",
    )
    args = parser.parse_args()
    main(push_to_hub=args.push, ethnicities=args.ethnicities)
