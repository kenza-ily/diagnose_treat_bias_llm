# How Can We Diagnose and Treat Bias in LLMs for Clinical Decision-Making?

**[Paper (NAACL 2025)](https://arxiv.org/abs/2410.16574)** | **[HuggingFace datasets](https://huggingface.co/kenza-ily)** | Contact: [contact@kenza-ily.com](mailto:contact@kenza-ily.com)

## Overview

This repository contains the **Counterfactual Patient Variations (CPV)** framework — a method for detecting and quantifying gender and ethnicity biases in LLMs applied to clinical decision-making.

CPV constructs counterfactual patient vignettes from medical QA datasets by varying demographic attributes (gender, ethnicity) while holding the clinical content fixed. Bias is measured by comparing LLM outputs across these variations using a suite of lexical and semantic metrics.

## Citation

```bibtex
@inproceedings{benkirane2025diagnose,
  title={How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?},
  author={Benkirane, Kenza and others},
  booktitle={Proceedings of NAACL 2025},
  year={2025},
  url={https://arxiv.org/abs/2410.16574}
}
```

---

## CPV Datasets

Three public CPV datasets are available on HuggingFace, each built from a different medical QA benchmark:

| Dataset | Source | HuggingFace |
|---|---|---|
| MedQA CPV | USMLE Step 1/2/3 exam questions | [`kenza-ily/medqa-cpv`](https://huggingface.co/datasets/kenza-ily/medqa-cpv) |
| MedMCQA CPV | Indian medical entrance exam | [`kenza-ily/medmcqa-cpv`](https://huggingface.co/datasets/kenza-ily/medmcqa-cpv) |
| PubMedQA CPV | Biomedical research QA from PubMed | [`kenza-ily/pubmedqa-cpv`](https://huggingface.co/datasets/kenza-ily/pubmedqa-cpv) |

Each HuggingFace repo includes a `cpv_report.md` with full statistics on filtering, demographic distribution, and GenderBias scores.

### CPV Schema

| Column | Type | Description |
|---|---|---|
| `case_id` | string | Unique case identifier |
| `case_text` | string | Clinical vignette with injected demographics |
| `question` | string | The MCQ question |
| `option_a`–`option_d` | string | Answer choices |
| `answer_idx` | string | Correct answer letter (A/B/C/D) |
| `gender_male` | int | 1 if male variant |
| `gender_female` | int | 1 if female variant |
| `gender_none` | int | 1 if no gender injection (original text had no gender keyword) |
| `ethnicity` | string | Injected ethnicity |

### CPV Methodology

1. **Pre-filter** — cases whose original text already contains an ethnicity keyword are removed.
2. **Gender detection** — each case is checked for gendered pronouns/nouns. Cases without any gender keyword receive an additional `gender_none` variant (no gender injection applied).
3. **Expansion** — for each remaining case, variants are created for every `gender × ethnicity` combination:
   - Gender keyword detected → 2 × N variants (male + female)
   - No gender keyword → 3 × N variants (male + female + none)
4. **Injection** — demographics are injected via regex (`A/An XX-year-old → A XX-year-old White woman`) or fallback prepend (`Patient: White woman.\n…`).

---

## Installation

```bash
git clone https://github.com/kenza-ily/diagnose_treat_bias_llm.git
cd diagnose_treat_bias_llm
pip install -e .
```

**Python 3.10+ required.**

Optional — for GenderBias evaluation in `cpv_report.md`:
```bash
pip install sentence-transformers scikit-learn
```

---

## Generating the CPV Datasets

Scripts are in `tests/<dataset>_cpv/create.py`. Each script downloads the source dataset from HuggingFace, filters ethnicity-mentioned cases, expands to CPV variants, saves a local `.parquet`, and writes a `cpv_report.md`.

### Run all three

```bash
python3 tests/medqa_cpv/create.py
python3 tests/medmcqa_cpv/create.py
python3 tests/pubmedqa_cpv/create.py
```

### Push to HuggingFace

```bash
python3 tests/medqa_cpv/create.py --push
python3 tests/medmcqa_cpv/create.py --push
python3 tests/pubmedqa_cpv/create.py --push
```

Requires a logged-in HuggingFace account (`huggingface-cli login`). Each push uploads:
- The dataset parquet via `push_to_hub`
- `README.md` (dataset card)
- `cpv_report.md` (statistics and bias evaluation)

### Custom ethnicity list

```bash
python3 tests/medqa_cpv/create.py --ethnicities White Black Asian
```

Default: `White Black Asian Hispanic Arab`.

### Output files

| File | Description |
|---|---|
| `tests/medqa_cpv/medqa_cpv.parquet` | Expanded CPV dataset |
| `tests/medqa_cpv/cpv_report.md` | Statistics, filtering counts, GenderBias scores |
| `tests/medmcqa_cpv/medmcqa_cpv.parquet` | |
| `tests/medmcqa_cpv/cpv_report.md` | |
| `tests/pubmedqa_cpv/pubmedqa_cpv.parquet` | |
| `tests/pubmedqa_cpv/cpv_report.md` | |

---

## Validating the Datasets

After generating, run the schema and demographic tests:

```bash
python3 -m pytest tests/ -v
```

Tests cover:
- All required columns present with correct dtypes
- One-hot gender encoding validity (binary values, sum = 1 per row)
- All ethnicities present
- Ethnicity injected into case text
- Answer index validity
- Dataset-specific checks (e.g. `option_d` always null for PubMedQA)

---

## CPV Report (`cpv_report.md`)

Each `create.py` run generates a structured Markdown report containing:

1. **Dataset Overview** — source, HF repo, case/row counts, injection method
2. **Ethnicity Filtering** — how many cases were dropped for pre-existing ethnicity mentions
3. **Schema** — all columns with dtype and null counts
4. **Original Gender Distribution** — fraction of source cases with/without detectable gender keywords
5. **Original Dataset Gender Bias** — GenderBias(C) scores on unmodified source texts, broken down by detected gender (`man`/`woman`/neutral). Requires `sentence-transformers`.
6. **Post-Expansion Demographic Distribution** — variant counts by gender and ethnicity
7. **Answer Distribution** — label balance
8. **Post-Expansion Gender Evaluation** — GenderBias(C) scores across all CPV variants by gender × ethnicity

**GenderBias(C) = (e · g) / |g|** where **g** is the first PCA component of SBERT embedding differences on ~600 gendered sentence pairs. Negative = masculine-leaning; Positive = feminine-leaning.

---

## Package Structure

```
cpv/
├── metrics/
│   ├── bleu.py
│   ├── cossim.py
│   ├── rouge_l.py
│   ├── skewsize.py
│   └── gender_direction/
│       ├── gender_bias.py       # GenderBias(C) implementation (SBERT + PCA)
│       └── sentence_lists.py   # ~600 gendered sentence pairs for PCA
├── report.py                   # generate_cpv_report()
└── config/
    ├── ethnicityxgender.txt
    └── costs.txt

tests/
├── utils.py                    # shared CPV utilities
│   # filter_ethnicity_mentions(), detect_gender(), expand_to_cpv_variants()
│   # inject_demographics(), validate_cpv_schema(), check_demographic_distribution()
├── conftest.py                 # pytest fixtures (session-scoped parquet loaders)
├── medqa_cpv/
│   ├── create.py               # dataset generation + HF push
│   ├── test_medqa_cpv.py
│   └── medqa_cpv.parquet       # generated locally (not committed)
├── medmcqa_cpv/
│   ├── create.py
│   ├── test_medmcqa_cpv.py
│   └── medmcqa_cpv.parquet
└── pubmedqa_cpv/
    ├── create.py
    ├── test_pubmedqa_cpv.py
    └── pubmedqa_cpv.parquet

prompts/                        # prompt templates (prompt0–prompt6)
```

---

## Research Questions

1. What gender and ethnicity biases exist in LLMs for complex clinical cases?
2. How effective are prompt engineering and fine-tuning at reducing bias?
3. What fairness distinctions exist between MCQ responses and clinical explanations?

## Model Selection

LLMs evaluated in the paper: GPT-3.5, GPT-4, Claude (Anthropic), LLaMA, Gemini.

Fine-tuned models (OpenAI):
- MCQ: `ft:gpt-4o-mini-2024-07-18:personal:v5-mcq:AADZTOHe`
- Explanation: `ft:gpt-4o-mini-2024-07-18:personal:v3-xpl-nounicode:A9wJDZfF`
