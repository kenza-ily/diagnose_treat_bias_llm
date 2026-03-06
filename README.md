# How Can We Diagnose and Treat Bias in LLMs for Clinical Decision-Making?

**[Paper on arXiv](https://arxiv.org/abs/2410.16574)** | Contact: [contact@kenza-ily.com](mailto:contact@kenza-ily.com)

## Overview

This repository contains the **Counterfactual Patient Variations (CPV)** framework — a method for detecting and quantifying gender and ethnicity biases in LLMs applied to clinical decision-making.

CPV constructs counterfactual patient vignettes from [JAMA Clinical Challenge](https://jamanetwork.com/collections/44038/clinical-challenge) cases by varying demographic attributes (gender, ethnicity) while holding the clinical content fixed. Bias is measured by comparing LLM outputs across these variations using a suite of lexical and semantic metrics.

## Installation

```bash
git clone https://github.com/kenza-ily/bias_llm_clinical.git
cd diagnose_treat_bias_llm
pip install -e .
```

For a full guide on dataset requirements, debiasing prompts, and alternative datasets (MedQA, MedMCQA), see [`docs/running_cpv.md`](docs/running_cpv.md).

## Dataset

The CPV dataset is derived from JAMA Clinical Challenge cases and requires a valid JAMA license. See [`data/README.md`](data/README.md) for instructions on obtaining the data and running the scraper.

## Package Structure

```
cpv/
├── metrics/
│   ├── bleu.py          # BLEU-based lexical similarity
│   ├── cossim.py        # Cosine similarity (embedding-based)
│   ├── rouge_l.py       # ROUGE-L recall
│   ├── skewsize.py      # Skew and effect-size measures
│   └── gender_direction/  # Gender-direction projection utilities
└── config/
    ├── ethnicityxgender.txt  # Demographic combination definitions
    └── costs.txt             # API cost reference

prompts/                 # Prompt templates (prompt0–prompt6)
data/                    # Dataset acquisition notebooks
```

## Usage

```python
from cpv.metrics import bleu, cossim, rouge_l, skewsize

# Compare two model outputs
score = bleu.compute(reference="The patient has hypertension.",
                     hypothesis="The patient presents with hypertension.")
```

## Research Questions

1. What gender and ethnicity biases exist in LLMs for complex clinical cases?
2. How effective are prompt engineering and fine-tuning at reducing bias?
3. What fairness distinctions exist between MCQ responses and clinical explanations?

## Citation

```bibtex
@article{benkirane2024diagnose,
  title={How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?},
  author={Benkirane, Kenza and others},
  journal={arXiv preprint arXiv:2410.16574},
  year={2024}
}
```
