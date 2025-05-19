## How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?
**Note:** This codebase is currently being updated incrementally.  

**[Access the research paper on arXiv](https://arxiv.org/html/2410.16574v1)**

## TL;DR

The study explores bias in large language models (LLMs) within clinical decision-making, focusing on gender and ethnicity biases. We introduce a Counterfactual Patient Variations (CPV) dataset from JAMA Clinical Challenge cases and a framework to assess and mitigate these biases. Key findings highlight biases affecting outcomes and _reasoning_ and examine strategies for debiasing.

## Dataset

The CPV dataset builds on the [JAMA Clinical Challenge](https://jamanetwork.com/collections/44038/clinical-challenge), a collection of complex clinical cases. The data scraper is based on [ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA) and has added functionality to capture each clinical case’s date. A valid license is required to download the original JAMA dataset.

## Methodology

1. **Model Selection**: A diverse set of LLMs, including GPT-3.5, GPT-4, Claude models, LLaMa and Gemini.
2. **Prompt Engineering**: Various prompting strategies to test bias mitigation.
3. **Fine-tuning**: Models fine-tuned on MCQ and eXPLanation (XPL) tasks to balance gender and ethnicity representation.
  To access the models from OpenAI's platform:
      - MCQ: ft:gpt-4o-mini-2024-07-18:personal:v5-mcq:AADZTOHe
      - XPL: ft:gpt-4o-mini-2024-07-18:personal:v3-xpl-nounicode:A9wJDZfF
4. **Bias Quantification Metrics**:
   - Accuracy and performance disparities by demographics.
   - Statistical consistency metrics.
   - SHAP analysis for feature interpretation.

## Research Questions

1. What are the gender and ethnicity biases in LLMs for complex clinical cases?
2. How effective are prompt engineering and fine-tuning in reducing bias?
3. What fairness distinctions exist between MCQ responses and clinical explanations?

## Findings

- Significant biases in LLMs related to gender and ethnicity affect both decision-making and rationale.
- Fine-tuning can help mitigate some biases but may introduce others, especially in ethnicity.
- Prompt effectiveness varies across models and demographics.

## Contact

For more information, please contact [kenza.benkirane.23@ucl.ac.uk](mailto:kenza.benkirane.23@ucl.ac.uk).


