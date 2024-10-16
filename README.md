## How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?

### Overview

This repository contains the code and dataset associated with the research paper titled *"How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?"* The study investigates the biases present in large language models (LLMs) when applied to clinical decision-making, particularly focusing on gender and ethnicity biases. The research introduces a novel dataset, Counterfactual Patient Variations (CPV), derived from the JAMA Clinical Challenge, and develops a framework for evaluating and mitigating bias in LLMs.

### Key Contributions

- **CPV Framework**: A systematic approach to evaluate bias in clinical scenarios using counterfactual variations.
  
- **Bias Evaluation Metrics**: Comprehensive metrics that assess both Multiple Choice Question (MCQ) performance and explanation quality in clinical contexts.
  
- **Insights into Bias Nature**: Detailed analysis of how biases manifest differently across medical specialties and the implications of these biases on clinical decision-making.

### Dataset

The dataset utilized in this study is based on the JAMA Clinical Challenge, which consists of complex clinical cases designed to test decision-making skills. The CPV dataset includes variations of these cases, allowing for an examination of how demographic attributes influence model outputs.

### Methodology

1. **Model Selection**: A diverse range of LLMs was evaluated, including GPT-3.5, GPT-4, Claude models, and Gemini.

2. **Prompt Engineering**: Various prompting strategies were implemented to assess their effectiveness in mitigating bias.

3. **Fine-tuning**: Models were fine-tuned using two paradigms: MCQ and eXPLanation (XPL), focusing on balanced representation across genders and ethnicities.

4. **Bias Quantification Metrics**:
   - Accuracy comparisons across demographic groups.
   - Statistical methods to assess performance consistency.
   - SHAP analysis for feature contribution interpretation.
   - Embedding-based evaluations to analyze reasoning quality.

### Research Questions

The study addresses three main research questions:

1. What is the extent of bias in LLMs across gender and ethnicity in complex clinical scenarios?
  
2. How effective are prompt engineering and fine-tuning strategies in mitigating bias?
  
3. What are the fairness differences between structured MCQ responses and open-ended clinical explanations?

### Findings

- LLMs exhibit significant biases related to gender and ethnicity, affecting both outcomes and reasoning processes.
  
- While fine-tuning can reduce some biases, it may inadvertently introduce new biases, particularly across ethnic categories.

- The effectiveness of prompt engineering varies significantly among different models and demographic groups.

### Installation

To use the code provided in this repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/kenza-ily/diagnose_treat_bias_llm.git
   cd diagnose_treat_bias_llm
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Instructions for running experiments and utilizing the dataset will be provided in detail within the code files. Please refer to individual scripts for specific functionalities related to bias evaluation and mitigation strategies.


### Contact

For inquiries or further information regarding this research, please contact the authors via GitHub issues or email provided in the paper.
