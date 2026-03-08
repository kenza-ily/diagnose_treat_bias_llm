# PubMedQA CPV — CPV Report

Generated: 2026-03-08 15:07 UTC

## Dataset Overview

| Field | Value |
|---|---|
| Source dataset | `qiaojin/PubMedQA` |
| Split | `train (pqa_labeled)` |
| HuggingFace repo | `kenza-ily/pubmedqa-cpv` |
| Unique cases (after filtering) | 950 |
| Total rows (after expansion) | 13800 |
| Injection method | fallback prepend (Patient: {ethnicity} {descriptor}.\n{context}) |
| Ethnicities | White, Black, Asian, Hispanic, Arab |

## Ethnicity Filtering

Cases with pre-existing ethnicity mentions in the source text were removed before CPV expansion.

| | Count |
|---|---|
| Cases dropped (ethnicity pre-mentioned) | 50 |
| Cases retained | 950 |

## Schema

| Column | dtype | Null count |
|---|---|---|
| `case_id` | object | 0 |
| `case_text` | object | 0 |
| `question` | object | 0 |
| `option_a` | object | 0 |
| `option_b` | object | 0 |
| `option_c` | object | 0 |
| `option_d` | object | 13800 |
| `answer_idx` | object | 0 |
| `explanation` | object | 0 |
| `gender_male` | int64 | 0 |
| `gender_female` | int64 | 0 |
| `gender_none` | int64 | 0 |
| `ethnicity` | object | 0 |

## Original Gender Distribution (Pre-Expansion)

Whether the source case text contained a detectable gender keyword (`man`, `he`, `woman`, `she`, etc.).

| Original text | Cases | % |
|---|---|---|
| Gender keyword present (male or female) | 90 | 9.5% |
| No gender keyword (gender-neutral) | 860 | 90.5% |

_Cases with a gender keyword receive 2 × N ethnicity variants; gender-neutral cases receive 3 × N variants (male + female + none)._

## Original Dataset Gender Bias (Pre-Expansion)

_Gender bias evaluation skipped (sentence-transformers not installed)._

## Post-Expansion Demographic Distribution

### Gender variants

| Gender | Count |
|---|---|
| female | 4750 |
| male | 4750 |
| none | 4300 |

### Ethnicity variants

| Ethnicity | Count |
|---|---|
| Arab | 2760 |
| Asian | 2760 |
| Black | 2760 |
| Hispanic | 2760 |
| White | 2760 |

### Gender × Ethnicity crosstab

|        | Arab | Asian | Black | Hispanic | White |
| ------ | ---- | ----- | ----- | -------- | ----- |
| female | 950  | 950   | 950   | 950      | 950   |
| male   | 950  | 950   | 950   | 950      | 950   |
| none   | 860  | 860   | 860   | 860      | 860   |

## Answer Distribution

| answer_idx | Count |
|---|---|
| A | 7555 |
| B | 4720 |
| C | 1525 |

## Post-Expansion Gender Evaluation

_Gender bias evaluation skipped (sentence-transformers not installed)._

