# MedMCQA CPV — CPV Report

Generated: 2026-03-08 15:07 UTC

## Dataset Overview

| Field | Value |
|---|---|
| Source dataset | `openlifescienceai/medmcqa` |
| Split | `validation` |
| HuggingFace repo | `kenza-ily/medmcqa-cpv` |
| Unique cases (after filtering) | 4162 |
| Total rows (after expansion) | 60520 |
| Injection method | fallback prepend (Patient: {ethnicity} {descriptor}.\n{question}) |
| Ethnicities | White, Black, Asian, Hispanic, Arab |

## Ethnicity Filtering

Cases with pre-existing ethnicity mentions in the source text were removed before CPV expansion.

| | Count |
|---|---|
| Cases dropped (ethnicity pre-mentioned) | 21 |
| Cases retained | 4162 |

## Schema

| Column | dtype | Null count |
|---|---|---|
| `case_id` | object | 0 |
| `case_text` | object | 0 |
| `question` | object | 0 |
| `option_a` | object | 0 |
| `option_b` | object | 0 |
| `option_c` | object | 0 |
| `option_d` | object | 0 |
| `answer_idx` | object | 0 |
| `explanation` | object | 29190 |
| `specialty` | object | 0 |
| `gender_male` | int64 | 0 |
| `gender_female` | int64 | 0 |
| `gender_none` | int64 | 0 |
| `ethnicity` | object | 0 |

## Original Gender Distribution (Pre-Expansion)

Whether the source case text contained a detectable gender keyword (`man`, `he`, `woman`, `she`, etc.).

| Original text | Cases | % |
|---|---|---|
| Gender keyword present (male or female) | 382 | 9.2% |
| No gender keyword (gender-neutral) | 3780 | 90.8% |

_Cases with a gender keyword receive 2 × N ethnicity variants; gender-neutral cases receive 3 × N variants (male + female + none)._

## Original Dataset Gender Bias (Pre-Expansion)

_Gender bias evaluation skipped (sentence-transformers not installed)._

## Post-Expansion Demographic Distribution

### Gender variants

| Gender | Count |
|---|---|
| female | 20810 |
| male | 20810 |
| none | 18900 |

### Ethnicity variants

| Ethnicity | Count |
|---|---|
| Arab | 12104 |
| Asian | 12104 |
| Black | 12104 |
| Hispanic | 12104 |
| White | 12104 |

### Gender × Ethnicity crosstab

|        | Arab | Asian | Black | Hispanic | White |
| ------ | ---- | ----- | ----- | -------- | ----- |
| female | 4162 | 4162  | 4162  | 4162     | 4162  |
| male   | 4162 | 4162  | 4162  | 4162     | 4162  |
| none   | 3780 | 3780  | 3780  | 3780     | 3780  |

## Answer Distribution

| answer_idx | Count |
|---|---|
| A | 19460 |
| B | 15825 |
| C | 13325 |
| D | 11910 |

## Post-Expansion Gender Evaluation

_Gender bias evaluation skipped (sentence-transformers not installed)._

