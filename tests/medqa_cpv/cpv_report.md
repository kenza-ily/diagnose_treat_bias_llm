# MedQA CPV — CPV Report

Generated: 2026-03-08 15:03 UTC

## Dataset Overview

| Field | Value |
|---|---|
| Source dataset | `GBaker/MedQA-USMLE-4-options` |
| Split | `test` |
| HuggingFace repo | `kenza-ily/medqa-cpv` |
| Unique cases (after filtering) | 1202 |
| Total rows (after expansion) | 12310 |
| Injection method | regex (age pattern replacement) |
| Ethnicities | White, Black, Asian, Hispanic, Arab |

## Ethnicity Filtering

Cases with pre-existing ethnicity mentions in the source text were removed before CPV expansion.

| | Count |
|---|---|
| Cases dropped (ethnicity pre-mentioned) | 71 |
| Cases retained | 1202 |

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
| `answer` | object | 0 |
| `gender_male` | int64 | 0 |
| `gender_female` | int64 | 0 |
| `gender_none` | int64 | 0 |
| `ethnicity` | object | 0 |

## Original Gender Distribution (Pre-Expansion)

Whether the source case text contained a detectable gender keyword (`man`, `he`, `woman`, `she`, etc.).

| Original text | Cases | % |
|---|---|---|
| Gender keyword present (male or female) | 1144 | 95.2% |
| No gender keyword (gender-neutral) | 58 | 4.8% |

_Cases with a gender keyword receive 2 × N ethnicity variants; gender-neutral cases receive 3 × N variants (male + female + none)._

## Original Dataset Gender Bias (Pre-Expansion)

_Gender bias evaluation skipped (sentence-transformers not installed)._

## Post-Expansion Demographic Distribution

### Gender variants

| Gender | Count |
|---|---|
| female | 6010 |
| male | 6010 |
| none | 290 |

### Ethnicity variants

| Ethnicity | Count |
|---|---|
| Arab | 2462 |
| Asian | 2462 |
| Black | 2462 |
| Hispanic | 2462 |
| White | 2462 |

### Gender × Ethnicity crosstab

|        | Arab | Asian | Black | Hispanic | White |
| ------ | ---- | ----- | ----- | -------- | ----- |
| female | 1202 | 1202  | 1202  | 1202     | 1202  |
| male   | 1202 | 1202  | 1202  | 1202     | 1202  |
| none   | 58   | 58    | 58    | 58       | 58    |

## Answer Distribution

| answer_idx | Count |
|---|---|
| A | 3470 |
| B | 2965 |
| C | 3335 |
| D | 2540 |

## Post-Expansion Gender Evaluation

_Gender bias evaluation skipped (sentence-transformers not installed)._

