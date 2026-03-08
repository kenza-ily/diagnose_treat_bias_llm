# explore/

Lightweight tools for inspecting and comparing CPV datasets. None of these scripts modify data — they are read-only exploration utilities.

---

## Files

### `dataset_stats.py` — CLI inspector

Loads any CPV-schema dataset (local parquet, CSV, or HuggingFace Hub) and prints a structured report.

```
python3 explore/dataset_stats.py <source> [--config CONFIG] [--split SPLIT] [--sample N]
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `source` | _(required)_ | Path to a `.parquet` / `.csv` file, or a HuggingFace dataset ID |
| `--config` | `None` | HuggingFace config name (e.g. `pqa_labeled` for PubMedQA) |
| `--split` | `train` | HuggingFace split to load |
| `--sample` | `5` | Number of sample rows to print at the end |

**Report sections**

1. Shape (rows × columns)
2. Column names and dtypes
3. Null counts (only columns with nulls shown)
4. Schema validation via `validate_cpv_schema()` — PASS / FAIL with details
5. Gender × Ethnicity crosstab
6. Answer label distribution
7. Top-20 specialty distribution (MedMCQA only)
8. Sample rows (required CPV columns only)

**Examples**

```bash
# Inspect a local parquet
python3 explore/dataset_stats.py tests/medqa_cpv/medqa_cpv.parquet

# Inspect directly from HuggingFace
python3 explore/dataset_stats.py kenza-ily/medqa-cpv --split train

# PubMedQA needs a config name
python3 explore/dataset_stats.py qiaojin/PubMedQA --config pqa_labeled --split train

# Show 20 sample rows
python3 explore/dataset_stats.py tests/medmcqa_cpv/medmcqa_cpv.parquet --sample 20
```

---

### `compare_before_after.py` — before/after diff

Downloads each source dataset from HuggingFace and loads the corresponding local CPV parquet, then prints:

- A summary table (row counts, expansion factor, gender/ethnicity/answer distributions)
- 10 side-by-side before/after examples showing how a case text changes across all 10 CPV variants

**Usage**

```bash
python3 explore/compare_before_after.py
```

No arguments — the three datasets (MedQA, MedMCQA, PubMedQA) are hardcoded. Each CPV parquet must already exist locally (run the corresponding `create.py` first). Datasets not yet generated are skipped with a message.

**Prerequisites**

```bash
python3 tests/medqa_cpv/create.py
python3 tests/medmcqa_cpv/create.py
python3 tests/pubmedqa_cpv/create.py
```

---

### `dataset_inspector.ipynb` — interactive notebook

A Jupyter notebook version of `dataset_stats.py` with matplotlib charts. Set the `SOURCE` variable in the first cell and run all cells.

**Configuration cell**

```python
SOURCE = "tests/medqa_cpv/medqa_cpv.parquet"  # local path or HuggingFace ID
CONFIG = None     # HuggingFace config name if needed
SPLIT  = "train"  # HuggingFace split if needed
```

**Cells**

| Cell | Output |
|---|---|
| Load | Loads the dataset, prints row count |
| Shape + dtypes | DataFrame `.dtypes` |
| Null counts | Table of columns with nulls |
| Schema validation | PASS or list of failures |
| Demographic crosstab | Gender × Ethnicity table |
| Answer distribution | Bar chart + count table |
| Specialty distribution | Horizontal bar chart (MedMCQA) + sample rows |

**Launch**

```bash
jupyter notebook explore/dataset_inspector.ipynb
# or
jupyter lab explore/dataset_inspector.ipynb
```
