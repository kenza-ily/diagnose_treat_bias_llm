# Dataset

The CPV dataset is derived from [JAMA Clinical Challenge](https://jamanetwork.com/collections/44038/clinical-challenge) cases. Raw data is **not included** in this repository due to licensing restrictions.

## Obtaining the Data

1. Obtain a valid JAMA Network license or institutional access.
2. Run `scraper.ipynb` to download the raw clinical cases.
3. Run `fetch.ipynb` to construct the counterfactual patient variations.

The resulting dataset should be placed at `data/CPV/raw.csv` (excluded from version control by `.gitignore`).
