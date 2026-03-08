"""Session-scoped fixtures loading local .parquet files for CPV dataset tests."""

import pytest
import pandas as pd
from pathlib import Path

MEDQA_PARQUET = Path(__file__).parent / "medqa_cpv" / "medqa_cpv.parquet"
MEDMCQA_PARQUET = Path(__file__).parent / "medmcqa_cpv" / "medmcqa_cpv.parquet"
PUBMEDQA_PARQUET = Path(__file__).parent / "pubmedqa_cpv" / "pubmedqa_cpv.parquet"


@pytest.fixture(scope="session")
def medqa_cpv_df() -> pd.DataFrame:
    if not MEDQA_PARQUET.exists():
        pytest.skip(f"MedQA parquet not found at {MEDQA_PARQUET}. Run tests/medqa_cpv/create.py first.")
    return pd.read_parquet(MEDQA_PARQUET)


@pytest.fixture(scope="session")
def medmcqa_cpv_df() -> pd.DataFrame:
    if not MEDMCQA_PARQUET.exists():
        pytest.skip(f"MedMCQA parquet not found at {MEDMCQA_PARQUET}. Run tests/medmcqa_cpv/create.py first.")
    return pd.read_parquet(MEDMCQA_PARQUET)


@pytest.fixture(scope="session")
def pubmedqa_cpv_df() -> pd.DataFrame:
    if not PUBMEDQA_PARQUET.exists():
        pytest.skip(f"PubMedQA parquet not found at {PUBMEDQA_PARQUET}. Run tests/pubmedqa_cpv/create.py first.")
    return pd.read_parquet(PUBMEDQA_PARQUET)
