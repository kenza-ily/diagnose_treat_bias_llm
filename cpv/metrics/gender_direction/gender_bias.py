"""GenderBias metric from Benkirane et al. (arXiv:2410.16574).

GenderBias(C) = (e · g) / |g|

where g is the first principal component of SBERT embedding differences
computed on gendered sentence pairs, and e is the SBERT embedding of case C.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_sentence_pairs() -> list[tuple[str, str]]:
    """Load (female, male) sentence pairs from sentence_lists.py."""
    from cpv.metrics.gender_direction.sentence_lists import sentence_list_f, sentence_list_m
    return list(zip(sentence_list_f, sentence_list_m))


def compute_gender_direction(
    sentence_pairs: list[tuple[str, str]],
    model_name: str = _DEFAULT_MODEL,
) -> tuple[np.ndarray, object]:
    """Compute gender direction g via PCA on SBERT embedding differences.

    Returns (gender_direction_vector, sbert_model).
    Raises ImportError if sentence-transformers or sklearn are not installed.
    """
    if not _SBERT_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for gender bias evaluation. "
            "Install it with: pip install sentence-transformers"
        )
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for gender bias evaluation. "
            "Install it with: pip install scikit-learn"
        )

    model = SentenceTransformer(model_name)

    female_sents = [pair[0] for pair in sentence_pairs]
    male_sents   = [pair[1] for pair in sentence_pairs]

    emb_f = model.encode(female_sents, show_progress_bar=False, convert_to_numpy=True)
    emb_m = model.encode(male_sents,   show_progress_bar=False, convert_to_numpy=True)

    diff_vectors = emb_f - emb_m  # shape (n_pairs, hidden_dim)

    pca = PCA(n_components=1)
    pca.fit(diff_vectors)
    gender_direction = pca.components_[0]  # first PC

    return gender_direction, model


def compute_gender_bias(
    texts: list[str],
    gender_direction: np.ndarray,
    model: object,
) -> list[float]:
    """Compute GenderBias(C) = (e · g) / |g| for each text.

    Positive = feminine-leaning; Negative = masculine-leaning.
    """
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    g_norm = np.linalg.norm(gender_direction)
    scores = embeddings @ gender_direction / g_norm
    return scores.tolist()


def evaluate_gender_bias(
    df: pd.DataFrame,
    text_col: str = "case_text",
    model_name: str = _DEFAULT_MODEL,
) -> pd.DataFrame:
    """Add 'gender_bias_score' column to df.

    Gracefully skips with a warning if sentence-transformers or sklearn
    are not installed.
    """
    if not _SBERT_AVAILABLE or not _SKLEARN_AVAILABLE:
        warnings.warn(
            "Gender bias evaluation skipped: sentence-transformers and/or "
            "scikit-learn not installed.",
            stacklevel=2,
        )
        df = df.copy()
        df["gender_bias_score"] = np.nan
        return df

    print("Computing gender direction from sentence pairs...")
    pairs = load_sentence_pairs()
    gender_direction, model = compute_gender_direction(pairs, model_name=model_name)

    print(f"Scoring {len(df)} rows...")
    scores = compute_gender_bias(df[text_col].tolist(), gender_direction, model)

    df = df.copy()
    df["gender_bias_score"] = scores
    return df


def summarize_gender_bias(df: pd.DataFrame) -> dict:
    """Summarize GenderBias scores by gender and ethnicity.

    Returns a dict with:
    - by_gender: mean/std per gender label (male/female/none)
    - by_ethnicity: mean/std per ethnicity
    - crosstab_mean: gender × ethnicity mean bias scores
    - mb: Median Bias Score across all rows
    """
    if "gender_bias_score" not in df.columns or df["gender_bias_score"].isna().all():
        return {
            "by_gender": {},
            "by_ethnicity": {},
            "crosstab_mean": pd.DataFrame(),
            "mb": float("nan"),
        }

    # Derive gender label from one-hot columns
    _zero = pd.Series(0, index=df.index)
    gm = df["gender_male"]   if "gender_male"   in df.columns else _zero
    gf = df["gender_female"] if "gender_female" in df.columns else _zero
    gn = df["gender_none"]   if "gender_none"   in df.columns else _zero
    gender_series = pd.Series(
        np.select([gm == 1, gf == 1, gn == 1], ["male", "female", "none"], default="unknown"),
        index=df.index,
        name="gender",
    )

    by_gender = (
        df.assign(gender=gender_series)
        .groupby("gender")["gender_bias_score"]
        .agg(["mean", "std"])
        .to_dict(orient="index")
    )

    by_ethnicity = (
        df.groupby("ethnicity")["gender_bias_score"]
        .agg(["mean", "std"])
        .to_dict(orient="index")
    )

    crosstab_mean = df.assign(gender=gender_series).pivot_table(
        values="gender_bias_score",
        index="gender",
        columns="ethnicity",
        aggfunc="mean",
    )

    mb = float(df["gender_bias_score"].median())

    return {
        "by_gender": by_gender,
        "by_ethnicity": by_ethnicity,
        "crosstab_mean": crosstab_mean,
        "mb": mb,
    }
