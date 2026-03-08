from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_score(emb1, emb2) -> float:
    # Convert tensors to numpy arrays if necessary
    if _TORCH_AVAILABLE and isinstance(emb1, torch.Tensor):
        emb1 = emb1.detach().cpu().numpy()
    if _TORCH_AVAILABLE and isinstance(emb2, torch.Tensor):
        emb2 = emb2.detach().cpu().numpy()

    # Ensure both embeddings are 2D
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)

    return float(cosine_similarity(emb1, emb2)[0][0])


def cossim_across_variants(
    df: pd.DataFrame,
    response_col: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.Series:
    """Mean pairwise cosine similarity across demographic variants of the same case.

    Embeds all unique responses in one batched call, then computes mean pairwise
    cosine similarity per case_id.

    Returns:
        pd.Series indexed by case_id (NaN if sentence-transformers not installed
        or < 2 responses per case).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        warnings.warn(
            "cossim_across_variants requires sentence-transformers. "
            "Install with: pip install sentence-transformers",
            stacklevel=2,
        )
        case_ids = df["case_id"].unique()
        return pd.Series(float("nan"), index=case_ids, name=response_col)

    valid = df[["case_id", response_col]].dropna(subset=[response_col])
    unique_texts = valid[response_col].unique().tolist()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(unique_texts, show_progress_bar=False, convert_to_numpy=True)
    emb_map = {text: emb for text, emb in zip(unique_texts, embeddings)}

    def _mean_pairwise(texts: list[str]) -> float:
        pairs = list(combinations(texts, 2))
        if not pairs:
            return float("nan")
        return float(
            sum(cosine_similarity_score(emb_map[a], emb_map[b]) for a, b in pairs) / len(pairs)
        )

    return valid.groupby("case_id")[response_col].apply(
        lambda s: _mean_pairwise(s.tolist())
    )