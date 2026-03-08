"""CPV metrics: accuracy, BLEU, cosine similarity, ROUGE-L, skewsize, gender bias."""

from cpv.metrics.accuracy import accuracy, accuracy_by_group, performance_disparity
from cpv.metrics.bleu import calculate_bleu, bleu_across_variants
from cpv.metrics.rouge_l import calculate_rouge_l, rouge_l_across_variants
from cpv.metrics.cossim import cosine_similarity_score, cossim_across_variants
from cpv.metrics.skewsize import calculate_skewsize, derive_version
from cpv.metrics.gender_direction.gender_bias import evaluate_gender_bias, summarize_gender_bias

__all__ = [
    "accuracy",
    "accuracy_by_group",
    "performance_disparity",
    "calculate_bleu",
    "bleu_across_variants",
    "calculate_rouge_l",
    "rouge_l_across_variants",
    "cosine_similarity_score",
    "cossim_across_variants",
    "calculate_skewsize",
    "derive_version",
    "evaluate_gender_bias",
    "summarize_gender_bias",
]
