"""Unified CPV evaluation layer.

Python API:
    from cpv.evaluate import evaluate
    results = evaluate(df, llm_col="llm_gpt4_answer", explanation_col="llm_gpt4_explanation")

CLI:
    python3 cpv/evaluate.py data.parquet --llm-col llm_gpt4_answer [--explanation-col ...] [--output results.json]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import pandas as pd

from cpv.metrics.accuracy import accuracy, accuracy_by_group, performance_disparity
from cpv.metrics.bleu import bleu_across_variants
from cpv.metrics.rouge_l import rouge_l_across_variants
from cpv.metrics.cossim import cossim_across_variants
from cpv.metrics.skewsize import calculate_skewsize, derive_version
from cpv.metrics.gender_direction.gender_bias import evaluate_gender_bias, summarize_gender_bias


@dataclass
class CPVResults:
    accuracy: float
    accuracy_by_group: pd.DataFrame
    performance_disparity: dict
    skewsize: dict                        # {llm_name: skew_value}
    skewsize_cases: dict                  # {llm_name: n_cases}
    bleu: "pd.Series | None" = field(default=None)
    rouge_l: "pd.Series | None" = field(default=None)
    cossim: "pd.Series | None" = field(default=None)
    gender_bias: dict = field(default_factory=dict)


def evaluate(
    df: pd.DataFrame,
    *,
    llm_col: str,
    explanation_col: str | None = None,
    llm_name: str | None = None,
) -> CPVResults:
    """Run all CPV bias metrics and return a CPVResults object.

    Args:
        df:              CPV DataFrame with standard columns.
        llm_col:         Column with predicted answer letter (e.g. 'llm_gpt4_answer').
        explanation_col: Optional column with LLM reasoning text (enables BLEU/ROUGE/cossim).
        llm_name:        Short LLM name for skewsize key. Inferred from llm_col if None.
    """
    if llm_name is None:
        # e.g. "llm_gpt4_answer" -> "gpt4_answer"
        llm_name = llm_col.removeprefix("llm_")

    df = df.copy()

    # Derive helper columns
    df["_perf"] = (df[llm_col] == df["answer_idx"]).astype(int)
    df[f"llm_{llm_name}_performance"] = df["_perf"]
    df["version"] = derive_version(df)

    # 1. Accuracy metrics
    acc = accuracy(df, llm_col)
    acc_by_group = accuracy_by_group(df, llm_col)
    perf_disp = performance_disparity(df, llm_col)

    # 2. SkewSize
    skewsize_results, cases_count = calculate_skewsize(df, llms=[llm_name])

    # 3. Explanation consistency (optional)
    bleu_series = rouge_series = cossim_series = None
    if explanation_col is not None and explanation_col in df.columns:
        bleu_series = bleu_across_variants(df, explanation_col)
        rouge_series = rouge_l_across_variants(df, explanation_col)
        cossim_series = cossim_across_variants(df, explanation_col)

    # 4. Gender bias
    df_with_bias = evaluate_gender_bias(df)
    gender_bias = summarize_gender_bias(df_with_bias)

    return CPVResults(
        accuracy=acc,
        accuracy_by_group=acc_by_group,
        performance_disparity=perf_disp,
        skewsize=skewsize_results,
        skewsize_cases=cases_count,
        bleu=bleu_series,
        rouge_l=rouge_series,
        cossim=cossim_series,
        gender_bias=gender_bias,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(results: CPVResults, parquet_path: str, llm_col: str, df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_cases = df["case_id"].nunique() if "case_id" in df.columns else n_rows
    import os
    print(f"\n=== CPV Evaluation — {os.path.basename(parquet_path)} ===")
    print(f"Rows: {n_rows}   Cases: {n_cases}   LLM: {llm_col}\n")

    print("--- Accuracy ---")
    print(f"Overall              {results.accuracy:.3f}")

    pd_disp = results.performance_disparity
    by_eth = pd_disp.get("by_ethnicity", {})
    by_gen = pd_disp.get("by_gender", {})

    if by_eth:
        row = "  " + "    ".join(f"{k:<12} {v:.3f}" for k, v in by_eth.items())
        print(row)
    if by_gen:
        row = "  " + "    ".join(f"{k:<8} {v:.3f}" for k, v in by_gen.items())
        print(row)
    print(f"  Max gap            {pd_disp['max_gap']:.3f}    Std     {pd_disp['std']:.3f}")

    print("\n--- SkewSize (Cramér's V skew) ---")
    for name, val in results.skewsize.items():
        n = results.skewsize_cases.get(name, "?")
        print(f"  {name:<20} {val:.3f}   ({n} cases)")

    if results.bleu is not None:
        print("\n--- Explanation Consistency (mean pairwise) ---")
        print(f"  BLEU               {results.bleu.mean():.3f}")
        print(f"  ROUGE-L            {results.rouge_l.mean():.3f}")
        print(f"  Cosine similarity  {results.cossim.mean():.3f}")

    gb = results.gender_bias
    if gb and not isinstance(gb.get("mb"), float) or (isinstance(gb.get("mb"), float)):
        print("\n--- GenderBias(C) ---")
        mb = gb.get("mb", float("nan"))
        print(f"  MB (median)        {mb:.3f}" if mb == mb else "  MB (median)        nan")
        by_gen_bias = gb.get("by_gender", {})
        if by_gen_bias:
            parts = "    ".join(
                f"{g:<8} {v['mean']:.3f}" for g, v in by_gen_bias.items()
            )
            print(f"  {parts}")
    print()


def _results_to_json(results: CPVResults) -> dict:
    def _series_to_dict(s):
        if s is None:
            return None
        return {str(k): (v if v == v else None) for k, v in s.items()}

    gb = results.gender_bias.copy() if results.gender_bias else {}
    # crosstab_mean is a DataFrame — convert to nested dict
    if "crosstab_mean" in gb and hasattr(gb["crosstab_mean"], "to_dict"):
        gb["crosstab_mean"] = gb["crosstab_mean"].to_dict()

    return {
        "accuracy": results.accuracy,
        "accuracy_by_group": results.accuracy_by_group.to_dict(orient="records"),
        "performance_disparity": results.performance_disparity,
        "skewsize": results.skewsize,
        "skewsize_cases": results.skewsize_cases,
        "bleu": _series_to_dict(results.bleu),
        "rouge_l": _series_to_dict(results.rouge_l),
        "cossim": _series_to_dict(results.cossim),
        "gender_bias": gb,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM bias on a CPV parquet file."
    )
    parser.add_argument("parquet", help="Path to CPV parquet file")
    parser.add_argument("--llm-col", required=True, help="Column with predicted answer letter")
    parser.add_argument("--explanation-col", default=None, help="Column with LLM explanation text")
    parser.add_argument("--output", default=None, help="Write results to this JSON file")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    results = evaluate(df, llm_col=args.llm_col, explanation_col=args.explanation_col)

    _print_results(results, args.parquet, args.llm_col, df)

    if args.output:
        payload = _results_to_json(results)
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
