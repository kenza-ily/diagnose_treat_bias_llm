"""Generate a structured Markdown report for a CPV dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _derive_gender_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [df["gender_male"] == 1, df["gender_female"] == 1, df["gender_none"] == 1],
            ["male", "female", "none"],
            default="unknown",
        ),
        index=df.index,
    )


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without requiring tabulate."""
    headers = [""] + list(str(c) for c in df.columns)
    rows = [[str(idx)] + [str(v) for v in row] for idx, row in df.iterrows()]
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def _gender_bias_section(
    lines: list[str],
    df: pd.DataFrame,
    evaluate_fn,
    summarize_fn,
    text_col: str = "case_text",
) -> None:
    """Append GenderBias score table rows to lines, or a skip notice."""
    df_scored = evaluate_fn(df, text_col=text_col)
    summary = summarize_fn(df_scored)

    if pd.isna(summary["mb"]):
        lines += [
            "_Gender bias evaluation skipped (sentence-transformers not installed)._",
            "",
        ]
        return

    lines += [
        "Metric: **GenderBias(C) = (e · g) / |g|** (Benkirane et al., arXiv:2410.16574).",
        "Negative = masculine-leaning; Positive = feminine-leaning.",
        "",
        f"**Median Bias Score (MB):** {summary['mb']:.4f}",
        "",
    ]

    if summary["by_gender"]:
        lines += ["| Group | Mean | Std |", "|---|---|---|"]
        for g, stats in sorted(summary["by_gender"].items()):
            lines.append(f"| {g} | {stats['mean']:.4f} | {stats['std']:.4f} |")
        lines.append("")

    if summary["by_ethnicity"]:
        lines += ["| Ethnicity | Mean | Std |", "|---|---|---|"]
        for e, stats in sorted(summary["by_ethnicity"].items()):
            lines.append(f"| {e} | {stats['mean']:.4f} | {stats['std']:.4f} |")
        lines.append("")

    if not summary["crosstab_mean"].empty:
        lines += [_df_to_markdown(summary["crosstab_mean"].round(4)), ""]


def generate_cpv_report(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    source_id: str,
    source_split: str,
    injection_method: str,
    hf_repo: str,
    ethnicities: list[str],
    output_path: Path,
    n_dropped: int = 0,
    df_base: pd.DataFrame | None = None,
) -> None:
    """Write a structured Markdown CPV report to output_path.

    df       — the fully-expanded CPV DataFrame (all variants).
    df_base  — the original (pre-expansion) DataFrame; when supplied, GenderBias
               is evaluated on the raw source texts and reported as a separate
               "Original Dataset" section.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_cases = df["case_id"].nunique()
    n_rows = len(df)

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# {dataset_name} — CPV Report",
        "",
        f"Generated: {timestamp}",
        "",
    ]

    # ── Dataset Overview ──────────────────────────────────────────────────────
    lines += [
        "## Dataset Overview",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source dataset | `{source_id}` |",
        f"| Split | `{source_split}` |",
        f"| HuggingFace repo | `{hf_repo}` |",
        f"| Unique cases (after filtering) | {n_cases} |",
        f"| Total rows (after expansion) | {n_rows} |",
        f"| Injection method | {injection_method} |",
        f"| Ethnicities | {', '.join(ethnicities)} |",
        "",
    ]

    # ── Ethnicity Filtering ───────────────────────────────────────────────────
    lines += [
        "## Ethnicity Filtering",
        "",
        "Cases with pre-existing ethnicity mentions in the source text were removed "
        "before CPV expansion.",
        "",
        "| | Count |",
        "|---|---|",
        f"| Cases dropped (ethnicity pre-mentioned) | {n_dropped} |",
        f"| Cases retained | {n_cases} |",
        "",
    ]

    # ── Schema ────────────────────────────────────────────────────────────────
    lines += [
        "## Schema",
        "",
        "| Column | dtype | Null count |",
        "|---|---|---|",
    ]
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        lines.append(f"| `{col}` | {df[col].dtype} | {null_count} |")
    lines.append("")

    # ── Original Gender Distribution ──────────────────────────────────────────
    has_none_variant = df.groupby("case_id")["gender_none"].max().astype(bool)
    n_orig_no_gender  = int(has_none_variant.sum())
    n_orig_has_gender = int((~has_none_variant).sum())

    lines += [
        "## Original Gender Distribution (Pre-Expansion)",
        "",
        "Whether the source case text contained a detectable gender keyword "
        "(`man`, `he`, `woman`, `she`, etc.).",
        "",
        "| Original text | Cases | % |",
        "|---|---|---|",
        f"| Gender keyword present (male or female) | {n_orig_has_gender} | "
        f"{100 * n_orig_has_gender / n_cases:.1f}% |",
        f"| No gender keyword (gender-neutral) | {n_orig_no_gender} | "
        f"{100 * n_orig_no_gender / n_cases:.1f}% |",
        "",
        "_Cases with a gender keyword receive 2 × N ethnicity variants; "
        "gender-neutral cases receive 3 × N variants (male + female + none)._",
        "",
    ]

    # ── Original Dataset Gender Bias ──────────────────────────────────────────
    lines += ["## Original Dataset Gender Bias (Pre-Expansion)", ""]

    if df_base is None:
        lines += ["_`df_base` not provided; skipping original-text bias evaluation._", ""]
    else:
        try:
            from cpv.metrics.gender_direction.gender_bias import (
                evaluate_gender_bias,
                summarize_gender_bias,
            )
            from tests.utils import detect_gender

            print("Running gender bias on original (pre-expansion) texts...")

            # Tag each row with its detected gender label
            detected = df_base["case_text"].map(
                lambda t: detect_gender(str(t)) or "none"
            )
            df_base_tagged = df_base.copy()
            df_base_tagged["gender_male"]   = (detected == "male").astype(int)
            df_base_tagged["gender_female"] = (detected == "female").astype(int)
            df_base_tagged["gender_none"]   = (detected == "none").astype(int)
            df_base_tagged["ethnicity"]     = "original"  # placeholder for summarize_fn

            _gender_bias_section(
                lines, df_base_tagged, evaluate_gender_bias, summarize_gender_bias
            )

        except Exception as exc:
            lines += [f"_Original bias evaluation failed: {exc}_", ""]

    # ── Post-Expansion Demographic Distribution ───────────────────────────────
    gender_series = _derive_gender_series(df)
    gender_counts = gender_series.value_counts().to_dict()
    ethnicity_counts = df["ethnicity"].value_counts().to_dict()
    crosstab = pd.crosstab(gender_series, df["ethnicity"])

    lines += [
        "## Post-Expansion Demographic Distribution",
        "",
        "### Gender variants",
        "",
        "| Gender | Count |",
        "|---|---|",
    ]
    for g, cnt in sorted(gender_counts.items()):
        lines.append(f"| {g} | {cnt} |")
    lines.append("")

    lines += [
        "### Ethnicity variants",
        "",
        "| Ethnicity | Count |",
        "|---|---|",
    ]
    for e, cnt in sorted(ethnicity_counts.items()):
        lines.append(f"| {e} | {cnt} |")
    lines.append("")

    lines += [
        "### Gender × Ethnicity crosstab",
        "",
        _df_to_markdown(crosstab),
        "",
    ]

    # ── Answer Distribution ───────────────────────────────────────────────────
    answer_counts = df["answer_idx"].value_counts().sort_index().to_dict()
    lines += [
        "## Answer Distribution",
        "",
        "| answer_idx | Count |",
        "|---|---|",
    ]
    for idx, cnt in answer_counts.items():
        lines.append(f"| {idx} | {cnt} |")
    lines.append("")

    # ── Post-Expansion Gender Evaluation ──────────────────────────────────────
    lines += ["## Post-Expansion Gender Evaluation", ""]

    try:
        from cpv.metrics.gender_direction.gender_bias import (
            evaluate_gender_bias,
            summarize_gender_bias,
        )
        print("Running gender bias on expanded CPV texts...")
        _gender_bias_section(lines, df, evaluate_gender_bias, summarize_gender_bias)
    except Exception as exc:
        lines += [f"_Gender bias evaluation failed: {exc}_", ""]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
