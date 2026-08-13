#!/usr/bin/env python3
"""Human-vs-LLM agreement statistics for the 100-article benchmark.

Design (fixed before inspecting results):
  * PRIMARY_RUN = 1 for BOTH prompt variants. Runs 2 and 3 are a robustness
    check reported as a spread, never as a basis for selecting a run.
  * Only binary yes/no frame labels enter the validity metrics. The free-text /
    categorical conditional fields (deservingness_direction,
    responsibility_responsible_actor, agency_agency_type) are excluded.
  * A cell is comparable only when BOTH the human and the LLM produced a
    yes/no value. Cross-corpus cells the human filled but the pipeline never
    ran (and vice versa) are dropped and itemised in the exclusion ledger.

Nothing here is treated as ground truth. This is a reference-standard
comparison; no run is designated correct, and no winner is declared.

Outputs (data/validity/):
  agreement_per_label.csv   one row per prompt x label, all statistics
  agreement_aggregate.csv   pooled and macro-averaged summaries
  agreement_exclusions.csv  every dropped cell, with the reason
  agreement_report.md       side-by-side Prompt A / Prompt B tables
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmark"
RUNS_DIR = ROOT / "data" / "annotated" / "benchmark_stability"
OUT_DIR = ROOT / "data" / "validity"

HUMAN_SHEET = BENCHMARK_DIR / "benchmark_annotation_sheet.csv.numbers"

PRIMARY_RUN = 1  # fixed a priori, both variants
RUNS = [1, 2, 3]
VARIANTS = ["A", "B"]

# Pooled base rate at or above this counts as strongly skewed marginals.
SKEW_THRESHOLD = 0.85

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20250810

BINARY = {"yes", "no"}

# The 16 binary frame labels, grouped by which corpus pass produces them.
SHARED_LABELS = [
    "conflict_present",
    "human_interest_present",
    "economic_present",
    "deservingness_present",
    "responsibility_present",
    "securitization_present",
    "othering_present",
    "agency_present",
]
MIGRATION_LABELS = ["humanitarian_present", "security_present", "policy_present"]
CLIMATE_LABELS = [
    "scientific_present",
    "crisis_present",
    "solutions_present",
    "victim_present",
    "skepticism_present",
]
LABELS = SHARED_LABELS + MIGRATION_LABELS + CLIMATE_LABELS

LABEL_SCOPE = {
    **{lab: "shared" for lab in SHARED_LABELS},
    **{lab: "migration-only" for lab in MIGRATION_LABELS},
    **{lab: "climate-only" for lab in CLIMATE_LABELS},
}

# Excluded from validity metrics by design (not binary).
NON_BINARY_FIELDS = [
    "deservingness_direction",
    "responsibility_responsible_actor",
    "agency_agency_type",
]


# ── loading ───────────────────────────────────────────────────────────────────


def load_human() -> pd.DataFrame:
    """Read the annotator's Numbers sheet into a normalised frame."""
    try:
        from numbers_parser import Document
    except ImportError:
        sys.exit(
            "numbers-parser is required to read the .numbers sheet: "
            "pip install numbers-parser"
        )

    rows = Document(str(HUMAN_SHEET)).sheets[0].tables[0].rows(values_only=True)
    df = pd.DataFrame(rows[1:], columns=list(rows[0]))
    for col in df.columns:
        df[col] = df[col].map(lambda v: None if v is None else str(v).strip())
    df["benchmark_idx"] = df["benchmark_idx"].astype(float).astype(int)
    return df


def load_run(variant: str, run: int) -> pd.DataFrame:
    df = pd.read_csv(RUNS_DIR / f"prompt{variant}_run{run}.csv", dtype=str)
    for col in df.columns:
        df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v).strip())
    df["benchmark_idx"] = df["article_idx"].astype(int)
    return df


def normalise(value) -> str | None:
    """Map a raw cell to 'yes'/'no', or None if it is not a usable binary."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in BINARY:
        return v
    return None  # covers '', 'null', 'nan', stray text


# ── statistics ────────────────────────────────────────────────────────────────


def percent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa, nominal, two raters. NaN when chance agreement is 1."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    p_o = percent_agreement(a, b)
    p_e = sum((np.mean(a == c)) * (np.mean(b == c)) for c in cats)
    if np.isclose(p_e, 1.0):
        return float("nan")  # no variance to correct for
    return float((p_o - p_e) / (1.0 - p_e))


def krippendorff_alpha(a: np.ndarray, b: np.ndarray) -> float:
    """Krippendorff's alpha, nominal metric, built from the coincidence matrix.

    Two raters with complete data on the comparable cells, so every unit
    contributes two ordered pairs. Kept in the general coincidence form so the
    arithmetic is auditable rather than a closed-form shortcut; the closed form
    is asserted against it in _self_test().
    """
    cats = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    coincidence = np.zeros((k, k), dtype=float)
    for va, vb in zip(a, b):
        i, j = index[va], index[vb]
        # m_u = 2 ratings per unit -> each ordered pair weighted 1/(m_u - 1) = 1
        coincidence[i, j] += 1.0
        coincidence[j, i] += 1.0

    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return float("nan")

    # nominal metric: delta^2 = 1 for c != k, 0 otherwise
    observed_disagreement = (coincidence.sum() - np.trace(coincidence)) / n
    expected_disagreement = (n**2 - np.sum(n_c**2)) / (n * (n - 1.0))

    if np.isclose(expected_disagreement, 0.0):
        return float("nan")
    return float(1.0 - observed_disagreement / expected_disagreement)


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, stat_fn, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float]:
    """Percentile bootstrap over articles. Wide at small N by construction."""
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        v = stat_fn(a[pick], b[pick])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return float("nan"), float("nan")  # too often degenerate to trust
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _self_test() -> None:
    """Validate alpha against its closed form for the 2-rater complete case.

    For two raters and no missing data, (1 - alpha) = (n-1)/n * (1 - Scott's pi),
    where n is the total number of ratings (2 * units).
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        size = int(rng.integers(5, 60))
        a = rng.choice(["yes", "no"], size)
        b = rng.choice(["yes", "no"], size)
        alpha = krippendorff_alpha(a, b)
        if np.isnan(alpha):
            continue
        pooled = np.concatenate([a, b])
        p_e_scott = sum(np.mean(pooled == c) ** 2 for c in ("yes", "no"))
        pi = 1.0 - (1.0 - percent_agreement(a, b)) / (1.0 - p_e_scott)
        n_ratings = 2 * size
        expected = 1.0 - (n_ratings - 1) / n_ratings * (1.0 - pi)
        assert np.isclose(alpha, expected, atol=1e-9), (alpha, expected)

    # Perfect agreement with variance present -> both statistics equal 1.
    a = np.array(["yes", "no", "yes", "no"])
    assert np.isclose(cohens_kappa(a, a), 1.0)
    assert np.isclose(krippendorff_alpha(a, a), 1.0)

    # Constant on both sides -> undefined, not 0 and not 1.
    const = np.array(["yes"] * 10)
    assert np.isnan(cohens_kappa(const, const))
    assert np.isnan(krippendorff_alpha(const, const))


# ── comparison assembly ───────────────────────────────────────────────────────


def build_pairs(human: pd.DataFrame, llm: pd.DataFrame, label: str):
    """Return aligned yes/no arrays plus the ledger of dropped cells."""
    merged = human[["benchmark_idx", "corpus", label]].merge(
        llm[["benchmark_idx", label]],
        on="benchmark_idx",
        suffixes=("_human", "_llm"),
        how="outer",
        indicator=True,
    )

    kept_h, kept_l, kept_idx, dropped = [], [], [], []
    for _, row in merged.iterrows():
        h = normalise(row.get(f"{label}_human"))
        l = normalise(row.get(f"{label}_llm"))
        if h is not None and l is not None:
            kept_h.append(h)
            kept_l.append(l)
            kept_idx.append(int(row["benchmark_idx"]))
            continue
        if h is None and l is None:
            reason = "not applicable to this corpus for either annotator"
        elif h is None:
            reason = "human blank / non-binary, LLM annotated"
        else:
            reason = "human annotated, LLM pass not run for this corpus"
        dropped.append(
            {
                "benchmark_idx": int(row["benchmark_idx"]),
                "corpus": row.get("corpus"),
                "label": label,
                "human_value": row.get(f"{label}_human"),
                "llm_value": row.get(f"{label}_llm"),
                "reason": reason,
            }
        )
    return (np.array(kept_h), np.array(kept_l), kept_idx, dropped)


def stats_for(a: np.ndarray, b: np.ndarray) -> dict:
    n = len(a)
    if n == 0:
        return {"n": 0}
    human_yes = float(np.mean(a == "yes"))
    llm_yes = float(np.mean(b == "yes"))
    pooled_yes = float(np.mean(np.concatenate([a, b]) == "yes"))
    return {
        "n": n,
        "human_yes_rate": human_yes,
        "llm_yes_rate": llm_yes,
        "pooled_yes_rate": pooled_yes,
        "raw_agreement": percent_agreement(a, b),
        "cohens_kappa": cohens_kappa(a, b),
        "krippendorff_alpha": krippendorff_alpha(a, b),
        "skewed_marginals": max(pooled_yes, 1 - pooled_yes) >= SKEW_THRESHOLD,
    }


def main() -> None:
    _self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    human = load_human()
    runs = {(v, r): load_run(v, r) for v in VARIANTS for r in RUNS}

    per_label_rows, exclusion_rows = [], []
    seen_exclusions = set()

    for variant in VARIANTS:
        for label in LABELS:
            by_run = {}
            for run in RUNS:
                a, b, _, dropped = build_pairs(human, runs[(variant, run)], label)
                by_run[run] = stats_for(a, b)
                if run == PRIMARY_RUN:
                    ci_lo, ci_hi = bootstrap_ci(a, b, cohens_kappa)
                    by_run[run]["kappa_ci_low"] = ci_lo
                    by_run[run]["kappa_ci_high"] = ci_hi
                for d in dropped:
                    key = (d["benchmark_idx"], d["label"], d["reason"])
                    if key not in seen_exclusions:
                        seen_exclusions.add(key)
                        exclusion_rows.append(d)

            primary = by_run[PRIMARY_RUN]
            row = {
                "prompt_variant": variant,
                "label": label,
                "scope": LABEL_SCOPE[label],
                "primary_run": PRIMARY_RUN,
                **{k: v for k, v in primary.items()},
            }
            for stat in ("raw_agreement", "cohens_kappa", "krippendorff_alpha"):
                vals = [
                    by_run[r][stat]
                    for r in RUNS
                    if by_run[r].get("n") and not np.isnan(by_run[r][stat])
                ]
                row[f"{stat}_run1"] = by_run[1].get(stat, float("nan"))
                row[f"{stat}_run2"] = by_run[2].get(stat, float("nan"))
                row[f"{stat}_run3"] = by_run[3].get(stat, float("nan"))
                row[f"{stat}_min"] = min(vals) if vals else float("nan")
                row[f"{stat}_max"] = max(vals) if vals else float("nan")
                row[f"{stat}_spread"] = (
                    (max(vals) - min(vals)) if vals else float("nan")
                )
            per_label_rows.append(row)

    per_label = pd.DataFrame(per_label_rows)

    # ── aggregates ────────────────────────────────────────────────────────────
    aggregate_rows = []
    for variant in VARIANTS:
        for run in RUNS:
            all_h, all_l = [], []
            for label in LABELS:
                a, b, _, _ = build_pairs(human, runs[(variant, run)], label)
                all_h.append(a)
                all_l.append(b)
            pooled_a = np.concatenate(all_h)
            pooled_b = np.concatenate(all_l)
            pooled = stats_for(pooled_a, pooled_b)

            sub = per_label[per_label.prompt_variant == variant]
            macro = {
                f"macro_mean_{stat}": float(
                    np.nanmean(sub[f"{stat}_run{run}"].to_numpy(dtype=float))
                )
                for stat in ("raw_agreement", "cohens_kappa", "krippendorff_alpha")
            }
            aggregate_rows.append(
                {
                    "prompt_variant": variant,
                    "run": run,
                    "is_primary": run == PRIMARY_RUN,
                    "n_cells": pooled["n"],
                    "pooled_raw_agreement": pooled["raw_agreement"],
                    "pooled_cohens_kappa": pooled["cohens_kappa"],
                    "pooled_krippendorff_alpha": pooled["krippendorff_alpha"],
                    "pooled_human_yes_rate": pooled["human_yes_rate"],
                    "pooled_llm_yes_rate": pooled["llm_yes_rate"],
                    **macro,
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)

    exclusions = pd.DataFrame(exclusion_rows)
    if not exclusions.empty:
        exclusions = exclusions.sort_values(["label", "benchmark_idx"])

    per_label.to_csv(OUT_DIR / "agreement_per_label.csv", index=False)
    aggregate.to_csv(OUT_DIR / "agreement_aggregate.csv", index=False)
    exclusions.to_csv(OUT_DIR / "agreement_exclusions.csv", index=False)

    write_report(per_label, aggregate, exclusions)
    print(f"wrote 4 files to {OUT_DIR}")


# ── reporting ─────────────────────────────────────────────────────────────────


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_report(
    per_label: pd.DataFrame, aggregate: pd.DataFrame, exclusions: pd.DataFrame
) -> None:
    a = per_label[per_label.prompt_variant == "A"].set_index("label")
    b = per_label[per_label.prompt_variant == "B"].set_index("label")

    lines = [
        "# Human vs LLM agreement — benchmark (n=100 articles)",
        "",
        f"Primary comparison run: **Run {PRIMARY_RUN}** for both variants, "
        "fixed before results were inspected. Runs 2 and 3 appear only as a "
        "robustness spread.",
        "",
        "No interpretation, ranking, or preferred variant is expressed here.",
        "",
        "## Per-label, primary run (Run 1)",
        "",
        "`prev` columns are yes-rates. `±` is the min–max spread of that "
        "statistic across runs 1–3. `skew` marks a pooled base rate at or "
        "beyond "
        f"{SKEW_THRESHOLD:.0%} in one direction, where kappa is unreliable and "
        "raw agreement should be read alongside it.",
        "",
    ]

    header = (
        "| label | scope | n | prev H | prev A | prev B | raw A | raw B | "
        "κ A | κ B | α A | α B | skew |"
    )
    lines += [header, "|" + "---|" * 13]
    for label in LABELS:
        ra, rb = a.loc[label], b.loc[label]
        skew = "yes" if (ra.skewed_marginals or rb.skewed_marginals) else ""
        lines.append(
            f"| {label.replace('_present','')} | {ra.scope} | {int(ra.n)} | "
            f"{fmt(ra.human_yes_rate,2)} | {fmt(ra.llm_yes_rate,2)} | "
            f"{fmt(rb.llm_yes_rate,2)} | "
            f"{fmt(ra.raw_agreement)} | {fmt(rb.raw_agreement)} | "
            f"{fmt(ra.cohens_kappa)} | {fmt(rb.cohens_kappa)} | "
            f"{fmt(ra.krippendorff_alpha)} | {fmt(rb.krippendorff_alpha)} | {skew} |"
        )

    lines += [
        "",
        "## Across-run spread (runs 1–3)",
        "",
        "| label | κ A run1 | κ A range | κ B run1 | κ B range | "
        "raw A range | raw B range |",
        "|" + "---|" * 7,
    ]
    for label in LABELS:
        ra, rb = a.loc[label], b.loc[label]
        lines.append(
            f"| {label.replace('_present','')} | {fmt(ra.cohens_kappa)} | "
            f"{fmt(ra.cohens_kappa_min)}–{fmt(ra.cohens_kappa_max)} | "
            f"{fmt(rb.cohens_kappa)} | "
            f"{fmt(rb.cohens_kappa_min)}–{fmt(rb.cohens_kappa_max)} | "
            f"{fmt(ra.raw_agreement_min)}–{fmt(ra.raw_agreement_max)} | "
            f"{fmt(rb.raw_agreement_min)}–{fmt(rb.raw_agreement_max)} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        "| variant | run | primary | cells | pooled raw | pooled κ | "
        "pooled α | macro κ | macro α |",
        "|" + "---|" * 9,
    ]
    for _, r in aggregate.iterrows():
        lines.append(
            f"| {r.prompt_variant} | {int(r.run)} | "
            f"{'*' if r.is_primary else ''} | {int(r.n_cells)} | "
            f"{fmt(r.pooled_raw_agreement)} | {fmt(r.pooled_cohens_kappa)} | "
            f"{fmt(r.pooled_krippendorff_alpha)} | "
            f"{fmt(r.macro_mean_cohens_kappa)} | "
            f"{fmt(r.macro_mean_krippendorff_alpha)} |"
        )

    lines += [
        "",
        "## Excluded cells",
        "",
        f"{len(exclusions)} cell(s) dropped as non-comparable "
        "(one side blank). Full ledger in `agreement_exclusions.csv`.",
        "",
    ]
    if not exclusions.empty:
        applicable = exclusions[~exclusions.reason.str.contains("not applicable")]
        if not applicable.empty:
            lines += [
                "| idx | corpus | label | human | llm | reason |",
                "|" + "---|" * 6,
            ]
            for _, r in applicable.iterrows():
                lines.append(
                    f"| {r.benchmark_idx} | {r.corpus} | {r.label} | "
                    f"{r.human_value} | {r.llm_value} | {r.reason} |"
                )

    (OUT_DIR / "agreement_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
