"""
Prompt 1 — on-topic distribution check across outlet × year strata.

Reads the flagged sample files from Prompt 0b and produces:
  - Per-cell (outlet × year) counts: total, on-topic, off-topic, off-topic proportion
  - Chi-square test of independence (stratum × on-topic status)
  - Flagged cells (disproportionately off-topic or thin on-topic count)
  - Topped-up sample files if oversampling is needed

Outputs (if oversampling triggered):
  data/sampled/climate_sample_topped_up.csv
  data/sampled/migration_sample_topped_up.csv
  data/sampled/strata_report.md
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CLIMATE_IN = ROOT / "data" / "sampled" / "climate_sample_flagged.csv"
MIGRATION_IN = ROOT / "data" / "sampled" / "migration_sample_flagged.csv"
CLIMATE_PROC = ROOT / "data" / "processed" / "climate_clean.csv"
MIGRATION_PROC = ROOT / "data" / "processed" / "migration_clean.csv"
CLIMATE_OUT = ROOT / "data" / "sampled" / "climate_sample_topped_up.csv"
MIGRATION_OUT = ROOT / "data" / "sampled" / "migration_sample_topped_up.csv"
REPORT_OUT = ROOT / "data" / "sampled" / "strata_report.md"

RANDOM_SEED = 42
SAMPLE_FRAC = 0.10  # original sampling fraction (from sample.py)

# Flag a cell as disproportionately off-topic if its off-topic rate exceeds
# the corpus-wide rate by this many percentage points.
OFFTOPIC_DEVIATION_THRESHOLD = 15.0  # pp


def sep(label: str) -> None:
    width = 68
    print(f"\n{'=' * width}\n  {label}\n{'=' * width}")


# ── per-cell stats ────────────────────────────────────────────────────────────


def cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per (outlet_clean × year) summary with on/off-topic counts."""
    grp = (
        df.groupby(["outlet_clean", "year"])["on_topic_flag"]
        .agg(total="count", on_topic="sum")
        .reset_index()
    )
    grp["off_topic"] = grp["total"] - grp["on_topic"]
    grp["off_topic_pct"] = 100 * grp["off_topic"] / grp["total"]
    grp["on_topic_pct"] = 100 * grp["on_topic"] / grp["total"]
    return grp


def print_cell_table(
    stats: pd.DataFrame, corpus_off_pct: float, flag_col: str = "flagged"
) -> None:
    header = (
        f"{'outlet':<12} {'year':>4}  {'total':>6}  "
        f"{'on':>5}  {'off':>5}  {'off%':>6}  {'flag':>5}"
    )
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")
    for _, r in stats.sort_values(["outlet_clean", "year"]).iterrows():
        flag_str = "<-- FLAGGED" if r.get(flag_col, False) else ""
        print(
            f"  {r['outlet_clean']:<12} {int(r['year']):>4}  "
            f"{int(r['total']):>6}  {int(r['on_topic']):>5}  "
            f"{int(r['off_topic']):>5}  {r['off_topic_pct']:>5.1f}%  "
            f"{flag_str}"
        )
    print(
        f"\n  Corpus-wide off-topic rate: {corpus_off_pct:.1f}%  "
        f"(flag threshold: ±{OFFTOPIC_DEVIATION_THRESHOLD:.0f} pp)"
    )


# ── chi-square test ───────────────────────────────────────────────────────────


def run_chi2(stats: pd.DataFrame) -> tuple[float, float, bool, str]:
    """
    Chi-square test of independence: stratum × on-topic status.
    Returns (chi2, p_value, valid, note).
    'valid' = False if expected counts < 5 made the test unreliable.
    """
    contingency = stats[["on_topic", "off_topic"]].values
    chi2, p, dof, expected = chi2_contingency(contingency)
    min_expected = expected.min()
    valid = min_expected >= 5
    note = f"min expected count = {min_expected:.1f} " + (
        "(≥5 — chi-square is valid)"
        if valid
        else "(< 5 — result unreliable; use deviation-threshold fallback)"
    )
    return chi2, p, valid, note


# ── oversampling ──────────────────────────────────────────────────────────────


def flag_offtopic(stats: pd.DataFrame, corpus_off_pct: float) -> pd.DataFrame:
    stats = stats.copy()
    stats["flagged"] = (
        stats["off_topic_pct"] - corpus_off_pct
    ).abs() > OFFTOPIC_DEVIATION_THRESHOLD
    return stats


def flag_thin(stats: pd.DataFrame, min_on_topic: int) -> pd.DataFrame:
    stats = stats.copy()
    if "flagged" not in stats.columns:
        stats["flagged"] = False
    stats["thin"] = stats["on_topic"] < min_on_topic
    stats["needs_topup"] = stats["flagged"] | stats["thin"]
    return stats


def compute_target_per_cell(proc_path: Path) -> pd.DataFrame:
    """Load processed corpus to get per-cell raw counts and 10% targets."""
    proc = pd.read_csv(proc_path, usecols=["outlet_clean", "year"])
    raw_counts = (
        proc.groupby(["outlet_clean", "year"]).size().reset_index(name="raw_count")
    )
    raw_counts["target_n"] = (raw_counts["raw_count"] * SAMPLE_FRAC).apply(
        lambda x: max(1, round(x))
    )
    return raw_counts


def oversample_cell(
    outlet: str,
    year: int,
    sample_df: pd.DataFrame,
    proc_path: Path,
    flag_fn,
    n_needed: int,
) -> pd.DataFrame | None:
    """
    Draw n_needed additional articles from the processed corpus for this cell,
    excluding articles already in sample_df (matched by title + date).
    Applies on_topic_flag logic to new draws.
    Returns DataFrame of new rows, or None if pool is exhausted.
    """
    proc = pd.read_csv(proc_path)
    if "year" not in proc.columns:
        proc["year"] = pd.to_datetime(proc["date"], errors="coerce").dt.year

    pool = proc[(proc["outlet_clean"] == outlet) & (proc["year"] == year)].copy()
    if pool.empty:
        return None

    # Exclude already-sampled articles by (date, title) composite key
    existing_keys = set(
        zip(sample_df["date"].astype(str), sample_df["title"].str.strip().str.lower())
    )
    pool["_key"] = list(
        zip(pool["date"].astype(str), pool["title"].str.strip().str.lower())
    )
    pool = pool[~pool["_key"].isin(existing_keys)].drop(columns=["_key"])

    if pool.empty:
        return None

    # Apply on_topic_flag to the pool
    pool = flag_fn(pool)

    # Prefer on-topic articles; if not enough, take whatever is available
    on_topic_pool = pool[pool["on_topic_flag"]]
    if len(on_topic_pool) >= n_needed:
        return on_topic_pool.sample(n=n_needed, random_state=RANDOM_SEED)
    elif not pool.empty:
        n_take = min(n_needed, len(pool))
        return pool.sample(n=n_take, random_state=RANDOM_SEED)
    return None


# ── corpus-specific analysis ──────────────────────────────────────────────────


def analyse_corpus(
    name: str,
    flagged_path: Path,
    proc_path: Path,
    out_path: Path,
    flag_fn,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Prompt 1 analysis for one corpus.
    Returns (flagged_sample, stats) DataFrames.
    """
    sep(f"{name.upper()} — stratum distribution check")

    df = pd.read_csv(flagged_path)
    n_total = len(df)
    n_on = df["on_topic_flag"].sum()
    n_off = n_total - n_on
    corpus_off_pct = 100 * n_off / n_total

    print(
        f"  Loaded {n_total:,} articles  |  on-topic {n_on:,} ({100*n_on/n_total:.1f}%)"
        f"  |  off-topic {n_off:,} ({corpus_off_pct:.1f}%)"
    )
    print(f"  on_topic_flag nulls: {df['on_topic_flag'].isnull().sum()}")

    stats = cell_stats(df)

    # ── chi-square ────────────────────────────────────────────────────────────
    if n_off == 0:
        print(f"\n  All articles on-topic — chi-square not applicable.")
        chi2_result = None
    else:
        chi2, p, valid, note = run_chi2(stats)
        chi2_result = (chi2, p, valid, note)
        print(f"\n  Chi-square test (stratum × on-topic status):")
        print(f"    χ²({len(stats)-1}) = {chi2:.3f},  p = {p:.4f}")
        print(f"    {note}")

    # ── flag cells ────────────────────────────────────────────────────────────
    stats = flag_offtopic(stats, corpus_off_pct)

    # Target: 10% of raw cell size.  "Thin" = on_topic < half the target.
    raw_targets = compute_target_per_cell(proc_path)
    stats = stats.merge(raw_targets, on=["outlet_clean", "year"], how="left")
    min_on_topic = (stats["target_n"] * 0.5).apply(lambda x: max(1, round(x)))
    stats["min_on_topic"] = min_on_topic.values
    stats = flag_thin(stats, min_on_topic=0)  # mark thin individually below
    stats["thin"] = stats["on_topic"] < stats["min_on_topic"]
    stats["needs_topup"] = stats["flagged"] | stats["thin"]

    print(
        f"\n  Per-cell table  "
        f"(flagged = off-topic% deviates >{OFFTOPIC_DEVIATION_THRESHOLD:.0f} pp from {corpus_off_pct:.1f}%):"
    )
    print_cell_table(stats, corpus_off_pct)

    n_flagged = stats["flagged"].sum()
    n_thin = stats["thin"].sum()
    n_topup = stats["needs_topup"].sum()
    print(f"\n  Flagged (disproportionate off-topic): {n_flagged} cells")
    print(f"  Thin (on-topic < half target):         {n_thin} cells")
    print(f"  Total needing top-up:                  {n_topup} cells")

    # ── on-topic counts per cell ──────────────────────────────────────────────
    print(f"\n  On-topic article count per cell (what actually goes to annotation):")
    hdr = f"{'outlet':<12} {'year':>4}  {'on_topic':>8}  {'target':>7}  {'status':>10}"
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for _, r in stats.sort_values(["outlet_clean", "year"]).iterrows():
        status = "THIN" if r["thin"] else ("FLAGGED" if r["flagged"] else "ok")
        print(
            f"  {r['outlet_clean']:<12} {int(r['year']):>4}  "
            f"{int(r['on_topic']):>8}  {int(r['target_n']):>7}  {status:>10}"
        )

    # ── oversampling ──────────────────────────────────────────────────────────
    if n_topup == 0:
        print(f"\n  No oversampling needed for {name}.")
        return df, stats

    print(f"\n  Oversampling {n_topup} cells...")
    new_rows_list = []
    topup_log = []

    topup_cells = stats[stats["needs_topup"]]
    for _, r in topup_cells.iterrows():
        outlet = r["outlet_clean"]
        year = int(r["year"])
        # How many additional on-topic articles do we need to reach the target?
        needed = max(0, int(r["target_n"]) - int(r["on_topic"]))
        if needed == 0:
            continue
        new_rows = oversample_cell(outlet, year, df, proc_path, flag_fn, needed)
        if new_rows is None or len(new_rows) == 0:
            print(f"    [{outlet} {year}] WARN: pool exhausted — could not top up")
            topup_log.append(
                (
                    outlet,
                    year,
                    int(r["on_topic"]),
                    int(r["target_n"]),
                    0,
                    "pool exhausted",
                )
            )
        else:
            n_added = len(new_rows)
            print(
                f"    [{outlet} {year}] added {n_added} articles "
                f"(was {int(r['on_topic'])} on-topic, target {int(r['target_n'])})"
            )
            new_rows_list.append(new_rows)
            topup_log.append(
                (outlet, year, int(r["on_topic"]), int(r["target_n"]), n_added, "ok")
            )

    if new_rows_list:
        topped_up = pd.concat([df] + new_rows_list, ignore_index=True)
        # Re-apply flag to any rows that came in without it
        if (
            "on_topic_flag" not in topped_up.columns
            or topped_up["on_topic_flag"].isnull().any()
        ):
            topped_up = flag_fn(topped_up)
        topped_up.to_csv(out_path, index=False)
        print(f"\n  Saved topped-up sample → {out_path}")
        print(
            f"  Total rows: {len(df):,} original + "
            f"{sum(r[4] for r in topup_log):,} added = {len(topped_up):,}"
        )
    else:
        print(f"\n  No rows added (all pools exhausted).")

    return df, stats


# ── flag functions for oversampled rows ───────────────────────────────────────


def climate_flag_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Apply climate keyword classifier to new rows (needs terms from raw)."""
    import ast

    CLIMATE_RAW = ROOT / "data" / "raw" / "climate.csv"
    raw = pd.read_csv(CLIMATE_RAW, usecols=["matched_keywords_all"])
    terms: set[str] = set()
    for val in raw["matched_keywords_all"]:
        try:
            for t in ast.literal_eval(val):
                terms.add(t.strip())
        except Exception:
            pass
    terms_list = sorted(terms, key=len, reverse=True)

    df = df.copy()

    def count_hits(text: str) -> int:
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(text_lower.count(t.lower()) for t in terms_list)

    df["title_keyword_count"] = df["title"].apply(count_hits)
    df["body_keyword_count"] = df["body"].apply(
        lambda b: count_hits(b) if isinstance(b, str) else 0
    )
    df["on_topic_flag_raw"] = (
        df["title_keyword_count"].astype(str)
        + " title hits / "
        + df["body_keyword_count"].astype(str)
        + " body hits"
    )
    df["on_topic_flag"] = (df["title_keyword_count"] >= 1) | (
        df["body_keyword_count"] >= 2
    )
    return df


MIGRATION_OFFTOPIC_META = {
    "Media, Arts, Culture & History",
    "Pandemic",
    "Sports",
    "Healthcare",
    "Other",
    "Nature",
}


def migration_flag_fn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "title_keyword_count" not in df.columns:
        df["title_keyword_count"] = None
        df["body_keyword_count"] = None
    df["on_topic_flag_raw"] = df["meta"]
    df["on_topic_flag"] = ~df["meta"].isin(MIGRATION_OFFTOPIC_META)
    return df


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    sep("CHECK STRATA — Prompt 1 distribution check")

    clim_df, clim_stats = analyse_corpus(
        "climate", CLIMATE_IN, CLIMATE_PROC, CLIMATE_OUT, climate_flag_fn
    )

    mig_df, mig_stats = analyse_corpus(
        "migration", MIGRATION_IN, MIGRATION_PROC, MIGRATION_OUT, migration_flag_fn
    )

    # ── markdown report ───────────────────────────────────────────────────────
    sep("Writing markdown report")

    lines = ["# Strata Distribution Report — Prompt 1\n"]
    lines.append(
        f"Threshold for flagging: off-topic% deviates > "
        f"{OFFTOPIC_DEVIATION_THRESHOLD:.0f} pp from corpus-wide rate.  "
    )
    lines.append(f"Thin = on-topic count < 50% of 10%-sample target for that cell.\n")

    for name, stats in [("Climate", clim_stats), ("Migration", mig_stats)]:
        lines.append(f"\n## {name}\n")
        buf = StringIO()
        (
            stats[
                [
                    "outlet_clean",
                    "year",
                    "total",
                    "on_topic",
                    "off_topic",
                    "off_topic_pct",
                    "target_n",
                    "min_on_topic",
                    "flagged",
                    "thin",
                ]
            ]
            .sort_values(["outlet_clean", "year"])
            .to_string(index=False, buf=buf)
        )
        lines.append("```\n" + buf.getvalue() + "\n```\n")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines))
    print(f"  Report saved → {REPORT_OUT}")

    sep("COMPLETE")
    print()


if __name__ == "__main__":
    main()
