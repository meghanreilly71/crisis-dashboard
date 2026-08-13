"""
Prompt 0b — deterministic on-topic flag for both corpora.

Climate  : keyword-threshold classifier (MaC_14 rule).
           Terms sourced from matched_keywords_all in data/raw/climate.csv
           (151 unique terms reconstructed from the raw Nexis export).
           on_topic_flag = True  if  title_keyword_count >= 1
                                 OR  body_keyword_count  >= 2

Migration: topic_meta exclusion.
           Uses the 'meta' column already in the sample (= topic_meta from raw,
           renamed during preprocessing).
           on_topic_flag = False if meta in MIGRATION_OFFTOPIC_META.

Outputs (new files, originals untouched):
  data/sampled/climate_sample_flagged.csv
  data/sampled/migration_sample_flagged.csv

NOTE: The climate term list is a reconstruction from the raw Nexis export's
matched-terms field, not a confirmed record of the supervisor's original search
string. Flag this in the methods write-up and sanity-check with the supervisor.
"""

import ast
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CLIMATE_RAW = ROOT / "data" / "raw" / "climate.csv"
CLIMATE_IN = ROOT / "data" / "sampled" / "climate_sample.csv"
MIGRATION_IN = ROOT / "data" / "sampled" / "migration_sample.csv"
CLIMATE_OUT = ROOT / "data" / "sampled" / "climate_sample_flagged.csv"
MIGRATION_OUT = ROOT / "data" / "sampled" / "migration_sample_flagged.csv"

# MaC_14 threshold
TITLE_MIN = 1
BODY_MIN = 2

# topic_meta values treated as off-topic for the migration corpus
MIGRATION_OFFTOPIC_META = {
    "Media, Arts, Culture & History",
    "Pandemic",
    "Sports",
    "Healthcare",
    "Other",
    "Nature",
}


def sep(label: str) -> None:
    width = 68
    print(f"\n{'=' * width}\n  {label}\n{'=' * width}")


# ── climate keyword helpers ───────────────────────────────────────────────────


def load_climate_terms() -> list[str]:
    """Parse matched_keywords_all and return sorted unique term list."""
    raw = pd.read_csv(CLIMATE_RAW, usecols=["matched_keywords_all"])
    terms: set[str] = set()
    for val in raw["matched_keywords_all"]:
        try:
            for t in ast.literal_eval(val):
                terms.add(t.strip())
        except Exception:
            pass
    # Sort longest first so substring overlap is predictable
    return sorted(terms, key=len, reverse=True)


def count_hits(text: str, terms: list[str]) -> int:
    """Case-insensitive total occurrence count of any term in text."""
    if not isinstance(text, str) or not text:
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(t.lower()) for t in terms)


# ── per-corpus flagging ───────────────────────────────────────────────────────


def flag_climate(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["title_keyword_count"] = df["title"].apply(lambda t: count_hits(t, terms))
    df["body_keyword_count"] = df["body"].apply(lambda b: count_hits(b, terms))
    df["on_topic_flag_raw"] = (
        df["title_keyword_count"].astype(str)
        + " title hits / "
        + df["body_keyword_count"].astype(str)
        + " body hits"
    )
    df["on_topic_flag"] = (df["title_keyword_count"] >= TITLE_MIN) | (
        df["body_keyword_count"] >= BODY_MIN
    )
    return df


def flag_migration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_keyword_count"] = None
    df["body_keyword_count"] = None
    df["on_topic_flag_raw"] = df["meta"]
    df["on_topic_flag"] = ~df["meta"].isin(MIGRATION_OFFTOPIC_META)
    return df


# ── threshold sanity check ────────────────────────────────────────────────────


def print_climate_threshold_examples(df: pd.DataFrame) -> None:
    """Show a handful of articles just below and just above the MaC_14 threshold."""
    print("\n  Just below threshold  (body=1, title=0)  → OFF-TOPIC")
    below = df[(df["body_keyword_count"] == 1) & (df["title_keyword_count"] == 0)].head(
        4
    )
    if below.empty:
        print("    (none in sample)")
    for _, r in below.iterrows():
        print(
            f"    [{r['outlet_clean']} {r.get('year', '')}]  {str(r['title'])[:70]!r}"
        )
        print(
            f"      title_kw={r['title_keyword_count']}  body_kw={r['body_keyword_count']}  "
            f"on_topic={r['on_topic_flag']}"
        )

    print("\n  Just above threshold  (body=2, title=0)  → ON-TOPIC")
    above = df[(df["body_keyword_count"] == 2) & (df["title_keyword_count"] == 0)].head(
        4
    )
    if above.empty:
        print("    (none in sample)")
    for _, r in above.iterrows():
        print(
            f"    [{r['outlet_clean']} {r.get('year', '')}]  {str(r['title'])[:70]!r}"
        )
        print(
            f"      title_kw={r['title_keyword_count']}  body_kw={r['body_keyword_count']}  "
            f"on_topic={r['on_topic_flag']}"
        )

    print("\n  Title hit examples (title_kw >= 1) → ON-TOPIC")
    title_hits = df[df["title_keyword_count"] >= 1].head(4)
    for _, r in title_hits.iterrows():
        print(
            f"    [{r['outlet_clean']} {r.get('year', '')}]  {str(r['title'])[:70]!r}"
        )
        print(
            f"      title_kw={r['title_keyword_count']}  body_kw={r['body_keyword_count']}  "
            f"on_topic={r['on_topic_flag']}"
        )

    print("\n  Off-topic examples (on_topic_flag=False)")
    off = df[~df["on_topic_flag"]].head(4)
    if off.empty:
        print("    (none — all articles passed the threshold)")
    for _, r in off.iterrows():
        print(
            f"    [{r['outlet_clean']} {r.get('year', '')}]  {str(r['title'])[:70]!r}"
        )
        print(
            f"      title_kw={r['title_keyword_count']}  body_kw={r['body_keyword_count']}  "
            f"on_topic={r['on_topic_flag']}"
        )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    sep("FLAG TOPICS — deterministic on-topic classification")
    CLIMATE_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── climate ───────────────────────────────────────────────────────────────
    sep("Step 1 — load climate search terms from raw export")
    terms = load_climate_terms()
    print(f"  {len(terms)} unique terms loaded from matched_keywords_all")
    print(f"  Most frequent terms (top 10 by length for readability):")
    for t in sorted(terms[:10]):
        print(f"    {t!r}")

    sep("Step 2 — flag climate sample")
    clim = pd.read_csv(CLIMATE_IN)
    print(f"  Loaded {len(clim):,} articles from {CLIMATE_IN.name}")
    clim_flagged = flag_climate(clim, terms)

    n_on = clim_flagged["on_topic_flag"].sum()
    n_off = (~clim_flagged["on_topic_flag"]).sum()
    total = len(clim_flagged)
    print(f"\n  Results:")
    print(f"    Total     : {total:,}")
    print(f"    On-topic  : {n_on:,}  ({100 * n_on  / total:.1f}%)")
    print(f"    Off-topic : {n_off:,}  ({100 * n_off / total:.1f}%)")
    print(f"\n  Keyword hit distribution:")
    print(
        f"    title_keyword_count  — mean {clim_flagged['title_keyword_count'].mean():.2f}  "
        f"median {clim_flagged['title_keyword_count'].median():.0f}  "
        f"max {clim_flagged['title_keyword_count'].max()}"
    )
    print(
        f"    body_keyword_count   — mean {clim_flagged['body_keyword_count'].mean():.2f}  "
        f"median {clim_flagged['body_keyword_count'].median():.0f}  "
        f"max {clim_flagged['body_keyword_count'].max()}"
    )

    print_climate_threshold_examples(clim_flagged)

    clim_flagged.to_csv(CLIMATE_OUT, index=False)
    print(f"\n  Saved → {CLIMATE_OUT}")

    # ── migration ─────────────────────────────────────────────────────────────
    sep("Step 3 — flag migration sample")
    mig = pd.read_csv(MIGRATION_IN)
    print(f"  Loaded {len(mig):,} articles from {MIGRATION_IN.name}")
    mig_flagged = flag_migration(mig)

    n_on = mig_flagged["on_topic_flag"].sum()
    n_off = (~mig_flagged["on_topic_flag"]).sum()
    total = len(mig_flagged)
    print(f"\n  Results:")
    print(f"    Total     : {total:,}")
    print(f"    On-topic  : {n_on:,}  ({100 * n_on  / total:.1f}%)")
    print(f"    Off-topic : {n_off:,}  ({100 * n_off / total:.1f}%)")
    print(f"\n  Off-topic meta distribution:")
    off_topic_mig = mig_flagged[~mig_flagged["on_topic_flag"]]
    print(off_topic_mig["meta"].value_counts().to_string())
    print(f"\n  On-topic meta distribution:")
    print(mig_flagged[mig_flagged["on_topic_flag"]]["meta"].value_counts().to_string())

    print("\n  Off-topic article examples:")
    for _, r in off_topic_mig.head(5).iterrows():
        print(f"    [{r['outlet_clean']} {r.get('year', '')}]  meta={r['meta']!r}")
        print(f"      {str(r['title'])[:80]!r}")

    mig_flagged.to_csv(MIGRATION_OUT, index=False)
    print(f"\n  Saved → {MIGRATION_OUT}")

    sep("COMPLETE")
    print(f"  climate_sample_flagged.csv  : {len(clim_flagged):,} rows")
    print(f"  migration_sample_flagged.csv: {len(mig_flagged):,} rows")
    print()


if __name__ == "__main__":
    main()
