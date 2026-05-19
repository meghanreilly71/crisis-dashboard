import re
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42          

ROOT          = Path(__file__).parent.parent
CLIMATE_RAW   = ROOT / "data" / "raw" / "climate.csv"
MIGRATION_RAW  = ROOT / "data" / "raw" / "migration.csv"
CLIMATE_OUT   = ROOT / "data" / "processed" / "climate_clean.csv"
MIGRATION_OUT  = ROOT / "data" / "processed" / "migration_clean.csv"

DATE_MIN = pd.Timestamp("2014-01-01")
DATE_MAX = pd.Timestamp("2023-12-31")

MIN_WORD_COUNT = 100

# Columns from a prior migration-only pipeline that are not used in this project.
RETIRED_COLS = [
    "processed", "nouns", "adjectives", "verbs",
    "topic", "topic_norm", "topic_label",
    "topic_meta", "topic_meta_original", "probability",
]

# Nexis boilerplate pattern (compiled once).
_BOILERPLATE = re.compile(
    r"(^Page\s+\d+\s+of\s+\d+\s*$"
    r"|^Load-Date:.*$"
    r"|^End of Document\s*$"
    r"|^Bekijk de oorspronkelijke pagina:.*$"
    r"|^Lees ook:.*$)",
    re.MULTILINE | re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# MAPPING RECOVERY
# Read outlet/brand → outlet_clean mappings from the existing clean files.
# Falls back to embedded defaults only when clean files are absent.
# ══════════════════════════════════════════════════════════════════════════════

_CLIMATE_OUTLET_FALLBACK: Dict[str, str] = {
    "AD": "AD", "Trouw": "Trouw", "Telegraaf": "Telegraaf",
    "Volkskrant": "Volkskrant", "NRC": "NRC", "FD": "FD",
}
_MIGRATION_BRAND_FALLBACK: Dict[str, str] = {
    "AD": "AD", "TR": "Trouw", "TG": "Telegraaf",
    "VK": "Volkskrant", "NRC": "NRC", "FD": "FD",
}


def _recover_climate_map() -> Tuple[Dict[str, str], Set[str]]:
    """Return (outlet → outlet_clean map, set of outlet_clean values to keep)."""
    if CLIMATE_OUT.exists():
        ref = pd.read_csv(CLIMATE_OUT, usecols=["outlet", "outlet_clean"])
        mapping = ref.drop_duplicates().set_index("outlet")["outlet_clean"].to_dict()
        keep    = set(mapping.values())
        print(f"  [map] Climate outlet map recovered from {CLIMATE_OUT.name} "
              f"({len(mapping)} entries)")
    else:
        mapping = _CLIMATE_OUTLET_FALLBACK
        keep    = set(mapping.values())
        print(f"  [map] {CLIMATE_OUT.name} not found — using fallback map")
    return mapping, keep


def _recover_migration_map() -> Tuple[Dict[str, str], Set[str]]:
    """Return (news_brand → outlet_clean map, set of outlet_clean values to keep)."""
    if MIGRATION_OUT.exists():
        ref = pd.read_csv(MIGRATION_OUT, usecols=["news_brand", "outlet_clean"])
        mapping = ref.drop_duplicates().set_index("news_brand")["outlet_clean"].to_dict()
        keep    = set(mapping.values())
        print(f"  [map] Migration brand map recovered from {MIGRATION_OUT.name} "
              f"({len(mapping)} entries)")
    else:
        mapping = _MIGRATION_BRAND_FALLBACK
        keep    = set(mapping.values())
        print(f"  [map] ⚠  {MIGRATION_OUT.name} not found — using fallback map")
    return mapping, keep


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sep(label: str) -> None:
    width = 68
    print(f"\n{'=' * width}\n  {label}\n{'=' * width}")


def step(n: int, label: str, corpus: str) -> None:
    print(f"\n  ── Step {n}: {label}  [{corpus}]")


def row_report(before: int, after: int, reason: str = "") -> None:
    dropped = before - after
    note = f"  ({reason})" if reason else ""
    print(f"     before={before:>7,}  dropped={dropped:>6,}  after={after:>7,}{note}")


def fix_doubled_title(title: str) -> str:
    """Return the first half of a self-concatenated title string."""
    if not isinstance(title, str):
        return title
    n = len(title)
    half = n // 2
    first = title[:half].strip()
    rest  = title[half:].strip()
    # Accept as doubled if halves match (allow for odd-length string edge case).
    if first == rest or title[: half + 1].strip() == rest:
        return first
    return title.strip()


def strip_boilerplate(text: str) -> str:
    """Remove Nexis Uni boilerplate lines; collapse excess blank lines."""
    if not isinstance(text, str):
        return text
    cleaned = _BOILERPLATE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def word_count(series: pd.Series) -> pd.Series:
    """Whitespace-delimited token count for each body string."""
    return series.fillna("").str.split().str.len()


def make_match_key(df: pd.DataFrame) -> pd.Series:
    """Canonical match key: outlet_clean | YYYY-MM-DD | lower-stripped title."""
    return (
        df["outlet_clean"].astype(str)
        + "|"
        + df["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["title"].str.strip().str.lower()
    )


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS CLIMATE
# ══════════════════════════════════════════════════════════════════════════════

def process_climate(
    outlet_map: Dict[str, str],
    keep_outlets: Set[str],
) -> pd.DataFrame:
    sep("CLIMATE — processing pipeline")
    print(f"  Source: {CLIMATE_RAW}")

    df = pd.read_csv(CLIMATE_RAW, index_col=0)
    n0 = len(df)
    print(f"  Raw rows: {n0:,}  |  columns: {list(df.columns)}")

    # Step 1 — fix doubled titles ────────────────────────────────────────────
    step(1, "Fix doubled titles", "climate")
    before = len(df)
    sample = df["title"].iloc[0]
    df["title"] = df["title"].map(fix_doubled_title)
    print(f"     sample before: {sample[:60]!r}")
    print(f"     sample after : {df['title'].iloc[0]!r}")
    row_report(before, len(df), "no rows dropped — transformation only")

    # Step 2 — strip Nexis boilerplate ───────────────────────────────────────
    step(2, "Strip Nexis boilerplate from body", "climate")
    before = len(df)
    wc_before = df["body"].dropna().str.split().str.len().median()
    df["body"] = df["body"].map(strip_boilerplate)
    wc_after  = df["body"].dropna().str.split().str.len().median()
    print(f"     median body words: {wc_before:.0f} → {wc_after:.0f}")
    row_report(before, len(df), "no rows dropped — transformation only")

    # Step 3 — parse dates; clip to shared time window ───────────────────────
    step(3, "Parse dates + clip to 2014-01-01 → 2023-12-31", "climate")
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    null_dates = df["date"].isnull().sum()
    print(f"     null dates after parse: {null_dates:,}")
    df = df[df["date"].notna()].copy()
    row_report(before, len(df), "null dates dropped")
    before = len(df)
    df = df[(df["date"] >= DATE_MIN) & (df["date"] <= DATE_MAX)].copy()
    row_report(before, len(df), "outside 2014–2023 dropped")
    print(f"     date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Step 4 — normalize outlet names ────────────────────────────────────────
    step(4, "Normalize outlet → outlet_clean", "climate")
    before = len(df)
    df["outlet_clean"] = df["outlet"].map(outlet_map)
    unmapped = df["outlet_clean"].isnull().sum()
    print(f"     mapping used: {outlet_map}")
    print(f"     unmapped values (will be dropped in step 5): {unmapped:,}")
    row_report(before, len(df), "no rows dropped yet")

    # Step 5 — filter to agreed outlet set ───────────────────────────────────
    step(5, "Filter to agreed outlet set", "climate")
    before = len(df)
    dropped_outlets = df[~df["outlet_clean"].isin(keep_outlets)]["outlet"].value_counts()
    if len(dropped_outlets):
        print(f"     outlets being dropped:")
        for o, n in dropped_outlets.items():
            print(f"       {o}: {n:,}")
    df = df[df["outlet_clean"].isin(keep_outlets)].copy()
    row_report(before, len(df), f"keep_outlets={sorted(keep_outlets)}")

    # Step 6 — compute word_count ────────────────────────────────────────────
    step(6, "Compute word_count from cleaned body", "climate")
    before = len(df)
    df["word_count"] = word_count(df["body"])
    print(f"     word_count: min={df['word_count'].min()}  "
          f"median={df['word_count'].median():.0f}  max={df['word_count'].max()}")
    row_report(before, len(df), "no rows dropped — computation only")

    # Step 7 — drop short articles ───────────────────────────────────────────
    step(7, f"Drop articles with word_count < {MIN_WORD_COUNT}", "climate")
    before = len(df)
    df = df[df["word_count"] >= MIN_WORD_COUNT].copy()
    row_report(before, len(df), f"word_count < {MIN_WORD_COUNT}")

    # Recalculate word_count on the filtered set (removes any rounding artefacts)
    df["word_count"] = word_count(df["body"])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS MIGRATION
# ══════════════════════════════════════════════════════════════════════════════

def process_migration(
    brand_map:    Dict[str, str],
    keep_outlets: Set[str],
) -> pd.DataFrame:
    sep("MIGRATION — processing pipeline")
    print(f"  Source: {MIGRATION_RAW}  (large file — reading in chunks)")

    chunks = []
    for chunk in pd.read_csv(MIGRATION_RAW, chunksize=10_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    n0 = len(df)
    print(f"  Raw rows: {n0:,}  |  columns: {list(df.columns)}")

    # Drop stray index column immediately if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("  Dropped stray 'Unnamed: 0' index column.")

    # Step 1 — fix doubled titles ────────────────────────────────────────────
    step(1, "Fix doubled titles", "migration")
    before = len(df)
    sample = df["title"].iloc[0]
    df["title"] = df["title"].map(fix_doubled_title)
    print(f"     sample before: {sample[:60]!r}")
    print(f"     sample after : {df['title'].iloc[0]!r}")
    row_report(before, len(df), "no rows dropped — transformation only")

    # Step 2 — strip Nexis boilerplate ───────────────────────────────────────
    step(2, "Strip Nexis boilerplate from body", "migration")
    before = len(df)
    wc_before = df["body"].dropna().str.split().str.len().median()
    df["body"] = df["body"].map(strip_boilerplate)
    wc_after  = df["body"].dropna().str.split().str.len().median()
    print(f"     median body words: {wc_before:.0f} → {wc_after:.0f}")
    row_report(before, len(df), "no rows dropped — transformation only")

    # Step 3 — parse dates; drop nulls; clip ─────────────────────────────────
    step(3, "Parse dates + drop nulls + clip to 2014-01-01 → 2023-12-31", "migration")
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    null_dates = df["date"].isnull().sum()
    print(f"     null dates after parse: {null_dates:,}")
    df = df[df["date"].notna()].copy()
    row_report(before, len(df), "null dates dropped")
    before = len(df)
    df = df[(df["date"] >= DATE_MIN) & (df["date"] <= DATE_MAX)].copy()
    row_report(before, len(df), "outside 2014–2023 dropped")
    print(f"     date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Step 4 — normalize outlet names via news_brand ─────────────────────────
    step(4, "Normalize news_brand → outlet_clean", "migration")
    before = len(df)
    df["outlet_clean"] = df["news_brand"].map(brand_map)
    unmapped = df["outlet_clean"].isnull().sum()
    print(f"     brand mapping used: {brand_map}")
    print(f"     unmapped news_brand values: {unmapped:,}")
    row_report(before, len(df), "no rows dropped yet")

    # Step 5 — filter to agreed outlet set ───────────────────────────────────
    step(5, "Filter to agreed outlet set", "migration")
    before = len(df)
    dropped_brands = df[~df["outlet_clean"].isin(keep_outlets)]["news_brand"].value_counts()
    if len(dropped_brands):
        print(f"     brands being dropped:")
        for b, n in dropped_brands.items():
            print(f"       {b}: {n:,}")
    else:
        print(f"     all news_brand values are mapped — 0 rows dropped")
    df = df[df["outlet_clean"].isin(keep_outlets)].copy()
    row_report(before, len(df), f"keep_outlets={sorted(keep_outlets)}")

    # Steps 8 & 9 — drop retired NLP columns and Unnamed artefacts ───────────
    step(8, "Drop retired NLP columns (safe check)", "migration")
    before = len(df)
    present = [c for c in RETIRED_COLS if c in df.columns]
    absent  = [c for c in RETIRED_COLS if c not in df.columns]
    print(f"     dropping ({len(present)}): {present}")
    if absent:
        print(f"     already absent ({len(absent)}): {absent}")
    if present:
        df = df.drop(columns=present)
    row_report(before, len(df), "no rows dropped — column removal only")

    step(9, "Drop 'Unnamed: 0' if present", "migration")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("     dropped.")
    else:
        print("     not present — skipped.")

    # Step 10 — deduplication ────────────────────────────────────────────────
    # NOTE: Deduplication runs BEFORE the word_count filter so that the
    # online-vs-print preference is applied on the full post-clip set.
    # This ordering matches the original pipeline and is required to reproduce
    # the verified output row count (68,760).

    step(10, "Dedup: (outlet_clean, date, title) — prefer Online over Print", "migration")
    before = len(df)
    df["_edition"] = df["outlet"].str.extract(r"(Online|Print)", expand=False).fillna("other")
    print(f"     edition breakdown before dedup: {df['_edition'].value_counts().to_dict()}")
    # "Online" < "Print" alphabetically → sort ascending so Online comes first
    df = (
        df.sort_values("_edition")
          .drop_duplicates(subset=["outlet_clean", "date", "title"], keep="first")
          .sort_index()
          .copy()
    )
    df = df.drop(columns=["_edition"])
    row_report(before, len(df), "Print editions superseded by Online dropped")

    step(10, "Dedup: exact-body match (secondary pass)", "migration")
    before = len(df)
    df = df.drop_duplicates(subset=["body"], keep="first").copy()
    row_report(before, len(df), "exact-body duplicates dropped")

    # Step 6 — compute word_count (after dedup, before filter) ───────────────
    step(6, "Compute word_count from cleaned body", "migration")
    before = len(df)
    df["word_count"] = word_count(df["body"])
    print(f"     word_count: min={df['word_count'].min()}  "
          f"median={df['word_count'].median():.0f}  max={df['word_count'].max()}")
    row_report(before, len(df), "no rows dropped — computation only")

    # Step 7 — drop short articles ───────────────────────────────────────────
    step(7, f"Drop articles with word_count < {MIN_WORD_COUNT}", "migration")
    before = len(df)
    df = df[df["word_count"] >= MIN_WORD_COUNT].copy()
    row_report(before, len(df), f"word_count < {MIN_WORD_COUNT}")

    # Final word_count recalculation on clean, filtered set
    df["word_count"] = word_count(df["body"])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CORPUS OVERLAP FLAG  (Step 11)
# ══════════════════════════════════════════════════════════════════════════════

def flag_corpus_overlap(
    clim: pd.DataFrame,
    mig:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sep("Step 11 — Flag corpus_overlap (both corpora)")
    print("  Matching on (outlet_clean, date[YYYY-MM-DD], title.lower().strip())")

    clim = clim.copy()
    mig  = mig.copy()

    clim_keys = set(make_match_key(clim))
    mig_keys  = set(make_match_key(mig))
    overlap   = clim_keys & mig_keys

    print(f"  Unique keys — climate: {len(clim_keys):,}  "
          f"migration: {len(mig_keys):,}")
    print(f"  Overlap (exact match): {len(overlap):,} articles "
          f"appear in both corpora")
    print(f"  These are retained in BOTH files as cross-crisis discourse evidence.")

    clim["corpus_overlap"] = make_match_key(clim).isin(overlap)
    mig["corpus_overlap"]  = make_match_key(mig).isin(overlap)

    print(f"  corpus_overlap=True — climate: {clim['corpus_overlap'].sum():,}  "
          f"migration: {mig['corpus_overlap'].sum():,}")
    return clim, mig


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS LABEL  (Step 12)
# ══════════════════════════════════════════════════════════════════════════════

def add_corpus_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["corpus"] = label
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sep("PREPROCESS_FINAL — full pipeline from raw inputs")
    print(f"  Inputs  : {CLIMATE_RAW.name}, {MIGRATION_RAW.name}")
    print(f"  Outputs : {CLIMATE_OUT.name}, {MIGRATION_OUT.name}")
    print(f"  Window  : {DATE_MIN.date()} → {DATE_MAX.date()}")
    print(f"  Min wc  : {MIN_WORD_COUNT} words")

    # Recover mappings from existing clean files (or use fallbacks)
    sep("Recovering outlet/brand mappings from existing clean files")
    climate_outlet_map, keep_outlets_c = _recover_climate_map()
    migration_brand_map, keep_outlets_m = _recover_migration_map()

    # Verify both mapping sets agree on which outlets to keep
    if keep_outlets_c != keep_outlets_m:
        print(f"  ⚠  Outlet sets differ between files:")
        print(f"     climate  : {sorted(keep_outlets_c)}")
        print(f"     migration: {sorted(keep_outlets_m)}")
        keep_outlets = keep_outlets_c | keep_outlets_m
        print(f"     Using union: {sorted(keep_outlets)}")
    else:
        keep_outlets = keep_outlets_c
        print(f"  Keep outlets (both files agree): {sorted(keep_outlets)}")

    # Process each corpus
    clim = process_climate(climate_outlet_map, keep_outlets)
    mig  = process_migration(migration_brand_map, keep_outlets)

    # Step 11 — cross-corpus overlap flag
    clim, mig = flag_corpus_overlap(clim, mig)

    # Step 12 — corpus label
    sep("Step 12 — Add corpus label")
    clim = add_corpus_label(clim, "climate")
    mig  = add_corpus_label(mig,  "migration")
    print(f"  corpus='climate'   → {len(clim):,} rows")
    print(f"  corpus='migration' → {len(mig):,} rows")

    # Save
    sep("Save outputs")
    clim.to_csv(CLIMATE_OUT,   index=False)
    mig.to_csv(MIGRATION_OUT,  index=False)
    print(f"  {CLIMATE_OUT}   ({len(clim):,} rows)")
    print(f"  {MIGRATION_OUT} ({len(mig):,} rows)")

    sep("PIPELINE COMPLETE")
    print(f"  climate_clean.csv  : {len(clim):,} rows")
    print(f"  migration_clean.csv: {len(mig):,} rows")
    print()


if __name__ == "__main__":
    main()
