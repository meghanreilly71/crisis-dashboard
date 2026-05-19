"""
build_corpus.py
Merges climate_clean.csv and migration_clean.csv into a single joint corpus.

Steps:
  1. Load both cleaned files and report their schemas
  2. Identify shared columns vs. file-specific columns
  3. Merge on shared columns only (using corpus to distinguish origin)
  4. Validate critical columns in the merged result
  5. Save to corpus_merged.csv
  6. Print full summary

Run: python3 build_corpus.py
     (Run preprocess_final.py first)
"""

import pandas as pd
from pathlib import Path

ROOT           = Path(__file__).parent.parent
CLIMATE_PATH   = ROOT / "data" / "processed" / "climate_clean.csv"
MIGRATION_PATH  = ROOT / "data" / "processed" / "migration_clean.csv"
MERGED_PATH    = ROOT / "data" / "processed" / "corpus_merged.csv"

# Columns that must be non-null in the final merged corpus.
CRITICAL_COLS = ["body", "date", "outlet_clean", "title", "corpus"]


# ── helpers ───────────────────────────────────────────────────────────────────

def sep(label: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep("BUILD_CORPUS — merge climate + migration into joint corpus")
    print(f"  Inputs : {CLIMATE_PATH.name}, {MIGRATION_PATH.name}")
    print(f"  Output : {MERGED_PATH.name}")

    # ── Step 1: load and report schemas ──────────────────────────────────────
    sep("Step 1 — Load files and inspect schemas")

    clim = pd.read_csv(CLIMATE_PATH)
    mig  = pd.read_csv(MIGRATION_PATH)

    print(f"\n  climate_clean.csv  : {len(clim):,} rows  ×  {len(clim.columns)} columns")
    print(f"  migration_clean.csv: {len(mig):,} rows  ×  {len(mig.columns)} columns")

    clim_cols = set(clim.columns)
    mig_cols  = set(mig.columns)

    # ── Step 2: column alignment report ──────────────────────────────────────
    sep("Step 2 — Column alignment")

    shared      = sorted(clim_cols & mig_cols)
    clim_only   = sorted(clim_cols - mig_cols)
    mig_only    = sorted(mig_cols  - clim_cols)

    print(f"\n  Shared columns ({len(shared)}):")
    for col in shared:
        clim_dtype = clim[col].dtype
        mig_dtype  = mig[col].dtype
        match = "✓" if clim_dtype == mig_dtype else f"⚠ dtype mismatch: climate={clim_dtype}, migration={mig_dtype}"
        print(f"    {col:<30} {match}")

    print(f"\n  Climate-only columns ({len(clim_only)}) — will be NULL for migration rows:")
    for col in clim_only:
        print(f"    {col!r}  (dtype={clim[col].dtype})")

    print(f"\n  Migration-only columns ({len(mig_only)}) — will be NULL for climate rows:")
    for col in mig_only:
        print(f"    {col!r}  (dtype={mig[col].dtype})")

    # ── Step 3: merge on shared columns only ─────────────────────────────────
    sep("Step 3 — Merge on shared columns")

    print(f"\n  Merging on {len(shared)} shared columns: {shared}")

    merged = pd.concat(
        [
            clim[shared].copy(),
            mig[shared].copy(),
        ],
        ignore_index=True,
    )

    print(f"\n  Merged shape: {merged.shape}")
    print(f"  Rows per corpus:")
    for label, n in merged["corpus"].value_counts().sort_index().items():
        print(f"    {label}: {n:,}")

    # ── Step 4: validate critical columns ────────────────────────────────────
    sep("Step 4 — Critical column validation")

    # Parse date for range checks
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    print(f"\n  Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")

    print(f"\n  Null counts in critical columns:")
    found_any = False
    for col in CRITICAL_COLS:
        if col in merged.columns:
            n = int(merged[col].isnull().sum())
            pct = 100 * n / len(merged)
            status = "⚠" if n > 0 else "✓"
            print(f"    {status} {col:<20}: {n:,} nulls ({pct:.2f}%)")
            if n > 0:
                found_any = True
    if not found_any:
        print("    All critical columns are fully populated.")

    # ── corpus_overlap cross-check ────────────────────────────────────────────
    if "corpus_overlap" in shared:
        n_overlap = merged["corpus_overlap"].sum()
        print(f"\n  corpus_overlap=True rows: {n_overlap:,} "
              f"({100*n_overlap/len(merged):.1f}% of merged corpus)")
        print(f"    These articles appear in both climate and migration corpora.")

    # ── Step 5: save ─────────────────────────────────────────────────────────
    sep("Step 5 — Save corpus_merged.csv")
    merged.to_csv(MERGED_PATH, index=False)
    print(f"\n  Saved: {MERGED_PATH}  ({len(merged):,} rows)")

    # ── Step 6: full summary ─────────────────────────────────────────────────
    sep("Step 6 — Full summary")

    print(f"\n  Shape            : {merged.shape}")
    print(f"  Date range       : {merged['date'].min().date()} → {merged['date'].max().date()}")

    print(f"\n  Rows per corpus:")
    for label, n in merged["corpus"].value_counts().sort_index().items():
        print(f"    {label:<12}: {n:,}")

    print(f"\n  Outlet distribution (outlet_clean) — combined:")
    outlet_summary = merged.groupby(["outlet_clean", "corpus"]).size().unstack(fill_value=0)
    outlet_summary["TOTAL"] = outlet_summary.sum(axis=1)
    outlet_summary = outlet_summary.sort_index()
    # Header
    col_labels = list(outlet_summary.columns)
    header = f"  {'Outlet':<16}" + "".join(f"{c:>12}" for c in col_labels)
    print(f"\n{header}")
    print("  " + "-" * (14 + 12 * len(col_labels)))
    for outlet, row in outlet_summary.iterrows():
        line = f"  {outlet:<16}" + "".join(f"{int(row[c]):>12,}" for c in col_labels)
        print(line)

    print(f"\n  Year distribution — combined:")
    year_summary = merged.groupby([merged["date"].dt.year, "corpus"]).size().unstack(fill_value=0)
    year_summary["TOTAL"] = year_summary.sum(axis=1)
    col_labels = list(year_summary.columns)
    header = f"  {'Year':<8}" + "".join(f"{str(c):>12}" for c in col_labels)
    print(f"\n{header}")
    print("  " + "-" * (6 + 12 * len(col_labels)))
    for yr, row in year_summary.iterrows():
        line = f"  {int(yr):<8}" + "".join(f"{int(row[c]):>12,}" for c in col_labels)
        print(line)

    print(f"\n  Columns in merged corpus ({len(merged.columns)}):")
    for col in merged.columns:
        n_null = int(merged[col].isnull().sum())
        null_str = f"  [{n_null:,} nulls]" if n_null else ""
        print(f"    {col!r:<30} dtype={merged[col].dtype}{null_str}")

    sep("BUILD_CORPUS COMPLETE")
    print(f"  Output: {MERGED_PATH}")
    print(f"  Total rows: {len(merged):,}  "
          f"(climate={merged['corpus'].eq('climate').sum():,}  "
          f"migration={merged['corpus'].eq('migration').sum():,})")
    print()


if __name__ == "__main__":
    main()
