import math
import pandas as pd
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_FRAC = 0.10

ROOT = Path(__file__).parent.parent
CLIMATE_IN = ROOT / "data" / "processed" / "climate_clean.csv"
MIGRATION_IN = ROOT / "data" / "processed" / "migration_clean.csv"
CLIMATE_OUT = ROOT / "data" / "sampled" / "climate_sample.csv"
MIGRATION_OUT = ROOT / "data" / "sampled" / "migration_sample.csv"


# ── helpers ───────────────────────────────────────────────────────────────────


def sep(label: str) -> None:
    width = 68
    print(f"\n{'=' * width}\n  {label}\n{'=' * width}")


def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Sample `frac` of each year × outlet_clean stratum, min 1 row per stratum."""
    strata = df.groupby(["year", "outlet_clean"], sort=False)
    parts = []
    for _, group in strata:
        n = max(1, round(len(group) * frac))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sort_index()


def print_report(name: str, raw: pd.DataFrame, sample: pd.DataFrame) -> None:
    pct = 100 * len(sample) / len(raw)
    print(f"\n  {name}")
    print(f"    Raw rows     : {len(raw):>8,}")
    print(f"    Sampled rows : {len(sample):>8,}  ({pct:.2f}%)")

    print(f"\n    By outlet_clean:")
    outlet_counts = sample["outlet_clean"].value_counts().sort_index()
    for outlet, n in outlet_counts.items():
        print(f"      {outlet:<20} {n:>6,}")

    print(f"\n    By year:")
    year_counts = sample["year"].value_counts().sort_index()
    for year, n in year_counts.items():
        print(f"      {year}  {n:>6,}")

    overlap_n = (
        sample["corpus_overlap"].sum() if "corpus_overlap" in sample.columns else "N/A"
    )
    print(
        f"\n    corpus_overlap=True : {overlap_n:>6,}"
        if isinstance(overlap_n, int)
        else f"\n    corpus_overlap=True : {overlap_n}"
    )

    label_nulls = sample["label"].isnull().sum() if "label" in sample.columns else "N/A"
    meta_nulls = sample["meta"].isnull().sum() if "meta" in sample.columns else "N/A"
    print(f"    label nulls         : {label_nulls}")
    print(f"    meta  nulls         : {meta_nulls}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    sep("SAMPLE — stratified 10% sample from preprocessed corpora")
    print(f"  Inputs  : {CLIMATE_IN.name}, {MIGRATION_IN.name}")
    print(f"  Outputs : {CLIMATE_OUT.name}, {MIGRATION_OUT.name}")
    print(f"  Fraction: {SAMPLE_FRAC:.0%}  |  Seed: {RANDOM_SEED}")
    print(f"  Strata  : year × outlet_clean  (min 1 row per stratum)")

    CLIMATE_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── climate ───────────────────────────────────────────────────────────────
    sep("Loading climate_clean.csv")
    clim = pd.read_csv(CLIMATE_IN, parse_dates=["date"])
    clim["year"] = clim["date"].dt.year
    print(f"  Rows: {len(clim):,}  |  columns: {list(clim.columns)}")

    sep("Sampling climate corpus")
    clim_sample = stratified_sample(clim, SAMPLE_FRAC, RANDOM_SEED)

    # ── migration ─────────────────────────────────────────────────────────────
    sep("Loading migration_clean.csv")
    mig = pd.read_csv(MIGRATION_IN, parse_dates=["date"])
    mig["year"] = mig["date"].dt.year
    print(f"  Rows: {len(mig):,}  |  columns: {list(mig.columns)}")

    sep("Sampling migration corpus")
    mig_sample = stratified_sample(mig, SAMPLE_FRAC, RANDOM_SEED)

    # ── save ──────────────────────────────────────────────────────────────────
    sep("Saving samples")
    clim_sample.to_csv(CLIMATE_OUT, index=False)
    mig_sample.to_csv(MIGRATION_OUT, index=False)
    print(f"  {CLIMATE_OUT}   ({len(clim_sample):,} rows)")
    print(f"  {MIGRATION_OUT} ({len(mig_sample):,} rows)")

    # ── summary report ────────────────────────────────────────────────────────
    sep("Summary report")
    print_report("climate", clim, clim_sample)
    print_report("migration", mig, mig_sample)

    sep("SAMPLING COMPLETE")
    print()


if __name__ == "__main__":
    main()
