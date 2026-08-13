"""
Prompt X0b — Rebuild 100-article benchmark from the X0-filtered pool.

Draws only from articles where on_topic_flag=True AND topic_central_flag=True.
Excludes articles already used in calibration runs (X1/X1b/X0) and the old benchmark.
Uses the same stratified sampling logic as the original Prompt 4.
Overwrites benchmark_annotation_sheet.csv and benchmark_ids.csv.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
SAMPLED_DIR = ROOT / "data" / "sampled"
BENCHMARK_DIR = ROOT / "data" / "benchmark"

RANDOM_SEED = 42
TARGET_N = 100

CENTRALITY_FILES = {
    "climate": "climate_sample_centrality.csv",
    "migration": "migration_sample_centrality.csv",
}
ORIGINAL_SAMPLE_FILES = {
    "climate": "climate_sample_flagged.csv",
    "migration": "migration_sample_topped_up.csv",
}

FRAME_COLS = [
    "conflict_present",
    "human_interest_present",
    "economic_present",
    "deservingness_present",
    "deservingness_direction",
    "responsibility_present",
    "responsibility_responsible_actor",
    "humanitarian_present",
    "security_present",
    "policy_present",
    "scientific_present",
    "crisis_present",
    "solutions_present",
    "victim_present",
    "skepticism_present",
    "securitization_present",
    "othering_present",
    "agency_present",
    "agency_agency_type",
]


def sep(msg: str) -> None:
    print(f"\n{'=' * 68}\n  {msg}\n{'=' * 68}")


def build_exclusion_set(corpus: str, old_bm_source_rows: set[int]) -> set[int]:
    """
    Returns source_row indices (position in on-topic subset) to exclude.
    Covers: old benchmark articles + X1/X1b/X0 calibration articles.
    Calibration used RANDOM_SEED=42, N=13 climate / 12 migration,
    from the on-topic pool minus old benchmark rows — reconstruct here.
    """
    N_CAL = {"climate": 13, "migration": 12}

    orig_df = pd.read_csv(SAMPLED_DIR / ORIGINAL_SAMPLE_FILES[corpus])
    on_topic = orig_df[orig_df["on_topic_flag"] == True].reset_index(drop=True)
    on_topic["source_row"] = range(len(on_topic))

    # Pool that X1/X1b/X0 calibration drew from (non-benchmark on-topic)
    cal_pool = on_topic[~on_topic["source_row"].isin(old_bm_source_rows)]
    cal_sample = cal_pool.sample(n=N_CAL[corpus], random_state=RANDOM_SEED)
    cal_rows = set(cal_sample["source_row"])

    return old_bm_source_rows | cal_rows


def stratified_draw(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    """
    Proportional stratified sample by outlet_clean × year.
    Each stratum gets max(1, round(stratum_size × target_n / pool_size)) articles.
    Falls back to random top-up / trim to hit target_n exactly.
    """
    pool_size = len(df)
    strata = df.groupby(["outlet_clean", "year"])
    draws = []

    for (outlet, year), grp in strata:
        n_draw = max(1, round(len(grp) * target_n / pool_size))
        n_draw = min(n_draw, len(grp))
        draws.append(grp.sample(n=n_draw, random_state=seed))

    result = pd.concat(draws)

    # Trim or top-up to hit target_n exactly
    if len(result) > target_n:
        result = result.sample(n=target_n, random_state=seed)
    elif len(result) < target_n:
        remaining = df[~df.index.isin(result.index)]
        shortfall = target_n - len(result)
        if len(remaining) >= shortfall:
            top_up = remaining.sample(n=shortfall, random_state=seed)
            result = pd.concat([result, top_up])

    return result.reset_index(drop=True)


def main() -> None:
    sep("X0b — Rebuild benchmark from X0-filtered pool")

    # ── load old benchmark for exclusion ─────────────────────────────────────
    old_bm = pd.read_csv(BENCHMARK_DIR / "benchmark_ids.csv")
    old_bm_rows = {
        "climate": set(old_bm[old_bm["corpus"] == "climate"]["source_row"].astype(int)),
        "migration": set(
            old_bm[old_bm["corpus"] == "migration"]["source_row"].astype(int)
        ),
    }
    print(f"\n  Old benchmark: {len(old_bm)} articles")
    print(f"    climate   : {len(old_bm_rows['climate'])} source_rows to exclude")
    print(f"    migration : {len(old_bm_rows['migration'])} source_rows to exclude")

    # ── load filtered pools ───────────────────────────────────────────────────
    pools = {}
    for corpus in ["climate", "migration"]:
        df = pd.read_csv(SAMPLED_DIR / CENTRALITY_FILES[corpus])
        # on_topic=True AND topic_central_flag=True
        central = df[
            (df["on_topic_flag"] == True) & (df["topic_central_flag"] == True)
        ].copy()
        central = central.reset_index(drop=True)
        central["source_row"] = range(len(central))  # position in the central subset

        # Build exclusion set (old benchmark + calibration articles)
        # Need to map exclusion source_rows (in on_topic subset) to central subset
        # Re-derive which articles are being excluded by matching on row identity
        orig_df = pd.read_csv(SAMPLED_DIR / ORIGINAL_SAMPLE_FILES[corpus])
        on_topic = orig_df[orig_df["on_topic_flag"] == True].reset_index(drop=True)
        on_topic["on_topic_row"] = range(len(on_topic))

        excl_on_topic_rows = build_exclusion_set(corpus, old_bm_rows[corpus])

        # Mark exclusions in the central df using the _row_idx column (original df row idx)
        # _row_idx in centrality CSV = original dataframe index
        central["_excl"] = central["_row_idx"].isin(
            on_topic[on_topic["on_topic_row"].isin(excl_on_topic_rows)].index
        )

        pool = central[~central["_excl"]].copy()
        pools[corpus] = pool
        print(
            f"\n  {corpus}: {len(central):,} central articles, "
            f"{central['_excl'].sum()} excluded (old benchmark + calibration) "
            f"→ {len(pool):,} in draw pool"
        )

    # ── corpus allocation proportional to pool size ───────────────────────────
    n_climate = len(pools["climate"])
    n_migration = len(pools["migration"])
    n_total = n_climate + n_migration

    target_climate = round(TARGET_N * n_climate / n_total)
    target_migration = TARGET_N - target_climate

    print(
        f"\n  Pool breakdown: climate={n_climate:,}  migration={n_migration:,}  total={n_total:,}"
    )
    print(
        f"  Benchmark allocation: climate={target_climate}  migration={target_migration}  "
        f"total={target_climate+target_migration}"
    )

    if target_climate < 10:
        print(
            f"  WARNING: climate side only {target_climate} articles — "
            f"thin stratum; note as limitation."
        )

    # ── stratified draw ───────────────────────────────────────────────────────
    sep("Stratified draw")
    bm_parts = {}
    for corpus, target_n in [
        ("climate", target_climate),
        ("migration", target_migration),
    ]:
        pool = pools[corpus]
        drawn = stratified_draw(pool, target_n, RANDOM_SEED)
        bm_parts[corpus] = drawn
        print(f"\n  {corpus}: drew {len(drawn)} articles")
        oy = drawn.groupby(["outlet_clean", "year"]).size().reset_index(name="n")
        for _, r in oy.iterrows():
            print(f"    {r['outlet_clean']:<20} {int(r['year'])}  n={r['n']}")

    # ── build output dataframes ───────────────────────────────────────────────
    sep("Building output files")

    bm_idx = 0
    ids_rows = []
    sheet_rows = []

    for corpus in ["climate", "migration"]:
        drawn = bm_parts[corpus]
        for _, row in drawn.iterrows():
            # benchmark_ids.csv: use source_row = position in central subset for this corpus
            ids_rows.append(
                {
                    "benchmark_idx": bm_idx,
                    "corpus": corpus,
                    "outlet_clean": row["outlet_clean"],
                    "year": int(row["year"]),
                    "date": row.get("date", ""),
                    "source_row": int(
                        row["source_row"]
                    ),  # position in central-filtered subset
                }
            )
            # annotation sheet: blank frame columns
            sheet_row = {
                "benchmark_idx": bm_idx,
                "corpus": corpus,
                "outlet_clean": row["outlet_clean"],
                "year": int(row["year"]),
                "date": row.get("date", ""),
                "title": row.get("title", ""),
                "body": row.get("body", ""),
            }
            for fc in FRAME_COLS:
                sheet_row[fc] = ""
            sheet_row["notes"] = ""
            sheet_rows.append(sheet_row)
            bm_idx += 1

    ids_df = pd.DataFrame(ids_rows)
    sheet_df = pd.DataFrame(sheet_rows)

    # ── save ──────────────────────────────────────────────────────────────────
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    ids_path = BENCHMARK_DIR / "benchmark_ids.csv"
    sheet_path = BENCHMARK_DIR / "benchmark_annotation_sheet.csv"

    ids_df.to_csv(ids_path, index=False)
    sheet_df.to_csv(sheet_path, index=False)

    print(f"\n  benchmark_ids.csv          → {len(ids_df)} rows  saved")
    print(f"  benchmark_annotation_sheet.csv → {len(sheet_df)} rows  saved")

    # ── deprecate old benchmark stability runs ─────────────────────────────────
    stab_dir = ROOT / "data" / "annotated" / "benchmark_stability"
    deprecated = []
    if stab_dir.exists():
        for f in sorted(stab_dir.iterdir()):
            if f.suffix == ".csv":
                new_name = f.with_name("DEPRECATED_" + f.name)
                f.rename(new_name)
                deprecated.append(new_name.name)
    if deprecated:
        print(f"\n  Deprecated old stability runs (renamed with DEPRECATED_ prefix):")
        for fn in deprecated:
            print(f"    {fn}")
    else:
        print(f"\n  No old stability run files found to deprecate.")

    # ── summary breakdown ─────────────────────────────────────────────────────
    sep("NEW BENCHMARK SUMMARY")
    print(f"\n  Total articles: {len(ids_df)}")
    for corpus in ["climate", "migration"]:
        sub = ids_df[ids_df["corpus"] == corpus]
        print(f"\n  {corpus.upper()} — {len(sub)} articles")
        oy_df = (
            sheet_df[sheet_df["corpus"] == corpus]
            .groupby(["outlet_clean", "year"])
            .size()
            .reset_index(name="n")
        )
        by_outlet = (
            oy_df.groupby("outlet_clean")["n"].sum().sort_values(ascending=False)
        )
        for outlet, n in by_outlet.items():
            print(f"    {outlet:<20} {n}")

    print(f"\n  NOTE: source_row in benchmark_ids.csv now refers to position in the")
    print(
        f"  X0-filtered (topic_central_flag=True) subset, not the on_topic_flag subset."
    )
    print(f"  Update any downstream scripts that load benchmark articles to use")
    print(f"  the centrality CSV files as the source, not the original sample files.")
    print()


if __name__ == "__main__":
    main()
