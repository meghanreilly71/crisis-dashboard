"""Stack the two per-corpus annotation CSVs into one tidy analysis frame.

The two files share an identical 53-column schema (verified: same names, same
order, same dtypes). Pass-specific frames are present as columns on BOTH sides
and simply null where the pass did not apply, so a plain concat aligns with no
reindexing.

Two things this script exists to get right:

  * article_idx is independently 0-indexed per corpus, so all 1,655 climate ids
    collide with migration ids. A compound `article_key` = "{corpus}-{idx}"
    (the same convention as the batch custom_ids) keeps them distinct.
  * exactly one article was drawn into both samples and annotated twice. It is
    deduplicated here, not silently left to double-count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
ANNOTATED_DIR = ROOT / "data" / "annotated"
OUT_DIR = ROOT / "data" / "final"
OUT_CSV = OUT_DIR / "annotated_tidy.csv"
OUT_SCHEMA = OUT_DIR / "frame_schema.json"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-meghanreilly-Desktop-thesis/"
    "4e169310-db94-4968-a390-cbc420a368cb/scratchpad"
)

SOURCES = {
    "climate": ANNOTATED_DIR / "climate_promptB_run1.csv",
    "migration": ANNOTATED_DIR / "migration_promptB_run1.csv",
}

# The known cross-corpus duplicate: same article sampled into both corpora and
# annotated independently. Keep the migration row (its topic label is
# informative; the climate topic model assigned NOISE).
DUP_CLIMATE_KEY = "climate-1414"
DUP_MIGRATION_KEY = "migration-3263"


def sep(msg: str) -> None:
    print(f"\n{'=' * 74}\n  {msg}\n{'=' * 74}")


# ── frame schema ──────────────────────────────────────────────────────────────
# Derived once here so no chart script has to re-derive it. Two irregularities
# are handled explicitly rather than by regex:
#
#   * doubled infixes — FRAME_EXTRA_FIELDS names othering's extras
#     "othering_type"/"othering_evidence" and agency's "agency_type", and the
#     flattener emits f"{slug}_{field}", giving othering_othering_type,
#     othering_othering_evidence, agency_agency_type.
#   * _present_model_stated — emitted only for othering and agency, the two
#     frames where derive_presence() overrides the model's own answer. A
#     naive r"(\w+)_present$" match misses these entirely.

PASS_FRAMES = {
    "pass1": ["conflict", "human_interest", "economic", "deservingness",
              "responsibility"],
    "pass2": ["humanitarian", "security", "policy"],
    "pass3": ["scientific", "crisis", "solutions", "victim", "skepticism"],
    "pass4": ["securitization", "othering", "agency"],
}

# Which corpus_type values a pass actually runs for (mirrors
# batch_pipeline.applicable_passes).
PASS_APPLIES_TO = {
    "pass1": ["climate", "migration", "intersection"],
    "pass2": ["migration", "intersection"],
    "pass3": ["climate", "intersection"],
    "pass4": ["climate", "migration", "intersection"],
}

FRAME_EXTRA_FIELDS = {
    "deservingness": ["direction"],
    "responsibility": ["responsible_actor"],
    "agency": ["agency_type"],
    "othering": ["othering_type", "othering_evidence"],
}

# Frames whose presence is derived rather than taken from the model.
MODEL_STATED_FRAMES = ["othering", "agency"]


def build_frame_schema() -> dict:
    schema: dict[str, dict] = {}
    for pass_name, slugs in PASS_FRAMES.items():
        for slug in slugs:
            extras = [f"{slug}_{f}" for f in FRAME_EXTRA_FIELDS.get(slug, [])]
            model_stated = (
                f"{slug}_present_model_stated" if slug in MODEL_STATED_FRAMES else None
            )
            entry = {
                "pass": pass_name,
                "applies_to": PASS_APPLIES_TO[pass_name],
                "present_col": f"{slug}_present",
                "indicators_col": f"{slug}_indicators",
                "extra_cols": extras,
                "model_stated_col": model_stated,
            }
            entry["all_cols"] = (
                [entry["present_col"], entry["indicators_col"]]
                + extras
                + ([model_stated] if model_stated else [])
            )
            schema[slug] = entry
    return schema


FRAME_SCHEMA = build_frame_schema()


def main() -> None:
    sep("TASK 1 — stack with compound key")
    frames = []
    for corpus, path in SOURCES.items():
        df = pd.read_csv(path, low_memory=False)
        annotated = df["skipped_reason"].isna()
        print(f"  {corpus:<10} {len(df):>6,} rows | annotated {int(annotated.sum()):>6,} "
              f"| skipped {int((~annotated).sum()):>6,} "
              f"{df.loc[~annotated, 'skipped_reason'].value_counts().to_dict()}")
        frames.append(df[annotated].copy())

    cols_a, cols_b = list(frames[0].columns), list(frames[1].columns)
    assert cols_a == cols_b, "schemas diverged — refusing to concat"
    print(f"\n  schema check: identical 53-col schema confirmed ({len(cols_a)} cols)")

    tidy = pd.concat(frames, ignore_index=True)
    tidy.insert(0, "article_key",
                tidy["corpus"].astype(str) + "-" + tidy["article_idx"].astype(int).astype(str))
    print(f"  concatenated: {len(tidy):,} annotated rows, {tidy.shape[1]} cols "
          f"(article_key inserted at position 0)")

    collide = set(frames[0]["article_idx"]) & set(frames[1]["article_idx"])
    print(f"  raw article_idx collisions that article_key resolves: {len(collide):,}")

    sep("TASK 2 — dedupe the cross-corpus duplicate")
    dup_rows = tidy[tidy["article_key"].isin([DUP_CLIMATE_KEY, DUP_MIGRATION_KEY])]
    print(f"  located {len(dup_rows)} rows: {dup_rows['article_key'].tolist()}")
    if len(dup_rows) != 2:
        raise SystemExit(f"expected 2 rows, got {len(dup_rows)} — ids moved, aborting")

    r_c = dup_rows[dup_rows["article_key"] == DUP_CLIMATE_KEY].iloc[0]
    r_m = dup_rows[dup_rows["article_key"] == DUP_MIGRATION_KEY].iloc[0]
    print(f"  climate  : idx={r_c['article_idx']} outlet={r_c['outlet_clean']} "
          f"date={r_c['date']} wc={r_c['word_count']} type={r_c['corpus_type']}")
    print(f"  migration: idx={r_m['article_idx']} outlet={r_m['outlet_clean']} "
          f"date={r_m['date']} wc={r_m['word_count']} type={r_m['corpus_type']}")

    print("\n  re-verifying all 16 frame columns agree:")
    agree = 0
    for slug in FRAME_SCHEMA:
        a, b = r_c[f"{slug}_present"], r_m[f"{slug}_present"]
        same = str(a) == str(b)
        agree += same
        print(f"     {slug:<16} climate={str(a):<5} migration={str(b):<5} "
              f"{'OK' if same else '<-- DIFFERS'}")
    print(f"\n  frame agreement: {agree}/16")
    if agree != 16:
        raise SystemExit("frames disagree — not a safe dedup, aborting")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    dup_rows.to_csv(SCRATCH / "duplicate_rows_preserved.csv", index=False)
    print(f"  both raw rows preserved -> {SCRATCH / 'duplicate_rows_preserved.csv'}")

    before = len(tidy)
    tidy = tidy[tidy["article_key"] != DUP_CLIMATE_KEY].reset_index(drop=True)
    print(f"  dropped {DUP_CLIMATE_KEY} (kept {DUP_MIGRATION_KEY}): "
          f"{before:,} -> {len(tidy):,}")

    sep("TASK 3 — frame schema mapping")
    print(f"  {len(FRAME_SCHEMA)} frames mapped")
    missing = [c for e in FRAME_SCHEMA.values() for c in e["all_cols"]
               if c not in tidy.columns]
    if missing:
        raise SystemExit(f"schema references non-existent columns: {missing}")
    mapped = {c for e in FRAME_SCHEMA.values() for c in e["all_cols"]}
    meta_cols = [c for c in tidy.columns if c not in mapped]
    print(f"  mapped frame columns: {len(mapped)}  |  metadata columns: {len(meta_cols)}")
    print(f"  {len(mapped)} + {len(meta_cols)} = {len(mapped) + len(meta_cols)} "
          f"== {tidy.shape[1]} total  "
          f"{'OK' if len(mapped) + len(meta_cols) == tidy.shape[1] else 'MISMATCH'}")
    for slug in ["deservingness", "othering", "agency", "conflict"]:
        print(f"     {slug:<15} {FRAME_SCHEMA[slug]['all_cols']}")

    sep("TASK 4 — shape verification")
    print(f"  rows: {len(tidy):,}   (expected 4,047)  "
          f"{'OK' if len(tidy) == 4047 else 'MISMATCH'}")
    print(f"  cols: {tidy.shape[1]}")
    print(f"  article_key unique: {tidy['article_key'].is_unique} "
          f"({tidy['article_key'].nunique():,} distinct)")
    print(f"  corpus       : {tidy['corpus'].value_counts().to_dict()}")
    print(f"  corpus_type  : {tidy['corpus_type'].value_counts().to_dict()}")
    print(f"  corpus_overlap=True: {int(tidy['corpus_overlap'].sum()):,}")

    print("\n  parse-gap check:")
    for reason in ["frames_key_not_found", "batch_parse_error"]:
        n = int((tidy["skipped_reason"] == reason).sum())
        print(f"     {reason:<24} {n}  {'OK' if n == 0 else '<-- INVESTIGATE'}")
    print(f"     skipped_reason all-null : {bool(tidy['skipped_reason'].isna().all())}")

    print("\n  null audit on *_present (expected = pass applicability):")
    ok = True
    for slug, e in FRAME_SCHEMA.items():
        expected = int(tidy["corpus_type"].isin(e["applies_to"]).sum())
        actual = int(tidy[e["present_col"]].notna().sum())
        good = expected == actual
        ok &= good
        print(f"     {slug:<16} {e['pass']}  non-null={actual:>5,}  "
              f"expected={expected:>5,}  {'OK' if good else '<-- UNEXPECTED'}")
    print(f"\n  null pattern matches pass applicability exactly: {ok}")

    sep("TASK 5 — save")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT_CSV, index=False)
    OUT_SCHEMA.write_text(json.dumps(FRAME_SCHEMA, indent=2))
    print(f"  wrote {OUT_CSV}  ({len(tidy):,} rows x {tidy.shape[1]} cols)")
    print(f"  wrote {OUT_SCHEMA}  ({len(FRAME_SCHEMA)} frames)")

    back = pd.read_csv(OUT_CSV, low_memory=False)
    print(f"  read-back verify: {len(back):,} rows x {back.shape[1]} cols  "
          f"article_key unique={back['article_key'].is_unique}")

    print("\n  5-row sample:")
    sample = back[["article_key", "corpus", "corpus_type", "outlet_clean", "date",
                   "year", "word_count", "meta", "conflict_present",
                   "humanitarian_present", "scientific_present", "agency_present"]].head(5)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
