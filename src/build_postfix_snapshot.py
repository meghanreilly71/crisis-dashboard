#!/usr/bin/env python3
"""Build human_postfix.csv — the post-codebook-fix reference standard.

Lineage (as specified):
    human_first_pass_n114.csv                       (114 articles, no model exposure)
      + codebook_recheck_worksheet.xlsx outcomes    (re-application of the coder's own
                                                     judgement to changed definitions)
      + enforced agency derivation                  (agency_present = agency_type != "none")
    = human_postfix.csv

Three sources of change, tracked separately:

  1. EXPLICIT REVISION — the coder filled "[REVISE] revised present".
  2. AFFIRMATION — the coder left it blank, meaning "the call shown still stands".
     The value shown in the worksheet is authoritative for every row the worksheet
     covered, because those are the cells she actually re-read.
  3. DERIVATION — agency_present is recomputed from agency_type per the restructured
     codebook rule. This is mechanical, not a judgement call.

human_first_pass.csv, human_first_pass_n114.csv and human_revision1.csv are left
untouched. This is a third, distinct snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "benchmark" / "snapshots"
BM = ROOT / "data" / "benchmark"
WORKSHEET = ROOT / "codebook_recheck_worksheet.xlsx"

LABELS = [
    "conflict",
    "human_interest",
    "economic",
    "deservingness",
    "responsibility",
    "securitization",
    "othering",
    "agency",
    "humanitarian",
    "security",
    "policy",
    "scientific",
    "crisis",
    "solutions",
    "victim",
    "skepticism",
]

SHEET_LABEL = {
    "1_scientific": "scientific",
    "2_deservingness": "deservingness",
    "3_responsibility": "responsibility",
    "4_othering": "othering",
}

BINARY = {"yes", "no"}


def norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    return s if s in BINARY else None


def load_agency_types() -> pd.Series:
    from numbers_parser import Document

    rows = (
        Document(str(BM / "benchmark_annotation_sheet.csv.numbers"))
        .sheets[0]
        .tables[0]
        .rows(values_only=True)
    )
    main = pd.DataFrame(rows[1:], columns=list(rows[0]))
    main["benchmark_idx"] = main.benchmark_idx.astype(float).round().astype(int)
    exp = pd.read_csv(
        BM / "benchmark_annotation_sheet_climate_expansion - climate_expansion.csv",
        dtype=str,
    )
    exp["benchmark_idx"] = exp.benchmark_idx.astype(int)
    both = pd.concat(
        [
            main[["benchmark_idx", "agency_agency_type"]],
            exp[["benchmark_idx", "agency_agency_type"]],
        ]
    )
    return both.set_index("benchmark_idx")["agency_agency_type"]


def main() -> None:
    base = pd.read_csv(SNAP / "human_first_pass_n114.csv").set_index("benchmark_idx")
    grid = {
        bi: {
            l: norm(base.loc[bi].get(f"{l}_present")) if bi in base.index else None
            for l in LABELS
        }
        for bi in range(114)
    }

    changes = []  # (bi, label, old, new, kind)

    # ── worksheet outcomes: revisions and affirmations ────────────────────────
    for sheet, label in SHEET_LABEL.items():
        d = pd.read_excel(WORKSHEET, sheet_name=sheet)
        if d.empty:
            continue
        for _, r in d.iterrows():
            bi = int(r["benchmark_idx"])
            revised = norm(r.get("[REVISE] revised present"))
            shown = norm(r.get("my original call"))
            authoritative = revised if revised is not None else shown
            if authoritative is None:
                continue
            old = grid[bi][label]
            if old != authoritative:
                changes.append(
                    (
                        bi,
                        label,
                        old,
                        authoritative,
                        (
                            "explicit revision"
                            if revised is not None
                            else "affirmation of worksheet value"
                        ),
                    )
                )
                grid[bi][label] = authoritative

    # ── enforced agency derivation ────────────────────────────────────────────
    atypes = load_agency_types()
    for bi in range(114):
        at = str(atypes.get(bi) or "").strip().lower()
        if at in ("", "nan"):
            continue
        derived = "no" if at in ("none", "null") else "yes"
        if grid[bi]["agency"] != derived:
            changes.append(
                (
                    bi,
                    "agency",
                    grid[bi]["agency"],
                    derived,
                    f"derivation (agency_type='{at}')",
                )
            )
            grid[bi]["agency"] = derived

    out = pd.DataFrame(grid).T.reset_index().rename(columns={"index": "benchmark_idx"})
    out.columns = ["benchmark_idx"] + [f"{l}_present" for l in LABELS]
    out.to_csv(SNAP / "human_postfix.csv", index=False)

    ch = pd.DataFrame(changes, columns=["benchmark_idx", "label", "old", "new", "kind"])
    ch.to_csv(SNAP / "human_postfix_changelog.csv", index=False)

    # ── report ────────────────────────────────────────────────────────────────
    total_cells = sum(1 for bi in range(114) for l in LABELS if grid[bi][l] is not None)
    print("=" * 78)
    print("  human_postfix.csv — cell changes vs human_first_pass_n114.csv")
    print("=" * 78)
    print(f"  comparable (non-null) cells : {total_cells}")
    print(
        f"  cells changed               : {len(ch)}  "
        f"({len(ch)/total_cells*100:.1f}%)\n"
    )

    print("  by label:")
    print(f"    {'label':16s} {'changed':>8s} {'yes->no':>8s} {'no->yes':>8s}")
    for label in LABELS:
        g = ch[ch.label == label]
        if g.empty:
            continue
        yn = ((g.old == "yes") & (g.new == "no")).sum()
        ny = ((g.old == "no") & (g.new == "yes")).sum()
        print(f"    {label:16s} {len(g):8d} {yn:8d} {ny:8d}")

    print("\n  by kind:")
    for kind, g in ch.groupby(ch.kind.str.split(" (", regex=False).str[0]):
        print(f"    {kind:34s} {len(g):3d}")

    print("\n  full changelog:")
    for _, r in ch.sort_values(["label", "benchmark_idx"]).iterrows():
        print(
            f"    bi={int(r.benchmark_idx):3d} {r.label:16s} "
            f"{str(r.old):4s} -> {r.new:4s}   [{r.kind}]"
        )

    # verification
    print("\n" + "=" * 78)
    bad = 0
    for bi in range(114):
        at = str(atypes.get(bi) or "").strip().lower()
        if at in ("", "nan"):
            continue
        if grid[bi]["agency"] != ("no" if at in ("none", "null") else "yes"):
            bad += 1
    print(
        f"  VERIFY agency_present == (agency_type != 'none') for all 114: "
        f"{'PASS' if bad == 0 else f'FAIL ({bad})'}"
    )
    for f in (
        "human_first_pass.csv",
        "human_first_pass_n114.csv",
        "human_revision1.csv",
    ):
        print(f"  untouched: {f}  ({len(pd.read_csv(SNAP/f))} rows)")


if __name__ == "__main__":
    main()
