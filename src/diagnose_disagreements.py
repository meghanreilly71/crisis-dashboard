#!/usr/bin/env python3
"""Independent recomputation of disagreement direction and exclusion structure.

Reads the human Numbers sheet and promptB_run1.csv directly. Does not read
any previously generated report, sheet, or summary, so its counts are an
independent check rather than a restatement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HUMAN_SHEET = ROOT / "data" / "benchmark" / "benchmark_annotation_sheet.csv.numbers"
RUN_CSV = ROOT / "data" / "annotated" / "benchmark_stability" / "promptB_run1.csv"

SHARED = [
    "conflict_present",
    "human_interest_present",
    "economic_present",
    "deservingness_present",
    "responsibility_present",
    "securitization_present",
    "othering_present",
    "agency_present",
]
MIGRATION = ["humanitarian_present", "security_present", "policy_present"]
CLIMATE = [
    "scientific_present",
    "crisis_present",
    "solutions_present",
    "victim_present",
    "skepticism_present",
]
LABELS = SHARED + MIGRATION + CLIMATE

BINARY = {"yes", "no"}


def load_human() -> pd.DataFrame:
    from numbers_parser import Document

    rows = Document(str(HUMAN_SHEET)).sheets[0].tables[0].rows(values_only=True)
    df = pd.DataFrame(rows[1:], columns=list(rows[0]))
    for c in df.columns:
        df[c] = df[c].map(lambda v: None if v is None else str(v).strip())
    df["benchmark_idx"] = df["benchmark_idx"].astype(float).astype(int)
    return df


def load_llm() -> pd.DataFrame:
    df = pd.read_csv(RUN_CSV, dtype=str)
    for c in df.columns:
        df[c] = df[c].map(lambda v: None if pd.isna(v) else str(v).strip())
    df["benchmark_idx"] = df["article_idx"].astype(int)
    return df


def norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return s if s in BINARY else None


def label_in_scope(label: str, corpus_type: str) -> bool:
    """Whether the pipeline is designed to produce this label for this article."""
    if label in SHARED:
        return True
    if label in MIGRATION:
        return corpus_type in ("migration", "intersection")
    return corpus_type in ("climate", "intersection")


def main() -> None:
    human, llm = load_human(), load_llm()
    merged = human.merge(llm, on="benchmark_idx", suffixes=("_h", "_l"))
    assert len(merged) == 100, len(merged)

    cells = []
    for _, r in merged.iterrows():
        for label in LABELS:
            cells.append(
                {
                    "benchmark_idx": r["benchmark_idx"],
                    "corpus": r["corpus_h"],
                    "corpus_type": r["corpus_type"],
                    "label": label,
                    "human": norm(r.get(f"{label}_h", r.get(label))),
                    "llm": norm(r.get(f"{label}_l", r.get(label))),
                    "human_raw": r.get(f"{label}_h", r.get(label)),
                    "llm_raw": r.get(f"{label}_l", r.get(label)),
                    "in_scope": label_in_scope(label, r["corpus_type"]),
                }
            )
    df = pd.DataFrame(cells)
    total = len(df)

    comparable = df[df.human.notna() & df.llm.notna()]
    noncomparable = df[df.human.isna() | df.llm.isna()]

    print("=" * 74)
    print("  CELL CENSUS  (100 articles x 16 binary labels)")
    print("=" * 74)
    print(f"  total cells          : {total}")
    print(f"  comparable           : {len(comparable)}")
    print(f"  non-comparable       : {len(noncomparable)}")

    print("\n" + "=" * 74)
    print("  DISAGREEMENT DIRECTION  (comparable cells only)")
    print("=" * 74)
    dis = comparable[comparable.human != comparable.llm]
    h_no_l_yes = ((dis.human == "no") & (dis.llm == "yes")).sum()
    h_yes_l_no = ((dis.human == "yes") & (dis.llm == "no")).sum()
    print(f"  agreements                   : {len(comparable) - len(dis)}")
    print(f"  disagreements                : {len(dis)}")
    print(f"  human=no  / LLM=yes          : {h_no_l_yes}")
    print(f"  human=yes / LLM=no           : {h_yes_l_no}")
    print(f"  check (sum == disagreements) : {h_no_l_yes + h_yes_l_no == len(dis)}")

    print("\n  by label:")
    print(f"    {'label':24s} {'n_dis':>6s} {'H=no/L=yes':>11s} {'H=yes/L=no':>11s}")
    for label in LABELS:
        s = dis[dis.label == label]
        a = ((s.human == "no") & (s.llm == "yes")).sum()
        b = ((s.human == "yes") & (s.llm == "no")).sum()
        print(f"    {label.replace('_present',''):24s} {len(s):6d} {a:11d} {b:11d}")

    print("\n" + "=" * 74)
    print("  NON-COMPARABLE BREAKDOWN")
    print("=" * 74)

    structural = noncomparable[
        (~noncomparable.in_scope)
        & noncomparable.human.isna()
        & noncomparable.llm.isna()
    ]
    human_only = noncomparable[noncomparable.human.notna() & noncomparable.llm.isna()]
    llm_only = noncomparable[noncomparable.human.isna() & noncomparable.llm.notna()]
    both_blank_in_scope = noncomparable[
        noncomparable.in_scope & noncomparable.human.isna() & noncomparable.llm.isna()
    ]

    print(
        f"  A. structural (label out of scope for corpus, both blank) : "
        f"{len(structural)}"
    )
    print(
        f"  B. human annotated, LLM pass not run                      : "
        f"{len(human_only)}"
    )
    print(
        f"  C. LLM annotated, human blank                             : "
        f"{len(llm_only)}"
    )
    print(
        f"  D. in scope but blank on BOTH sides                       : "
        f"{len(both_blank_in_scope)}"
    )
    print(
        f"  ---- sum: {len(structural)+len(human_only)+len(llm_only)+len(both_blank_in_scope)}"
        f"  (should equal {len(noncomparable)})"
    )

    for name, sub in (
        ("B. human annotated, LLM pass not run", human_only),
        ("C. LLM annotated, human blank", llm_only),
        ("D. in scope, blank both sides", both_blank_in_scope),
    ):
        if sub.empty:
            continue
        print(f"\n  --- {name} ---")
        for _, r in sub.sort_values(["label", "benchmark_idx"]).iterrows():
            print(
                f"    bi={r.benchmark_idx:3d} {r.corpus:9s} corpus_type={r.corpus_type:12s} "
                f"{r.label:24s} human={str(r.human_raw):8s} llm={str(r.llm_raw)}"
            )

    print("\n" + "=" * 74)
    print("  BASE RATES on comparable cells  (yes-rate, human vs LLM)")
    print("=" * 74)
    print(
        f"    {'label':24s} {'n':>4s} {'human':>7s} {'LLM':>7s} {'diff':>7s} "
        f"{'pooled':>7s} {'raw agr':>8s}"
    )
    for label in LABELS:
        s = comparable[comparable.label == label]
        hy = (s.human == "yes").mean()
        ly = (s.llm == "yes").mean()
        pooled = ((s.human == "yes").sum() + (s.llm == "yes").sum()) / (2 * len(s))
        raw = (s.human == s.llm).mean()
        print(
            f"    {label.replace('_present',''):24s} {len(s):4d} {hy:7.3f} {ly:7.3f} "
            f"{ly-hy:+7.3f} {pooled:7.3f} {raw:8.3f}"
        )

    df.to_csv(ROOT / "data" / "validity" / "cell_census.csv", index=False)
    print(f"\n  wrote data/validity/cell_census.csv ({total} rows)")


if __name__ == "__main__":
    main()
