#!/usr/bin/env python3
"""Diagnostics for the codebook's presence rule ("present if any indicator = yes").

Two analyses, both on already-recorded data (no API calls, nothing re-run):

  A. Human agreement with the LLM's present=yes, broken down by how many of the
     four sub-indicators supported it.
  B. Counterfactual re-derivation of `present` from the recorded indicator
     answers at thresholds >=1 / >=2 / >=3, with the resulting kappa.

(B) is a diagnostic, NOT a tuning procedure. Selecting a per-frame threshold by
whichever value maximises kappa here would fit the threshold to this n=100
benchmark and make the reported kappa optimistic. Any threshold change must be
justified from the frame's definition and then validated on data not used to
choose it.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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


def load():
    from numbers_parser import Document

    rows = (
        Document(str(ROOT / "data/benchmark/benchmark_annotation_sheet.csv.numbers"))
        .sheets[0]
        .tables[0]
        .rows(values_only=True)
    )
    h = pd.DataFrame(rows[1:], columns=list(rows[0]))
    for c in h.columns:
        h[c] = h[c].map(lambda v: None if v is None else str(v).strip())
    h["bi"] = h.benchmark_idx.astype(float).astype(int)
    l = pd.read_csv(
        ROOT / "data/annotated/benchmark_stability/promptB_run1.csv", dtype=str
    )
    for c in l.columns:
        l[c] = l[c].map(lambda v: None if pd.isna(v) else str(v).strip())
    l["bi"] = l.article_idx.astype(int)
    return h, l


def norm(v):
    return v.lower() if v and str(v).lower() in ("yes", "no") else None


def kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pa, pb = (a == "yes").mean(), (b == "yes").mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (np.nan if np.isclose(pe, 1) else (po - pe) / (1 - pe)), po


def pairs(h, l, label, threshold=None):
    A, B = [], []
    for _, hr in h.iterrows():
        lr = l[l.bi == hr.bi]
        if lr.empty:
            continue
        lr = lr.iloc[0]
        hv, lv = norm(hr.get(label + "_present")), norm(lr.get(label + "_present"))
        raw = lr.get(label + "_indicators")
        if hv is None or lv is None or not raw:
            continue
        if threshold is None:
            B.append(lv)
        else:
            k = sum(1 for v in json.loads(raw).values() if str(v).lower() == "yes")
            B.append("yes" if k >= threshold else "no")
        A.append(hv)
    return A, B


def main():
    h, l = load()
    print("=== B. Presence-rule threshold counterfactual (Prompt B, run 1) ===")
    print("    >=1 reproduces the current codebook rule.\n")
    hdr = f"{'label':16s} {'n':>4s} |" + "|".join(
        f"{'>='+str(k):^17s}" for k in (1, 2, 3)
    )
    print(hdr)
    print(
        f"{'':16s} {'':>4s} |"
        + "|".join(f"{'kappa':>8s}{'raw':>9s}" for _ in (1, 2, 3))
    )
    pooled = {1: ([], []), 2: ([], []), 3: ([], [])}
    for lab in LABELS:
        cells, n = [], 0
        for thr in (1, 2, 3):
            A, B = pairs(h, l, lab, thr)
            n = len(A)
            k, po = kappa(A, B)
            cells.append(
                f"{k:8.3f}{po:9.3f}" if not np.isnan(k) else f"{'n/a':>8s}{po:9.3f}"
            )
            pooled[thr][0].extend(A)
            pooled[thr][1].extend(B)
        print(f"{lab:16s} {n:4d} |" + "|".join(cells))
    cells = []
    for thr in (1, 2, 3):
        k, po = kappa(*pooled[thr])
        cells.append(f"{k:8.3f}{po:9.3f}")
    print(f"\n{'POOLED':16s} {len(pooled[1][0]):4d} |" + "|".join(cells))


if __name__ == "__main__":
    main()
