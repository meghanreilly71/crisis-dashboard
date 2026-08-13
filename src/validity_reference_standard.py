#!/usr/bin/env python3
"""Reference-standard reporting: first-pass (primary) vs revision 1 (adjudication).

Reads frozen snapshots in data/benchmark/snapshots/ rather than the live
Numbers sheet, so these numbers cannot drift as the working file is edited.

  human_first_pass.csv  — annotation before any revision. PRIMARY validation
                          reference: produced without sight of model output.
  human_revision1.csv   — after the 57-cell reconsideration pass. Reported
                          separately as an adjudication result, never blended
                          into the primary validity figure.

Revision 2 (11 cells: othering x10, conflict x1) was discarded and is not
represented here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "benchmark" / "snapshots"
RUNS = ROOT / "data" / "annotated" / "benchmark_stability"
OUT = ROOT / "data" / "validity"

PRIMARY_RUN = 1
BINARY = {"yes", "no"}

# Labels touched by either revision, reported here in full.
FOCUS = ["crisis", "othering", "responsibility", "conflict"]


def norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    return s if s in BINARY else None


def load_llm(variant: str, run: int) -> pd.DataFrame:
    df = pd.read_csv(RUNS / f"prompt{variant}_run{run}.csv", dtype=str)
    df["benchmark_idx"] = df["article_idx"].astype(int)
    return df


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    n = len(a)
    po = float((a == b).mean())
    pa, pb = float((a == "yes").mean()), float((b == "yes").mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = float("nan") if np.isclose(pe, 1) else (po - pe) / (1 - pe)
    kmax = float("nan") if np.isclose(pe, 1) else ((1 - abs(pa - pb)) - pe) / (1 - pe)

    # Krippendorff alpha, nominal, two raters, complete data
    pooled = np.concatenate([a, b])
    pe_scott = sum((pooled == c).mean() ** 2 for c in ("yes", "no"))
    n_ratings = 2 * n
    if np.isclose(pe_scott, 1):
        alpha = float("nan")
    else:
        pi = 1 - (1 - po) / (1 - pe_scott)
        alpha = 1 - (n_ratings - 1) / n_ratings * (1 - pi)

    return {
        "n": n,
        "human_yes": pa,
        "llm_yes": pb,
        "raw": po,
        "kappa": kappa,
        "alpha": alpha,
        "kappa_max": kmax,
        "pabak": 2 * po - 1,
    }


def compare(human: pd.DataFrame, llm: pd.DataFrame, label: str) -> dict:
    col = f"{label}_present"
    m = human[["benchmark_idx", col]].merge(
        llm[["benchmark_idx", col]], on="benchmark_idx", suffixes=("_h", "_l")
    )
    pairs = [(norm(r[f"{col}_h"]), norm(r[f"{col}_l"])) for _, r in m.iterrows()]
    pairs = [(h, l) for h, l in pairs if h is not None and l is not None]
    return stats(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


def main() -> None:
    sheets = {
        "first-pass": pd.read_csv(SNAP / "human_first_pass.csv"),
        "revision 1": pd.read_csv(SNAP / "human_revision1.csv"),
    }

    rows = []
    for variant in ("A", "B"):
        llm = load_llm(variant, PRIMARY_RUN)
        for name, hs in sheets.items():
            for label in FOCUS:
                rows.append(
                    {
                        "sheet": name,
                        "variant": variant,
                        "label": label,
                        **compare(hs, llm, label),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reference_standard_comparison.csv", index=False)

    f = lambda v: " n/a " if isinstance(v, float) and np.isnan(v) else f"{v:.3f}"
    for variant in ("A", "B"):
        print(f"\n{'='*88}")
        print(
            f"  PROMPT {variant}, run {PRIMARY_RUN}   —   first-pass (primary) vs revision 1 (adjudication)"
        )
        print("=" * 88)
        print(
            f"  {'label':16s} {'sheet':12s} {'n':>4s} {'H yes':>6s} {'L yes':>6s} "
            f"{'raw':>7s} {'kappa':>7s} {'alpha':>7s} {'k_max':>7s} {'PABAK':>7s}"
        )
        for label in FOCUS:
            for name in sheets:
                r = df[
                    (df.variant == variant) & (df.label == label) & (df.sheet == name)
                ].iloc[0]
                print(
                    f"  {label if name=='first-pass' else '':16s} {name:12s} {int(r['n']):4d} "
                    f"{r['human_yes']:6.2f} {r['llm_yes']:6.2f} {f(r['raw']):>7s} "
                    f"{f(r['kappa']):>7s} {f(r['alpha']):>7s} {f(r['kappa_max']):>7s} "
                    f"{f(r['pabak']):>7s}"
                )
            print()

    print(f"  wrote {OUT/'reference_standard_comparison.csv'}")


if __name__ == "__main__":
    main()
