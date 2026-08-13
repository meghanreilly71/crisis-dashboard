#!/usr/bin/env python3
"""FINAL validity: human_postfix (n=114) vs Prompt B run 1 under the final codebook.

Supersedes every earlier version — the first-pass n=100 tables, the revision-1
adjudication figures, the discarded revision-2 numbers, and the pre-fix n=40
discussion that was never run.

Comparable-cell arithmetic is recomputed from scratch for the new corpus
composition (climate is now 40 articles, not 26); the old 1155 figure is not
reused anywhere.

Per-label is the primary reporting unit. Pooled and macro figures are computed
for completeness but are deliberately not presented as the headline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "benchmark" / "snapshots"
RUN = ROOT / "data" / "annotated" / "final_codebook" / "promptB_run1_final.csv"
OUT = ROOT / "data" / "validity"

SHARED = [
    "conflict",
    "human_interest",
    "economic",
    "deservingness",
    "responsibility",
    "securitization",
    "othering",
    "agency",
]
MIGRATION = ["humanitarian", "security", "policy"]
CLIMATE = ["scientific", "crisis", "solutions", "victim", "skepticism"]
LABELS = SHARED + MIGRATION + CLIMATE
SCOPE = {
    **{l: "shared" for l in SHARED},
    **{l: "migration-only" for l in MIGRATION},
    **{l: "climate-only" for l in CLIMATE},
}

N_BOOT, SEED = 2000, 20250810
BINARY = {"yes", "no"}


def norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    return s if s in BINARY else None


def kappa(a, b):
    po = float((a == b).mean())
    pa, pb = float((a == "yes").mean()), float((b == "yes").mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if np.isclose(pe, 1) else (po - pe) / (1 - pe)


def alpha(a, b):
    """Krippendorff's alpha, nominal, two raters, complete data."""
    n = len(a)
    po = float((a == b).mean())
    pooled = np.concatenate([a, b])
    pe = sum((pooled == c).mean() ** 2 for c in ("yes", "no"))
    if np.isclose(pe, 1):
        return float("nan")
    pi = 1 - (1 - po) / (1 - pe)
    m = 2 * n
    return 1 - (m - 1) / m * (1 - pi)


def kappa_max(a, b):
    pa, pb = float((a == "yes").mean()), float((b == "yes").mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if np.isclose(pe, 1) else ((1 - abs(pa - pb)) - pe) / (1 - pe)


def boot_ci(a, b, fn, n_boot=N_BOOT):
    """Percentile bootstrap over articles. Wide at small n by construction."""
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(a[idx], b[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tier(label, n, k, kmax, human_yes, llm_yes, n_pos):
    """Reliability tier. A label is only 'structural' when a structural cause
    actually accounts for the shortfall.

    Small n widens the confidence interval but does not by itself depress kappa,
    so it is recorded as an aggravating factor rather than as the explanation.
    Where kappa_max >= 0.6 the marginals permit good agreement and a low kappa
    means the two coders disagree about WHICH articles — genuine divergence,
    not arithmetic.
    """
    if not np.isnan(k) and k >= 0.6:
        return "clears 0.6", ""

    small_n = f" (aggravated by small subsample n={n}, " f"wide CI)" if n <= 40 else ""

    if not np.isnan(kmax) and kmax < 0.6:
        return (
            "below 0.6 — structural",
            f"base-rate ceiling: kappa_max={kmax:.3f} < 0.6, so 0.6 is "
            f"unreachable given marginals ({human_yes:.2f} vs {llm_yes:.2f})" + small_n,
        )
    if n_pos <= 6:
        return (
            "below 0.6 — structural",
            f"prevalence floor: only {n_pos} positive cases at n={n}; "
            f"kappa unstable at any feasible benchmark size",
        )
    return (
        "below 0.6 — unresolved",
        f"kappa_max={kmax:.3f} permits >=0.6, so the marginals are not the "
        f"constraint — the two coders disagree about which articles qualify" + small_n,
    )


def main() -> None:
    human = pd.read_csv(SNAP / "human_postfix.csv").set_index("benchmark_idx")
    llm = pd.read_csv(RUN).set_index("benchmark_idx")

    # ── comparable-cell census, recomputed from scratch ───────────────────────
    census = []
    for bi in human.index:
        if bi not in llm.index:
            continue
        for l in LABELS:
            h = norm(human.loc[bi].get(f"{l}_present"))
            m = norm(llm.loc[bi].get(f"{l}_present"))
            census.append({"benchmark_idx": bi, "label": l, "human": h, "llm": m})
    cen = pd.DataFrame(census)
    comparable = cen[cen.human.notna() & cen.llm.notna()]

    print("=" * 96)
    print("  CELL CENSUS — recomputed for the n=114 corpus (climate 40, migration 74)")
    print("=" * 96)
    print(f"  articles                : {human.index.nunique()}")
    print(f"  total cells (114 x 16)  : {len(cen)}")
    print(f"  comparable              : {len(comparable)}")
    print(f"  non-comparable          : {len(cen) - len(comparable)}")

    rows = []
    for l in LABELS:
        s = comparable[comparable.label == l]
        a, b = s.human.to_numpy(), s.llm.to_numpy()
        n = len(a)
        if n == 0:
            continue
        k, al, km = kappa(a, b), alpha(a, b), kappa_max(a, b)
        po = float((a == b).mean())
        hy, ly = float((a == "yes").mean()), float((b == "yes").mean())
        n_pos = int((a == "yes").sum())
        lo, hi = boot_ci(a, b, kappa)
        t, why = tier(l, n, k, km, hy, ly, n_pos)
        rows.append(
            {
                "label": l,
                "scope": SCOPE[l],
                "n": n,
                "n_human_yes": n_pos,
                "human_yes": hy,
                "llm_yes": ly,
                "raw_agreement": po,
                "cohens_kappa": k,
                "krippendorff_alpha": al,
                "kappa_max": km,
                "pabak": 2 * po - 1,
                "kappa_ci_low": lo,
                "kappa_ci_high": hi,
                "tier": t,
                "tier_reason": why,
            }
        )
    df = pd.DataFrame(rows).sort_values("cohens_kappa", ascending=False)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "validity_FINAL.csv", index=False)

    f = lambda v: " n/a " if isinstance(v, float) and np.isnan(v) else f"{v:.3f}"
    print("\n" + "=" * 96)
    print("  FINAL VALIDITY — human_postfix (n=114) vs Prompt B run 1, final codebook")
    print("=" * 96)
    print(
        f"  {'label':16s}{'scope':15s}{'n':>4s}{'H yes':>7s}{'L yes':>7s}"
        f"{'raw':>7s}{'kappa':>8s}{'alpha':>8s}{'k_max':>7s}{'95% CI':>17s}  tier"
    )
    for _, r in df.iterrows():
        ci = (
            f"[{r.kappa_ci_low:.2f},{r.kappa_ci_high:.2f}]"
            if not np.isnan(r.kappa_ci_low)
            else "  —  "
        )
        print(
            f"  {r.label:16s}{r.scope:15s}{int(r.n):4d}{r.human_yes:7.2f}"
            f"{r.llm_yes:7.2f}{f(r.raw_agreement):>7s}{f(r.cohens_kappa):>8s}"
            f"{f(r.krippendorff_alpha):>8s}{f(r.kappa_max):>7s}{ci:>17s}  {r.tier}"
        )

    print("\n  tier reasons:")
    for _, r in df[df.tier != "clears 0.6"].iterrows():
        print(f"    {r.label:16s} {r.tier_reason}")

    print(
        "\n  tier counts: "
        + ", ".join(f"{k}={v}" for k, v in df.tier.value_counts().items())
    )

    pa = comparable.human.to_numpy()
    pb = comparable.llm.to_numpy()
    print(
        f"\n  [secondary, not headline] pooled kappa={kappa(pa,pb):.3f}  "
        f"raw={float((pa==pb).mean()):.3f}  "
        f"macro kappa={np.nanmean(df.cohens_kappa):.3f}"
    )
    print(f"\n  wrote {OUT/'validity_FINAL.csv'}")


if __name__ == "__main__":
    main()
