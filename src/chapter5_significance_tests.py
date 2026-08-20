#!/usr/bin/env python3
"""Systematic recomputation of every chi-square / Cramer's V result in Chapter 5.

NEW SCRIPT. Reads only data/final/annotated_tidy.csv (+ frame_schema.json).
Writes only data/final/chapter5_significance_tests.csv. Alters nothing else.

Tests are declared as structured data in TESTS below, not hardcoded one-offs, so
the table regenerates if the underlying corpus changes.

────────────────────────────────────────────────────────────────────────────────
CONVENTIONS (established from the data, not assumed)
────────────────────────────────────────────────────────────────────────────────
corpus scope
    "climate corpus"  = corpus == 'climate'   (n=1069, INCLUDING 18 intersection)
    "migration corpus"= corpus == 'migration' (n=2978, INCLUDING  9 intersection)
    This is the scope under which the Chapter 5 prevalence figures reproduce.

meta categories (topic tests only)
    annotated_tidy.csv stores UNNORMALISED topic labels: 'Economy & Finance '
    (n=108) and 'Economy & Finance' (n=4) are distinct, likewise Water Management
    and three Global Politics variants. build_rq1_heatmap.py collapses whitespace
    before plotting, and Figure 5.1 / the §5.1 prose use the collapsed form.
    This script therefore applies the same whitespace normalisation and, for the
    climate corpus, the same NOISE exclusion the heatmap uses. Both are switchable
    (META_NORMALISE / DROP_NOISE) and the effect is reported.

continuity correction
    For 2x2 tables scipy applies Yates by default. The thesis values sit between
    the corrected and uncorrected results, so BOTH are reported for every test and
    neither is presented as canonical. RxC tables (year, outlet, 3-group) take no
    correction; there the two columns are identical by construction.

Cramer's V
    V = sqrt(chi2 / (n * (min(rows, cols) - 1))), computed from the same chi2 that
    produced the p-value in that column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
SCHEMA = ROOT / "data" / "final" / "frame_schema.json"
OUT = ROOT / "data" / "final" / "chapter5_significance_tests.csv"

META_NORMALISE = True
DROP_NOISE = True

# ── declarative test inventory ────────────────────────────────────────────────
# kind: category | year | corpus | corpus_type3 | outlet
# Every chi-square / Cramer's V figure reported in Chapter 5 appears here.
TESTS = [
    # §5.1 topic x frame
    dict(sec="5.1", kind="category", corpus="migration", frame="policy",
         category="Migration & Housing",
         claim="p = .062, V = .036"),
    dict(sec="5.1", kind="category", corpus="climate", frame="responsibility",
         category="Economy & Finance",
         claim="p = .020, V = .074"),
    dict(sec="5.1", kind="category", corpus="migration", frame="human_interest",
         category="Population Trends",
         claim="V = .111, p < .001"),
    dict(sec="5.1", kind="category", corpus="migration", frame="human_interest",
         category="Natural Disasters & Accidents",
         claim="V = .071, p < .001"),
    dict(sec="5.1", kind="category", corpus="migration", frame="humanitarian",
         category="Population Trends",
         claim="p < .001 (no V reported)"),
    # §5.2 temporal (year x frame)
    dict(sec="5.2", kind="year", corpus="climate", frame="human_interest",
         claim="p = .013, V = .143"),
    dict(sec="5.2", kind="year", corpus="migration", frame="security",
         claim="p < .001, V = .211"),
    dict(sec="5.2", kind="year", corpus="migration", frame="othering",
         claim="p < .001, V = .193"),
    dict(sec="5.2", kind="year", corpus="climate", frame="crisis",
         claim="p = .120 (n.s.)"),
    dict(sec="5.2", kind="year", corpus="climate", frame="economic",
         claim="p = .303 (n.s.)"),
    # §5.3.1 cross-corpus shared frames
    dict(sec="5.3.1", kind="corpus", frame="securitization",
         claim="p < .001, V = .259"),
    dict(sec="5.3.1", kind="corpus", frame="othering",
         claim="p < .001, V = .389"),
    dict(sec="5.3.1", kind="corpus", frame="deservingness",
         claim="p < .001, V = .374"),
    dict(sec="5.3.1", kind="corpus", frame="economic",
         claim="p < .001, V = .253"),
    # §5.3.2 three-group intersection comparison
    dict(sec="5.3.2", kind="corpus_type3", frame="securitization",
         claim="p < .001, V = .267"),
    # §5.4 outlet x frame
    dict(sec="5.4", kind="outlet", corpus="migration", frame="economic",
         claim="p < .001, V = .183"),
    dict(sec="5.4", kind="outlet", corpus="climate", frame="economic",
         claim="p < .001, V = .219"),
    dict(sec="5.4", kind="outlet", corpus="migration", frame="humanitarian",
         claim="p < .001, V = .212"),
    dict(sec="5.4", kind="outlet", corpus="climate", frame="human_interest",
         claim="p < .001, V = .213"),
    dict(sec="5.4", kind="outlet", corpus="migration", frame="agency",
         claim="p = .174, V = .051"),
    dict(sec="5.4", kind="outlet", corpus="climate", frame="agency",
         claim="p = .076, V = .098"),
]


def sep(msg: str) -> None:
    print(f"\n{'=' * 100}\n  {msg}\n{'=' * 100}")


def load() -> pd.DataFrame:
    schema = json.loads(SCHEMA.read_text())
    df = pd.read_csv(TIDY, low_memory=False)
    for slug, e in schema.items():
        df[slug] = df[e["present_col"]].map({"yes": 1, "no": 0})
    df["meta_n"] = df["meta"].astype(str)
    if META_NORMALISE:
        df["meta_n"] = df["meta_n"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def stats(tab: np.ndarray) -> dict:
    tab = np.asarray(tab, dtype=float)
    tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {}
    n = tab.sum()
    dfree = min(tab.shape) - 1
    out = {"n": int(n), "rows": tab.shape[0], "cols": tab.shape[1]}
    for corr, tag in ((True, "yates"), (False, "nocorr")):
        chi2, p, dof, exp = chi2_contingency(tab, correction=corr)
        out[f"chi2_{tag}"] = chi2
        out[f"p_{tag}"] = p
        out[f"V_{tag}"] = float(np.sqrt(chi2 / (n * dfree)))
        out["dof"] = dof
        out["expected_min"] = float(exp.min())
    if tab.shape == (2, 2):
        out["p_fisher"] = float(fisher_exact(tab)[1])
    return out


def build_table(df: pd.DataFrame, t: dict) -> tuple[pd.DataFrame, str]:
    f = t["frame"]
    if t["kind"] == "category":
        sub = df[df["corpus"] == t["corpus"]]
        if DROP_NOISE:
            sub = sub[sub["meta_n"] != "NOISE"]
        sub = sub[["meta_n", f]].dropna()
        grp = np.where(sub["meta_n"] == t["category"], t["category"], "all other topics")
        desc = (f"{f} | {t['category']} vs rest of {t['corpus']} corpus")
        return pd.crosstab(pd.Series(grp, index=sub.index), sub[f]), desc
    if t["kind"] == "year":
        sub = df[df["corpus"] == t["corpus"]][["year", f]].dropna()
        return pd.crosstab(sub["year"], sub[f]), f"{f} | year (2014-2023) x {t['corpus']} corpus"
    if t["kind"] == "corpus":
        sub = df[["corpus", f]].dropna()
        return pd.crosstab(sub["corpus"], sub[f]), f"{f} | climate corpus vs migration corpus"
    if t["kind"] == "corpus_type3":
        sub = df[["corpus_type", f]].dropna()
        return pd.crosstab(sub["corpus_type"], sub[f]), \
            f"{f} | intersection vs climate-only vs migration-only"
    if t["kind"] == "outlet":
        sub = df[df["corpus"] == t["corpus"]][["outlet_clean", f]].dropna()
        return pd.crosstab(sub["outlet_clean"], sub[f]), f"{f} | outlet (6) x {t['corpus']} corpus"
    raise ValueError(t["kind"])


def main() -> None:
    df = load()
    sep("INPUT")
    print(f"  {TIDY.relative_to(ROOT)} — {len(df):,} articles")
    print(f"  META_NORMALISE={META_NORMALISE}  DROP_NOISE={DROP_NOISE}")
    if META_NORMALISE:
        raw = df["meta"].astype(str).nunique()
        print(f"  meta categories: {raw} raw -> {df['meta_n'].nunique()} after whitespace collapse")

    rows = []
    for t in TESTS:
        tab, desc = build_table(df, t)
        s = stats(tab.values)
        pct = None
        if t["kind"] == "category":
            r = tab.loc[t["category"]]
            pct = f"{100*r[1.0]/r.sum():.2f}% ({int(r[1.0])}/{int(r.sum())})"
        rows.append(dict(
            section=t["sec"], kind=t["kind"], frame=t["frame"],
            scope=t.get("corpus", "both"), grouping=t.get("category", ""),
            description=desc, thesis_claim=t["claim"],
            n=s["n"], table=f"{s['rows']}x{s['cols']}", dof=s["dof"],
            focal_pct=pct,
            chi2_yates=round(s["chi2_yates"], 4), p_yates=s["p_yates"],
            V_yates=round(s["V_yates"], 4),
            chi2_nocorr=round(s["chi2_nocorr"], 4), p_nocorr=s["p_nocorr"],
            V_nocorr=round(s["V_nocorr"], 4),
            p_fisher=s.get("p_fisher", np.nan),
            expected_min=round(s["expected_min"], 2),
        ))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    sep("RESULTS — all Chapter 5 chi-square / Cramer's V tests")
    hdr = (f"{'§':<7}{'test':<52}{'n':>6}{'tab':>6}"
           f"{'p (Yates)':>12}{'V':>8}{'p (no corr)':>13}{'V':>8}   thesis")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for _, r in res.iterrows():
        pv = lambda p: ("<.0001" if p < 1e-4 else f"{p:.4f}")
        print(f"{r['section']:<7}{r['description'][:51]:<52}{r['n']:>6}{r['table']:>6}"
              f"{pv(r['p_yates']):>12}{r['V_yates']:>8.3f}"
              f"{pv(r['p_nocorr']):>13}{r['V_nocorr']:>8.3f}   {r['thesis_claim']}")

    sep("2x2 TESTS — Fisher's exact alongside chi-square")
    for _, r in res[res["table"] == "2x2"].iterrows():
        print(f"  {r['description'][:60]:<62} min expected={r['expected_min']:>7.2f}  "
              f"Fisher p={r['p_fisher']:.4g}")

    print(f"\n  wrote {OUT.relative_to(ROOT)}  ({len(res)} tests)")


if __name__ == "__main__":
    main()
