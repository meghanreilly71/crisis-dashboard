"""Join title + body from the raw corpora onto the annotated set.

Writes data/final/article_table_data.csv, keyed by article_key, covering only
the 4,047 rows of annotated_tidy.csv. ADDITIVE — annotated_tidy.csv and the four
RQ outputs are never opened for writing.

Join key is (outlet, date, title) with fix_doubled_title applied to the raw side,
the same key verified earlier for topic joining: 100% match and globally unique
on climate. Migration's raw file contains genuine duplicate keys, so 85 annotated
articles match more than one raw row. Those get an exact word_count tiebreak
against the sampled file — the same mechanism used to confirm the duplicate Trouw
article — and anything still ambiguous is written with body = null and
body_ambiguous = True rather than guessed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
OUT = ROOT / "data" / "final" / "article_table_data.csv"
SAMPLES = {
    "climate": ROOT / "data" / "sampled" / "climate_sample_centrality.csv",
    "migration": ROOT / "data" / "sampled" / "migration_sample_centrality.csv",
}
RAW = {
    "climate": ROOT / "data" / "raw" / "climate.csv",
    "migration": ROOT / "data" / "raw" / "migration.csv",
}

_spec = importlib.util.spec_from_file_location("pp", ROOT / "src" / "preprocess_final.py")
_pp = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_pp)
except SystemExit:
    pass
fix_doubled_title = _pp.fix_doubled_title
strip_boilerplate = _pp.strip_boilerplate
word_count_fn = _pp.word_count


def keyify(df: pd.DataFrame, outlet_col: str, title_col: str) -> pd.Series:
    d = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return (df[outlet_col].astype(str).str.strip() + "|" + d.astype(str) + "|"
            + df[title_col].astype(str).str.strip().str.lower())


def load_raw(corpus: str) -> pd.DataFrame:
    cols = ["Unnamed: 0", "outlet", "date", "title", "body", "word_count"]
    if corpus == "climate":
        raw = pd.read_csv(RAW[corpus], usecols=cols, low_memory=False)
    else:
        raw = pd.concat(
            list(pd.read_csv(RAW[corpus], usecols=cols, chunksize=50_000,
                             low_memory=False)),
            ignore_index=True)
    raw["title_fixed"] = raw["title"].map(fix_doubled_title)
    raw["_k"] = keyify(raw, "outlet", "title_fixed")
    # The raw word_count column predates boilerplate stripping, so it does NOT
    # equal the sampled file's word_count (recomputed from the cleaned body in
    # preprocess_final). Reproduce the cleaned body and its word count here so
    # the tiebreak compares like with like.
    raw["body_clean"] = raw["body"].map(strip_boilerplate)
    raw["wc_derived"] = word_count_fn(raw["body_clean"])
    return raw


def resolve(cand: pd.DataFrame, samp_body: str, samp_wc) -> tuple:
    """Pick the body for an ambiguous key. Exact only — never a guess.

    Resolved whenever the surviving candidates all carry the SAME body text,
    because then the body is determined regardless of which raw row is the true
    source row. Returns (title, body, method) or (None, None, None).
    """
    def unique_body(sel: pd.DataFrame):
        if len(sel) and sel["body_clean"].astype(str).str.strip().nunique() == 1:
            return sel.iloc[0]
        return None

    target = str(samp_body).strip()
    hit = unique_body(cand[cand["body_clean"].astype(str).str.strip() == target])
    if hit is not None:
        return hit["title"], hit["body_clean"], "body_exact"
    hit = unique_body(cand[cand["wc_derived"] == samp_wc])
    if hit is not None:
        return hit["title"], hit["body_clean"], "word_count_derived"
    hit = unique_body(cand)
    if hit is not None:
        return hit["title"], hit["body_clean"], "candidates_identical"
    return None, None, None


def main() -> None:
    tidy = pd.read_csv(TIDY, low_memory=False)
    print(f"annotated_tidy.csv: {len(tidy):,} rows (read-only)")

    out_rows = []
    for corpus in ["climate", "migration"]:
        print(f"\n{'=' * 78}\n  {corpus.upper()}\n{'=' * 78}")
        samp = pd.read_csv(SAMPLES[corpus], low_memory=False)
        samp["article_idx"] = range(len(samp))
        sub = tidy[tidy["corpus"] == corpus][["article_key", "article_idx"]]
        # sampled file carries title/body/word_count; article_idx is its row position
        s = sub.merge(samp[["article_idx", "outlet", "date", "title", "body",
                            "word_count"]],
                      on="article_idx", how="left", validate="one_to_one")
        assert s["title"].notna().all(), "sampled join lost rows"
        s["_k"] = keyify(s, "outlet", "title")
        print(f"  annotated rows: {len(s):,}")

        raw = load_raw(corpus)
        counts = raw["_k"].value_counts()
        hits = s["_k"].map(counts).fillna(0).astype(int)
        print(f"  match rate            : {int((hits >= 1).sum()):,}/{len(s):,}")
        print(f"  unambiguous (1 raw)   : {int((hits == 1).sum()):,}")
        print(f"  ambiguous  (>1 raw)   : {int((hits > 1).sum()):,}")
        print(f"  no match              : {int((hits == 0).sum()):,}")

        uniq = raw[~raw["_k"].duplicated(keep=False)].set_index("_k")
        dup = raw[raw["_k"].duplicated(keep=False)]

        methods = {"body_exact": 0, "word_count_derived": 0,
                   "candidates_identical": 0}
        still_ambiguous = 0
        for _, r in s.iterrows():
            k = r["_k"]
            n = counts.get(k, 0)
            body = title = None
            ambiguous = False
            if n == 1:
                body = uniq.loc[k]["body_clean"]
            elif n > 1:
                title, body, how = resolve(dup[dup["_k"] == k], r["body"],
                                           r["word_count"])
                if how is None:
                    ambiguous = True
                    still_ambiguous += 1
                else:
                    methods[how] += 1
            out_rows.append({
                "article_key": r["article_key"],
                "corpus": corpus,
                # ALWAYS the sampled title. The raw title is used only to build
                # the join key (via fix_doubled_title); storing it would put
                # self-concatenated headlines back into the output for the ~1,900
                # articles that preprocessing de-doubled.
                "title": r["title"],
                "body": body,
                "body_ambiguous": ambiguous,
            })

        n_amb = int((hits > 1).sum())
        if n_amb:
            print(f"\n  tiebreak on {n_amb} ambiguous keys:")
            for m, c in methods.items():
                print(f"     {m:<22} {c}")
            print(f"     {'UNRESOLVED':<22} {still_ambiguous}  "
                  f"(body=null, body_ambiguous=True)")

    df = pd.DataFrame(out_rows)
    assert len(df) == len(tidy), f"row count drift: {len(df)} vs {len(tidy)}"
    assert df["article_key"].is_unique, "duplicate article_key"
    assert set(df["article_key"]) == set(tidy["article_key"]), "article_key mismatch"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    mb = OUT.stat().st_size / 1048576
    print(f"\n{'=' * 78}")
    print(f"  wrote {OUT}")
    print(f"     rows            : {len(df):,}")
    print(f"     body non-null   : {int(df['body'].notna().sum()):,}")
    print(f"     body_ambiguous  : {int(df['body_ambiguous'].sum()):,}")
    print(f"     file size       : {mb:.1f} MB")
    print(f"     median body len : "
          f"{int(df['body'].dropna().str.len().median()):,} chars")


if __name__ == "__main__":
    main()
