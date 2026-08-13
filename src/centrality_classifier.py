"""
Prompt X0 — Centrality classifier: is this article primarily about the topic?

Uses gpt-4o-mini (single call per article) to distinguish articles that are genuinely
about climate change / migration from articles that only mention the topic in passing.
Adds topic_central_flag and centrality_justification columns to new output files.

Modes:
  --mode calibrate   Run ~25 articles, measure real output tokens, report full-corpus cost.
  --mode run         Run all on-topic articles (both corpora). Resume-safe.
  --mode spotcheck   Print spot-check sample for human review (run after --mode run).

Usage:
  python3 src/centrality_classifier.py --mode calibrate
  python3 src/centrality_classifier.py --mode run
  python3 src/centrality_classifier.py --mode spotcheck
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

SAMPLED_DIR = ROOT / "data" / "sampled"
BENCHMARK_DIR = ROOT / "data" / "benchmark"

SAMPLE_FILES = {
    "climate": "climate_sample_flagged.csv",
    "migration": "migration_sample_topped_up.csv",
}
OUTPUT_FILES = {
    "climate": "climate_sample_centrality.csv",
    "migration": "migration_sample_centrality.csv",
}

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 150  # yes/no + one sentence fits comfortably in 150 tokens
MAX_RETRIES = 3
RANDOM_SEED = 42
N_CAL = {"climate": 13, "migration": 12}  # mirrors X1/X1b calibration counts

TOPIC_DESC = {
    "climate": "climate change, global warming, climate policy, or the energy transition",
    "migration": "migration, refugees, asylum seekers, or immigration policy",
}

SYSTEM_PROMPT = """\
You are a media content classifier. Given a Dutch newspaper article, determine whether \
its PRIMARY subject is the specified topic, or whether it only mentions the topic while \
being mainly about something else.

Classify as:
  "yes" — the topic is the main focus: what the article is centrally reporting on or arguing about
  "no"  — the article mainly covers something else (politics, a person, an event, business, \
crime, etc.) and the topic appears only incidentally — as background, context, or a brief mention

The article is in Dutch. Judge the content meaning, not word frequency.

Respond with ONLY a valid JSON object, nothing else:
{"central": "yes", "justification": "one sentence explaining the decision"}\
"""


def build_user_message(corpus: str, title: str, body: str) -> str:
    topic = TOPIC_DESC[corpus]
    return (
        f'Topic to assess: "{topic}"\n\n'
        f"Title: {title}\n\n"
        f"Article body:\n{body}\n\n"
        f'Is this article primarily about "{topic}"?'
    )


def extract_result(raw: str) -> tuple[bool, str]:
    """Parse JSON from model response. Returns (central_flag, justification)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in response: {raw[:200]!r}")
    obj = json.loads(m.group())
    central = str(obj.get("central", "")).strip().lower() == "yes"
    justification = str(obj.get("justification", "")).strip()
    return central, justification


def call_api(
    client: openai.OpenAI, corpus: str, title: str, body: str
) -> tuple[str, int, int]:
    """Returns (raw_content, prompt_tokens, completion_tokens)."""
    user_msg = build_user_message(corpus, title, body)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return (
                resp.choices[0].message.content or "",
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )
        except (openai.RateLimitError, openai.APIError) as exc:
            wait = 2 ** (attempt + 1)
            print(f"    [retry {attempt+1}] {type(exc).__name__}: waiting {wait}s")
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                raise
    raise RuntimeError("Exceeded retries")


def sep(label: str) -> None:
    print(f"\n{'=' * 68}\n  {label}\n{'=' * 68}")


# ── MODE: calibrate ───────────────────────────────────────────────────────────


def mode_calibrate(client: openai.OpenAI) -> None:
    sep("X0 CALIBRATE — measure real gpt-4o-mini output token length")
    print(f"  Model : {MODEL}  ($0.15/M in, $0.60/M out — standard API)")
    print()

    # Exclude benchmark articles (same discipline as X1/X1b)
    bm = pd.read_csv(BENCHMARK_DIR / "benchmark_ids.csv")
    bm_rows = {
        "climate": set(bm[bm["corpus"] == "climate"]["source_row"]),
        "migration": set(bm[bm["corpus"] == "migration"]["source_row"]),
    }

    PRICE_IN = 0.15 / 1e6
    PRICE_OUT = 0.60 / 1e6

    records = []
    total_cost = 0.0

    for corpus in ["climate", "migration"]:
        df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
        on_topic = df[df["on_topic_flag"] == True].reset_index(drop=True)
        on_topic["source_row"] = range(len(on_topic))
        pool = on_topic[~on_topic["source_row"].isin(bm_rows[corpus])]
        sample = pool.sample(n=N_CAL[corpus], random_state=RANDOM_SEED)

        print(f"  {corpus}: pool={len(pool):,}  calibrating on {len(sample)} articles")

        for i, (_, row) in enumerate(sample.iterrows()):
            title = str(row.get("title", "") or "")
            body = str(row.get("body", "") or "")
            raw, ptok, ctok = call_api(client, corpus, title, body)
            cost = ptok * PRICE_IN + ctok * PRICE_OUT
            total_cost += cost
            try:
                central, justif = extract_result(raw)
                flag_str = "central" if central else "not-central"
            except Exception as e:
                flag_str = f"PARSE ERROR: {e}"
            records.append(
                {
                    "corpus": corpus,
                    "prompt_tokens": ptok,
                    "completion_tokens": ctok,
                    "cost_usd": cost,
                    "result": flag_str,
                }
            )
            print(
                f"    [{corpus}:{i+1}]  prompt={ptok:>4}  compl={ctok:>3}  "
                f"${cost:.5f}  → {flag_str}"
            )

    sep("CALIBRATION RESULTS")
    df_cal = pd.DataFrame(records)
    overall = df_cal["completion_tokens"]
    print(f"\n  API calls  : {len(df_cal)}")
    print(f"  Total cost : ${total_cost:.4f}")
    print()
    print(
        f"  Output tokens — mean={overall.mean():.0f}  median={overall.median():.0f}  "
        f"min={overall.min()}  max={overall.max()}"
    )
    print()

    # Full-corpus cost projection
    sep("FULL-CORPUS COST ESTIMATE")
    N_ON_TOPIC = {"climate": 1655, "migration": 5979}
    total_articles = sum(N_ON_TOPIC.values())
    mean_in = float(df_cal["prompt_tokens"].mean())
    mean_out = float(df_cal["completion_tokens"].mean())

    total_in_tok = total_articles * mean_in
    total_out_tok = total_articles * mean_out

    cost_std = total_in_tok * 0.15 / 1e6 + total_out_tok * 0.60 / 1e6
    cost_batch = total_in_tok * 0.075 / 1e6 + total_out_tok * 0.30 / 1e6

    print(
        f"\n  On-topic articles : {total_articles:,}  "
        f"(climate={N_ON_TOPIC['climate']:,}, migration={N_ON_TOPIC['migration']:,})"
    )
    print(f"  Mean input tokens : {mean_in:.0f}")
    print(f"  Mean output tokens: {mean_out:.0f}")
    print()
    print(f"  Standard API cost : ${cost_std:.2f}")
    print(f"  Batch API cost    : ${cost_batch:.2f}")
    print()
    print("  *** Confirm cost above, then run:")
    print("      python3 src/centrality_classifier.py --mode run")


# ── MODE: run ─────────────────────────────────────────────────────────────────


def mode_run(client: openai.OpenAI) -> None:
    sep("X0 RUN — centrality classification on full on-topic pool")

    for corpus in ["climate", "migration"]:
        sep(f"Corpus: {corpus.upper()}")
        df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
        out_path = SAMPLED_DIR / OUTPUT_FILES[corpus]

        # Resume: which rows are already done?
        done_idxs: set[int] = set()
        if out_path.exists():
            done_df = pd.read_csv(out_path, usecols=["_row_idx"])
            done_idxs = set(done_df["_row_idx"].dropna().astype(int))
            print(f"  Resuming: {len(done_idxs):,} already done")

        write_header = not out_path.exists()
        n_done = n_failed = n_skipped = 0
        PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6
        total_cost = 0.0

        for row_idx, row in df.iterrows():
            if row_idx in done_idxs:
                continue

            on_topic = bool(row.get("on_topic_flag", False))
            out_row = row.to_dict()
            out_row["_row_idx"] = row_idx

            if not on_topic:
                out_row["topic_central_flag"] = False
                out_row["centrality_justification"] = "excluded_by_stage1"
                n_skipped += 1
            else:
                title = str(row.get("title", "") or "")
                body = str(row.get("body", "") or "")
                try:
                    raw, ptok, ctok = call_api(client, corpus, title, body)
                    central, justif = extract_result(raw)
                    out_row["topic_central_flag"] = central
                    out_row["centrality_justification"] = justif
                    cost = ptok * PRICE_IN + ctok * PRICE_OUT
                    total_cost += cost
                    n_done += 1
                    if n_done % 100 == 0 or n_done <= 5:
                        flag_str = "central" if central else "not-central"
                        print(
                            f"  [{n_done:>5}]  {corpus}  "
                            f"row={row_idx}  → {flag_str}  (cumul ${total_cost:.2f})"
                        )
                except Exception as exc:
                    print(f"  FAILED row={row_idx}: {exc}")
                    out_row["topic_central_flag"] = None
                    out_row["centrality_justification"] = f"ERROR: {exc}"
                    n_failed += 1

            pd.DataFrame([out_row]).to_csv(
                out_path, mode="a", index=False, header=write_header
            )
            write_header = False

        sep(f"{corpus.upper()} COMPLETE")
        print(f"  Classified (LLM) : {n_done:,}")
        print(f"  Skipped (off-topic): {n_skipped:,}")
        print(f"  Failed           : {n_failed:,}")
        print(f"  LLM cost         : ${total_cost:.4f}")
        print(f"  Output           : {out_path}")

    # Post-run: summary breakdown + benchmark overlap check
    sep("SUMMARY — centrality filter results")
    _print_summary()


def _print_summary() -> None:
    for corpus in ["climate", "migration"]:
        out_path = SAMPLED_DIR / OUTPUT_FILES[corpus]
        if not out_path.exists():
            print(f"  {corpus}: output file not found")
            continue
        df = pd.read_csv(out_path)
        on = df[df["on_topic_flag"] == True]
        central = on[on["topic_central_flag"] == True]
        not_central = on[on["topic_central_flag"] == False]
        print(f"\n  {corpus.upper()}")
        print(f"    on_topic total         : {len(on):,}")
        print(
            f"    topic_central = True   : {len(central):,}  "
            f"({100*len(central)/max(1,len(on)):.1f}%)"
        )
        print(
            f"    topic_central = False  : {len(not_central):,}  "
            f"({100*len(not_central)/max(1,len(on)):.1f}%)"
        )

        # Outlet/year breakdown
        if len(not_central) > 0:
            print(f"\n    Not-central by outlet:")
            oc = not_central.groupby("outlet_clean").size().sort_values(ascending=False)
            for outlet, n in oc.items():
                pct = (
                    100 * n / len(on[on["outlet_clean"] == outlet])
                    if len(on[on["outlet_clean"] == outlet]) > 0
                    else 0
                )
                print(f"      {outlet:<20} {n:>4} excluded  ({pct:.0f}% of outlet)")

    # Benchmark overlap
    bm_path = BENCHMARK_DIR / "benchmark_ids.csv"
    if bm_path.exists():
        bm = pd.read_csv(bm_path)
        print(
            f"\n  BENCHMARK IMPACT (articles that would be excluded by topic_central_flag=False):"
        )
        n_excluded = 0
        for corpus in ["climate", "migration"]:
            out_path = SAMPLED_DIR / OUTPUT_FILES[corpus]
            if not out_path.exists():
                continue
            df = pd.read_csv(out_path)
            on = df[df["on_topic_flag"] == True].reset_index(drop=True)
            on["source_row"] = range(len(on))
            bm_corpus = bm[bm["corpus"] == corpus]
            for _, bm_row in bm_corpus.iterrows():
                src = int(bm_row["source_row"])
                if src < len(on):
                    flag = on.iloc[src]["topic_central_flag"]
                    if flag == False or str(flag).lower() == "false":
                        n_excluded += 1
                        print(
                            f"    EXCLUDED: benchmark_idx={int(bm_row['benchmark_idx'])}  "
                            f"corpus={corpus}  outlet={bm_row['outlet_clean']}  "
                            f"year={int(bm_row['year'])}"
                        )
        total_bm = len(bm)
        print(f"\n  → {n_excluded}/{total_bm} benchmark articles would be excluded")
        print(
            "    *** Decision needed: keep benchmark as-is or rebuild from filtered pool (X0b)?"
        )


# ── MODE: spotcheck ───────────────────────────────────────────────────────────


def mode_spotcheck() -> None:
    sep("X0 SPOTCHECK — articles near decision boundary for human review")
    N_SHOW = 25

    for corpus in ["climate", "migration"]:
        out_path = SAMPLED_DIR / OUTPUT_FILES[corpus]
        if not out_path.exists():
            print(f"  {corpus}: output file not found — run --mode run first")
            continue

        df = pd.read_csv(out_path)
        on = df[df["on_topic_flag"] == True].copy()
        on["kw_total"] = on["title_keyword_count"].fillna(0) + on[
            "body_keyword_count"
        ].fillna(0)

        false_df = on[on["topic_central_flag"] == False].copy()
        true_df = on[on["topic_central_flag"] == True].copy()

        # "Boundary" not-central: highest keyword counts (model said no despite many keywords)
        # "Boundary" central:     lowest keyword counts (model said yes despite few keywords)
        false_boundary = false_df.nlargest(N_SHOW // 2, "kw_total")
        false_other = false_df[~false_df.index.isin(false_boundary.index)].sample(
            min(N_SHOW // 2, max(0, N_SHOW - len(false_boundary))),
            random_state=RANDOM_SEED,
        )
        true_boundary = true_df.nsmallest(N_SHOW // 2, "kw_total")
        true_other = true_df[~true_df.index.isin(true_boundary.index)].sample(
            min(N_SHOW // 2, max(0, N_SHOW - len(true_boundary))),
            random_state=RANDOM_SEED,
        )

        sep(f"SPOTCHECK — {corpus.upper()}")
        print(
            f"  Not-central (topic_central_flag=False) — {len(false_df):,} total, "
            f"showing {len(false_boundary)+len(false_other)} (boundary-first):"
        )
        for _, row in pd.concat([false_boundary, false_other]).iterrows():
            title_short = str(row.get("title", ""))[:80]
            body_excerpt = str(row.get("body", ""))[:200].replace("\n", " ")
            print(
                f"\n  [{row.get('outlet_clean','?')} {int(row.get('year',0))}]  "
                f"kw={int(row.get('kw_total',0))}"
            )
            print(f"  Title  : {title_short}")
            print(f"  Excerpt: {body_excerpt}...")
            print(f"  Model  : {row.get('centrality_justification','')}")

        print(
            f"\n\n  Central (topic_central_flag=True) — {len(true_df):,} total, "
            f"showing {len(true_boundary)+len(true_other)} (boundary-first):"
        )
        for _, row in pd.concat([true_boundary, true_other]).iterrows():
            title_short = str(row.get("title", ""))[:80]
            body_excerpt = str(row.get("body", ""))[:200].replace("\n", " ")
            print(
                f"\n  [{row.get('outlet_clean','?')} {int(row.get('year',0))}]  "
                f"kw={int(row.get('kw_total',0))}"
            )
            print(f"  Title  : {title_short}")
            print(f"  Excerpt: {body_excerpt}...")
            print(f"  Model  : {row.get('centrality_justification','')}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="X0 centrality classifier")
    parser.add_argument(
        "--mode",
        choices=["calibrate", "run", "spotcheck"],
        required=True,
        help="calibrate: cost estimate on 25 articles | run: full corpus | spotcheck: review sample",
    )
    args = parser.parse_args()

    if args.mode == "spotcheck":
        mode_spotcheck()
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    if args.mode == "calibrate":
        mode_calibrate(client)
    elif args.mode == "run":
        mode_run(client)


if __name__ == "__main__":
    main()
