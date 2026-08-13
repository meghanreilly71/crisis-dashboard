"""
Prompt X3 — A vs. B comparison + stability sub-study on the rebuilt benchmark sample.

Runs all 100 benchmark articles through both Prompt A and Prompt B, 3 runs each,
using the standard synchronous API. Benchmark articles come from the X0-filtered
(topic_central_flag=True) pool — source_row in benchmark_ids.csv indexes into the
centrality CSV, not the original sample file.

Output:
  data/annotated/benchmark_stability/promptA_run{1,2,3}.csv
  data/annotated/benchmark_stability/promptB_run{1,2,3}.csv

Usage:
  python3 src/benchmark_stability_x3.py                      # all 6 runs
  python3 src/benchmark_stability_x3.py --prompt A --runs 1  # specific
  python3 src/benchmark_stability_x3.py --dry-run            # cost only
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

BENCHMARK_DIR = ROOT / "data" / "benchmark"
SAMPLED_DIR = ROOT / "data" / "sampled"
OUTPUT_DIR = ROOT / "data" / "annotated" / "benchmark_stability"

MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = {"A": 1500, "B": 4000}
MAX_RETRIES = 3

CENTRALITY_FILES = {
    "climate": "climate_sample_centrality.csv",
    "migration": "migration_sample_centrality.csv",
}

# ── pull codebook + both prompts from annotate.py ────────────────────────────
_src = (ROOT / "src" / "annotate.py").read_text()
_patched = _src.replace("Path(__file__).parent.parent", "Path('.')").replace(
    'if __name__ == "__main__":\n    main()', ""
)
_sandbox: dict = {"__file__": str(ROOT / "src" / "annotate.py"), "__name__": "__x3__"}
exec(compile(_patched, "annotate.py", "exec"), _sandbox)

SYSTEM = {"A": _sandbox["SYSTEM_PROMPT_A"], "B": _sandbox["SYSTEM_PROMPT_B"]}
build_user_message = _sandbox["build_user_message"]
flatten_frame_result = _sandbox["flatten_frame_result"]
null_frame_row = _sandbox["null_frame_row"]
extract_json = _sandbox["extract_json"]
derive_corpus_type = _sandbox["derive_corpus_type"]
CARRY_COLS = _sandbox["CARRY_COLS"]

PASSES_DEF = [
    ("pass1", _sandbox["PASS1_DEFS"], _sandbox["PASS1_FRAMES"]),
    ("pass2", _sandbox["PASS2_DEFS"], _sandbox["PASS2_FRAMES"]),
    ("pass3", _sandbox["PASS3_DEFS"], _sandbox["PASS3_FRAMES"]),
    ("pass4", _sandbox["PASS4_DEFS"], _sandbox["PASS4_FRAMES"]),
]


def sep(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 68}\n  [{ts}] {msg}\n{'=' * 68}")


def applicable_passes(row: pd.Series) -> list[str]:
    corpus_type = derive_corpus_type(row)
    passes = ["pass1"]
    if corpus_type in ("migration", "intersection"):
        passes.append("pass2")
    if corpus_type in ("climate", "intersection"):
        passes.append("pass3")
    passes.append("pass4")
    return passes


def call_api(
    client: openai.OpenAI, prompt_variant: str, user_msg: str
) -> tuple[str, int, int]:
    """Returns (content, prompt_tokens, completion_tokens)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS[prompt_variant],
                messages=[
                    {"role": "system", "content": SYSTEM[prompt_variant]},
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


def load_benchmark_articles() -> pd.DataFrame:
    """Load article text + metadata for all 100 benchmark articles.

    source_row in benchmark_ids.csv = position in the X0-filtered (topic_central_flag=True)
    subset of each corpus's centrality CSV.
    """
    bm_ids = pd.read_csv(BENCHMARK_DIR / "benchmark_ids.csv")
    bm_sheet = pd.read_csv(
        BENCHMARK_DIR / "benchmark_annotation_sheet.csv",
        usecols=[
            "benchmark_idx",
            "corpus",
            "outlet_clean",
            "year",
            "date",
            "title",
            "body",
        ],
    )
    rows = []
    for _, bm_row in bm_ids.iterrows():
        corpus = bm_row["corpus"]
        src_row = int(bm_row["source_row"])

        # Load from centrality file, filter to central-only subset
        cent = pd.read_csv(SAMPLED_DIR / CENTRALITY_FILES[corpus])
        central = cent[
            (cent["on_topic_flag"] == True) & (cent["topic_central_flag"] == True)
        ].reset_index(drop=True)
        source_art = central.iloc[src_row] if src_row < len(central) else None

        sheet_row = bm_sheet[bm_sheet["benchmark_idx"] == bm_row["benchmark_idx"]].iloc[
            0
        ]

        rows.append(
            {
                "benchmark_idx": int(bm_row["benchmark_idx"]),
                "article_idx": int(bm_row["benchmark_idx"]),
                "corpus": corpus,
                "outlet_clean": sheet_row["outlet_clean"],
                "year": sheet_row["year"],
                "date": sheet_row["date"],
                "title": sheet_row["title"],
                "body": sheet_row["body"],
                "on_topic_flag": True,
                "corpus_overlap": (
                    bool(source_art["corpus_overlap"])
                    if source_art is not None
                    else False
                ),
                "label": (
                    source_art["label"]
                    if source_art is not None and "label" in source_art
                    else None
                ),
                "meta": (
                    source_art["meta"]
                    if source_art is not None and "meta" in source_art
                    else None
                ),
                "word_count": (
                    source_art["word_count"]
                    if source_art is not None and "word_count" in source_art
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def annotate_one(
    idx: int,
    row: pd.Series,
    client: openai.OpenAI,
    prompt_variant: str,
    api_counter: list[int],
) -> dict:
    body = str(row.get("body", "") or "")
    result: dict[str, Any] = {"corpus_type": derive_corpus_type(row)}
    passes = applicable_passes(row)

    for pass_name, defs, frames in PASSES_DEF:
        if pass_name not in passes:
            result.update(null_frame_row(frames))
            continue
        user_msg = build_user_message(defs, body, frames)
        raw, _, _ = call_api(client, prompt_variant, user_msg)
        api_counter[0] += 1
        try:
            parsed = extract_json(raw)
            result.update(flatten_frame_result(parsed, frames))
        except Exception as e:
            print(f"    [parse error {pass_name}]: {e}")
            result.update(null_frame_row(frames))

    return result


def run_one(
    articles_df: pd.DataFrame,
    client: openai.OpenAI,
    prompt_variant: str,
    run_number: int,
) -> float:
    """Run one prompt/run combination. Returns total API cost."""
    output_path = OUTPUT_DIR / f"prompt{prompt_variant}_run{run_number}.csv"
    failed_path = OUTPUT_DIR / f"failed_prompt{prompt_variant}_run{run_number}.txt"

    # Resume
    completed_idxs: set[int] = set()
    write_header = True
    if output_path.exists():
        existing = pd.read_csv(output_path, usecols=["article_idx"])
        completed_idxs = set(existing["article_idx"].dropna().astype(int))
        write_header = False
        remaining = len(articles_df) - len(completed_idxs)
        print(f"  Resuming: {len(completed_idxs)} done, {remaining} remaining")

    # Per-pass output means for cost tracking
    OUT_MEANS = {
        "A": {"pass1": 569, "pass2": 350, "pass3": 582, "pass4": 412},
        "B": {"pass1": 1436, "pass2": 982, "pass3": 1480, "pass4": 1126},
    }
    IN_MEANS = {
        "A": {"pass1": 2515, "pass2": 1930, "pass3": 2634, "pass4": 2218},
        "B": {"pass1": 2646, "pass2": 2061, "pass3": 2765, "pass4": 2349},
    }
    PRICE_IN = 2.50 / 1e6
    PRICE_OUT = 10.00 / 1e6

    api_counter = [0]
    n_done = len(completed_idxs)
    n_failed = 0
    failed_ids: list[int] = []
    total = len(articles_df)
    approx_cost = 0.0

    for _, row in articles_df.iterrows():
        idx = int(row["article_idx"])
        if idx in completed_idxs:
            continue

        passes = applicable_passes(row)
        for p in passes:
            approx_cost += (
                IN_MEANS[prompt_variant][p] * PRICE_IN
                + OUT_MEANS[prompt_variant][p] * PRICE_OUT
            )

        print(
            f"\n  [{prompt_variant}/run{run_number}] {n_done+1}/{total}  "
            f"idx={idx}  {row.get('outlet_clean','?')} {row.get('year','?')}"
        )
        try:
            annotation = annotate_one(idx, row, client, prompt_variant, api_counter)
            out_row: dict[str, Any] = {"article_idx": idx}
            for col in CARRY_COLS:
                out_row[col] = row.get(col)
            out_row.update(annotation)
            out_row["skipped_reason"] = None
            out_row["prompt_variant"] = prompt_variant
            out_row["run_number"] = run_number
            pd.DataFrame([out_row]).to_csv(
                output_path,
                mode="a",
                index=False,
                header=(write_header and n_done == 0),
            )
            write_header = False
            n_done += 1
            print(f"    done [{n_done}/{total}]  api_calls={api_counter[0]}")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed_ids.append(idx)
            n_failed += 1

    sep(f"PROMPT {prompt_variant} RUN {run_number} COMPLETE")
    print(f"  Annotated : {n_done}")
    print(f"  Failed    : {n_failed}")
    print(f"  API calls : {api_counter[0]}")
    print(f"  Approx cost: ${approx_cost:.2f}")
    print(f"  Output    : {output_path}")

    if failed_ids:
        failed_path.write_text("\n".join(str(i) for i in failed_ids) + "\n")
        print(f"  Failures  : {failed_path}")

    return approx_cost


def cost_estimate(articles_df: pd.DataFrame) -> None:
    OUT_MEANS = {
        "A": {"pass1": 569, "pass2": 350, "pass3": 582, "pass4": 412},
        "B": {"pass1": 1436, "pass2": 982, "pass3": 1480, "pass4": 1126},
    }
    IN_MEANS = {
        "A": {"pass1": 2515, "pass2": 1930, "pass3": 2634, "pass4": 2218},
        "B": {"pass1": 2646, "pass2": 2061, "pass3": 2765, "pass4": 2349},
    }
    PRICE_IN = 2.50 / 1e6
    PRICE_OUT = 10.00 / 1e6

    for pv in ["A", "B"]:
        in_tok = out_tok = calls = 0
        for _, row in articles_df.iterrows():
            for p in applicable_passes(row):
                in_tok += IN_MEANS[pv][p]
                out_tok += OUT_MEANS[pv][p]
                calls += 1
        cost_1run = in_tok * PRICE_IN + out_tok * PRICE_OUT
        print(
            f"  Prompt {pv}: {calls} calls/run  "
            f"→ ${cost_1run:.2f}/run  → ${cost_1run*3:.2f} for 3 runs"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="X3 benchmark stability sub-study.")
    parser.add_argument(
        "--prompt",
        choices=["A", "B", "both"],
        default="both",
        help="Which prompt variant(s) to run (default: both)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        metavar="N",
        help="Which run numbers to execute (default: 1 2 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show cost estimate only, no API calls"
    )
    args = parser.parse_args()

    prompts = ["A", "B"] if args.prompt == "both" else [args.prompt]

    sep("X3 — A vs. B stability sub-study on 100-article benchmark (X0-filtered)")
    print(f"  Prompts : {prompts}")
    print(f"  Runs    : {args.runs}")
    print(f"  Output  : {OUTPUT_DIR}")

    articles_df = load_benchmark_articles()
    print(f"  Articles loaded: {len(articles_df)}")
    print(f"  Corpus: {articles_df['corpus'].value_counts().to_dict()}")

    print("\n  COST ESTIMATE:")
    cost_estimate(articles_df)

    if args.dry_run:
        sep("DRY RUN — no API calls made")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    overall_start = time.time()

    for prompt_variant in prompts:
        for run_number in args.runs:
            sep(f"PROMPT {prompt_variant}  RUN {run_number}")
            total_cost += run_one(articles_df, client, prompt_variant, run_number)

    elapsed = time.time() - overall_start
    sep("ALL RUNS COMPLETE")
    print(f"  Wall time    : {elapsed/60:.0f}m")
    print(f"  Total cost   : ~${total_cost:.2f}")
    print()
    print("  Files written:")
    for pv in prompts:
        for rn in args.runs:
            p = OUTPUT_DIR / f"prompt{pv}_run{rn}.csv"
            n = len(pd.read_csv(p)) if p.exists() else 0
            print(f"    {p.name}  ({n} rows)")
    print()
    print("  Feed into Prompt 5 for stability metrics (Krippendorff's Alpha,")
    print("  exact-match rate, Hamming distance) and A-vs-B comparison.")


if __name__ == "__main__":
    main()
