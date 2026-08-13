"""
Prompt 3 — orchestration script for the full 40-run annotation pipeline.

Design:
  5 runs × 4 passes × 2 corpora = 40 "pass-level" runs.
  Since annotate.py handles all 4 passes in a single invocation, the actual
  CLI invocations are: N_RUNS × N_CORPORA × N_PROMPTS.

  Default (--prompt A, 5 runs, 2 corpora): 10 invocations = 40 pass-level runs.
  Pass --prompt B or --prompt both for additional variants.

Resume:
  annotate.py already handles per-article resume via completed_indices.
  This script checks whether each invocation is already fully complete
  (output row count == expected total rows) and skips it entirely if so.

Usage:
  python3 src/run_all.py --dry-run       # show plan and cost estimate, do nothing
  python3 src/run_all.py                 # run prompt A only (default)
  python3 src/run_all.py --prompt B
  python3 src/run_all.py --prompt both
  python3 src/run_all.py --corpus climate   # single corpus
  python3 src/run_all.py --runs 1 2 3      # specific runs only
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SAMPLED_DIR = ROOT / "data" / "sampled"
ANNOTATED_DIR = ROOT / "data" / "annotated"
LOG_DIR = ROOT / "data" / "annotated"

SAMPLE_FILES = {
    "climate": "climate_sample_flagged.csv",
    "migration": "migration_sample_topped_up.csv",
}

# Expected total rows per output file = all articles (on-topic annotated + off-topic skipped)
CORPUS_TOTALS = {
    "climate": len(
        pd.read_csv(SAMPLED_DIR / SAMPLE_FILES["climate"], usecols=["outlet_clean"])
    ),
    "migration": len(
        pd.read_csv(SAMPLED_DIR / SAMPLE_FILES["migration"], usecols=["outlet_clean"])
    ),
}
ON_TOPIC_COUNTS = {
    "climate": int(
        pd.read_csv(SAMPLED_DIR / SAMPLE_FILES["climate"], usecols=["on_topic_flag"])[
            "on_topic_flag"
        ].sum()
    ),
    "migration": int(
        pd.read_csv(SAMPLED_DIR / SAMPLE_FILES["migration"], usecols=["on_topic_flag"])[
            "on_topic_flag"
        ].sum()
    ),
}


def sep(msg: str) -> None:
    width = 72
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * width}\n  [{ts}] {msg}\n{'=' * width}")


def output_path(corpus: str, prompt: str, run: int) -> Path:
    return ANNOTATED_DIR / f"{corpus}_prompt{prompt}_run{run}.csv"


def completed_count(path: Path) -> int:
    """Number of articles already written to the output file."""
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path, usecols=["article_idx"]))
    except Exception:
        return 0


def is_fully_complete(corpus: str, prompt: str, run: int) -> bool:
    path = output_path(corpus, prompt, run)
    expected = CORPUS_TOTALS[corpus]
    actual = completed_count(path)
    return actual >= expected


def run_invocation(corpus: str, prompt: str, run: int, log_file: Path) -> int:
    """
    Launch annotate.py for one corpus/prompt/run combination.
    Streams output to stdout AND to log_file.
    Returns the subprocess return code.
    """
    cmd = [
        sys.executable,
        str(ROOT / "src" / "annotate.py"),
        "--corpus",
        corpus,
        "--prompt",
        prompt,
        "--run",
        str(run),
    ]
    sep(f"Launching: corpus={corpus}  prompt={prompt}  run={run}")
    print(f"  Command  : {' '.join(cmd)}")
    print(f"  Log file : {log_file}")
    print(
        f"  Expected : {CORPUS_TOTALS[corpus]:,} rows total "
        f"({ON_TOPIC_COUNTS[corpus]:,} on-topic, "
        f"{CORPUS_TOTALS[corpus] - ON_TOPIC_COUNTS[corpus]:,} skipped)"
    )
    print()

    with open(log_file, "w") as lf:
        lf.write(f"# annotate.py log: corpus={corpus} prompt={prompt} run={run}\n")
        lf.write(f"# started: {datetime.now().isoformat()}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

        lf.write(
            f"\n# finished: {datetime.now().isoformat()}  returncode={proc.returncode}\n"
        )

    return proc.returncode


def cost_estimate(corpora: list[str], prompts: list[str], runs: list[int]) -> None:
    """Print cost/time estimate before launching anything."""
    sep("COST / TIME ESTIMATE")

    # API calls per run (corpus depends on how many passes each article triggers)
    PASSES = {
        "climate": {"intersection": 4, "default": 3},
        "migration": {"intersection": 4, "default": 3},
    }
    total_calls_per_run: dict[str, int] = {}
    for corpus in corpora:
        df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
        df_on = df[df["on_topic_flag"] == True]
        calls = 0
        for _, row in df_on.iterrows():
            overlap = bool(row.get("corpus_overlap", False))
            calls += 4 if overlap else 3
        total_calls_per_run[corpus] = calls

    total_calls_all = (
        sum(total_calls_per_run[c] for c in corpora) * len(runs) * len(prompts)
    )

    print(f"  Corpora          : {', '.join(corpora)}")
    print(f"  Prompt variants  : {', '.join(prompts)}")
    print(f"  Runs             : {runs}")
    print(f"  CLI invocations  : {len(corpora) * len(runs) * len(prompts)}")
    print()
    for corpus in corpora:
        print(
            f"  {corpus:10s}  on-topic={ON_TOPIC_COUNTS[corpus]:,}  "
            f"API calls/run={total_calls_per_run[corpus]:,}  "
            f"total across runs+prompts={total_calls_per_run[corpus]*len(runs)*len(prompts):,}"
        )
    print(f"\n  Total API calls  : {total_calls_all:,}")

    # Token and cost estimates
    INPUT_TOKENS = 2_000  # conservative per-call average (system+defs+article+format)
    OUT_TOKENS_A = 500  # Prompt A: JSON only
    OUT_TOKENS_B = 2_000  # Prompt B: CoT reasoning + JSON

    calls_A = (
        sum(total_calls_per_run[c] for c in corpora) * len(runs) * prompts.count("A")
    )
    calls_B = (
        sum(total_calls_per_run[c] for c in corpora) * len(runs) * prompts.count("B")
    )

    in_tok = (calls_A + calls_B) * INPUT_TOKENS
    out_tok = calls_A * OUT_TOKENS_A + calls_B * OUT_TOKENS_B

    # GPT-4o pricing (as of mid-2025; verify at platform.openai.com/docs/pricing)
    PRICE_IN = 2.50 / 1_000_000
    PRICE_OUT = 10.00 / 1_000_000

    cost_in = in_tok * PRICE_IN
    cost_out = out_tok * PRICE_OUT
    cost_tot = cost_in + cost_out

    print(f"\n  Input tokens est  : {in_tok/1e6:.1f}M  → ${cost_in:.2f}")
    print(f"  Output tokens est : {out_tok/1e6:.1f}M  → ${cost_out:.2f}")
    print(f"  TOTAL COST EST    : ~${cost_tot:.0f}")
    print()
    print("  NOTE: GPT-4o pricing used here: $2.50/M input, $10.00/M output.")
    print("        Verify at platform.openai.com/docs/pricing before launching.")
    print("        Output token estimates are approximate (Prompt B is much higher).")
    print("        Actual cost depends on real article lengths and response verbosity.")
    print()

    # Time estimate: ~5s per API call with retries / rate limits
    AVG_SECONDS_PER_CALL = 4
    total_seconds = total_calls_all * AVG_SECONDS_PER_CALL
    hours = total_seconds / 3600
    print(
        f"  Time estimate     : ~{hours:.0f} hours  ({total_seconds:,}s at ~{AVG_SECONDS_PER_CALL}s/call)"
    )
    print(f"  (Actual depends on OpenAI rate limits and tier.)")


def summarise_progress(corpora: list[str], prompts: list[str], runs: list[int]) -> None:
    """Print a progress table for all combinations."""
    sep("PROGRESS SUMMARY")
    total_complete = 0
    total_invocations = len(corpora) * len(prompts) * len(runs)
    print(
        f"  {'Corpus':<12} {'Prompt':<8} {'Run':<6} {'Written':>8} {'Expected':>9} {'Status'}"
    )
    print(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*8} {'-'*9} {'-'*10}")
    for corpus in corpora:
        for prompt in prompts:
            for run in runs:
                path = output_path(corpus, prompt, run)
                written = completed_count(path)
                expected = CORPUS_TOTALS[corpus]
                pct = 100 * written / expected if expected else 0
                status = (
                    "DONE"
                    if written >= expected
                    else (f"{pct:.0f}%" if written > 0 else "not started")
                )
                if written >= expected:
                    total_complete += 1
                print(
                    f"  {corpus:<12} {prompt:<8} {run:<6} {written:>8,} {expected:>9,} {status}"
                )
    print(f"\n  Complete: {total_complete}/{total_invocations} invocations")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full annotation pipeline for all combinations."
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        choices=["climate", "migration"],
        default=["climate", "migration"],
    )
    parser.add_argument(
        "--prompt",
        choices=["A", "B", "both"],
        default="A",
        help="Prompt variant(s) to run. 'both' runs A then B. Default: A",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=list(range(1, 6)),
        metavar="N",
        help="Run numbers to execute (1-5). Default: all 5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost estimate and plan, then exit without running anything.",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Show progress summary and exit."
    )
    args = parser.parse_args()

    corpora = args.corpus
    prompts = ["A", "B"] if args.prompt == "both" else [args.prompt]
    runs = args.runs

    if args.progress:
        summarise_progress(corpora, prompts, runs)
        return

    sep("ANNOTATE — full pipeline orchestration")
    print(f"  Corpora         : {', '.join(corpora)}")
    print(f"  Prompt variants : {', '.join(prompts)}")
    print(f"  Runs            : {runs}")
    print(f"  Dry run         : {args.dry_run}")

    cost_estimate(corpora, prompts, runs)

    if args.dry_run:
        sep("DRY RUN — plan shown above, nothing executed")
        return

    # Check already-complete invocations
    combinations = [
        (corpus, prompt, run)
        for corpus in corpora
        for prompt in prompts
        for run in runs
    ]
    already_done = [(c, p, r) for c, p, r in combinations if is_fully_complete(c, p, r)]
    to_run = [(c, p, r) for c, p, r in combinations if not is_fully_complete(c, p, r)]

    print(f"  Invocations total     : {len(combinations)}")
    print(f"  Already complete      : {len(already_done)}")
    print(f"  To run (new/resume)   : {len(to_run)}")
    if already_done:
        print("  Skipping:")
        for c, p, r in already_done:
            print(f"    {c} prompt{p} run{r} — already complete")

    if not to_run:
        sep("ALL INVOCATIONS ALREADY COMPLETE")
        summarise_progress(corpora, prompts, runs)
        return

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # ── main loop ──────────────────────────────────────────────────────────────
    overall_start = time.time()
    failed_runs: list[tuple] = []
    completed_runs: list[tuple] = []

    for i, (corpus, prompt, run) in enumerate(to_run, 1):
        run_start = time.time()
        log_file = ANNOTATED_DIR / f"log_{corpus}_prompt{prompt}_run{run}.txt"

        sep(f"Invocation {i}/{len(to_run)}: {corpus} / prompt {prompt} / run {run}")
        already = completed_count(output_path(corpus, prompt, run))
        if already > 0:
            print(f"  Resuming from {already:,} already-written rows")

        rc = run_invocation(corpus, prompt, run, log_file)
        elapsed = time.time() - run_start

        if rc == 0:
            written = completed_count(output_path(corpus, prompt, run))
            print(
                f"\n  Invocation finished in {elapsed:.0f}s  |  rows written: {written:,}"
            )
            completed_runs.append((corpus, prompt, run))
        else:
            print(f"\n  Invocation FAILED (returncode={rc})  —  check log: {log_file}")
            failed_runs.append((corpus, prompt, run))

    # ── final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    sep("ALL RUNS COMPLETE")
    print(f"  Total wall time     : {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)")
    print(f"  Invocations run     : {len(to_run)}")
    print(f"  Successful          : {len(completed_runs)}")
    print(f"  Failed              : {len(failed_runs)}")
    if failed_runs:
        print("  Failed invocations (re-run individually):")
        for c, p, r in failed_runs:
            print(f"    python3 src/annotate.py --corpus {c} --prompt {p} --run {r}")
    summarise_progress(corpora, prompts, runs)


if __name__ == "__main__":
    main()
