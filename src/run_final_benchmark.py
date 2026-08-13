#!/usr/bin/env python3
"""Prompt B, run 1, over all 114 benchmark articles under the FINAL codebook.

Single run — X3 stability was established separately and is not being re-tested.

Differences from benchmark_stability_x3.py:
  * covers bi=0-113 (original 100 + climate expansion 14)
  * records per-call token usage, so the X1/X1b calibration is empirical rather
    than estimated (folded into this batch rather than run separately)
  * persists the RAW model response alongside the parsed JSON. The earlier runs
    discarded it, which is why Prompt B's chain-of-thought was unavailable for the
    three-way review protocol. Fixed here.

Outputs:
  data/annotated/final_codebook/promptB_run1_final.csv    parsed annotations
  data/annotated/final_codebook/promptB_run1_raw.jsonl    raw responses + usage
  data/annotated/final_codebook/token_usage.csv           per-call usage
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BM = ROOT / "data" / "benchmark"
OUT_DIR = ROOT / "data" / "annotated" / "final_codebook"

MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4000
MAX_RETRIES = 5
PRICE_IN, PRICE_OUT = 2.50 / 1e6, 10.00 / 1e6

_sandbox: dict = {"__file__": str(ROOT / "src" / "annotate.py")}
exec(
    (ROOT / "src" / "annotate.py").read_text().split("# ── annotation pipeline")[0],
    _sandbox,
)

SYSTEM = _sandbox["SYSTEM_PROMPT_B"]
PASSES = [
    ("pass1", _sandbox["PASS1_DEFS"], _sandbox["PASS1_FRAMES"]),
    ("pass2", _sandbox["PASS2_DEFS"], _sandbox["PASS2_FRAMES"]),
    ("pass3", _sandbox["PASS3_DEFS"], _sandbox["PASS3_FRAMES"]),
    ("pass4", _sandbox["PASS4_DEFS"], _sandbox["PASS4_FRAMES"]),
]
build_user_message = _sandbox["build_user_message"]
extract_json = _sandbox["extract_json"]
flatten_frame_result = _sandbox["flatten_frame_result"]
null_frame_row = _sandbox["null_frame_row"]
ALL_FRAMES = _sandbox["ALL_FRAMES"]
FRAME_EXTRA_FIELDS = _sandbox["FRAME_EXTRA_FIELDS"]


def load_articles() -> pd.DataFrame:
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
    cols = ["benchmark_idx", "corpus", "outlet_clean", "year", "date", "title", "body"]
    df = pd.concat([main[cols], exp[cols]], ignore_index=True)

    ids = pd.read_csv(BM / "benchmark_ids.csv")
    climate = set(ids[ids.corpus == "climate"].benchmark_idx.astype(int)) | set(
        range(100, 114)
    )
    overlap = set(pd.read_csv(BM / "benchmark_ids.csv").benchmark_idx.astype(int))
    df["corpus_type"] = df.benchmark_idx.map(
        lambda b: (
            "intersection" if b == 13 else ("climate" if b in climate else "migration")
        )
    )
    return df.sort_values("benchmark_idx").reset_index(drop=True)


def applicable(corpus_type: str) -> list[str]:
    p = ["pass1"]
    if corpus_type in ("migration", "intersection"):
        p.append("pass2")
    if corpus_type in ("climate", "intersection"):
        p.append("pass3")
    p.append("pass4")
    return p


def call(client, user_msg: str):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            return (
                r.choices[0].message.content or "",
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            )
        except (openai.RateLimitError, openai.APIError) as exc:
            wait = 2 ** (attempt + 1)
            print(
                f"      [retry {attempt+1}/{MAX_RETRIES}] {type(exc).__name__} "
                f"— waiting {wait}s"
            )
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                raise
    raise RuntimeError("exceeded retries")


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found in .env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI()

    arts = load_articles()
    print(f"  articles: {len(arts)}  " f"({arts.corpus_type.value_counts().to_dict()})")

    raw_path = OUT_DIR / "promptB_run1_raw.jsonl"
    rows, usage = [], []
    parse_failures = []

    with raw_path.open("w") as raw_f:
        for n, (_, a) in enumerate(arts.iterrows(), 1):
            bi = int(a.benchmark_idx)
            result = {"corpus_type": a.corpus_type}
            passes = applicable(a.corpus_type)
            for pname, defs, frames in PASSES:
                if pname not in passes:
                    result.update(null_frame_row(frames))
                    continue
                msg = build_user_message(defs, str(a.body or ""), frames)
                txt, pt, ct = call(client, msg)
                usage.append(
                    {
                        "benchmark_idx": bi,
                        "corpus_type": a.corpus_type,
                        "pass": pname,
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "cost_usd": pt * PRICE_IN + ct * PRICE_OUT,
                    }
                )
                raw_f.write(
                    json.dumps(
                        {
                            "benchmark_idx": bi,
                            "pass": pname,
                            "raw": txt,
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                        }
                    )
                    + "\n"
                )
                try:
                    result.update(flatten_frame_result(extract_json(txt), frames))
                except Exception as exc:
                    parse_failures.append((bi, pname, str(exc)[:80]))
                    result.update(null_frame_row(frames))
            rows.append(
                {
                    "benchmark_idx": bi,
                    "outlet_clean": a.outlet_clean,
                    "date": a.date,
                    "year": a.year,
                    "corpus": a.corpus,
                    **result,
                    "prompt_variant": "B",
                    "run_number": 1,
                }
            )
            spent = sum(u["cost_usd"] for u in usage)
            print(
                f"  [{n:3d}/{len(arts)}] bi={bi:3d} {a.corpus_type:12s} "
                f"{len(passes)} calls   running cost ${spent:.2f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "promptB_run1_final.csv", index=False)
    u = pd.DataFrame(usage)
    u.to_csv(OUT_DIR / "token_usage.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  articles annotated : {len(df)}")
    print(f"  API calls          : {len(u)}")
    print(f"  input tokens       : {u.prompt_tokens.sum():,}")
    print(f"  output tokens      : {u.completion_tokens.sum():,}")
    print(f"  ACTUAL COST        : ${u.cost_usd.sum():,.2f}")
    print(f"  parse failures     : {len(parse_failures)}")
    for bi, p, e in parse_failures:
        print(f"      bi={bi} {p}: {e}")
    print(f"  raw responses      : {raw_path}")


if __name__ == "__main__":
    main()
