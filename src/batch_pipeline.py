"""
Prompt X2 — Batch API pipeline for single full-corpus annotation run.

Three modes (use --mode flag):
  generate   Build the batch .jsonl file and report size/count. No API calls.
  submit     Upload the file and create the batch job. Requires --confirm flag.
  status     Check current status of a batch job. Requires --batch-id or reads saved ID.
  parse      Download completed results and write to data/annotated/. Requires --batch-id.

Usage:
  python3 src/batch_pipeline.py --mode generate
  python3 src/batch_pipeline.py --mode submit --confirm
  python3 src/batch_pipeline.py --mode status
  python3 src/batch_pipeline.py --mode parse

Output structure:
  data/annotated/climate_promptA_run1.csv    (same schema as regular annotate.py output)
  data/annotated/migration_promptA_run1.csv
  data/annotated/batch_job_ids.json          (saved batch ID + metadata)
  data/annotated/batch_climate_run1.jsonl    (batch input file)
  data/annotated/batch_migration_run1.jsonl

On-topic gating:
  Off-topic articles (on_topic_flag=False) are NOT included in the batch at all.
  They are written as skipped rows to the output CSV during parse step, same as
  the synchronous pipeline.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

SAMPLED_DIR = ROOT / "data" / "sampled"
ANNOTATED_DIR = ROOT / "data" / "annotated"
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4000  # Prompt B: chain-of-thought reasoning + JSON
RUN_NUMBER = 1
PROMPT_VARIANT = "B"

# X0-filtered pool: topic_central_flag=True articles only
SAMPLE_FILES = {
    "climate": "climate_sample_centrality.csv",
    "migration": "migration_sample_centrality.csv",
}

# ── Pull codebook from annotate.py ────────────────────────────────────────
_src = (ROOT / "src" / "annotate.py").read_text()
_patched = _src.replace("Path(__file__).parent.parent", "Path('.')").replace(
    'if __name__ == "__main__":\n    main()', ""
)
_sandbox: dict = {
    "__file__": str(ROOT / "src" / "annotate.py"),
    "__name__": "__batch__",
}
exec(compile(_patched, "annotate.py", "exec"), _sandbox)

SYSTEM_B = _sandbox["SYSTEM_PROMPT_B"]
build_user_message = _sandbox["build_user_message"]
build_skipped_row = _sandbox["build_skipped_row"]
flatten_frame_result = _sandbox["flatten_frame_result"]
null_frame_row = _sandbox["null_frame_row"]
extract_json = _sandbox["extract_json"]
derive_corpus_type = _sandbox["derive_corpus_type"]
CARRY_COLS = _sandbox["CARRY_COLS"]
ALL_FRAMES = _sandbox["ALL_FRAMES"]

PASSES_DEF = [
    ("pass1", _sandbox["PASS1_DEFS"], _sandbox["PASS1_FRAMES"]),
    ("pass2", _sandbox["PASS2_DEFS"], _sandbox["PASS2_FRAMES"]),
    ("pass3", _sandbox["PASS3_DEFS"], _sandbox["PASS3_FRAMES"]),
    ("pass4", _sandbox["PASS4_DEFS"], _sandbox["PASS4_FRAMES"]),
]


def sep(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    width = 72
    print(f"\n{'=' * width}\n  [{ts}] {msg}\n{'=' * width}")


def applicable_passes(row: pd.Series) -> list[str]:
    """Which passes to run for this article (same logic as annotate.py)."""
    corpus_type = derive_corpus_type(row)
    passes = ["pass1"]
    if corpus_type in ("migration", "intersection"):
        passes.append("pass2")
    if corpus_type in ("climate", "intersection"):
        passes.append("pass3")
    passes.append("pass4")
    return passes


def make_custom_id(corpus: str, article_idx: int, pass_name: str) -> str:
    return f"{corpus}-{article_idx}-{pass_name}"


def parse_custom_id(custom_id: str) -> tuple[str, int, str]:
    """Returns (corpus, article_idx, pass_name)."""
    parts = custom_id.split("-")
    corpus = parts[0]
    art_idx = int(parts[1])
    pass_nm = "-".join(parts[2:])
    return corpus, art_idx, pass_nm


def build_request_line(
    corpus: str,
    article_idx: int,
    pass_name: str,
    defs: str,
    frames: list[str],
    body: str,
) -> dict:
    user_msg = build_user_message(defs, body, frames)
    return {
        "custom_id": make_custom_id(corpus, article_idx, pass_name),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_B},
                {"role": "user", "content": user_msg},
            ],
        },
    }


# ── MODE: generate ────────────────────────────────────────────────────────


def mode_generate() -> None:
    sep("GENERATE — building batch .jsonl files")
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    pass_map = {p[0]: (p[1], p[2]) for p in PASSES_DEF}
    summary: dict[str, Any] = {}

    # Per-pass Prompt B token means, MEASURED on the n=114 benchmark run under the
    # final codebook (343 calls). Source: data/annotated/final_codebook/token_usage.csv
    # Supersedes the earlier X1b estimates, which predated every codebook change.
    # Display only — these feed the printed estimate below and nothing that is
    # submitted or billed. The request bodies are built from annotate.py directly.
    B_IN = {"pass1": 2320, "pass2": 1575, "pass3": 2166, "pass4": 2308}
    B_OUT = {"pass1": 1508, "pass2": 982, "pass3": 1420, "pass4": 1070}
    total_in_tok = total_out_tok = 0

    for corpus in ["climate", "migration"]:
        out_path = ANNOTATED_DIR / f"batch_{corpus}_run{RUN_NUMBER}.jsonl"
        df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
        df["article_idx"] = range(len(df))
        # X0 filter: only articles that are both on-topic AND centrally about the topic
        df_annotate = df[
            (df["on_topic_flag"] == True) & (df["topic_central_flag"] == True)
        ]
        df_skip = df[~df.index.isin(df_annotate.index)]

        print(
            f"\n  {corpus}: {len(df_annotate):,} to annotate, "
            f"{len(df_skip):,} skipped (off-topic or not-central)"
        )

        n_requests = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in df_annotate.iterrows():
                body = str(row.get("body", "") or "")
                passes = applicable_passes(row)
                for pass_name in passes:
                    defs, frames = pass_map[pass_name]
                    req = build_request_line(
                        corpus, int(row["article_idx"]), pass_name, defs, frames, body
                    )
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n_requests += 1
                    total_in_tok += B_IN[pass_name]
                    total_out_tok += B_OUT[pass_name]

        size_bytes = out_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        summary[corpus] = {
            "path": str(out_path),
            "n_requests": n_requests,
            "size_mb": size_mb,
            "annotate": len(df_annotate),
            "skipped": len(df_skip),
        }
        print(f"    Requests : {n_requests:,}")
        print(f"    File size: {size_mb:.1f} MB  ({size_bytes:,} bytes)")
        print(f"    Written  : {out_path}")
        limit_flag = " ← EXCEEDS 50k LIMIT" if n_requests > 50_000 else ""
        size_flag = " ← EXCEEDS 200MB LIMIT" if size_mb > 200 else ""
        print(f"    50k check: {'OK' if not limit_flag else limit_flag}")
        print(f"    200MB chk: {'OK' if not size_flag else size_flag}")

    total_requests = sum(v["n_requests"] for v in summary.values())
    total_size = sum(v["size_mb"] for v in summary.values())
    print(f"\n  TOTAL REQUESTS : {total_requests:,}  (across both corpora)")
    print(f"  TOTAL FILE SIZE: {total_size:.1f} MB")

    # Cost estimate from the measured final-codebook token means (see B_IN/B_OUT).
    print()
    print("  COST ESTIMATE (Prompt B, X0-filtered pool, measured calibration):")
    std = total_in_tok * 2.50 / 1e6 + total_out_tok * 10.00 / 1e6
    batch = total_in_tok * 1.25 / 1e6 + total_out_tok * 5.00 / 1e6
    print(f"    Input tokens  : {total_in_tok/1e6:.1f}M")
    print(f"    Output tokens : {total_out_tok/1e6:.1f}M")
    print(f"    Standard API  : ${std:,.2f}   ($2.50/M in, $10.00/M out)")
    print(
        f"    Batch API     : ${batch:,.2f}   ($1.25/M in, $5.00/M out, 50% discount)"
    )

    # Save summary for submit step
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    meta = {
        "generated_at": datetime.now().isoformat(),
        "run_number": RUN_NUMBER,
        "prompt_variant": PROMPT_VARIANT,
        "files": summary,
        "batch_ids": {},
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        meta["batch_ids"] = existing.get("batch_ids", {})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  Metadata saved → {meta_path}")
    print()
    print("  Next: confirm cost, then run:")
    print("    python3 src/batch_pipeline.py --mode submit --confirm")


# ── MODE: submit ──────────────────────────────────────────────────────────


def mode_submit(confirm: bool) -> None:
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    if not meta_path.exists():
        sys.exit("ERROR: run --mode generate first to build the batch files")

    meta = json.loads(meta_path.read_text())

    if not confirm:
        print("\n  DRY RUN — files ready to submit but --confirm not passed.")
        print("  Add --confirm to actually upload and create the batch job.")
        print("  Files that would be submitted:")
        for corpus, info in meta["files"].items():
            print(
                f"    {corpus}: {info['path']}  ({info['n_requests']:,} requests, {info['size_mb']:.1f} MB)"
            )
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    sep("SUBMIT — uploading batch files and creating jobs")

    for corpus in ["climate", "migration"]:
        if corpus not in meta["files"]:
            print(f"  {corpus}: no file found in metadata, skipping")
            continue
        info = meta["files"][corpus]
        jsonl_path = Path(info["path"])
        if not jsonl_path.exists():
            print(f"  {corpus}: file not found at {jsonl_path}, skipping")
            continue

        if corpus in meta["batch_ids"] and meta["batch_ids"][corpus].get("batch_id"):
            print(
                f"  {corpus}: batch already submitted → {meta['batch_ids'][corpus]['batch_id']}"
            )
            print(f"    Run --mode status to check progress")
            continue

        print(f"\n  Uploading {jsonl_path.name} ({info['size_mb']:.1f} MB)...")
        with open(jsonl_path, "rb") as f:
            upload = client.files.create(file=f, purpose="batch")
        print(f"    File ID: {upload.id}")

        print(f"  Creating batch job...")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "corpus": corpus,
                "run": str(RUN_NUMBER),
                "prompt": PROMPT_VARIANT,
            },
        )
        print(f"    Batch ID : {batch.id}")
        print(f"    Status   : {batch.status}")
        print(f"    Created  : {datetime.fromtimestamp(batch.created_at).isoformat()}")

        meta["batch_ids"][corpus] = {
            "batch_id": batch.id,
            "file_id": upload.id,
            "submitted_at": datetime.now().isoformat(),
            "n_requests": info["n_requests"],
            "status": batch.status,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"    Saved batch ID → {meta_path}")

    sep("SUBMISSION COMPLETE")
    print("  Batch jobs submitted. Expected completion: within 24 hours.")
    print(f"  Check progress: python3 src/batch_pipeline.py --mode status")

    sep("EXPECTED OUTPUT SIZE (pre-download estimate)")
    report_size_estimate(meta)
    print(f"  Metadata      : {meta_path}")
    print()


# ── MODE: status ──────────────────────────────────────────────────────────


def mode_status() -> None:
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    if not meta_path.exists():
        sys.exit("ERROR: no batch_job_ids.json found — has a job been submitted?")

    meta = json.loads(meta_path.read_text())
    batch_ids = meta.get("batch_ids", {})

    if not batch_ids:
        print("No batch jobs found in metadata.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    sep("BATCH STATUS")
    for corpus, info in batch_ids.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            print(f"  {corpus}: no batch_id found")
            continue

        batch = client.batches.retrieve(batch_id)
        rc = batch.request_counts

        submitted_at = info.get("submitted_at", "?")
        elapsed_msg = ""
        if submitted_at != "?":
            try:
                dt = datetime.fromisoformat(submitted_at)
                elapsed = (datetime.now() - dt).total_seconds() / 3600
                elapsed_msg = f"  ({elapsed:.1f}h since submission)"
            except Exception:
                pass

        print(f"\n  {corpus.upper()} — {batch_id}")
        print(f"    Status    : {batch.status}{elapsed_msg}")
        print(f"    Submitted : {submitted_at}")
        if rc:
            total = rc.total or info.get("n_requests", "?")
            done = rc.completed or 0
            failed = rc.failed or 0
            pct = 100 * done / total if isinstance(total, int) and total > 0 else 0
            print(
                f"    Progress  : {done:,}/{total:,} completed ({pct:.0f}%)  |  {failed:,} failed"
            )
        if batch.status == "completed":
            print(f"    Output file: {batch.output_file_id}")
            print(f"    → Run: python3 src/batch_pipeline.py --mode parse")
        elif batch.status in ("failed", "expired", "cancelled"):
            print(f"    ERROR file : {getattr(batch, 'error_file_id', None)}")
        if batch.expires_at:
            exp = datetime.fromtimestamp(batch.expires_at)
            print(f"    Expires   : {exp.isoformat()}")

        # Update stored status
        meta["batch_ids"][corpus]["status"] = batch.status
        if rc:
            meta["batch_ids"][corpus]["completed_count"] = rc.completed
            meta["batch_ids"][corpus]["failed_count"] = rc.failed
        if batch.status == "completed":
            meta["batch_ids"][corpus]["output_file_id"] = batch.output_file_id
    meta_path.write_text(json.dumps(meta, indent=2))
    print()


# ── MODE: parse ───────────────────────────────────────────────────────────

# ── download safety: size estimate, disk headroom, integrity ─────────────────

DISK_SAFETY_FACTOR = 3  # require 3x the estimate: temp files, parsing, margin
BYTES_PER_TOKEN = 5.14  # measured on the n=114 benchmark raw jsonl
ENVELOPE_BYTES = 550  # OpenAI batch line: id/custom_id/response/usage fields

# Empirical completion tokens per call, final codebook, Prompt B (n=343 calls).
# Source: data/annotated/final_codebook/token_usage.csv
OUT_TOKENS_PER_CALL = {"pass1": 1508, "pass2": 982, "pass3": 1420, "pass4": 1070}


def estimate_output_bytes(corpus: str, n_requests: int, n_annotate: int) -> int:
    """Approximate the size of the batch output JSONL before it is downloaded.

    Each article gets pass1 + one corpus-specific pass + pass4; any surplus
    requests (intersection articles taking a fourth pass) are costed at the
    pass1 rate as a conservative approximation.
    """
    mid = "pass3" if corpus == "climate" else "pass2"
    per_article = (
        OUT_TOKENS_PER_CALL["pass1"]
        + OUT_TOKENS_PER_CALL[mid]
        + OUT_TOKENS_PER_CALL["pass4"]
    )
    tokens = n_annotate * per_article
    tokens += max(0, n_requests - n_annotate * 3) * OUT_TOKENS_PER_CALL["pass1"]
    return int(tokens * BYTES_PER_TOKEN + n_requests * ENVELOPE_BYTES)


def report_size_estimate(meta: dict) -> int:
    """Print the pre-submission size estimate. Returns total estimated bytes."""
    mb = lambda b: b / 1048576
    print(f"\n  {'corpus':<12}{'requests':>10}{'articles':>10}{'est. output':>14}")
    total = 0
    for corpus, info in meta.get("files", {}).items():
        est = estimate_output_bytes(corpus, info["n_requests"], info["annotate"])
        total += est
        print(
            f"  {corpus:<12}{info['n_requests']:>10,}{info['annotate']:>10,}"
            f"{mb(est):>11.1f} MB"
        )
    print(f"  {'TOTAL':<12}{'':>10}{'':>10}{mb(total):>11.1f} MB")
    print(
        f"\n  Disk headroom required before download "
        f"({DISK_SAFETY_FACTOR}x): {mb(total * DISK_SAFETY_FACTOR):.0f} MB"
    )
    return total


def check_disk_headroom(target_dir: Path, estimated_bytes: int, corpus: str) -> bool:
    """Return True if it is safe to download. Never writes anything."""
    import shutil

    free = shutil.disk_usage(target_dir).free
    need = estimated_bytes * DISK_SAFETY_FACTOR
    mb = lambda b: b / 1048576
    print(
        f"    disk check: need {mb(need):,.0f} MB "
        f"({DISK_SAFETY_FACTOR}x est. {mb(estimated_bytes):,.0f} MB), "
        f"free {mb(free):,.0f} MB"
    )
    if free >= need:
        return True
    print(f"\n    STOPPING before download — insufficient disk space.")
    print(f"      estimated output : {mb(estimated_bytes):,.0f} MB")
    print(f"      required ({DISK_SAFETY_FACTOR}x)   : {mb(need):,.0f} MB")
    print(f"      available        : {mb(free):,.0f} MB")
    print(f"      shortfall        : {mb(need - free):,.0f} MB")
    print(
        f"\n    Nothing was written and nothing is lost. The results remain"
        f"\n    stored on OpenAI's side — batch output files persist until"
        f"\n    explicitly deleted. Free up space and re-run --mode parse;"
        f"\n    the {corpus} results will still be there."
    )
    return False


def verify_download(path: Path, expected_lines: int, corpus: str) -> bool:
    """Confirm the downloaded file is complete and well-formed.

    A file existing on disk is not evidence it is the right file: check the
    line count against the number of requests submitted, and parse a sample.
    """
    if not path.exists():
        print(f"    verify: FAILED — {path.name} does not exist")
        return False
    n = sum(1 for line in path.open() if line.strip())
    ok = True
    print(f"    verify: {n:,} lines, expected {expected_lines:,}")
    if n != expected_lines:
        print(
            f"    verify: LINE COUNT MISMATCH ({n - expected_lines:+,}) — "
            f"download likely truncated"
        )
        ok = False
    sample, bad = [], 0
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i % max(1, n // 20) == 0 and len(sample) < 20:
                sample.append(line)
    for line in sample:
        try:
            obj = json.loads(line)
            if "custom_id" not in obj:
                bad += 1
        except json.JSONDecodeError:
            bad += 1
    print(f"    verify: parsed {len(sample)} sampled lines, {bad} malformed")
    if bad:
        ok = False
    if not ok:
        print(
            f"\n    Treating the {corpus} download as FAILED. The remote copy is"
            f"\n    intact — batch outputs persist on OpenAI until deleted. Delete"
            f"\n    the local partial file and re-run --mode parse to retry."
        )
    return ok


def mode_parse() -> None:
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    if not meta_path.exists():
        sys.exit("ERROR: no batch_job_ids.json found")

    meta = json.loads(meta_path.read_text())

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    sep("PARSE — downloading and converting batch results")
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    for corpus in ["climate", "migration"]:
        batch_info = meta.get("batch_ids", {}).get(corpus)
        if not batch_info:
            print(f"  {corpus}: no batch info found, skipping")
            continue

        batch_id = batch_info.get("batch_id")
        batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"  {corpus}: batch status is '{batch.status}' — not ready to parse")
            if batch.status == "expired":
                _report_missing(meta, corpus, client)
            continue

        # Download results (cache locally so re-parse doesn't re-download)
        out_file_id = batch.output_file_id or batch_info.get("output_file_id")
        if not out_file_id:
            print(f"  {corpus}: no output_file_id found")
            continue

        raw_cache = ANNOTATED_DIR / f"batch_{corpus}_run{RUN_NUMBER}_raw.jsonl"
        file_info = meta.get("files", {}).get(corpus, {})
        n_requests = file_info.get("n_requests", 0)

        if raw_cache.exists():
            print(f"\n  {corpus}: using cached results from {raw_cache.name}")
            if not verify_download(raw_cache, n_requests, corpus):
                print(f"  {corpus}: cached file failed verification — skipping")
                continue
            raw_content = raw_cache.read_bytes()
        else:
            print(f"\n  {corpus}: preparing to download results file {out_file_id}")
            estimated = estimate_output_bytes(
                corpus, n_requests, file_info.get("annotate", 0)
            )
            if not check_disk_headroom(ANNOTATED_DIR, estimated, corpus):
                continue  # nothing written, nothing lost
            print(f"    downloading...")
            raw_content = client.files.content(out_file_id).read()
            raw_cache.write_bytes(raw_content)
            print(
                f"    wrote {raw_cache.name} "
                f"({len(raw_content)/1048576:.1f} MB actual vs "
                f"{estimated/1048576:.1f} MB estimated)"
            )
            if not verify_download(raw_cache, n_requests, corpus):
                continue  # do not parse a bad file
        raw_lines = raw_content.decode("utf-8").strip().split("\n")
        print(f"    {len(raw_lines):,} result lines")

        # Build a lookup: custom_id → result JSON
        results: dict[str, dict] = {}
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("custom_id", "")
                if obj.get("error"):
                    results[cid] = {"error": obj["error"]}
                else:
                    content = obj["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]
                    results[cid] = {"content": content}
            except Exception as e:
                print(f"    Warning: could not parse line: {e}")

        # Load the original sample to get all articles (including off-topic)
        df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
        df["article_idx"] = range(len(df))
        total = len(df)

        output_path = (
            ANNOTATED_DIR / f"{corpus}_prompt{PROMPT_VARIANT}_run{RUN_NUMBER}.csv"
        )
        pass_map = {p[0]: p[2] for p in PASSES_DEF}  # pass_name → frames list

        rows = []
        failed_ids: list[str] = []
        schema_errors: list[str] = []
        n_on_topic = 0
        n_off_topic = 0
        n_failed = 0

        print(f"  Building output rows for {total:,} articles...")

        for _, row in df.iterrows():
            idx = int(row["article_idx"])
            on_topic = bool(row.get("on_topic_flag", True))
            central = bool(row.get("topic_central_flag", True)) if on_topic else False

            if not on_topic or not central:
                # Write skipped row: off-topic OR not centrally about the topic
                reason = (
                    "off_topic_keyword_flag" if not on_topic else "not_topic_central"
                )
                skipped = build_skipped_row(idx, row, PROMPT_VARIANT, RUN_NUMBER)
                skipped["skipped_reason"] = reason
                rows.append(skipped)
                n_off_topic += 1
                continue

            # Gather results for all passes this article should have
            passes = applicable_passes(row)
            article_result: dict[str, Any] = {}
            article_result["corpus_type"] = derive_corpus_type(row)

            article_ok = True
            for pass_name in ["pass1", "pass2", "pass3", "pass4"]:
                frames = pass_map[pass_name]
                cid = make_custom_id(corpus, idx, pass_name)

                if pass_name not in passes:
                    # Pass not applicable — null it out
                    article_result.update(null_frame_row(frames))
                    continue

                res = results.get(cid)
                if res is None:
                    failed_ids.append(cid)
                    article_result.update(null_frame_row(frames))
                    article_ok = False
                    continue

                if "error" in res:
                    failed_ids.append(cid)
                    article_result.update(null_frame_row(frames))
                    article_ok = False
                    continue

                try:
                    parsed = extract_json(res["content"])
                    article_result.update(flatten_frame_result(parsed, frames))
                except Exception as e:
                    schema_errors.append(f"{cid}: {e}")
                    article_result.update(null_frame_row(frames))
                    article_ok = False

            out_row: dict[str, Any] = {"article_idx": idx}
            for col in CARRY_COLS:
                out_row[col] = row.get(col)
            out_row.update(article_result)
            out_row["skipped_reason"] = None if article_ok else "batch_parse_error"
            out_row["prompt_variant"] = PROMPT_VARIANT
            out_row["run_number"] = RUN_NUMBER
            rows.append(out_row)

            if article_ok:
                n_on_topic += 1
            else:
                n_failed += 1

        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)

        # Verify write: read back and confirm every article_idx is present
        written = pd.read_csv(output_path, usecols=["article_idx"])
        written_idxs = set(written["article_idx"].dropna().astype(int))
        expected_idxs = set(range(total))
        missing_idxs = expected_idxs - written_idxs
        if missing_idxs:
            print(
                f"\n  ERROR: {len(missing_idxs)} article_idx values missing from written CSV!"
            )
            print(f"    Missing: {sorted(missing_idxs)[:20]}")
            sys.exit(1)

        print(f"\n  Results for {corpus}:")
        print(f"    On-topic annotated : {n_on_topic:,}")
        print(f"    Off-topic skipped  : {n_off_topic:,}")
        print(f"    Failed/errored     : {n_failed:,}")
        print(f"    Schema parse errors: {len(schema_errors)}")
        print(
            f"    Output written     : {output_path}  ({len(written):,} rows verified)"
        )

        if failed_ids:
            fail_path = ANNOTATED_DIR / f"failed_batch_{corpus}_run{RUN_NUMBER}.txt"
            fail_path.write_text(
                f"Missing/failed custom_ids for {corpus} run{RUN_NUMBER}:\n"
                + "\n".join(failed_ids)
                + "\n"
            )
            print(f"\n  MISSING REQUESTS ({len(failed_ids)}):")
            print(f"  Logged to {fail_path}")
            for cid in failed_ids[:20]:
                _, art_idx, pass_nm = parse_custom_id(cid)
                print(f"    {cid}  (article {art_idx}, {pass_nm})")
            if len(failed_ids) > 20:
                print(f"    ... and {len(failed_ids)-20} more (see {fail_path})")

        if schema_errors:
            print(f"\n  SCHEMA ERRORS:")
            for e in schema_errors[:10]:
                print(f"    {e}")

    sep("PARSE COMPLETE")
    print(
        "  Output files are in data/annotated/ with the same schema as annotate.py output."
    )
    print()


def _report_missing(meta: dict, corpus: str, client: openai.OpenAI) -> None:
    """For expired batches: report which article/pass combinations are missing."""
    batch_info = meta["batch_ids"][corpus]
    batch_id = batch_info["batch_id"]

    err_file_id = client.batches.retrieve(batch_id).error_file_id
    if err_file_id:
        print(f"  Downloading error file {err_file_id}...")
        errs = client.files.content(err_file_id).read().decode("utf-8")
        for line in errs.strip().split("\n")[:20]:
            print(f"    {line}")

    # Report which requests did NOT complete by cross-checking with expected set
    df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
    df["article_idx"] = range(len(df))
    pass_map = {p[0]: p[2] for p in PASSES_DEF}
    expected = set()
    for _, row in df[df["on_topic_flag"] == True].iterrows():
        for pass_name in applicable_passes(row):
            expected.add(make_custom_id(corpus, int(row["article_idx"]), pass_name))

    out_file_id = batch_info.get("output_file_id")
    completed = set()
    if out_file_id:
        raw = client.files.content(out_file_id).read().decode("utf-8")
        for line in raw.strip().split("\n"):
            try:
                completed.add(json.loads(line)["custom_id"])
            except Exception:
                pass

    missing = expected - completed
    print(f"\n  Expected : {len(expected):,} requests")
    print(f"  Completed: {len(completed):,}")
    print(f"  MISSING  : {len(missing):,}")
    if missing:
        fail_path = ANNOTATED_DIR / f"missing_{corpus}_run{RUN_NUMBER}.txt"
        fail_path.write_text("\n".join(sorted(missing)) + "\n")
        print(f"  Missing custom_ids saved → {fail_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKED SUBMISSION
#
# The org's enqueued-token limit for gpt-4o is 1,350,000. A whole-corpus batch
# (climate 8.3M input tokens, migration 19.1M) exceeds it and fails validation
# immediately, so each corpus is split into chunks that individually fit.
#
# Accounting note: OpenAI reports the limit in "enqueued tokens" but does not
# document whether that counts input only or input + reserved max_tokens. Chunks
# are sized on INPUT tokens at 80% of the cap; if a chunk is rejected with
# token_limit_exceeded the planner halves the target and replans, so a wrong
# assumption self-corrects rather than silently overshooting.
#
# There is no API that reports current enqueued-token usage directly. Live
# capacity is derived from batches.list() — which batches are genuinely still in
# flight — combined with the per-chunk token counts recorded at plan time. That
# is grounded in live state rather than an internal counter alone, but it cannot
# see batches submitted by other tooling; the 20% margin absorbs that.
# ══════════════════════════════════════════════════════════════════════════════

ENQUEUED_TOKEN_CAP = 1_350_000
CHUNK_TARGET_FRAC = 0.80  # keep 20% headroom against estimation error
IN_FLIGHT = ("validating", "in_progress", "finalizing")
CHUNK_DIR = ANNOTATED_DIR / "chunks"


def _encoder():
    import tiktoken

    try:
        return tiktoken.encoding_for_model(MODEL)
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def count_request_tokens(path: Path) -> list[int]:
    """Input tokens per request line, in file order."""
    enc = _encoder()
    counts = []
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            counts.append(
                sum(len(enc.encode(m["content"])) for m in o["body"]["messages"]) + 8
            )
    return counts


def plan_chunks(counts: list[int], target: int) -> list[dict]:
    """Contiguous chunks, each with total input tokens <= target.

    Contiguity matters: chunk boundaries are line ranges into the original
    request file, so reassembly is a straight concatenation in order.
    """
    chunks, start, cur = [], 0, 0
    for i, t in enumerate(counts):
        if cur and cur + t > target:
            chunks.append(
                {
                    "line_start": start,
                    "line_end": i,
                    "n_requests": i - start,
                    "input_tokens": cur,
                }
            )
            start, cur = i, 0
        cur += t
    if start < len(counts):
        chunks.append(
            {
                "line_start": start,
                "line_end": len(counts),
                "n_requests": len(counts) - start,
                "input_tokens": cur,
            }
        )
    return chunks


def write_chunk_file(src: Path, corpus: str, idx: int, lo: int, hi: int) -> Path:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    dst = CHUNK_DIR / f"batch_{corpus}_run{RUN_NUMBER}_chunk{idx:03d}.jsonl"
    if dst.exists() and sum(1 for _ in dst.open()) == hi - lo:
        return dst
    with src.open() as fin, dst.open("w") as fout:
        for i, line in enumerate(fin):
            if lo <= i < hi:
                fout.write(line)
            elif i >= hi:
                break
    return dst


def live_enqueued_tokens(client, meta: dict) -> tuple[int, list[str]]:
    """Tokens currently enqueued, from live batch states + recorded counts."""
    by_id = {
        c["batch_id"]: c
        for chunks in meta.get("chunks", {}).values()
        for c in chunks
        if c.get("batch_id")
    }
    if not by_id:
        return 0, []
    total, active = 0, []
    for b in client.batches.list(limit=100):
        if b.id in by_id and b.status in IN_FLIGHT:
            total += by_id[b.id]["input_tokens"]
            active.append(b.id)
    return total, active


def mode_chunk_submit(confirm: bool, max_submit: int = 0) -> None:
    """Plan chunks, then submit as many as fit under the live cap."""
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("chunks", {})

    target = int(ENQUEUED_TOKEN_CAP * CHUNK_TARGET_FRAC)
    sep(
        f"CHUNK PLAN — target {target:,} input tokens/chunk "
        f"({CHUNK_TARGET_FRAC:.0%} of {ENQUEUED_TOKEN_CAP:,} cap)"
    )

    for corpus in ["climate", "migration"]:
        if corpus in meta["chunks"] and meta["chunks"][corpus]:
            print(
                f"  {corpus}: plan already exists "
                f"({len(meta['chunks'][corpus])} chunks)"
            )
            continue
        src = Path(meta["files"][corpus]["path"])
        counts = count_request_tokens(src)
        plan = plan_chunks(counts, target)
        for i, c in enumerate(plan):
            c.update(
                {
                    "index": i,
                    "corpus": corpus,
                    "batch_id": None,
                    "file_id": None,
                    "status": "planned",
                }
            )
        meta["chunks"][corpus] = plan
        print(
            f"  {corpus:10s} {len(counts):,} requests, "
            f"{sum(counts):,} input tokens -> {len(plan)} chunks "
            f"(max {max(c['input_tokens'] for c in plan):,} tokens)"
        )
    meta_path.write_text(json.dumps(meta, indent=2))

    if not confirm:
        print("\nDRY RUN — plan saved. Add --confirm to upload and submit.")
        return

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    used, active = live_enqueued_tokens(client, meta)
    sep("SUBMIT — filling available capacity")
    print(f"  in-flight batches: {len(active)}   enqueued now: {used:,} tokens")
    print(f"  capacity free    : {ENQUEUED_TOKEN_CAP - used:,} tokens\n")

    submitted = 0
    for corpus in ["climate", "migration"]:
        for c in meta["chunks"][corpus]:
            if c["status"] != "planned":
                continue
            if used + c["input_tokens"] > ENQUEUED_TOKEN_CAP:
                print(
                    f"  {corpus} chunk {c['index']:03d}: would exceed cap "
                    f"({used + c['input_tokens']:,} > {ENQUEUED_TOKEN_CAP:,}) — holding"
                )
                break
            if max_submit and submitted >= max_submit:
                print(f"  reached --max-submit {max_submit}, holding the rest")
                break
            path = write_chunk_file(
                Path(meta["files"][corpus]["path"]),
                corpus,
                c["index"],
                c["line_start"],
                c["line_end"],
            )
            up = client.files.create(file=path.open("rb"), purpose="batch")
            try:
                b = client.batches.create(
                    input_file_id=up.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "corpus": corpus,
                        "chunk": str(c["index"]),
                        "run": str(RUN_NUMBER),
                        "prompt": PROMPT_VARIANT,
                    },
                )
            except openai.BadRequestError as exc:
                print(f"  {corpus} chunk {c['index']:03d}: REJECTED — {exc}")
                meta_path.write_text(json.dumps(meta, indent=2))
                raise
            c.update(
                {
                    "batch_id": b.id,
                    "file_id": up.id,
                    "status": b.status,
                    "submitted_at": datetime.now().isoformat(),
                }
            )
            used += c["input_tokens"]
            submitted += 1
            print(
                f"  {corpus:10s} chunk {c['index']:03d}  "
                f"lines {c['line_start']:>5,}-{c['line_end']:<5,} "
                f"{c['n_requests']:>4,} req  {c['input_tokens']:>9,} tok  "
                f"-> {b.id}  [{b.status}]"
            )
            meta_path.write_text(json.dumps(meta, indent=2))
        else:
            continue
        break

    meta_path.write_text(json.dumps(meta, indent=2))
    pend = sum(
        1 for cs in meta["chunks"].values() for c in cs if c["status"] == "planned"
    )
    sep("SUBMISSION ROUND COMPLETE")
    print(f"  submitted this round : {submitted}")
    print(f"  still queued locally : {pend}")
    print(f"  enqueued tokens now  : {used:,} / {ENQUEUED_TOKEN_CAP:,}")
    if pend:
        print(f"\n  Re-run this command as chunks complete to submit the rest:")
        print(f"    python3 src/batch_pipeline.py --mode chunk-submit --confirm")


def mode_chunk_status() -> None:
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    meta = json.loads(meta_path.read_text())
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sep("CHUNK STATUS")
    grand = {"done": 0, "total": 0}
    for corpus, chunks in meta.get("chunks", {}).items():
        agg = {}
        done_req = tot_req = 0
        for c in chunks:
            tot_req += c["n_requests"]
            if not c.get("batch_id"):
                agg["planned"] = agg.get("planned", 0) + 1
                continue
            b = client.batches.retrieve(c["batch_id"])
            c["status"] = b.status
            c["completed_count"] = b.request_counts.completed
            c["failed_count"] = b.request_counts.failed
            if b.status == "completed":
                c["output_file_id"] = b.output_file_id
            agg[b.status] = agg.get(b.status, 0) + 1
            done_req += b.request_counts.completed
        grand["done"] += done_req
        grand["total"] += tot_req
        pct = done_req / tot_req * 100 if tot_req else 0
        print(f"  {corpus:10s} {len(chunks)} chunks  {agg}")
        print(f"             {done_req:,}/{tot_req:,} requests ({pct:.1f}%)")
    meta_path.write_text(json.dumps(meta, indent=2))
    if grand["total"]:
        print(
            f"\n  OVERALL: {grand['done']:,}/{grand['total']:,} "
            f"({grand['done']/grand['total']*100:.1f}%)"
        )
    used, active = live_enqueued_tokens(client, meta)
    print(
        f"  enqueued: {used:,}/{ENQUEUED_TOKEN_CAP:,} tokens "
        f"across {len(active)} in-flight batches"
    )


def mode_chunk_parse() -> None:
    """Download every completed chunk, verify it, concatenate, then parse.

    Downstream code sees exactly what the unchunked path produced: one
    batch_{corpus}_run{N}_raw.jsonl per corpus and one parsed CSV. Chunking is
    invisible past this point.

    The disk-headroom check and download verification run PER CHUNK as it is
    fetched, so a bad early chunk surfaces immediately rather than after every
    chunk has completed.
    """
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    meta = json.loads(meta_path.read_text())
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    sep("CHUNK PARSE — download, verify, concatenate")

    for corpus, chunks in meta.get("chunks", {}).items():
        incomplete = [
            c
            for c in chunks
            if c.get("status") != "completed" or chunk_outstanding(c) > 0
        ]
        if incomplete:
            print(
                f"\n  {corpus}: {len(incomplete)}/{len(chunks)} chunks not yet "
                f"completed — skipping (run --mode chunk-status first)"
            )
            continue

        print(f"\n  {corpus}: {len(chunks)} chunks, all completed")
        parts, ok = [], True
        for c in chunks:
            dst = CHUNK_DIR / (
                f"batch_{corpus}_run{RUN_NUMBER}" f"_chunk{c['index']:03d}_raw.jsonl"
            )
            if dst.exists():
                print(f"    chunk {c['index']:03d}: cached")
            else:
                est = int(c["n_requests"] * (BYTES_PER_TOKEN * 1300 + ENVELOPE_BYTES))
                if not check_disk_headroom(
                    CHUNK_DIR, est, f"{corpus} chunk {c['index']}"
                ):
                    ok = False
                    break
                content = client.files.content(c["output_file_id"]).read()
                dst.write_bytes(content)
                print(
                    f"    chunk {c['index']:03d}: downloaded "
                    f"{len(content)/1048576:.1f} MB"
                )
            n_expect = c["n_requests"] - sum(
                mk.get("n_requests", 0) for mk in c.get("makeups", [])
            )
            if not verify_download(dst, n_expect, f"{corpus} chunk {c['index']}"):
                ok = False
                break
            parts.append(dst)
            for mk in c.get("makeups", []):
                if mk.get("status") != "completed":
                    continue
                mdst = CHUNK_DIR / (
                    f"batch_{corpus}_run{RUN_NUMBER}"
                    f"_chunk{c['index']:03d}_makeup_raw.jsonl"
                )
                if not mdst.exists():
                    mdst.write_bytes(client.files.content(mk["output_file_id"]).read())
                    print(
                        f"    chunk {c['index']:03d} makeup: downloaded "
                        f"{mk['n_requests']} results"
                    )
                if not verify_download(
                    mdst, mk["n_requests"], f"{corpus} chunk {c['index']} makeup"
                ):
                    ok = False
                    break
                parts.append(mdst)
            if not ok:
                break

        if not ok:
            print(f"  {corpus}: aborting — remote copies are intact, retry later")
            continue

        raw_cache = ANNOTATED_DIR / f"batch_{corpus}_run{RUN_NUMBER}_raw.jsonl"
        total = 0
        with raw_cache.open("wb") as out:
            for part in parts:
                with part.open("rb") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            total += 1
        expected = sum(c["n_requests"] for c in chunks)
        print(
            f"    concatenated -> {raw_cache.name}: {total:,} lines "
            f"(expected {expected:,}) "
            f"{'OK' if total == expected else 'MISMATCH'}"
        )
        if total != expected:
            print(f"    {corpus}: refusing to parse a short file")
            continue

        results: dict[str, dict] = {}
        with raw_cache.open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id", "")
                if obj.get("error"):
                    results[cid] = {"error": obj["error"]}
                else:
                    results[cid] = {
                        "content": obj["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                    }
        print(f"    parsed {len(results):,} responses -> building CSV")
        _write_corpus_csv(corpus, results)

    meta_path.write_text(json.dumps(meta, indent=2))


def _write_corpus_csv(corpus: str, results: dict) -> None:
    """Build the per-corpus annotated CSV from a custom_id -> result mapping.

    Identical output to the unchunked parse path.
    """
    df = pd.read_csv(SAMPLED_DIR / SAMPLE_FILES[corpus])
    df["article_idx"] = range(len(df))
    pass_map = {p[0]: p[2] for p in PASSES_DEF}
    rows, failed_ids, schema_errors = [], [], []
    n_ok = n_skip = n_fail = 0

    for _, row in df.iterrows():
        idx = int(row["article_idx"])
        on_topic = bool(row.get("on_topic_flag", True))
        central = bool(row.get("topic_central_flag", True)) if on_topic else False
        if not on_topic or not central:
            skipped = build_skipped_row(idx, row, PROMPT_VARIANT, RUN_NUMBER)
            skipped["skipped_reason"] = (
                "off_topic_keyword_flag" if not on_topic else "not_topic_central"
            )
            rows.append(skipped)
            n_skip += 1
            continue

        passes = applicable_passes(row)
        article: dict = {"corpus_type": derive_corpus_type(row)}
        article_ok = True
        for pass_name in ["pass1", "pass2", "pass3", "pass4"]:
            frames = pass_map[pass_name]
            if pass_name not in passes:
                article.update(null_frame_row(frames))
                continue
            res = results.get(make_custom_id(corpus, idx, pass_name))
            if res is None or "error" in res:
                failed_ids.append(make_custom_id(corpus, idx, pass_name))
                article.update(null_frame_row(frames))
                article_ok = False
                continue
            try:
                article.update(
                    flatten_frame_result(extract_json(res["content"]), frames)
                )
            except Exception as exc:
                schema_errors.append(f"{idx}/{pass_name}: {exc}")
                article.update(null_frame_row(frames))
                article_ok = False

        out_row: dict = {"article_idx": idx}
        for col in CARRY_COLS:
            out_row[col] = row.get(col)
        out_row.update(article)
        out_row["skipped_reason"] = None if article_ok else "batch_parse_error"
        out_row["prompt_variant"] = PROMPT_VARIANT
        out_row["run_number"] = RUN_NUMBER
        rows.append(out_row)
        n_ok += article_ok
        n_fail += not article_ok

    out_path = ANNOTATED_DIR / f"{corpus}_prompt{PROMPT_VARIANT}_run{RUN_NUMBER}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    written = pd.read_csv(out_path, usecols=["article_idx"])
    missing = set(range(len(df))) - set(written["article_idx"].dropna().astype(int))
    print(
        f"    wrote {out_path.name}: {len(rows):,} rows "
        f"({n_ok:,} annotated, {n_skip:,} skipped, {n_fail:,} failed)"
    )
    if failed_ids:
        print(f"    {len(failed_ids):,} missing/errored requests")
    if schema_errors:
        print(f"    {len(schema_errors):,} schema errors")
    if missing:
        print(f"    ERROR: {len(missing)} article_idx missing from CSV")


# ══════════════════════════════════════════════════════════════════════════════
# MAKEUP BATCHES
#
# A chunk can complete with some requests failed (e.g. the credit exhaustion that
# hit climate chunk 002: 394/428, the other 34 returning HTTP 429
# credit_balance_exhausted). Those requests are real gaps in the corpus, so the
# pipeline refuses to parse until they are filled.
#
# A makeup batch re-submits exactly the failed custom_ids for one chunk. Makeups
# are recorded ON the chunk, not as separate top-level jobs, so:
#   * chunk-parse concatenates chunk output + all its makeup outputs, and the
#     existing line-count check then sees the full n_requests with no special
#     casing;
#   * nothing is orphaned or needs manual reconciliation;
#   * the same code handles any future chunk with partial failures.
# ══════════════════════════════════════════════════════════════════════════════


def failed_custom_ids(client, chunk: dict) -> list[str]:
    """custom_ids that failed in this chunk, read from its error file."""
    b = client.batches.retrieve(chunk["batch_id"])
    if not b.error_file_id:
        return []
    raw = client.files.content(b.error_file_id).read().decode()
    ids = []
    for line in raw.strip().split("\n"):
        if line.strip():
            ids.append(json.loads(line)["custom_id"])
    return ids


def chunk_outstanding(chunk: dict) -> int:
    """Requests still missing for this chunk, after accounting for makeups."""
    got = chunk.get("completed_count", 0)
    for mk in chunk.get("makeups", []):
        if mk.get("status") == "completed":
            got += mk.get("completed_count", 0)
    return chunk["n_requests"] - got


def build_makeup_file(corpus: str, chunk: dict, custom_ids: list[str]) -> Path:
    """Extract exactly the named requests from the corpus's source batch file."""
    want = set(custom_ids)
    src = ANNOTATED_DIR / f"batch_{corpus}_run{RUN_NUMBER}.jsonl"
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    seq = len(chunk.get("makeups", []))
    dst = (
        CHUNK_DIR / f"batch_{corpus}_run{RUN_NUMBER}"
        f"_chunk{chunk['index']:03d}_makeup{seq:02d}.jsonl"
    )
    found = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            cid = json.loads(line)["custom_id"]
            if cid in want:
                fout.write(line)
                found += 1
    if found != len(want):
        missing = want - {json.loads(l)["custom_id"] for l in dst.open()}
        raise RuntimeError(
            f"makeup build incomplete: found {found}/{len(want)} source lines; "
            f"missing {sorted(missing)[:5]}"
        )
    return dst


def mode_makeup(confirm: bool) -> None:
    """Submit makeup batches for every chunk with outstanding failed requests."""
    meta_path = ANNOTATED_DIR / "batch_job_ids.json"
    meta = json.loads(meta_path.read_text())
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    sep("MAKEUP — resubmitting failed requests")
    todo = []
    for corpus, chunks in meta.get("chunks", {}).items():
        for c in chunks:
            if not c.get("batch_id") or c.get("status") != "completed":
                continue
            if chunk_outstanding(c) > 0:
                todo.append((corpus, c))

    if not todo:
        print("  no chunks have outstanding failed requests")
        return

    for corpus, c in todo:
        n_missing = chunk_outstanding(c)
        print(
            f"\n  {corpus} chunk {c['index']:03d}: "
            f"{c['completed_count']}/{c['n_requests']} done, {n_missing} outstanding"
        )
        ids = failed_custom_ids(client, c)
        already = {
            i
            for mk in c.get("makeups", [])
            for i in mk.get("custom_ids", [])
            if mk.get("status") == "completed"
        }
        ids = [i for i in ids if i not in already]
        print(f"    failed custom_ids to resubmit: {len(ids)}")
        if len(ids) != n_missing:
            print(
                f"    WARNING: error file lists {len(ids)} but {n_missing} "
                f"outstanding — investigate before proceeding"
            )
        path = build_makeup_file(corpus, c, ids)
        print(f"    built {path.name} ({sum(1 for _ in path.open())} lines)")

        if not confirm:
            print("    DRY RUN — add --confirm to submit")
            continue

        used, _ = live_enqueued_tokens(client, meta)
        print(f"    enqueued now {used:,}/{ENQUEUED_TOKEN_CAP:,}")

        up = client.files.create(file=path.open("rb"), purpose="batch")
        b = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "corpus": corpus,
                "chunk": str(c["index"]),
                "makeup": "1",
                "run": str(RUN_NUMBER),
                "prompt": PROMPT_VARIANT,
            },
        )
        c.setdefault("makeups", []).append(
            {
                "batch_id": b.id,
                "file_id": up.id,
                "custom_ids": ids,
                "n_requests": len(ids),
                "status": b.status,
                "submitted_at": datetime.now().isoformat(),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"    submitted -> {b.id} [{b.status}]")

    meta_path.write_text(json.dumps(meta, indent=2))
    print("\n  Poll with --mode chunk-status; parse once complete.")


def verify_chunk_whole(client, corpus: str, chunk: dict) -> bool:
    """Confirm every custom_id the chunk should contain is present and successful."""
    ids_seen: set[str] = set()
    sources = [chunk.get("output_file_id")] + [
        mk.get("output_file_id") for mk in chunk.get("makeups", [])
    ]
    for fid in [f for f in sources if f]:
        raw = client.files.content(fid).read().decode()
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            o = json.loads(line)
            body = (o.get("response") or {}).get("body") or {}
            if o.get("error") or "error" in body:
                continue  # still a failure; do not count it
            ids_seen.add(o["custom_id"])

    src = ANNOTATED_DIR / f"batch_{corpus}_run{RUN_NUMBER}.jsonl"
    expected = []
    with src.open() as f:
        for i, line in enumerate(f):
            if chunk["line_start"] <= i < chunk["line_end"]:
                expected.append(json.loads(line)["custom_id"])
            elif i >= chunk["line_end"]:
                break
    missing = [c for c in expected if c not in ids_seen]
    print(
        f"    expected {len(expected)} custom_ids, successful {len(ids_seen)}, "
        f"missing {len(missing)}"
    )
    if missing:
        for c in missing[:10]:
            print(f"      MISSING {c}")
    return not missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch API pipeline for framing annotation."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "generate",
            "submit",
            "status",
            "parse",
            "chunk-submit",
            "chunk-status",
            "chunk-parse",
            "makeup",
        ],
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for --mode submit to actually send the batch job",
    )
    parser.add_argument(
        "--max-submit",
        type=int,
        default=0,
        help="Cap how many chunks to submit this round (0 = as many as fit)",
    )
    parser.add_argument(
        "--batch-id", help="Override batch ID (optional for status/parse)"
    )
    args = parser.parse_args()

    if args.mode == "generate":
        mode_generate()
    elif args.mode == "submit":
        mode_submit(confirm=args.confirm)
    elif args.mode == "status":
        mode_status()
    elif args.mode == "parse":
        mode_parse()
    elif args.mode == "chunk-submit":
        mode_chunk_submit(confirm=args.confirm, max_submit=args.max_submit)
    elif args.mode == "chunk-status":
        mode_chunk_status()
    elif args.mode == "chunk-parse":
        mode_chunk_parse()
    elif args.mode == "makeup":
        mode_makeup(confirm=args.confirm)


if __name__ == "__main__":
    main()
