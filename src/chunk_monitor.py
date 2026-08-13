#!/usr/bin/env python3
"""Automated sequential chunk submission for the full-corpus batch run.

Polls OpenAI for chunk status, submits queued chunks whenever enqueued-token
capacity frees up, and parses each corpus once all its chunks complete.

State lives entirely in data/annotated/batch_job_ids.json — nothing is tracked
only in memory. Every cycle re-reads that file, so killing and restarting the
monitor resumes exactly where it left off: submitted chunks are not resubmitted,
completed chunks are not re-downloaded.

Failure policy: STOP AND FLAG, never skip. Any chunk reaching failed / expired /
cancelled, or any completed chunk containing failed requests, halts the monitor
with a non-zero exit and a loud log entry. Silent continuation would leave gaps
in the corpus that are hard to detect downstream.

Usage
  python3 src/chunk_monitor.py            # run the monitor loop
  python3 src/chunk_monitor.py --status   # one-shot status, no polling
  python3 src/chunk_monitor.py --interval 300
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "annotated" / "chunk_monitor.log"
META = ROOT / "data" / "annotated" / "batch_job_ids.json"

_spec = importlib.util.spec_from_file_location("bp", ROOT / "src" / "batch_pipeline.py")
bp = importlib.util.module_from_spec(_spec)
# batch_pipeline builds an ArgumentParser at import; neutralise argv for the
# import only, then restore it so this script's own flags still parse.
_argv, sys.argv = sys.argv, ["chunk_monitor"]
_spec.loader.exec_module(bp)
sys.argv = _argv

TERMINAL_BAD = ("failed", "expired", "cancelled", "cancelling")

# Batch API pricing for gpt-4o (50% of standard). Discount confirmed applying
# against the OpenAI usage dashboard for the crisis_dashboard project.
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 5.00 / 1e6

# Calibrated fallback output-tokens-per-request, used to project corpora that
# have no completed chunk yet. Source: final_codebook/token_usage.csv.
FALLBACK_OUT_PER_REQ = {"climate": 1333, "migration": 1187}
CALIBRATED_TOTAL = 106.79


def chunk_cost(c: dict) -> float | None:
    """Actual $ for a completed chunk, from the API's own usage object."""
    u = c.get("usage")
    if not u:
        return None
    return u["input_tokens"] * PRICE_IN + u["output_tokens"] * PRICE_OUT


def cost_report(meta: dict) -> tuple[str, float, float]:
    """(text, actual_so_far, projected_total) — all from measured usage."""
    actual = 0.0
    done_req = 0
    ratio: dict[str, float] = {}
    for corpus, chunks in meta.get("chunks", {}).items():
        out_tok = in_tok = n = 0
        for c in chunks:
            cc = chunk_cost(c)
            if cc is None:
                continue
            actual += cc
            done_req += c["n_requests"]
            out_tok += c["usage"]["output_tokens"]
            in_tok += c["usage"]["input_tokens"]
            n += c["n_requests"]
            for mk in c.get("makeups", []):
                mc = chunk_cost(mk)
                if mc is not None:
                    actual += mc
                    out_tok += mk["usage"]["output_tokens"]
        if n:
            ratio[corpus] = out_tok / n  # measured output tokens/request

    projected = 0.0
    for corpus, chunks in meta.get("chunks", {}).items():
        per_req_out = ratio.get(corpus, FALLBACK_OUT_PER_REQ.get(corpus, 1300))
        for c in chunks:
            cc = chunk_cost(c)
            if cc is not None:
                projected += cc
            else:
                projected += (
                    c["input_tokens"] * PRICE_IN
                    + c["n_requests"] * per_req_out * PRICE_OUT
                )

    src = (
        ", ".join(f"{k}={v:.0f}tok/req(measured)" for k, v in ratio.items())
        or "none yet"
    )
    txt = (
        f"  COST       actual ${actual:,.2f} over {done_req:,} req  |  "
        f"projected total ${projected:,.2f}  |  calibrated ${CALIBRATED_TOTAL:,.2f}\n"
        f"             output rate: {src}"
    )
    return txt, actual, projected


def log(msg: str, echo: bool = True) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")
    if echo:
        print(line, flush=True)


def load() -> dict:
    """Always read state from disk — never trust an in-memory copy."""
    return json.loads(META.read_text())


def refresh(client, meta: dict) -> dict:
    """Update every chunk's status from the API and persist."""
    for corpus, chunks in meta.get("chunks", {}).items():
        for c in chunks:
            if not c.get("batch_id"):
                continue
            if (
                c.get("status") == "completed"
                and c.get("output_file_id")
                and c.get("usage")
            ):
                continue  # terminal and fully recorded; no need to re-poll
            b = client.batches.retrieve(c["batch_id"])
            c["status"] = b.status
            c["completed_count"] = b.request_counts.completed
            c["failed_count"] = b.request_counts.failed
            if b.status == "completed":
                c["output_file_id"] = b.output_file_id
                u = getattr(b, "usage", None)
                if u is not None:
                    c["usage"] = {
                        "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens,
                    }
            if b.status in TERMINAL_BAD:
                errs = getattr(getattr(b, "errors", None), "data", None) or []
                c["error"] = (
                    "; ".join(
                        f"{getattr(e,'code','?')}: {getattr(e,'message','')}"[:200]
                        for e in errs
                    )
                    or b.status
                )
            # keep makeup batches attached to this chunk up to date too
            for mk in c.get("makeups", []):
                if mk.get("status") == "completed" and mk.get("usage"):
                    continue
                mb = client.batches.retrieve(mk["batch_id"])
                mk["status"] = mb.status
                mk["completed_count"] = mb.request_counts.completed
                mk["failed_count"] = mb.request_counts.failed
                if mb.status == "completed":
                    mk["output_file_id"] = mb.output_file_id
                    mu = getattr(mb, "usage", None)
                    if mu is not None:
                        mk["usage"] = {
                            "input_tokens": mu.input_tokens,
                            "output_tokens": mu.output_tokens,
                        }
    META.write_text(json.dumps(meta, indent=2))
    return meta


def problems(meta: dict) -> list[str]:
    """Anything that should stop the run."""
    out = []
    for corpus, chunks in meta.get("chunks", {}).items():
        for c in chunks:
            if c.get("status") in TERMINAL_BAD:
                out.append(
                    f"{corpus} chunk {c['index']:03d} [{c['status']}] "
                    f"{c.get('error','')}"
                )
            elif c.get("status") == "completed":
                short = bp.chunk_outstanding(c)
                if short > 0:
                    mk = len(c.get("makeups", []))
                    out.append(
                        f"{corpus} chunk {c['index']:03d} completed but {short} of "
                        f"{c['n_requests']} requests still MISSING"
                        + (f" (after {mk} makeup batch(es))" if mk else "")
                        + " — run: python3 src/batch_pipeline.py --mode makeup --confirm"
                    )
    return out


def summarise(meta: dict, started: datetime | None = None) -> str:
    lines = []
    g_done = g_tot = 0
    for corpus, chunks in meta.get("chunks", {}).items():
        agg: dict[str, int] = {}
        done = tot = 0
        for c in chunks:
            st = c.get("status") or "planned"
            if not c.get("batch_id"):
                st = "queued"
            agg[st] = agg.get(st, 0) + 1
            tot += c["n_requests"]
            done += c.get("completed_count", 0)
            done += sum(
                mk.get("completed_count", 0)
                for mk in c.get("makeups", [])
                if mk.get("status") == "completed"
            )
        g_done += done
        g_tot += tot
        parts = "  ".join(f"{k}={v}" for k, v in sorted(agg.items()))
        pct = done / tot * 100 if tot else 0
        lines.append(f"  {corpus:10s} {len(chunks):2d} chunks | {parts}")
        lines.append(f"             {done:,}/{tot:,} requests ({pct:.1f}%)")
    pct = g_done / g_tot * 100 if g_tot else 0
    lines.append(f"  OVERALL    {g_done:,}/{g_tot:,} requests ({pct:.1f}%)")
    lines.append(cost_report(meta)[0])
    if started:
        el = datetime.now() - started
        lines.append(f"  elapsed    {str(el).split('.')[0]}")
        if g_done:
            eta = el / g_done * (g_tot - g_done)
            lines.append(
                f"  est. remaining {str(timedelta(seconds=int(eta.total_seconds())))}"
            )
    return "\n".join(lines)


def all_submitted(meta: dict) -> bool:
    return all(c.get("batch_id") for chunks in meta["chunks"].values() for c in chunks)


def corpus_complete(meta: dict, corpus: str) -> bool:
    return all(c.get("status") == "completed" for c in meta["chunks"][corpus])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--interval", type=int, default=180, help="seconds between polls (default 180)"
    )
    ap.add_argument("--status", action="store_true", help="one-shot status, then exit")
    ap.add_argument(
        "--cost-cap",
        type=float,
        default=150.0,
        help="Stop and alert if the projected total approaches this "
        "(default 150; set to your real OpenAI usage cap)",
    )
    a = ap.parse_args()

    import openai, os

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if a.status:
        meta = refresh(client, load())
        print(summarise(meta))
        for p in problems(meta):
            print(f"  !! {p}")
        return

    started = datetime.now()
    log("=" * 68, echo=False)
    log(
        f"MONITOR START — poll every {a.interval}s, cap "
        f"{bp.ENQUEUED_TOKEN_CAP:,} tokens"
    )
    parsed: set[str] = set()

    while True:
        try:
            meta = refresh(client, load())
        except Exception as exc:
            # Network blip or API hiccup: log and retry, do not lose state.
            log(f"poll error ({type(exc).__name__}: {exc}) — retrying next cycle")
            time.sleep(a.interval)
            continue

        _txt, actual, projected = cost_report(meta)
        if projected > a.cost_cap * 0.90:
            log("!" * 68)
            log(
                f"STOPPING — projected total ${projected:,.2f} is within 10% of "
                f"the ${a.cost_cap:,.2f} cap"
            )
            log(f"  actual spent so far: ${actual:,.2f}")
            log("  No further chunks submitted. Raise --cost-cap or the OpenAI")
            log("  usage cap, then restart the monitor to resume.")
            log("!" * 68)
            sys.exit(2)

        probs = problems(meta)
        if probs:
            log("!" * 68)
            log("STOPPING — chunk failure detected:")
            for p in probs:
                log(f"   {p}")
            log("Nothing was skipped. Completed chunks remain on OpenAI and in")
            log("metadata; fix the cause and restart the monitor to resume.")
            log("!" * 68)
            sys.exit(1)

        # Submit whatever now fits under the cap.
        if not all_submitted(meta):
            used, active = bp.live_enqueued_tokens(client, meta)
            free = bp.ENQUEUED_TOKEN_CAP - used
            nxt = next(
                (
                    c
                    for chunks in meta["chunks"].values()
                    for c in chunks
                    if not c.get("batch_id")
                ),
                None,
            )
            if nxt and nxt["input_tokens"] <= free:
                log(f"capacity {free:,} free — submitting next chunk(s)")
                try:
                    bp.mode_chunk_submit(confirm=True)
                    meta = load()
                except Exception as exc:
                    log(f"SUBMIT FAILED ({type(exc).__name__}: {exc}) — stopping")
                    sys.exit(1)
            else:
                log(
                    f"waiting — {len(active)} in flight, {used:,}/"
                    f"{bp.ENQUEUED_TOKEN_CAP:,} enqueued",
                    echo=False,
                )

        # Parse each corpus as soon as all its chunks are done.
        for corpus in list(meta.get("chunks", {})):
            if corpus not in parsed and corpus_complete(meta, corpus):
                log(f"{corpus}: all chunks complete — downloading and parsing")
                try:
                    bp.mode_chunk_parse()
                    parsed.add(corpus)
                    log(f"{corpus}: parse finished")
                except Exception as exc:
                    log(
                        f"PARSE FAILED for {corpus} "
                        f"({type(exc).__name__}: {exc}) — stopping"
                    )
                    sys.exit(1)

        if all_submitted(meta) and all(
            corpus_complete(meta, c) for c in meta["chunks"]
        ):
            log("ALL CHUNKS COMPLETE")
            log(summarise(meta, started))
            return

        log(summarise(meta, started), echo=False)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
