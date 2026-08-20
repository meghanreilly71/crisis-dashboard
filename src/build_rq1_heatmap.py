"""RQ1 panel 1 — meta-topic x frame prevalence heatmaps.

Aggregates the tidy frame to (corpus, meta, frame) prevalence and emits a
standalone Plotly.js page with the aggregates embedded inline. No per-article
data of any kind reaches the HTML — only counts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
SCHEMA = ROOT / "data" / "final" / "frame_schema.json"
OUT_CSV = ROOT / "data" / "final" / "rq1_heatmap_long.csv"
OUT_HTML = ROOT / "dashboard" / "rq1_heatmap.html"

CORPUS_PASSES = {
    "climate": ["pass1", "pass3", "pass4"],
    "migration": ["pass1", "pass2", "pass4"],
}

FRAME_LABELS = {
    "conflict": "Conflict", "human_interest": "Human Interest",
    "economic": "Economic", "deservingness": "Deservingness",
    "responsibility": "Responsibility", "humanitarian": "Humanitarian",
    "security": "Security", "policy": "Policy", "scientific": "Scientific",
    "crisis": "Crisis", "solutions": "Solutions", "victim": "Victim",
    "skepticism": "Skepticism", "securitization": "Securitization",
    "othering": "Othering", "agency": "Agency",
}

# Reliability is read from validity_FINAL.csv, the verified source of truth, so
# the panel can never drift from it again. It previously lived in a hardcoded
# dict where `crisis` and `skepticism` carried stale values (.091 and .375
# against the correct .516 and .554) — a column-shift error caught during thesis
# table generation.
#
# The statistic displayed is COHEN'S KAPPA, not Krippendorff's alpha. The old
# dict was labelled alpha but held kappa for all 16 frames; both are in the CSV
# and they differ (crisis kappa .516 vs alpha .519), so the label is now correct
# rather than the number being silently swapped.
VALIDITY = ROOT / "data" / "validity" / "validity_FINAL.csv"

_TIER_MAP = {
    "clears 0.6": "clears",
    "below 0.6 - unresolved": "unresolved",
    "below 0.6 - structural": "structural",
}


def load_reliability() -> dict[str, tuple[float, str]]:
    """frame -> (cohen's kappa, tier) straight from validity_FINAL.csv."""
    v = pd.read_csv(VALIDITY)
    out: dict[str, tuple[float, str]] = {}
    for _, r in v.iterrows():
        # normalise the en/em dash used in the tier strings
        key = re.sub(r"[‐-―]", "-", str(r["tier"])).strip()
        assert key in _TIER_MAP, f"unrecognised tier {r['tier']!r} for {r['label']}"
        out[str(r["label"])] = (round(float(r["cohens_kappa"]), 3), _TIER_MAP[key])
    return out


RELIABILITY = load_reliability()

THIN_THRESHOLD = 10

# Meta-topics excluded from THIS PANEL's topic axis only. NOISE is BERTopic's
# outlier / no-cluster bucket, not a thematic category, so it does not belong on
# a topic axis. The articles stay in annotated_tidy.csv and remain available to
# every other panel — this is a display-scope exclusion, not a corpus one.
HEATMAP_EXCLUDE_META = {"climate": {"NOISE"}, "migration": set()}

# The three frames that remain below kappa = 0.6 with no resolved cause. These
# are the only frames marked in the panel. Asserted against the CSV's own tier
# column in aggregate(), so a tier change upstream fails the build rather than
# silently changing which frames carry the asterisk.
CAVEAT_FRAMES = {"crisis", "solutions", "security"}


def aggregate() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(TIDY, low_memory=False)
    schema = json.load(open(SCHEMA))

    present_vals = set()
    for e in schema.values():
        present_vals |= set(df[e["present_col"]].dropna().unique())
    assert present_vals <= {"yes", "no"}, f"unexpected encoding: {present_vals}"
    print(f"  value encoding confirmed: {sorted(present_vals)} (string yes/no)")

    # Three climate meta labels differ only by trailing whitespace (one by a
    # trailing tab), which splits a single topic across two heatmap rows and
    # makes the remainder look like a thin group. Trimming is a labelling fix,
    # not a data change: no article moves category and no count is lost.
    raw_meta = df["meta"].astype(str)
    df["meta"] = raw_meta.str.replace(r"\s+", " ", regex=True).str.strip()
    merged = (
        pd.DataFrame({"raw": raw_meta, "clean": df["meta"]})
        .groupby("clean")["raw"].nunique()
    )
    merged = merged[merged > 1]
    if len(merged):
        print(f"\n  whitespace-normalised meta labels ({len(merged)} merges):")
        for clean in merged.index:
            variants = sorted(raw_meta[df["meta"] == clean].unique())
            counts = [int((raw_meta == v).sum()) for v in variants]
            print(f"     {clean!r}")
            for v, n in zip(variants, counts):
                print(f"        n={n:>4}  {v!r}")

    assert {s for s, (_, t) in RELIABILITY.items() if t == "unresolved"} == CAVEAT_FRAMES, \
        "CAVEAT_FRAMES drifted from the unresolved tier in RELIABILITY"

    rows, payload = [], {}
    for corpus, passes in CORPUS_PASSES.items():
        sub = df[df["corpus"] == corpus]
        frames = [s for s, e in schema.items() if e["pass"] in passes]

        drop = HEATMAP_EXCLUDE_META[corpus]
        if drop:
            n_before, g_before = len(sub), sub["meta"].nunique()
            hit = sub["meta"].isin(drop)
            sub = sub[~hit]
            print(f"\n  {corpus}: excluded meta {sorted(drop)} from this panel — "
                  f"{int(hit.sum())} articles")
            print(f"     rows        {n_before:,} -> {len(sub):,}")
            print(f"     meta groups {g_before} -> {sub['meta'].nunique()}")

        print(f"\n  corpus == '{corpus}': {len(sub):,} rows, {len(frames)} frames")

        for slug in frames:
            col = schema[slug]["present_col"]
            n_null = int(sub[col].isna().sum())
            assert n_null == 0, f"{corpus}/{slug}: {n_null} nulls in an applicable frame"

        sizes = (sub.groupby("meta").size().sort_values(ascending=False))
        metas = list(sizes.index)

        for meta in metas:
            g = sub[sub["meta"] == meta]
            for slug in frames:
                col = schema[slug]["present_col"]
                applicable = g[col].notna()
                n_app = int(applicable.sum())
                assert n_app > 0, f"{corpus}/{meta}/{slug}: n_applicable == 0"
                n_pres = int((g.loc[applicable, col] == "yes").sum())
                rows.append({
                    "corpus": corpus, "meta": meta, "frame": slug,
                    "frame_label": FRAME_LABELS[slug],
                    "n_present": n_pres, "n_applicable": n_app,
                    "pct": round(100.0 * n_pres / n_app, 4),
                    "alpha": RELIABILITY[slug][0], "tier": RELIABILITY[slug][1],
                })

        long = pd.DataFrame([r for r in rows if r["corpus"] == corpus])
        z, cd = [], []
        for meta in metas:
            zr, cr = [], []
            for slug in frames:
                r = long[(long["meta"] == meta) & (long["frame"] == slug)].iloc[0]
                zr.append(r["pct"])
                cr.append([int(r["n_present"]), int(r["n_applicable"])])
            z.append(zr)
            cd.append(cr)

        payload[corpus] = {
            "n_rows": int(len(sub)),
            "metas": metas,
            "meta_n": [int(sizes[m]) for m in metas],
            "thin": [bool(sizes[m] < THIN_THRESHOLD) for m in metas],
            "frames": [FRAME_LABELS[s] for s in frames],
            "frame_slugs": frames,
            "alpha": [RELIABILITY[s][0] for s in frames],
            "tier": [RELIABILITY[s][1] for s in frames],
            "z": z,
            "customdata": cd,
        }

        print(f"  meta groups: {len(metas)}  |  thin (n<{THIN_THRESHOLD}): "
              f"{sum(sizes < THIN_THRESHOLD)}")

    return pd.DataFrame(rows), payload


def report_sizes(payload: dict) -> None:
    for corpus, p in payload.items():
        print(f"\n{'=' * 74}\n  META GROUP SIZES — {corpus} (n={p['n_rows']:,})\n{'=' * 74}")
        print(f"  {'meta':<56}{'n':>7}   flag")
        for m, n, thin in zip(p["metas"], p["meta_n"], p["thin"]):
            print(f"  {m[:54]:<56}{n:>7,}   {'THIN (n<10)' if thin else ''}")
        print(f"  {'TOTAL':<56}{sum(p['meta_n']):>7,}")


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ1 — Frame prevalence by meta-topic</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --bg:#ffffff; --fg:#16181d; --muted:#5c6370; --line:#e3e6ea; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa2ad; --line:#2a2e35; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:28px 24px 56px; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; }
  h1 { font-size:23px; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 22px; }
  .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px 6px;
           margin-bottom:26px; }
  .panel h2 { font-size:16px; margin:2px 0 2px; }
  .panel .meta { color:var(--muted); font-size:13px; margin:0 0 8px; }
  /* only the plot scrolls — headings stay put */
  .chart-scroll { overflow-x:auto; }
  .chart { min-width:880px; }
  .caption { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
             margin-top:10px; padding:11px 2px 8px; }
  .key { display:flex; flex-wrap:wrap; gap:16px; margin:6px 0 2px; font-size:12.5px;
         color:var(--muted); }
  .key span { display:flex; align-items:center; gap:6px; }
  .sw { width:22px; height:12px; border-radius:2px; display:inline-block; }
</style>
</head>
<body>
<div class="wrap">
  <h1>RQ1 &middot; Frame prevalence by meta-topic</h1>
  <p class="sub">Share of articles in each meta-topic where the frame is present.
     Cell colour is that percentage; hover for the underlying counts.</p>

  <div class="panel">
    <h2>Climate corpus</h2>
    <p class="meta" id="m-climate"></p>
    <div class="chart-scroll"><div id="climate" class="chart"></div></div>
  </div>

  <div class="panel">
    <h2>Migration corpus</h2>
    <p class="meta" id="m-migration"></p>
    <div class="chart-scroll"><div id="migration" class="chart"></div></div>
  </div>

  <div class="caption">
    <div class="key">
      <span><span class="sw" style="background:#440154"></span>0%</span>
      <span><span class="sw" style="background:#21918c"></span>50%</span>
      <span><span class="sw" style="background:#fde725"></span>100%</span>
      <span>&dagger; thin group (n &lt; 10)</span>
    </div>
    <p>&dagger; marks meta-topics with fewer than 10 articles. These are retained,
      not suppressed, but single articles move their percentages a long way.</p>
  </div>
</div>

<script>
const DATA = __DATA__;

// Frame labels carry no reliability marker. Each cell's kappa is still in the
// hover tooltip, and the tier data stays in the payload.
function tickLabel(name, tier) {
  return name;
}

function render(key, elId) {
  const d = DATA[key];
  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const fg = isDark ? '#e8eaed' : '#16181d';
  const muted = isDark ? '#9aa2ad' : '#5c6370';
  const grid = isDark ? '#2a2e35' : '#e3e6ea';

  const xLabels = d.frames.map((f, i) => tickLabel(f, d.tier[i]));
  const yLabels = d.metas.map((m, i) =>
      (d.thin[i] ? '\\u2020 ' : '') + m + '  (' + d.meta_n[i] + ')');

  const cd = d.customdata.map((row, r) => row.map((c, i) =>
      [c[0], c[1], d.alpha[i].toFixed(3), d.tier[i]]));

  const trace = {
    type: 'heatmap',
    x: xLabels, y: yLabels, z: d.z, customdata: cd,
    colorscale: 'Viridis', zmin: 0, zmax: 100,
    xgap: 1.5, ygap: 1.5,
    colorbar: {
      title: { text: '% present', side: 'right', font: { size: 12, color: muted } },
      tickfont: { size: 11, color: muted }, thickness: 12, len: 0.85,
      outlinewidth: 0, ticksuffix: '%'
    },
    hovertemplate:
      '<b>%{y}</b><br>' +
      'Frame: %{x}<br>' +
      'Present: <b>%{z:.1f}%</b><br>' +
      'Count: %{customdata[0]} / %{customdata[1]} articles<br>' +
      // literal alpha: Plotly hover labels do not decode HTML entities.
      // customdata[3] is the tier; it stays in the payload for other consumers
      // but is deliberately not rendered here.
      // kappa, not alpha: the value shown is Cohen's kappa from validity_FINAL.csv
      'Reliability: \\u03ba = %{customdata[2]}' +
      '<extra></extra>'
  };

  const height = Math.max(300, 46 + d.metas.length * 21);
  Plotly.newPlot(elId, [trace], {
    height: height,
    margin: { l: 300, r: 96, t: 12, b: 104 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: fg, size: 12 },
    xaxis: { side: 'bottom', tickangle: -42, tickfont: { size: 11.5, color: fg },
             gridcolor: grid, ticks: '', automargin: true },
    yaxis: { tickfont: { size: 11, color: fg }, autorange: 'reversed',
             gridcolor: grid, ticks: '', automargin: true }
  }, { displayModeBar: false, responsive: true });

  document.getElementById('m-' + key).textContent =
      d.n_rows.toLocaleString() + ' articles \\u00b7 ' + d.metas.length +
      ' meta-topics \\u00b7 ' + d.frames.length + ' frames';
}

render('climate', 'climate');
render('migration', 'migration');
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  render('climate', 'climate'); render('migration', 'migration');
});
</script>
</body>
</html>
""".replace("__DATA__", data_json)


def main() -> None:
    print(f"{'=' * 74}\n  TASK 1 — aggregate to long format\n{'=' * 74}")
    long, payload = aggregate()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(long):,} rows)")
    print(f"     climate rows  : {int((long['corpus'] == 'climate').sum()):,} "
          f"({payload['climate']['metas'].__len__()} metas x 13 frames)")
    print(f"     migration rows: {int((long['corpus'] == 'migration').sum()):,} "
          f"({payload['migration']['metas'].__len__()} metas x 11 frames)")

    report_sizes(payload)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"\n  wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
