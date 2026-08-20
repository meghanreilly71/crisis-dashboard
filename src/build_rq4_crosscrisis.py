"""RQ4 panel 4 — cross-crisis small multiples.

The 8 frames coded identically in BOTH corpora, one subplot each, climate vs
migration overlaid. Aggregates are re-used from rq2_line_long.csv (all-outlets
level) rather than recomputed, so this panel cannot drift from RQ2.

Layout is a single 4x2 figure. Frame order is pass1 (5) then pass4 (3), so the
row break falls inside pass1 and Responsibility sits on the bottom row beside
the cross-crisis frames. Rows therefore do NOT equal blocks, and the headings
are drawn over *runs* of consecutive same-block subplots instead of over whole
rows — with a vertical divider in row 2 marking where pass1 ends and pass4
begins, so the distinction survives the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RQ2_CSV = ROOT / "data" / "final" / "rq2_line_long.csv"
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
SCHEMA = ROOT / "data" / "final" / "frame_schema.json"
OUT_CSV = ROOT / "data" / "final" / "rq4_crosscrisis_long.csv"
OUT_HTML = ROOT / "dashboard" / "rq4_crosscrisis.html"

YEARS = list(range(2014, 2024))

PASS1 = ["conflict", "human_interest", "economic", "deservingness", "responsibility"]
PASS4 = ["securitization", "othering", "agency"]

FRAME_LABELS = {
    "conflict": "Conflict", "human_interest": "Human Interest",
    "economic": "Economic", "deservingness": "Deservingness",
    "responsibility": "Responsibility", "securitization": "Securitization",
    "othering": "Othering", "agency": "Agency",
}

CORPUS_COLORS = {"climate": "#4269d0", "migration": "#ef7d18"}


def build() -> tuple[pd.DataFrame, dict]:
    rq2 = pd.read_csv(RQ2_CSV, low_memory=False)
    both = PASS1 + PASS4

    src = rq2[(rq2["level"] == "all_outlets") & (rq2["frame"].isin(both))].copy()
    print(f"  source rows from rq2_line_long.csv (all-outlets, 8 frames): {len(src)}")
    assert src["outlet_clean"].isna().all(), "all-outlets level must have null outlet"

    # completeness: every (frame, corpus, year) must be present exactly once
    missing = []
    for f in both:
        for c in ["climate", "migration"]:
            for y in YEARS:
                n = len(src[(src.frame == f) & (src.corpus == c) & (src.year == y)])
                if n != 1:
                    missing.append((f, c, y, n))
    print(f"  (frame, corpus, year) combos missing/duplicated: {len(missing)}")
    assert not missing, missing[:10]
    assert (src["n_applicable"] > 0).all(), "n_applicable == 0 present"
    print(f"  n_applicable range: {int(src.n_applicable.min())}"
          f"–{int(src.n_applicable.max())}  (zero gaps)")

    series: dict[str, dict] = {}
    for f in both:
        series[f] = {}
        for c in ["climate", "migration"]:
            g = src[(src.frame == f) & (src.corpus == c)].sort_values("year")
            assert g["year"].tolist() == YEARS
            series[f][c] = {
                "pct": [round(float(v), 4) for v in g["pct"]],
                "np": [int(v) for v in g["n_present"]],
                "na": [int(v) for v in g["n_applicable"]],
            }

    thin = [(f, c, y, n) for f in both for c in ("climate", "migration")
            for y, n in zip(YEARS, series[f][c]["na"]) if n < 10]
    print(f"  cells with n_applicable < 10: {len(thin)}"
          f"{' -> ' + str(thin) if thin else ''}")

    payload = {
        "years": YEARS,
        "blocks": [
            {"title": "Generic frames",
             "note": "coded identically across both corpora",
             "frames": PASS1},
            {"title": "Cross-crisis frames",
             "note": "designed to test securitization parallels between "
                     "migration and climate",
             "frames": PASS4},
        ],
        "frame_labels": FRAME_LABELS,
        "colors": CORPUS_COLORS,
        "series": series,
    }
    out = src[["corpus", "year", "frame", "frame_label",
               "pct", "n_present", "n_applicable"]].sort_values(
        ["frame", "corpus", "year"])
    return out, payload


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ4 — Cross-crisis frame comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --bg:#ffffff; --fg:#16181d; --muted:#5c6370; --line:#e3e6ea; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa2ad; --line:#2a2e35; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:28px 24px 56px; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; }
  h1 { font-size:23px; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 20px; }
  .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px 6px; }
  .chart-scroll { overflow-x:auto; }
  .chart { min-width:960px; }
  .caption { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
             margin-top:6px; padding:11px 2px 8px; }
  .caption p { margin:0 0 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>RQ4 &middot; Cross-crisis frame comparison</h1>
  <p class="sub">The eight frames coded identically in both corpora, climate
     against migration. Click a legend entry to hide or show a corpus.</p>

  <div class="panel">
    <div class="chart-scroll"><div id="chart" class="chart"></div></div>
  </div>
</div>

<script>
const DATA = __DATA__;

const NCOL = 4, XGAP = 0.06;
const COLW = (1 - XGAP * (NCOL - 1)) / NCOL;
const ROWS = [[0.60, 1.00], [0.00, 0.40]];

// flat frame order with its block index; grid position is derived from this
const FLAT = [];
DATA.blocks.forEach((b, bi) => b.frames.forEach(f => FLAT.push({ slug: f, block: bi })));

// consecutive same-block subplots within a row, so a heading can span exactly
// the subplots it describes rather than assuming one block per row
function runsFor(row) {
  const out = [];
  for (let c = 0; c < NCOL; c++) {
    const i = row * NCOL + c;
    if (i >= FLAT.length) break;
    const b = FLAT[i].block;
    if (out.length && out[out.length - 1].block === b) out[out.length - 1].end = c;
    else out.push({ block: b, start: c, end: c });
  }
  return out;
}

function render() {
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const fg = dark ? '#e8eaed' : '#16181d';
  const muted = dark ? '#9aa2ad' : '#5c6370';
  const grid = dark ? '#2a2e35' : '#e3e6ea';
  const paper = dark ? '#14161a' : '#ffffff';
  const GREY = dark ? '#8b9098' : '#999999';

  const traces = [];
  const layout = {
    height: 640,
    // top margin must clear the first block heading, which sits above the
    // row-1 plotting area at paper y = 1.075
    margin: { l: 52, r: 150, t: 78, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: fg, size: 11.5 },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: dark ? '#20242b' : '#ffffff', bordercolor: grid,
                  font: { size: 11.5, color: fg } },
    legend: { orientation: 'v', x: 1.008, xanchor: 'left', y: 0.99,
              font: { size: 12, color: fg } },
    annotations: [], shapes: []
  };

  // One shared legend for the whole grid: two proxy traces own it, every real
  // trace is showlegend:false in the matching legendgroup. Also keeps per-point
  // marker arrays from ever reaching the legend icon.
  ['climate', 'migration'].forEach(c => {
    traces.push({
      type: 'scatter', mode: 'lines+markers',
      name: c.charAt(0).toUpperCase() + c.slice(1),
      legendgroup: c, showlegend: true, hoverinfo: 'skip',
      x: [null], y: [null],
      line: { color: DATA.colors[c], width: 2 },
      marker: { size: 10, symbol: 'circle', color: DATA.colors[c],
                line: { color: DATA.colors[c], width: 2 } },
      xaxis: 'x', yaxis: 'y'
    });
  });

  const seen = {};
  ROWS.forEach((rng, row) => {
    const [y0, y1] = rng;

    // one heading + rule per run of same-block subplots in this row
    runsFor(row).forEach(run => {
      const b = DATA.blocks[run.block];
      const xa0 = run.start * (COLW + XGAP);
      const xa1 = run.end * (COLW + XGAP) + COLW;
      const first = !seen[run.block];
      seen[run.block] = true;
      layout.annotations.push({
        text: '<b>' + b.title + '</b>' + (first ? ' \\u2014 ' + b.note : ' (cont.)'),
        xref: 'paper', yref: 'paper', x: xa0, y: y1 + 0.075,
        xanchor: 'left', yanchor: 'bottom', showarrow: false,
        font: { size: 12.5, color: muted }
      });
      layout.shapes.push({
        type: 'line', xref: 'paper', yref: 'paper',
        x0: xa0, x1: xa1, y0: y1 + 0.062, y1: y1 + 0.062,
        line: { color: grid, width: 1 }
      });
      // vertical divider where one block ends mid-row
      if (run.start > 0) {
        layout.shapes.push({
          type: 'line', xref: 'paper', yref: 'paper',
          x0: xa0 - XGAP / 2, x1: xa0 - XGAP / 2,
          y0: y0, y1: y1 + 0.062,
          line: { color: grid, width: 1 }
        });
      }
    });

    for (let ci = 0; ci < NCOL; ci++) {
      const flatIdx = row * NCOL + ci;
      if (flatIdx >= FLAT.length) break;
      const slug = FLAT[flatIdx].slug;
      const ax = flatIdx + 1;
      const xa = 'x' + (ax === 1 ? '' : ax), ya = 'y' + (ax === 1 ? '' : ax);
      const x0 = ci * (COLW + XGAP);

      layout['xaxis' + (ax === 1 ? '' : ax)] = {
        domain: [x0, x0 + COLW], anchor: ya.replace('y', 'y'),
        tick0: 2015, dtick: 4, tickfont: { size: 10, color: muted },
        gridcolor: grid, zeroline: false, range: [2013.4, 2023.6]
      };
      layout['yaxis' + (ax === 1 ? '' : ax)] = {
        domain: [y0, y1], anchor: xa,
        range: [-4, 104], tick0: 0, dtick: 25,
        showticklabels: ci === 0, ticksuffix: '%',
        tickfont: { size: 10, color: muted },
        gridcolor: grid, zeroline: false
      };
      layout.annotations.push({
        text: DATA.frame_labels[slug],
        xref: 'paper', yref: 'paper',
        x: x0 + COLW / 2, y: y1 + 0.012,
        xanchor: 'center', yanchor: 'bottom', showarrow: false,
        font: { size: 12, color: fg }
      });

      ['climate', 'migration'].forEach(c => {
        const s = DATA.series[slug][c];
        const colour = DATA.colors[c];
        traces.push({
          type: 'scatter', mode: 'lines+markers',
          name: c.charAt(0).toUpperCase() + c.slice(1),
          legendgroup: c, showlegend: false,
          x: DATA.years, y: s.pct, xaxis: xa, yaxis: ya,
          customdata: s.na.map((n, j) => [
            s.np[j], n,
            (n > 0 && n <= 2) ? '  \\u00b7  n=' + n + ', not meaningful' : ''
          ]),
          line: { color: colour, width: 2 },
          marker: {
            size:   s.na.map(n => n <= 2 ? 8 : (n < 10 ? 5 : 8)),
            symbol: s.na.map(n => n <= 2 ? 'diamond-open' : 'circle'),
            color:  s.na.map(n => n <= 2 ? GREY : (n < 10 ? paper : colour)),
            line: { color: s.na.map(n => n <= 2 ? GREY : colour),
                    width: s.na.map(n => n <= 2 ? 1.8 : (n < 10 ? 1.5 : 1.5)) }
          },
          connectgaps: false,
          hovertemplate:
            '%{y:.1f}%  \\u00b7  %{customdata[0]}/%{customdata[1]} articles' +
            '%{customdata[2]}'
        });
      });
    }
  });

  Plotly.react('chart', traces, layout, { displayModeBar: false, responsive: true });
}

render();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
</script>
</body>
</html>
"""


def main() -> None:
    print(f"{'=' * 78}\n  RQ4 — reuse RQ2 aggregates\n{'=' * 78}")
    long, payload = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(long):,} rows)")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(
        _TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8")
    print(f"  wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
