"""RQ3 panel 3 — frame prevalence by outlet.

No time dimension and no topic dimension: this panel is purely
(corpus x outlet x frame). Aggregation is over all years combined.

Carries the two fixes RQ2 needed, applied from the start:
  * legend swatches come from empty proxy traces in a legendgroup, so
    per-bar opacity/pattern arrays can never leak into the legend icon;
  * hovermode is 'x unified', so every outlet in a frame group is readable
    in one box instead of only the nearest bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
SCHEMA = ROOT / "data" / "final" / "frame_schema.json"
OUT_CSV = ROOT / "data" / "final" / "rq3_outlet_long.csv"
OUT_HTML = ROOT / "dashboard" / "rq3_outlet.html"

THIN = 10

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

CAVEAT_FRAMES = {"crisis", "solutions", "security"}

# Fixed order and colour per outlet, identical in both corpus tabs so a bar
# keeps its identity when the reader switches. No grey.
OUTLET_ORDER = ["AD", "FD", "NRC", "Telegraaf", "Trouw", "Volkskrant"]
OUTLET_COLORS = {
    "AD": "#4269d0",         # blue
    "FD": "#efb118",         # gold
    "NRC": "#a463f2",        # violet
    "Telegraaf": "#3ca951",  # green
    "Trouw": "#ff725c",      # coral
    "Volkskrant": "#17becf", # cyan
}


def aggregate() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(TIDY, low_memory=False)
    schema = json.load(open(SCHEMA))

    vals = set()
    for e in schema.values():
        vals |= set(df[e["present_col"]].dropna().unique())
    assert vals <= {"yes", "no"}, f"unexpected encoding: {vals}"
    print(f"  value encoding confirmed: {sorted(vals)}")

    rows, payload = [], {}
    for corpus, passes in CORPUS_PASSES.items():
        sub = df[df["corpus"] == corpus]
        frames = [s for s, e in schema.items() if e["pass"] in passes]
        for slug in frames:
            assert sub[schema[slug]["present_col"]].isna().sum() == 0, \
                f"{corpus}/{slug}: nulls in an applicable frame"

        present = [o for o in OUTLET_ORDER if o in set(sub["outlet_clean"])]
        missing = set(sub["outlet_clean"]) - set(OUTLET_ORDER)
        assert not missing, f"{corpus}: unmapped outlets {missing}"
        print(f"\n  {corpus}: {len(sub):,} rows | {len(frames)} frames | "
              f"{len(present)} outlets")

        series, thin_cells = {}, []
        for outlet in present:
            g = sub[sub["outlet_clean"] == outlet]
            pct, npres, napp = [], [], []
            for slug in frames:
                col = schema[slug]["present_col"]
                applicable = g[g[col].notna()]
                n_app = len(applicable)
                assert n_app > 0, f"{corpus}/{outlet}/{slug}: n_applicable == 0"
                n_pres = int((applicable[col] == "yes").sum())
                p = round(100.0 * n_pres / n_app, 4)
                pct.append(p); npres.append(n_pres); napp.append(n_app)
                if n_app < THIN:
                    thin_cells.append((outlet, slug, n_app))
                rows.append({
                    "corpus": corpus, "outlet_clean": outlet, "frame": slug,
                    "frame_label": FRAME_LABELS[slug],
                    "pct": p, "n_present": n_pres, "n_applicable": n_app,
                })
            series[outlet] = {"pct": pct, "np": npres, "na": napp}

        print(f"     per-outlet totals: "
              f"{ {o: series[o]['na'][0] for o in present} }")
        print(f"     cells with n_applicable < {THIN}: "
              f"{len(thin_cells)}{' -> ' + str(thin_cells) if thin_cells else ''}")

        payload[corpus] = {
            "frames": frames,
            "frame_labels": [FRAME_LABELS[s] for s in frames],
            "caveat": [s in CAVEAT_FRAMES for s in frames],
            "outlets": present,
            "colors": [OUTLET_COLORS[o] for o in present],
            "outlet_n": [series[o]["na"][0] for o in present],
            "n_total": int(len(sub)),
            "has_thin": bool(thin_cells),
            "series": series,
        }

    return pd.DataFrame(rows), payload


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ3 — Frame prevalence by outlet</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --bg:#ffffff; --fg:#16181d; --muted:#5c6370; --line:#e3e6ea; --accent:#4269d0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa2ad; --line:#2a2e35; --accent:#97bbf5; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:28px 24px 56px; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; }
  h1 { font-size:23px; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 20px; }
  .controls { display:flex; flex-wrap:wrap; gap:20px; align-items:center; margin-bottom:16px; }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .seg button { appearance:none; border:0; background:transparent; color:var(--fg);
                padding:7px 16px; font-size:14px; cursor:pointer; }
  .seg button[aria-selected="true"] { background:var(--accent); color:#fff; }
  .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px 6px; }
  .panel .meta { color:var(--muted); font-size:13px; margin:0 0 6px; }
  .chart-scroll { overflow-x:auto; }
  .chart { min-width:1000px; }
  .caption { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
             margin-top:10px; padding:11px 2px 8px; }
  .caption p { margin:0 0 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>RQ3 &middot; Frame prevalence by outlet</h1>
  <p class="sub">Share of each outlet's articles where the frame is present, all
     years combined. Click a legend entry to hide or show that outlet.</p>

  <div class="controls">
    <div class="seg" role="tablist" id="corpusToggle">
      <button role="tab" data-corpus="climate" aria-selected="true">Climate</button>
      <button role="tab" data-corpus="migration" aria-selected="false">Migration</button>
    </div>
  </div>

  <div class="panel">
    <p class="meta" id="meta"></p>
    <div class="chart-scroll"><div id="chart" class="chart"></div></div>
    <div class="caption" id="caption">
      <p id="thinNote"></p>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
let corpus = 'climate';

function render() {
  const d = DATA[corpus];
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const fg = dark ? '#e8eaed' : '#16181d';
  const muted = dark ? '#9aa2ad' : '#5c6370';
  const grid = dark ? '#2a2e35' : '#e3e6ea';

  // no reliability marker on the labels; the tier data stays in the payload
  const xLabels = d.frame_labels.slice();

  // Legend swatches come from empty proxy traces, never from the bar traces
  // themselves: per-bar opacity/pattern arrays would otherwise be sampled at
  // index 0 and become the legend icon (the RQ2 legend bug).
  const proxies = d.outlets.map((o, i) => ({
    type: 'scatter', mode: 'markers', name: o, legendgroup: o,
    x: [null], y: [null],
    marker: { size: 11, symbol: 'square', color: d.colors[i] },
    showlegend: true, hoverinfo: 'skip'
  }));

  const bars = d.outlets.map((o, i) => {
    const s = d.series[o];
    return {
      type: 'bar', name: o, legendgroup: o, showlegend: false,
      x: xLabels, y: s.pct,
      customdata: s.na.map((n, j) => [s.np[j], n]),
      marker: {
        color: d.colors[i],
        // low-confidence cells: faded + hatched. Never fires on the current
        // data (every outlet total is >= 76) but is kept so a future rerun
        // with a thinner corpus cannot silently plot them at full weight.
        opacity: s.na.map(n => n < 10 ? 0.45 : 1),
        pattern: { shape: s.na.map(n => n < 10 ? '/' : ''), size: 5, solidity: 0.35 }
      },
      hovertemplate: '%{y:.1f}%  \\u00b7  %{customdata[0]}/%{customdata[1]} articles'
    };
  });

  Plotly.react('chart', proxies.concat(bars), {
    height: 540,
    margin: { l: 62, r: 150, t: 10, b: 116 },
    barmode: 'group', bargap: 0.28, bargroupgap: 0.06,
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: fg, size: 12 },
    xaxis: { type: 'category', tickangle: -38, gridcolor: grid,
             tickfont: { size: 11.5, color: fg }, automargin: true },
    yaxis: { title: { text: '% of articles with frame present',
                      font: { size: 12.5, color: muted } },
             range: [0, 104], tick0: 0, dtick: 20, ticksuffix: '%',
             gridcolor: grid, zeroline: false, tickfont: { color: fg } },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: dark ? '#20242b' : '#ffffff', bordercolor: grid,
                  font: { size: 11.5, color: fg } },
    legend: { orientation: 'v', x: 1.01, xanchor: 'left', y: 1,
              font: { size: 11.5, color: fg } }
  }, { displayModeBar: false, responsive: true });

  document.getElementById('meta').textContent =
    d.n_total.toLocaleString() + ' articles \\u00b7 ' + d.outlets.length +
    ' outlets \\u00b7 ' + d.frames.length + ' frames \\u00b7 ' + corpus + ' corpus (' +
    d.outlets.map((o, i) => o + ' ' + d.outlet_n[i]).join(', ') + ')';

  // Only says anything when there is actually something to flag; when no cell
  // is thin the paragraph is emptied and hidden rather than stating a negative.
  const note = document.getElementById('thinNote');
  note.textContent = d.has_thin
    ? 'Faded, hatched bars rest on fewer than 10 articles and should not be read as estimates.'
    : '';
  note.style.display = d.has_thin ? '' : 'none';
  // the thin note is now the caption's only content, so hide the whole block
  // (and its top border) rather than leave an empty bordered box
  document.getElementById('caption').style.display = d.has_thin ? '' : 'none';
}

document.getElementById('corpusToggle').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  corpus = b.dataset.corpus;
  [...e.currentTarget.querySelectorAll('button')].forEach(x =>
    x.setAttribute('aria-selected', String(x === b)));
  render();
});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);

render();
</script>
</body>
</html>
"""


def main() -> None:
    print(f"{'=' * 78}\n  RQ3 — aggregate\n{'=' * 78}")
    long, payload = aggregate()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(long):,} rows)")
    assert "meta" not in long.columns and "label" not in long.columns
    print(f"  columns: {list(long.columns)}")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(
        _TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8")
    print(f"  wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
