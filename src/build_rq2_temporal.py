"""RQ2 panel 2 — frame prevalence over time.

Two aggregation levels are precomputed and embedded, so the outlet dropdown
switches arrays rather than recomputing anything:

  (a) year x frame                  -> "All outlets"
  (b) year x frame x outlet_clean   -> one entry per outlet

NOISE is NOT excluded here. That exclusion was scoped to the RQ1 heatmap's
topic axis (PROJECT_CONTEXT.md 18.1); this panel has no topic axis, so all
1,069 climate / 2,978 migration annotated articles are in scope.

Cells with n_applicable == 0 are emitted as null rather than 0% and are not
written to the long CSV at all: no articles means no measurement. Cells with
1 <= n < 10 ARE plotted, with a hollow marker, so a thin year is visible as
thin instead of reading like a solid estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
SCHEMA = ROOT / "data" / "final" / "frame_schema.json"
OUT_CSV = ROOT / "data" / "final" / "rq2_line_long.csv"
OUT_HTML = ROOT / "dashboard" / "rq2_temporal.html"

YEARS = list(range(2014, 2024))
ALL_OUTLETS = "All outlets"
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

# Same three frames marked in RQ1: below alpha 0.6, cause unresolved.
CAVEAT_FRAMES = {"crisis", "solutions", "security"}

# Colour is keyed to the FRAME, not to its position in the corpus's frame list.
# A positional palette gave the same frame different colours in the two corpora
# (securitization was red in climate but brown in migration) and put a grey on
# whichever frame landed on that slot — skepticism in climate, othering in
# migration — which collided with the grey reserved for n<=2 markers.
#
# pass2 and pass3 never appear in the same chart, so they share one pool of five.
# That caps simultaneous colours at 8 + 5 = 13 (climate) and 8 + 3 = 11
# (migration), and keeps every hue distinct within a chart. No grey anywhere.
_SHARED = {          # pass1 + pass4: present in BOTH corpora
    "conflict": "#4269d0",        # blue
    "human_interest": "#efb118",  # gold
    "economic": "#ff725c",        # coral
    "deservingness": "#6cc5b0",   # aqua
    "responsibility": "#3ca951",  # green
    "securitization": "#e45756",  # red
    "othering": "#9c6b4e",        # brown
    "agency": "#b279a2",          # mauve
}
_EXCLUSIVE = ["#a463f2", "#97bbf5", "#ff8ab7", "#17becf", "#ff7f0e"]
#              violet     pale blue  pink       cyan       orange
FRAME_COLORS = {
    **_SHARED,
    # pass2 — migration only
    "humanitarian": _EXCLUSIVE[0],
    "security": _EXCLUSIVE[1],
    "policy": _EXCLUSIVE[2],
    # pass3 — climate only
    "scientific": _EXCLUSIVE[0],
    "crisis": _EXCLUSIVE[1],
    "solutions": _EXCLUSIVE[2],
    "victim": _EXCLUSIVE[3],
    "skepticism": _EXCLUSIVE[4],
}


def series_for(g: pd.DataFrame, col: str) -> tuple[list, list, list]:
    """Per-year (pct, n_present, n_applicable); pct is None where n == 0."""
    pct, npres, napp = [], [], []
    for y in YEARS:
        yr = g[g["year"] == y]
        applicable = yr[yr[col].notna()]
        n_app = len(applicable)
        if n_app == 0:
            pct.append(None); npres.append(0); napp.append(0)
            continue
        assert n_app > 0
        n_pres = int((applicable[col] == "yes").sum())
        pct.append(round(100.0 * n_pres / n_app, 4))
        npres.append(n_pres)
        napp.append(n_app)
    return pct, npres, napp


def aggregate() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(TIDY, low_memory=False)
    schema = json.load(open(SCHEMA))

    vals = set()
    for e in schema.values():
        vals |= set(df[e["present_col"]].dropna().unique())
    assert vals <= {"yes", "no"}, f"unexpected encoding: {vals}"
    print(f"  value encoding confirmed: {sorted(vals)}")

    # no frame may reuse the low-confidence grey, and no two frames drawn in the
    # same chart may share a colour
    for corpus, passes in CORPUS_PASSES.items():
        fr = [s for s, e in schema.items() if e["pass"] in passes]
        cols = [FRAME_COLORS[s] for s in fr]
        assert len(set(cols)) == len(cols), f"{corpus}: duplicate frame colours"
        assert not ({c.lower() for c in cols} & {"#999999", "#8b9098", "#9498a0"}), \
            f"{corpus}: a frame colour collides with the n<=2 grey"
    print(f"  frame colours: unique within each corpus, none grey")

    rows, payload = [], {}
    for corpus, passes in CORPUS_PASSES.items():
        sub = df[df["corpus"] == corpus]
        frames = [s for s, e in schema.items() if e["pass"] in passes]
        for slug in frames:
            assert sub[schema[slug]["present_col"]].isna().sum() == 0, \
                f"{corpus}/{slug}: nulls in an applicable frame"
        outlets = sorted(sub["outlet_clean"].unique())
        print(f"\n  {corpus}: {len(sub):,} rows | {len(frames)} frames | "
              f"{len(outlets)} outlets | years {min(YEARS)}-{max(YEARS)}")

        series: dict[str, dict] = {}
        for group in [ALL_OUTLETS] + outlets:
            g = sub if group == ALL_OUTLETS else sub[sub["outlet_clean"] == group]
            series[group] = {}
            for slug in frames:
                col = schema[slug]["present_col"]
                pct, npres, napp = series_for(g, col)
                series[group][slug] = {"pct": pct, "np": npres, "na": napp}
                for y, p, a, b in zip(YEARS, pct, npres, napp):
                    if b == 0:
                        continue  # no articles -> no measurement, no row
                    rows.append({
                        "corpus": corpus, "year": y, "frame": slug,
                        "frame_label": FRAME_LABELS[slug],
                        "outlet_clean": "" if group == ALL_OUTLETS else group,
                        "level": "all_outlets" if group == ALL_OUTLETS else "by_outlet",
                        "pct": p, "n_present": a, "n_applicable": b,
                    })

        payload[corpus] = {
            "years": YEARS,
            "frames": frames,
            "frame_labels": [FRAME_LABELS[s] for s in frames],
            "caveat": [s in CAVEAT_FRAMES for s in frames],
            "colors": [FRAME_COLORS[s] for s in frames],
            "outlets": [ALL_OUTLETS] + outlets,
            "n_total": int(len(sub)),
            "series": series,
        }

    return pd.DataFrame(rows), payload


def report_thin(payload: dict) -> None:
    for corpus, p in payload.items():
        print(f"\n{'=' * 78}\n  THIN CELLS — {corpus} (n_applicable < {THIN})\n{'=' * 78}")
        base = p["frames"][0]  # pass1 frame: same denominator as every article
        allrow = p["series"][ALL_OUTLETS][base]["na"]
        print(f"  all-outlets denominators by year: "
              f"{dict(zip(p['years'], allrow))}")
        thin_all = [(y, n) for y, n in zip(p["years"], allrow) if n < THIN]
        print(f"  all-outlets thin years: {thin_all if thin_all else 'none'}")
        print(f"\n  {'outlet':<12}" + "".join(f"{y:>6}" for y in p["years"]))
        n_thin = n_zero = 0
        for o in p["outlets"][1:]:
            na = p["series"][o][base]["na"]
            cells = "".join(
                (f"{n:>5}*" if 0 < n < THIN else (f"{'-':>6}" if n == 0 else f"{n:>6}"))
                for n in na)
            n_thin += sum(1 for n in na if 0 < n < THIN)
            n_zero += sum(1 for n in na if n == 0)
            print(f"  {o:<12}{cells}")
        print(f"\n  * = thin (1-9 articles): {n_thin} cells   "
              f"- = zero articles (no line point): {n_zero} cells")


def build_html(payload: dict) -> str:
    return _TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ2 — Frame prevalence over time</title>
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
  label { font-size:13.5px; color:var(--muted); display:flex; align-items:center; gap:8px; }
  select { font:inherit; font-size:13.5px; padding:6px 9px; border-radius:7px;
           border:1px solid var(--line); background:var(--bg); color:var(--fg); }
  .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px 6px; }
  .panel .meta { color:var(--muted); font-size:13px; margin:0 0 6px; }
  .chart-scroll { overflow-x:auto; }
  .chart { min-width:980px; }
  .caption { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
             margin-top:10px; padding:11px 2px 8px; }
  .caption p { margin:0 0 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>RQ2 &middot; Frame prevalence over time</h1>
  <p class="sub">Share of articles per year where each frame is present.
     Click a legend entry to hide or show that frame.</p>

  <div class="controls">
    <div class="seg" role="tablist" id="corpusToggle">
      <button role="tab" data-corpus="climate" aria-selected="true">Climate</button>
      <button role="tab" data-corpus="migration" aria-selected="false">Migration</button>
    </div>
    <label>Outlet <select id="outlet"></select></label>
  </div>

  <div class="panel">
    <p class="meta" id="meta"></p>
    <div class="chart-scroll"><div id="chart" class="chart"></div></div>
    <div class="caption">
      <p>Marker style shows how much data a year rests on: <b>solid</b> = 10 or more
         articles; <b>small hollow</b> = 3&ndash;9 articles; <b>grey open diamond</b>
         = 1&ndash;2 articles, where only two or three percentages are arithmetically
         possible and the point carries no trend information. The line is drawn
         through all of them. Years with no articles at all are left as a gap rather
         than plotted as 0%.</p>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
let corpus = 'climate';

function isDark() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function fillOutlets() {
  const sel = document.getElementById('outlet');
  const prev = sel.value;
  sel.innerHTML = '';
  DATA[corpus].outlets.forEach(o => {
    const opt = document.createElement('option');
    opt.value = o; opt.textContent = o; sel.appendChild(opt);
  });
  sel.value = DATA[corpus].outlets.includes(prev) ? prev : DATA[corpus].outlets[0];
}

function render() {
  const d = DATA[corpus];
  const outlet = document.getElementById('outlet').value;
  const dark = isDark();
  const fg = dark ? '#e8eaed' : '#16181d';
  const muted = dark ? '#9aa2ad' : '#5c6370';
  const grid = dark ? '#2a2e35' : '#e3e6ea';
  const paper = dark ? '#14161a' : '#ffffff';
  const GREY = dark ? '#8b9098' : '#999999';

  // Three marker tiers by denominator:
  //   n == 0      no point at all (y is null) — styling below is never drawn
  //   n <= 2      grey open diamond: 1-2 articles can only produce 2-3 distinct
  //               percentages, so the point carries no trend information
  //   3 <= n < 9  small hollow circle, outlined in the trace colour
  //   n >= 10     full solid circle
  // Plotly draws a legend swatch from index 0 of any per-point marker array,
  // so the 2014 point's style became every frame's legend icon — on NRC that
  // made all 13 identical grey diamonds. Fix: each frame is a legendgroup with
  // an empty proxy trace that owns the legend (fixed solid circle in the frame
  // colour) and the real data trace carrying showlegend:false. groupclick
  // defaults to 'togglegroup', so clicking the proxy toggles the real trace.
  const legendProxies = d.frames.map((slug, i) => ({
    type: 'scatter', mode: 'lines+markers',
    name: d.frame_labels[i],
    legendgroup: slug,
    x: [null], y: [null],
    line: { color: d.colors[i], width: 2 },
    marker: { size: 10, symbol: 'circle', color: d.colors[i],
              line: { color: d.colors[i], width: 2 } },
    showlegend: true, hoverinfo: 'skip'
  }));

  const traces = d.frames.map((slug, i) => {
    const s = d.series[outlet][slug];
    const colour = d.colors[i];
    const sizes   = s.na.map(n => n <= 2 ? 8 : (n < 10 ? 5 : 11));
    const symbols = s.na.map(n => n <= 2 ? 'diamond-open' : 'circle');
    const fills   = s.na.map(n => n <= 2 ? GREY : (n < 10 ? paper : colour));
    const edges   = s.na.map(n => n <= 2 ? GREY : colour);
    const widths  = s.na.map(n => n <= 2 ? 1.8 : (n < 10 ? 1.5 : 2));
    return {
      type: 'scatter', mode: 'lines+markers',
      name: d.frame_labels[i],
      legendgroup: slug, showlegend: false,
      x: d.years, y: s.pct,
      customdata: s.na.map((n, j) => [
        s.np[j], n,
        (n > 0 && n <= 2) ? '  \\u00b7  n=' + n + ', not meaningful' : ''
      ]),
      line: { color: colour, width: 2 },
      marker: { size: sizes, symbol: symbols, color: fills,
                line: { color: edges, width: widths } },
      connectgaps: false,
      // no <extra></extra>: in x-unified the trace name is the row label
      // (with its colour swatch), which is exactly what identifies each frame
      hovertemplate:
        '%{y:.1f}%  \\u00b7  %{customdata[0]}/%{customdata[1]} articles' +
        '%{customdata[2]}'
    };
  });

  Plotly.react('chart', legendProxies.concat(traces), {
    height: 520,
    // right margin must hold the outside legend, or it gets clipped
    margin: { l: 60, r: 168, t: 10, b: 48 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: fg, size: 12 },
    xaxis: { title: { text: 'Year', font: { size: 12.5, color: muted } },
             dtick: 1, gridcolor: grid, zeroline: false, tickfont: { color: fg } },
    yaxis: { title: { text: '% of articles with frame present',
                      font: { size: 12.5, color: muted } },
             // pad past 0/100 so markers sitting exactly on those values
             // (every n=1 point does) are not sliced in half by the axis
             range: [-4, 104], tick0: 0, dtick: 20,
             ticksuffix: '%', gridcolor: grid, zeroline: false,
             tickfont: { color: fg } },
    // unified: every frame at a shared year stacks into one box, so points
    // sitting on identical percentages stay reachable
    hovermode: 'x unified',
    hoverlabel: { bgcolor: dark ? '#20242b' : '#ffffff',
                  bordercolor: grid,
                  font: { size: 11.5, color: fg } },
    legend: { orientation: 'v', x: 1.01, xanchor: 'left', y: 1,
              font: { size: 11.5, color: fg }, itemclick: 'toggle',
              itemdoubleclick: 'toggleothers' }
  }, { displayModeBar: false, responsive: true });

  const base = d.frames[0];
  const n = d.series[outlet][base].na.reduce((a, b) => a + b, 0);
  document.getElementById('meta').textContent =
    n.toLocaleString() + ' articles \\u00b7 ' + outlet + ' \\u00b7 ' +
    d.frames.length + ' frames \\u00b7 ' + corpus + ' corpus';
}

document.getElementById('corpusToggle').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  corpus = b.dataset.corpus;
  [...e.currentTarget.querySelectorAll('button')].forEach(x =>
    x.setAttribute('aria-selected', String(x === b)));
  fillOutlets(); render();
});
document.getElementById('outlet').addEventListener('change', render);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);

fillOutlets();
render();
</script>
</body>
</html>
"""


def main() -> None:
    print(f"{'=' * 78}\n  RQ2 — aggregate\n{'=' * 78}")
    long, payload = aggregate()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(long):,} rows)")
    for lvl in ["all_outlets", "by_outlet"]:
        print(f"     {lvl:<12} {int((long['level'] == lvl).sum()):>5,} rows")

    report_thin(payload)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"\n  wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
