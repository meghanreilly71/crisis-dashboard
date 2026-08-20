"""Filterable article table panel.

Reads annotated_tidy.csv (frames, meta, outlet, date) and article_table_data.csv
(title, body) and emits dashboard/rq_table.html. Both inputs are read-only.

Payload size is the constraint here: bodies plus indicator blobs for 4,047
articles. Two reductions, neither of which loses anything the panel can show:
  * indicators are stored ONLY for frames where present == "yes", since the
    filter is frame_present == "yes" and no other indicator is ever displayed;
  * presence is implied by key existence rather than stored as a separate field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TIDY = ROOT / "data" / "final" / "annotated_tidy.csv"
BODIES = ROOT / "data" / "final" / "article_table_data.csv"
SCHEMA = ROOT / "data" / "final" / "frame_schema.json"
OUT_HTML = ROOT / "dashboard" / "rq_table.html"

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


def build_payload() -> dict:
    tidy = pd.read_csv(TIDY, low_memory=False)
    bodies = pd.read_csv(BODIES, low_memory=False)
    schema = json.load(open(SCHEMA))

    df = tidy.merge(bodies[["article_key", "title", "body", "body_ambiguous"]],
                    on="article_key", how="left", validate="one_to_one")
    assert len(df) == len(tidy), "merge changed row count"
    assert df["title"].notna().all(), "missing titles after merge"
    df["meta"] = df["meta"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    payload = {}
    for corpus, passes in CORPUS_PASSES.items():
        sub = df[df["corpus"] == corpus]
        frames = [s for s, e in schema.items() if e["pass"] in passes]
        metas = sorted(sub["meta"].unique())

        rows = []
        for _, r in sub.iterrows():
            present = {}
            for slug in frames:
                if r[schema[slug]["present_col"]] == "yes":
                    ind = r[schema[slug]["indicators_col"]]
                    present[slug] = "" if pd.isna(ind) else str(ind)
            body = r["body"]
            rows.append({
                "k": r["article_key"],
                "o": r["outlet_clean"],
                "d": str(r["date"])[:10],
                "t": str(r["title"]),
                "b": None if pd.isna(body) else str(body),
                "m": r["meta"],
                "amb": bool(r["body_ambiguous"]) if not pd.isna(r["body_ambiguous"]) else False,
                "f": present,
            })

        payload[corpus] = {
            "metas": metas,
            "frames": [{"slug": s, "label": FRAME_LABELS[s],
                        "caveat": s in CAVEAT_FRAMES} for s in frames],
            "rows": rows,
        }
        print(f"  {corpus}: {len(rows):,} rows | {len(metas)} meta-topics | "
              f"{len(frames)} frames | "
              f"{sum(1 for x in rows if x['amb'])} body_ambiguous | "
              f"{sum(1 for x in rows if x['b'] is None)} null body")
    return payload


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Article table — filter by topic and frame</title>
<style>
  :root { --bg:#ffffff; --fg:#16181d; --muted:#5c6370; --line:#e3e6ea;
          --accent:#4269d0; --soft:#f6f7f9; --warn:#b45309; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa2ad; --line:#2a2e35;
            --accent:#97bbf5; --soft:#1b1e24; --warn:#fbbf24; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:28px 24px 64px; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; }
  h1 { font-size:23px; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 20px; }
  .controls { display:flex; flex-wrap:wrap; gap:18px; align-items:center; margin-bottom:14px; }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .seg button { appearance:none; border:0; background:transparent; color:var(--fg);
                padding:7px 16px; font-size:14px; cursor:pointer; }
  .seg button[aria-selected="true"] { background:var(--accent); color:#fff; }
  label { font-size:13.5px; color:var(--muted); display:flex; align-items:center; gap:8px; }
  select { font:inherit; font-size:13.5px; padding:6px 9px; border-radius:7px; max-width:340px;
           border:1px solid var(--line); background:var(--bg); color:var(--fg); }
  .count { font-size:13.5px; color:var(--muted); margin:0 0 12px; }
  .count b { color:var(--fg); }
  table { width:100%; border-collapse:collapse; font-size:13.5px; }
  th { text-align:left; font-weight:600; font-size:12.5px; color:var(--muted);
       border-bottom:1px solid var(--line); padding:8px 10px; white-space:nowrap; }
  td { border-bottom:1px solid var(--line); padding:10px; vertical-align:top; }
  tr:hover td { background:var(--soft); }
  .nowrap { white-space:nowrap; color:var(--muted); }
  .title { font-weight:600; }
  .body { color:var(--muted); margin-top:4px; }
  .toggle { appearance:none; border:0; background:transparent; color:var(--accent);
            font:inherit; font-size:12.5px; cursor:pointer; padding:2px 0; }
  .ind { margin-top:6px; font-size:12px; color:var(--muted); }
  .ind summary { cursor:pointer; color:var(--accent); }
  .ind ul { margin:6px 0 0; padding-left:18px; }
  .amb { display:inline-block; margin-top:5px; font-size:12px; color:var(--warn); }
  .empty { padding:26px 10px; color:var(--muted); font-size:14px; }
  .caption { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
             margin-top:18px; padding:11px 2px 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Article table</h1>
  <p class="sub">Articles where a chosen frame is present within a chosen
     meta-topic, with the indicator answers behind that frame's coding.</p>

  <div class="controls">
    <div class="seg" role="tablist" id="corpusToggle">
      <button role="tab" data-corpus="climate" aria-selected="true">Climate</button>
      <button role="tab" data-corpus="migration" aria-selected="false">Migration</button>
    </div>
    <label>Topic <select id="meta"></select></label>
    <label>Frame <select id="frame"></select></label>
  </div>

  <p class="count" id="count"></p>
  <table>
    <thead><tr>
      <th style="width:96px">Outlet</th>
      <th style="width:96px">Date</th>
      <th>Article</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>

<script>
const DATA = __DATA__;
const PREVIEW = 300;
let corpus = 'climate';

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fillSelects() {
  const d = DATA[corpus];
  const m = document.getElementById('meta'), f = document.getElementById('frame');
  const pm = m.value, pf = f.value;
  m.innerHTML = ''; f.innerHTML = '';
  d.metas.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v; m.appendChild(o);
  });
  d.frames.forEach(fr => {
    const o = document.createElement('option');
    o.value = fr.slug; o.textContent = fr.label;
    f.appendChild(o);
  });
  if (d.metas.includes(pm)) m.value = pm;
  if (d.frames.some(x => x.slug === pf)) f.value = pf;
}

function indicatorList(raw) {
  if (!raw) return '';
  let obj;
  try { obj = JSON.parse(raw); } catch (e) { return '<div class="ind">' + esc(raw) + '</div>'; }
  if (!obj || typeof obj !== 'object') return '';
  const items = Object.keys(obj).map(q =>
    '<li><b>' + esc(obj[q]) + '</b> &mdash; ' + esc(q) + '</li>').join('');
  return '<details class="ind"><summary>Indicators (' +
         Object.keys(obj).length + ')</summary><ul>' + items + '</ul></details>';
}

function render() {
  const d = DATA[corpus];
  const meta = document.getElementById('meta').value;
  const frame = document.getElementById('frame').value;
  const hits = d.rows.filter(r => r.m === meta && r.f[frame] !== undefined);

  const total = d.rows.filter(r => r.m === meta).length;
  const fr = d.frames.find(x => x.slug === frame);
  document.getElementById('count').innerHTML =
    '<b>' + hits.length + '</b> of ' + total + ' articles in &ldquo;' + esc(meta) +
    '&rdquo; have <b>' + esc(fr ? fr.label : frame) + '</b> present';

  const tb = document.getElementById('rows');
  if (!hits.length) {
    tb.innerHTML = '<tr><td colspan="3" class="empty">No articles match this ' +
                   'combination.</td></tr>';
    return;
  }
  tb.innerHTML = hits.map((r, i) => {
    const body = r.b || '';
    const short = body.length > PREVIEW ? body.slice(0, PREVIEW) + '\\u2026' : body;
    const needsToggle = body.length > PREVIEW;
    let bodyHtml;
    if (r.b === null) {
      bodyHtml = '<div class="amb">Body not shown &mdash; this article matched more ' +
                 'than one source record and could not be resolved unambiguously.</div>';
    } else {
      bodyHtml = '<div class="body" id="b' + i + '">' + esc(short) + '</div>' +
        (needsToggle
          ? '<button class="toggle" data-i="' + i + '">Show full</button>'
          : '');
    }
    return '<tr>' +
      '<td class="nowrap">' + esc(r.o) + '</td>' +
      '<td class="nowrap">' + esc(r.d) + '</td>' +
      '<td><div class="title">' + esc(r.t) + '</div>' + bodyHtml +
      indicatorList(r.f[frame]) + '</td>' +
    '</tr>';
  }).join('');

  tb.querySelectorAll('.toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = +btn.dataset.i, el = document.getElementById('b' + i);
      const full = hits[i].b || '';
      const open = btn.textContent === 'Show less';
      el.textContent = open ? full.slice(0, PREVIEW) + '\\u2026' : full;
      btn.textContent = open ? 'Show full' : 'Show less';
    });
  });
}

document.getElementById('corpusToggle').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  corpus = b.dataset.corpus;
  [...e.currentTarget.querySelectorAll('button')].forEach(x =>
    x.setAttribute('aria-selected', String(x === b)));
  fillSelects(); render();
});
document.getElementById('meta').addEventListener('change', render);
document.getElementById('frame').addEventListener('change', render);

fillSelects();
render();
</script>
</body>
</html>
"""


def main() -> None:
    print(f"{'=' * 78}\n  ARTICLE TABLE PANEL\n{'=' * 78}")
    payload = build_payload()
    blob = json.dumps(payload, ensure_ascii=False)
    print(f"\n  embedded JSON: {len(blob) / 1048576:.1f} MB")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(_TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    mb = OUT_HTML.stat().st_size / 1048576
    print(f"  wrote {OUT_HTML}  ({mb:.1f} MB)")
    if mb > 20:
        print(f"  *** OVER 20 MB — flag before combining inline ***")


if __name__ == "__main__":
    main()
