"""Assemble the four verified RQ panels into one tabbed dashboard.

This is assembly, not a rebuild: each panel's embedded JSON payload is lifted
byte-for-byte out of its already-verified HTML file and asserted identical in
the output. Only the surrounding code is rewritten, and only to namespace DOM
ids so the four panels can share one document.

The article table (rq_table.html) is deliberately NOT included and NOT linked.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DASH = ROOT / "dashboard"
OUT = DASH / "dashboard.html"

PANELS = [
    {"key": "rq1", "file": "rq1_heatmap.html",     "tab": "Topics × Frames",
     "rq": "RQ1", "ids": ["climate", "migration", "m-climate", "m-migration"],
     "minw": 880},
    {"key": "rq2", "file": "rq2_temporal.html",    "tab": "Over Time",
     "rq": "RQ2", "ids": ["chart", "corpusToggle", "meta", "outlet"],
     "minw": 980},
    {"key": "rq3", "file": "rq3_outlet.html",      "tab": "By Outlet",
     "rq": "RQ3", "ids": ["chart", "corpusToggle", "meta", "thinNote"],
     "minw": 1000},
    {"key": "rq4", "file": "rq4_crosscrisis.html", "tab": "Cross-Crisis",
     "rq": "RQ4", "ids": ["chart"], "minw": 960},
]


def split_script(script: str) -> tuple[str, str]:
    """Split an inline script into (DATA statement, remaining code).

    Brace-counted so the payload is isolated exactly; nothing downstream ever
    rewrites it, which is what keeps it byte-identical.
    """
    i = script.index("const DATA = ")
    j = script.index("{", i)
    depth, k, in_str, esc = 0, j, False, False
    while k < len(script):
        c = script[k]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    end = script.index(";", k) + 1
    return script[i:end], script[:i] + script[end:]


def payload_of(script: str) -> str:
    stmt, _ = split_script(script)
    return stmt[stmt.index("{"):stmt.rindex("}") + 1]


def namespace(code: str, p: dict) -> str:
    """Prefix every DOM id reference with the panel key."""
    k = p["key"]
    for _id in p["ids"]:
        code = code.replace(f"getElementById('{_id}')", f"getElementById('{k}-{_id}')")
    # rq1 builds ids dynamically and passes plot ids into render()
    if k == "rq1":
        code = code.replace("getElementById('m-' +", f"getElementById('{k}-m-' +")
        code = code.replace("render('climate', 'climate')", f"render('climate', '{k}-climate')")
        code = code.replace("render('migration', 'migration')", f"render('migration', '{k}-migration')")
    for call in ["Plotly.react('chart'", "Plotly.newPlot('chart'"]:
        code = code.replace(call, call.replace("'chart'", f"'{k}-chart'"))
    return code


def extract(p: dict) -> dict:
    html = (DASH / p["file"]).read_text()
    script = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    body = re.sub(r"<script>.*?</script>", "", body, flags=re.S)
    body = re.search(r'<div class="wrap">(.*)</div>', body, re.S).group(1)
    sub = re.search(r'<p class="sub">(.*?)</p>', body, re.S)
    body = re.sub(r"<h1>.*?</h1>", "", body, flags=re.S)
    body = re.sub(r'<p class="sub">.*?</p>', "", body, flags=re.S)
    for _id in p["ids"]:
        body = body.replace(f'id="{_id}"', f'id="{p["key"]}-{_id}"')

    data_stmt, code = split_script(script)
    return {
        "payload": payload_of(script),
        "data_stmt": data_stmt,
        "code": namespace(code, p),
        "body": body.strip(),
        "sub": " ".join(sub.group(1).split()) if sub else "",
    }


SHELL_CSS = """
  :root { --bg:#ffffff; --fg:#16181d; --muted:#5c6370; --line:#e3e6ea;
          --accent:#4269d0; --soft:#f6f7f9; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa2ad; --line:#2a2e35;
            --accent:#97bbf5; --soft:#1b1e24; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:0 0 56px; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:0 24px; }
  header { border-bottom:1px solid var(--line); padding:26px 0 0; margin-bottom:22px; }
  header h1 { font-size:22px; margin:0 0 3px; letter-spacing:-0.01em; }
  header .thesis { color:var(--muted); font-size:13.5px; margin:0 0 16px; }
  .tabs { display:flex; gap:2px; flex-wrap:wrap; }
  .tabs button { appearance:none; border:0; border-bottom:2px solid transparent;
                 background:transparent; color:var(--muted); cursor:pointer;
                 font:inherit; font-size:14px; padding:9px 15px 10px; }
  .tabs button:hover { color:var(--fg); }
  .tabs button[aria-selected="true"] { color:var(--fg); border-bottom-color:var(--accent);
                                       font-weight:600; }
  .tabs .rq { color:var(--muted); font-weight:400; font-size:12px; margin-right:6px; }
  .tabs button[aria-selected="true"] .rq { color:var(--accent); }
  .panelwrap { display:none; }
  .panelwrap.active { display:block; }
  .panelwrap > .sub { color:var(--muted); font-size:14px; margin:0 0 18px; }
  /* carried over from the individual panels, unchanged */
  .controls { display:flex; flex-wrap:wrap; gap:20px; align-items:center; margin-bottom:16px; }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .seg button { appearance:none; border:0; background:transparent; color:var(--fg);
                padding:7px 16px; font-size:14px; cursor:pointer; }
  .seg button[aria-selected="true"] { background:var(--accent); color:#fff; }
  label { font-size:13.5px; color:var(--muted); display:flex; align-items:center; gap:8px; }
  select { font:inherit; font-size:13.5px; padding:6px 9px; border-radius:7px;
           border:1px solid var(--line); background:var(--bg); color:var(--fg); }
  .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px 6px;
           margin-bottom:26px; }
  .panel h2 { font-size:16px; margin:2px 0 2px; }
  .panel .meta { color:var(--muted); font-size:13px; margin:0 0 8px; }
  .chart-scroll { overflow-x:auto; }
  .caption { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
             margin-top:10px; padding:11px 2px 8px; }
  .caption p { margin:0 0 8px; }
  .key { display:flex; flex-wrap:wrap; gap:16px; margin:6px 0 2px; font-size:12.5px;
         color:var(--muted); }
  .key span { display:flex; align-items:center; gap:6px; }
  .sw { width:22px; height:12px; border-radius:2px; display:inline-block; }
"""


def main() -> None:
    print(f"{'=' * 78}\n  ASSEMBLE COMBINED DASHBOARD\n{'=' * 78}")
    parts = []
    for p in PANELS:
        e = extract(p)
        p.update(e)
        src = (DASH / p["file"]).stat().st_size
        print(f"  {p['key']}: payload {len(e['payload']):>9,} chars | "
              f"code {len(e['code']):>6,} | body {len(e['body']):>5,} | "
              f"source {src/1024:.0f} KB")
        parts.append(p)

    css = SHELL_CSS + "\n" + "\n".join(
        f'  #{p["key"]}-root .chart {{ min-width:{p["minw"]}px; }}' for p in parts)

    tabs = "\n".join(
        f'      <button role="tab" data-panel="{p["key"]}" '
        f'aria-selected="{"true" if i == 0 else "false"}">'
        f'<span class="rq">{p["rq"]}</span>{p["tab"]}</button>'
        for i, p in enumerate(parts))

    bodies = "\n".join(
        f'    <section class="panelwrap{" active" if i == 0 else ""}" '
        f'id="{p["key"]}-root" data-panel="{p["key"]}">\n'
        f'      <p class="sub">{p["sub"]}</p>\n{p["body"]}\n    </section>'
        for i, p in enumerate(parts))

    scripts = "\n".join(
        f"// ---- {p['key'].upper()} ----------------------------------------\n"
        f"(function() {{\n{p['data_stmt']}\n{p['code']}\n}})();"
        for p in parts)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Framing climate and migration in the Dutch press</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{css}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Framing climate and migration in the Dutch press</h1>
    <p class="thesis">Frame prevalence across 4,047 annotated articles from six
       Dutch national outlets, 2014&ndash;2023.</p>
    <nav class="tabs" role="tablist">
{tabs}
    </nav>
  </div>
</header>

<div class="wrap">
{bodies}
</div>

<script>
{scripts}

// ---- tab shell -------------------------------------------------------
// Every panel renders at load. Plotly sizes to a hidden container as zero
// width, so each plot in a newly shown tab is resized on activation.
(function() {{
  const tabs = document.querySelector('.tabs');
  tabs.addEventListener('click', e => {{
    const b = e.target.closest('button[data-panel]');
    if (!b) return;
    const key = b.dataset.panel;
    tabs.querySelectorAll('button[data-panel]').forEach(x =>
      x.setAttribute('aria-selected', String(x === b)));
    document.querySelectorAll('.panelwrap').forEach(s =>
      s.classList.toggle('active', s.dataset.panel === key));
    document.querySelectorAll('#' + key + '-root .js-plotly-plot')
      .forEach(el => Plotly.Plots.resize(el));
  }});
}})();
</script>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")

    print(f"\n  payload byte-for-byte check (source -> combined):")
    combined = OUT.read_text()
    ok = True
    for p in parts:
        present = p["payload"] in combined
        ok &= present
        print(f"     {p['key']}: {len(p['payload']):>9,} chars present verbatim: {present}")
    if not ok:
        raise SystemExit("payload altered in transit")

    kb = OUT.stat().st_size / 1024
    src_total = sum((DASH / p["file"]).stat().st_size for p in parts) / 1024
    print(f"\n  wrote {OUT}  ({kb:.0f} KB)")
    print(f"  sum of the four source panels: {src_total:.0f} KB")
    assert "rq_table" not in combined, "combined file references the article table"
    assert "Article Explorer" not in combined
    print("  no reference to rq_table / Article Explorer: confirmed")


if __name__ == "__main__":
    main()
