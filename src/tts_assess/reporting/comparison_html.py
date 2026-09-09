from __future__ import annotations

import html
from typing import Any


def render_comparison_html(data: dict[str, Any]) -> str:
    """Render the minimal black-and-white comparative report."""
    labels = data["labels"]
    subtitle = data.get("subtitle")
    sub_html = f'<p class="sub">{html.escape(subtitle)}</p>' if subtitle else ""
    question = data.get("question")
    question_html = (
        f'<p class="question">{html.escape(question)}</p>' if question else ""
    )
    metric_html = _metric_table(data["metric_table"], labels)
    health_html = _health_table(data["health_table"], labels)
    violations_html = _violations_section(data["violations"]) if data.get("violations") else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(data["title"])}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>{html.escape(data["title"])}</h1>
  {sub_html}
  {question_html}
  {metric_html}
  {health_html}
  {violations_html}
</main>
</body>
</html>
"""


def _violations_section(v: dict[str, Any]) -> str:
    counts = v.get("counts", [])
    if not counts:
        return (
            "<section><h2>Threshold Violations</h2>"
            "<p class='legend'>No configured threshold was violated.</p></section>"
        )
    crows = "".join(
        f"<tr><th class='rowhead'>{html.escape(c['metric'])}</th>"
        f"<td>{c['warn']}</td><td>{c['fail']}</td><td>{c['total']}</td></tr>"
        for c in counts
    )
    counts_tbl = (
        "<div class='scroll'><table>"
        "<tr><th class='rowhead'>metric</th><th>warn</th><th>fail</th><th>total</th></tr>"
        f"{crows}</table></div>"
    )

    ex = v.get("examples", {})
    has_audio = ex.get("has_audio")
    cols = ["metric", "voice", "value", "expected", "heard"] + (["audio"] if has_audio else [])
    header = "".join(f"<th class='rowhead'>{c}</th>" for c in cols)
    erows = []
    for r in ex.get("rows", []):
        cells = [
            f"<td class='rule'>{html.escape(r['metric'])}</td>",
            f"<td>{html.escape(str(r['voice']))}</td>",
            f"<td><span class='val'>{html.escape(str(r['value']))}</span></td>",
            f"<td class='txt'>{html.escape(r['expected'])}</td>",
            f"<td class='txt'>{html.escape(r['heard'])}</td>",
        ]
        if has_audio:
            cells.append(f"<td>{_audio_cell(r.get('audio'))}</td>")
        erows.append(f"<tr>{''.join(cells)}</tr>")
    examples_tbl = f"<div class='scroll'><table><tr>{header}</tr>{''.join(erows)}</table></div>"
    legend = (
        f"<p class='legend'>{html.escape(v.get('state', ''))} Compare <b>expected</b> vs "
        "<b>heard</b> (the normalized strings scored) to tell a real error from a normalization "
        "mismatch. Each sample appears once, under its most-violated metric.</p>"
    )
    return (
        "<section><h2>Threshold Violations</h2>"
        "<p class='legend'>Warn / fail counts per metric.</p>"
        f"{counts_tbl}{examples_tbl}{legend}</section>"
    )


def _audio_cell(src: str | None) -> str:
    if not src:
        return ""
    safe = html.escape(str(src), quote=True)
    return f"<audio class='row-audio' controls preload='none' src='{safe}'></audio>"


def _header_row(labels: list[str], lead: tuple[str, ...] = ("metric",)) -> str:
    head = "".join(f"<th class='rowhead'>{html.escape(h)}</th>" for h in lead)
    cells = "".join(f"<th>{html.escape(str(label))}</th>" for label in labels)
    return f"<tr>{head}{cells}</tr>"


def _metric_table(table: dict[str, Any], labels: list[str]) -> str:
    ci_label = table.get("ci_label", "95% CI")
    span = len(labels) + 1
    body = []
    for group in table["groups"]:
        body.append(
            f"<tr class='group'><th class='rowhead' colspan='{span}'>"
            f"{html.escape(group['title'])}</th></tr>"
        )
        for row in group["rows"]:
            arrow = {"lower": " ↓", "higher": " ↑"}.get(row.get("better"), "")
            name = f"{html.escape(row['metric'])}<span class='dir'>{arrow}</span>"
            cells = "".join(_metric_cell(c) for c in row["cells"])
            body.append(f"<tr><th class='rowhead'>{name}</th>{cells}</tr>")
    method = (
        "bootstrap over texts (clips sharing a text are resampled together)"
        if table.get("ci_method") == "cluster"
        else "bootstrap over clips"
    )
    legend = (
        "<p class='legend'>Each cell: mean with "
        f"{html.escape(ci_label)} below ({method}). "
        "<span class='chip best'></span> best &nbsp; "
        "<span class='chip worst'></span> worst &nbsp; "
        "<span class='star'>*</span> the paired per-text difference to the best run "
        "excludes zero (↓/↑ = better direction). "
        "<span class='cov'>n=…</span> marks a mean over fewer clips than the run has: "
        "unmeasured clips are excluded from the mean, so read it with its coverage. "
        "The <i>Others</i> group is shown without best/worst colouring."
        "</p>"
    )
    return (
        "<section><h2>Metric Comparison</h2>"
        f"<div class='scroll'><table>{_header_row(labels)}{''.join(body)}</table></div>"
        f"{legend}</section>"
    )


def _coverage(cell: dict[str, Any]) -> str:
    """'n=990/1000 · 10 failed' when a cell covers fewer clips than the run has."""
    n = cell.get("n", cell.get("measured"))
    total = cell.get("total")
    failed = cell.get("failed") or 0
    if n is None or total is None or (n == total and not failed):
        return ""
    text = f"n={n}/{total}"
    if failed:
        text += f" · {failed} failed"
    return f"<span class='cov'>{html.escape(text)}</span>"


def _metric_cell(cell: dict[str, Any]) -> str:
    if cell.get("na") or cell.get("mean") is None:
        return f"<td class='na'>—{_coverage(cell)}</td>"
    cls = "best" if cell.get("best") else ("worst" if cell.get("worst") else "")
    star = "<span class='star'>*</span>" if cell.get("sig") else ""
    ci = ""
    if cell.get("lo") is not None and cell.get("hi") is not None:
        ci = f"<span class='ci'>{_num(cell['lo'])}–{_num(cell['hi'])}</span>"
    return (
        f"<td class='{cls}'>"
        f"<span class='val'>{_num(cell['mean'])}{star}</span>{ci}{_coverage(cell)}</td>"
    )


def _health_table(table: dict[str, Any], labels: list[str]) -> str:
    good = round(table["good_rate"] * 100)
    warn = round(table["warn_rate"] * 100)
    body = []
    for row in table["rows"]:
        rule = html.escape(row.get("pass_rule", "—"))
        cells = "".join(_health_cell(c) for c in row["cells"])
        body.append(
            f"<tr><th class='rowhead'>{html.escape(row['metric'])}</th>"
            f"<td class='rule'>{rule}</td>{cells}</tr>"
        )
    if not body:
        return (
            "<section><h2>Model Health</h2>"
            "<p class='legend'>No threshold-backed metric was evaluated.</p></section>"
        )
    header = _header_row(labels, lead=("metric", "pass if"))
    legend = (
        "<p class='legend'>Two steps. "
        "<b>1 — per sample:</b> a clip <i>passes</i> a metric when its value meets that row's "
        "<b>pass if</b> rule (otherwise it warns/fails). "
        "<b>2 — per model:</b> each cell is the share of that model's clips that pass, graded "
        f"<span class='chip good'></span> good (≥{good}%) &nbsp; "
        f"<span class='chip warn'></span> warn (≥{warn}%) &nbsp; "
        f"<span class='chip fail'></span> fail (&lt;{warn}%). "
        "A clip whose measurement failed (decode, ASR, or a model error) counts as not "
        "passing; <span class='cov'>n=…</span> shows measured/total and failed counts."
        "</p>"
    )
    inner = (
        f"<div class='scroll'><table>{header}{''.join(body)}</table></div>{legend}"
    )
    return f"<section><h2>Model Health</h2>{inner}</section>"


def _health_cell(cell: dict[str, Any]) -> str:
    if cell.get("na") or cell.get("good_rate") is None:
        return "<td class='na'>—</td>"
    status = cell.get("status", "")
    # One decimal, trimmed — so the shown % never contradicts its band at a
    # boundary (e.g. 79.9% fail must not display as "80% fail").
    pct = f"{cell['good_rate'] * 100:.1f}".rstrip("0").rstrip(".")
    return f"<td class='{status}'><span class='val'>{pct}%</span>{_coverage(cell)}</td>"


def _num(value: float) -> str:
    """At most 3 decimals, trailing zeros trimmed."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


_CSS = """
:root{--rule:#cfcfcf;--ink:#111;--muted:#777;--best:#e6f2e6;--worst:#f6e7e7;--warn:#faf0d7}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 4rem}
h1{font-size:1.6rem;font-weight:600;margin:0 0 .15rem;letter-spacing:.01em}
.sub{color:var(--muted);margin:0 0 1.5rem;font-size:.9rem}
.question{margin:.5rem 0 1.5rem;font-size:.95rem}
section{margin:2.25rem 0}
h2{font-size:1.05rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;border-bottom:2px solid var(--ink);padding-bottom:.35rem;margin:0 0 1rem}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--rule);padding:.4rem .55rem;text-align:center}
thead,th{}
tr>th:first-child,td:first-child{text-align:left}
th{font-weight:600;background:#fafafa}
th.rowhead{font-weight:500;white-space:nowrap;position:sticky;left:0;background:#fafafa;z-index:1}
.dir{color:var(--muted);font-weight:400}
tr.group th{text-align:left;background:#efefef;font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:.72rem;color:#333}
td{vertical-align:middle}
td .val{display:block;font-size:.95rem;font-weight:600;font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
td .ci{display:block;font-size:.62rem;color:var(--muted);margin-top:.1rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cov{display:block;font-size:.62rem;color:#a33;margin-top:.1rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.legend .cov{display:inline}
.star{color:#000;font-weight:700;margin-left:.1rem}
td.na{color:#bbb}
td.rule{text-align:left;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;white-space:nowrap}
td.txt{text-align:left;font-size:.82rem;max-width:320px;white-space:normal;color:var(--ink)}
.row-audio{height:1.9rem;max-width:200px}
.legend b{color:var(--ink)}
td.best{background:var(--best)}
td.worst{background:var(--worst)}
td.good{background:var(--best)}
td.warn{background:var(--warn)}
td.fail{background:var(--worst)}
.legend{color:var(--muted);font-size:.75rem;margin:.6rem 0 0}
.chip{display:inline-block;width:.7rem;height:.7rem;border:1px solid var(--rule);vertical-align:middle;margin-right:.15rem}
.chip.best,.chip.good{background:var(--best)}
.chip.worst,.chip.fail{background:var(--worst)}
.chip.warn{background:var(--warn)}
"""
