"""HTML report for ``lattix validate`` (PLAN §6 task 4.2): map differences vs s.

One section per engine pair with the summary row of :class:`~lattix.oracles.compare.PairComparison`
and an inline SVG of the per-boundary maximum absolute difference of every named block of the
cumulative map (log scale) against the position ``s``.  No external assets: the file is
self-contained and can be attached to a CI run.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

from lattix.oracles.compare import BLOCKS, PairComparison

_COLORS = {
    "T4x4": "#1f77b4", "disp": "#ff7f0e", "path": "#2ca02c", "R56": "#d62728",
    "R5x_z": "#9467bd", "E_row": "#8c564b", "z_col": "#e377c2",
}
FLOOR = 1e-16  # differences below this are drawn at the floor (exact agreement)

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem;color:#222;background:#fff}
h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}
pre{background:#f4f4f4;padding:.6rem;overflow-x:auto;font-size:.85rem}
.legend span{display:inline-block;margin-right:1rem;font-size:.85rem}
.legend i{display:inline-block;width:1.2em;height:.6em;margin-right:.3em;vertical-align:middle}
svg{max-width:100%;height:auto;border:1px solid #ddd}
"""


def _svg(pc: PairComparison, width: int = 900, height: int = 320) -> str:
    pts = pc.per_boundary
    if not pts:
        return "<p>no shared boundaries to plot</p>"
    ml, mr, mt, mb = 60, 20, 16, 40
    xs = [s for s, _ in pts]
    x0, x1 = min(xs), max(xs)
    if x1 <= x0:
        x1 = x0 + 1.0
    vals = [max(v, FLOOR) for _, d in pts for v in d.values()] or [FLOOR]
    lo, hi = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))
    if hi <= lo:
        hi = lo + 1

    def X(s: float) -> float:
        return ml + (s - x0) / (x1 - x0) * (width - ml - mr)

    def Y(v: float) -> float:
        return mt + (hi - math.log10(max(v, FLOOR))) / (hi - lo) * (height - mt - mb)

    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="map differences {html.escape(pc.a)} vs {html.escape(pc.b)}">']
    step = max(1, (hi - lo) // 8)
    for dec in range(lo, hi + 1, step):
        y = Y(10.0 ** dec)
        out.append(f'<line x1="{ml}" x2="{width - mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e0e0e0"/>'
                   f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11">1e{dec}</text>')
    for k in range(6):
        s = x0 + (x1 - x0) * k / 5
        x = X(s)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{height - mb}" y2="{height - mb + 5}" stroke="#333"/>'
                   f'<text x="{x:.1f}" y="{height - mb + 18}" text-anchor="middle" font-size="11">{s:.4g}</text>')
    out.append(f'<text x="{(ml + width - mr) / 2:.0f}" y="{height - 4}" text-anchor="middle" '
               f'font-size="12">s [m]</text>')
    out.append(f'<text x="14" y="{(mt + height - mb) / 2:.0f}" text-anchor="middle" font-size="12" '
               f'transform="rotate(-90 14 {(mt + height - mb) / 2:.0f})">max |ΔR_cum| per block</text>')
    for name, color in _COLORS.items():
        poly = " ".join(f"{X(s):.1f},{Y(d.get(name, FLOOR)):.1f}" for s, d in pts)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{poly}">'
                   f'<title>{name}</title></polyline>')
    out.append("</svg>")
    return "".join(out)


def render_validate_html(comparisons: list[PairComparison], title: str = "lattix validate") -> str:
    legend = "".join(f'<span><i style="background:{c}"></i>{k}</span>' for k, c in _COLORS.items()
                     if k in BLOCKS)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><style>{_CSS}</style>",
             f"<h1>lattix validate — {html.escape(title)}</h1>",
             "<p>Cumulative maps in the common basis (x, px/p0, y, py/p0, z, δ), rescaled to constant p0, "
             "compared at the boundaries both engines share.  Each curve is the largest absolute difference "
             "inside one block of the 6×6 map at that boundary; a curve at the floor means exact agreement.</p>",
             f'<p class="legend">{legend}</p>']
    for pc in comparisons:
        parts.append(f"<h2>{html.escape(pc.a)} vs {html.escape(pc.b)}</h2>")
        parts.append(f"<pre>{html.escape(pc.row())}</pre>")
        if pc.notes:
            parts.append("<p>" + "; ".join(html.escape(n) for n in pc.notes) + "</p>")
        parts.append(_svg(pc))
    return "\n".join(parts) + "\n"


def write_validate_html(path: str | Path, comparisons: list[PairComparison], title: str = "lattix validate") -> Path:
    path = Path(path)
    path.write_text(render_validate_html(comparisons, title), encoding="utf-8")
    return path
