"""``lattix validate --html``: the per-boundary report renders from a PairComparison alone."""
from __future__ import annotations

from lattix.oracles.compare import BLOCKS, PairComparison
from lattix.report_html import render_validate_html, write_validate_html


def _pair(n: int = 5) -> PairComparison:
    per_boundary = [(0.5 * i, {k: 10.0 ** (-8 - i) for k in BLOCKS}) for i in range(n)]
    return PairComparison("madx", "bmad", n, 2.0, 2.0, 1e-8, 1e-9, 1e-12, 0.0, 1e-12,
                          {k: 1e-8 for k in BLOCKS}, [], per_boundary=per_boundary)


def test_one_polyline_per_block_and_the_names_appear():
    page = render_validate_html([_pair()], title="fodo.madx, fodo.bmad")
    assert page.count("<polyline") == len(BLOCKS)
    assert "madx vs bmad" in page and "fodo.madx, fodo.bmad" in page
    assert "<script" not in page and "http" not in page.replace("http://www.w3.org/2000/svg", "")


def test_pair_without_shared_boundaries_renders_a_note():
    pc = PairComparison("a", "b", 0, 1.0, 1.0, float("nan"), float("nan"), float("nan"), float("nan"),
                        None, {}, ["no shared s boundaries"])
    page = render_validate_html([pc])
    assert "no shared s boundaries" in page and "<polyline" not in page


def test_write_creates_the_file(tmp_path):
    out = write_validate_html(tmp_path / "r.html", [_pair(3)])
    assert out.exists() and out.read_text(encoding="utf-8").startswith("<!doctype html>")
