"""Every public deck through every writer: the page's per-element diff must find no unexplained difference
wherever the battery's round trip passes, and its round-trip verdict must be the battery's."""
from __future__ import annotations

from pathlib import Path

import pytest

from lattix.crossval import DECKS, PUBLIC, _read_options, _suffix, readable_formats, run_case
from lattix.formats import read, write
from lattix.ui.compare import compare_translation

_SKIP_DST = {"madx", "xtrack", "madng"}          # engines-only re-reads (cpymad/xtrack) live in the oracle job


def _translate(rel, src, dst, tmp_path, opts):
    tmp_path.mkdir(parents=True, exist_ok=True)
    lat, _ = read(PUBLIC / rel, src, **(opts or {}))
    out = tmp_path / ("ImpactT.in" if dst == "impactt" else "ImpactZ.in" if dst == "impactz" else f"x{_suffix(dst)}")
    rep = write(lat, out, dst)
    lat2, rep2 = read(out, dst, **_read_options(dst, lat.reference.species))
    return compare_translation(lat, rep, lat2, rep2, src_fmt=src, dst_fmt=dst)


@pytest.mark.crossval
@pytest.mark.parametrize("rel,src,opts", DECKS, ids=[Path(r).name for r, _, _ in DECKS])
@pytest.mark.parametrize("dst", [f for f in readable_formats() if f not in _SKIP_DST])
def test_no_unexplained_diffs_where_the_battery_passes(tmp_path, rel, src, opts, dst):
    if dst == src:
        pytest.skip("same format")
    res = run_case(PUBLIC / rel, src, dst, tmp_path / "case", read_options=opts)
    if res.error:
        pytest.skip(res.error)
    c = _translate(rel, src, dst, tmp_path / "ui", opts)
    assert (c["ir"]["ok"], c["ir"]["problems"]) == (res.ir_ok, res.ir_problems)
    if res.ir_ok:
        bad = [(c["alignment"]["elements"][d["i"]]["name"], {q: v["class"] for q, v in d["quantities"].items()
                                                              if v["class"] != "equal"}, d["energy"]["class"],
                d["position"]["class"]) for d in c["diffs"] if d["worst"] == "DIFF"]
        assert not bad, bad[:3]
