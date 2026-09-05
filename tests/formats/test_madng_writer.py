"""MAD-NG output (Phase 5.1): a writer-only format rendered by xtrack's ``mad_writer``.

The IR goes through :func:`lattix.formats.xtrack.to_line` (so every xtrack ledger row applies) and
xtrack writes the Lua ``sequence``; each element gets an extra ``EQUIVALENT:VIA_XTRACK`` row.  Knobs
come out as MAD-NG deferred expressions (``k2 =\\ (k2bi4bsw1l11)``)."""
from __future__ import annotations

from pathlib import Path

import pytest

xt = pytest.importorskip("xtrack")

from lattix import read, write  # noqa: E402
from lattix.fidelity import TranslationError  # noqa: E402
from lattix.formats.base import FORMATS  # noqa: E402
from lattix.ir.elements import Drift, Foil, Quadrupole  # noqa: E402
from lattix.ir.lattice import Lattice  # noqa: E402
from lattix.ir.reference import ReferenceParticle, species  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public"


def test_madng_is_registered_as_a_writer_only_format():
    spec = FORMATS["madng"]
    assert spec.reader_attr is None
    assert spec.writer_attr
    from lattix.formats.madng import Writer

    assert Writer.format == "madng"


def test_fodo_renders_a_lua_sequence_with_the_via_xtrack_ledger(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.madng"
    rep = write(lat, out, "madng")
    text = out.read_text()
    assert text.startswith("-- lattix ")
    assert "-- lattix: energy_mode=delta" in text
    assert "quadrupole 'qf'" in text and "sbend 'b1'" in text and "drift" in text
    assert "kfocus = 0.6" in text                        # the deck's variables are MAD-NG variables
    assert rep.codes()["VIA_XTRACK"] == 9                # one row per placed element
    assert rep.target_format == "madng"
    assert all(e.cls in ("EXACT", "EQUIVALENT") for e in rep.entries)


def test_psb_knobs_become_deferred_expressions(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "xtrack" / "psb.seq", "madx", species="proton", kinetic_energy_eV=160e6)
    out = tmp_path / "psb.madng"
    write(lat, out, "madng")
    text = out.read_text()
    assert text.count("=\\ (") >= 100                    # MAD-NG deferred expressions for the knobs
    assert "k2 =\\ (k2bi4bsw1l11)" in text


def test_strict_mode_raises_on_a_lossy_element(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=8e8)
    lat = Lattice.from_sequence("s", [Drift(name="d", length=0.5), Foil(name="f"),
                                      Quadrupole(name="q", length=0.3)], ref)
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "s.madng", "madng", strict=True)
    rep = write(lat, tmp_path / "s.madng", "madng")
    assert "FOIL_TO_MARKER" in rep.codes()
    assert "marker 'f'" in (tmp_path / "s.madng").read_text()


def test_cli_convert_to_madng(tmp_path, capsys):
    pytest.importorskip("cpymad")
    from lattix.cli import main

    out = tmp_path / "fodo.madng"
    rc = main(["convert", str(DATA / "helix" / "fodo.madx"), str(out), "--to", "madng"])
    assert rc == 0
    assert "sbend 'b1'" in out.read_text()
