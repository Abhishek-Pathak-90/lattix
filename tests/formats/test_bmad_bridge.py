"""The Bmad bridge: Astra, GPT, CSRtrack, Merlin++, SLICKTRACK and SAD written through Bmad's own
converters, SAD and SXF read through them.  Needs the ``bmad`` environment; the Python converters
need a Bmad source tree (``LATTIX_BMAD_UTIL_DIR``) and are skipped without one."""
from __future__ import annotations

from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import compare_profiles, profile, readable_formats
from lattix.fidelity import TranslationError
from lattix.formats import bmad_bridge as bb
from lattix.formats.base import FORMATS

pytestmark = pytest.mark.oracle_bmad
DATA = Path(__file__).resolve().parents[1] / "data" / "public"


def _fodo():
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    return lat


def _need(name: str):
    ok, why = bb.available(name)
    if not ok:
        pytest.skip(f"{name} bridge unavailable: {why}")


def test_registry_keeps_bridged_formats_out_of_the_battery():
    for f in ("astra", "gpt", "csrtrack", "merlin", "slicktrack", "sad", "sxf", "at"):
        assert FORMATS[f].options.get("bridge") and FORMATS[f].options.get("doc") == "bmad_bridge"
        assert f not in readable_formats()
    assert FORMATS["sad"].reader_attr and FORMATS["sad"].writer_attr
    assert FORMATS["astra"].reader_attr is None and FORMATS["sxf"].writer_attr is None


def test_unavailable_reader_explains_itself(monkeypatch):
    for var in bb._UTIL_VARS:
        monkeypatch.delenv(var, raising=False)
    ok, why = bb.available("at")
    assert not ok and ("LATTIX_BMAD_UTIL_DIR" in why or "bmad" in why.lower())


@pytest.mark.parametrize("target, token", [("astra", "&QUADRUPOLE"), ("gpt", "qf"), ("csrtrack", "_m1"),
                                           ("merlin", "qf"), ("slicktrack", "qf"), ("sad", "QUAD QF")])
def test_write_targets_and_bmad_consistency(tmp_path, target, token):
    """The file appears with the lattice's elements in it, every element carries VIA_BMAD, and the
    Bmad file written on the way re-reads to the IR."""
    _need(target)
    lat = _fodo()
    out = tmp_path / f"fodo.{target}"
    rep = write(lat, out, target, bmad_copy=tmp_path / "fodo.bmad", keep_workdir=tmp_path / "wd")
    assert out.is_file() and out.stat().st_size > 100
    assert token.lower() in out.read_text().lower()
    assert rep.codes()["VIA_BMAD"] == len(list(lat.flatten()))
    assert (tmp_path / "wd" / "convert.log").is_file()
    back, _ = read(tmp_path / "fodo.bmad", "bmad")
    diff = compare_profiles(profile(lat, {}), profile(back, {}))
    assert diff.ok, diff.problems[:5]
    if target == "gpt":
        # MEASURED: Bmad 20260828 has no GPT translation for bends and says so
        losses = [e.message for e in rep.entries if e.code == "BMAD_CONVERTER_LOSS"]
        assert any("BEND" in m.upper() for m in losses), losses
        with pytest.raises(TranslationError):
            write(lat, tmp_path / "strict.gpt", "gpt", strict=True)


def test_sad_round_trip_through_bmad(tmp_path):
    """Tao writes the SAD file, sad_to_bmad.py reads it back: SAD carries the momentum but no
    species, so the species is a read option; the elements and strengths return."""
    _need("sad")
    ok, why = bb.available("sad")
    if bb.util_dir() is None:
        pytest.skip(f"no Bmad util_programs for the SAD reader: {why}")
    lat = _fodo()
    write(lat, tmp_path / "fodo.sad", "sad")
    back, rep = read(tmp_path / "fodo.sad", "sad", species="proton", keep_workdir=tmp_path / "wd")
    assert rep.codes()["VIA_BMAD"] == len(list(back.flatten())) and back.meta["bmad_bridge"]["tool"] == "sad_to_bmad.py"
    assert back.reference.kinetic_energy_eV == pytest.approx(lat.reference.kinetic_energy_eV, rel=1e-8)
    diff = compare_profiles(profile(lat, {}), profile(back, {}))
    assert diff.ok, diff.problems[:5]


@pytest.mark.oracle_madx
def test_sxf_written_by_madx_reads_back(tmp_path):
    """MAD-X's ``sxfwrite`` makes the SXF file; sxf_to_bmad.py turns it into superimposed Bmad
    elements; SXF has no energy, so species and energy are read options."""
    _need("sxf")
    if bb.util_dir() is None:
        pytest.skip("no Bmad util_programs for the SXF reader")
    cpymad = pytest.importorskip("cpymad.madx")
    lat = _fodo()
    m = cpymad.Madx(stdout=False)
    m.call(str(DATA / "helix" / "fodo.madx"))
    m.use(sequence="fodo")
    m.input(f"sxfwrite, file='{tmp_path / 'fodo.sxf'}';")
    m.quit()
    back, rep = read(tmp_path / "fodo.sxf", "sxf", species="proton", kinetic_energy_eV=lat.reference.kinetic_energy_eV)
    assert rep.codes()["VIA_BMAD"] == len(list(back.flatten()))
    diff = compare_profiles(profile(lat, {}), profile(back, {}))
    assert diff.ok, diff.problems[:5]
