"""The field-map degradation ladder in every writer (PLAN §4.3).

A field map has no equivalent in MAD-X, MAD8, elegant, Bmad or PALS, so a map that lattix
integrated is written as the element that reproduces what the map does to the reference:

===============================  ===============================  ==========================
map                              target element                   code
===============================  ===============================  ==========================
RF (``kind = "rf"``)             cavity of the map's own length   ``FM_TO_CAVITY``
static solenoid                  hard edge ``L_eff``/``B_eff``    ``FM_SOL_HARDEDGE``
static quadrupole (digit 9)      hard edge preserving ∫G, ∫G²     ``FM_QUAD_HARDEDGE``
nothing known                    drift                            ``FM_TO_DRIFT`` (LOSSY)
===============================  ===============================  ==========================

Every case is checked in both regimes (permissive records, strict raises on the LOSSY one),
for the emitted numbers, and for the two invariants that must never move: Σlength (I-1) and
the reference energy the target engine will compute (I-6).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.ir.elements import RFP, FieldMap, Marker
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate

WRITERS = ("madx", "mad8", "elegant", "bmad", "pals")
SUFFIX = {"madx": ".madx", "mad8": ".flat", "elegant": ".lte", "bmad": ".bmad", "pals": ".pals.yaml"}

L_MAP = 0.3
DE = 1.25e6
V_C = DE / math.cos(math.radians(-30.0))
INT_B, INT_B2 = -0.24, 0.36           # → L_eff = 0.16 m, B_eff = −1.5 T
INT_G, INT_G2 = 1.2, 7.2              # → L_eff = 0.2 m, G_eff = 6 T/m


def _ref(ke: float = 1.0e8, sp: str = "proton") -> ReferenceParticle:
    return ReferenceParticle(species=species(sp), kinetic_energy_eV=ke, rf_frequency_Hz=162.5e6)


def rf_map(name: str = "fm") -> FieldMap:
    """A field map the reader integrated: an RF cavity of 1.25 MeV at φs = −30°."""
    return FieldMap(
        name=name, length=L_MAP, geom=7700, files=["QWR"],
        rf=RFP(frequency_Hz=162.5e6, phase_rad=math.radians(-30.0), phase_is_sync=True,
               voltage_V=1.5e6, ttf=0.72, dE_ref_eV=DE),
        meta={"map_summary": {"kind": "rf", "length_m": L_MAP, "frequency_Hz": 162.5e6,
                              "v_c_V": V_C, "v_eff_V": 1.5e6, "ttf": 0.72,
                              "phase_sync_rad": math.radians(-30.0), "dE_ref_eV": DE,
                              "int_Ez_V": 2.0e6, "int_Bz2_T2m": 0.0}})


def solenoid_map(name: str = "fms") -> FieldMap:
    return FieldMap(
        name=name, length=L_MAP, geom=10, files=["SOL"],
        rf=RFP(dE_ref_eV=0.0),
        meta={"map_summary": {"kind": "solenoid", "length_m": L_MAP, "int_Bz_Tm": INT_B,
                              "int_Bz2_T2m": INT_B2, "L_eff_m": INT_B**2 / INT_B2,
                              "B_eff_T": INT_B2 / INT_B, "dE_ref_eV": 0.0}})


def quad_map(name: str = "fmq") -> FieldMap:
    return FieldMap(
        name=name, length=L_MAP, geom=90, files=["Q"],
        rf=RFP(dE_ref_eV=0.0),
        meta={"map_summary": {"kind": "quad", "length_m": L_MAP, "int_Gz_Tm_per_m": INT_G,
                              "int_Gz2_T2m_per_m2": INT_G2, "dE_ref_eV": 0.0}})


def blind_map(name: str = "fmb") -> FieldMap:
    """Files were missing, so nothing about the map's physics is known."""
    return FieldMap(name=name, length=L_MAP, geom=7700, files=["gone"])


def _lattice(el) -> Lattice:
    return Lattice.from_sequence("fm_line", [Marker(name="m0"), el, Marker(name="m1")], _ref())


def _write(fmt: str, lat: Lattice, tmp_path: Path, **kw):
    import importlib

    mod = importlib.import_module(f"lattix.formats.{fmt}")
    path = tmp_path / f"out{SUFFIX[fmt]}"
    rep = mod.Writer().write(lat, path, **kw)
    return path, rep


def _read_back(fmt: str, path: Path) -> Lattice:
    """Read the written deck back with the source beam (an elegant ``.lte`` carries none)."""
    import importlib

    kw = {"species": "proton", "kinetic_energy_eV": 1.0e8, "frequency_Hz": 162.5e6} \
        if fmt == "elegant" else {}
    return importlib.import_module(f"lattix.formats.{fmt}").Reader().read(path, **kw)[0]


def _codes(rep) -> dict:
    return rep.codes()


# ------------------------------------------------------------------ the ladder
@pytest.mark.parametrize("fmt", WRITERS)
def test_rf_map_becomes_a_cavity(fmt, tmp_path):
    lat = _lattice(rf_map())
    path, rep = _write(fmt, lat, tmp_path)
    assert "FM_TO_CAVITY" in _codes(rep) and "FM_TO_DRIFT" not in _codes(rep)
    entry = next(e for e in rep.entries if e.code == "FM_TO_CAVITY")
    assert entry.cls.value == "EQUIVALENT" and entry.kind == "FieldMap"
    assert entry.details["dE_ref_eV"] == pytest.approx(DE)
    assert entry.details["voltage_V"] == pytest.approx(V_C)
    assert rep.ok                                  # a known gain is never a loss
    cavity = {"madx": "rfcavity", "mad8": "rfcavity", "elegant": "rfca",
              "bmad": "lcavity", "pals": "rfcavity"}[fmt]
    assert cavity in path.read_text().lower()


@pytest.mark.parametrize("fmt", WRITERS)
def test_static_solenoid_map_becomes_a_hard_edge(fmt, tmp_path):
    lat = _lattice(solenoid_map())
    path, rep = _write(fmt, lat, tmp_path)
    assert "FM_SOL_HARDEDGE" in _codes(rep)
    entry = next(e for e in rep.entries if e.code == "FM_SOL_HARDEDGE")
    assert entry.cls.value == "EQUIVALENT"
    assert entry.details["L_eff_m"] == pytest.approx(0.16)
    assert entry.details["strength"] == pytest.approx(-1.5)
    assert entry.details["strength"] * entry.details["L_eff_m"] == pytest.approx(INT_B)
    assert rep.ok


@pytest.mark.parametrize("fmt", WRITERS)
def test_static_quadrupole_map_becomes_a_hard_edge(fmt, tmp_path):
    lat = _lattice(quad_map())
    path, rep = _write(fmt, lat, tmp_path)
    assert "FM_QUAD_HARDEDGE" in _codes(rep)
    entry = next(e for e in rep.entries if e.code == "FM_QUAD_HARDEDGE")
    assert entry.cls.value == "EQUIVALENT"
    assert entry.details["L_eff_m"] == pytest.approx(0.2)
    assert entry.details["strength"] == pytest.approx(6.0)
    assert entry.details["strength"] * entry.details["L_eff_m"] == pytest.approx(INT_G)


@pytest.mark.parametrize("fmt", WRITERS)
def test_map_without_a_summary_is_still_a_lossy_drift(fmt, tmp_path):
    lat = _lattice(blind_map())
    _path, rep = _write(fmt, lat, tmp_path)
    assert "FM_TO_DRIFT" in _codes(rep)
    assert next(e for e in rep.entries if e.code == "FM_TO_DRIFT").cls.value == "LOSSY"
    with pytest.raises(TranslationError):
        _write(fmt, lat, tmp_path, strict=True)


@pytest.mark.parametrize("fmt", WRITERS)
@pytest.mark.parametrize("factory", [rf_map, solenoid_map, quad_map], ids=["rf", "sol", "quad"])
def test_degraded_map_is_never_fatal_in_strict_mode(fmt, factory, tmp_path):
    """An integrated map is EQUIVALENT, not LOSSY: strict translation must go through."""
    _write(fmt, _lattice(factory()), tmp_path, strict=True)


# ------------------------------------------------------------------ invariants
@pytest.mark.parametrize("fmt", ["madx", "mad8", "elegant", "bmad"])
@pytest.mark.parametrize("factory", [rf_map, solenoid_map, quad_map, blind_map],
                         ids=["rf", "sol", "quad", "blind"])
def test_length_survives_the_degradation(fmt, factory, tmp_path):
    """I-1: the hard edge is centred in the map and padded, so Σlength never moves."""
    lat = _lattice(factory())
    path, _rep = _write(fmt, lat, tmp_path)
    assert _read_back(fmt, path).total_length == pytest.approx(lat.total_length, abs=1e-12)


@pytest.mark.parametrize("fmt", ["madx", "mad8", "elegant", "bmad"])
def test_hard_edge_is_centred_in_the_map(fmt, tmp_path):
    lat = _lattice(solenoid_map())
    path, _rep = _write(fmt, lat, tmp_path)
    back = _read_back(fmt, path)
    placed = [p for p in propagate(back) if p.element.kind == "Solenoid"]
    assert len(placed) == 1
    sol = placed[0]
    assert sol.element.length == pytest.approx(0.16)
    assert sol.element.solenoid.Bsol_T == pytest.approx(-1.5, rel=1e-9)
    assert (sol.s_in + sol.s_out) / 2 == pytest.approx(L_MAP / 2, abs=1e-12)   # map centre


@pytest.mark.parametrize("fmt", ["mad8", "elegant", "bmad"])
def test_quad_hard_edge_keeps_the_integrated_gradient(fmt, tmp_path):
    lat = _lattice(quad_map())
    path, _rep = _write(fmt, lat, tmp_path)
    back = _read_back(fmt, path)
    quads = [p.element for p in propagate(back) if p.element.kind == "Quadrupole"]
    assert len(quads) == 1
    assert quads[0].length * quads[0].gradient == pytest.approx(INT_G, rel=1e-9)


@pytest.mark.parametrize("fmt", ["elegant", "bmad"])
def test_p0_following_targets_reproduce_the_maps_reference_gain(fmt, tmp_path):
    """I-6 on the engines that follow p0: reading the written deck back gives the map's dE."""
    lat = _lattice(rf_map())
    path, _rep = _write(fmt, lat, tmp_path)
    back = _read_back(fmt, path)
    got = propagate(back)[-1].ref_out.kinetic_energy_eV - back.reference.kinetic_energy_eV
    assert got == pytest.approx(DE, rel=1e-9)


def test_madx_cavity_lag_and_volt(tmp_path):
    from lattix.ir.rf import madx_lag

    path, _rep = _write("madx", _lattice(rf_map()), tmp_path)
    line = next(ln for ln in path.read_text().splitlines() if ln.startswith("fm:"))
    assert "rfcavity" in line and "l=0.3" in line
    assert f"volt={V_C * 1e-6:.15g}" in line
    assert f"lag={madx_lag(math.radians(-30.0)):.15g}" in line
    assert "freq=162.5" in line


def test_elegant_cavity_changes_p0(tmp_path):
    path, _rep = _write("elegant", _lattice(rf_map()), tmp_path)
    line = next(ln for ln in path.read_text().splitlines() if ln.upper().startswith("FM:"))
    assert "RFCA" in line.upper() and "CHANGE_P0=1" in line.upper()


def test_bmad_cavity_is_a_thick_lcavity(tmp_path):
    path, _rep = _write("bmad", _lattice(rf_map()), tmp_path)
    line = next(ln for ln in path.read_text().splitlines() if ln.startswith("fm:"))
    assert "lcavity" in line and "l = 0.3" in line
    assert f"voltage = {V_C:.15g}" in line


def test_pals_hard_edge_is_a_union_of_the_map_length(tmp_path):
    import yaml

    path, _rep = _write("pals", _lattice(solenoid_map()), tmp_path)
    doc = yaml.safe_load(path.read_text())
    node = next(v for entry in doc["PALS"]["facility"] if isinstance(entry, dict)
                for k, v in entry.items() if isinstance(v, dict) and v.get("kind") == "UnionEle")
    assert node["length"] == pytest.approx(L_MAP)
    child = next(iter(node["elements"].values()))
    assert child["kind"] == "Solenoid" and child["length"] == pytest.approx(0.16)
    assert child["SolenoidP"]["Bsol"] == pytest.approx(-1.5)


def test_pals_rf_map_carries_dE_ref(tmp_path):
    import yaml

    path, _rep = _write("pals", _lattice(rf_map()), tmp_path)
    doc = yaml.safe_load(path.read_text())
    node = next(v for entry in doc["PALS"]["facility"] if isinstance(entry, dict)
                for k, v in entry.items() if isinstance(v, dict) and v.get("kind") == "RFCavity")
    assert node["RFP"]["dE_ref"] == pytest.approx(DE)
    assert node["RFP"]["voltage"] == pytest.approx(V_C)
    assert node["length"] == pytest.approx(L_MAP)


def test_line_mode_keeps_the_padding_as_a_sub_line(tmp_path):
    """MAD-X ``mode='line'`` has no implicit drifts, so the padded hard edge is a sub-line."""
    from lattix.formats.madx import Reader

    lat = _lattice(solenoid_map())
    path, _rep = _write("madx", lat, tmp_path, mode="line")
    text = path.read_text()
    assert "fms_fm: line=(" in text
    back, _ = Reader().read(path)
    assert back.total_length == pytest.approx(lat.total_length, abs=1e-12)


# ------------------------------------------------------------------ clusters and options
@pytest.mark.parametrize("fmt", WRITERS)
def test_superposed_field_maps_take_the_ladder_too(fmt, tmp_path):
    """A SUPERPOSE_MAP cluster's children are field maps: they must not fall back to drifts."""
    from lattix.ir.elements import Superposition
    from lattix.ir.lattice import Lattice, Line, LineItem

    cav, sol = rf_map("cfm"), solenoid_map("sfm")
    cluster = Superposition(name="sup", length=0.6, children=[(0.0, "cfm"), (0.3, "sfm")],
                            rf=RFP(dE_ref_eV=DE))
    lat = Lattice(name="c", reference=_ref(),
                  elements={"cfm": cav, "sfm": sol, "sup": cluster, "m0": Marker(name="m0")},
                  lines={"c": Line(name="c", items=[LineItem(ref="m0"), LineItem(ref="sup")])},
                  use="c")
    _path, rep = _write(fmt, lat, tmp_path)
    assert "FM_TO_CAVITY" in _codes(rep) and "FM_SOL_HARDEDGE" in _codes(rep)
    assert "FM_TO_DRIFT" not in _codes(rep)


def test_thin_cavity_form_is_a_zero_length_traveling_wave_between_two_drifts():
    """``replacement_for(thin_cavity=True)`` is the Bmad-safe thin form (docs/oracles.md: a
    zero-length standing-wave lcavity is a fatal Bmad error)."""
    from lattix.ir.fieldmap import replacement_for

    r = replacement_for(rf_map(), thin_cavity=True)
    assert r.code == "FM_TO_CAVITY" and r.cls == "EQUIVALENT"
    kinds = [p.kind for p in r.parts]
    assert kinds == ["Drift", "RFCavity", "Drift"]
    assert sum(p.length for p in r.parts) == pytest.approx(L_MAP)
    cav = r.main
    assert cav.length == 0.0 and cav.rf.cavity_type == "TRAVELING_WAVE"
    assert cav.rf.voltage_V == pytest.approx(V_C) and cav.rf.dE_ref_eV == pytest.approx(DE)
    assert r.parts[0].length == pytest.approx(L_MAP / 2)


def test_rf_map_with_a_static_magnet_channel_reports_the_dropped_field():
    """A map that is both a cavity and a magnet cannot be one target element."""
    from lattix.ir.fieldmap import replacement_for

    fm = rf_map()
    fm.meta["map_summary"] = dict(fm.meta["map_summary"], int_Bz_Tm=-0.2, int_Bz2_T2m=0.3)
    r = replacement_for(fm)
    assert r.code == "FM_TO_CAVITY"
    assert [c for c, _code, _m in r.extra] == ["LOSSY"]
    assert r.extra[0][1] == "FM_STATIC_B_DROPPED"


def test_cavity_without_a_summary_but_with_dE_ref_still_becomes_a_cavity():
    """A lattice from the HELIX adapter carries dE_ref but no map_summary."""
    from lattix.ir.fieldmap import replacement_for

    fm = FieldMap(name="fm", length=0.4,
                  rf=RFP(phase_rad=math.radians(-30.0), dE_ref_eV=DE, frequency_Hz=325e6))
    r = replacement_for(fm)
    assert r.code == "FM_TO_CAVITY" and r.cls == "EQUIVALENT"
    assert r.main.rf.voltage_V == pytest.approx(DE / math.cos(math.radians(-30.0)))
