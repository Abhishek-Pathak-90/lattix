"""Ocelot writer: rules coverage, the module, the measured conventions, surrogate thin gaps, goldens."""
from __future__ import annotations

import ast
import math
import os
import re
from pathlib import Path

import pytest

from lattix import read, write
from lattix.formats.base import check_rules_coverage
from lattix.formats.ocelot import Writer
from lattix.formats.ocelot.writer import _ident
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    Sextupole,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ocelot"
KE = 2.1e6


def ref_at(ke=KE, sp="electron", f=162.5e6):
    return ReferenceParticle(species=species(sp), kinetic_energy_eV=ke, rf_frequency_Hz=f)


def all_kinds_lattice(sp="electron") -> Lattice:
    ref = ref_at(sp=sp)
    els = [
        Drift(name="d1", length=0.5),
        Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.2}), shift=BodyShiftP(x_offset=1e-3)),
        Sextupole(name="s1", length=0.1, multipole=MagneticMultipoleP(Bn={2: 3.0})),
        Octupole(name="o1", length=0.1, multipole=MagneticMultipoleP(Bn={3: 4.0})),
        Multipole(name="m1", multipole=MagneticMultipoleP(BnL={0: 0.01, 1: 0.02}, BsL={0: 0.02})),
        Bend(name="b1", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.45, hgap=0.03)),
        Solenoid(name="sol1", length=0.3, solenoid=SolenoidP(Bsol_T=0.5)),
        Drift(name="d2", length=0.3),
        RFCavity(name="c1", length=0.0, rf=RFP(voltage_V=8e4, phase_rad=math.radians(-85.0), frequency_Hz=162.5e6)),
        Drift(name="d3", length=0.3),
        RFCavity(name="c2", length=0.2, rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0), frequency_Hz=162.5e6,
                                               cavity_type="TRAVELING_WAVE")),
        Kicker(name="k1", length=0.1, hkick=1e-3, vkick=-2e-3),
        Collimator(name="col1", length=0.05, aperture=ApertureP.rect(0.01, 0.02)),
        Marker(name="mk1"),
        Instrument(name="bpm1", family="BPM"),
        Instrument(name="scr1", family="SCREEN"),
        Foil(name="f1", material="C", thickness_kg_per_m2=1e-3),
        Taylor(name="t1", length=0.1, matrix=[[1.0 if i == j else (0.1 if (i, j) == (0, 1) else 0.0) for j in range(6)]
                                              for i in range(6)]),
        Patch(name="p1", x_offset=1e-3),
        ReferenceChange(name="rc1", dE_ref_eV=1e3),
        Freq(name="fq1", frequency_Hz=162.5e6),
        Directive(name="dir1", format="tracewin", card="LATTICE", args=["4", "0"], role="period_start"),
        Drift(name="d4", length=0.2, aperture=ApertureP.circle(0.015)),
    ]
    return Lattice.from_sequence("all_kinds", els, ref)


def _module(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "lat.ocelot.py"
    rep = write(lat, out, "ocelot", **opts)
    return parse_module(out.read_text()), rep, out


def parse_module(text: str) -> dict[str, tuple[str, dict]]:
    """``{var: (constructor, kwargs)}`` of the element assignments (literal values only)."""
    out = {}
    for st in ast.parse(text).body:
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) and isinstance(st.targets[0], ast.Name):
            ctor = getattr(st.value.func, "id", "")
            if ctor in ("MagneticLattice", "Twiss"):
                continue
            kw = {k.arg: ast.literal_eval(k.value) for k in st.value.keywords}
            out[st.targets[0].id] = (ctor, kw)
    return out


_VERSION_RE = re.compile(r"\blattix \d[\w.+!-]*")      # the writer's banner: "lattix 0.2.0", "lattix 0.2.0rc1"


def _norm(text: str) -> str:
    return _VERSION_RE.sub("lattix <version>", text)


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = path.read_text()
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(_norm(text))
    assert _norm(text) == _norm(golden.read_text()), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_all_kinds_module_types_and_ledger(tmp_path):
    lat = all_kinds_lattice()
    mod, rep, out = _module(lat, tmp_path)
    compile(out.read_text(), str(out), "exec")                     # a valid Python module
    types = {var: ctor for var, (ctor, _) in mod.items()}
    assert types["q1"] == "Quadrupole" and types["s1"] == "Sextupole" and types["o1"] == "Octupole"
    assert types["m1"] == "Multipole" and types["b1"] == "SBend" and types["sol1"] == "Solenoid"
    assert types["c1"] == "Cavity" and types["c2"] == "Cavity" and types["k1"] == "Hcor" and types["k1_v"] == "Vcor"
    assert types["col1"] == "Aperture" and types["col1_body"] == "Drift" and types["bpm1"] == "Monitor"
    assert types["t1"] == "Matrix" and types["rc1"] == "Matrix" and types["fq1"] == "Marker"
    assert types["d4_aper_in"] == "Aperture" and types["d4_aper_out"] == "Aperture" and types["mk1"] == "Marker"
    text = out.read_text()
    assert "# lattix: reference species=\"electron\"" in text
    assert "lattice = MagneticLattice(cell)" in text and "tws0.E = 0.00261099895" in text
    cell = text.split("cell = (")[1].split(")")[0]
    assert cell.index("d4_aper_in") < cell.index("d4,") < cell.index("d4_aper_out")
    codes = rep.codes()
    for code in ("INSTRUMENT_AS_MONITOR", "THIN_GAP_AS_SHORT_CAVITY", "KICKER_SPLIT_HV", "FOIL_TO_MARKER",
                 "PATCH_DROPPED", "REFCHANGE_AS_MATRIX", "FOREIGN_DIRECTIVE", "APERTURE_AS_ELEMENT",
                 "OCELOT_SKEW_MULTIPOLE_DROPPED", "TAYLOR_BASIS_OCELOT"):
        assert code in codes, code
    assert "OCELOT_ELECTRON_ONLY" not in codes and "THIN_GAP_ADDS_LENGTH" not in codes


def test_measured_conventions(tmp_path):
    lat = all_kinds_lattice()
    mod, _, _ = _module(lat, tmp_path)
    brho = lat.reference.brho_signed
    assert brho < 0                                                    # an electron
    q1 = mod["q1"][1]
    assert q1["k1"] == pytest.approx(1.2 / brho) and q1["k1"] < 0
    assert mod["sol1"][1]["k"] == pytest.approx(0.5 / (2 * brho))      # k = B/(2Bρ_signed)
    b1 = mod["b1"][1]
    assert b1["gap"] == pytest.approx(0.06) and b1["fint"] == 0.45 and b1["e1"] == 0.05 and b1["angle"] == 0.1
    c2 = mod["c2"][1]
    assert c2["v"] == pytest.approx(1e-3) and c2["phi"] == pytest.approx(+30.0)     # GV, phi = −φs
    assert c2["freq"] == 162.5e6 and c2["l"] == 0.2
    m1 = mod["m1"][1]
    assert m1["kn"] == pytest.approx([0.01 / brho, 0.02 / brho])                    # MAD's knl
    assert mod["k1"][1]["angle"] == 1e-3 and mod["k1_v"][1]["angle"] == -2e-3
    col = mod["col1"][1]
    assert col["xmax"] == 0.01 and col["ymax"] == 0.02 and col["type"] == "rect"
    assert mod["d4_aper_in"][1]["type"] == "ellipt" and mod["d4_aper_in"][1]["xmax"] == 0.015
    t1 = mod["t1"][1]
    assert t1["r12"] == pytest.approx(0.1) and t1["r11"] == 1.0 and t1["r66"] == 1.0
    assert mod["rc1"][1]["delta_e"] == pytest.approx(1e-6)                           # GeV


def test_misalignment_is_an_attribute_line(tmp_path):
    lat = all_kinds_lattice()
    _, _, out = _module(lat, tmp_path)
    assert "q1.dx = 0.001" in out.read_text()


def test_non_electron_is_lossy_but_keeps_the_normalized_strengths(tmp_path):
    lat = all_kinds_lattice("proton")
    mod, rep, _ = _module(lat, tmp_path)
    brho = lat.reference.brho_signed
    assert brho > 0 and mod["q1"][1]["k1"] == pytest.approx(1.2 / brho)
    assert "OCELOT_ELECTRON_ONLY" in rep.codes()
    entry = next(e for e in rep.entries if e.code == "OCELOT_ELECTRON_ONLY")
    assert entry.cls == "LOSSY" and entry.details["species"] == "proton"


def test_names_are_python_identifiers():
    used: set[str] = set()
    assert _ident("bi4.bsw1l1.1", used) == "bi4_bsw1l1_1" and _ident("1q", used) == "e_1q"
    assert _ident("bi4.bsw1l1.1", used) == "bi4_bsw1l1_1_2"
    assert _ident("cell", used) == "e_cell" and _ident("Drift", used) == "e_Drift"


def test_thin_gap_takes_its_length_from_the_drifts(tmp_path):
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0),
                                                                frequency_Hz=162.5e6)),
                                      Drift(name="d2", length=0.5)], ref_at())
    mod, rep, out = _module(lat, tmp_path)
    L = mod["c"][1]["l"]
    assert 1e-3 <= L <= 0.5 * lat.reference.wavelength_m
    assert mod["d1"][1]["l"] + mod["d2"][1]["l"] + L == pytest.approx(1.0, abs=1e-12)
    assert mod["d1"][1]["l"] == pytest.approx(0.5 - L / 2) and mod["d2"][1]["l"] == pytest.approx(0.5 - L / 2)
    assert f"kind=RFCavity L=0 pad={L / 2:.15g}" in out.read_text().replace(f"pad={round(L / 2, 12):.15g}",
                                                                              f"pad={L / 2:.15g}")
    assert "THIN_GAP_ADDS_LENGTH" not in rep.codes() and "THIN_GAP_AS_SHORT_CAVITY" in rep.codes()


def test_thin_gap_without_drifts_grows_the_line(tmp_path):
    lat = Lattice.from_sequence("g", [Solenoid(name="s", length=0.2, solenoid=SolenoidP(Bsol_T=0.1)),
                                      RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=-0.5, frequency_Hz=1.3e9)),
                                      Marker(name="m")], ref_at())
    mod, rep, _ = _module(lat, tmp_path)
    assert "THIN_GAP_ADDS_LENGTH" in rep.codes()
    assert mod["s"][1]["l"] == 0.2 and mod["c"][1]["l"] > 0


def test_field_map_becomes_the_replacement_cavity(tmp_path):
    from lattix.ir.elements import FieldMap

    fm = FieldMap(name="fm", length=0.4, geom=100, files=["x"], rf=RFP(frequency_Hz=162.5e6, voltage_V=5e5,
                                                                        phase_rad=0.3, dE_ref_eV=3e5))
    fm.meta["map_summary"] = {"kind": "rf", "v_c_V": 5.1e5, "phase_sync_rad": -0.4, "dE_ref_eV": 3e5}
    lat = Lattice.from_sequence("s", [Drift(name="d", length=0.1), fm], ref_at())
    mod, rep, out = _module(lat, tmp_path)
    assert mod["fm"][0] == "Cavity" and mod["fm"][1]["v"] == pytest.approx(5.1e5 * 1e-9)
    assert mod["fm"][1]["phi"] == pytest.approx(math.degrees(0.4)) and mod["fm"][1]["l"] == 0.4
    assert "FM_TO_CAVITY" in rep.codes() and "from=FieldMap" in out.read_text()


def test_reused_definition_is_placed_twice(tmp_path):
    lat = Lattice.from_sequence("r", [Drift(name="d", length=1.0)], ref_at())
    lat.lines["r"].items[0].repeat = 3
    _, _, out = _module(lat, tmp_path)
    text = out.read_text()
    assert text.count("Drift(") == 1 and "d, d, d," in text


def test_strict_mode_raises_on_a_lossy_kind(tmp_path):
    from lattix.fidelity import TranslationError

    lat = Lattice.from_sequence("s", [Foil(name="f")], ref_at())
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "f.ocelot.py", "ocelot", strict=True)


def test_goldens(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.ocelot.py"
    write(lat, out, "ocelot")
    _golden("fodo.ocelot.py", out)
    out2 = tmp_path / "all_kinds.ocelot.py"
    write(all_kinds_lattice(), out2, "ocelot")
    _golden("all_kinds.ocelot.py", out2)
