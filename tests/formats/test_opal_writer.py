"""OPAL-T writer: rules coverage, the deck's statements and ledger, the source-derived conventions (ELEMEDGE
placement, strengths over the BEAM's P0/c, LAG relative to the crest, DESIGNENERGY as the crest energy, the
generated 1-D maps), goldens."""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from lattix import read, write
from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.impactt.rfprofile import transit_factor
from lattix.formats.opal import Writer
from lattix.formats.opal.maps import DYNAMIC, STATIC, parse_map
from lattix.formats.opal.reader import parse_attributes, statements
from lattix.ir.elements import RFP, Drift, Multipole, RFCavity
from lattix.ir.elements import MagneticMultipoleP as MP
from lattix.ir.lattice import Lattice
from tests.formats.test_ocelot_writer import all_kinds_lattice, ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "opal"
C = 299_792_458.0


def defs(text: str) -> dict[str, tuple[str, dict, dict]]:
    """``{identifier: (TYPE, attributes, tag)}`` of the element definitions, plus ``LINE``/``BEAM``/``TRACK``."""
    out = {}
    stmts, _ = statements(text)
    for st in stmts:
        t = st.text.strip()
        if ":" in t.split(",")[0]:
            name, rest = t.split(":", 1)
            rest = rest.strip()
            if rest.upper().startswith("LINE"):
                inner = rest.split("(", 1)[1].rsplit(")", 1)[0]
                out[name.strip()] = ("LINE", {"members": [m.strip() for m in inner.split(",")]}, st.tag)
                continue
            etype, _, body = rest.partition(",")
            out[name.strip()] = (etype.strip().upper(), parse_attributes(body), st.tag)
        else:
            key, _, body = t.partition(",")
            out[key.strip().upper()] = (key.strip().upper(), parse_attributes(body), st.tag)
    return out


def _deck(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "lat.opal.in"
    rep = write(lat, out, "opal", **opts)
    return defs(out.read_text()), rep, out


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = path.read_text()
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def f(v) -> float:
    return float(str(v).strip('"'))


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_all_kinds_deck_statements_and_ledger(tmp_path):
    lat = all_kinds_lattice("proton")
    d, rep, out = _deck(lat, tmp_path)
    types = {k: v[0] for k, v in d.items()}
    assert types["q1"] == "QUADRUPOLE" and types["s1"] == "SEXTUPOLE" and types["o1"] == "OCTUPOLE"
    assert types["m1_k"] == "KICKER" and types["m1"] == "MULTIPOLE" and types["b1"] == "SBEND"
    assert types["sol1"] == "SOLENOID" and types["c1"] == "RFCAVITY" and types["c2"] == "RFCAVITY"
    assert types["k1"] == "KICKER" and types["col1"] == "RCOLLIMATOR" and types["mk1"] == "MARKER"
    assert types["bpm1"] == "MONITOR" and types["f1"] == "MARKER" and types["t1"] == "DRIFT"
    assert types["p1"] == "MARKER" and types["rc1"] == "MARKER" and types["fq1"] == "MARKER" and types["d4"] == "DRIFT"
    assert types["all_kinds"] == "LINE" and types["lattix_beam"] == "BEAM" and types["TRACK"] == "TRACK"
    members = d["all_kinds"][1]["members"]
    assert members[:8] == ["d1", "q1", "s1", "o1", "m1_k", "m1", "b1", "sol1"] and members[-1] == "d4"
    # every element sits at its entrance path length
    assert f(d["q1"][1]["ELEMEDGE"]) == 0.5 and f(d["b1"][1]["ELEMEDGE"]) == 0.9 and f(d["d4"][1]["ELEMEDGE"]) == 3.25
    assert d["d4"][1]["APERTURE"] == '"CIRCLE(0.03)"' and f(d["col1"][1]["XSIZE"]) == 0.01
    assert d["lattix_beam"][1]["PARTICLE"] == "PROTON" and f(d["lattix_beam"][1]["BFREQ"]) == 162.5
    assert f(d["TRACK"][1]["ZSTOP"]) == pytest.approx(3.45)
    codes = {e.code for e in rep.entries}
    assert {"OPAL_CAVITY_MAP", "OPAL_SOLENOID_MAP", "OPAL_BEND_DEFAULT_PROFILE", "MULTIPOLE_AS_SHORT",
            "APERTURE_AS_ATTRIBUTE", "INSTRUMENT_AS_MONITOR", "FOIL_TO_MARKER", "TAYLOR_DROPPED", "PATCH_DROPPED",
            "REFCHANGE_AS_TAG", "FOREIGN_DIRECTIVE"} <= codes
    assert sorted(p.name for p in out.parent.iterdir()) == ["lat.opal.in", "lat_c1.1dd", "lat_c2.1dd", "lat_sol1.1dms"]
    text = out.read_text()
    assert "// lattix: directive=1 name=dir1" in text and 'RUN, METHOD="PARALLEL-T"' in text
    assert "OPTION, AUTOPHASE=4" in text


@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_strengths_over_the_beam_rigidity(tmp_path, sp):
    """K1 = G/(P0/c) with the BEAM's unsigned rigidity: the written field is the lab field for either charge."""
    lat = all_kinds_lattice(sp)
    ref = lat.reference
    d, rep, out = _deck(lat, tmp_path)
    assert f(d["q1"][1]["K1"]) == pytest.approx(1.2 / ref.brho_abs, rel=1e-12)
    assert f(d["q1"][1]["K1"]) > 0                      # the same sign for H⁻ (MAD-X would flip it)
    assert f(d["s1"][1]["K2"]) == pytest.approx(3.0 / ref.brho_abs, rel=1e-12)
    assert f(d["o1"][1]["K3"]) == pytest.approx(4.0 / ref.brho_abs, rel=1e-12)
    assert f(d["sol1"][1]["KS"]) == pytest.approx(0.5 / ref.brho_abs, rel=1e-12)
    # a thin multipole: dipole terms as a KICKER (deflections of the actual particle), the quadrupole term
    # over the surrogate length (1 mm), both centred on the element
    k, m = d["m1_k"][1], d["m1"][1]
    assert f(k["HKICK"]) == pytest.approx(-0.01 / ref.brho_signed, rel=1e-12)
    assert f(k["VKICK"]) == pytest.approx(0.02 / ref.brho_signed, rel=1e-12)
    assert f(k["L"]) == 1e-3 and f(k["ELEMEDGE"]) == pytest.approx(0.9 - 0.5e-3)
    assert m["KN"] == "{0," + f"{0.02 / (1e-3 * ref.brho_abs):.15g}" + "}"
    assert d["m1"][2] == {"kind": "Multipole", "L": "0", "role": "higher"}
    # kicker: deflection in rad at the local energy (after the two cavities)
    ke = 2.1e6 + 8e4 * math.cos(math.radians(-85.0)) + 1e6 * math.cos(math.radians(-30.0))
    assert f(d["k1"][1]["HKICK"]) == 1e-3 and f(d["k1"][1]["VKICK"]) == -2e-3
    assert f(d["k1"][1]["DESIGNENERGY"]) == pytest.approx(ke * 1e-6, rel=1e-12)
    # bend: geometric angle, MAD faces, DESIGNENERGY [MeV kinetic] at the entrance, OPAL's default profile
    b = d["b1"][1]
    assert f(b["ANGLE"]) == 0.1 and f(b["E1"]) == 0.05 and f(b["E2"]) == 0.05 and f(b["HGAP"]) == 0.03
    assert f(b["GAP"]) == 0.06 and f(b["FINT"]) == 0.45 and f(b["DESIGNENERGY"]) == pytest.approx(2.1)
    assert b["FMAPFN"] == '"1DPROFILE1-DEFAULT"'


def test_cavities_lag_designenergy_and_maps(tmp_path):
    lat = all_kinds_lattice("proton")
    ref = lat.reference
    d, rep, out = _deck(lat, tmp_path)
    c2 = d["c2"][1]
    ke_in = 2.1e6 + 8e4 * math.cos(math.radians(-85.0))
    assert f(c2["LAG"]) == pytest.approx(math.radians(-30.0))            # the synchronous phase, crest-relative
    assert f(c2["FREQ"]) == 162.5 and c2["TYPE"] == '"STANDING"'
    assert f(c2["DESIGNENERGY"]) == pytest.approx((ke_in + 1e6) * 1e-6, rel=1e-12)   # the crest energy
    assert f(c2["L"]) == 0.2 and f(c2["ELEMEDGE"]) == 2.8
    m = parse_map((out.parent / c2["FMAPFN"].strip('"')).read_text())
    assert m.kind == DYNAMIC and m.frequency_Hz == 162.5e6 and m.length_m == pytest.approx(0.2)
    assert m.peak == pytest.approx(1.0) and m.z_m[0] == 0.0 and len(m.values) == 201
    # VOLT [MV/m] is the peak field whose first-order transit-time integral at the entrance velocity is V
    beta = ref.advanced(dE_eV=ke_in - 2.1e6).beta
    F = abs(transit_factor(m.z_m, m.values, 2.0 * math.pi * 162.5e6 / (beta * C)))
    assert f(c2["VOLT"]) * 1e6 * F == pytest.approx(1e6, rel=1e-9)
    # the thin gap: a surrogate length centred on the gap, tag L=0
    c1 = d["c1"][1]
    assert 1e-3 <= f(c1["L"]) <= 0.5 * ref.beta * C / 162.5e6
    assert f(c1["ELEMEDGE"]) == pytest.approx(2.5 - 0.5 * f(c1["L"]))
    assert d["c1"][2]["L"] == "0" and f(d["c1"][2]["V"]) == 8e4
    assert f(c1["DESIGNENERGY"]) == pytest.approx(2.18)
    assert d["c2"][2]["tw"] == "1"


def test_solenoid_map_keeps_the_field_integral(tmp_path):
    lat = all_kinds_lattice("proton")
    d, rep, out = _deck(lat, tmp_path)
    s = d["sol1"][1]
    m = parse_map((out.parent / s["FMAPFN"].strip('"')).read_text())
    assert m.kind == STATIC and m.length_m == pytest.approx(0.32) and f(s["L"]) == pytest.approx(0.32)
    assert f(s["ELEMEDGE"]) == pytest.approx(1.9 - 0.01)
    assert d["sol1"][2] == {"kind": "Solenoid", "L": "0.3", "pad": "0.01"}
    dz = m.z_m[1] - m.z_m[0]
    integral = sum(0.5 * (a + b) * dz for a, b in zip(m.values[:-1], m.values[1:], strict=True))
    assert integral == pytest.approx(0.3, rel=1e-3)                    # ∫B dz = Bsol · L
    assert max(m.values) == 1.0 and m.values[0] == 0.0 and m.values[-1] == 0.0


def test_negative_voltage_and_missing_frequency(tmp_path):
    lat = Lattice.from_sequence("t", [
        Drift(name="d", length=0.1),
        RFCavity(name="nof", length=0.1, rf=RFP(voltage_V=2e5, phase_rad=0.0)),      # no RF clock yet
        RFCavity(name="neg", length=0.1, rf=RFP(voltage_V=-2e5, phase_rad=0.0, frequency_Hz=352.2e6)),
    ], ref_at(sp="proton", f=None))
    d, rep, out = _deck(lat, tmp_path)
    assert f(d["neg"][1]["LAG"]) == pytest.approx(math.pi) and f(d["neg"][1]["DESIGNENERGY"]) == pytest.approx(2.5)
    assert d["nof"][0] == "DRIFT" and "OPAL_CAVITY_TO_DRIFT" in {e.code for e in rep.entries}
    assert "BFREQ" not in d["lattix_beam"][1]


def test_fieldmap_written_as_its_own_map(tmp_path):
    lat, _ = read(DATA / "lightwin" / "example.dat", species="proton", kinetic_energy_eV=20e6, frequency_Hz=352.2e6)
    d, rep, out = _deck(lat, tmp_path)
    fm = next(v for k, v in d.items() if v[2].get("kind") == "FieldMap" and v[0] == "RFCAVITY")
    m = parse_map((out.parent / fm[1]["FMAPFN"].strip('"')).read_text())
    assert m.kind == DYNAMIC and m.frequency_Hz == pytest.approx(352.2e6) and m.peak == pytest.approx(1.0)
    assert "file" in fm[2] and "V" in fm[2] and f(fm[1]["L"]) == pytest.approx(m.length_m)
    codes = {e.code for e in rep.entries}
    assert "FM_AS_OPAL_MAP" in codes and "FM_TO_CAVITY" not in codes


def test_names_become_identifiers(tmp_path):
    lat = Lattice.from_sequence("t", [Drift(name="d.1", length=0.1), Drift(name="d_1", length=0.2),
                                      Multipole(name="line", multipole=MP(BnL={1: 0.01}))], ref_at(sp="proton"))
    d, rep, out = _deck(lat, tmp_path)
    assert d["d_1"][2]["name"] == "d.1" and d["d_1_2"][2]["name"] == "d_1" and "e_line" in d
    assert d["t"][1]["members"] == ["d_1", "d_1_2", "e_line"]


def test_strict_raises_on_lossy(tmp_path):
    with pytest.raises(TranslationError):
        write(all_kinds_lattice("proton"), tmp_path / "s.opal.in", "opal", strict=True)


def test_goldens(tmp_path):
    d, rep, out = _deck(all_kinds_lattice("proton"), tmp_path)
    _golden("all_kinds.opal.in", out)
    lat, _ = read(DATA / "helix" / "fodo.madx")
    out2 = tmp_path / "fodo.opal.in"
    write(lat, out2, "opal")
    _golden("fodo.opal.in", out2)
