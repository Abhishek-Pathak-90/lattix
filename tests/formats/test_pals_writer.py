"""PALS writer tests (PLAN §6 task 2.4): goldens, rules coverage, dual regimes, idempotence.

Golden snapshots live in ``tests/golden/pals``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_pals_writer.py``.  The only normalisation
applied is the lattix version in the two header comment lines and in ``notes``.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest
import yaml

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.pals import Reader, Writer, sanitize
from lattix.ir import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    Expression,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    Lattice,
    Line,
    LineItem,
    MagneticMultipoleP,
    Marker,
    Multipole,
    NCells,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceChange,
    ReferenceParticle,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    SolenoidP,
    Superposition,
    Taylor,
    Variable,
    species,
)
from lattix.ir.rf import pals_phase
from lattix.testing import needs

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "pals"
_VERSION = re.compile(r"lattix \d[^\s]*")


def _norm(text: str) -> str:
    return _VERSION.sub("lattix <version>", text)


def assert_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("LATTIX_UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_norm(text))
    assert path.exists(), f"missing golden {path}; rerun with LATTIX_UPDATE_GOLDEN=1"
    assert _norm(text) == path.read_text()


def proton_ref(ke: float = 8e8) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke)


def demo_lattice() -> Lattice:
    """FODO with two bends, a solenoid, a thin cavity, a kicker and a collimator."""
    ref = proton_ref()
    b = ref.brho_signed
    els = [
        Quadrupole(name="QF", length=0.3, multipole=MagneticMultipoleP(Bn={1: 0.6 * b})),
        Drift(name="D1", length=0.5),
        Bend(name="B1", length=1.0,
             bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.5, hgap=0.02)),
        Drift(name="D2", length=0.5),
        Solenoid(name="SOL", length=0.4, solenoid=SolenoidP(Bsol_T=0.3 * b)),
        Drift(name="D3", length=0.2),
        RFCavity(name="CAV", length=0.0,
                 rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6, frequency_Hz=325e6)),
        Drift(name="D4", length=0.2),
        Kicker(name="COR", length=0.0, hkick=1e-3, vkick=-2e-3),
        Drift(name="D5", length=0.3),
        Collimator(name="COLL", length=0.1, aperture=ApertureP.rect(0.02, 0.03)),
        Drift(name="D6", length=0.5),
        Bend(name="B2", length=1.0, bend=BendP(angle=-0.1, e1=-0.05, e2=-0.05, rect=True)),
        Drift(name="D7", length=0.5),
        Quadrupole(name="QD", length=0.3, multipole=MagneticMultipoleP(Bn={1: -0.6 * b}),
                   shift=BodyShiftP(x_offset=1e-4, tilt=math.pi / 4)),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("demo", els, ref)


def zoo_lattice() -> Lattice:
    """One element of every remaining IR kind, so the ledger covers the whole RULES table."""
    ref = proton_ref(1e8)
    els = [
        Sextupole(name="sx", length=0.2, multipole=MagneticMultipoleP(Bn={2: 12.0})),
        Octupole(name="oc", length=0.2, multipole=MagneticMultipoleP(Bn={3: 30.0})),
        Multipole(name="mp", multipole=MagneticMultipoleP(BnL={2: 0.4}, BsL={3: -0.1})),
        FieldMap(name="fm", length=0.5, rf=RFP(voltage_V=2e6, phase_rad=-0.4,
                                               frequency_Hz=162.5e6, dE_ref_eV=1.8e6)),
        FieldMap(name="fm_blind", length=0.5),
        NCells(name="nc", length=0.6),
        RFQCell(name="rq", length=0.05),
        Instrument(name="bpm", family="BPM", params={"gain": 2.0}),
        Foil(name="foil", material="C", thickness_kg_per_m2=1e-3),
        Taylor(name="tay"),
        Patch(name="pat", z_offset=0.1, y_rot=0.02),
        ReferenceChange(name="rc", dE_ref_eV=5e5),
        Freq(name="fq", frequency_Hz=325e6),
        Directive(name="dir", format="tracewin", card="ADJUST", args=["1", "2"], role="matching"),
        Directive(name="per", format="tracewin", card="LATTICE", args=[], role="period_start"),
        Directive(name="ttl", format="tracewin", card="TITLE", args=["a demo"], role="title"),
    ]
    els[9].matrix[0][1] = 0.75
    els[9].offset[2] = 1e-4
    lat = Lattice.from_sequence("zoo", els, ref)
    child_a = Solenoid(name="sup_sol", length=0.8, solenoid=SolenoidP(Bsol_T=0.2))
    child_b = RFCavity(name="sup_cav", length=0.3,
                       rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=325e6))
    lat.add_element(child_a)
    lat.add_element(child_b)
    sup = Superposition(name="sup", length=1.0, children=[(0.1, "sup_sol"), (0.5, "sup_cav")],
                        rf=RFP(dE_ref_eV=1e6))
    lat.add_element(sup)
    lat.lines["zoo"].items.append(LineItem(ref="sup"))
    return lat


def one_element(el: Element, *, ref: ReferenceParticle | None = None, **kw) -> Lattice:
    return Lattice.from_sequence("s", [el], ref or proton_ref(), **kw)


def dump(lat: Lattice, tmp_path, name="o.pals.yaml", **kw):
    out = Path(tmp_path) / name
    rep = Writer().write(lat, out, **kw)
    return yaml.safe_load(out.read_text()), rep, out


def facility(doc: dict) -> dict:
    """``{name: node}`` of every one-key facility entry."""
    out = {}
    for entry in doc["PALS"]["facility"]:
        if isinstance(entry, dict) and len(entry) == 1:
            k, v = next(iter(entry.items()))
            out[k] = v
    return out


# ── rules table ───────────────────────────────────────────────────────────────────────────
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_every_rule_has_a_builder():
    w = Writer()
    for kind in w.RULES:
        if kind in ("Freq", "Directive"):
            continue
        assert hasattr(w, f"_def_{kind.lower()}"), kind


# ── goldens ───────────────────────────────────────────────────────────────────────────────
def test_golden_demo_lattice(tmp_path):
    out = tmp_path / "demo.pals.yaml"
    rep = Writer().write(demo_lattice(), out)
    assert rep.ok
    assert_golden("demo.pals.yaml", out.read_text())


def test_golden_demo_lattice_normalized(tmp_path):
    out = tmp_path / "demo_n.pals.yaml"
    rep = Writer().write(demo_lattice(), out, normalized=True)
    assert rep.ok
    assert_golden("demo_normalized.pals.yaml", out.read_text())


def test_golden_zoo_lattice(tmp_path):
    out = tmp_path / "zoo.pals.yaml"
    rep = Writer().write(zoo_lattice(), out)
    assert not rep.ok                     # the zoo exists to exercise the downgrades
    assert_golden("zoo.pals.yaml", out.read_text())


@needs("madx")
def test_golden_from_madx_fodo(tmp_path):
    from lattix.formats.madx import Reader as MadxReader
    lat, _ = MadxReader().read(DATA / "helix" / "fodo.madx")
    out = tmp_path / "fodo.pals.yaml"
    rep = Writer().write(lat, out, strict=True)
    assert rep.ok
    assert_golden("fodo_from_madx.pals.yaml", out.read_text())


def test_golden_pals_fodo_roundtrip(tmp_path):
    lat, _ = Reader().read(DATA / "pals" / "fodo.pals.yaml")
    out = tmp_path / "fodo.pals.yaml"
    Writer().write(lat, out)
    assert_golden("fodo_roundtrip.pals.yaml", out.read_text())


@pytest.mark.parametrize("flavor", ["flat", "beamline"])
def test_golden_flavors(tmp_path, flavor):
    lat, _ = Reader().read(DATA / "pals" / "fodo.pals.yaml")
    out = tmp_path / f"{flavor}.pals.yaml"
    Writer().write(lat, out, flavor=flavor, beginning=False)
    assert_golden(f"fodo_{flavor}.pals.yaml", out.read_text())


# ── document shape ────────────────────────────────────────────────────────────────────────
def test_document_shape_and_use_statement(tmp_path):
    doc, rep, out = dump(demo_lattice(), tmp_path)
    assert list(doc) == ["PALS"]
    root = doc["PALS"]
    assert root["version"] is None
    assert any("lattix" in n for n in root["notes"])
    entries = root["facility"]
    assert entries[-1] == {"use": "demo_lattice"}
    lattice_node = entries[-2]["demo_lattice"]
    assert lattice_node == {"kind": "Lattice", "branches": ["demo"]}
    line = facility(doc)["demo"]
    assert line["kind"] == "BeamLine"
    first = line["line"][0]
    assert next(iter(first.values()))["kind"] == "BeginningEle"
    assert out.read_text().startswith("# PALS document written by lattix ")


def test_beginning_ele_carries_the_reference_particle(tmp_path):
    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=8e8, time_s=1e-9)
    doc, _rep, _out = dump(one_element(Drift(name="d", length=1.0), ref=ref), tmp_path)
    begin = next(iter(facility(doc)["s"]["line"][0].values()))
    assert begin["kind"] == "BeginningEle"
    assert begin["ReferenceP"]["species_ref"] == "#1H-1"
    assert begin["ReferenceP"]["E_tot_ref"] == pytest.approx(ref.total_energy_eV)
    assert begin["ReferenceP"]["time_ref"] == pytest.approx(1e-9)


def test_beginning_can_be_omitted_with_a_ledger_entry(tmp_path):
    doc, rep, _ = dump(one_element(Drift(name="d", length=1.0)), tmp_path, beginning=False)
    assert "PALS_NO_BEGINNING_ELE" in rep.codes()
    assert facility(doc)["s"]["line"] == ["d"]       # no BeginningEle at the head
    with pytest.raises(TranslationError):
        Writer().write(one_element(Drift(name="d", length=1.0)), Path(tmp_path) / "x.pals.yaml",
                       beginning=False, strict=True)


def test_nested_lines_repeat_and_direction(tmp_path):
    lat = Lattice(name="m", reference=proton_ref())
    lat.add_element(Drift(name="d", length=1.0))
    lat.add_element(Bend(name="b", length=2.0, bend=BendP(angle=0.1)))
    lat.lines["cell"] = Line(name="cell", items=[LineItem(ref="d"), LineItem(ref="b")])
    lat.lines["m"] = Line(name="m", items=[LineItem(ref="cell", repeat=3),
                                           LineItem(ref="cell", reverse=True)])
    lat.use = "m"
    doc, _rep, _out = dump(lat, tmp_path)
    f = facility(doc)
    assert f["cell"]["line"] == ["d", "b"]
    assert f["m"]["line"][1:] == [{"cell": {"repeat": 3}}, {"cell": {"direction": -1}}]
    # sublines are defined before the line that uses them
    keys = [next(iter(e)) for e in doc["PALS"]["facility"] if isinstance(e, dict)]
    assert keys.index("cell") < keys.index("m")


def test_json_output(tmp_path):
    out = Path(tmp_path) / "d.pals.json"
    Writer().write(demo_lattice(), out)
    doc = json.loads(out.read_text())
    assert doc["PALS"]["facility"][-1] == {"use": "demo_lattice"}
    out2 = Path(tmp_path) / "d.txt"
    Writer().write(demo_lattice(), out2, format="json")
    assert json.loads(out2.read_text()) == doc
    with pytest.raises(ValueError):
        Writer().write(demo_lattice(), out2, format="toml")
    with pytest.raises(ValueError):
        Writer().write(demo_lattice(), out2, flavor="nope")


# ── parameter groups ──────────────────────────────────────────────────────────────────────
def test_quadrupole_field_and_normalized_forms(tmp_path):
    ref = proton_ref()
    q = Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 2.0}, tilt={1: 0.1}))
    doc, _r, _o = dump(one_element(q, ref=ref), tmp_path)
    assert facility(doc)["q"]["MagneticMultipoleP"] == {"Bn1": 2.0, "tilt1": 0.1}
    doc, _r, _o = dump(one_element(q, ref=ref), tmp_path, name="n.pals.yaml", normalized=True)
    assert facility(doc)["q"]["MagneticMultipoleP"]["Kn1"] == pytest.approx(2.0 / ref.brho_signed)


def test_bend_group_names(tmp_path):
    """PALS calls the angle ``angle_ref`` and folds ``fint``·``hgap`` into ``edge1_int``."""
    b = Bend(name="b", length=1.0,
             bend=BendP(angle=0.2, e1=0.03, e2=0.04, edge_int1=0.5, edge_int2=0.7, hgap=0.02,
                        tilt_ref=math.pi / 2))
    doc, rep, _o = dump(one_element(b), tmp_path)
    g = facility(doc)["b"]["BendP"]
    assert g["angle_ref"] == pytest.approx(0.2)
    assert (g["e1"], g["e2"]) == pytest.approx((0.03, 0.04))
    assert g["edge1_int"] == pytest.approx(0.01)         # 0.5 * 0.02
    assert g["edge2_int"] == pytest.approx(0.014)
    assert g["tilt_ref"] == pytest.approx(math.pi / 2)
    assert "g_ref" not in g and "radius_ref" not in g     # only two shape parameters allowed
    assert facility(doc)["b"]["length"] == pytest.approx(1.0)
    assert rep.ok


def test_rectangular_bend_uses_rect_edges(tmp_path):
    b = Bend(name="b", length=1.0, bend=BendP(angle=0.2, e1=0.1, e2=0.1, rect=True))
    doc, _r, _o = dump(one_element(b), tmp_path)
    g = facility(doc)["b"]["BendP"]
    assert "e1" not in g and "e2" not in g
    assert (g["e1_rect"], g["e2_rect"]) == pytest.approx((0.0, 0.0))


def test_rf_group(tmp_path):
    cav = RFCavity(name="c", length=1.0,
                   rf=RFP(voltage_V=2e6, phase_rad=-math.pi / 6, frequency_Hz=650e6,
                          L_active_m=0.8, n_cell=5))
    doc, rep, _o = dump(one_element(cav), tmp_path)
    g = facility(doc)["c"]["RFP"]
    assert g["voltage"] == pytest.approx(2e6)
    assert g["phase"] == pytest.approx(pals_phase(-math.pi / 6))
    assert g["zero_phase"] == "ACCELERATING"
    assert g["frequency"] == pytest.approx(650e6)
    assert g["L_active"] == pytest.approx(0.8)
    assert g["num_cells"] == 5
    assert g["dE_ref"] == pytest.approx(2e6 * math.cos(-math.pi / 6))
    assert rep.ok


def test_rf_gradient_form_is_preserved_through_a_round_trip(tmp_path):
    lat, _ = Reader().read(DATA / "pals" / "rf_gradient.pals.yaml")
    doc, _rep, _out = dump(lat, tmp_path)
    g = facility(doc)["cav"]["RFP"]
    assert g["gradient"] == pytest.approx(1.0e8)
    assert "voltage" not in g


def test_a_pals_source_keeps_its_own_parameter_forms(tmp_path):
    """``native["pals"]`` remembers whether the source spelled a strength normalized and how a
    bend was parameterised, so a pals->pals conversion does not churn the file."""
    lat, _ = Reader().read(DATA / "pals" / "iota.pals.yaml")
    doc, rep, _o = dump(lat, tmp_path)
    f = facility(doc)
    assert f["qa1"]["MagneticMultipoleP"]["Kn1"] == pytest.approx(-8.78017699)   # not Bn1
    assert "Bn1" not in f["qa1"]["MagneticMultipoleP"]
    # radius_ref + length in, g_ref + length out (both are one curvature + one length)
    assert f["sbend30"]["BendP"]["g_ref"] == pytest.approx(1 / 0.822230996255981)
    assert "angle_ref" not in f["sbend30"]["BendP"]
    assert rep.ok
    # a lattice that never came from PALS keeps the IR's canonical lab fields
    doc, _r, _o = dump(demo_lattice(), tmp_path, name="ir.pals.yaml")
    assert "Bn1" in facility(doc)["QF"]["MagneticMultipoleP"]
    assert "angle_ref" in facility(doc)["B1"]["BendP"]


def test_solenoid_and_kicker_signs(tmp_path):
    ref = proton_ref()
    sol = Solenoid(name="sol", length=0.4, solenoid=SolenoidP(Bsol_T=0.3))
    doc, _r, _o = dump(one_element(sol, ref=ref), tmp_path)
    assert facility(doc)["sol"]["SolenoidP"] == {"Bsol": 0.3}
    doc, _r, _o = dump(one_element(sol, ref=ref), tmp_path, name="k.pals.yaml", normalized=True)
    assert facility(doc)["sol"]["SolenoidP"]["Ksol"] == pytest.approx(0.3 / ref.brho_signed)

    kick = Kicker(name="c", hkick=1e-3, vkick=-2e-3)
    doc, _r, _o = dump(one_element(kick, ref=ref), tmp_path, name="c.pals.yaml")
    g = facility(doc)["c"]["MagneticMultipoleP"]
    assert g["Bn0L"] == pytest.approx(-1e-3 * ref.brho_signed)   # +hkick bends towards +x
    assert g["Bs0L"] == pytest.approx(-2e-3 * ref.brho_signed)


def test_kicker_round_trips_through_the_reader(tmp_path):
    ref = proton_ref()
    lat = one_element(Kicker(name="c", hkick=1.5e-3, vkick=-2.5e-3), ref=ref)
    _doc, _rep, out = dump(lat, tmp_path)
    back, _ = Reader().read(out)
    assert back.elements["c"].hkick == pytest.approx(1.5e-3)
    assert back.elements["c"].vkick == pytest.approx(-2.5e-3)


def test_aperture_and_body_shift_names(tmp_path):
    d = Drift(name="d", length=1.0, aperture=ApertureP.circle(0.02),
              shift=BodyShiftP(x_offset=1e-3, tilt=0.2, y_rot=-0.01))
    doc, _r, _o = dump(one_element(d), tmp_path)
    node = facility(doc)["d"]
    assert node["ApertureP"] == {"x_min": -0.02, "x_max": 0.02, "y_min": -0.02, "y_max": 0.02,
                                 "shape": "ELLIPTICAL", "location": "BOTH_ENDS"}
    assert node["BodyShiftP"] == {"x_offset": 1e-3, "y_rot": -0.01, "z_rot": 0.2}


def test_collimator_becomes_a_mask(tmp_path):
    doc, rep, _o = dump(one_element(
        Collimator(name="cl", length=0.1, aperture=ApertureP.rect(0.01, 0.02))), tmp_path)
    node = facility(doc)["cl"]
    assert node["kind"] == "Mask"
    assert node["ApertureP"]["shape"] == "RECTANGULAR"
    assert rep.ok


def test_superposition_becomes_a_unionele(tmp_path):
    lat = Lattice(name="main", reference=proton_ref())
    lat.add_element(Solenoid(name="sa", length=0.8, solenoid=SolenoidP(Bsol_T=0.2)))
    lat.add_element(Drift(name="db", length=0.2))
    lat.add_element(Superposition(name="u", length=1.0, children=[(0.15, "sa"), (0.5, "db")]))
    lat.lines["main"] = Line(name="main", items=[LineItem(ref="u")])
    lat.use = "main"
    doc, rep, out = dump(lat, tmp_path)
    node = facility(doc)["u"]
    assert node["kind"] == "UnionEle"
    # children are centred on the union's centre, so z_offset = offset + L_child/2 - L_union/2
    assert node["elements"]["sa"]["BodyShiftP"]["z_offset"] == pytest.approx(0.15 + 0.4 - 0.5)
    assert node["elements"]["db"]["BodyShiftP"]["z_offset"] == pytest.approx(0.5 + 0.1 - 0.5)
    assert "SUPERPOSITION_AS_UNIONELE" in rep.codes()
    back, _ = Reader().read(out)
    assert sorted(back.elements["u"].children) == [pytest.approx((0.15, "sa")),
                                                   pytest.approx((0.5, "db"))]


def test_instrument_family_and_meta_alias(tmp_path):
    doc, rep, _o = dump(one_element(Instrument(name="bpm.1", family="BPM")), tmp_path)
    node = facility(doc)["bpm_1"]
    assert node["kind"] == "Instrument"
    assert node["MetaP"] == {"label": "BPM", "alias": "bpm.1"}    # . is not a PALS name char
    assert rep.ok


def test_names_are_sanitized_and_uniquified(tmp_path):
    assert sanitize("q.1") == "q_1"
    assert sanitize("2q") == "e_2q"
    assert sanitize("") == "e_"
    lat = Lattice(name="s", reference=proton_ref())
    lat.add_element(Drift(name="a-b", length=1.0))
    lat.add_element(Drift(name="a b", length=1.0))
    lat.lines["s"] = Line(name="s", items=[LineItem(ref="a-b"), LineItem(ref="a b")])
    lat.use = "s"
    doc, _r, _o = dump(lat, tmp_path)
    assert {"a_b", "a_b_2"} <= set(facility(doc))


def test_numbers_keep_full_precision(tmp_path):
    d = Drift(name="d", length=1.0 / 3.0)
    _doc, _rep, out = dump(one_element(d), tmp_path)
    assert repr(1.0 / 3.0) in out.read_text()
    back, _ = Reader().read(out)
    assert back.elements["d"].length == 1.0 / 3.0


def test_expressions_are_reused_when_they_still_evaluate(tmp_path):
    lat = Lattice(name="s", reference=proton_ref(),
                  variables={"lq": Variable(value=0.3), "kf": Variable(value=0.6)})
    q = Quadrupole(name="q", length=0.3,
                   multipole=MagneticMultipoleP(Bn={1: 0.6 * proton_ref().brho_signed}))
    q.expressions["length"] = Expression(text="lq", deferred=True)
    q.expressions["multipole.Bn[1]"] = Expression(text="kf", deferred=True)
    lat.add_element(q)
    lat.lines["s"] = Line(name="s", items=[LineItem(ref="q")])
    lat.use = "s"
    doc, _r, _o = dump(lat, tmp_path, normalized=True)
    assert {"variables": [{"lq": 0.3}, {"kf": 0.6}]} in doc["PALS"]["facility"]
    assert facility(doc)["q"]["length"] == "lq"
    assert facility(doc)["q"]["MagneticMultipoleP"]["Kn1"] == "kf"
    # ... and a stale expression falls back to the number
    q.expressions["length"] = Expression(text="lq * 2")
    doc, _r, _o = dump(lat, tmp_path, name="e2.pals.yaml", normalized=True)
    assert facility(doc)["q"]["length"] == 0.3
    # expressions=False writes numbers only
    doc, _r, _o = dump(lat, tmp_path, name="e3.pals.yaml", expressions=False)
    assert not any("variables" in e for e in doc["PALS"]["facility"] if isinstance(e, dict))


def test_variable_with_an_unwritable_name_is_reported(tmp_path):
    lat = one_element(Drift(name="d", length=1.0))
    lat.variables["a.b"] = Variable(value=1.0)
    _doc, rep, _o = dump(lat, tmp_path)
    assert "VARIABLE_DROPPED" in rep.codes()


# ── dual regime: every LOSSY / DROPPED path ───────────────────────────────────────────────
_PROBLEMS = [
    ("FM_TO_DRIFT", FieldMap(name="fm", length=0.5), "LOSSY"),
    ("NCELLS_TO_DRIFT", NCells(name="nc", length=0.6), "LOSSY"),
    ("RFQ_TO_DRIFT", RFQCell(name="rq", length=0.05), "LOSSY"),
    ("FOIL_MATERIAL_DROPPED", Foil(name="fo", material="C", thickness_kg_per_m2=1e-3), "LOSSY"),
    ("INSTRUMENT_PARAMS_DROPPED", Instrument(name="in", params={"gain": 1.0}), "LOSSY"),
    ("COLLIMATOR_WITHOUT_APERTURE", Collimator(name="cl", length=0.1), "LOSSY"),
    ("ELECTRIC_KICKER_AS_MAGNETIC", Kicker(name="ek", hkick=1e-3, electric=True), "LOSSY"),
    ("PALS_TTF_DROPPED", RFCavity(name="cv", rf=RFP(voltage_V=1e6, ttf=0.8)), "LOSSY"),
    ("PALS_FRINGE_K2_DROPPED",
     Bend(name="bk", length=1.0, bend=BendP(angle=0.1, fringe_k2=2.8)), "LOSSY"),
    ("PALS_EDGE_INT_NEEDS_HGAP",
     Bend(name="be", length=1.0, bend=BendP(angle=0.1, edge_int1=0.45, hgap=0.0)), "LOSSY"),
    ("PALS_REF_PHASE_DROPPED", ReferenceChange(name="rp", dphase_rad=0.3), "LOSSY"),
    ("FOREIGN_DIRECTIVE",
     Directive(name="dv", card="ADJUST", args=["1"], role="matching"), "DROPPED"),
]


@pytest.mark.parametrize(("code", "element", "cls"), _PROBLEMS, ids=[p[0] for p in _PROBLEMS])
def test_downgrade_permissive_records_strict_raises(tmp_path, code, element, cls):
    lat = one_element(element.model_copy(deep=True))
    rep = Writer().write(lat, Path(tmp_path) / "p.pals.yaml")
    entry = next((e for e in rep.entries if e.code == code), None)
    assert entry is not None, f"{code} not recorded; got {rep.codes()}"
    assert entry.cls == cls
    with pytest.raises(TranslationError) as exc:
        Writer().write(one_element(element.model_copy(deep=True)),
                       Path(tmp_path) / "s.pals.yaml", strict=True)
    assert code in str(exc.value) or exc.value.entry.code in {code, "PALS_NO_BEGINNING_ELE"}


_EQUIVALENT = [
    ("FM_TO_CAVITY", FieldMap(name="fm", length=0.5,
                                rf=RFP(voltage_V=1e6, phase_rad=0.0, dE_ref_eV=9e5))),
    ("PALS_PHASE_AS_SYNC", RFCavity(name="cv", rf=RFP(voltage_V=1e6, phase_is_sync=False))),
    ("PERIOD_AS_MARKER", Directive(name="pd", card="LATTICE", role="period_start")),
]


@pytest.mark.parametrize(("code", "element"), _EQUIVALENT, ids=[c for c, _ in _EQUIVALENT])
def test_equivalent_paths_are_recorded_but_never_fatal(tmp_path, code, element):
    rep = Writer().write(one_element(element), Path(tmp_path) / "e.pals.yaml", strict=True)
    assert code in rep.codes()
    assert rep.ok


def test_title_directive_becomes_a_note_and_period_a_marker(tmp_path):
    lat = Lattice.from_sequence("t", [
        Directive(name="ttl", card="TITLE", args=["my machine"], role="title"),
        Directive(name="per", card="LATTICE", role="period_start"),
        Drift(name="d", length=1.0),
    ], proton_ref())
    doc, rep, _o = dump(lat, tmp_path)
    assert any("my machine" in n for n in doc["PALS"]["notes"])
    assert facility(doc)["per"] == {"kind": "Marker", "MetaP": {"label": "period_start"}}
    assert "ttl" not in facility(doc)
    assert rep.ok


def test_freq_is_not_written(tmp_path):
    lat = Lattice.from_sequence("f", [Freq(name="fq", frequency_Hz=325e6),
                                      Drift(name="d", length=1.0)], proton_ref())
    doc, rep, _o = dump(lat, tmp_path)
    assert "fq" not in facility(doc)
    assert facility(doc)["f"]["line"][1:] == ["d"]
    assert rep.ok


def test_multi_energy_definition_is_reported(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=1e8)
    lat = Lattice(name="m", reference=ref)
    lat.add_element(RFCavity(name="cav", rf=RFP(voltage_V=5e6, phase_rad=0.0)))
    lat.add_element(Quadrupole(name="q", length=0.3,
                               multipole=MagneticMultipoleP(Bn={1: 1.0})))
    lat.lines["m"] = Line(name="m", items=[LineItem(ref="q"), LineItem(ref="cav"),
                                           LineItem(ref="q")])
    lat.use = "m"
    _doc, rep, _o = dump(lat, tmp_path, normalized=True)
    assert "PALS_MULTI_ENERGY_NORMALIZED" not in rep.codes()   # reader-side code
    assert "PALS_MULTI_ENERGY_DEFINITION" in rep.codes()


def test_one_entry_per_element_definition(tmp_path):
    lat = demo_lattice()
    _doc, rep, _o = dump(lat, tmp_path)
    named = [e for e in rep.entries if e.element is not None]
    assert {e.element for e in named} == set(lat.elements)


# ── idempotence ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["fodo", "iota", "bend_angle_radius", "rf_voltage",
                                  "rf_gradient", "drift_quad_bend"])
def test_write_read_write_is_a_fixed_point(tmp_path, name):
    lat, _ = Reader().read(DATA / "pals" / f"{name}.pals.yaml")
    a = Path(tmp_path) / "a.pals.yaml"
    b = Path(tmp_path) / "b.pals.yaml"
    Writer().write(lat, a)
    lat2, _ = Reader().read(a)
    Writer().write(lat2, b)
    assert a.read_text() == b.read_text()
    assert [(p.element.kind, p.length) for p in lat.flatten()] == \
           [(p.element.kind, p.length) for p in lat2.flatten()]


def test_demo_and_zoo_lattices_are_fixed_points(tmp_path):
    for i, lat in enumerate((demo_lattice(), zoo_lattice())):
        a, b = Path(tmp_path) / f"a{i}.pals.yaml", Path(tmp_path) / f"b{i}.pals.yaml"
        Writer().write(lat, a)
        lat2, _ = Reader().read(a)
        Writer().write(lat2, b)
        assert a.read_text() == b.read_text()


@needs("madx")
def test_madx_fodo_survives_a_pals_round_trip(tmp_path):
    from lattix.formats.madx import Reader as MadxReader
    lat, _ = MadxReader().read(DATA / "helix" / "fodo.madx")
    out = Path(tmp_path) / "f.pals.yaml"
    Writer().write(lat, out)
    back, rep = Reader().read(out)
    assert rep.ok
    a, b = lat.flatten(), back.flatten()
    assert [p.element.kind for p in a] == [p.element.kind for p in b]
    assert [round(p.length, 12) for p in a] == [round(p.length, 12) for p in b]
    assert back.elements["qf"].multipole.Bn[1] == pytest.approx(lat.elements["qf"].multipole.Bn[1])
    assert back.elements["b1"].bend.angle == pytest.approx(lat.elements["b1"].bend.angle)
    assert back.elements["b1"].bend.e1 == pytest.approx(lat.elements["b1"].bend.e1)
    assert back.reference.total_energy_eV == pytest.approx(lat.reference.total_energy_eV)
