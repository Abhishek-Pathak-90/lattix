"""ImpactX writer tests (PLAN §6 task 3.2): goldens, rules coverage, dual regimes.

Golden snapshots live in ``tests/golden/impactx``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_impactx_writer.py``.  The only
normalisation applied is the lattix version + source format in the header comment.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.impactx import Reader, Writer
from lattix.formats.impactx.writer import (
    INPUTS_TYPE,
    MAX_NAME_LEN,
    Emit,
    is_valid,
    sanitize,
    to_elements,
)
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    NCells,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    SolenoidP,
    Superposition,
    Taylor,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "impactx"
_VERSION_RE = re.compile(r"^(#[ ]?)lattix \S+ from .*$", re.MULTILINE)


def _norm(text: str) -> str:
    return _VERSION_RE.sub(r"\1lattix <version> from <fmt>", text)


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
        Collimator(name="COLL", length=0.0, aperture=ApertureP.rect(0.02, 0.03)),
        Drift(name="D6", length=0.5),
        Bend(name="B2", length=1.0, bend=BendP(angle=-0.1, e1=-0.05, e2=-0.05, rect=True)),
        Drift(name="D7", length=0.5),
        Quadrupole(name="QD", length=0.3, multipole=MagneticMultipoleP(Bn={1: -0.6 * b})),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("demo", els, ref)


def all_kinds_lattice() -> Lattice:
    """One element of every one of the 22 IR kinds."""
    ref = proton_ref(1e8)
    b = ref.brho_signed
    child = Drift(name="sup_child", length=0.2)
    els = [
        Drift(name="dr", length=0.4),
        Quadrupole(name="qp", length=0.3,
                   multipole=MagneticMultipoleP(Bn={1: 0.6 * b}, tilt={1: 0.02})),
        Sextupole(name="sx", length=0.2, multipole=MagneticMultipoleP(Bn={2: 1.5 * b})),
        Octupole(name="oc", length=0.2, multipole=MagneticMultipoleP(Bn={3: 2.5 * b})),
        Multipole(name="mp", multipole=MagneticMultipoleP(BnL={0: 0.01 * b, 2: 0.3 * b},
                                                          BsL={1: 0.03 * b})),
        Bend(name="bd", length=1.0,
             bend=BendP(angle=0.1, e1=0.02, e2=0.03, edge_int1=0.45, edge_int2=0.5,
                        hgap=0.03, tilt_ref=0.1)),
        Solenoid(name="sl", length=0.4, solenoid=SolenoidP(Bsol_T=0.3 * b)),
        RFCavity(name="cv", length=0.5,
                 rf=RFP(voltage_V=2e6, phase_rad=-math.pi / 6, frequency_Hz=325e6, n_cell=5)),
        FieldMap(name="fm", length=0.6,
                 rf=RFP(voltage_V=0.0, phase_rad=-math.pi / 6, frequency_Hz=325e6,
                        dE_ref_eV=1.5e6), files=["map.edz"]),
        NCells(name="nc", length=0.7),
        RFQCell(name="rq", length=0.05),
        Kicker(name="kk", length=0.1, hkick=1e-3, vkick=-2e-3),
        Collimator(name="cl", length=0.1,
                   aperture=ApertureP(shape="ELLIPTICAL", x_limits=(-0.02, 0.02),
                                      y_limits=(-0.03, 0.03))),
        Marker(name="mk"),
        Instrument(name="bp", length=0.0, family="BPM"),
        Foil(name="fl", thickness_kg_per_m2=0.237, material="C"),
        Taylor(name="ty", length=0.2),
        Patch(name="pt", x_offset=1e-3, y_rot=2e-3, tilt=0.3),
        ReferenceChange(name="rc", energy_eV=1.2e8),
        Freq(name="fq", frequency_Hz=650e6),
        Directive(name="dv", format="tracewin", card="ADJUST", role="matching",
                  args=["1", "2"]),
        Superposition(name="sp", length=0.2, children=[(0.0, "sup_child")]),
    ]
    lat = Lattice.from_sequence("allkinds", els, ref)
    lat.elements["sup_child"] = child
    ty = lat.elements["ty"]
    ty.matrix[0][1] = 0.5
    ty.matrix[4][5] = 0.25
    return lat


def linac_lattice() -> Lattice:
    """Three thin 300 kV gaps at -30 deg with quads, proton at 2.1 MeV."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6,
                            rf_frequency_Hz=162.5e6)
    b = ref.brho_signed
    els = []
    for i in range(3):
        sign = 1.0 if i % 2 == 0 else -1.0
        els += [
            Quadrupole(name=f"q{i + 1}", length=0.1,
                       multipole=MagneticMultipoleP(Bn={1: sign * 5.0 * b})),
            Drift(name=f"da{i + 1}", length=0.2),
            RFCavity(name=f"cav{i + 1}", length=0.0,
                     rf=RFP(voltage_V=3e5, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
            Drift(name=f"db{i + 1}", length=0.2),
        ]
    return Lattice.from_sequence("linac", els, ref)


# ---------------------------------------------------------------- structure
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_every_emitted_class_has_an_inputs_type_or_is_marker():
    for lat in (demo_lattice(), all_kinds_lattice(), linac_lattice()):
        emits, _ = to_elements(lat, flavor="inputs")
        for e in emits:
            assert e.cls in INPUTS_TYPE, e.cls
            assert INPUTS_TYPE[e.cls] is not None, f"{e.cls} has no inputs type"


def test_names_are_python_identifiers_and_unique():
    lat = all_kinds_lattice()
    emits, _ = to_elements(lat)
    names = [e.name for e in emits]
    for n in names:
        assert is_valid(n), n
    # a definition used once per occurrence keeps a stable name; duplicates only appear
    # when the same IR element is placed twice
    assert len(set(names)) == len(names)


def test_sanitize_rules():
    assert sanitize("QF.1") == "QF_1"
    assert sanitize("2bad") == "e_2bad"
    assert sanitize("lattice") == "lattice_x"
    assert sanitize("class") == "class_x"
    assert len(sanitize("x" * 200)) == MAX_NAME_LEN
    assert not is_valid("beam")
    assert not is_valid("import")
    assert is_valid("qf_1")


# ---------------------------------------------------------------- goldens
def test_golden_demo():
    lat = demo_lattice()
    assert_golden("demo.impactx.py", Writer().render(lat))
    assert_golden("demo.impactx.in", Writer().render(lat, flavor="inputs"))


def test_golden_all_kinds():
    assert_golden("all_kinds.impactx.py", Writer().render(all_kinds_lattice()))


def test_golden_linac():
    assert_golden("linac.impactx.py", Writer().render(linac_lattice()))
    assert_golden("linac.impactx.in", Writer().render(linac_lattice(), flavor="inputs"))


@pytest.mark.oracle_madx
def test_golden_fodo_from_madx():
    pytest.importorskip("cpymad")
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "fodo.madx")
    assert_golden("fodo_from_madx.impactx.py", Writer().render(lat))
    assert_golden("fodo_from_madx.impactx.in", Writer().render(lat, flavor="inputs"))


# ---------------------------------------------------------------- physics
def test_quad_k_uses_signed_rigidity():
    ref = proton_ref(1e8)
    b = ref.brho_signed
    lat = Lattice.from_sequence(
        "q", [Quadrupole(name="q", length=0.3,
                         multipole=MagneticMultipoleP(Bn={1: 12.0}))], ref)
    emits, _ = to_elements(lat)
    assert emits[0].cls == "Quad"
    assert emits[0].params["k"] == pytest.approx(12.0 / b)

    hm = ReferenceParticle(species=species("h-"), kinetic_energy_eV=1e8)
    lat2 = Lattice.from_sequence(
        "q", [Quadrupole(name="q", length=0.3,
                         multipole=MagneticMultipoleP(Bn={1: 12.0}))], hm)
    e2, _ = to_elements(lat2)
    assert e2[0].params["k"] < 0, "H- flips the sign of k through the signed rigidity"


def test_solenoid_ks_and_tilted_quad_rotation():
    ref = proton_ref(1e8)
    b = ref.brho_signed
    lat = Lattice.from_sequence("s", [
        Solenoid(name="s", length=0.4, solenoid=SolenoidP(Bsol_T=0.75)),
        Quadrupole(name="q", length=0.2,
                   multipole=MagneticMultipoleP(Bn={1: 3.0}, tilt={1: math.pi / 4})),
    ], ref)
    emits, _ = to_elements(lat)
    assert emits[0].params["ks"] == pytest.approx(0.75 / b)
    assert emits[1].params["rotation"] == pytest.approx(45.0)


def test_bend_becomes_dipedge_sbend_dipedge():
    ref = proton_ref()
    lat = Lattice.from_sequence("b", [
        Bend(name="b", length=2.0,
             bend=BendP(angle=0.2, e1=0.11, e2=0.09, edge_int1=0.45, edge_int2=0.5,
                        hgap=0.025, tilt_ref=math.pi / 2)),
    ], ref)
    emits, rep = to_elements(lat)
    assert [e.cls for e in emits] == ["DipEdge", "Sbend", "DipEdge"]
    assert emits[0].params == {"psi": 0.11, "rc": 10.0, "g": 0.05, "K2": 0.45,
                               "location": "entry", "rotation": pytest.approx(90.0)}
    assert emits[1].params["rc"] == pytest.approx(10.0)
    assert emits[2].params["K2"] == pytest.approx(0.5)
    assert emits[2].params["location"] == "exit"
    # ImpactX's DipEdge default is K2 = 1 (MAD-X's FINT default is 0): always explicit
    assert "K2" in emits[0].params and "K2" in emits[2].params
    assert rep.ok


def test_bend_with_gradient_becomes_cfbend():
    ref = proton_ref()
    b = ref.brho_signed
    lat = Lattice.from_sequence("b", [
        Bend(name="b", length=2.0, bend=BendP(angle=0.2),
             multipole=MagneticMultipoleP(Bn={1: 0.3 * b}))], ref)
    emits, _ = to_elements(lat)
    assert [e.cls for e in emits] == ["CFbend"]
    assert emits[0].params["k"] == pytest.approx(0.3)


def test_shortrf_voltage_and_phase_convention():
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    lat = Lattice.from_sequence("c", [
        RFCavity(name="c", length=0.0,
                 rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0), frequency_Hz=162.5e6))],
        ref)
    emits, _ = to_elements(lat)
    assert emits[0].cls == "ShortRF"
    # V is dimensionless: max energy gain / rest energy
    assert emits[0].params["V"] == pytest.approx(1e6 / ref.species.mass_eV)
    # phase in degrees, 0 = crest, identical to the IR synchronous phase
    assert emits[0].params["phase"] == pytest.approx(-30.0)
    assert emits[0].params["freq"] == pytest.approx(162.5e6)


def test_multipole_uses_madx_knl_convention():
    ref = proton_ref(1e8)
    b = ref.brho_signed
    lat = Lattice.from_sequence("m", [
        Sextupole(name="sx", length=0.2, multipole=MagneticMultipoleP(Bn={2: 1.5 * b})),
    ], ref)
    emits, rep = to_elements(lat)
    assert [e.cls for e in emits] == ["Drift", "Multipole", "Drift"]
    m = emits[1]
    assert m.params["multipole"] == 3          # m = 3 is a sextupole in ImpactX
    assert m.params["k_normal"] == pytest.approx(1.5 * 0.2)   # k2 * L, no factorial
    assert any(e.code == "THICK_TO_THIN_MULTIPOLE" for e in rep.entries)


def test_kicker_and_collimator():
    ref = proton_ref()
    lat = Lattice.from_sequence("k", [
        Kicker(name="k", hkick=1e-3, vkick=-2e-3),
        Collimator(name="c", aperture=ApertureP(shape="RECTANGULAR", x_limits=(-0.01, 0.03),
                                                y_limits=(-0.02, 0.02))),
    ], ref)
    emits, _ = to_elements(lat)
    assert emits[0].params["xkick"] == pytest.approx(1e-3)
    assert emits[0].params["ykick"] == pytest.approx(-2e-3)
    assert emits[0].params["unit"] == "dimensionless"
    assert emits[1].cls == "Aperture"
    assert emits[1].params["aperture_x"] == pytest.approx(0.02)   # half-aperture
    assert emits[1].params["dx"] == pytest.approx(0.01)           # off-centre -> dx
    assert emits[1].params["shape"] == "rectangular"


def test_body_shift_becomes_dx_dy_rotation():
    ref = proton_ref()
    lat = Lattice.from_sequence("d", [
        Drift(name="d", length=1.0,
              shift=BodyShiftP(x_offset=1e-3, y_offset=-2e-3, tilt=math.pi / 6)),
    ], ref)
    emits, rep = to_elements(lat)
    assert emits[0].params["dx"] == pytest.approx(1e-3)
    assert emits[0].params["dy"] == pytest.approx(-2e-3)
    assert emits[0].params["rotation"] == pytest.approx(30.0)
    assert rep.ok


def test_energy_mode_local_is_exact_for_accelerating_lattices():
    lat = linac_lattice()
    placed = propagate(lat)
    emits_local, rep_local = to_elements(lat, energy_mode="local")
    emits_const, rep_const = to_elements(lat, energy_mode="constant")
    quads_local = [e for e in emits_local if e.cls == "Quad"]
    quads_const = [e for e in emits_const if e.cls == "Quad"]
    # the reference gains energy, so the later quads must be weaker in the local mode
    assert abs(quads_local[-1].params["k"]) < abs(quads_const[-1].params["k"])
    quad_places = [p for p in placed if p.element.kind == "Quadrupole"]
    for e, p in zip(quads_local, quad_places, strict=True):
        assert e.params["k"] == pytest.approx(
            p.element.multipole.Bn[1] / p.ref_in.brho_signed)
    assert rep_local.ok and any(e.code == "FOLLOWS_P0" for e in rep_local.entries)
    assert any(e.code == "CONST_START_RIGIDITY" for e in rep_const.entries)


# ---------------------------------------------------- dual-regime downgrades
DOWNGRADES = [
    ("FM_TO_DRIFT", lambda ref: FieldMap(name="fm", length=0.6, files=["m.edz"])),
    ("FM_TO_CAVITY", lambda ref: FieldMap(
        name="fm", length=0.6, rf=RFP(voltage_V=1e6, phase_rad=-0.5, frequency_Hz=325e6))),
    ("NCELLS_TO_DRIFT", lambda ref: NCells(name="nc", length=0.7)),
    ("RFQ_TO_DRIFT", lambda ref: RFQCell(name="rq", length=0.05)),
    ("FOIL_TO_MARKER", lambda ref: Foil(name="fl", thickness_kg_per_m2=0.2)),
    ("PATCH_DROPPED", lambda ref: Patch(name="pt", x_offset=1e-3, y_rot=2e-3)),
    ("REFCHANGE_DROPPED", lambda ref: ReferenceChange(name="rc", dE_ref_eV=1e5)),
    ("FOREIGN_DIRECTIVE", lambda ref: Directive(name="dv", card="ADJUST", role="matching")),
    ("THICK_TO_THIN_MULTIPOLE", lambda ref: Sextupole(
        name="sx", length=0.2, multipole=MagneticMultipoleP(Bn={2: 1.5 * ref.brho_signed}))),
    ("THIN_SOLENOID_DROPPED", lambda ref: Solenoid(
        name="sl", length=0.0, solenoid=SolenoidP(Bsol_T=0.3))),
    ("APERTURE_UNSET", lambda ref: Collimator(name="cl")),
    ("MISALIGN_DROPPED", lambda ref: Drift(name="d", length=1.0,
                                           shift=BodyShiftP(z_offset=1e-3))),
    ("APERTURE_DROPPED", lambda ref: Kicker(name="k", hkick=1e-3,
                                            aperture=ApertureP.circle(0.02))),
    ("RF_VOLTAGE_UNKNOWN", lambda ref: RFCavity(
        name="c", length=0.3, rf=RFP(frequency_Hz=325e6))),
    ("THIN_BEND_KICK", lambda ref: Bend(name="b", length=0.0, bend=BendP(angle=0.05))),
    ("RF_FREQUENCY_UNKNOWN", lambda ref: RFCavity(
        name="c", length=0.0, rf=RFP(voltage_V=1e6, phase_rad=0.0))),
    ("EMPTY_BEND", lambda ref: Bend(name="b", length=0.0, bend=BendP(angle=0.0))),
    ("ELECTRIC_KICKER", lambda ref: Kicker(name="k", hkick=1e-3, electric=True)),
    ("APERTURE_PARTIAL", lambda ref: Drift(
        name="d", length=1.0, aperture=ApertureP(shape="ELLIPTICAL", x_limits=(-0.02, 0.02)))),
    ("SUPERPOSITION_FLATTENED", None),
    ("SUPERPOSITION_CHILD_MISSING", None),
    ("TAYLOR_OFFSET_DROPPED", None),
]


@pytest.mark.parametrize("code", [c for c, _ in DOWNGRADES])
def test_downgrade_in_both_regimes(code, tmp_path):
    ref = proton_ref(1e8)
    factory = dict(DOWNGRADES)[code]
    if code == "SUPERPOSITION_CHILD_MISSING":
        lat = Lattice.from_sequence(
            "s", [Superposition(name="sp", length=0.2, children=[(0.0, "ghost")])], ref)
    elif code == "SUPERPOSITION_FLATTENED":
        child = Drift(name="child", length=0.2)
        lat = Lattice.from_sequence(
            "s", [Superposition(name="sp", length=0.2, children=[(0.0, "child")])], ref)
        lat.elements["child"] = child
    elif code == "TAYLOR_OFFSET_DROPPED":
        t = Taylor(name="ty", length=0.2)
        t.offset[4] = 1e-3
        lat = Lattice.from_sequence("t", [t], ref)
    else:
        lat = Lattice.from_sequence("x", [factory(ref)], ref)

    rep = Writer().write(lat, tmp_path / "p.impactx.py")          # permissive
    assert code in rep.codes(), f"{code} not recorded; got {sorted(rep.codes())}"
    if code in ("FOREIGN_DIRECTIVE",) or rep.problems():
        with pytest.raises(TranslationError) as exc:
            Writer().write(lat, tmp_path / "s.impactx.py", strict=True)
        assert exc.value.entry.code in rep.codes()


def test_allowlist_lets_a_known_downgrade_through(tmp_path):
    ref = proton_ref(1e8)
    lat = Lattice.from_sequence("x", [NCells(name="nc", length=0.7)], ref)
    rep = Writer().write(lat, tmp_path / "a.impactx.py")
    rep.allowlist = {"NCELLS_TO_DRIFT"}
    rep.raise_if(True)                       # does not raise
    assert rep.ok


def test_species_not_representable_in_inputs(tmp_path):
    ref = ReferenceParticle(species=species("deuteron"), kinetic_energy_eV=1e8)
    lat = Lattice.from_sequence("x", [Drift(name="d", length=1.0)], ref)
    rep = Writer().write(lat, tmp_path / "d.impactx.in", flavor="inputs")
    assert "SPECIES_NOT_REPRESENTABLE" in rep.codes()
    # the Python flavour is exact: set_charge_qe/set_mass_MeV take any species
    rep2 = Writer().write(lat, tmp_path / "d.impactx.py", flavor="python", strict=True)
    assert rep2.ok
    text = (tmp_path / "d.impactx.py").read_text()
    assert "ref.set_mass_MeV(1875.6129426)" in text


def test_marker_has_no_inputs_type_and_becomes_a_zero_drift(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("m", [Marker(name="mk")], ref)
    py = Writer().render(lat, flavor="python")
    inp = Writer().render(lat, flavor="inputs")
    assert "elements.Marker(name=\"mk\")" in py
    assert "mk.type = drift" in inp and "mk.ds = 0" in inp
    assert INPUTS_TYPE["Marker"] is None


def test_instrument_can_become_a_beam_monitor():
    ref = proton_ref()
    lat = Lattice.from_sequence("i", [Instrument(name="bpm", family="BPM")], ref)
    emits, rep = to_elements(lat, instrument="beam_monitor")
    assert emits[0].cls == "BeamMonitor"
    assert rep.ok
    emits2, rep2 = to_elements(lat)
    assert emits2[0].cls == "Marker"
    assert "INSTRUMENT_AS_MARKER" in rep2.codes()


def test_emit_rendering_round_trips_matrix():
    e = Emit("m1", "LinearMap", {"ds": 0.0, "R": [[1.0 if i == j else 0.0 for j in range(6)]
                                                  for i in range(6)]})
    e.params["R"][0][1] = 0.5
    assert "R=_map6([" in e.python_call()
    lines = e.inputs_lines()
    assert "m1.type = linear_map" in lines
    assert "m1.R12 = 0.5" in lines
    assert not any(".R11" in ln for ln in lines), "identity entries are omitted"


def test_write_rejects_bad_options(tmp_path):
    lat = demo_lattice()
    with pytest.raises(ValueError):
        Writer().write(lat, tmp_path / "x.impactx.py", flavor="fortran")
    with pytest.raises(ValueError):
        Writer().write(lat, tmp_path / "x.impactx.py", energy_mode="whatever")
    with pytest.raises(ValueError):
        Writer().write(lat, tmp_path / "x.impactx.py", nslice=0)


def test_python_deck_is_syntactically_valid(tmp_path):
    import ast

    for lat in (demo_lattice(), all_kinds_lattice(), linac_lattice()):
        text = Writer().render(lat)
        ast.parse(text)          # a runnable script, whatever ImpactX is installed
        assert "sim.lattice.extend(lattice)" in text
        # no tracking call is emitted: every mention of track_particles is a comment
        for line in text.splitlines():
            if "track_particles" in line:
                assert line.lstrip().startswith("#") or line.lstrip().startswith("Add a beam")


def test_inputs_deck_reads_back(tmp_path):
    lat = demo_lattice()
    p = tmp_path / "demo.impactx.in"
    Writer().write(lat, p, flavor="inputs")
    lat2, rep = Reader().read(p)
    assert lat2.total_length == pytest.approx(lat.total_length)
    kinds = [e.kind for e in (pl.element for pl in lat2.flatten())]
    assert "Bend" in kinds and "Solenoid" in kinds and "RFCavity" in kinds
    assert rep.ok


def test_registry_picks_the_flavour_from_the_file_name(tmp_path):
    """``write()`` through the format registry passes no ``flavor``: the suffix decides."""
    from lattix.formats.base import FORMATS, guess_format
    from lattix.formats.base import write as registry_write

    assert "impactx" in FORMATS, "add the impactx FormatSpec to lattix/formats/base.py"
    assert guess_format("x.impactx.in") == "impactx"
    assert guess_format("x.impactx.py") == "impactx"
    lat = demo_lattice()
    inp = tmp_path / "d.impactx.in"
    py = tmp_path / "d.impactx.py"
    registry_write(lat, inp)
    registry_write(lat, py)
    assert inp.read_text().startswith("###")
    assert py.read_text().startswith("#!/usr/bin/env python3")
    assert "lattice.elements = " in inp.read_text()
    assert "from impactx import ImpactX, elements" in py.read_text()
    # an explicit flavour still wins over the suffix
    other = tmp_path / "e.impactx.in"
    Writer().write(lat, other, flavor="python")
    assert other.read_text().startswith("#!/usr/bin/env python3")


def _source_codes(module) -> set[str]:
    """Every LOSSY/DROPPED code a module can emit: the RULES table plus the literal
    ``rep.lossy("X"`` / ``rep.dropped("X"`` call sites."""
    import inspect

    src = inspect.getsource(module)
    codes = set(re.findall(r'rep\.(?:lossy|dropped)\(\s*"([A-Z_]+)"', src))
    for rule in getattr(module, "Writer", type("_", (), {"RULES": {}})).RULES.values():
        if rule.cls in ("LOSSY", "DROPPED"):
            codes.add(rule.code)
    return codes


def test_every_writer_downgrade_code_is_exercised():
    """PLAN §5.3: every downgrade is tested in BOTH regimes — so no code may be missing
    from the DOWNGRADES table above."""
    from lattix.formats.impactx import writer as writer_mod

    tested = {c for c, _ in DOWNGRADES} | {"SPECIES_NOT_REPRESENTABLE"}
    missing = _source_codes(writer_mod) - tested
    assert not missing, f"untested writer downgrades: {sorted(missing)}"
