"""Bmad writer tests (PLAN §6 task 2.2): goldens, rules coverage, dual regimes.

Golden snapshots live in ``tests/golden/bmad``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_bmad_writer.py``.  The only
normalisation applied is the lattix version in the header comment.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.bmad import Reader, Writer
from lattix.formats.bmad.naming import MAX_NAME_LEN, is_valid, sanitize
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
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.rf import bmad_phi0
from lattix.ir.walk import propagate

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "bmad"
_VERSION_RE = re.compile(r"^! lattix \S+ from ", re.MULTILINE)


def _norm(text: str) -> str:
    return _VERSION_RE.sub("! lattix <version> from ", text)


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
        Quadrupole(name="qp", length=0.3, multipole=MagneticMultipoleP(Bn={1: 0.6 * b},
                                                                      tilt={1: 0.02})),
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
        Instrument(name="bp", length=0.05, family="BPM"),
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
    ty.offset[4] = 1e-3
    return lat


def linac_lattice() -> Lattice:
    """Three thin 300 kV gaps at -30 deg with a quad triplet, proton at 2.1 MeV."""
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


def test_golden_demo_nested_and_flat(tmp_path):
    lat = demo_lattice()
    assert_golden("demo.bmad", Writer().render(lat))
    assert_golden("demo_flat.bmad", Writer().render(lat, line_mode="flat"))


def test_golden_all_kinds():
    assert_golden("all_kinds.bmad", Writer().render(all_kinds_lattice()))


def test_golden_linac():
    assert_golden("linac.bmad", Writer().render(linac_lattice()))


@pytest.mark.oracle_madx
def test_golden_fodo_from_madx(tmp_path):
    pytest.importorskip("cpymad")
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "fodo.madx")
    assert_golden("fodo_from_madx.bmad", Writer().render(lat))


def test_nested_lines_keep_repeat_and_reflection():
    ref = proton_ref()
    lat = Lattice(name="root", reference=ref)
    lat.elements["a"] = Drift(name="a", length=1.0)
    lat.elements["b"] = Marker(name="b")
    lat.lines["sub"] = Line(name="sub", items=[LineItem(ref="a"), LineItem(ref="b")])
    lat.lines["root"] = Line(name="root", items=[
        LineItem(ref="sub"), LineItem(ref="sub", reverse=True), LineItem(ref="a", repeat=3)])
    lat.use = "root"
    text = Writer().render(lat)
    assert "sub: line = (a, b)" in text
    assert "root: line = (sub, -sub, 3*a)" in text
    assert "use, root" in text
    # flat mode spells every occurrence out
    flat = Writer().render(lat, line_mode="flat")
    assert "root: line = (a, b, b, a, a, a, a)" in flat


def test_long_line_wraps_on_commas():
    """Bmad's line limit is 500 characters; a trailing comma continues a statement."""
    ref = proton_ref()
    els = [Drift(name=f"drift_with_a_long_name_{i}", length=0.1) for i in range(60)]
    lat = Lattice.from_sequence("long", els, ref)
    text = Writer().render(lat)
    line_block = [ln for ln in text.splitlines() if ln.strip()]
    assert max(len(ln) for ln in line_block) < 500
    assert any(ln.rstrip().endswith(",") for ln in line_block)
    back, _ = Reader().read(_written(lat))
    assert len(back.flatten()) == 60


def _written(lat: Lattice, tmp: Path | None = None, **kw) -> Path:
    import tempfile

    d = tmp or Path(tempfile.mkdtemp(prefix="lattix_bmad_test_"))
    p = d / "out.bmad"
    Writer().write(lat, p, **kw)
    return p


# ---------------------------------------------------------------- conventions
def test_quadrupole_uses_the_local_signed_rigidity():
    lat = linac_lattice()
    text = Writer().render(lat)
    walk = propagate(lat)
    for p in walk:
        if p.element.kind != "Quadrupole":
            continue
        k1 = p.element.multipole.Bn[1] / p.ref_in.brho_signed
        assert f"{p.element.name}: quadrupole, l = 0.1, k1 = {k1:.15g}" in text
    # the reference energy rises, so the later quads have a smaller |k1|
    ks = [float(m) for m in re.findall(r"k1 = (-?[\d.eE+-]+)", text)]
    assert abs(ks[0]) > abs(ks[1]) > abs(ks[2])


def test_h_minus_flips_the_sign_of_every_normalized_strength():
    for name, expect in (("proton", 1.0), ("h-", -1.0)):
        ref = ReferenceParticle(species=species(name), kinetic_energy_eV=8e8)
        lat = Lattice.from_sequence(
            "s", [Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 3.0}))],
            ref)
        text = Writer().render(lat)
        k1 = float(re.search(r"k1 = (-?[\d.eE+-]+)", text).group(1))
        assert math.copysign(1.0, k1) == expect
    assert "parameter[particle] = #1H-" in text


def test_thin_cavity_is_a_zero_length_traveling_wave_gap():
    from lattix import write as write_deck

    lat = linac_lattice()
    rep = Writer().write(lat, _written(lat))
    text = _written(lat).read_text()
    assert "cav1: lcavity, l = 0, voltage = 300000, phi0 = " in text
    assert "cavity_type = traveling_wave" in text
    assert f"phi0 = {bmad_phi0(-math.pi / 6):.15g}" in text
    codes = rep.codes()
    assert "THIN_CAVITY_TRAVELING_WAVE" in codes
    # measured: a zero-length Bmad lcavity has no radial RF kick, an off-crest
    # TraceWin/HELIX gap does -> the note is EQUIVALENT because the registry-level
    # lattix.write() pre-pass (base.with_rf_focusing) writes the kick as a 'taylor'
    # lens right after the cavity
    assert "THIN_CAVITY_NO_RF_FOCUSING" in codes
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="lattix_bmad_test_")) / "linac.bmad"
    rep2 = write_deck(lat, out, "bmad", strict=True)                # strict passes: nothing is lost
    text2 = out.read_text()
    assert "THIN_GAP_RF_FOCUSING_AS_MATRIX" in rep2.codes()
    # HELIX gives 0.7617 / 0.6511 / 0.5649 for the three gaps (module docstring): the lenses carry them
    assert "cav1_rfdefocus: taylor, l = 0, {2: 0.761657298976769|1}, {4: 0.761657298976769|3}" in text2
    assert "cav3_rfdefocus: taylor, l = 0, {2: 0.56485399073776|1}" in text2


def test_on_crest_thin_cavity_has_no_rf_focusing_entry():
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    lat = Lattice.from_sequence(
        "c", [RFCavity(name="c1", length=0.0,
                       rf=RFP(voltage_V=3e5, phase_rad=0.0, frequency_Hz=162.5e6))], ref)
    rep = Writer().write(lat, _written(lat))
    assert "THIN_CAVITY_NO_RF_FOCUSING" not in rep.codes()


def test_sub_millimetre_cavity_is_refused_by_bmad_and_recorded():
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    lat = Lattice.from_sequence(
        "c", [RFCavity(name="c1", length=5e-4,
                       rf=RFP(voltage_V=3e5, phase_rad=0.0, frequency_Hz=162.5e6))], ref)
    p = _written(lat)
    rep = Writer().write(lat, p)
    assert "SHORT_CAVITY_ZERO_LENGTH" in rep.codes()
    assert "c1: lcavity, l = 0," in p.read_text()
    with pytest.raises(TranslationError):
        Writer().write(lat, p, strict=True)


def test_rbend_is_written_as_a_chord_with_shifted_pole_faces():
    ref = proton_ref()
    angle = 0.08
    arc = 0.8 * (angle / 2) / math.sin(angle / 2)
    lat = Lattice.from_sequence(
        "b", [Bend(name="br", length=arc,
                   bend=BendP(angle=angle, e1=0.01 + angle / 2, e2=0.02 + angle / 2, rect=True))],
        ref)
    text = Writer().render(lat)
    assert f"br: rbend, l = {0.8:.15g}" in text
    assert "e1 = 0.01" in text and "e2 = 0.02" in text
    back, _ = Reader().read(_written(lat))
    b = back.elements["br"]
    assert b.length == pytest.approx(arc)
    assert b.bend.e1 == pytest.approx(0.01 + angle / 2)
    assert b.bend.rect is True


def test_bend_roll_and_ref_tilt_are_different_attributes():
    ref = proton_ref()
    el = Bend(name="bd", length=1.0, bend=BendP(angle=0.1, tilt_ref=0.2),
              shift=BodyShiftP(tilt=0.05))
    text = Writer().render(Lattice.from_sequence("b", [el], ref))
    assert "ref_tilt = 0.2" in text
    assert "roll = 0.05" in text
    assert ", tilt = " not in text


def test_multipole_k0l_needs_a_status():
    """Measured: a Bmad multipole with k0l and the default k0l_status is fatal."""
    ref = proton_ref()
    el = Multipole(name="mp", multipole=MagneticMultipoleP(BnL={0: 0.01 * ref.brho_signed}))
    text = Writer().render(Lattice.from_sequence("m", [el], ref))
    assert "k0l = 0.01" in text
    assert "k0l_status = straight_reference" in text


def test_elliptical_aperture_writes_all_four_limits():
    """Bmad: 'FOR AN ELLIPTICAL APERTURE ALL FOUR X1_LIMIT ... MUST BE SET'."""
    ref = proton_ref()
    el = Quadrupole(name="q", length=0.3,
                    aperture=ApertureP(shape="ELLIPTICAL", x_limits=(-0.01, 0.02),
                                       y_limits=(-0.03, 0.04)))
    text = Writer().render(Lattice.from_sequence("a", [el], ref))
    for attr, value in (("x1_limit", 0.01), ("x2_limit", 0.02),
                        ("y1_limit", 0.03), ("y2_limit", 0.04)):
        assert f"{attr} = {value:.15g}" in text
    assert "aperture_type = elliptical" in text


def test_expressions_are_re_emitted_without_the_madx_colon_equals():
    ref = proton_ref()
    lat = Lattice.from_sequence(
        "e", [Quadrupole(name="q", length=0.3,
                         multipole=MagneticMultipoleP(Bn={1: 0.6 * ref.brho_signed}))], ref)
    lat.variables["kqf"] = Variable(value=0.6)
    lat.elements["q"].native["madx"] = {"k1_expr": "kqf"}
    text = Writer().render(lat)
    assert "kqf = 0.6" in text
    assert "k1 = kqf" in text
    assert ":=" not in text
    # a stale expression falls back to the number and says so
    lat.variables["kqf"] = Variable(value=0.7)
    rep_text = Writer().render(lat)
    assert "k1 = 0.6" in rep_text


# ---------------------------------------------------------------- downgrades
@pytest.mark.parametrize("element, code", [
    (NCells(name="nc", length=0.7), "NCELLS_TO_DRIFT"),
    (RFQCell(name="rq", length=0.05), "RFQ_TO_DRIFT"),
    (ReferenceChange(name="rc", energy_eV=1.2e8), "REFCHANGE_DROPPED"),
    (Directive(name="dv", card="ADJUST", role="matching"), "FOREIGN_DIRECTIVE"),
    (Kicker(name="ek", hkick=1e-3, electric=True), "EKICK_AS_MAGNETIC"),
    (FieldMap(name="fm", length=0.6, files=["m.edz"]), "FM_TO_DRIFT"),
    (Foil(name="fl"), "FOIL_THICKNESS_UNKNOWN"),
])
def test_every_downgrade_in_both_regimes(element, code):
    lat = Lattice.from_sequence("d", [element], proton_ref(1e8))
    rep = Writer().write(lat, _written(lat))
    assert code in rep.codes()
    with pytest.raises(TranslationError):
        Writer().write(lat, _written(lat), strict=True)


def test_fieldmap_with_a_known_gain_becomes_an_lcavity():
    ref = proton_ref(1e8)
    fm = FieldMap(name="fm", length=0.6, files=["m.edz"],
                  rf=RFP(phase_rad=-math.pi / 6, frequency_Hz=325e6, dE_ref_eV=1.5e6))
    lat = Lattice.from_sequence("f", [fm], ref)
    p = _written(lat)
    rep = Writer().write(lat, p)
    assert "FM_TO_CAVITY" in rep.codes()
    volt = 1.5e6 / math.cos(-math.pi / 6)
    assert f"voltage = {volt:.15g}" in p.read_text()
    back, _ = Reader().read(p)
    assert propagate(back)[-1].ref_out.kinetic_energy_eV == \
        pytest.approx(ref.kinetic_energy_eV + 1.5e6)


def test_freq_and_comment_role_directives_are_exact():
    lat = Lattice.from_sequence(
        "f", [Freq(name="fq", frequency_Hz=650e6),
              Directive(name="ttl", card="TITLE", role="title"),
              Drift(name="d", length=1.0)], proton_ref())
    p = _written(lat)
    rep = Writer().write(lat, p)
    assert rep.ok
    assert "fq" not in p.read_text()
    back, _ = Reader().read(p)
    assert back.total_length == pytest.approx(1.0)


def test_superposition_children_are_written_in_order():
    ref = proton_ref()
    child = Quadrupole(name="child", length=0.2,
                       multipole=MagneticMultipoleP(Bn={1: 0.5 * ref.brho_signed}))
    sup = Superposition(name="sup", length=0.2, children=[(0.0, "child")])
    lat = Lattice.from_sequence("s", [Drift(name="d", length=0.4), sup], ref)
    lat.elements["child"] = child
    p = _written(lat)
    rep = Writer().write(lat, p)
    assert "SUPERPOSITION_SUPERIMPOSED" in rep.codes()
    back, _ = Reader().read(p)
    assert [e.element.kind for e in back.flatten()] == ["Drift", "Quadrupole"]


def test_missing_superposition_child_and_empty_line_are_recorded():
    ref = proton_ref()
    lat = Lattice.from_sequence(
        "s", [Superposition(name="sup", length=0.2, children=[(0.0, "gone")])], ref)
    p = _written(lat)
    rep = Writer().write(lat, p)
    assert "SUPERPOSITION_CHILD_MISSING" in rep.codes()
    assert "EMPTY_LINE" in rep.codes()
    assert "s_nil: marker" in p.read_text()          # Bmad rejects an empty line
    # the flat layout takes the same route
    rep_flat = Writer().write(lat, p, line_mode="flat")
    assert "EMPTY_LINE" in rep_flat.codes()
    with pytest.raises(TranslationError):
        Writer().write(lat, p, strict=True)


def test_custom_species_uses_bmads_bare_mass_syntax():
    """Bmad keeps only two decimals of an ``@M<mass in u>`` species (measured)."""
    from lattix.ir.reference import Species

    ref = ReferenceParticle(species=Species(name="c12_6plus", mass_eV=11.1747e9, charge=6),
                            kinetic_energy_eV=1e9)
    lat = Lattice.from_sequence("ion", [Drift(name="d", length=1.0)], ref)
    p = _written(lat)
    rep = Writer().write(lat, p)
    text = p.read_text()
    assert "parameter[particle] = @M12.00++++++" in text
    assert "SPECIES_MASS_ROUNDED" in rep.codes()
    with pytest.raises(TranslationError):
        Writer().write(lat, p, strict=True)


def test_superimposed_source_element_is_only_commented(tmp_path):
    """Re-emitting ``name[superimpose] = T`` would place a second copy in Bmad."""
    deck = tmp_path / "sup.bmad"
    deck.write_text("parameter[particle] = proton\nparameter[e_tot] = 1738272000\n"
                    "parameter[geometry] = open\n"
                    "d1: drift, l = 1.0\nq1: quadrupole, l = 0.2, k1 = 0.5\n"
                    "mk: marker, superimpose, ref = q1, offset = 0.3\n"
                    "l1: line = (d1, q1, d1)\nuse, l1\n")
    lat, _ = Reader().read(deck)
    out = tmp_path / "again.bmad"
    Writer().write(lat, out)
    text = out.read_text()
    assert "[superimpose]" not in text
    assert "! lattix: mk was superimposed" in text
    again, _ = Reader().read(out)
    assert [e.element.name for e in again.flatten()] == \
        [e.element.name for e in lat.flatten()]
    assert again.total_length == pytest.approx(2.2)


# ---------------------------------------------------------------- names
def test_name_limit_is_measured_at_40():
    assert MAX_NAME_LEN == 40
    assert is_valid("q" + "a" * 39)
    assert not is_valid("q" + "a" * 40)
    assert not is_valid("quadrupole")            # a reserved element key
    assert sanitize("QF.1$") == "qf_1_"
    assert len(sanitize("x" * 60)) == 40


def test_illegal_names_are_sanitized_and_tagged_reversibly():
    ref = proton_ref()
    lat = Lattice.from_sequence("n", [Drift(name="D 1.a", length=1.0),
                                      Drift(name="D-1.a", length=1.0)], ref)
    p = _written(lat)
    text = p.read_text()
    assert "d_1_a: drift" in text
    assert 'name="D 1.a"' in text and 'name="D-1.a"' in text
    back, _ = Reader().read(p)
    originals = {e.provenance.original_name for e in back.elements.values()}
    assert {"D 1.a", "D-1.a"} <= originals


# ---------------------------------------------------------------- round trips
@pytest.mark.parametrize("factory", [demo_lattice, all_kinds_lattice, linac_lattice])
def test_write_read_write_is_a_fixed_point(factory):
    lat = factory()
    first = _written(lat)
    back, _ = Reader().read(first)
    text2 = Writer().render(back)
    third = Writer().render(Reader().read(_written(back))[0])
    assert text2 == third, "write o read is not a fixed point"
    assert back.total_length == pytest.approx(lat.total_length)


def test_golden_idempotence():
    for name in ("demo.bmad", "all_kinds.bmad", "linac.bmad"):
        path = GOLDEN / name
        lat, _ = Reader().read(path)
        again, _ = Reader().read(_written(lat))
        assert [e.kind for e in again.elements.values()] == \
               [e.kind for e in lat.elements.values()]
        assert again.total_length == pytest.approx(lat.total_length)
