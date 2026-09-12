"""MAD-X writer tests (PLAN §6 task 1.5): goldens, rules coverage, dual regimes.

Golden snapshots live in ``tests/golden/madx``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_madx_writer.py``.  The only
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
from lattix.formats.madx import Reader, Writer
from lattix.formats.madx.naming import MAX_NAME_LEN, sanitize
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
from lattix.ir.rf import madx_lag
from lattix.ir.walk import propagate
from lattix.testing import needs

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "madx"
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
    """FODO with two sbends, a solenoid, a thin cavity, a kicker and a collimator."""
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


def linac_lattice() -> Lattice:
    """Two 5 MV cavities with a quad before, between and after them."""
    ref = proton_ref(1e8)
    b = ref.brho_signed
    els = [
        Quadrupole(name="q1", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0 * b})),
        Drift(name="dr1", length=0.5),
        RFCavity(name="cav1", length=0.0, rf=RFP(voltage_V=5e6, phase_rad=0.0, frequency_Hz=325e6)),
        Drift(name="dr2", length=0.5),
        Quadrupole(name="q2", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0 * b})),
        Drift(name="dr3", length=0.5),
        RFCavity(name="cav2", length=0.0, rf=RFP(voltage_V=5e6, phase_rad=0.0, frequency_Hz=325e6)),
        Drift(name="dr4", length=0.5),
        Quadrupole(name="q3", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0 * b})),
    ]
    return Lattice.from_sequence("linac", els, ref)


def one_element(el, *, ref: ReferenceParticle | None = None, **kw) -> Lattice:
    return Lattice.from_sequence("s", [el], ref or proton_ref(), **kw)


def _k1_of(text: str, name: str) -> float:
    m = re.search(rf"^{name}: quadrupole,[^;]*?k1=([-\d.eE+]+)", text, re.MULTILINE)
    assert m, f"no k1 for {name} in\n{text}"
    return float(m.group(1))


# ------------------------------------------------------------------ rules table
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_every_rule_has_a_builder():
    w = Writer()
    for kind in w.RULES:
        if kind in ("Freq", "Directive"):
            continue
        assert hasattr(w, f"_def_{kind.lower()}"), kind


# ---------------------------------------------------------------------- goldens
@needs("madx")
@pytest.mark.parametrize("deck", ["fodo.madx", "transport.madx"])
def test_golden_sequence_mode(deck, tmp_path):
    lat, _ = Reader().read(DATA / deck)
    out = tmp_path / deck
    rep = Writer().write(lat, out)
    assert rep.ok
    assert_golden(deck, out.read_text())


def test_golden_demo_lattice(tmp_path):
    out = tmp_path / "demo.madx"
    rep = Writer().write(demo_lattice(), out)
    assert rep.ok                                   # only EQUIVALENT entries
    assert set(rep.codes()) == {"CONST_P0", "CONST_P0_DELTA_RIGIDITY"}
    assert_golden("demo_fodo.madx", out.read_text())


def test_golden_demo_lattice_constant_energy(tmp_path):
    out = tmp_path / "demo_const.madx"
    rep = Writer().write(demo_lattice(), out, energy_mode="constant")
    assert set(rep.codes()) == {"CONST_P0", "CONST_P0_START_RIGIDITY"}
    assert_golden("demo_fodo_constant.madx", out.read_text())


def test_golden_line_mode(tmp_path):
    lat = demo_lattice()
    lat.lines["cell"] = Line(name="cell", items=[LineItem(ref="QF"), LineItem(ref="D1")])
    lat.lines["demo"] = Line(name="demo", items=[
        LineItem(ref="cell", repeat=2), LineItem(ref="cell", reverse=True),
        LineItem(ref="B1"), LineItem(ref="END")])
    out = tmp_path / "demo_line.madx"
    Writer().write(lat, out, mode="line")
    text = out.read_text()
    assert "cell: line=(qf, d1);" in text          # names sanitized to lower case
    assert "demo: line=(2*cell, -cell, b1, end);" in text
    assert "qf: quadrupole" in text and "d1: drift, l=0.5;" in text   # drifts are explicit
    assert_golden("demo_line.madx", text)


# ------------------------------------------------------------------ conventions
def test_beam_line_per_species(tmp_path):
    for name, particle in (("proton", "proton"), ("electron", "electron"),
                           ("positron", "positron"), ("h-", "ion"), ("deuteron", "ion")):
        ref = ReferenceParticle(species=species(name), kinetic_energy_eV=8e8)
        out = tmp_path / f"{name}.madx"
        Writer().write(one_element(Drift(name="d", length=1.0), ref=ref), out)
        beam = next(ln for ln in out.read_text().splitlines() if ln.startswith("beam,"))
        assert f"particle={particle}," in beam
        assert f"charge={ref.species.charge}" in beam
        assert f"mass={ref.species.mass_eV / 1e9:.15g}" in beam
        assert f"energy={ref.total_energy_eV / 1e9:.15g}" in beam


def test_rbend_writes_the_chord_length(tmp_path):
    angle = 0.08
    arc = 0.8 * (angle / 2) / math.sin(angle / 2)
    lat = one_element(Bend(name="br", length=arc,
                           bend=BendP(angle=angle, e1=angle / 2, e2=angle / 2, rect=True)))
    out = tmp_path / "rb.madx"
    Writer().write(lat, out)
    line = next(ln for ln in out.read_text().splitlines() if ln.startswith("br:"))
    assert line.startswith("br: rbend, l=0.8, angle=0.08;"), line   # e1/e2 back to chord = 0


def test_rfcavity_units_and_lag(tmp_path):
    phase = -math.pi / 6
    lat = one_element(RFCavity(name="cav", length=0.0,
                               rf=RFP(voltage_V=1e6, phase_rad=phase, frequency_Hz=325e6)))
    out = tmp_path / "rf.madx"
    Writer().write(lat, out)
    line = next(ln for ln in out.read_text().splitlines() if ln.startswith("cav:"))
    assert "volt=1" in line and "freq=325" in line and "l=0" in line
    assert f"lag={madx_lag(phase):.15g}" in line


def test_normalized_strengths_use_the_signed_rigidity(tmp_path):
    for name in ("proton", "h-"):
        ref = ReferenceParticle(species=species(name), kinetic_energy_eV=8e8)
        lat = one_element(Quadrupole(name="q", length=0.3,
                                     multipole=MagneticMultipoleP(Bn={1: 2.5})), ref=ref)
        out = tmp_path / f"q_{name}.madx"
        Writer().write(lat, out)
        assert _k1_of(out.read_text(), "q") == pytest.approx(2.5 / ref.brho_signed, rel=1e-12)
    assert ref.brho_signed < 0                        # h- last: k1 came out negative


def test_thin_multipole_and_thick_multipoles(tmp_path):
    ref = proton_ref()
    b = ref.brho_signed
    lat = Lattice.from_sequence("s", [
        Multipole(name="mp", multipole=MagneticMultipoleP(BnL={1: 0.2 * b, 3: 0.5 * b},
                                                          BsL={1: 0.3 * b})),
        Sextupole(name="sx", length=0.2, multipole=MagneticMultipoleP(Bn={2: 2.0 * b})),
        Octupole(name="oc", length=0.2, multipole=MagneticMultipoleP(Bn={3: 4.0 * b},
                                                                    Bs={3: 1.0 * b})),
    ], ref)
    out = tmp_path / "mp.madx"
    Writer().write(lat, out)
    text = out.read_text()
    assert "mp: multipole, knl={0,0.2,0,0.5}, ksl={0,0.3};" in text
    assert "sx: sextupole, l=0.2, k2=2;" in text
    assert "oc: octupole, l=0.2, k3=4, k3s=1;" in text


def test_instrument_families_and_collimators(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("s", [
        Instrument(name="bpm", family="BPM"),
        Instrument(name="hm", family="HMONITOR"),
        Instrument(name="ins", family="SCREEN"),
        Collimator(name="rc", length=0.1, aperture=ApertureP.rect(0.02, 0.03)),
        Collimator(name="ec", length=0.1,
                   aperture=ApertureP(shape="ELLIPTICAL", x_limits=(-0.02, 0.02),
                                      y_limits=(-0.03, 0.03))),
    ], ref)
    out = tmp_path / "inst.madx"
    Writer().write(lat, out)
    text = out.read_text()
    assert "bpm: monitor, l=0;" in text
    assert "hm: hmonitor, l=0;" in text
    assert "ins: instrument, l=0;" in text
    assert "rc: rcollimator, l=0.1, xsize=0.02, ysize=0.03;" in text
    assert "ec: ecollimator, l=0.1, xsize=0.02, ysize=0.03;" in text


def test_taylor_and_aperture_and_ealign(tmp_path):
    m = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    m[0][1] = 0.5
    lat = Lattice.from_sequence("s", [
        Taylor(name="mx", matrix=m, offset=[1e-3, 0, 0, 0, 0, 0]),
        Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0}),
                   aperture=ApertureP.circle(0.02),
                   shift=BodyShiftP(x_offset=1e-3, tilt=2e-4)),
    ], proton_ref())
    out = tmp_path / "tay.madx"
    Writer().write(lat, out)
    text = out.read_text()
    assert "mx: matrix, l=0, rm12=0.5, kick1=0.001;" in text
    assert "apertype=ellipse, aperture={0.02,0.02}" in text
    assert "select, flag=error, range=q;" in text
    assert "ealign, dx=0.001, dy=0, ds=0, dphi=0, dtheta=0, dpsi=0.0002;" in text


# ------------------------------------------------------------------ energy mode
def test_energy_mode_local_vs_constant(tmp_path):
    lat = linac_lattice()
    placed = propagate(lat)
    refs = {p.name: p.ref_in for p in placed}
    b1, b2, b3 = (refs[n].brho_signed for n in ("q1", "q2", "q3"))
    assert b1 < b2 < b3                                       # two 5 MV cavities on crest

    loc = tmp_path / "loc.madx"
    rep_loc = Writer().write(lat, loc, energy_mode="local")
    con = tmp_path / "con.madx"
    rep_con = Writer().write(lat, con, energy_mode="constant")
    tl, tc = loc.read_text(), con.read_text()

    # constant mode: one rigidity, so identical k1 everywhere
    k_c = [_k1_of(tc, n) for n in ("q1", "q2", "q3")]
    assert k_c[0] == pytest.approx(k_c[1]) == pytest.approx(k_c[2])
    assert k_c[0] == pytest.approx(1.0)

    # local mode: k1 scales with the local rigidity ratio
    k_l = [_k1_of(tl, n) for n in ("q1", "q2", "q3")]
    assert k_l[1] == pytest.approx(k_l[0] * b1 / b2, rel=1e-12)
    assert k_l[2] == pytest.approx(k_l[0] * b1 / b3, rel=1e-12)
    assert k_l[0] > k_l[1] > k_l[2]

    assert set(rep_loc.codes()) == {"CONST_P0", "CONST_P0_LOCAL_RIGIDITY"}
    assert set(rep_con.codes()) == {"CONST_P0", "CONST_P0_START_RIGIDITY"}
    assert rep_loc.codes()["CONST_P0_LOCAL_RIGIDITY"] == 2      # one per accelerating element
    assert rep_loc.ok and rep_con.ok                            # EQUIVALENT never blocks


def test_non_accelerating_cavity_records_nothing(tmp_path):
    """A bunching cavity at φ = −π/2 gains nothing, so no CONST_P0 entry."""
    lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0),
                                      RFCavity(name="cav", length=0.0,
                                               rf=RFP(voltage_V=1.5e6, phase_rad=-math.pi / 2,
                                                      frequency_Hz=325e6))], proton_ref())
    rep = Writer().write(lat, tmp_path / "b.madx")
    assert rep.codes() == {}


def test_multi_rigidity_definition_is_flagged(tmp_path):
    ref = proton_ref(1e8)
    q = Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: ref.brho_signed}))
    lat = Lattice(name="s", reference=ref)
    lat.add_element(q)
    lat.add_element(RFCavity(name="cav", length=0.0,
                             rf=RFP(voltage_V=5e6, phase_rad=0.0, frequency_Hz=325e6)))
    lat.lines["s"] = Line(name="s", items=[LineItem(ref="q"), LineItem(ref="cav"),
                                           LineItem(ref="q")])
    lat.use = "s"
    rep = Writer().write(lat, tmp_path / "mr.madx", energy_mode="local")
    assert "MULTI_RIGIDITY_DEFINITION" in rep.codes()
    assert rep.ok


# ------------------------------------------------------------------ expressions
@needs("madx")
def test_expression_is_re_emitted(tmp_path):
    src = tmp_path / "src.madx"
    src.write_text("""
beam, particle=proton, energy=1.738272;
kqf := 0.31 + 0.29;
qf: quadrupole, l=0.3, k1:=kqf;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(src)
    out = tmp_path / "out.madx"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert "kqf := 0.31+0.29;" in text
    assert "qf: quadrupole, l=0.3, k1 := kqf;" in text
    assert "EXPRESSION_DROPPED" not in rep.codes()

    # the same deck with use_expressions=False writes plain numbers
    plain = tmp_path / "plain.madx"
    Writer().write(lat, plain, use_expressions=False)
    assert "kqf" not in plain.read_text()
    assert _k1_of(plain.read_text(), "qf") == pytest.approx(0.6)


@needs("madx")
def test_stale_expression_is_dropped_for_the_number(tmp_path):
    src = tmp_path / "src.madx"
    src.write_text("""
beam, particle=proton, energy=1.738272;
kqf := 0.6;
qf: quadrupole, l=0.3, k1:=kqf;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(src)
    lat.elements["qf"].multipole.Bn[1] *= 2.0          # IR edited: the expression is now stale
    out = tmp_path / "out.madx"
    rep = Writer().write(lat, out)
    assert "EXPRESSION_DROPPED" in rep.codes()
    assert _k1_of(out.read_text(), "qf") == pytest.approx(1.2)
    assert rep.ok                                     # EQUIVALENT, not a blocker


def test_variables_are_emitted_in_dependency_order(tmp_path):
    from lattix.ir.expr import Expression

    lat = one_element(Drift(name="d", length=1.0))
    lat.variables["kb"] = Variable(value=0.6, expression=Expression(text="ka * 2", deferred=True))
    lat.variables["ka"] = Variable(value=0.3)
    lat.variables["bad name"] = Variable(value=1.0)
    out = tmp_path / "v.madx"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert text.index("ka = 0.3;") < text.index("kb := ka * 2;")
    assert "VARIABLE_DROPPED" in rep.codes()
    assert "bad name" not in text


# ----------------------------------------------------------------------- naming
def test_names_are_sanitized_uniquified_and_tagged(tmp_path):
    from lattix.ir.elements import Provenance

    a = Drift(name="Q/F#1", length=0.5,
              provenance=Provenance(format="tracewin", original_name="Q/F#1",
                                    original_type="DRIFT"))
    b = Drift(name="Q F#1", length=0.5,
              provenance=Provenance(format="tracewin", original_name="Q F#1"))
    c = Drift(name="sequence", length=0.5)
    lat = Lattice.from_sequence("s", [a, b, c], proton_ref())
    out = tmp_path / "n.madx"
    Writer().write(lat, out, mode="line")
    text = out.read_text()
    assert 'q_f_1: drift, l=0.5;   ! lattix: name="Q/F#1" type="DRIFT"' in text
    assert 'q_f_1_2: drift, l=0.5;   ! lattix: name="Q F#1"' in text
    assert "sequence_x: drift" in text                    # reserved word avoided
    for line in text.splitlines():
        if ": drift" in line:
            assert re.match(r"^[a-z][a-z0-9_.]*:", line)


@needs("madx")
def test_sanitized_names_survive_a_read_back(tmp_path):
    from lattix.ir.elements import Provenance

    ref = proton_ref()
    a = Quadrupole(name="Q/F#1", length=0.5,
                   multipole=MagneticMultipoleP(Bn={1: ref.brho_signed}),
                   provenance=Provenance(format="tracewin", original_name="Q/F#1"))
    lat = Lattice.from_sequence("s", [a, Marker(name="end")], ref)
    out = tmp_path / "n.madx"
    Writer().write(lat, out)
    back, _ = Reader().read(out)
    assert back.elements["q_f_1"].provenance.original_name == "Q/F#1"
    assert back.elements["q_f_1"].gradient == pytest.approx(ref.brho_signed, rel=1e-14)


def test_sanitize_rules():
    assert sanitize("Q/F#1") == "q_f_1"
    assert sanitize("1st") == "e_1st"
    assert sanitize("drift") == "drift_x"
    assert len(sanitize("q" * 80)) == MAX_NAME_LEN


# ------------------------------------------------ dual regime: LOSSY and DROPPED
_LOSSY_CASES = [
    ("FM_TO_DRIFT", lambda: one_element(FieldMap(name="fm", length=0.4))),
    ("NCELLS_TO_DRIFT", lambda: one_element(NCells(name="nc", length=0.4))),
    ("RFQ_TO_DRIFT", lambda: one_element(RFQCell(name="rq", length=0.4))),
    ("FOIL_TO_MARKER", lambda: one_element(Foil(name="fo"))),
    ("PATCH_ATTR_DROPPED", lambda: one_element(Patch(name="pa", x_offset=1e-3, t_offset_s=1e-9))),
    ("FOREIGN_DIRECTIVE",
     lambda: one_element(Directive(name="dv", card="SET_ADV", args=["1"], role="matching"))),
    ("EKICK_AS_MAGNETIC",
     lambda: one_element(Kicker(name="ek", hkick=1e-3, electric=True))),
]


@pytest.mark.parametrize("code,build", _LOSSY_CASES, ids=[c for c, _ in _LOSSY_CASES])
def test_downgrade_permissive_then_strict(code, build, tmp_path):
    out = tmp_path / f"{code.lower()}.madx"
    rep = Writer().write(build(), out)
    assert code in rep.codes(), rep.summary()
    assert not rep.ok
    with pytest.raises(TranslationError) as exc:
        Writer().write(build(), out, strict=True)
    assert code in str(exc.value)


def test_superposition_is_flattened(tmp_path):
    ref = proton_ref()
    lat = Lattice(name="s", reference=ref)
    lat.add_element(Quadrupole(name="qa", length=0.2,
                               multipole=MagneticMultipoleP(Bn={1: ref.brho_signed})))
    lat.add_element(Solenoid(name="sa", length=0.2,
                             solenoid=SolenoidP(Bsol_T=0.1 * ref.brho_signed)))
    lat.add_element(Superposition(name="sup", length=0.4,
                                  children=[(0.0, "qa"), (0.2, "sa")]))
    lat.lines["s"] = Line(name="s", items=[LineItem(ref="sup")])
    lat.use = "s"
    out = tmp_path / "sup.madx"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert "SUPERPOSITION_FLATTENED" in rep.codes()
    assert "qa, at=0.1;" in text and "sa, at=0.3;" in text
    with pytest.raises(TranslationError):
        Writer().write(lat, tmp_path / "sup2.madx", strict=True)


def test_superposition_with_a_missing_child(tmp_path):
    ref = proton_ref()
    lat = Lattice(name="s", reference=ref)
    lat.add_element(Superposition(name="sup", length=0.4, children=[(0.0, "ghost")]))
    lat.lines["s"] = Line(name="s", items=[LineItem(ref="sup")])
    lat.use = "s"
    rep = Writer().write(lat, tmp_path / "g.madx")
    assert "SUPERPOSITION_CHILD_MISSING" in rep.codes()
    with pytest.raises(TranslationError):
        Writer().write(lat, tmp_path / "g2.madx", strict=True)


def test_line_mode_without_line_structure(tmp_path):
    ref = proton_ref()
    lat = Lattice(name="s", reference=ref)
    lat.add_element(Drift(name="d", length=1.0))
    lat.use = "d"                       # a bare element as the root: no Line to preserve
    out = tmp_path / "nl.madx"
    rep = Writer().write(lat, out, mode="line")
    assert "NO_LINE_STRUCTURE" in rep.codes()
    assert "s: line=(d);" in out.read_text()
    with pytest.raises(TranslationError):
        Writer().write(lat, tmp_path / "nl2.madx", mode="line", strict=True)


def test_directive_roles_that_are_only_comments(tmp_path):
    for role in sorted(Writer.COMMENT_ROLES):
        lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0),
                                          Directive(name="dv", card="LATTICE", args=["4"], role=role)],
                                    proton_ref())
        out = tmp_path / f"{role}.madx"
        rep = Writer().write(lat, out, strict=True)      # EXACT: strict must not raise
        assert rep.ok
        assert "! lattix directive: LATTICE 4" in out.read_text()


def test_freq_writes_nothing(tmp_path):
    lat = Lattice.from_sequence("s", [Freq(name="f1", frequency_Hz=162.5e6),
                                      Drift(name="d", length=1.0)], proton_ref())
    out = tmp_path / "f.madx"
    rep = Writer().write(lat, out, mode="line", strict=True)
    assert rep.ok
    assert "f1" not in out.read_text()


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="mode"):
        Writer().write(demo_lattice(), Path("/dev/null"), mode="nope")
    with pytest.raises(ValueError, match="energy_mode"):
        Writer().write(demo_lattice(), Path("/dev/null"), energy_mode="nope")


# ------------------------------------------------------------------ idempotence
@needs("madx")
@pytest.mark.parametrize("deck", ["fodo.madx", "transport.madx"])
def test_write_read_write_is_a_fixed_point(deck, tmp_path):
    """I-13: write(read(write(read(x)))) is byte-identical to write(read(x))."""
    r, w = Reader(), Writer()
    lat1, _ = r.read(DATA / deck)
    first = tmp_path / "a.madx"
    w.write(lat1, first)
    lat2, _ = r.read(first)
    second = tmp_path / "b.madx"
    w.write(lat2, second)
    assert second.read_bytes() == first.read_bytes()


@needs("madx")
def test_written_deck_reproduces_the_ir(tmp_path):
    """A demo IR written and read back keeps its lab fields (constant mode round trip)."""
    lat = demo_lattice()
    out = tmp_path / "demo.madx"
    Writer().write(lat, out, energy_mode="constant")
    back, _ = Reader().read(out)
    assert back.total_length == pytest.approx(lat.total_length, abs=1e-12)
    assert back.elements["qf"].gradient == pytest.approx(lat.elements["QF"].gradient, rel=1e-14)
    assert back.elements["sol"].solenoid.Bsol_T == pytest.approx(
        lat.elements["SOL"].solenoid.Bsol_T, rel=1e-14)
    assert back.elements["cav"].rf.phase_rad == pytest.approx(
        lat.elements["CAV"].rf.phase_rad, abs=1e-14)
    assert back.elements["b2"].bend.rect is True
    assert back.elements["b2"].length == pytest.approx(lat.elements["B2"].length, rel=1e-14)
    assert back.elements["coll"].aperture.shape == "RECTANGULAR"
    # the kicker sits after the cavity: written as kick·r and read back as kick/r (roundoff only)
    assert back.elements["cor"].hkick == pytest.approx(1e-3, rel=1e-12)
    assert back.elements["cor"].vkick == pytest.approx(-2e-3, rel=1e-12)


@needs("madx")
def test_element_attribute_expression_falls_back_to_the_number(tmp_path):
    """``sc->l`` is MAD-X syntax lattix's evaluator does not speak: number + ledger."""
    src = tmp_path / "attr.madx"
    src.write_text("""
beam, particle=proton, energy=1.738272;
sc: drift, l=0.5;
qf: quadrupole, l=0.3, k1 := sc->l * 2;
s: sequence, l=1.0, refer=centre; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(src)
    assert lat.elements["qf"].native["madx"]["k1_expr"] == "sc->l * 2"
    assert not any(n.startswith("__") for n in lat.variables)   # MAD-X temporaries filtered
    out = tmp_path / "out.madx"
    rep = Writer().write(lat, out)
    entry = next(e for e in rep.entries if e.code == "EXPRESSION_DROPPED")
    assert "cannot be evaluated" in entry.message
    assert _k1_of(out.read_text(), "qf") == pytest.approx(1.0)
    assert rep.ok


def test_aperture_on_a_drift_is_dropped_with_a_ledger_entry(tmp_path):
    """Measured: MAD-X 5.09.03 rejects apertype on drift/matrix/translation."""
    lat = one_element(Drift(name="d", length=1.0, aperture=ApertureP.circle(0.02)))
    out = tmp_path / "d.madx"
    rep = Writer().write(lat, out)
    assert "APERTURE_DROPPED" in rep.codes()
    assert "apertype" not in out.read_text()
    with pytest.raises(TranslationError):
        Writer().write(lat, tmp_path / "d2.madx", strict=True)
    # ... while a quadrupole keeps it
    lat2 = one_element(Quadrupole(name="q", length=0.3,
                                  multipole=MagneticMultipoleP(Bn={1: 1.0}),
                                  aperture=ApertureP.circle(0.02)))
    rep2 = Writer().write(lat2, tmp_path / "q.madx", strict=True)
    assert rep2.ok
    assert "apertype=ellipse" in (tmp_path / "q.madx").read_text()


def test_drift_body_shift_is_dropped_in_sequence_mode(tmp_path):
    lat = one_element(Drift(name="d", length=1.0, shift=BodyShiftP(x_offset=1e-3)))
    rep = Writer().write(lat, tmp_path / "ds.madx")
    assert "DRIFT_SHIFT_DROPPED" in rep.codes()
    with pytest.raises(TranslationError):
        Writer().write(lat, tmp_path / "ds2.madx", strict=True)


def test_patch_round_trip_through_madx_frame_cards(tmp_path):
    """A combined patch is written as one card per component (translation, yrotation, xrotation,
    srotation at the same position) and reads back as four patches with the same exit frame; a
    single-component patch is one card, exactly."""
    import numpy as np

    from lattix.ir import frame_survey

    combined = Patch(name="pa", x_offset=1e-3, y_offset=-2e-3, z_offset=5e-4, x_rot=0.01, y_rot=-0.02, tilt=0.3)
    lat = Lattice.from_sequence("s", [Drift(name="d1", length=1.0), combined, Drift(name="d2", length=1.0)],
                                proton_ref())
    out = tmp_path / "patch.madx"
    rep = Writer().write(lat, out)
    assert "PATCH_AS_CARDS" in rep.codes() and "PATCH_DROPPED" not in rep.codes(), rep.summary()
    text = out.read_text()
    for card in ("pa_t: translation, dx=", "pa_y: yrotation, angle=", "pa_x: xrotation, angle=",
                 "pa_s: srotation, angle="):
        assert card in text, text
    back, _ = Reader().read(out)
    assert [p.element.kind for p in back.flatten()].count("Patch") == 4
    want, got = frame_survey(lat.flatten())[-1].exit, frame_survey(back.flatten())[-1].exit
    np.testing.assert_allclose(got.V, want.V, atol=1e-12)
    np.testing.assert_allclose(got.W, want.W, atol=1e-12)
    single = Lattice.from_sequence("s", [Drift(name="d1", length=1.0), Patch(name="yaw", y_rot=0.02),
                                         Drift(name="d2", length=1.0)], proton_ref())
    rep1 = Writer().write(single, tmp_path / "one.madx")
    assert rep1.ok and "yaw: yrotation, angle=0.02;" in (tmp_path / "one.madx").read_text()
    back1, _ = Reader().read(tmp_path / "one.madx")
    assert back1.elements["yaw"].y_rot == 0.02
