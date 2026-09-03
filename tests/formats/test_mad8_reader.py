"""MAD8 flat-file reader tests (PLAN §6 task 2.3).

The inline decks mirror HELIX ``tests/io/test_mad8_parser.py`` one for one (so a
regression in either project is visible against the same fixtures) plus the cases the
lattix IR adds: kicks that survive, ``SQRT`` in a parameter, MAD8 class inheritance,
anonymous sub-lines and the LATTICE brackets as IR ``Directive``s.

The corpus anchors (``pytest.mark.corpus``) run against the private PIP-II decks named in
``$LATTIX_CORPUS_DIR/manifest.yaml``.
"""
from __future__ import annotations

import math

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.mad8 import Reader
from lattix.formats.mad8.reader import logical_lines, split_top_level, strip_comment
from lattix.ir.elements import Bend, Directive, Instrument, Kicker, Marker, Quadrupole
from lattix.testing import corpus_dir

BRHO = 4.881          # T·m, the value the PIP-II BTL/BAL decks declare

MINI = """
! minimal H- line
BRHO := 4.881
LQ := 0.2
KF := 1.5
D1: DRIFT, L=0.5
QF: QUADRUPOLE, L=LQ, K1=KF
QD: QUADRUPOLE, L=LQ, K1=-KF
CELL: LINE=(D1, QF, D1, QD)
TOP: LINE=(CELL)
RETURN
"""


def write(tmp_path, text, name="mini.lat"):
    p = tmp_path / name
    p.write_text(text)
    return p


def read(tmp_path, text, name="mini.lat", **kw):
    return Reader().read(write(tmp_path, text, name), **kw)


def kinds(lat):
    return [p.element.kind for p in lat.flatten()]


# --------------------------------------------------------------- dialect front-end
def test_logical_lines_join_continuations_and_strip_comments():
    text = "A: DRIFT, &\n  L=0.25   ! trailing comment\n! whole line\nB: MARKER\n"
    assert logical_lines(text) == [(1, "A: DRIFT,   L=0.25"), (4, "B: MARKER")]


def test_strip_comment_honours_quoted_strings():
    assert strip_comment('TITLE, "a ! b"  ! real comment') == 'TITLE, "a ! b"'


def test_split_top_level_respects_parentheses():
    assert split_top_level("A, (B, C), D") == ["A", "(B, C)", "D"]


def test_basic_parse_and_units(tmp_path):
    lat, rep = read(tmp_path, MINI)
    assert kinds(lat) == ["Drift", "Quadrupole", "Drift", "Quadrupole"]
    assert lat.flatten()[0].length == pytest.approx(0.5)        # metres, not HELIX mm
    assert lat.use == "TOP"
    assert lat.lines["TOP"].items[0].ref == "CELL"
    assert lat.meta["mad8_root_line"] == "TOP"
    assert rep.ok                                               # only EQUIVALENT entries


def test_continuation_and_comments(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, &
    L=0.25   ! trailing comment on the continued line
TOP: LINE=(D1, D1)
"""
    lat, _ = read(tmp_path, text)
    assert [p.length for p in lat.flatten()] == [0.25, 0.25]


def test_title_and_beam_and_use_are_commands(tmp_path):
    text = """TITLE, "PIP-II beam transfer line"
BRHO := 4.881
D1: DRIFT, L=0.5
SUB: LINE=(D1)
TOP: LINE=(SUB, SUB)
USE, TOP
"""
    lat, _ = read(tmp_path, text)
    assert lat.meta["mad8_title"] == "PIP-II beam transfer line"
    assert lat.use == "TOP"                       # the USE target, not the size heuristic


@pytest.mark.parametrize("use_card", ["USE, TOP", "USE TOP", "USE, PERIOD=TOP",
                                      'USE, LINE="TOP"'])
def test_use_card_spellings(tmp_path, use_card):
    text = f"""BRHO := 4.881
D1: DRIFT, L=0.5
BIG: LINE=(D1, D1, D1)
TOP: LINE=(D1)
{use_card}
"""
    lat, _ = read(tmp_path, text)
    assert lat.use == "TOP"                       # not BIG, the larger expansion


def test_unrecognised_statement_is_recorded(tmp_path):
    text = """BRHO := 4.881
this is not MAD8
D1: DRIFT, L=0.5
TOP: LINE=(D1)
"""
    lat, rep = read(tmp_path, text)
    assert "UNRECOGNISED_STATEMENT" in rep.codes()
    with pytest.raises(TranslationError, match="UNRECOGNISED_STATEMENT"):
        read(tmp_path, text, name="strict.lat", strict=True)


# ------------------------------------------------------------------- expressions
def test_deferred_params_and_attr_refs(tmp_path):
    """``NAME[L]`` element-attribute references and chained ``:=`` params must resolve —
    not silently coerce to zero (the ``_gf``-default trap)."""
    text = """BRHO := 4.881
A := 0.1
B := 2.0*A
D1: DRIFT, L=B
D2: DRIFT, L=0.899-D1[L]
TOP: LINE=(D1, D2)
"""
    lat, _ = read(tmp_path, text)
    assert [p.length for p in lat.flatten()] == pytest.approx([0.2, 0.699])
    assert lat.variables["a"].value == 0.1
    assert lat.variables["b"].expression.text == "2.0*A"
    assert lat.elements["D2"].expressions["length"].text == "0.899-D1[L]"


def test_sqrt_and_friends_in_parameters(tmp_path):
    """The BAL deck's ``BRHO := P0/C*1e11`` chain — where HELIX's resolver stops
    (docs/corpus.md)."""
    text = """MASS_HMINUS := 9.39294E2
E0 := 8.0E2
P0 := SQRT(E0*(2.0*MASS_HMINUS+E0))
C := 2.997925E10
BRHO := P0/C*1.0E8
SC := ABS(SIN(0.0)+COS(0.0))
D1: DRIFT, L=0.5*SC
TOP: LINE=(D1)
"""
    lat, _ = read(tmp_path, text)
    p0 = math.sqrt(800.0 * (2 * 939.294 + 800.0))
    assert lat.variables["p0"].value == pytest.approx(p0, rel=1e-12)
    assert lat.reference.brho_abs == pytest.approx(p0 / 2.997925e10 * 1e8, rel=1e-12)
    assert lat.flatten()[0].length == pytest.approx(0.5)


def test_file_parameter_shadows_a_builtin_constant(tmp_path):
    """MAD8 scoping: a deck's own ``E``/``PI`` wins over lattix's constant table, at
    every nesting level."""
    text = """BRHO := 4.881
E := 3.0
F := 2.0*E
D1: DRIFT, L=F
TOP: LINE=(D1)
"""
    lat, _ = read(tmp_path, text)
    assert lat.flatten()[0].length == pytest.approx(6.0)


def test_circular_parameter_is_reported(tmp_path):
    text = """BRHO := 4.881
A := B
B := A
D1: DRIFT, L=A
TOP: LINE=(D1)
"""
    lat, rep = read(tmp_path, text)
    assert "UNRESOLVED_ATTRIBUTE" in rep.codes()
    assert lat.flatten()[0].length == 0.0
    with pytest.raises(TranslationError):
        read(tmp_path, text, name="s.lat", strict=True)


def test_sci_notation_not_identifier(tmp_path):
    lat, _ = read(tmp_path, "BRHO := 4.881\nD1: DRIFT, L=1E-03\nTOP: LINE=(D1)\n")
    assert lat.flatten()[0].length == pytest.approx(1e-3)


def test_apostrophe_param_recorded_and_skipped(tmp_path):
    text = """BRHO := 4.881
QX' := -9.129452648695
D1: DRIFT, L=0.5
TOP: LINE=(D1)
"""
    lat, rep = read(tmp_path, text)
    assert lat.flatten()[0].length == pytest.approx(0.5)
    assert rep.codes()["UNPARSEABLE_IDENTIFIER"] == 1
    assert lat.meta["mad8_unparseable_params"] == {"QX'": "-9.129452648695"}
    assert "qx'" not in lat.variables
    with pytest.raises(TranslationError, match="UNPARSEABLE_IDENTIFIER"):
        read(tmp_path, text, name="s.lat", strict=True)


def test_negative_drift_survives(tmp_path):
    """MAD overlap-bookkeeping drifts (BTL: ``DBV3NT`` = −204.288 mm)."""
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
DN: DRIFT, L=-0.204288
TOP: LINE=(D1, DN, D1)
"""
    lat, _ = read(tmp_path, text)
    assert lat.flatten()[1].length == pytest.approx(-0.204288)
    assert lat.total_length == pytest.approx(0.795712)


# ------------------------------------------------------------------------- lines
def test_line_reversal_and_repetition(tmp_path):
    text = """BRHO := 4.881
DA: DRIFT, L=0.1
DB: DRIFT, L=0.2
SUB: LINE=(DA, DB)
TOP: LINE=(2*DA, -SUB)
"""
    lat, _ = read(tmp_path, text)
    assert [p.length for p in lat.flatten()] == pytest.approx([0.1, 0.1, 0.2, 0.1])
    top = lat.lines["TOP"].items
    assert (top[0].ref, top[0].repeat, top[0].reverse) == ("DA", 2, False)
    assert (top[1].ref, top[1].repeat, top[1].reverse) == ("SUB", 1, True)


def test_anonymous_sublines(tmp_path):
    text = """BRHO := 4.881
DA: DRIFT, L=0.1
DB: DRIFT, L=0.2
TOP: LINE=(DA, 2*(DA, DB))
"""
    lat, _ = read(tmp_path, text)
    assert [p.length for p in lat.flatten()] == pytest.approx([0.1, 0.1, 0.2, 0.1, 0.2])


def test_root_line_is_the_largest_unreferenced(tmp_path):
    text = """BRHO := 4.881
DA: DRIFT, L=0.1
SMALL: LINE=(DA)
BIG: LINE=(DA, DA, DA)
"""
    lat, rep = read(tmp_path, text)
    assert lat.use == "BIG"
    assert "AMBIGUOUS_ROOT_LINE" in rep.codes()


def test_undefined_reference_is_reported(tmp_path):
    text = """BRHO := 4.881
DA: DRIFT, L=0.1
TOP: LINE=(DA, NOSUCH)
"""
    lat, rep = read(tmp_path, text)
    assert [p.name for p in lat.flatten()] == ["DA"]
    assert "UNDEFINED_REFERENCE" in rep.codes()
    with pytest.raises(TranslationError, match="UNDEFINED_REFERENCE"):
        read(tmp_path, text, name="s.lat", strict=True)


def test_class_inheritance(tmp_path):
    text = """BRHO := 4.881
QBASE: QUADRUPOLE, L=0.2, K1=1.0
QF: QBASE
QD: QBASE, K1=-1.0
TOP: LINE=(QF, QD)
"""
    lat, _ = read(tmp_path, text)
    qf, qd = lat.elements["QF"], lat.elements["QD"]
    assert isinstance(qf, Quadrupole) and qf.length == pytest.approx(0.2)
    assert qf.multipole.Bn[1] == pytest.approx(-1.0 * BRHO, rel=1e-9)
    assert qd.multipole.Bn[1] == pytest.approx(+1.0 * BRHO, rel=1e-9)


# ---------------------------------------------------------------------- rigidity
def test_charge_sign_hminus_default(tmp_path):
    """H⁻ (default): G = sign(q)·K1·Bρ = −K1·Bρ — the legacy ``mad2tw`` convention
    (btl.dat header: ``variable mad2tw -4.8829``)."""
    lat, _ = read(tmp_path, MINI)
    assert lat.elements["QF"].multipole.Bn[1] == pytest.approx(-1.5 * BRHO, rel=1e-9)
    assert lat.reference.species.name == "h-"
    assert lat.reference.kinetic_energy_eV / 1e6 == pytest.approx(799.52, abs=0.2)


def test_charge_sign_proton_flips(tmp_path):
    lat, _ = read(tmp_path, MINI, species="proton")
    assert lat.elements["QF"].multipole.Bn[1] == pytest.approx(+1.5 * BRHO, rel=1e-9)


def test_no_rigidity_is_a_hard_error(tmp_path):
    text = """D1: DRIFT, L=0.5
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(D1, Q1)
"""
    with pytest.raises(ValueError, match="rigidity"):
        read(tmp_path, text)


def test_brho_argument_wins_over_the_file(tmp_path):
    lat, rep = read(tmp_path, MINI, brho=9.762)
    assert lat.reference.brho_abs == pytest.approx(9.762, rel=1e-12)
    assert lat.elements["QF"].multipole.Bn[1] == pytest.approx(-1.5 * 9.762, rel=1e-9)
    entry = next(e for e in rep.entries if e.code == "RIGIDITY_FROM_BRHO")
    assert entry.details["brho"] == pytest.approx(9.762)


def test_brho_argument_fallback_without_a_file_value(tmp_path):
    text = """D1: DRIFT, L=0.5
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(D1, Q1)
"""
    lat, rep = read(tmp_path, text, brho=4.881)
    assert lat.elements["Q1"].multipole.Bn[1] == pytest.approx(-4.881, rel=1e-9)
    # Bρ = 4.881 T·m inverted with the physical H⁻ ion mass (939.294 MeV) gives
    # 799.52 MeV kinetic; the "800 MeV" label pairs with 4.881 only under the
    # proton-mass convention, evidence that the BTL lineage treats H⁻ as a bare proton.
    assert lat.reference.kinetic_energy_eV / 1e6 == pytest.approx(799.52, abs=0.2)
    assert "RIGIDITY_FROM_BRHO" in rep.codes()


def test_beam_statement(tmp_path):
    text = """BEAM, PARTICLE=PROTON, ENERGY=1.938272
D1: DRIFT, L=0.5
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(D1, Q1)
"""
    lat, rep = read(tmp_path, text)
    assert lat.reference.species.name == "proton"
    assert lat.reference.kinetic_energy_eV / 1e6 == pytest.approx(1000.0, abs=0.5)
    assert lat.elements["Q1"].multipole.Bn[1] > 0            # proton: +K1·Bρ
    assert "RIGIDITY_FROM_BRHO" not in rep.codes()


def test_beam_mass_and_charge_identify_hminus(tmp_path):
    """What this package's own writer emits for H⁻ (MAD8 has no HMINUS name)."""
    text = """BEAM, MASS=0.93929408606, CHARGE=-1, ENERGY=1.73881631804175
D1: DRIFT, L=0.5
TOP: LINE=(D1)
"""
    lat, _ = read(tmp_path, text)
    assert lat.reference.species.name == "h-"
    assert lat.reference.kinetic_energy_eV / 1e6 == pytest.approx(799.52, abs=0.1)


def test_brho_beats_beam_and_the_conflict_is_recorded(tmp_path):
    text = """BEAM, PARTICLE=PROTON, ENERGY=1.938272
BRHO := 4.881
D1: DRIFT, L=0.5
TOP: LINE=(D1)
"""
    lat, rep = read(tmp_path, text)
    assert lat.reference.species.name == "proton"          # species still from BEAM
    assert lat.reference.brho_abs == pytest.approx(4.881, rel=1e-12)
    assert "RIGIDITY_CONFLICT" in rep.codes()


# ------------------------------------------------------------- element mapping
def test_kicker_keeps_the_kick_and_the_body_length(tmp_path):
    """The IR gain over HELIX, which wrote Marker + body Drift and dropped the kick."""
    text = """BRHO := 4.881
K1: HKICKER, L=0.06, KICK=1.5E-3
K2: VKICKER, L=0.06, KICK=-2.0E-3
K3: KICKER, L=0.1, HKICK=1E-3, VKICK=2E-3
TOP: LINE=(K1, K2, K3)
"""
    lat, rep = read(tmp_path, text)
    k1, k2, k3 = (lat.elements[n] for n in ("K1", "K2", "K3"))
    assert all(isinstance(k, Kicker) for k in (k1, k2, k3))
    assert (k1.hkick, k1.vkick, k1.length) == pytest.approx((1.5e-3, 0.0, 0.06))
    assert (k2.hkick, k2.vkick) == pytest.approx((0.0, -2.0e-3))
    assert (k3.hkick, k3.vkick, k3.length) == pytest.approx((1e-3, 2e-3, 0.1))
    assert lat.total_length == pytest.approx(0.22)
    assert rep.ok


def test_monitor_vs_hvmonitor(tmp_path):
    text = """BRHO := 4.881
M1: MONITOR
HP: HMONITOR
VP: VMONITOR
IN: INSTRUMENT, L=0.1
TOP: LINE=(M1, HP, VP, IN)
"""
    lat, _ = read(tmp_path, text)
    fams = [lat.elements[n].family for n in ("M1", "HP", "VP", "IN")]
    assert fams == ["MONITOR", "BPM", "BPM", "INSTRUMENT"]
    assert all(isinstance(lat.elements[n], Instrument) for n in ("M1", "HP", "VP", "IN"))
    assert lat.elements["HP"].native["mad8"]["type"] == "hmonitor"    # exact round trip


def test_skew_quad_tilt(tmp_path):
    text = """BRHO := 4.881
QS: QUADRUPOLE, L=0.2, K1=0.0, TILT=0.7853981634
QB: QUADRUPOLE, L=0.2, K1=1.0, TILT
TOP: LINE=(QS, QB)
"""
    lat, _ = read(tmp_path, text)
    assert lat.elements["QS"].multipole.tilt[1] == pytest.approx(math.pi / 4, rel=1e-9)
    assert lat.elements["QB"].multipole.tilt[1] == pytest.approx(math.pi / 4)   # bare flag


def test_rbend_length_is_the_arc_and_pole_faces_shift(tmp_path):
    """MAD8 has no ``OPTION, RBARC``: L is the arc (anchored on the PIP-II BTL export,
    where ``LBA = 2.408`` with ``BAANG = 0.11455892`` gives ρ = 21.019751 m)."""
    text = """BRHO := 4.881
LBA := 2.408
BAANG := 0.11455892
BA: RBEND, L=LBA, ANGLE=BAANG
TOP: LINE=(BA)
"""
    lat, _ = read(tmp_path, text)
    b = lat.elements["BA"]
    assert isinstance(b, Bend) and b.bend.rect
    assert b.length == pytest.approx(2.408)
    assert b.length / b.bend.angle == pytest.approx(21.019751, abs=1e-6)
    assert b.bend.e1 == pytest.approx(0.11455892 / 2)
    assert b.bend.e2 == pytest.approx(0.11455892 / 2)


def test_sbend_edges_fint_hgap_and_k1(tmp_path):
    text = """BRHO := 4.881
B: SBEND, L=1.0, ANGLE=0.1, E1=0.02, E2=0.03, FINT=0.5, HGAP=0.025, K1=0.4
TOP: LINE=(B)
"""
    lat, _ = read(tmp_path, text)
    b = lat.elements["B"]
    assert not b.bend.rect
    assert (b.bend.e1, b.bend.e2) == pytest.approx((0.02, 0.03))
    assert (b.bend.edge_int1, b.bend.hgap) == pytest.approx((0.5, 0.025))
    assert b.multipole.Bn[1] == pytest.approx(0.4 * -BRHO, rel=1e-9)


def test_vertical_bend_keeps_the_signed_tilt(tmp_path):
    """MAD8 spells a vertical bend ``TILT = ±π/2``; the IR keeps the sign (HELIX had to
    fold it into an ``hv`` flag with ρ > 0)."""
    text = """BRHO := 4.881
BV: RBEND, L=1.05, ANGLE=0.0416, TILT=-1.570796327
TOP: LINE=(BV)
"""
    lat, _ = read(tmp_path, text)
    b = lat.elements["BV"]
    assert b.bend.tilt_ref == pytest.approx(-math.pi / 2, abs=1e-9)
    assert b.bend.angle == pytest.approx(0.0416)
    assert b.length == pytest.approx(1.05)


def test_zero_angle_rbend_stays_a_bend_of_the_same_length(tmp_path):
    lat, _ = read(tmp_path, "BRHO := 4.881\nB0: RBEND, L=3.05, ANGLE=0.0\nTOP: LINE=(B0)\n")
    b = lat.elements["B0"]
    assert isinstance(b, Bend) and b.bend.angle == 0.0
    assert lat.total_length == pytest.approx(3.05)


def test_solenoid_sextupole_octupole_multipole_rfcavity(tmp_path):
    text = """BRHO := 4.881
S: SOLENOID, L=0.4, KS=0.3
SX: SEXTUPOLE, L=0.2, K2=1.5
OC: OCTUPOLE, L=0.2, K3=2.5
MP: MULTIPOLE, K2L=0.4, T2=0.1, LRAD=0.05
CAV: RFCAVITY, L=0.0, VOLT=1.0, LAG=0.25, FREQ=325.0, HARMON=7
TOP: LINE=(S, SX, OC, MP, CAV)
"""
    lat, rep = read(tmp_path, text)
    assert lat.elements["S"].solenoid.Bsol_T == pytest.approx(0.3 * -BRHO, rel=1e-9)
    assert lat.elements["SX"].multipole.Bn[2] == pytest.approx(1.5 * -BRHO, rel=1e-9)
    assert lat.elements["OC"].multipole.Bn[3] == pytest.approx(2.5 * -BRHO, rel=1e-9)
    assert lat.elements["MP"].multipole.BnL[2] == pytest.approx(0.4 * -BRHO, rel=1e-9)
    assert lat.elements["MP"].multipole.tilt[2] == pytest.approx(0.1)
    cav = lat.elements["CAV"]
    assert cav.rf.voltage_V == pytest.approx(1e6)          # MAD8 VOLT is MV
    assert cav.rf.frequency_Hz == pytest.approx(325e6)     # MAD8 FREQ is MHz
    assert cav.rf.phase_rad == pytest.approx(0.0, abs=1e-12)   # lag 1/4 = crest
    assert cav.native["mad8"]["harmon"] == 7
    assert rep.ok


def test_collimators(tmp_path):
    text = """BRHO := 4.881
RC: RCOLLIMATOR, L=0.1, XSIZE=0.02, YSIZE=0.03
EC: ECOLLIMATOR, L=0.1, XSIZE=0.02, YSIZE=0.02
TOP: LINE=(RC, EC)
"""
    lat, _ = read(tmp_path, text)
    assert lat.elements["RC"].aperture.shape == "RECTANGULAR"
    assert lat.elements["RC"].aperture.half_y == pytest.approx(0.03)
    assert lat.elements["EC"].aperture.shape == "ELLIPTICAL"


def test_aperture_attribute_becomes_a_circle(tmp_path):
    lat, _ = read(tmp_path, "BRHO := 4.881\nD: DRIFT, L=0.5, APERTURE=0.0254\nTOP: LINE=(D)\n")
    assert lat.elements["D"].aperture.half_x == pytest.approx(0.0254)


def test_unknown_type_is_a_marker_of_the_same_length(tmp_path):
    text = """BRHO := 4.881
X1: OCTOPUS, L=0.5
TOP: LINE=(X1)
"""
    lat, rep = read(tmp_path, text)
    el = lat.elements["X1"]
    assert isinstance(el, Marker) and el.length == pytest.approx(0.5)
    assert el.native["mad8"]["type"] == "octopus"
    assert rep.codes()["UNSUPPORTED_MAD8_TYPE"] == 1
    with pytest.raises(TranslationError, match="UNSUPPORTED_MAD8_TYPE"):
        read(tmp_path, text, name="s.lat", strict=True)


def test_elseparator_becomes_an_electric_kicker(tmp_path):
    text = """BEAM, PARTICLE=PROTON, ENERGY=1.938272
SEP: ELSEPARATOR, L=1.0, E=0.5
TOP: LINE=(SEP)
"""
    lat, rep = read(tmp_path, text)
    sep = lat.elements["SEP"]
    assert isinstance(sep, Kicker) and sep.electric and sep.length == pytest.approx(1.0)
    ref = lat.reference
    assert sep.vkick == pytest.approx(0.5e6 * 1.0 / (ref.beta * ref.pc_eV), rel=1e-12)
    assert "ESEPARATOR_AS_KICKER" in rep.codes()


# ---------------------------------------------------------------- periodicity
FODO_PERIODS = """BRHO := 4.881
D1: DRIFT, L=0.5
QF: QUADRUPOLE, L=0.2, K1=1.5
QD: QUADRUPOLE, L=0.2, K1=-1.5
CELL1: LINE=(D1, QF, D1, QD)
CELL2: LINE=(D1, QF, D1, QD)
CELL3: LINE=(D1, QF, D1, QD)
FODO: LINE=(CELL1, CELL2, CELL3)
TOP: LINE=(FODO)
"""


def test_auto_periods_bracket_the_repeating_cells(tmp_path):
    lat, _ = read(tmp_path, FODO_PERIODS)
    periods = lat.meta["mad8_periods"]
    assert len(periods) == 1
    assert periods[0]["n_repeats"] == 3
    assert periods[0]["n_sig"] == 4                 # 2 drifts + 2 quads per cell
    flat = lat.flatten()
    opens = [p for p in flat if isinstance(p.element, Directive) and p.element.role == "period_start"]
    closes = [p for p in flat if isinstance(p.element, Directive) and p.element.role == "period_end"]
    assert len(opens) == len(closes) == 1
    assert opens[0].element.card == "LATTICE" and opens[0].element.args == ["4", "0"]
    assert closes[0].element.card == "LATTICE_END"
    # the bracket wraps the three cells, i.e. all 12 physical elements
    assert opens[0].index == 0 and closes[0].index == len(flat) - 1


def test_auto_periods_off(tmp_path):
    lat, _ = read(tmp_path, FODO_PERIODS, auto_periods=False)
    assert lat.meta["mad8_periods"] == []
    assert not any(isinstance(p.element, Directive) for p in lat.flatten())


def test_a_cell_without_focusing_is_not_a_period(tmp_path):
    """HELIX's rule: matching wire-scanner wrapper LINEs are not transport periods."""
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
WS: MONITOR
W1: LINE=(D1, WS)
W2: LINE=(D1, WS)
SEC: LINE=(W1, W2)
TOP: LINE=(SEC)
"""
    lat, _ = read(tmp_path, text)
    assert lat.meta["mad8_periods"] == []


def test_periods_do_not_change_the_geometry(tmp_path):
    a, _ = read(tmp_path, FODO_PERIODS, name="a.lat")
    b, _ = read(tmp_path, FODO_PERIODS, name="b.lat", auto_periods=False)
    assert a.total_length == pytest.approx(b.total_length)


# ------------------------------------------------- dual regime (reader downgrades)
DUAL_DECKS = {
    "UNRECOGNISED_STATEMENT": "BRHO := 4.881\nthis is not MAD8\nD: DRIFT, L=1\nT: LINE=(D)\n",
    "CALL_NOT_FOLLOWED": 'BRHO := 4.881\nCALL, FILENAME="more.lat"\nD: DRIFT, L=1\nT: LINE=(D)\n',
    "UNPARSEABLE_IDENTIFIER": "BRHO := 4.881\nQX' := 1.0\nD: DRIFT, L=1\nT: LINE=(D)\n",
    "UNRESOLVED_VARIABLE": "BRHO := 4.881\nA := NOSUCH\nD: DRIFT, L=1\nT: LINE=(D)\n",
    "UNDEFINED_REFERENCE": "BRHO := 4.881\nD: DRIFT, L=1\nT: LINE=(D, GHOST)\n",
    "UNRESOLVED_ATTRIBUTE": "BRHO := 4.881\nD: DRIFT, L=NOSUCH\nT: LINE=(D)\n",
    "UNSUPPORTED_MAD8_TYPE": "BRHO := 4.881\nX: OCTOPUS, L=1\nT: LINE=(X)\n",
    "BEND_ATTR_DROPPED": "BRHO := 4.881\nB: SBEND, L=1, ANGLE=0.1, H1=0.2\nT: LINE=(B)\n",
    "KICKER_TILT_DROPPED": "BRHO := 4.881\nK: KICKER, L=0.1, HKICK=1E-3, TILT=0.3\nT: LINE=(K)\n",
}


@pytest.mark.parametrize("code", sorted(DUAL_DECKS))
def test_dual_regime_every_reader_downgrade(code, tmp_path):
    lat, rep = read(tmp_path, DUAL_DECKS[code], name=f"{code.lower()}.lat")
    assert code in rep.codes(), rep.summary()
    assert lat.use is not None
    with pytest.raises(TranslationError, match=code):
        read(tmp_path, DUAL_DECKS[code], name=f"{code.lower()}_s.lat", strict=True)


def test_species_assumed_for_an_unknown_beam_particle(tmp_path):
    text = """BEAM, PARTICLE=MUON, ENERGY=1.0
D: DRIFT, L=1
T: LINE=(D)
"""
    lat, rep = read(tmp_path, text, brho=4.881)
    assert "SPECIES_ASSUMED" in rep.codes()
    assert lat.reference.species.name == "h-"          # the species= fallback


# ------------------------------------------------------- format registry wiring
def test_registry_reads_lat_and_flat(tmp_path):
    from lattix.formats import guess_format, translate
    from lattix.formats import read as registry_read

    for name in ("deck.lat", "DECK.FLAT"):
        assert guess_format(tmp_path / name) == "mad8"
        p = tmp_path / name
        p.write_text(MINI)
        lat, rep = registry_read(p)
        assert lat.total_length == pytest.approx(1.4)
        assert rep.source_format == "mad8"

    out = tmp_path / "out.lat"
    rep = translate(tmp_path / "deck.lat", out)
    assert out.exists() and rep.target_format == "mad8"


# ------------------------------------------------------------------- corpus
pytest_corpus = pytest.mark.corpus


def _entry(deck_id: str) -> str:
    root = corpus_dir()
    if root is None or not (root / "manifest.yaml").is_file():
        pytest.skip("LATTIX_CORPUS_DIR is not set or has no manifest.yaml")
    from lattix.corpus import load_manifest, resolve_path

    for e in load_manifest(root):
        if e["id"] == deck_id:
            path = resolve_path(root, e)
            if not path.is_file():
                pytest.skip(f"{deck_id} is listed but missing on disk")
            return str(path)
    pytest.skip(f"{deck_id} is not in the corpus manifest")
    raise AssertionError                                        # pragma: no cover


@pytest_corpus
def test_corpus_btl_2025_anchor():
    """The PIP-II BTL: 307.969918 m, H⁻ at 799.52 MeV from ``BRHO := 4.881``, and the ten
    periodicity brackets the hand-checked TraceWin export declares."""
    lat, rep = Reader().read(_entry("helix-pipii-root/btl2025v0703.lat"))
    assert lat.total_length == pytest.approx(307.969918, abs=1e-5)
    assert lat.reference.species.name == "h-"
    assert lat.reference.brho_abs == pytest.approx(4.881, rel=1e-12)
    assert lat.reference.kinetic_energy_eV / 1e6 == pytest.approx(799.52, abs=0.2)

    periods = lat.meta["mad8_periods"]
    assert [(p["n_sig"], p["n_repeats"]) for p in periods] == [
        (41, 2), (37, 1), (27, 1), (32, 1), (34, 1), (41, 1), (43, 1), (37, 6), (41, 1), (37, 2)]

    flat = lat.flatten()
    counts: dict[str, int] = {}
    for p in flat:
        counts[p.element.kind] = counts.get(p.element.kind, 0) + 1
    assert counts["Bend"] == 38 and counts["Quadrupole"] == 53 and counts["Kicker"] == 52
    assert counts["Directive"] == 20                       # ten LATTICE/LATTICE_END pairs
    # HELIX counts EDGE+BEND+EDGE per dipole and a marker + body drift per kicker: its
    # 949 "transport elements" is this element set re-expanded.
    n_zero_angle = sum(1 for p in flat if p.element.kind == "Bend" and p.element.bend.angle == 0)
    helix_equivalent = (counts["Drift"] + counts["Kicker"] + counts["Quadrupole"]
                        + n_zero_angle + 3 * (counts["Bend"] - n_zero_angle))
    assert helix_equivalent == 949
    assert set(rep.codes()) <= {"UNPARSEABLE_IDENTIFIER", "RIGIDITY_FROM_BRHO",
                                "AMBIGUOUS_ROOT_LINE"}


@pytest_corpus
def test_corpus_bal_2025_parses_where_helix_stops():
    """HELIX's MAD8 resolver stops at ``SQRT`` in this deck's parameter block
    (docs/corpus.md); lattix reads it.

    The deck's own ``BRHO := P0/C*1.0E11`` is off by 1000 (the units want ``*1.0E8``:
    P0 [MeV] / c [cm/s] × 1e8 = 4.8829 T·m), so the physics assertions pass the intended
    rigidity explicitly — the point of the test is that the file parses at all.
    """
    path = _entry("helix-pipii-root/bal2025v0213.flat")
    raw, _ = Reader().read(path)
    assert raw.reference.brho_abs == pytest.approx(4.8828927 * 1000, rel=1e-6)

    lat, rep = Reader().read(path, brho=4.881)
    flat = lat.flatten()
    counts: dict[str, int] = {}
    for p in flat:
        counts[p.element.kind] = counts.get(p.element.kind, 0) + 1
    assert lat.use == "BAL2023V0914"
    assert len(flat) == 402
    assert lat.total_length == pytest.approx(125.403964, abs=1e-5)
    assert counts == {"Drift": 256, "Instrument": 84, "Quadrupole": 18, "Bend": 19,
                      "Kicker": 19, "Directive": 4, "Marker": 2}
    assert lat.variables["p0"].value == pytest.approx(
        math.sqrt(800.0 * (2 * 939.294 + 800.0)), rel=1e-12)
    assert set(rep.codes()) <= {"UNPARSEABLE_IDENTIFIER", "RIGIDITY_FROM_BRHO",
                                "AMBIGUOUS_ROOT_LINE"}


@pytest_corpus
def test_corpus_btl_2022_newcol():
    deck = ("pipii-anchors/studies_and_related_material/beam_dynamics_studies/btl/"
            "btl_lattice_with_spacecharge/mad_lattice/btl2022v0922_newcol.flat")
    lat, rep = Reader().read(_entry(deck))
    assert lat.use == "BTL2020V0929NBS"
    assert lat.total_length == pytest.approx(308.181594, abs=1e-5)
    assert len(lat.flatten()) == 867
    assert len(lat.meta["mad8_periods"]) == 7
    assert lat.reference.brho_abs == pytest.approx(4.881, rel=1e-12)
    assert set(rep.codes()) <= {"UNPARSEABLE_IDENTIFIER", "RIGIDITY_FROM_BRHO",
                                "AMBIGUOUS_ROOT_LINE"}
