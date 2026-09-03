"""MAD8 writer tests (PLAN §6 task 2.3): goldens, rules coverage, dual regimes,
idempotence and a MAD-X-backed round trip.

Golden snapshots live in ``tests/golden/mad8``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_mad8_writer.py``.  The only
normalisation applied is the lattix version in the header comment.

MAD8 itself is not installable here (no binary exists on this machine and none ships with
any conda channel), so the engine-backed leg goes through MAD-X: ``fodo.madx`` → IR →
``.lat`` (this writer) → IR (this reader) → ``.madx``, then cpymad on the original and on
the final deck must agree per element.  Every MAD8-specific spelling is covered by the
golden files instead.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.mad8 import Reader, Writer
from lattix.formats.mad8.writer import MAX_NAME_LEN, RESERVED, is_valid, sanitize, wrap
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
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
    Provenance,
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
from lattix.testing import needs

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "mad8"
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


def one_element(el: Element, *, ref: ReferenceParticle | None = None, **kw) -> Lattice:
    return Lattice.from_sequence("S", [el], ref or proton_ref(), **kw)


def demo_lattice() -> Lattice:
    """One element of every IR kind, so the golden pins all 22 writer rules."""
    ref = proton_ref()
    b = ref.brho_signed
    els: list[Element] = [
        Quadrupole(name="QF", length=0.3, multipole=MagneticMultipoleP(Bn={1: 0.6 * b})),
        Drift(name="D1", length=0.5),
        Bend(name="B1", length=1.0,
             bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.5, hgap=0.02)),
        Drift(name="D2", length=0.5),
        Solenoid(name="SOL", length=0.4, solenoid=SolenoidP(Bsol_T=0.3 * b)),
        Sextupole(name="SX", length=0.2, multipole=MagneticMultipoleP(Bn={2: 1.5 * b})),
        Octupole(name="OC", length=0.2, multipole=MagneticMultipoleP(Bn={3: 2.5 * b})),
        Multipole(name="MP", multipole=MagneticMultipoleP(BnL={2: 0.4 * b}, BsL={1: 0.1 * b})),
        RFCavity(name="CAV", length=0.0,
                 rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6, frequency_Hz=325e6)),
        Kicker(name="COR", length=0.05, hkick=1e-3, vkick=-2e-3),
        Collimator(name="COLL", length=0.1, aperture=ApertureP.rect(0.02, 0.03)),
        Instrument(name="BPM1", length=0.0, family="BPM"),
        FieldMap(name="FM", length=0.3),
        NCells(name="NC", length=0.4),
        RFQCell(name="RQ", length=0.1),
        Foil(name="FOIL"),
        Taylor(name="TAY", length=0.0),
        Patch(name="PA"),
        ReferenceChange(name="RC", dE_ref_eV=1e6),
        Freq(name="FR", frequency_Hz=162.5e6),
        Directive(name="LATTICE_1", format="tracewin", card="LATTICE", args=["4", "0"],
                  role="period_start"),
        Directive(name="LATTICE_END_1", format="tracewin", card="LATTICE_END",
                  role="period_end"),
        Bend(name="BV", length=1.05,
             bend=BendP(angle=0.0416, e1=0.0208, e2=0.0208, tilt_ref=-math.pi / 2, rect=True)),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("DEMO", els, ref)


def hminus_lattice() -> Lattice:
    """An H⁻ FODO with knobs — the PIP-II shape the ``BRHO :=`` convention exists for."""
    ref = ReferenceParticle(species=species("h-"),
                            kinetic_energy_eV=ReferenceParticle.from_brho("h-", 4.881).kinetic_energy_eV)
    b = ref.brho_signed
    qf = Quadrupole(name="QF", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.5 * b}))
    qd = Quadrupole(name="QD", length=0.2, multipole=MagneticMultipoleP(Bn={1: -1.5 * b}))
    qf.expressions["multipole.Bn[1]"] = _expr("KF")
    qd.expressions["multipole.Bn[1]"] = _expr("-KF")
    d1 = Drift(name="D1", length=0.5)
    lat = Lattice(name="TOP", reference=ref,
                  variables={"kf": Variable(value=1.5, expression=_expr("1.5")),
                             "lq": Variable(value=0.2, expression=_expr("0.2"))})
    for el in (d1, qf, qd):
        lat.elements[el.name] = el
    lat.lines["CELL"] = Line(name="CELL", items=[LineItem(ref="D1"), LineItem(ref="QF"),
                                                 LineItem(ref="D1"), LineItem(ref="QD")])
    lat.lines["TOP"] = Line(name="TOP", items=[LineItem(ref="CELL", repeat=3),
                                               LineItem(ref="CELL", reverse=True)])
    lat.use = "TOP"
    return lat


def _expr(text: str):
    from lattix.ir.expr import Expression

    return Expression(text=text, deferred=True, dialect="infix")


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
def test_golden_demo_lattice(tmp_path):
    out = tmp_path / "demo.lat"
    rep = Writer().write(demo_lattice(), out)
    assert_golden("demo.lat", out.read_text())
    assert set(rep.codes()) == {"CONST_P0", "CONST_P0_START_RIGIDITY", "FM_TO_DRIFT",
                                "NCELLS_TO_DRIFT", "RFQ_TO_DRIFT", "FOIL_TO_MARKER",
                                "TAYLOR_DROPPED", "PATCH_DROPPED", "REFCHANGE_DROPPED"}


def test_golden_hminus_line_structure(tmp_path):
    out = tmp_path / "hminus.lat"
    rep = Writer().write(hminus_lattice(), out)
    text = out.read_text()
    assert_golden("hminus_fodo.lat", text)
    assert "BEAM, MASS=" in text and "PARTICLE=" not in text     # MAD8 has no HMINUS
    assert "BRHO := 4.881" in text
    assert "CELL: LINE=(D1, QF, D1, QD)" in text     # a command name is fine as a line
    assert "TOP: LINE=(3*CELL, -CELL)" in text
    assert "K1=KF" in text and "K1=-KF" in text
    assert "BEAM_MASS_CHARGE" in rep.codes()


@needs("madx")
@pytest.mark.parametrize("deck", ["fodo.madx", "transport.madx"])
def test_golden_from_madx(deck, tmp_path):
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / deck)
    out = tmp_path / deck.replace(".madx", ".lat")
    rep = Writer().write(lat, out, strict=True)
    assert rep.ok
    assert_golden(out.name, out.read_text())


# ------------------------------------------------------------------ conventions
def test_beam_line_per_species(tmp_path):
    for name, particle in (("proton", "PROTON"), ("electron", "ELECTRON"),
                           ("positron", "POSITRON"), ("antiproton", "ANTI-PROTON"),
                           ("h-", None), ("deuteron", None)):
        ref = ReferenceParticle(species=species(name), kinetic_energy_eV=8e8)
        out = tmp_path / f"{name}.lat"
        rep = Writer().write(one_element(Drift(name="D", length=1.0), ref=ref), out)
        text = out.read_text()
        beam = "".join(ln.rstrip("&") for ln in
                       re.search(r"^BEAM,.*?(?=\n\n)", text, re.MULTILINE | re.DOTALL)
                       .group(0).splitlines())
        if particle:
            assert f"PARTICLE={particle}," in beam
            assert "BEAM_MASS_CHARGE" not in rep.codes()
        else:
            assert "PARTICLE=" not in beam
            assert "BEAM_MASS_CHARGE" in rep.codes()
        assert f"MASS={ref.species.mass_eV / 1e9:.15g}" in beam
        assert f"CHARGE={ref.species.charge}" in beam
        assert f"ENERGY={ref.total_energy_eV / 1e9:.15g}" in beam
        assert f"BRHO := {abs(ref.brho_signed):.15g}" in text


def test_numbers_use_15_significant_digits(tmp_path):
    out = tmp_path / "n.lat"
    Writer().write(one_element(Drift(name="D", length=1 / 3)), out)
    assert "L=0.333333333333333" in out.read_text()


def test_rbend_length_stays_the_arc(tmp_path):
    """MAD-X writes the chord (``rbarc``); MAD8 has no such option, so the arc is written
    unchanged and the reader gets the same number back."""
    b = Bend(name="BR", length=2.408, bend=BendP(angle=0.11455892, e1=0.11455892 / 2,
                                                 e2=0.11455892 / 2, rect=True))
    out = tmp_path / "rb.lat"
    Writer().write(one_element(b), out)
    text = out.read_text()
    assert "BR: RBEND, L=2.408, ANGLE=0.11455892" in text
    assert "E1=" not in text                       # e1 == angle/2 is the rbend default
    lat2, _ = Reader().read(out, species="proton")
    back = lat2.elements["BR"]
    assert back.length == pytest.approx(2.408, rel=1e-15)
    assert back.bend.e1 == pytest.approx(0.11455892 / 2, rel=1e-15)


def test_kicker_type_follows_the_native_spelling(tmp_path):
    k = Kicker(name="HT", length=0.06, hkick=0.0, vkick=0.0)
    k.native["mad8"] = {"type": "hkicker"}
    out = tmp_path / "k.lat"
    Writer().write(one_element(k), out)
    assert "HT: HKICKER, L=0.06, KICK=0" in out.read_text()

    plain = Kicker(name="C1", length=0.0, hkick=1e-3)
    Writer().write(one_element(plain), out)
    assert "C1: HKICKER, L=0, KICK=0.001" in out.read_text()


def test_monitor_family_round_trips(tmp_path):
    inst = Instrument(name="VP", length=0.0, family="BPM")
    inst.native["mad8"] = {"type": "vmonitor"}
    out = tmp_path / "m.lat"
    Writer().write(one_element(inst), out)
    assert "VP: VMONITOR" in out.read_text()
    Writer().write(one_element(Instrument(name="IP", family="MONITOR")), out)
    assert "IP: MONITOR" in out.read_text()


def test_rf_uses_mv_mhz_and_the_madx_lag(tmp_path):
    from lattix.ir.rf import madx_lag

    cav = RFCavity(name="CAV", length=0.0,
                   rf=RFP(voltage_V=2.5e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6))
    out = tmp_path / "rf.lat"
    rep = Writer().write(one_element(cav), out)
    text = out.read_text()
    assert f"VOLT=2.5, LAG={madx_lag(-math.pi / 6):.15g}, FREQ=162.5" in text
    assert "CONST_P0" in rep.codes()


def test_energy_modes(tmp_path):
    ref = proton_ref(1e8)
    b = ref.brho_signed
    els = [Quadrupole(name="Q1", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0 * b})),
           RFCavity(name="CAV", rf=RFP(voltage_V=5e7, phase_rad=0.0, frequency_Hz=325e6)),
           Quadrupole(name="Q2", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0 * b}))]
    lat = Lattice.from_sequence("LIN", els, ref)
    const = tmp_path / "const.lat"
    local = tmp_path / "local.lat"
    rc = Writer().write(lat, const, energy_mode="constant")
    rl = Writer().write(lat, local, energy_mode="local")
    assert "CONST_P0_START_RIGIDITY" in rc.codes() and "CONST_P0_LOCAL_RIGIDITY" in rl.codes()

    def k1(text, name):
        return float(re.search(rf"^{name}: QUADRUPOLE, L=0.3, K1=([-\d.eE+]+)$",
                               text, re.MULTILINE).group(1))
    assert k1(const.read_text(), "Q2") == pytest.approx(k1(const.read_text(), "Q1"))
    assert k1(local.read_text(), "Q2") < k1(local.read_text(), "Q1")   # higher rigidity
    with pytest.raises(ValueError, match="energy_mode"):
        Writer().write(lat, tmp_path / "bad.lat", energy_mode="nope")


# ------------------------------------------------------------------------ names
def test_sanitize_upper_cases_and_caps_at_16():
    assert sanitize("q f/1") == "Q_F_1"
    assert sanitize("9lead") == "E_9LEAD"
    assert len(sanitize("a" * 40)) == MAX_NAME_LEN
    assert sanitize("DRIFT") not in RESERVED
    assert is_valid("QLF1") and not is_valid("qlf1") and not is_valid("DRIFT")


def test_names_are_uniquified_and_tagged(tmp_path):
    a = Drift(name="a-name", length=1.0,
              provenance=Provenance(format="tracewin", original_name="a-name",
                                    original_type="DRIFT"))
    b = Drift(name="a+name", length=2.0)
    lat = Lattice.from_sequence("S", [a, b], proton_ref())
    out = tmp_path / "names.lat"
    Writer().write(lat, out)
    text = out.read_text()
    assert '! lattix: name="a-name" type="DRIFT"' in text
    assert "A_NAME: DRIFT, L=1" in text
    assert "A_NAME_2: DRIFT, L=2" in text
    from lattix.formats.mad8.reader import parse_tags

    assert parse_tags(text)["A_NAME"] == {"name": "a-name", "type": "DRIFT"}


def test_upper_casing_alone_is_not_a_rename(tmp_path):
    lat = one_element(Drift(name="qf", length=1.0))
    out = tmp_path / "u.lat"
    Writer().write(lat, out)
    assert "! lattix:" not in out.read_text()          # MAD8 is case insensitive


def test_reserved_names_are_renamed(tmp_path):
    lat = one_element(Drift(name="LINE", length=1.0))
    out = tmp_path / "r.lat"
    Writer().write(lat, out)
    assert "LINE_X: DRIFT" in out.read_text()


# ------------------------------------------------------------ continuation cards
def test_wrap_breaks_at_commas_with_an_ampersand():
    long = "X: LINE=(" + ", ".join(f"E{i:03d}" for i in range(40)) + ")"
    lines = wrap(long)
    assert len(lines) > 1
    assert all(len(ln) <= 79 for ln in lines)
    assert all(ln.endswith("&") for ln in lines[:-1])
    assert "".join(ln.rstrip("&") for ln in lines).replace(" ", "") == long.replace(" ", "")


def test_long_line_round_trips_through_the_reader(tmp_path):
    els = [Drift(name=f"D{i:03d}", length=0.1) for i in range(60)]
    lat = Lattice.from_sequence("LONG", els, proton_ref())
    out = tmp_path / "long.lat"
    Writer().write(lat, out)
    assert max(len(ln) for ln in out.read_text().splitlines()) <= 79
    assert "&" in out.read_text()
    back, _ = Reader().read(out, species="proton")
    assert len(back.flatten()) == 60
    assert back.total_length == pytest.approx(6.0)


# ------------------------------------------------------------------ expressions
def test_expressions_are_re_emitted_and_can_be_switched_off(tmp_path):
    lat = hminus_lattice()
    on, off = tmp_path / "on.lat", tmp_path / "off.lat"
    Writer().write(lat, on)
    Writer().write(lat, off, use_expressions=False)
    assert "K1=KF" in on.read_text() and "KF := 1.5" in on.read_text()
    assert "K1=KF" not in off.read_text() and "KF := " not in off.read_text()
    assert "K1=1.5" in off.read_text()


def test_attribute_reference_expressions_survive(tmp_path):
    """``L=D1[L]+0.1`` is re-emitted only when the element it points at is in the deck."""
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
D2: DRIFT, L=D1[L]+0.1
D3: DRIFT, L=GHOST[L]+0.1
GHOST: DRIFT, L=1.0
TOP: LINE=(D1, D2, D3)
"""
    src = tmp_path / "attr.lat"
    src.write_text(text)
    lat, _ = Reader().read(src)
    out = tmp_path / "attr_out.lat"
    rep = Writer().write(lat, out)
    written = out.read_text()
    assert "D2: DRIFT, L=D1[L]+0.1" in written
    assert "D3: DRIFT, L=1.1" in written              # GHOST is not part of the deck
    assert rep.codes()["EXPRESSION_DROPPED"] == 1


def test_stale_expression_falls_back_to_the_number(tmp_path):
    lat = hminus_lattice()
    lat.variables["kf"] = Variable(value=99.0, expression=_expr("99.0"))
    out = tmp_path / "stale.lat"
    rep = Writer().write(lat, out)
    assert "K1=1.5" in out.read_text()
    assert rep.codes()["EXPRESSION_DROPPED"] == 2


def test_variable_that_mad8_cannot_spell_is_folded_in(tmp_path):
    lat = hminus_lattice()
    lat.variables["k f"] = Variable(value=1.0, expression=_expr("1.0"))
    out = tmp_path / "v.lat"
    rep = Writer().write(lat, out)
    assert "VARIABLE_DROPPED" in rep.codes()


def test_brho_variable_conflict_is_reported(tmp_path):
    lat = hminus_lattice()
    lat.variables["brho"] = Variable(value=9.9, expression=_expr("9.9"))
    out = tmp_path / "b.lat"
    rep = Writer().write(lat, out)
    assert "BRHO_REPLACED" in rep.codes()
    assert "BRHO := 4.881" in out.read_text()


# ----------------------------------------------------- dual regime (LOSSY/DROPPED)
def _shifted() -> Lattice:
    el = Drift(name="D", length=1.0, shift=BodyShiftP(x_offset=1e-3))
    return one_element(el)


def _superposed() -> Lattice:
    child = FieldMap(name="CH", length=0.2)
    sup = Superposition(name="SUP", length=0.4, children=[(0.0, "CH")])
    lat = Lattice.from_sequence("S", [sup], proton_ref())
    lat.elements["CH"] = child
    return lat


def _superposition_child_missing() -> Lattice:
    sup = Superposition(name="SUP", length=0.4, children=[(0.0, "NOPE")])
    return Lattice.from_sequence("S", [sup], proton_ref())


def _no_line_structure() -> Lattice:
    lat = Lattice.from_sequence("S", [Drift(name="D", length=1.0)], proton_ref())
    lat.use = "D"                    # a bare element, not a line
    return lat


DUAL_CASES = {
    "FM_TO_DRIFT": lambda: one_element(FieldMap(name="FM", length=0.3)),
    "NCELLS_TO_DRIFT": lambda: one_element(NCells(name="NC", length=0.3)),
    "RFQ_TO_DRIFT": lambda: one_element(RFQCell(name="RQ", length=0.3)),
    "FOIL_TO_MARKER": lambda: one_element(Foil(name="FO")),
    "TAYLOR_DROPPED": lambda: one_element(Taylor(name="TA")),
    "PATCH_DROPPED": lambda: one_element(Patch(name="PA")),
    "REFCHANGE_DROPPED": lambda: one_element(ReferenceChange(name="RC", dE_ref_eV=0.0)),
    "FOREIGN_DIRECTIVE": lambda: one_element(
        Directive(name="ERR", format="tracewin", card="ERROR_QUAD_NCPL_STAT", role="error")),
    "SUPERPOSITION_FLATTENED": _superposed,
    "SUPERPOSITION_CHILD_MISSING": _superposition_child_missing,
    "MISALIGN_DROPPED": _shifted,
    "APERTURE_DROPPED": lambda: one_element(Marker(name="MK", aperture=ApertureP.circle(0.02))),
    "APERTURE_APPROXIMATED": lambda: one_element(
        Drift(name="D", length=1.0, aperture=ApertureP.rect(0.02, 0.03))),
    "FINTX_DROPPED": lambda: one_element(
        Bend(name="B", length=1.0, bend=BendP(angle=0.1, edge_int1=0.5, edge_int2=0.7))),
    "SKEW_COMPONENT_DROPPED": lambda: one_element(
        Sextupole(name="SX", length=0.2, multipole=MagneticMultipoleP(Bs={2: 1.0}))),
    "EKICK_AS_MAGNETIC": lambda: one_element(
        Kicker(name="EK", length=0.1, vkick=1e-3, electric=True)),
    "KICK_COMPONENT_DROPPED": lambda: one_element(
        _hkicker_with_vkick()),
    "MARKER_LENGTH_AS_DRIFT": lambda: one_element(Marker(name="MK", length=0.2)),
    "NO_LINE_STRUCTURE": _no_line_structure,
}


def _hkicker_with_vkick() -> Kicker:
    k = Kicker(name="HK", length=0.1, hkick=1e-3, vkick=2e-3)
    k.native["mad8"] = {"type": "hkicker"}
    return k


@pytest.mark.parametrize("code", sorted(DUAL_CASES))
def test_dual_regime_every_downgrade(code, tmp_path):
    lat = DUAL_CASES[code]()
    out = tmp_path / f"{code.lower()}.lat"
    rep = Writer().write(lat, out)
    assert code in rep.codes(), rep.summary()
    assert out.exists()
    # SUPERPOSITION_CHILD_MISSING can only follow its parent's SUPERPOSITION_FLATTENED,
    # and strict mode raises on the first problem it meets.
    pattern = "SUPERPOSITION_" if code.startswith("SUPERPOSITION_") else code
    with pytest.raises(TranslationError, match=pattern):
        Writer().write(DUAL_CASES[code](), tmp_path / f"{code.lower()}_strict.lat", strict=True)


def test_permissive_still_writes_a_readable_deck(tmp_path):
    out = tmp_path / "demo.lat"
    Writer().write(demo_lattice(), out)
    back, _ = Reader().read(out, species="proton")
    assert back.total_length == pytest.approx(demo_lattice().total_length)


def test_multi_rigidity_definition_is_recorded(tmp_path):
    """One definition reused at two reference energies: MAD8 has one number per
    attribute, so the first occurrence's rigidity wins and the ledger says so."""
    ref = proton_ref(1e8)
    q = Quadrupole(name="Q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0 * ref.brho_signed}))
    cav = RFCavity(name="CAV", rf=RFP(voltage_V=5e7, phase_rad=0.0, frequency_Hz=325e6))
    lat = Lattice(name="L", reference=ref)
    lat.elements.update({"Q": q, "CAV": cav})
    lat.lines["L"] = Line(name="L", items=[LineItem(ref="Q"), LineItem(ref="CAV"),
                                           LineItem(ref="Q")])
    lat.use = "L"
    rep = Writer().write(lat, tmp_path / "m.lat", energy_mode="local")
    assert "MULTI_RIGIDITY_DEFINITION" in rep.codes()


def test_skew_multipole_is_folded_into_a_tilt(tmp_path):
    b = proton_ref().brho_signed
    mp = Multipole(name="MP", multipole=MagneticMultipoleP(BnL={2: 0.3 * b}, BsL={2: 0.4 * b}))
    out = tmp_path / "skew.lat"
    rep = Writer().write(one_element(mp), out)
    assert "SKEW_MULTIPOLE_AS_TILT" in rep.codes()
    text = out.read_text()
    assert f"K2L={math.hypot(0.3, 0.4):.15g}" in text
    assert "T2=" in text


def test_multipole_round_trips(tmp_path):
    b = proton_ref().brho_signed
    mp = Multipole(name="MP", multipole=MagneticMultipoleP(BnL={2: 0.4 * b}, BsL={1: 0.1 * b}))
    out = tmp_path / "mp.lat"
    Writer().write(one_element(mp), out)
    back, _ = Reader().read(out, species="proton")
    got = back.elements["MP"].multipole
    assert got.BnL[2] == pytest.approx(0.4 * b, rel=1e-12)
    assert got.BnL[1] == pytest.approx(0.1 * b, rel=1e-12)       # skew quad as a tilt
    assert got.tilt[1] == pytest.approx(math.pi / 4, rel=1e-12)


# ------------------------------------------------------------------ idempotence
@pytest.mark.parametrize("build", [demo_lattice, hminus_lattice])
def test_write_read_write_is_a_fixed_point(build, tmp_path):
    """I-13: ``write(read(write(read(x))))`` is byte-identical to ``write(read(x))``."""
    first = tmp_path / "a.lat"
    Writer().write(build(), first)
    lat1, _ = Reader().read(first, species=build().reference.species.name)
    second = tmp_path / "b.lat"
    Writer().write(lat1, second)
    lat2, _ = Reader().read(second, species=build().reference.species.name)
    third = tmp_path / "c.lat"
    Writer().write(lat2, third)
    assert second.read_text() == third.read_text()
    assert lat2.total_length == pytest.approx(lat1.total_length)


@needs("madx")
def test_fodo_fixed_point(tmp_path):
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "fodo.madx")
    a, b, c = (tmp_path / n for n in ("a.lat", "b.lat", "c.lat"))
    Writer().write(lat, a)
    lat1, _ = Reader().read(a)
    Writer().write(lat1, b)
    lat2, _ = Reader().read(b)
    Writer().write(lat2, c)
    assert b.read_text() == c.read_text()


# -------------------------------------------------------------- MAD-X oracle leg
@needs("madx")
@pytest.mark.oracle_madx
@pytest.mark.parametrize("deck", ["fodo.madx", "transport.madx"])
def test_madx_roundtrip_through_mad8(deck, tmp_path):
    """``fodo.madx`` → IR → ``.lat`` → IR → ``.madx``: cpymad on the first and last decks
    must give the same per-element optics (there is no MAD8 binary to ask instead)."""
    from lattix.formats.madx import Reader as MadxReader
    from lattix.formats.madx import Writer as MadxWriter
    from lattix.oracles import get_oracle
    from lattix.oracles.compare import compare_pair

    src = DATA / deck
    lat, r_in = MadxReader().read(src)
    mad8 = tmp_path / deck.replace(".madx", ".lat")
    r8w = Writer().write(lat, mad8, strict=True)
    lat2, r8r = Reader().read(mad8, strict=False)
    final = tmp_path / f"final_{deck}"
    r_out = MadxWriter().write(lat2, final, mode="line", strict=True)
    assert r_in.ok and r8w.ok and r8r.ok and r_out.ok
    assert set(r8r.codes()) == {"RIGIDITY_FROM_BRHO"}

    oracle = get_oracle("madx")
    a = oracle.run(src, workdir=tmp_path / "a")
    b = oracle.run(final, workdir=tmp_path / "b")
    cmp = compare_pair(a, b)
    assert cmp.n_shared >= 8
    assert cmp.length_a == pytest.approx(cmp.length_b, abs=1e-12)
    assert cmp.blocks["T4x4"] < 1e-9
    assert cmp.blocks["disp"] < 1e-9
    assert cmp.notes == []
