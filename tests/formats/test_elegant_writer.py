"""Elegant ``.lte`` writer tests (PLAN §6 task 2.1): goldens, RULES coverage,
idempotence and both regimes for every downgrade.

Golden snapshots live in ``tests/golden/elegant``; regenerate them deliberately
with ``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_elegant_writer.py``.  The
only normalisation applied is the lattix version in the header comment.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.elegant import Reader, Writer, sanitize
from lattix.formats.elegant.naming import RESERVED, is_valid, parse_tags
from lattix.ir.elements import (
    ALL_KINDS,
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
from lattix.ir.rf import elegant_phase_deg

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "elegant"
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
    """FODO with two bends (one rectangular), a solenoid, a cavity, a kicker and a
    collimator — the same shape as the MAD-X writer's golden so the two are comparable."""
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
        Sextupole(name="SX", length=0.2, multipole=MagneticMultipoleP(Bn={2: 2.0 * b})),
        Octupole(name="OC", length=0.2, multipole=MagneticMultipoleP(Bn={3: 3.0 * b})),
        Multipole(name="MP", length=0.0, multipole=MagneticMultipoleP(BnL={1: 0.01 * b,
                                                                          2: 0.02 * b})),
        Instrument(name="BPM1", length=0.0, family="BPM"),
        Taylor(name="MAT", length=0.1),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("DEMO", els, ref)


def nested_lattice() -> Lattice:
    """Nested lines with a 3× repeat and a reflection (the structure must survive)."""
    ref = proton_ref()
    b = ref.brho_signed
    lat = Lattice(name="RING", reference=ref)
    for el in (Quadrupole(name="QF", length=0.3, multipole=MagneticMultipoleP(Bn={1: 0.6 * b})),
               Drift(name="DR", length=0.5),
               Marker(name="MK")):
        lat.add_element(el)
    lat.lines["CELL"] = Line(name="CELL", items=[LineItem(ref="QF"),
                                                 LineItem(ref="DR", repeat=3),
                                                 LineItem(ref="MK")])
    lat.lines["RING"] = Line(name="RING", items=[LineItem(ref="CELL"),
                                                 LineItem(ref="CELL", reverse=True),
                                                 LineItem(ref="CELL", repeat=2)])
    lat.use = "RING"
    lat.variables["LQ"] = Variable(value=0.3)
    return lat


# --------------------------------------------------------------------------- goldens
def test_golden_demo(tmp_path):
    out = tmp_path / "demo.lte"
    rep = Writer().write(demo_lattice(), out)
    assert_golden("demo.lte", out.read_text())
    assert rep.counts.get("LOSSY", 0) == 0 and rep.counts.get("DROPPED", 0) == 0


def test_golden_nested_lines(tmp_path):
    out = tmp_path / "nested.lte"
    Writer().write(nested_lattice(), out)
    assert_golden("nested.lte", out.read_text())
    text = out.read_text()
    assert "CELL: LINE=(QF, 3*DR, MK)" in text
    assert "RING: LINE=(CELL, -CELL, 2*CELL)" in text
    assert text.rstrip().endswith("USE, RING")


def test_golden_fodo_from_madx(tmp_path):
    """The MAD-X reader exists and works; this is the cross-format golden."""
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "fodo.madx")
    out = tmp_path / "fodo.lte"
    rep = Writer().write(lat, out, strict=True)
    assert_golden("fodo_from_madx.lte", out.read_text())
    assert rep.ok


def test_golden_transport_from_madx(tmp_path):
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "transport.madx")
    out = tmp_path / "transport.lte"
    Writer().write(lat, out, strict=True)
    assert_golden("transport_from_madx.lte", out.read_text())


# --------------------------------------------------------------------------- idempotence
@pytest.mark.parametrize("build", [demo_lattice, nested_lattice], ids=["demo", "nested"])
def test_write_read_write_is_a_fixed_point(tmp_path, build):
    a, b = tmp_path / "a.lte", tmp_path / "b.lte"
    Writer().write(build(), a)
    lat2, _ = Reader().read(a, species="proton", kinetic_energy_eV=8e8)
    Writer().write(lat2, b)
    lat3, _ = Reader().read(b, species="proton", kinetic_energy_eV=8e8)
    c = tmp_path / "c.lte"
    Writer().write(lat3, c)
    assert b.read_text() == c.read_text()


def test_fodo_from_madx_round_trip_is_a_fixed_point(tmp_path):
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "fodo.madx")
    a = tmp_path / "a.lte"
    Writer().write(lat, a)
    ke = lat.reference.kinetic_energy_eV
    lat2, _ = Reader().read(a, species="proton", kinetic_energy_eV=ke)
    b = tmp_path / "b.lte"
    Writer().write(lat2, b)
    lat3, _ = Reader().read(b, species="proton", kinetic_energy_eV=ke)
    c = tmp_path / "c.lte"
    Writer().write(lat3, c)
    assert b.read_text() == c.read_text()
    assert lat3.total_length == pytest.approx(lat.total_length, abs=1e-12)


def test_read_write_preserves_the_source_types_and_unmapped_attributes(tmp_path):
    src = tmp_path / "src.lte"
    src.write_text("\n".join([
        "% 0.35 sto LQ",
        "Q: QUAD, L=LQ, K1=0.6, N_KICKS=8",
        "D: CSRDRIFT, L=0.500, N_KICKS=10",
        "B: SBEN, L=1, ANGLE=0.1, FINT=0.5",
        "L1: LINE=(Q,D,B)",
        "USE, L1",
    ]))
    lat, _ = Reader().read(src, species="proton", kinetic_energy_eV=8e8)
    out = tmp_path / "out.lte"
    Writer().write(lat, out)
    text = out.read_text()
    assert "% 0.35 sto LQ" in text
    assert "Q: QUAD, L=LQ, K1=0.6, N_KICKS=8" in text          # type, expression and extras kept
    assert "D: CSRDRIFT, L=0.500, N_KICKS=10" in text
    assert "B: SBEN, L=1, ANGLE=0.1" in text                    # FINT == default, not re-emitted


# --------------------------------------------------------------------------- rules / naming
def test_rules_cover_every_ir_kind():
    assert check_rules_coverage(Writer()) == set()
    assert set(Writer.RULES) == set(ALL_KINDS)


def test_every_kind_is_writable(tmp_path):
    """One lattice holding one element of every IR kind must write without an exception."""
    ref = proton_ref()
    b = ref.brho_signed
    els = [
        Drift(name="DR", length=0.1),
        Quadrupole(name="QU", length=0.1, multipole=MagneticMultipoleP(Bn={1: b})),
        Sextupole(name="SE", length=0.1, multipole=MagneticMultipoleP(Bn={2: b})),
        Octupole(name="OC", length=0.1, multipole=MagneticMultipoleP(Bn={3: b})),
        Multipole(name="MU", multipole=MagneticMultipoleP(BnL={2: 0.1 * b})),
        Bend(name="BE", length=1.0, bend=BendP(angle=0.1)),
        Solenoid(name="SO", length=0.1, solenoid=SolenoidP(Bsol_T=0.2)),
        RFCavity(name="CA", length=0.1, rf=RFP(voltage_V=1e5, frequency_Hz=1e8)),
        FieldMap(name="FM", length=0.3),
        NCells(name="NC", length=0.3),
        RFQCell(name="RQ", length=0.3),
        Kicker(name="KI", hkick=1e-3),
        Collimator(name="CO", aperture=ApertureP.circle(0.01)),
        Marker(name="MA"),
        Instrument(name="IN", family="BPM"),
        Foil(name="FO"),
        Taylor(name="TA"),
        Patch(name="PA", x_offset=1e-3),
        ReferenceChange(name="RC", dE_ref_eV=1e5),
        Freq(name="FR", frequency_Hz=1e8),
        Directive(name="DI", format="tracewin", card="SET_ADV", role="matching"),
    ]
    lat = Lattice.from_sequence("ALLKINDS", els, ref)
    sup_child = Drift(name="SUPKID", length=0.2)
    lat.add_element(sup_child)
    sup = Superposition(name="SU", length=0.2, children=[(0.0, "SUPKID")])
    lat.add_element(sup)
    lat.lines["ALLKINDS"].items.append(LineItem(ref="SU"))
    rep = Writer().write(lat, tmp_path / "all.lte")
    kinds_seen = {e.kind for e in rep.entries if e.kind}
    assert kinds_seen >= set(ALL_KINDS) - {"Superposition"} or "Superposition" in kinds_seen
    assert (tmp_path / "all.lte").is_file()


@pytest.mark.parametrize("bad,good", [
    ("q f", "Q_F"), ("1quad", "E_1QUAD"), ("q-1", "Q_1"), ("q#1", "Q_1"), ("QUAD", "QUAD_X"),
])
def test_sanitize(bad, good):
    assert sanitize(bad) == good
    assert is_valid(sanitize(bad))


def test_reserved_type_keywords_are_never_used_as_names():
    assert "QUAD" in RESERVED and "LINE" in RESERVED and "MARK" in RESERVED
    assert not is_valid("QUAD")


def test_renamed_elements_carry_a_reversible_tag(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [Drift(name="my drift", length=1.0),
                                      Marker(name="MARK")], ref)
    out = tmp_path / "n.lte"
    Writer().write(lat, out)
    text = out.read_text()
    tags = parse_tags(text)
    assert tags["MY_DRIFT"]["name"] == "my drift"
    assert tags["MARK_X"]["name"] == "MARK"
    back, _ = Reader().read(out, species="proton", kinetic_energy_eV=8e8)
    originals = {e.provenance.original_name for e in back.elements.values()}
    assert {"my drift", "MARK"} <= originals


def test_duplicate_names_are_uniquified(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [Drift(name="d", length=1.0), Drift(name="D", length=2.0)],
                                ref)
    out = tmp_path / "u.lte"
    Writer().write(lat, out)
    text = out.read_text()
    assert "D: DRIF, L=1" in text and "D_2: DRIF, L=2" in text


# --------------------------------------------------------------------------- physics
def test_quadrupole_k1_is_the_gradient_over_the_signed_rigidity(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [Quadrupole(name="Q", length=0.3,
                                                 multipole=MagneticMultipoleP(
                                                     Bn={1: 0.6 * ref.brho_signed}))], ref)
    out = tmp_path / "q.lte"
    Writer().write(lat, out)
    assert "Q: KQUAD, L=0.3, K1=0.6" in out.read_text()


@pytest.mark.parametrize("sp,crest", [("proton", -90.0), ("h-", 90.0), ("electron", 90.0)])
def test_rfca_phase_is_written_for_the_lattice_species(tmp_path, sp, crest):
    ref = ReferenceParticle(species=species(sp), kinetic_energy_eV=2.1e6)
    lat = Lattice.from_sequence("L", [RFCavity(name="C", rf=RFP(voltage_V=1e6,
                                                               phase_rad=-math.pi / 6,
                                                               frequency_Hz=325e6))], ref)
    out = tmp_path / "c.lte"
    rep = Writer().write(lat, out)
    text = out.read_text()
    want = elegant_phase_deg(-math.pi / 6, ref.species.charge)
    assert want == pytest.approx((-30.0 + crest) % 360.0)
    assert f"PHASE={want:.15g}" in text
    assert "CHANGE_P0=1" in text
    codes = rep.codes()
    assert "ELEGANT_PHASE_FOR_SPECIES" in codes and "RFCA_CHANGE_P0" in codes
    entry = next(e for e in rep.entries if e.code == "ELEGANT_PHASE_FOR_SPECIES")
    assert entry.details["species"] == ref.species.name


def test_rbend_is_written_with_the_chord_length(tmp_path):
    ref = proton_ref()
    arc = 1.000416788226488                       # elegant's own arc for chord 1, angle 0.1
    lat = Lattice.from_sequence("L", [Bend(name="B", length=arc,
                                           bend=BendP(angle=0.1, e1=0.05, e2=0.05, rect=True))],
                                ref)
    out = tmp_path / "b.lte"
    Writer().write(lat, out)
    line = next(ln for ln in out.read_text().splitlines() if ln.startswith("B:"))
    assert line.startswith("B: RBEN, L=1")
    assert float(line.split("L=")[1].split(",")[0]) == pytest.approx(1.0, abs=1e-14)
    assert "E1=" not in line                      # e1 == angle/2 exactly, so nothing to write


def test_fint_is_written_whenever_it_differs_from_elegants_default(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [
        Bend(name="B0", length=1.0, bend=BendP(angle=0.1, edge_int1=0.0, hgap=0.03)),
        Bend(name="B5", length=1.0, bend=BendP(angle=0.1, edge_int1=0.5, hgap=0.03)),
        Bend(name="BX", length=1.0, bend=BendP(angle=0.1, edge_int1=0.3, edge_int2=0.7)),
    ], ref)
    out = tmp_path / "f.lte"
    Writer().write(lat, out)
    text = out.read_text()
    assert "B0: CSBEND, L=1, ANGLE=0.1, FINT=0, HGAP=0.03" in text
    assert "B5: CSBEND, L=1, ANGLE=0.1, HGAP=0.03" in text          # 0.5 is elegant's default
    assert "BX: CSBEND, L=1, ANGLE=0.1, FINT1=0.3, FINT2=0.7" in text


def test_multipole_is_split_one_element_per_order(tmp_path):
    ref = proton_ref()
    b = ref.brho_signed
    lat = Lattice.from_sequence("L", [Multipole(name="M", multipole=MagneticMultipoleP(
        BnL={1: 0.01 * b, 3: 0.03 * b}))], ref)
    out = tmp_path / "m.lte"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert "M: MULT, L=0, ORDER=1, KNL=0.01" in text
    assert "M__2: MULT, L=0, ORDER=3, KNL=0.03" in text
    assert "MULT_SPLIT_BY_ORDER" in rep.codes()
    assert "M__ALL: LINE=(M, M__2)" in text
    assert "L: LINE=(M__ALL)" in text


def test_electric_kickers_use_ehkick_and_evkick(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [Kicker(name="EH", hkick=1e-3, electric=True),
                                      Kicker(name="EV", vkick=2e-3, electric=True),
                                      Kicker(name="EB", hkick=1e-3, vkick=2e-3, electric=True)],
                                ref)
    out = tmp_path / "k.lte"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert "EH: EHKICK, L=0, KICK=0.001" in text
    assert "EV: EVKICK, L=0, KICK=0.002" in text
    assert "EB: KICKER" in text
    assert "EKICK_AS_MAGNETIC" in rep.codes()


def test_misalignments_are_written_only_where_elegant_accepts_them(tmp_path):
    """Measured: DRIF takes no DX/DY/DZ at all, RFCA takes DX/DY but not DZ, KQUAD takes
    DX/DY/DZ/PITCH/YAW, and a plain SEXT has no PITCH/YAW."""
    ref = proton_ref()
    shift = BodyShiftP(x_offset=1e-3, y_offset=2e-3, z_offset=3e-3, x_rot=1e-4)
    lat = Lattice.from_sequence("L", [
        Drift(name="D", length=0.5, shift=shift),
        Quadrupole(name="Q", length=0.3, multipole=MagneticMultipoleP(Bn={1: ref.brho_signed}),
                   shift=shift),
        # a plain SEXT (the source type is kept) has no PITCH/YAW, unlike KSEXT
        Sextupole(name="S", length=0.2, multipole=MagneticMultipoleP(Bn={2: ref.brho_signed}),
                  shift=shift, native={"elegant": {"type": "SEXT", "attrs": {}, "consumed": []}}),
        RFCavity(name="C", length=0.2, rf=RFP(voltage_V=1e5, frequency_Hz=1e8), shift=shift),
    ], ref)
    out = tmp_path / "al.lte"
    rep = Writer().write(lat, out)
    lines = {ln.split(":")[0]: ln for ln in out.read_text().splitlines() if ":" in ln}
    assert "DX=" not in lines["D"]                                   # DRIF has no alignment
    assert "DX=0.001, DY=0.002, DZ=0.003, PITCH=-0.0001" in lines["Q"]
    assert "DX=0.001, DY=0.002, DZ=0.003" in lines["S"] and "PITCH" not in lines["S"]
    assert "DX=0.001, DY=0.002" in lines["C"] and "DZ=" not in lines["C"]
    dropped = {tuple(e.details["attrs"]) for e in rep.entries if e.code == "MISALIGN_DROPPED"}
    assert ("DX", "DY", "DZ", "PITCH") in dropped                    # the drift
    assert ("PITCH",) in dropped                                     # the sextupole
    assert ("DZ", "PITCH") in dropped                                # the cavity


def test_collimator_shapes(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [
        Collimator(name="E", length=0.1, aperture=ApertureP.circle(0.02)),
        Collimator(name="R", length=0.1, aperture=ApertureP.rect(0.03, 0.04)),
    ], ref)
    out = tmp_path / "c.lte"
    Writer().write(lat, out)
    text = out.read_text()
    assert "E: ECOL, L=0.1, X_MAX=0.02, Y_MAX=0.02" in text
    assert "R: RCOL, L=0.1, X_MAX=0.03, Y_MAX=0.04" in text


def test_instrument_families(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [Instrument(name="B", family="BPM", length=0.05),
                                      Instrument(name="H", family="HMON"),
                                      Instrument(name="W", family="WATCH"),
                                      Instrument(name="X", family="PROFILE")], ref)
    out = tmp_path / "i.lte"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert "B: MONI, L=0.05" in text and "H: HMON" in text
    assert 'W: WATCH, FILENAME="%s.w"' in text
    assert "X: MARK" in text
    assert "INSTRUMENT_AS_MARKER" in rep.codes()
    assert rep.ok        # a zero-length diagnostic loses no optics: EQUIVALENT, not LOSSY


def test_instrument_with_a_length_keeps_it_as_a_monitor(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [Instrument(name="P", family="PROFILE", length=0.12)], ref)
    out = tmp_path / "p.lte"
    rep = Writer().write(lat, out, strict=True)
    assert "P: MONI, L=0.12" in out.read_text()
    assert "INSTRUMENT_AS_MONITOR" in rep.codes()


def test_native_elegant_directive_is_re_emitted(tmp_path):
    ref = proton_ref()
    lat = Lattice.from_sequence("L", [
        Directive(name="Q0", format="elegant", card="CHARGE", role="beam",
                  args=["TOTAL=1e-9"]),
        Drift(name="D", length=1.0)], ref)
    out = tmp_path / "d.lte"
    rep = Writer().write(lat, out, strict=True)
    assert "Q0: CHARGE, TOTAL=1e-9" in out.read_text()
    assert "NATIVE_DIRECTIVE" in [e.code for e in rep.entries]


def test_use_expressions_false_writes_plain_numbers(tmp_path):
    src = tmp_path / "src.lte"
    src.write_text("% 0.35 sto LQ\nQ: QUAD, L=LQ, K1=0.6\nL1: LINE=(Q)\nUSE,L1")
    lat, _ = Reader().read(src, species="proton", kinetic_energy_eV=8e8)
    out = tmp_path / "out.lte"
    Writer().write(lat, out, use_expressions=False)
    text = out.read_text()
    assert "sto LQ" not in text
    assert "Q: QUAD, L=0.35, K1=0.6" in text


def test_line_name_option(tmp_path):
    out = tmp_path / "n.lte"
    Writer().write(nested_lattice(), out, line_name="MYRING")
    assert out.read_text().rstrip().endswith("USE, MYRING")


# --------------------------------------------------------------------------- downgrades
def _one(el, **kw) -> Lattice:
    ref = proton_ref()
    return Lattice.from_sequence("L", [el, Drift(name="PAD", length=0.1)], ref, **kw)


DOWNGRADES = {
    "FM_TO_DRIFT": lambda: _one(FieldMap(name="FM", length=0.3)),
    "NCELLS_TO_DRIFT": lambda: _one(NCells(name="NC", length=0.3)),
    "RFQ_TO_DRIFT": lambda: _one(RFQCell(name="RQ", length=0.3)),
    "FOIL_TO_MARKER": lambda: _one(Foil(name="FO")),
    "PATCH_DROPPED": lambda: _one(Patch(name="PA", x_offset=1e-3)),
    "REFCHANGE_DROPPED": lambda: _one(ReferenceChange(name="RC", dE_ref_eV=1e5)),
    "FOREIGN_DIRECTIVE": lambda: _one(Directive(name="DI", format="tracewin", card="SET_ADV",
                                                role="matching")),
    "EKICK_AS_MAGNETIC": lambda: _one(Kicker(name="EB", hkick=1e-3, vkick=1e-3, electric=True)),
    "SKEW_MULTIPOLE_DROPPED": lambda: _one(Multipole(name="MS", multipole=MagneticMultipoleP(
        BsL={1: 0.01}))),
    "BEND_FINTX_DROPPED": lambda: _one(Bend(name="BF", length=1.0,
                                            bend=BendP(angle=0.1, rect=True, edge_int1=0.1,
                                                       edge_int2=0.9))),
    "MISALIGN_DROPPED": lambda: _one(Solenoid(name="SR", length=0.2,
                                              solenoid=SolenoidP(Bsol_T=0.1),
                                              shift=BodyShiftP(x_rot=1e-3))),
    "NCELL_DROPPED": lambda: _one(RFCavity(name="CN", length=0.2,
                                           rf=RFP(voltage_V=1e6, frequency_Hz=1e8, n_cell=5))),
    "SUPERPOSITION_FLATTENED": None,          # built below (needs an extra child element)
}


def _superposition_lattice() -> Lattice:
    ref = proton_ref()
    lat = Lattice(name="L", reference=ref)
    lat.add_element(Drift(name="KID", length=0.2))
    lat.add_element(Superposition(name="SU", length=0.2, children=[(0.0, "KID")]))
    lat.lines["L"] = Line(name="L", items=[LineItem(ref="SU")])
    lat.use = "L"
    return lat


DOWNGRADES["SUPERPOSITION_FLATTENED"] = _superposition_lattice


@pytest.mark.parametrize("code", sorted(DOWNGRADES))
def test_every_downgrade_is_recorded_in_permissive_mode(tmp_path, code):
    rep = Writer().write(DOWNGRADES[code](), tmp_path / f"{code}.lte")
    assert code in rep.codes(), rep.summary()


@pytest.mark.parametrize("code", sorted(DOWNGRADES))
def test_every_downgrade_raises_in_strict_mode(tmp_path, code):
    with pytest.raises(TranslationError) as e:
        Writer().write(DOWNGRADES[code](), tmp_path / f"{code}.lte", strict=True)
    assert e.value.entry.cls in ("LOSSY", "DROPPED")


def test_directive_roles_that_carry_no_physics_are_exact(tmp_path):
    lat = _one(Directive(name="DI", format="tracewin", card="LATTICE", role="period_start"))
    rep = Writer().write(lat, tmp_path / "ok.lte", strict=True)
    assert rep.ok
    assert "! lattix directive: LATTICE" in (tmp_path / "ok.lte").read_text()


def test_freq_writes_nothing_and_stays_exact(tmp_path):
    lat = _one(Freq(name="FR", frequency_Hz=325e6))
    rep = Writer().write(lat, tmp_path / "fr.lte", strict=True)
    text = (tmp_path / "fr.lte").read_text()
    assert "FR" not in text
    assert rep.ok


def test_unused_definitions_are_still_written(tmp_path):
    ref = proton_ref()
    lat = Lattice(name="L", reference=ref)
    lat.add_element(Drift(name="USED", length=1.0))
    lat.add_element(Drift(name="SPARE", length=2.0))
    lat.lines["L"] = Line(name="L", items=[LineItem(ref="USED")])
    lat.use = "L"
    out = tmp_path / "u.lte"
    rep = Writer().write(lat, out)
    text = out.read_text()
    assert "SPARE: DRIF, L=2" in text and "LINE=(USED)" in text
    assert "DEFINITION_NOT_IN_LINE" in rep.codes()
