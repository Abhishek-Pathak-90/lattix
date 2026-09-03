"""Elegant ``.lte`` reader tests (PLAN §6 task 2.1).

Every convention asserted here was measured against ``elegant 2026.3.0`` itself
(see ``lattix/formats/elegant/reader.py``); the numbers are hand-computed, not
copied from the reader.  ``tests/data/public`` has no ``.lte`` (cheetah's and
ocelot's are GPL-3 and must not be vendored), so the file-level tests use inline
decks and the two private PIP-II anchors behind ``pytest.mark.corpus``.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.elegant import DEFAULT_FINT, DEFAULT_RF_FREQUENCY_HZ, Reader, logical_statements
from lattix.formats.elegant.reader import split_attrs, split_top
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.rf import elegant_phase_deg
from lattix.testing import corpus_dir


def read(tmp_path: Path, text: str, name: str = "deck.lte", **kw):
    p = tmp_path / name
    p.write_text(text)
    return Reader().read(p, **kw)


def brho(species_name: str = "proton", ke: float = 8e8) -> float:
    return ReferenceParticle(species=species(species_name), kinetic_energy_eV=ke).brho_signed


PROTON = {"species": "proton", "kinetic_energy_eV": 8e8}


# --------------------------------------------------------------------------- tokenizer
def test_comments_and_ampersand_continuation():
    text = "\n".join([
        "! a comment line",
        "Q1: QUAD, L=0.3, &   ! trailing comment after the ampersand",
        "     K1=0.6",
        "L1: LINE=(Q1)",
    ])
    stmts = logical_statements(text)
    assert [s for _, s in stmts] == ["Q1: QUAD, L=0.3, K1=0.6", "L1: LINE=(Q1)"]
    assert stmts[0][0] == 2                       # the statement's first line number


def test_trailing_comma_continues_only_when_the_next_line_is_not_a_statement():
    """elegant drops the tail of a comma-continued statement; lattix keeps it, but it
    must never glue two definitions together."""
    joined = [s for _, s in logical_statements("Q1: QUAD, L=0.3,\n  K1=0.6\nL1: LINE=(Q1)")]
    assert joined == ["Q1: QUAD, L=0.3, K1=0.6", "L1: LINE=(Q1)"]
    separate = [s for _, s in logical_statements("Q1: QUAD, L=0.3,\nQ2: QUAD, L=0.4")]
    assert separate == ["Q1: QUAD, L=0.3,", "Q2: QUAD, L=0.4"]


def test_unbalanced_parenthesis_continues_a_beam_line():
    text = "L1: LINE=(A,B,\n C,D)\nA: DRIF, L=1"
    assert [s for _, s in logical_statements(text)][0] == "L1: LINE=(A,B, C,D)"


def test_an_unterminated_beam_line_never_swallows_the_next_definition():
    """The PIP-II BTL/BAL anchors continue a LINE with '&' onto a line that is entirely a
    comment; without a guard the unbalanced '(' would eat the rest of the deck."""
    text = "\n".join(["L1: LINE=(A,B,&", "! ,C,D)", "L2: LINE=(A)", "A: DRIF, L=1"])
    joined = [s for _, s in logical_statements(text)]
    assert joined == ["L1: LINE=(A,B,", "L2: LINE=(A)", "A: DRIF, L=1"]


def test_a_template_definition_still_joins_across_a_trailing_comma():
    """The guard must not fire on a user-defined template base (QBASE is not a type)."""
    text = "QBASE: QUAD, L=0.1, K1=1\nQ1: QBASE,\n  K1=2\nL1: LINE=(Q1)"
    joined = [s for _, s in logical_statements(text)]
    assert joined[1] == "Q1: QBASE, K1=2"


def test_split_top_and_split_attrs_respect_quotes_and_nesting():
    assert split_top("A,(B,C),D") == ["A", "(B,C)", "D"]
    attrs, bare = split_attrs('L="LQ 2 *", FILENAME="a,b", ORDER')
    assert attrs == {"L": '"LQ 2 *"', "FILENAME": '"a,b"'}
    assert bare == ["ORDER"]


# --------------------------------------------------------------------------- statements
def test_rpn_store_variables(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "% 0.35 sto LQ",
        "% LQ 2 * sto LQ2",
        "% 30 dsin sto HALF",
        "D1: DRIF, L=LQ2",
        'D2: DRIF, L="LQ 1 +"',
        "L1: LINE=(D1,D2)",
    ]), **PROTON)
    assert lat.variables["LQ"].value == 0.35
    assert lat.variables["LQ2"].value == pytest.approx(0.7)
    assert lat.variables["HALF"].value == pytest.approx(0.5)
    assert lat.variables["LQ2"].expression.dialect == "rpn"
    assert lat.elements["D1"].length == pytest.approx(0.7)
    assert lat.elements["D2"].length == pytest.approx(1.35)
    assert rep.ok


def test_element_template_inherits_the_base_attributes(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "QBASE: QUAD, L=0.3, K1=0.6",
        "Q1: QBASE",
        "Q2: QBASE, K1=-0.6",
        "L1: LINE=(Q1,Q2)",
    ]), **PROTON)
    b = brho()
    assert lat.elements["Q1"].length == 0.3
    assert lat.elements["Q1"].multipole.Bn[1] == pytest.approx(0.6 * b)
    assert lat.elements["Q2"].length == 0.3
    assert lat.elements["Q2"].multipole.Bn[1] == pytest.approx(-0.6 * b)
    assert lat.elements["Q2"].provenance.original_type == "QUAD"


def test_name_property_override(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "Q1: QUAD, L=0.3, K1=0.6",
        "Q1[K1] = 2.5",
        "L1: LINE=(Q1)",
    ]), **PROTON)
    assert lat.elements["Q1"].multipole.Bn[1] == pytest.approx(2.5 * brho())
    assert "ELEMENT_PROPERTY_OVERRIDE" in rep.codes()


def test_override_of_an_unknown_element_is_dropped(tmp_path):
    _, rep = read(tmp_path, "NOPE[K1] = 1\nD: DRIF, L=1\nL1: LINE=(D)", **PROTON)
    assert "OVERRIDE_UNKNOWN_ELEMENT" in rep.codes()


def test_use_selects_the_root_and_otherwise_the_last_line(tmp_path):
    body = "D: DRIF, L=1\nA: LINE=(D)\nB: LINE=(D,D)\n"
    lat, _ = read(tmp_path, body, **PROTON)
    assert lat.use == "B"
    lat, _ = read(tmp_path, body + "USE, A", **PROTON)
    assert lat.use == "A"
    lat, _ = read(tmp_path, body, line="A", **PROTON)
    assert lat.use == "A"


def test_line_repeat_reverse_and_nested_sublists_are_preserved(tmp_path):
    lat, _ = read(tmp_path, "\n".join([
        "Q: QUAD, L=0.1, K1=1",
        "D: DRIF, L=0.2",
        "M: MARK",
        "CELL: LINE=(Q,3*D,M)",
        "RING: LINE=(CELL,-CELL,2*(Q,D))",
        "USE, RING",
    ]), **PROTON)
    ring = lat.lines["RING"]
    assert [(i.ref, i.repeat, i.reverse) for i in ring.items[:2]] == [("CELL", 1, False),
                                                                     ("CELL", 1, True)]
    sub = ring.items[2]
    assert sub.repeat == 2 and not sub.reverse and sub.ref in lat.lines
    assert [(i.ref, i.repeat) for i in lat.lines[sub.ref].items] == [("Q", 1), ("D", 1)]
    flat = [p.element.name for p in lat.flatten()]
    assert flat == ["Q", "D", "D", "D", "M",                  # CELL
                    "M", "D", "D", "D", "Q",                  # -CELL
                    "Q", "D", "Q", "D"]                       # 2*(Q,D)
    assert lat.total_length == pytest.approx(2 * (0.1 + 0.6) + 2 * 0.3)


def test_case_insensitive_names_keep_their_source_spelling(tmp_path):
    lat, _ = read(tmp_path, "o1aBQDD: drif, l=0.5\nl1: line=(O1ABQDD)\nuse,L1", **PROTON)
    assert "o1aBQDD" in lat.elements
    assert lat.lines["l1"].items[0].ref == "o1aBQDD"
    assert lat.total_length == 0.5


def test_return_ends_the_deck(tmp_path):
    lat, _ = read(tmp_path, "D: DRIF, L=1\nL1: LINE=(D)\nUSE,L1\nRETURN\nX: DRIF, L=9", **PROTON)
    assert "X" not in lat.elements


# --------------------------------------------------------------------------- elements
def test_drifts_including_the_collective_ones(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "D1: DRIF, L=0.5",
        "D2: DRIFT, L=0.5",
        "D3: EDRIFT, L=0.5",
        "D4: CSRDRIFT, L=0.5, N_KICKS=10",
        "D5: LSCDRIFT, L=0.5",
        "L1: LINE=(D1,D2,D3,D4,D5)",
    ]), **PROTON)
    assert all(lat.elements[f"D{i}"].kind == "Drift" for i in range(1, 6))
    assert lat.total_length == pytest.approx(2.5)
    assert rep.codes()["COLLECTIVE_DRIFT_AS_DRIFT"] == 2
    assert lat.elements["D4"].native["elegant"]["attrs"]["N_KICKS"] == "10"


def test_quadrupole_k1_uses_the_signed_rigidity_and_tilt_is_the_design_roll(tmp_path):
    lat, rep = read(tmp_path,
                    "Q: KQUAD, L=0.3, K1=0.6, TILT=0.1, DX=1e-3, DY=2e-3, DZ=3e-3, PITCH=1e-4\n"
                    "L1: LINE=(Q)", **PROTON)
    q = lat.elements["Q"]
    assert q.kind == "Quadrupole"
    assert q.multipole.Bn[1] == pytest.approx(0.6 * brho())
    assert q.multipole.tilt[1] == 0.1                     # design roll, not a misalignment
    assert (q.shift.x_offset, q.shift.y_offset, q.shift.z_offset) == (1e-3, 2e-3, 3e-3)
    assert q.shift.x_rot == pytest.approx(-1e-4)          # elegant PITCH is -x_rot
    assert rep.ok


def test_quadrupole_k1_flips_sign_for_a_negative_species(tmp_path):
    """K1 -> G goes through the SIGNED rigidity, so the same deck read as an antiproton
    gives exactly the opposite gradient (same mass, opposite charge)."""
    text = "Q: QUAD, L=0.3, K1=0.6\nL1: LINE=(Q)"
    p, _ = read(tmp_path, text, species="proton", kinetic_energy_eV=8e8, name="p.lte")
    a, _ = read(tmp_path, text, species="antiproton", kinetic_energy_eV=8e8, name="a.lte")
    assert p.elements["Q"].multipole.Bn[1] == pytest.approx(-a.elements["Q"].multipole.Bn[1])
    h, _ = read(tmp_path, text, species="h-", kinetic_energy_eV=8e8, name="h.lte")
    assert h.elements["Q"].multipole.Bn[1] < 0 < p.elements["Q"].multipole.Bn[1]


def test_sextupole_octupole_and_multipole(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "S: KSEXT, L=0.1, K2=2.0",
        "O: KOCT, L=0.1, K3=3.0",
        "M: MULT, KNL=0.01, ORDER=2, FACTOR=2",
        "L1: LINE=(S,O,M)",
    ]), **PROTON)
    b = brho()
    assert lat.elements["S"].multipole.Bn[2] == pytest.approx(2.0 * b)
    assert lat.elements["O"].multipole.Bn[3] == pytest.approx(3.0 * b)
    assert lat.elements["M"].kind == "Multipole"
    assert lat.elements["M"].multipole.BnL[2] == pytest.approx(0.02 * b)   # KNL * FACTOR


def test_multipole_default_order_is_one(tmp_path):
    lat, _ = read(tmp_path, "M: MULT, KNL=0.03\nL1: LINE=(M)", **PROTON)
    assert set(lat.elements["M"].multipole.BnL) == {1}


def test_sector_bend_edges_fint_and_combined_function(tmp_path):
    lat, rep = read(tmp_path,
                    "B: CSBEND, L=1.0, ANGLE=0.1, E1=0.02, E2=0.03, HGAP=0.02, FINT=0.4, "
                    "K1=0.5, TILT=1.5707963267948966, ETILT=1e-3\nL1: LINE=(B)", **PROTON)
    b = lat.elements["B"]
    assert b.kind == "Bend" and b.length == 1.0
    assert (b.bend.angle, b.bend.e1, b.bend.e2) == (0.1, 0.02, 0.03)
    assert b.bend.edge_int1 == 0.4 and b.bend.edge_int2 is None
    assert b.bend.hgap == 0.02
    assert b.bend.tilt_ref == pytest.approx(math.pi / 2)   # vertical bend, a design property
    assert b.shift.tilt == pytest.approx(1e-3)             # ETILT is the misalignment
    assert b.multipole.Bn[1] == pytest.approx(0.5 * brho())
    assert not b.bend.rect


def test_bend_fint_default_is_half(tmp_path):
    lat, _ = read(tmp_path, "B: SBEN, L=1, ANGLE=0.1, HGAP=0.03\nL1: LINE=(B)", **PROTON)
    assert lat.elements["B"].bend.edge_int1 == DEFAULT_FINT == 0.5


def test_csbend_fint1_fint2_override_fint(tmp_path):
    lat, _ = read(tmp_path, "B: CSBEND, L=1, ANGLE=0.1, FINT1=0.3, FINT2=0.7\nL1: LINE=(B)",
                  **PROTON)
    assert lat.elements["B"].bend.edge_int1 == 0.3
    assert lat.elements["B"].bend.edge_int2 == 0.7


def test_rben_length_is_the_chord(tmp_path):
    """Measured with elegant's own &save_lattice: RBEN L=1, ANGLE=0.1 becomes
    SBEN L=1.000416788226488, E1=E2=0.05."""
    lat, rep = read(tmp_path, "B: RBEN, L=1.0, ANGLE=0.1\nL1: LINE=(B)", **PROTON)
    b = lat.elements["B"]
    assert b.length == pytest.approx(1.000416788226488, abs=1e-14)
    assert b.bend.e1 == pytest.approx(0.05) and b.bend.e2 == pytest.approx(0.05)
    assert b.bend.rect
    assert "RBEN_CHORD_TO_ARC" in rep.codes()


def test_rben_pole_faces_add_to_the_source_values(tmp_path):
    lat, _ = read(tmp_path, "B: RBEND, L=1.0, ANGLE=0.2, E1=0.01, E2=-0.01\nL1: LINE=(B)", **PROTON)
    assert lat.elements["B"].bend.e1 == pytest.approx(0.11)
    assert lat.elements["B"].bend.e2 == pytest.approx(0.09)


def test_solenoid_ks_and_b(tmp_path):
    lat, _ = read(tmp_path, "S1: SOLE, L=0.3, KS=0.5\nS2: SOLE, L=0.3, B=1.25\nL1: LINE=(S1,S2)",
                  **PROTON)
    assert lat.elements["S1"].solenoid.Bsol_T == pytest.approx(0.5 * brho())
    assert lat.elements["S2"].solenoid.Bsol_T == 1.25


@pytest.mark.parametrize("sp,crest", [("proton", -90.0), ("h-", +90.0), ("electron", +90.0)])
def test_rfca_phase_convention_follows_the_charge_sign(tmp_path, sp, crest):
    """phi_s = -30 deg must be PHASE = -30 + crest, and the walk must gain +V*cos30."""
    phase_deg = -30.0 + crest
    lat, rep = read(tmp_path,
                    f"C: RFCA, L=0, VOLT=1e6, PHASE={phase_deg}, FREQ=325e6, CHANGE_P0=1\n"
                    "L1: LINE=(C)", species=sp, kinetic_energy_eV=2.1e6)
    cav = lat.elements["C"]
    assert math.degrees(cav.rf.phase_rad) == pytest.approx(-30.0)
    assert cav.rf.voltage_V * math.cos(cav.rf.phase_rad) == pytest.approx(1e6 * math.cos(
        math.radians(30)))
    assert elegant_phase_deg(cav.rf.phase_rad, lat.reference.species.charge) % 360 == \
        pytest.approx(phase_deg % 360)
    assert "SPECIES_ASSUMED" in rep.codes()


def test_rfca_default_frequency_and_change_p0(tmp_path):
    lat, rep = read(tmp_path, "C: RFCA, L=0.25, VOLT=1e6, PHASE=90\nL1: LINE=(C)",
                    species="electron", kinetic_energy_eV=1e8)
    assert lat.elements["C"].rf.frequency_Hz == DEFAULT_RF_FREQUENCY_HZ == 500e6
    codes = rep.codes()
    assert "RFCA_DEFAULT_FREQ" in codes and "RFCA_NO_P0_CHANGE" in codes


def test_kickers(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "H: HKICK, L=0.1, KICK=1e-3",
        "V: VKICK, L=0.1, KICK=2e-3",
        "K: KICKER, L=0.1, HKICK=1e-3, VKICK=-2e-3",
        "E: EHKICK, L=0, KICK=5e-4",
        "L1: LINE=(H,V,K,E)",
    ]), **PROTON)
    assert (lat.elements["H"].hkick, lat.elements["H"].vkick) == (1e-3, 0.0)
    assert (lat.elements["V"].hkick, lat.elements["V"].vkick) == (0.0, 2e-3)
    assert (lat.elements["K"].hkick, lat.elements["K"].vkick) == (1e-3, -2e-3)
    assert lat.elements["E"].electric and not lat.elements["H"].electric
    assert rep.ok


def test_collimators_maxamp_and_apertures(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "E1: ECOL, L=0.1, X_MAX=0.02, Y_MAX=0.01",
        "R1: RCOL, L=0.0, X_MAX=0.03, Y_MAX=0.04",
        "MA: MAXAMP, X_MAX=0.05, Y_MAX=0.05, ELLIPTICAL=1",
        "L1: LINE=(E1,R1,MA)",
    ]), **PROTON)
    e, r, ma = (lat.elements[k] for k in ("E1", "R1", "MA"))
    assert e.kind == r.kind == ma.kind == "Collimator"
    assert e.aperture.shape == "ELLIPTICAL" and (e.aperture.half_x, e.aperture.half_y) == (0.02,
                                                                                           0.01)
    assert r.aperture.shape == "RECTANGULAR" and r.aperture.half_x == 0.03
    assert ma.aperture.shape == "ELLIPTICAL" and ma.length == 0.0
    assert "MAXAMP_AS_COLLIMATOR" in rep.codes()


def test_markers_and_diagnostics(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "M: MARK",
        "BPM: MONI, L=0.05",
        "HM: HMON",
        "VM: VMON",
        'W: WATCH, FILENAME="%s.w1"',
        "L1: LINE=(M,BPM,HM,VM,W)",
    ]), **PROTON)
    assert lat.elements["M"].kind == "Marker"
    fam = {n: lat.elements[n].family for n in ("BPM", "HM", "VM", "W")}
    assert fam == {"BPM": "BPM", "HM": "HMON", "VM": "VMON", "W": "WATCH"}
    assert lat.elements["W"].params["filename"] == "%s.w1"
    assert rep.ok


def test_ematrix_first_order_becomes_a_taylor(tmp_path):
    lat, rep = read(tmp_path,
                    "E: EMATRIX, L=0.5, ORDER=1, R11=1.5, R12=2.0, R56=-0.3, C1=1e-3\n"
                    "L1: LINE=(E)", **PROTON)
    t = lat.elements["E"]
    assert t.kind == "Taylor" and t.length == 0.5
    assert t.matrix[0][0] == 1.5 and t.matrix[0][1] == 2.0 and t.matrix[4][5] == -0.3
    assert t.offset[0] == 1e-3
    assert t.matrix[2][2] == 1.0                        # untouched diagonal stays identity
    assert rep.ok


def test_ematrix_higher_order_is_lossy(tmp_path):
    lat, rep = read(tmp_path, "E: EMATRIX, L=0, ORDER=2, T111=1\nL1: LINE=(E)", **PROTON)
    assert lat.elements["E"].kind == "Marker"
    assert "EMATRIX_HIGHER_ORDER" in rep.codes()


def test_beam_data_and_physics_and_unsupported_types(tmp_path):
    lat, rep = read(tmp_path, "\n".join([
        "Q: CHARGE, TOTAL=0.02e-9",
        "SC: SCATTER, DP=1e-4",
        "DF: RFDF, L=0.2, VOLTAGE=1e3, FREQUENCY=1e8",
        "ZZ: NOTATYPE, L=0.4",
        "YY: ALSONOTATYPE",
        "L1: LINE=(Q,SC,DF,ZZ,YY)",
    ]), **PROTON)
    assert lat.elements["Q"].kind == "Directive" and lat.elements["Q"].role == "beam"
    assert lat.elements["Q"].args == ["TOTAL=0.02e-9"]
    assert lat.elements["SC"].kind == "Marker"
    assert lat.elements["DF"].kind == "Marker"
    assert lat.elements["ZZ"].kind == "Drift" and lat.elements["ZZ"].length == 0.4
    assert lat.elements["YY"].kind == "Marker"
    codes = rep.codes()
    assert codes["BEAM_DATA_DROPPED"] == 1
    assert codes["PHYSICS_ELEMENT_DROPPED"] == 1
    assert codes["UNSUPPORTED_ELEGANT_TYPE"] == 3


def test_native_keeps_every_raw_attribute(tmp_path):
    lat, rep = read(tmp_path, "Q: KQUAD, L=0.3, K1=0.6, N_KICKS=8, SYNCH_RAD=1\nL1: LINE=(Q)",
                    **PROTON)
    nat = lat.elements["Q"].native["elegant"]
    assert nat["type"] == "KQUAD"
    assert nat["attrs"] == {"L": "0.3", "K1": "0.6", "N_KICKS": "8", "SYNCH_RAD": "1"}
    assert set(nat["consumed"]) >= {"L", "K1"}
    assert "UNMAPPED_ELEGANT_ATTRIBUTE" in rep.codes()


def test_rigidity_uses_the_local_reference_energy_through_a_cavity(tmp_path):
    """A quad after an accelerating cavity converts K1 with the *post-cavity* Bρ."""
    text = "\n".join([
        "C: RFCA, L=0, VOLT=1e8, PHASE=-90, FREQ=325e6, CHANGE_P0=1",
        "Q: QUAD, L=0.3, K1=0.6",
        "L1: LINE=(Q,C,Q)",
    ])
    lat, rep = read(tmp_path, text, species="proton", kinetic_energy_eV=1e8)
    placed = lat.flatten()
    assert len(placed) == 3
    b_in = ReferenceParticle(species=species("proton"), kinetic_energy_eV=1e8).brho_signed
    assert lat.elements["Q"].multipole.Bn[1] == pytest.approx(0.6 * b_in)
    assert "MULTI_RIGIDITY_DEFINITION" in rep.codes()


# --------------------------------------------------------------------------- downgrades
DOWNGRADES = {
    "BEAM_DATA_DROPPED": "Q: CHARGE, TOTAL=1e-9\nD: DRIF, L=1\nL1: LINE=(Q,D)",
    "PHYSICS_ELEMENT_DROPPED": "S: SCATTER, DP=1e-4\nD: DRIF, L=1\nL1: LINE=(S,D)",
    "UNSUPPORTED_ELEGANT_TYPE": "Z: NOSUCHTYPE, L=1\nL1: LINE=(Z)",
    "UNPARSED_STATEMENT": "this is not elegant\nD: DRIF, L=1\nL1: LINE=(D)",
    "UNDEFINED_LINE_MEMBER": "D: DRIF, L=1\nL1: LINE=(D,GHOST)",
    "NO_BEAMLINE": "D: DRIF, L=1",
    "UNKNOWN_ROOT_LINE": "D: DRIF, L=1\nL1: LINE=(D)\nUSE, NOPE",
    "OVERRIDE_UNKNOWN_ELEMENT": "GHOST[K1]=1\nD: DRIF, L=1\nL1: LINE=(D)",
    "BARE_ATTRIBUTE_TOKEN": "D: DRIF, , L\nL1: LINE=(D)",
    "UNRESOLVED_EXPRESSION": "D: DRIF, L=NOSUCHVAR\nL1: LINE=(D)",
    "EMATRIX_HIGHER_ORDER": "E: EMATRIX, ORDER=3\nL1: LINE=(E)",
    "KICKER_TILT_KEPT_NATIVE": "K: KICKER, L=0, HKICK=1e-3, TILT=0.5\nL1: LINE=(K)",
    "BEND_POLE_CURVATURE_DROPPED": "B: CSBEND, L=1, ANGLE=0.1, H1=0.2\nL1: LINE=(B)",
    "ELEGANT_INCLUDE": "#include: other.lte\nD: DRIF, L=1\nL1: LINE=(D)",
    "RPN_STORE_UNRESOLVED": "% NOPE 2 * sto X\nD: DRIF, L=1\nL1: LINE=(D)",
    "RPN_COMMAND_DROPPED": "% 1 2 +\nD: DRIF, L=1\nL1: LINE=(D)",
    "UNTERMINATED_LINE": "D: DRIF, L=1\nL1: LINE=(D,&\n! everything here is a comment\nL2: LINE=(D)",
}


@pytest.mark.parametrize("code", sorted(DOWNGRADES))
def test_every_downgrade_is_recorded_in_permissive_mode(tmp_path, code):
    _, rep = read(tmp_path, DOWNGRADES[code], **PROTON)
    assert code in rep.codes(), rep.summary()


@pytest.mark.parametrize("code", sorted(DOWNGRADES))
def test_every_downgrade_raises_in_strict_mode(tmp_path, code):
    with pytest.raises(TranslationError) as e:
        read(tmp_path, DOWNGRADES[code], strict=True, **PROTON)
    assert e.value.entry.cls in ("LOSSY", "DROPPED")


@pytest.mark.parametrize("code", ["COLLECTIVE_DRIFT_AS_DRIFT", "MAXAMP_AS_COLLIMATOR",
                                  "RBEN_CHORD_TO_ARC", "RFCA_NO_P0_CHANGE", "SPECIES_ASSUMED",
                                  "ENERGY_ASSUMED", "ROOT_LINE_IS_LAST",
                                  "UNMAPPED_ELEGANT_ATTRIBUTE"])
def test_equivalent_codes_never_raise_in_strict_mode(tmp_path, code):
    text = "\n".join([
        "D: CSRDRIFT, L=1",
        "MA: MAXAMP, X_MAX=0.01, Y_MAX=0.01",
        "B: RBEN, L=1, ANGLE=0.1",
        "C: RFCA, L=0, VOLT=1e6, PHASE=90, FREQ=1e8",
        "Q: KQUAD, L=0.1, K1=1, N_KICKS=4",
        "L1: LINE=(D,MA,B,C,Q)",
    ])
    _, rep = read(tmp_path, text, strict=True, species="electron")
    assert code in rep.codes(), rep.summary()


# --------------------------------------------------------------------------- corpus
CORPUS_IDS = {
    "btl": "pipii-anchors/studies_and_related_material/beam_dynamics_studies/btl/"
           "btl_lattice_with_spacecharge/mad_lattice/elegant_lattice.lte",
    "bal": "pipii-anchors/studies_and_related_material/beam_dynamics_studies/bal/"
           "bal_lattice_with_spacecharge/mad_lattice/elegant_lattice.lte",
    "hwr": "pipii-anchors/studies_and_related_material/virtual-accelerator/prototypes/"
           "v1_14_august_2024/tracewin_elegant_lattice.lte",
}


def corpus_deck(key: str) -> Path:
    from lattix.corpus import load_manifest, resolve_path

    root = corpus_dir()
    if root is None:
        pytest.skip("LATTIX_CORPUS_DIR is not set")
    try:
        entries = load_manifest(root)
    except FileNotFoundError as e:                       # pragma: no cover - environment
        pytest.skip(str(e))
    for e in entries:
        if e.get("id") == CORPUS_IDS[key]:
            p = resolve_path(root, e)
            if not p.is_file():
                pytest.skip(f"corpus entry {key} is listed but absent: {p}")
            return p
    pytest.skip(f"corpus manifest has no entry {CORPUS_IDS[key]!r}")


@pytest.mark.corpus
def test_corpus_hwr_cryomodule():
    """The PIP-II HWR cryomodule .lte: 34 definitions, 8 cavities, 8 solenoids, 5.9012 m."""
    lat, rep = corpus_read("hwr")
    kinds = Counter(p.element.kind for p in lat.flatten())
    assert len(lat.elements) == 34
    assert kinds == {"Drift": 18, "Solenoid": 8, "RFCavity": 8}
    assert lat.total_length == pytest.approx(5.901207, abs=1e-6)
    assert lat.use == "HWR_CM"
    codes = rep.codes()
    # the deck writes APERTURE on DRIFT/SOLENOID/RFCA, which real elegant rejects
    assert codes["UNMAPPED_ELEGANT_ATTRIBUTE"] == 34
    assert codes["RFCA_NO_P0_CHANGE"] == 8
    assert lat.elements["CAV1"].rf.frequency_Hz == 162.5e6
    # PHASE = -49.9998 deg read as H- (crest +90) -> phi_s = -139.9998 deg
    assert math.degrees(lat.elements["CAV1"].rf.phase_rad) == pytest.approx(-139.9998, abs=1e-9)


@pytest.mark.corpus
@pytest.mark.parametrize("key", ["btl", "bal"])
def test_corpus_btl_bal_skeleton_decks(key):
    """Both PIP-II BTL/BAL anchors are *value-less* skeletons: every definition reads
    ``NAME: TYPE, , L`` with no ``=``, so elegant itself refuses them
    (``get_param_name(): no parameter name found in string L``).  lattix reads the
    structure, zeroes the geometry and says so in the ledger."""
    lat, rep = corpus_read(key)
    kinds = Counter(p.element.kind for p in lat.flatten())
    assert len(lat.elements) == 192
    assert len(lat.lines) == 78
    assert lat.use == "BTL2020V0929NBS"
    assert kinds == {"Drift": 527, "Quadrupole": 53, "Bend": 40}
    assert lat.total_length == 0.0
    codes = rep.codes()
    assert codes["BARE_ATTRIBUTE_TOKEN"] == 192
    # SS04 and SS05 each have their whole continuation line commented out
    assert codes["UNTERMINATED_LINE"] == 2
    assert codes["UNDEFINED_LINE_MEMBER"] == 276


def corpus_read(key: str):
    return Reader().read(corpus_deck(key), species="h-", kinetic_energy_eV=8e8)
