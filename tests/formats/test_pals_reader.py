"""PALS reader tests (PLAN §6 task 2.4).

The vendored files under ``tests/data/public/pals`` come from the PALS standard repository
itself (CC-BY-4.0, commit ``a2b1083``; see that directory's README), so every number checked
against them is the standard's own worked example.  Inline documents cover the constructs the
examples do not: ``include``, ``load``, expressions, ``direction: -1``, ``zero_phase``,
apertures, kickers, ``UnionEle`` and the ``pals-schema`` 0.3.0 spellings.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.pals import Reader
from lattix.ir import (
    Bend,
    Collimator,
    Drift,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    Patch,
    ReferenceChange,
    RFCavity,
    Solenoid,
    Superposition,
)
from lattix.ir.units import turns_to_rad

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "pals"
IMPACTX = Path(__file__).resolve().parents[1] / "data" / "public" / "impactx"


def read(path, **kw):
    return Reader().read(Path(path), **kw)


def parse(tmp_path, text: str, name: str = "t.pals.yaml", **kw):
    p = tmp_path / name
    p.write_text(text)
    return read(p, **kw)


# ── the standard's own examples ───────────────────────────────────────────────────────────
def test_fodo_example_expands_to_three_cells():
    """``examples/fodo.pals.yaml``: 3 × (drift1, quad1, drift2, quad2, drift1) = 15 elements."""
    lat, rep = read(DATA / "fodo.pals.yaml")
    placed = lat.flatten()
    assert len(placed) == 15
    assert sum(p.length for p in placed) == pytest.approx(9.0)
    assert [p.element.kind for p in placed[:5]] == \
        ["Drift", "Quadrupole", "Drift", "Quadrupole", "Drift"]
    # `drift1` is referenced twice in the cell: one definition, two placements
    assert placed[0].element is placed[4].element
    # quad2 `inherit: quad1` keeps the length and overrides Bn1
    assert lat.elements["quad2"].length == 1.0
    assert lat.elements["quad2"].multipole.Bn[1] == -1.0
    assert lat.elements["quad1"].multipole.Bn[1] == 1.0
    assert lat.use == "fodo_channel"
    assert set(lat.lines) == {"fodo_channel", "fodo_cell"}
    assert lat.meta["pals_lattice"] == "fodo_lattice"
    assert rep.ok


def test_iota_example_ring():
    """``examples/iota.pals.yaml``: the real IOTA circumference, `Kn1` through the rigidity,
    an inline ``BeginningEle`` and ``repeat: -1`` (reversed element order)."""
    lat, rep = read(DATA / "iota.pals.yaml")
    placed = lat.flatten()
    assert len(placed) == 101
    assert sum(p.length for p in placed) == pytest.approx(39.968229715, abs=1e-9)

    ref = lat.reference
    assert ref.species.name == "proton"
    assert ref.kinetic_energy_eV == pytest.approx(2.5e6, rel=1e-9)

    # Kn1 = -8.78017699 [1/m^2] -> Bn1 = Kn1 * Brho(2.5 MeV proton)
    qa1 = lat.elements["qa1"]
    assert qa1.multipole.Bn[1] == pytest.approx(-8.78017699 * ref.brho_signed, rel=1e-12)

    # radius_ref + length -> angle; edge integrals are the fint*hgap product
    sb30 = lat.elements["sbend30"]
    assert sb30.bend.angle == pytest.approx(0.4305191429 / 0.822230996255981, rel=1e-12)
    assert math.degrees(sb30.bend.angle) == pytest.approx(30.0, abs=2e-3)
    assert sb30.bend.edge_int1 * sb30.bend.hgap == pytest.approx(0.0145)

    # monitor(0), half(1..49), qe3(50), mirrored half(51..99), monitor(100)
    names = [p.element.name for p in placed]
    assert names[0] == names[100] == "monitor" and names[50] == "qe3"
    assert names[1:50] == names[99:50:-1]
    assert not any(p.reversed for p in placed)          # order-only reversal
    assert "PALS_REVERSED_ORDER_EXPANDED" in rep.codes()
    assert lat.meta["pals_periodic"] == ["iota_ring"]
    assert lat.meta["pals_extensions"]["impactx_setup"]["ImpactX"]["npart"] == 10000
    assert rep.ok


def test_bend_given_by_angle_and_radius_derives_its_length():
    """``unit_tests/elements/bend_angle_radius``: angle_ref 0.2 with radius_ref 5 -> L = 1.0."""
    lat, rep = read(DATA / "bend_angle_radius.pals.yaml")
    bnd = lat.elements["bnd"]
    assert isinstance(bnd, Bend)
    assert bnd.length == pytest.approx(1.0)
    assert bnd.bend.angle == pytest.approx(0.2)
    assert bnd.rho == pytest.approx(5.0)
    assert lat.reference.species.name == "electron"
    assert lat.reference.total_energy_eV == pytest.approx(1.0e9)
    assert rep.ok


@pytest.mark.parametrize(("name", "voltage", "gradient"),
                         [("rf_voltage", 8.0e6, 2.0e7), ("rf_gradient", 4.0e7, 1.0e8)])
def test_rf_voltage_and_gradient_are_linked_by_the_active_length(name, voltage, gradient):
    """``parameters/rf.md``: ``voltage = gradient * L_active``, ``L_active`` defaults to ``L``."""
    lat, rep = read(DATA / f"{name}.pals.yaml")
    cav = lat.elements["cav"]
    assert isinstance(cav, RFCavity)
    assert cav.length == pytest.approx(0.4)
    assert cav.rf.voltage_V == pytest.approx(voltage)
    assert cav.rf.gradient_V_per_m == pytest.approx(gradient)
    assert cav.rf.voltage_V == pytest.approx(cav.rf.gradient_V_per_m * cav.length)
    assert rep.ok


def test_drift_quad_bend_example():
    lat, rep = read(DATA / "drift_quad_bend.pals.yaml")
    placed = lat.flatten()
    assert [p.element.kind for p in placed] == ["Drift", "Quadrupole", "Bend", "Drift", "Marker"]
    assert sum(p.length for p in placed) == pytest.approx(5.0)
    assert lat.elements["q1"].multipole.Bn[1] == pytest.approx(1.2)
    assert lat.elements["b1"].bend.angle == pytest.approx(0.15)
    assert lat.elements["b1"].bend.g_ref(1.5) == pytest.approx(0.1)
    assert rep.ok


def test_impactx_bare_beamline_document_is_read_with_a_ledger_entry():
    """ImpactX's ``examples/pals/fodo.pals.yaml`` has no ``PALS:`` root node."""
    lat, rep = read(IMPACTX / "fodo.pals.yaml")
    assert len(lat.flatten()) == 5
    assert sum(p.length for p in lat.flatten()) == pytest.approx(3.0)
    assert "PALS_ROOT_MISSING" in rep.codes()
    assert rep.ok                      # EQUIVALENT, never fatal


# ── line construction ─────────────────────────────────────────────────────────────────────
_HEAD = """
PALS:
  facility:
    - d:
        kind: Drift
        length: 1.0
"""


def test_repeat_and_direction(tmp_path):
    lat, rep = parse(tmp_path, _HEAD + """
    - b:
        kind: Bend
        length: 2.0
        BendP: {angle_ref: 0.1, e1: 0.05, e2: 0.0}
    - cell:
        kind: BeamLine
        line: [d, b]
    - main:
        kind: BeamLine
        line:
          - cell: {repeat: 2}
          - cell: {direction: -1}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    placed = lat.flatten()
    assert [p.element.name for p in placed] == ["d", "b", "d", "b", "b", "d"]
    assert [p.reversed for p in placed] == [False] * 4 + [True, True]
    assert rep.ok


def test_negative_repeat_reverses_order_without_reversing_elements(tmp_path):
    """``beamlines.md`` §Repetition: 'reverse order does not mean direction reversal'."""
    lat, rep = parse(tmp_path, _HEAD + """
    - b:
        kind: Bend
        length: 2.0
        BendP: {angle_ref: 0.1, e1: 0.05, e2: 0.0}
    - cell:
        kind: BeamLine
        line: [d, b]
    - main:
        kind: BeamLine
        line:
          - cell: {repeat: -2}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    placed = lat.flatten()
    assert [p.element.name for p in placed] == ["b", "d", "b", "d"]
    assert not any(p.reversed for p in placed)          # order only, e1/e2 keep their ends
    assert placed[0].element.bend.e1 == pytest.approx(0.05)
    assert "cell__reversed" in lat.lines
    assert "PALS_REVERSED_ORDER_EXPANDED" in rep.codes()


def test_in_place_definition_and_per_occurrence_override(tmp_path):
    lat, _ = parse(tmp_path, _HEAD + """
    - q:
        kind: Quadrupole
        length: 0.3
        MagneticMultipoleP: {Bn1: 2.0}
    - main:
        kind: BeamLine
        line:
          - q
          - q_strong:
              inherit: q
              MagneticMultipoleP: {Bn1: 3.0}
          - q2:
              kind: Quadrupole
              length: 0.4
              MagneticMultipoleP: {Bn1: 1.0}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    placed = lat.flatten()
    assert [p.element.multipole.Bn[1] for p in placed] == [2.0, 3.0, 1.0]
    assert placed[1].element.length == 0.3               # inherited from q
    assert placed[2].element.length == 0.4


def test_use_option_selects_a_branch_or_beamline(tmp_path):
    text = _HEAD + """
    - a: {kind: BeamLine, line: [d]}
    - b: {kind: BeamLine, line: [d, d]}
    - lat1: {kind: Lattice, branches: [a]}
    - lat2: {kind: Lattice, branches: [b]}
"""
    lat, _ = parse(tmp_path, text)                       # default: the LAST Lattice
    assert len(lat.flatten()) == 2
    lat, _ = parse(tmp_path, text + "    - use: lat1\n", name="u.pals.yaml")
    assert len(lat.flatten()) == 1
    lat, _ = parse(tmp_path, text, name="v.pals.yaml", use="b")
    assert len(lat.flatten()) == 2
    lat, rep = parse(tmp_path, text, name="w.pals.yaml", use="nope")
    assert "PALS_USE_NOT_FOUND" in rep.codes()


def test_extra_branches_are_reported(tmp_path):
    lat, rep = parse(tmp_path, _HEAD + """
    - a: {kind: BeamLine, line: [d]}
    - b: {kind: BeamLine, line: [d, d]}
    - lat: {kind: Lattice, branches: [a, b]}
    - use: lat
""")
    assert len(lat.flatten()) == 1
    assert "PALS_EXTRA_BRANCHES" in rep.codes()


# ── expressions, constants, include ───────────────────────────────────────────────────────
def test_constants_variables_and_expressions(tmp_path):
    lat, rep = parse(tmp_path, """
PALS:
  facility:
    - constants:
        - lq: 0.3
        - twice: 2 * lq
    - variables:
        - g: 1.5
    - q:
        kind: Quadrupole
        length: twice
        MagneticMultipoleP: {Bn1: "g * 2"}
    - c:
        kind: Drift
        length: "c_light * 1e-9"
    - main: {kind: BeamLine, line: [q, c]}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    assert lat.elements["q"].length == pytest.approx(0.6)
    assert lat.elements["q"].multipole.Bn[1] == pytest.approx(3.0)
    assert lat.elements["c"].length == pytest.approx(0.299792458)
    assert lat.variables["lq"].value == 0.3
    assert rep.ok


def test_unevaluable_expression_is_lossy_not_fatal(tmp_path):
    lat, rep = parse(tmp_path, _HEAD.replace("length: 1.0", 'length: "no_such_function(2.7)"') + """
    - main: {kind: BeamLine, line: [d]}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    assert lat.elements["d"].length == 0.0
    assert "PALS_EXPRESSION_UNEVALUABLE" in rep.codes()
    with pytest.raises(TranslationError):
        parse(tmp_path, _HEAD.replace("length: 1.0", 'length: "no_such_function(2.7)"') + """
    - main: {kind: BeamLine, line: [d]}
""", name="s.pals.yaml", strict=True)


def test_include_is_followed_relative_to_the_including_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "q.subpals.yaml").write_text(
        "MagneticMultipoleP:\n  Bn1: 4.0\n")
    (tmp_path / "els.subpals.yaml").write_text("""
- q:
    kind: Quadrupole
    length: 0.5
    include: sub/q.subpals.yaml
""")
    lat, rep = parse(tmp_path, """
PALS:
  facility:
    - include: els.subpals.yaml
    - main: {kind: BeamLine, line: [q]}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    assert lat.elements["q"].multipole.Bn[1] == 4.0
    assert rep.ok


def test_missing_include_is_lossy(tmp_path):
    lat, rep = parse(tmp_path, """
PALS:
  facility:
    - include: nowhere.yaml
    - d: {kind: Drift, length: 1.0}
    - main: {kind: BeamLine, line: [d]}
""")
    assert "INCLUDE_NOT_FOLLOWED" in rep.codes()
    assert len(lat.flatten()) == 1


def test_load_merges_whole_pals_nodes(tmp_path):
    (tmp_path / "layout.pals.yaml").write_text("""
PALS:
  notes: [layout]
  facility:
    - d: {kind: Drift, length: 2.0}
    - main: {kind: BeamLine, line: [d, d]}
    - lat: {kind: Lattice, branches: [main]}
""")
    lat, rep = parse(tmp_path, """
PALS:
  load:
    - layout.pals.yaml
    - SELF
  notes: [joiner]
  facility:
    - use: lat
""")
    assert len(lat.flatten()) == 2
    assert lat.meta["pals_notes"] == ["layout", "joiner"]


# ── reference particle ────────────────────────────────────────────────────────────────────
def test_beginning_ele_sets_the_reference_and_leaves_the_line(tmp_path):
    lat, rep = parse(tmp_path, _HEAD + """
    - main:
        kind: BeamLine
        line:
          - begin:
              kind: BeginningEle
              ReferenceP: {species_ref: electron, E_tot_ref: 1.0e+9, time_ref: 1.0e-6}
          - d
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    assert [p.element.kind for p in lat.flatten()] == ["Drift"]     # BeginningEle is not an element
    assert lat.reference.species.name == "electron"
    assert lat.reference.total_energy_eV == pytest.approx(1e9)
    assert lat.reference.time_s == pytest.approx(1e-6)
    assert lat.meta["pals_beginning"] == "begin"
    assert "PALS_BEGINNING_AS_REFERENCE" in {e.code for e in rep.entries}


def test_reference_from_pc_ref_and_species_aliases(tmp_path):
    lat, _ = parse(tmp_path, _HEAD + """
    - main:
        kind: BeamLine
        line:
          - begin:
              kind: BeginningEle
              ReferenceP: {species_ref: "#1H-1", pc_ref: 1.0e+9}
          - d
""")
    assert lat.reference.species.name == "h-"
    assert lat.reference.pc_eV == pytest.approx(1e9)


def test_unknown_species_falls_back_to_a_proton(tmp_path):
    lat, rep = parse(tmp_path, _HEAD + """
    - main:
        kind: BeamLine
        line:
          - begin: {kind: BeginningEle, ReferenceP: {species_ref: "Au+79", E_tot_ref: 1.0e+11}}
          - d
""")
    assert lat.reference.species.name == "proton"
    assert "PALS_SPECIES_UNKNOWN" in rep.codes()


def test_reference_is_assumed_when_the_document_has_none(tmp_path):
    lat, rep = parse(tmp_path, _HEAD + "    - main: {kind: BeamLine, line: [d]}\n")
    assert "PALS_REFERENCE_ASSUMED" in rep.codes()
    assert rep.ok
    lat, rep = parse(tmp_path, _HEAD + "    - main: {kind: BeamLine, line: [d]}\n",
                     name="o.pals.yaml", species="h-", kinetic_energy_eV=8e8)
    assert lat.reference.species.name == "h-"
    assert lat.reference.kinetic_energy_eV == 8e8
    assert "PALS_REFERENCE_ASSUMED" not in rep.codes()


# ── parameter groups ──────────────────────────────────────────────────────────────────────
def _one(tmp_path, body: str, name="x.pals.yaml", **kw):
    lat, rep = parse(tmp_path, "PALS:\n  facility:\n" + body + """
    - main: {kind: BeamLine, line: [e]}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""", name=name, **kw)
    return lat.elements["e"], lat, rep


def test_normalized_strengths_use_the_local_rigidity(tmp_path):
    el, lat, rep = _one(tmp_path, """
    - e:
        kind: Quadrupole
        length: 0.5
        MagneticMultipoleP: {Kn1: 0.6, Ks2L: 0.1, tilt1: 0.3}
""", species="proton", kinetic_energy_eV=8e8)
    brho = lat.reference.brho_signed
    assert el.multipole.Bn[1] == pytest.approx(0.6 * brho)
    assert el.multipole.Bs[2] == pytest.approx(0.1 * brho / 0.5)     # integrated -> per length
    assert el.multipole.tilt[1] == pytest.approx(0.3)
    assert "PALS_INTEGRATED_TO_FIELD" in rep.codes()


def test_thin_multipole_keeps_integrated_strengths(tmp_path):
    el, _lat, rep = _one(tmp_path, """
    - e:
        kind: Multipole
        MagneticMultipoleP: {Bn2L: 0.7, Bs3L: -0.2}
""")
    assert isinstance(el, Multipole)
    assert el.multipole.BnL[2] == pytest.approx(0.7)
    assert el.multipole.BsL[3] == pytest.approx(-0.2)
    assert rep.ok


def test_thick_multipole_is_integrated_over_its_length(tmp_path):
    el, _lat, rep = _one(tmp_path, """
    - e:
        kind: Multipole
        length: 0.25
        MagneticMultipoleP: {Bn2: 8.0}
""")
    assert el.multipole.BnL[2] == pytest.approx(2.0)
    assert "PALS_MULTIPOLE_INTEGRATED" in rep.codes()


def test_solenoid_field_and_normalized_forms(tmp_path):
    el, _l, _r = _one(tmp_path, "    - e: {kind: Solenoid, length: 1.0, SolenoidP: {Bsol: 0.4}}\n")
    assert isinstance(el, Solenoid) and el.solenoid.Bsol_T == pytest.approx(0.4)
    el, lat, _r = _one(tmp_path, "    - e: {kind: Solenoid, length: 1.0, SolenoidP: {Ksol: 0.2}}\n",
                       name="k.pals.yaml", species="proton", kinetic_energy_eV=8e8)
    assert el.solenoid.Bsol_T == pytest.approx(0.2 * lat.reference.brho_signed)


def test_kicker_order_zero_multipole_signs(tmp_path):
    """``parameters/bend.md``: positive ``Kn0`` bends towards −x, so ``hkick = −Kn0L``."""
    el, lat, _r = _one(tmp_path, """
    - e:
        kind: Kicker
        MagneticMultipoleP: {Kn0L: 0.001, Bs0L: 0.02}
""", species="proton", kinetic_energy_eV=8e8)
    assert isinstance(el, Kicker)
    assert el.hkick == pytest.approx(-0.001)
    assert el.vkick == pytest.approx(0.02 / lat.reference.brho_signed)


def test_thick_kicker_uses_the_length_for_per_length_fields(tmp_path):
    el, lat, _r = _one(tmp_path, """
    - e:
        kind: Kicker
        length: 0.5
        MagneticMultipoleP: {Bn0: 0.1}
""", species="proton", kinetic_energy_eV=8e8)
    assert el.hkick == pytest.approx(-0.1 * 0.5 / lat.reference.brho_signed)


def test_aperture_min_max_and_center_width(tmp_path):
    el, _l, _r = _one(tmp_path, """
    - e:
        kind: Drift
        length: 1.0
        ApertureP:
          x_min: -0.02
          x_max: 0.02
          y_center: 0.001
          y_width: 0.01
          shape: RECTANGULAR
          location: BOTH_ENDS
""")
    assert el.aperture.shape == "RECTANGULAR"
    assert el.aperture.x_limits == (-0.02, 0.02)
    assert el.aperture.y_limits == pytest.approx((-0.004, 0.006))
    assert el.aperture.aperture_at == "BOTH_ENDS"


def test_aperture_unsupported_shape_and_inactive(tmp_path):
    el, _l, rep = _one(tmp_path, """
    - e:
        kind: Drift
        length: 1.0
        ApertureP: {shape: VERTICES, vertices: {list: [{point: [0.01, 0.0]}]}}
""")
    assert el.aperture is None
    assert "PALS_APERTURE_SHAPE_UNSUPPORTED" in rep.codes()
    el, _l, rep = _one(tmp_path, """
    - e:
        kind: Drift
        length: 1.0
        ApertureP: {x_min: -0.01, x_max: 0.01, aperture_active: false}
""", name="ia.pals.yaml")
    assert el.aperture is None
    assert "PALS_APERTURE_INACTIVE" in rep.codes()


def test_body_shift_z_rot_is_the_ir_tilt(tmp_path):
    el, _l, _r = _one(tmp_path, """
    - e:
        kind: Quadrupole
        length: 0.3
        MagneticMultipoleP: {Bn1: 1.0}
        BodyShiftP: {x_offset: 0.001, z_rot: 0.7854, y_rot: -0.002}
""")
    assert el.shift.x_offset == pytest.approx(0.001)
    assert el.shift.tilt == pytest.approx(0.7854)
    assert el.shift.y_rot == pytest.approx(-0.002)


def test_rf_zero_phase_transition_conventions(tmp_path):
    for zero, shift in (("BELOW_TRANSITION", -math.pi / 2), ("ABOVE_TRANSITION", math.pi / 2)):
        el, _l, rep = _one(tmp_path, f"""
    - e:
        kind: RFCavity
        length: 1.0
        RFP: {{voltage: 1.0e+6, phase: 0.05, zero_phase: {zero}}}
""", name=f"{zero}.pals.yaml")
        assert el.rf.phase_rad == pytest.approx(turns_to_rad(0.05) + shift)
        assert "PALS_ZERO_PHASE_TRANSITION" in rep.codes()
        assert rep.ok


def test_rf_extras(tmp_path):
    el, _l, rep = _one(tmp_path, """
    - e:
        kind: RFCavity
        length: 2.0
        RFP:
          frequency: 650.0e+6
          voltage: 1.0e+7
          phase: -0.0833333333
          L_active: 1.6
          num_cells: 5
          dE_ref: 8.0e+6
          cavity_type: TRAVELING_WAVE
""")
    assert el.rf.frequency_Hz == pytest.approx(650e6)
    assert el.rf.L_active_m == pytest.approx(1.6)
    assert el.rf.n_cell == 5
    assert el.rf.dE_ref_eV == pytest.approx(8e6)
    assert el.rf.cavity_type == "TRAVELING_WAVE"
    assert el.rf.gradient_V_per_m == pytest.approx(1e7 / 1.6)
    assert rep.ok


def test_rf_harmon_without_frequency_is_lossy(tmp_path):
    _el, _l, rep = _one(tmp_path, """
    - e: {kind: RFCavity, length: 1.0, RFP: {harmon: 84, voltage: 1.0e+6}}
""")
    assert "PALS_HARMON_UNRESOLVED" in rep.codes()


def test_bend_rectangular_edges(tmp_path):
    """``e1 = e1_rect + angle_ref/2`` for ``ref_geometry: ARC``."""
    el, _l, _r = _one(tmp_path, """
    - e:
        kind: Bend
        length: 1.0
        BendP: {angle_ref: 0.2, e1_rect: 0.0, e2_rect: 0.01}
""")
    assert el.bend.rect is True
    assert el.bend.e1 == pytest.approx(0.1)
    assert el.bend.e2 == pytest.approx(0.11)


def test_bend_from_reference_field_needs_the_rigidity(tmp_path):
    el, lat, _r = _one(tmp_path, """
    - e: {kind: Bend, length: 2.0, BendP: {Bn0_ref: 0.5}}
""", species="proton", kinetic_energy_eV=8e8)
    assert el.bend.angle == pytest.approx(0.5 / lat.reference.brho_signed * 2.0)


def test_bend_from_chord_and_rectangle(tmp_path):
    el, _l, _r = _one(tmp_path, "    - e: {kind: Bend, BendP: {angle_ref: 0.4, L_chord: 1.0}}\n")
    assert el.length == pytest.approx(1.0 * 0.2 / math.sin(0.2))
    el, _l, _r = _one(tmp_path, "    - e: {kind: Bend, BendP: {angle_ref: 0.4, L_rectangle: 1.0}}\n",
                      name="lr.pals.yaml")
    assert el.length == pytest.approx(0.4 / math.sin(0.4))


def test_bend_pole_face_curvature_and_geometry_are_lossy(tmp_path):
    _el, _l, rep = _one(tmp_path, """
    - e:
        kind: Bend
        length: 1.0
        BendP: {angle_ref: 0.1, h1: 0.5, ref_geometry: CHORD}
""")
    assert "PALS_POLE_FACE_CURVATURE_DROPPED" in rep.codes()
    assert "PALS_BEND_GEOMETRY_IGNORED" in rep.codes()


def test_mask_becomes_a_collimator_and_unionele_a_superposition(tmp_path):
    el, _l, _r = _one(tmp_path, """
    - e: {kind: Mask, ApertureP: {x_min: -0.01, x_max: 0.01}}
""")
    assert isinstance(el, Collimator)
    assert el.aperture.x_limits == (-0.01, 0.01)

    el, lat, rep = _one(tmp_path, """
    - e:
        kind: UnionEle
        length: 2.0
        elements:
          sa:
            kind: Solenoid
            length: 1.0
            SolenoidP: {Bsol: 0.3}
          ra:
            kind: RFCavity
            length: 0.4
            BodyShiftP: {z_offset: 0.5}
            RFP: {voltage: 1.0e+6}
""", name="u.pals.yaml")
    assert isinstance(el, Superposition)
    assert dict((n, round(o, 12)) for o, n in el.children) == {"sa": 0.5, "ra": 1.3}
    assert "PALS_UNIONELE_AS_SUPERPOSITION" in rep.codes()


def test_reference_change_and_patch(tmp_path):
    el, lat, _r = _one(tmp_path, """
    - e: {kind: ReferenceChange, ReferenceChangeP: {dE_ref: 1.0e+6, dtime_ref: 2.0e-9}}
""")
    assert isinstance(el, ReferenceChange)
    assert el.dE_ref_eV == pytest.approx(1e6)
    assert el.dtime_s == pytest.approx(2e-9)

    el, lat, _r = _one(tmp_path, """
    - e: {kind: ReferenceChange, ReferenceChangeP: {E_tot_ref: 1.0e+9}}
""", name="rc.pals.yaml", species="proton")
    assert el.energy_eV == pytest.approx(1e9 - lat.reference.species.mass_eV)

    el, _l, _r = _one(tmp_path, """
    - e: {kind: Patch, PatchP: {x_offset: 0.01, z_rot: 0.2, y_rot: 0.05}}
""", name="p.pals.yaml")
    assert isinstance(el, Patch)
    assert (el.x_offset, el.tilt, el.y_rot) == pytest.approx((0.01, 0.2, 0.05))


def test_floorshift_becomes_a_patch_and_instrument_keeps_its_family(tmp_path):
    el, _l, rep = _one(tmp_path, """
    - e: {kind: FloorShift, CoordinateSetP: {z_offset: 0.3}}
""")
    assert isinstance(el, Patch) and el.z_offset == pytest.approx(0.3)
    assert "PALS_FLOORSHIFT_AS_PATCH" in rep.codes()

    el, _l, _r = _one(tmp_path, "    - e: {kind: Instrument, MetaP: {label: BPM}}\n",
                      name="i.pals.yaml")
    assert isinstance(el, Instrument) and el.family == "BPM"


def test_taylor_first_order_terms(tmp_path):
    el, _l, rep = _one(tmp_path, """
    - e:
        kind: Taylor
        TaylorP:
          x_out: ["term 1.0 1 0 0 0 0 0", "term 2.5 0 1 0 0 0 0", "term 0.4 2 0 0 0 0 0"]
          px_out: ["term 0.1 0 0 0 0 0 0"]
""")
    assert el.matrix[0][0] == pytest.approx(1.0)
    assert el.matrix[0][1] == pytest.approx(2.5)
    assert el.offset[1] == pytest.approx(0.1)
    assert "PALS_TAYLOR_ORDER_TRUNCATED" in rep.codes()


# ── unsupported kinds ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("kind", "length", "cls"),
                         [("Wiggler", 2.0, Drift), ("CrabCavity", 0.0, Marker),
                          ("Fork", 0.0, Marker), ("Girder", 0.0, Marker),
                          ("Match", 0.0, Marker), ("Placeholder", 0.0, Marker)])
def test_unsupported_kind_keeps_the_length_and_is_dropped(tmp_path, kind, length, cls):
    el, _l, rep = _one(tmp_path, f"    - e: {{kind: {kind}, length: {length}}}\n",
                       name=f"{kind}.pals.yaml")
    assert isinstance(el, cls)
    assert el.length == pytest.approx(length)
    assert "UNSUPPORTED_PALS_KIND" in rep.codes()
    with pytest.raises(TranslationError):
        _one(tmp_path, f"    - e: {{kind: {kind}, length: {length}}}\n",
             name=f"{kind}_s.pals.yaml", strict=True)


def test_set_and_superimpose_commands_are_reported(tmp_path):
    _el, _l, rep = _one(tmp_path, """
    - e: {kind: Drift, length: 1.0}
    - set: {parameter: "e>MagneticMultipoleP.Kn1", value: 0.5}
    - superimpose: {place: e, placement: {base_item: e}}
""")
    assert rep.codes()["PALS_COMMAND_IGNORED"] == 2


def test_placement_and_undefined_items(tmp_path):
    lat, rep = parse(tmp_path, _HEAD + """
    - main:
        kind: BeamLine
        line:
          - d: {placement: {offset: 1.0}}
          - ghost
""")
    assert "PALS_PLACEMENT_IGNORED" in rep.codes()
    assert "PALS_UNDEFINED_ITEM" in rep.codes()
    assert len(lat.flatten()) == 1


# ── pals-schema 0.3.0 spellings ───────────────────────────────────────────────────────────
def test_pals_schema_field_name_aliases_are_accepted(tmp_path):
    """``pals-schema`` 0.3.0 is a snapshot of an earlier draft: ``SBend``/``RBend`` instead of
    ``Bend``, ``rho_ref``/``edge_int1``/``n_cell``/``x_limits``/``extra_dtime_ref``."""
    lat, rep = parse(tmp_path, """
PALS:
  facility:
    - b:
        kind: SBend
        length: 1.0
        BendP: {rho_ref: 5.0, edge_int1: 0.02, edge_int2: 0.03}
        ApertureP: {x_limits: [-0.03, 0.03], y_limits: [-0.02, 0.02]}
    - r:
        kind: RBend
        length: 1.0
        BendP: {rho_ref: 5.0, e1_rect: 0.0, e2_rect: 0.0}
    - c:
        kind: RFCavity
        length: 1.0
        RFP: {voltage: 1.0e+6, n_cell: 9}
    - rc:
        kind: ReferenceChange
        ReferenceChangeP: {dE_ref: 1.0e+6, extra_dtime_ref: 3.0e-9}
    - main: {kind: BeamLine, line: [b, r, c, rc]}
    - lat: {kind: Lattice, branches: [main]}
    - use: lat
""")
    b = lat.elements["b"]
    assert isinstance(b, Bend) and b.bend.angle == pytest.approx(0.2)
    assert b.bend.edge_int1 * b.bend.hgap == pytest.approx(0.02)
    assert b.bend.edge_int2 * b.bend.hgap == pytest.approx(0.03)
    assert b.aperture.x_limits == (-0.03, 0.03)
    assert lat.elements["r"].bend.rect is True
    assert lat.elements["c"].rf.n_cell == 9
    assert lat.elements["rc"].dtime_s == pytest.approx(3e-9)
    assert rep.ok


# ── ledger shape ──────────────────────────────────────────────────────────────────────────
def test_document_without_a_beamline_is_salvaged(tmp_path):
    """A definitions-only fragment (a `load`-able settings file) still yields a usable
    lattice, with the recovery recorded as DROPPED so strict mode refuses it."""
    lat, rep = parse(tmp_path, """
PALS:
  facility:
    - d: {kind: Drift, length: 1.0}
    - q: {kind: Quadrupole, length: 0.3, MagneticMultipoleP: {Bn1: 2.0}}
""")
    assert [p.element.name for p in lat.flatten()] == ["d", "q"]
    assert "PALS_NO_LINE" in rep.codes()
    with pytest.raises(TranslationError):
        parse(tmp_path, "PALS:\n  facility:\n    - d: {kind: Drift, length: 1.0}\n",
              name="nl.pals.yaml", strict=True)


def test_json_documents_read_the_same_as_yaml(tmp_path):
    import json

    import yaml
    src = (DATA / "fodo.pals.yaml").read_text()
    (tmp_path / "f.pals.json").write_text(json.dumps(yaml.safe_load(src)))
    a, _ = read(DATA / "fodo.pals.yaml")
    b, _ = read(tmp_path / "f.pals.json")
    assert [(p.element.kind, p.length) for p in a.flatten()] == \
           [(p.element.kind, p.length) for p in b.flatten()]


def test_every_element_definition_gets_one_ledger_entry():
    _lat, rep = read(DATA / "drift_quad_bend.pals.yaml")
    elements = [e for e in rep.entries if e.kind in
                {"Drift", "Quadrupole", "Bend", "Marker"}]
    assert len(elements) == 5           # d1, q1, b1, d2, end (one per definition)
    assert all(e.cls == "EXACT" for e in elements)


def test_provenance_records_the_pals_kind():
    lat, _ = read(DATA / "drift_quad_bend.pals.yaml")
    b = lat.elements["b1"]
    assert b.provenance.format == "pals"
    assert b.provenance.original_type == "Bend"
    assert b.provenance.original_name == "b1"
