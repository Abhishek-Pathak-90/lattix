"""IR ⇄ ``xt.Line`` conversion rules (PLAN §6 task 3.5): one test per RULES row.

No tracking happens here (no ``build_tracker``), so this module runs under any xtrack
build; the engine-backed sign and map checks live in ``test_xtrack_oracle.py``.
"""
from __future__ import annotations

import math

import pytest

xt = pytest.importorskip("xtrack")

from lattix.fidelity import TranslationError  # noqa: E402
from lattix.formats.base import check_rules_coverage  # noqa: E402
from lattix.formats.xtrack import (  # noqa: E402
    METADATA_KEY,
    RULES,
    NameMap,
    Writer,
    from_line,
    phase_from_xtrack,
    sanitize,
    to_line,
    xtrack_lag_deg,
    xtrack_phase_rad,
)
from lattix.formats.xtrack.convert import _new  # noqa: E402
from lattix.ir.elements import (  # noqa: E402
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
from lattix.ir.lattice import Lattice  # noqa: E402
from lattix.ir.reference import ReferenceParticle, species  # noqa: E402
from lattix.ir.walk import propagate  # noqa: E402


def proton_ref(ke: float = 8e8, freq: float | None = None) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke,
                             rf_frequency_Hz=freq)


def one(el: Element, ref: ReferenceParticle | None = None, **kw):
    """``to_line`` of a single-element lattice → (xtrack element, line, report)."""
    from lattix.fidelity import FidelityReport

    ref = ref or proton_ref()
    lat = Lattice.from_sequence("t", [el], ref)
    rep = FidelityReport()
    line = to_line(lat, report=rep, **kw)
    return line.element_dict[line.element_names[0]], line, rep


def all_kinds_lattice() -> Lattice:
    """One element of every one of the 22 IR kinds (mirrors the Bmad/MAD-X zoos)."""
    ref = proton_ref(1e8, freq=325e6)
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
        Instrument(name="bp", length=0.05, family="BPM"),
        Foil(name="fl", thickness_kg_per_m2=0.237, material="C"),
        Taylor(name="ty", length=0.2),
        Patch(name="pt", x_offset=1e-3, y_rot=2e-3, tilt=0.3),
        ReferenceChange(name="rc", energy_eV=1.2e8),
        Freq(name="fq", frequency_Hz=650e6),
        Directive(name="dv", format="tracewin", card="ADJUST", role="matching", args=["1", "2"]),
        Superposition(name="sp", length=0.2, children=[(0.0, "sup_child")]),
    ]
    lat = Lattice.from_sequence("allkinds", els, ref)
    lat.elements["sup_child"] = child
    ty = lat.elements["ty"]
    ty.matrix[0][1] = 0.5
    ty.offset[4] = 1e-3
    return lat


# ------------------------------------------------------------------ structure
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()
    assert set(RULES) == set(check_rules_coverage(Writer()) | set(RULES))


def test_every_kind_produces_at_least_one_xtrack_element():
    lat = all_kinds_lattice()
    line = to_line(lat)
    rows = line.metadata[METADATA_KEY]["elements"]
    covered = {row["kind"] for row in rows.values() if "kind" in row}
    # Superposition is expanded into its children, which carry the child's kind
    assert set(RULES) - covered == {"Superposition"}
    assert len(line.element_names) >= len(lat.flatten())


def test_names_are_unique_and_sanitised():
    ref = proton_ref()
    lat = Lattice(name="l", reference=ref)
    from lattix.ir.lattice import Line, LineItem

    lat.elements["q 1"] = Quadrupole(name="q 1", length=0.1)
    lat.lines["l"] = Line(name="l", items=[LineItem(ref="q 1"), LineItem(ref="q 1")])
    lat.use = "l"
    line = to_line(lat)
    assert line.element_names == ["q_1", "q_1_2"]
    assert len(set(line.element_names)) == len(line.element_names)


def test_sanitize_and_namemap():
    assert sanitize("a-b c") == "a_b_c"
    assert sanitize("9lead") == "lead"
    assert sanitize("!!!") == "___"
    assert sanitize("") == "el"
    nm = NameMap()
    assert [nm.assign("x"), nm.assign("x"), nm.assign("x")] == ["x", "x_2", "x_3"]


def test_metadata_carries_lattix_provenance():
    lat = Lattice.from_sequence("demo", [Drift(name="d", length=1.0)], proton_ref())
    lat.meta["source_format"] = "madx"
    line = to_line(lat)
    meta = line.metadata[METADATA_KEY]
    assert meta["source_format"] == "madx"
    assert meta["lattice"] == "demo"
    assert meta["energy_mode"] == "delta"
    assert meta["species"] == "proton"
    assert meta["elements"]["d"] == {"role": "main", "name": "d", "kind": "Drift", "group": 0}


def test_particle_ref_comes_from_the_lattice_reference():
    import numpy as np

    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=2.1e6)
    _, line, _ = one(Drift(name="d", length=1.0), ref)
    pref = line.particle_ref
    assert float(np.atleast_1d(pref.mass0)[0]) == pytest.approx(ref.species.mass_eV)
    assert int(round(float(np.atleast_1d(pref.q0)[0]))) == -1
    assert float(np.atleast_1d(pref.kinetic_energy0)[0]) == pytest.approx(2.1e6)


# --------------------------------------------------------------- per-rule
def test_drift():
    el, _, rep = one(Drift(name="d", length=1.25))
    assert isinstance(el, xt.Drift) and float(el.length) == 1.25
    assert rep.ok


def test_quadrupole_k1_uses_the_signed_rigidity():
    ref = proton_ref()
    G = 0.6 * ref.brho_signed
    el, _, _ = one(Quadrupole(name="q", length=0.3,
                              multipole=MagneticMultipoleP(Bn={1: G})), ref)
    assert isinstance(el, xt.Quadrupole)
    assert float(el.k1) == pytest.approx(0.6, rel=1e-14)


def test_quadrupole_hminus_flips_k1():
    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=8e8)
    G = 0.6 * abs(ref.brho_abs)
    el, _, _ = one(Quadrupole(name="q", length=0.3,
                              multipole=MagneticMultipoleP(Bn={1: G})), ref)
    assert float(el.k1) == pytest.approx(-0.6, rel=1e-14)


def test_quadrupole_tilt_becomes_rot_s_rad_and_skew_becomes_k1s():
    ref = proton_ref()
    b = ref.brho_signed
    el, _, _ = one(Quadrupole(name="q", length=0.3,
                              multipole=MagneticMultipoleP(Bn={1: 0.6 * b}, Bs={1: 0.1 * b},
                                                           tilt={1: 0.25})), ref)
    assert float(el.rot_s_rad) == pytest.approx(0.25)
    assert float(el.k1s) == pytest.approx(0.1)
    assert float(el.k1) == pytest.approx(0.6)


@pytest.mark.parametrize("cls,order,attr,skew", [(Sextupole, 2, "k2", "k2s"),
                                                 (Octupole, 3, "k3", "k3s")])
def test_thick_multipoles(cls, order, attr, skew):
    ref = proton_ref()
    b = ref.brho_signed
    el, _, _ = one(cls(name="m", length=0.2,
                       multipole=MagneticMultipoleP(Bn={order: 1.5 * b}, Bs={order: 0.5 * b},
                                                    tilt={order: 0.1})), ref)
    assert float(getattr(el, attr)) == pytest.approx(1.5)
    assert float(getattr(el, skew)) == pytest.approx(0.5)
    assert float(el.rot_s_rad) == pytest.approx(0.1)


def test_thin_multipole_knl_ksl():
    ref = proton_ref()
    b = ref.brho_signed
    el, _, _ = one(Multipole(name="m", multipole=MagneticMultipoleP(BnL={0: 0.01 * b,
                                                                        2: 0.3 * b},
                                                                    BsL={1: 0.03 * b})), ref)
    assert isinstance(el, xt.Multipole)
    assert [float(v) for v in el.knl][:3] == pytest.approx([0.01, 0.0, 0.3])
    assert [float(v) for v in el.ksl][:3] == pytest.approx([0.0, 0.03, 0.0])
    assert not bool(el.isthick)


def test_bend_geometry_and_edges():
    ref = proton_ref()
    b = ref.brho_signed
    el, _, _ = one(Bend(name="bd", length=1.0,
                        multipole=MagneticMultipoleP(Bn={1: 0.2 * b}),
                        bend=BendP(angle=0.1, e1=0.02, e2=0.03, edge_int1=0.45,
                                   edge_int2=0.5, hgap=0.03, tilt_ref=0.1)), ref)
    assert isinstance(el, xt.Bend)
    assert float(el.length) == pytest.approx(1.0)
    assert float(el.h) == pytest.approx(0.1)
    assert float(el.angle) == pytest.approx(0.1)
    assert bool(el.k0_from_h)                       # angle drives the dipole field
    assert float(el.edge_entry_angle) == pytest.approx(0.02)
    assert float(el.edge_exit_angle) == pytest.approx(0.03)
    assert float(el.edge_entry_fint) == pytest.approx(0.45)
    assert float(el.edge_exit_fint) == pytest.approx(0.5)
    assert float(el.edge_entry_hgap) == float(el.edge_exit_hgap) == pytest.approx(0.03)
    assert float(el.k1) == pytest.approx(0.2)
    assert float(el.rot_s_rad) == pytest.approx(0.1)


def test_rect_bend_is_a_sector_bend_with_total_edge_angles():
    """The IR keeps the arc length and e1/e2 already containing angle/2, so an rbend is
    an ``xt.Bend`` — no RBend chord/sagitta model is involved."""
    el, _, _ = one(Bend(name="rb", length=1.0,
                        bend=BendP(angle=0.1, e1=0.05, e2=0.05, rect=True)))
    assert isinstance(el, xt.Bend)
    assert float(el.length) == pytest.approx(1.0)
    assert float(el.edge_entry_angle) == pytest.approx(0.05)


def test_thin_bend_becomes_a_multipole_with_hxl():
    el, _, rep = one(Bend(name="b0", length=0.0, bend=BendP(angle=0.01)))
    assert isinstance(el, xt.Multipole)
    assert float(el.hxl) == pytest.approx(0.01)
    assert "THIN_BEND_AS_MULTIPOLE" in rep.codes()


def test_solenoid_ks():
    ref = proton_ref()
    el, _, _ = one(Solenoid(name="s", length=0.4,
                            solenoid=SolenoidP(Bsol_T=0.3 * ref.brho_signed)), ref)
    assert isinstance(el, xt.UniformSolenoid)
    assert float(el.ks) == pytest.approx(0.3)


def test_thin_solenoid_is_lossy():
    _, _, rep = one(Solenoid(name="s", length=0.0, solenoid=SolenoidP(Bsol_T=0.3)))
    assert "THIN_SOLENOID_DROPPED" in rep.codes()


# --------------------------------------------------------------------- RF
def test_cavity_phase_convention():
    """IR φ → ``phase = φ + π/2`` ≡ ``lag = φ·180/π + 90`` (both add in xtrack)."""
    phi = -math.pi / 6
    el, _, rep = one(RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=phi,
                                               frequency_Hz=325e6)))
    assert isinstance(el, xt.Cavity)
    assert float(el.voltage) == pytest.approx(1e6)
    assert float(el.frequency) == pytest.approx(325e6)
    assert float(el.phase) == pytest.approx(phi + math.pi / 2)
    assert xtrack_lag_deg(phi) == pytest.approx(60.0)
    assert xtrack_phase_rad(phi) == pytest.approx(math.radians(60.0))
    assert phase_from_xtrack(0.0, float(el.phase)) == pytest.approx(phi)
    assert phase_from_xtrack(xtrack_lag_deg(phi), 0.0) == pytest.approx(phi)
    assert "CONST_P0" in rep.codes()


def test_cavity_gradient_is_folded_into_a_voltage():
    el, _, _ = one(RFCavity(name="c", length=0.5,
                            rf=RFP(gradient_V_per_m=2e6, L_active_m=0.4, phase_rad=0.0,
                                   frequency_Hz=325e6)))
    assert float(el.voltage) == pytest.approx(8e5)
    assert float(el.length) == pytest.approx(0.5)


def test_cavity_without_frequency_uses_the_reference_clock():
    ref = proton_ref(2.1e6, freq=162.5e6)
    el, _, rep = one(RFCavity(name="c", rf=RFP(voltage_V=1e5, phase_rad=0.0)), ref)
    assert float(el.frequency) == pytest.approx(162.5e6)
    assert "RF_FREQUENCY_FROM_CLOCK" in rep.codes()


def test_cavity_with_no_frequency_at_all_is_lossy():
    _, _, rep = one(RFCavity(name="c", rf=RFP(voltage_V=1e5, phase_rad=0.0)))
    assert "RF_FREQUENCY_MISSING" in rep.codes()


def test_fieldmap_with_a_voltage_becomes_a_cavity():
    el, _, rep = one(FieldMap(name="fm", length=0.6,
                              rf=RFP(voltage_V=1.5e6, phase_rad=-math.pi / 6,
                                     frequency_Hz=325e6)))
    assert isinstance(el, xt.Cavity)
    assert float(el.length) == pytest.approx(0.6)
    assert "FM_AS_CAVITY" in rep.codes()


def test_fieldmap_from_de_ref_only():
    el, _, _ = one(FieldMap(name="fm", length=0.6,
                            rf=RFP(phase_rad=-math.pi / 3, frequency_Hz=325e6,
                                   dE_ref_eV=1e6)))
    assert isinstance(el, xt.Cavity)
    assert float(el.voltage) * math.cos(-math.pi / 3) == pytest.approx(1e6)


def test_fieldmap_without_a_voltage_becomes_a_drift():
    el, _, rep = one(FieldMap(name="fm", length=0.6, rf=RFP(frequency_Hz=325e6)))
    assert isinstance(el, xt.Drift) and float(el.length) == pytest.approx(0.6)
    assert "FM_TO_DRIFT" in rep.codes()


@pytest.mark.parametrize("cls,code", [(NCells, "NCELLS_TO_DRIFT"), (RFQCell, "RFQ_TO_DRIFT")])
def test_cell_trains_become_drifts(cls, code):
    el, _, rep = one(cls(name="x", length=0.7))
    assert isinstance(el, xt.Drift) and float(el.length) == pytest.approx(0.7)
    assert code in rep.codes()


# ------------------------------------------------------------------ kicker
def test_kicker_sign_convention():
    """Measured against xtrack: ``knl[0] = -hkick``, ``ksl[0] = +vkick``."""
    el, _, rep = one(Kicker(name="k", hkick=1e-3, vkick=-2e-3))
    assert isinstance(el, xt.Multipole)
    assert float(el.knl[0]) == pytest.approx(-1e-3)
    assert float(el.ksl[0]) == pytest.approx(-2e-3)
    assert rep.ok


def test_thick_kicker_advances_s():
    el, line, _ = one(Kicker(name="k", length=0.5, hkick=1e-3))
    assert bool(el.isthick)
    assert float(line.get_length()) == pytest.approx(0.5)


def test_electric_kicker_is_lossy():
    _, _, rep = one(Kicker(name="k", hkick=1e-3, electric=True))
    assert "EKICK_AS_MAGNETIC" in rep.codes()


# --------------------------------------------------------------- apertures
def test_collimator_rect_and_ellipse():
    el, line, _ = one(Collimator(name="c", length=0.1,
                                 aperture=ApertureP.rect(0.02, 0.03)))
    assert isinstance(el, xt.LimitRect)
    assert (float(el.min_x), float(el.max_x)) == (-0.02, 0.02)
    assert len(line.element_names) == 2                 # limit + the collimator's drift
    el2, _, _ = one(Collimator(name="c", aperture=ApertureP.circle(0.01)))
    assert isinstance(el2, xt.LimitEllipse)
    assert float(el2.a) == pytest.approx(0.01)


def test_collimator_without_an_aperture_is_lossy():
    el, _, rep = one(Collimator(name="c", length=0.1))
    assert isinstance(el, xt.Drift)
    assert "COLLIMATOR_WITHOUT_APERTURE" in rep.codes()


def test_element_aperture_becomes_limit_elements_at_both_ends():
    q = Quadrupole(name="q", length=0.3, aperture=ApertureP.circle(0.02))
    _, line, _ = one(q)
    assert line.element_names == ["q_aper_in", "q", "q_aper_out"]
    assert isinstance(line.element_dict["q_aper_in"], xt.LimitEllipse)


def test_aperture_install_can_be_switched_off():
    q = Quadrupole(name="q", length=0.3, aperture=ApertureP.circle(0.02))
    _, line, _ = one(q, install_apertures=False)
    assert line.element_names == ["q"]


def test_entrance_only_aperture():
    q = Quadrupole(name="q", length=0.3,
                   aperture=ApertureP(shape="ELLIPTICAL", x_limits=(-0.02, 0.02),
                                      y_limits=(-0.02, 0.02), aperture_at="ENTRANCE"))
    _, line, _ = one(q)
    assert line.element_names == ["q_aper_in", "q"]


# ------------------------------------------------------- misc element kinds
def test_marker_and_freq_and_directive():
    el, _, rep = one(Marker(name="m"))
    assert isinstance(el, xt.Marker) and rep.ok
    el, _, rep = one(Freq(name="f", frequency_Hz=650e6))
    assert isinstance(el, xt.Marker) and rep.ok
    el, _, rep = one(Directive(name="d", card="ADJUST", role="matching"))
    assert isinstance(el, xt.Marker)
    assert "FOREIGN_DIRECTIVE" in rep.codes()
    el, _, rep = one(Directive(name="d", card="LATTICE", role="period_start"))
    assert rep.ok                                    # a comment role loses nothing


def test_instrument_and_foil():
    el, _, rep = one(Instrument(name="bp", length=0.05, family="BPM"))
    assert isinstance(el, xt.Drift) and float(el.length) == pytest.approx(0.05)
    assert "MONITOR_AS_DRIFT" in rep.codes()
    el, _, _ = one(Instrument(name="bp", family="BPM"))
    assert isinstance(el, xt.Marker)
    el, _, rep = one(Foil(name="fl"))
    assert isinstance(el, xt.Marker)
    assert "FOIL_TO_MARKER" in rep.codes()


def test_taylor_map():
    t = Taylor(name="ty", length=0.2)
    t.matrix[0][1] = 0.5
    t.offset[4] = 1e-3
    el, _, rep = one(t)
    assert isinstance(el, xt.FirstOrderTaylorMap)
    import numpy as np

    assert np.asarray(el.m1).reshape(6, 6)[0, 1] == pytest.approx(0.5)
    assert "TAYLOR_BASIS_XTRACK" in rep.codes()


def test_patch_becomes_frame_elements():
    _, line, rep = one(Patch(name="pt", x_offset=1e-3, y_offset=-2e-3, y_rot=2e-3, tilt=0.3,
                             z_offset=0.05))
    kinds = [type(line.element_dict[n]).__name__ for n in line.element_names]
    assert kinds == ["XYShift", "YRotation", "SRotation", "Drift"]
    assert float(line.element_dict[line.element_names[0]].dx) == pytest.approx(1e-3)
    assert float(line.element_dict["pt_srot"].angle) == pytest.approx(math.degrees(0.3))
    assert rep.ok


def test_empty_patch_is_a_marker():
    el, _, _ = one(Patch(name="pt"))
    assert isinstance(el, xt.Marker)


def test_reference_change_becomes_a_p0c_increase():
    ref = proton_ref(1e8)
    el, _, rep = one(ReferenceChange(name="rc", energy_eV=1.2e8), ref)
    assert isinstance(el, xt.ReferenceEnergyIncrease)
    expect = ref.advanced(dE_eV=2e7).pc_eV - ref.pc_eV
    assert float(el.Delta_p0c) == pytest.approx(expect, rel=1e-12)
    assert "REFCHANGE_AS_P0C" in rep.codes()


def test_reference_time_change_becomes_a_zeta_shift():
    ref = proton_ref(1e8)
    el, _, _ = one(ReferenceChange(name="rc", dtime_s=1e-9), ref)
    assert isinstance(el, xt.ZetaShift)
    from lattix.ir.units import C_LIGHT

    assert float(el.dzeta) == pytest.approx(-ref.beta * C_LIGHT * 1e-9)


def test_superposition_is_flattened():
    ref = proton_ref()
    lat = Lattice.from_sequence("s", [Superposition(name="sp", length=0.2,
                                                    children=[(0.0, "child")])], ref)
    lat.elements["child"] = Drift(name="child", length=0.2)
    from lattix.fidelity import FidelityReport

    rep = FidelityReport()
    line = to_line(lat, report=rep)
    assert line.element_names == ["child"]
    assert "SUPERPOSITION_FLATTENED" in rep.codes()


def test_superposition_missing_child_is_dropped():
    ref = proton_ref()
    lat = Lattice.from_sequence("s", [Superposition(name="sp", children=[(0.0, "nope")])], ref)
    from lattix.fidelity import FidelityReport

    rep = FidelityReport()
    to_line(lat, report=rep)
    assert "SUPERPOSITION_CHILD_MISSING" in rep.codes()


# ----------------------------------------------------------- misalignments
def test_body_shift_uses_the_element_attributes():
    sh = BodyShiftP(x_offset=1e-3, y_offset=-2e-3, z_offset=3e-3, tilt=0.01,
                    x_rot=2e-4, y_rot=-3e-4)
    el, _, rep = one(Quadrupole(name="q", length=0.3, shift=sh))
    assert float(el.shift_x) == pytest.approx(1e-3)
    assert float(el.shift_y) == pytest.approx(-2e-3)
    assert float(el.shift_s) == pytest.approx(3e-3)
    assert float(el.rot_s_rad_no_frame) == pytest.approx(0.01)
    assert float(el.rot_x_rad) == pytest.approx(2e-4)
    assert float(el.rot_y_rad) == pytest.approx(-3e-4)
    assert rep.ok


def test_body_shift_on_a_drift_is_a_documented_no_op():
    el, _, rep = one(Drift(name="d", length=1.0,
                           shift=BodyShiftP(x_offset=1e-3, tilt=0.2)))
    assert isinstance(el, xt.Drift)
    assert "SHIFT_NO_OP" in rep.codes()


def test_bend_keeps_design_tilt_and_error_roll_apart():
    el, _, _ = one(Bend(name="b", length=1.0, bend=BendP(angle=0.1, tilt_ref=math.pi / 2),
                        shift=BodyShiftP(tilt=0.01)))
    assert float(el.rot_s_rad) == pytest.approx(math.pi / 2)
    assert float(el.rot_s_rad_no_frame) == pytest.approx(0.01)


# ------------------------------------------------------------- dual regime
def _linac() -> Lattice:
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=1e8,
                            rf_frequency_Hz=325e6)
    b = ref.brho_signed
    els = [
        Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * b})),
        RFCavity(name="c1", rf=RFP(voltage_V=5e7, phase_rad=0.0, frequency_Hz=325e6)),
        Quadrupole(name="q2", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * b})),
    ]
    return Lattice.from_sequence("linac", els, ref)


def test_energy_mode_local_vs_constant():
    lat = _linac()
    local = to_line(lat, energy_mode="local")
    const = to_line(lat, energy_mode="constant")
    assert float(local.element_dict["q1"].k1) == pytest.approx(float(const.element_dict["q1"].k1))
    # q2 sits behind a 50 MV cavity: local rigidity is larger, so |k1| is smaller
    assert abs(float(local.element_dict["q2"].k1)) < abs(float(const.element_dict["q2"].k1))
    placed = propagate(lat)
    expect = 5.0 * lat.reference.brho_signed / placed[2].ref_in.brho_signed
    assert float(local.element_dict["q2"].k1) == pytest.approx(expect, rel=1e-12)
    assert float(const.element_dict["q2"].k1) == pytest.approx(5.0, rel=1e-12)


def test_energy_mode_is_recorded():
    from lattix.fidelity import FidelityReport

    for mode, code in (("delta", "CONST_P0_DELTA_RIGIDITY"), ("local", "CONST_P0_LOCAL_RIGIDITY"),
                       ("constant", "CONST_P0_START_RIGIDITY")):
        rep = FidelityReport()
        to_line(_linac(), energy_mode=mode, report=rep)
        assert code in rep.codes()
        assert "CONST_P0" in rep.codes()


def test_bad_energy_mode_raises():
    with pytest.raises(ValueError, match="energy_mode"):
        to_line(_linac(), energy_mode="nope")


def test_strict_raises_on_a_lossy_element():
    lat = Lattice.from_sequence("s", [NCells(name="nc", length=0.5)], proton_ref())
    with pytest.raises(TranslationError):
        to_line(lat, strict=True)


# ------------------------------------------------------------------ from_line
def test_from_line_reads_the_particle_ref():
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    lat = from_line(line)
    assert lat.reference.species.name == "proton"
    assert lat.reference.kinetic_energy_eV == pytest.approx(8e8)


def test_from_line_without_a_particle_ref_warns():
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
    lat = from_line(line)
    assert any("particle_ref" in w for w in lat.warnings)
    assert lat.reference.kinetic_energy_eV == 1e9


def test_from_line_explicit_reference_wins():
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=2.1e6)
    lat = from_line(line, ref)
    assert lat.reference.species.name == "h-"


def _round_trip_kinds(elements, names, ref=None):
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    from lattix.fidelity import FidelityReport

    rep = FidelityReport()
    lat = from_line(line, ref, report=rep)
    return lat, rep


def test_from_line_maps_every_known_class():
    ref = proton_ref()
    b = ref.brho_signed
    els = [xt.Drift(length=1.0), xt.Quadrupole(length=0.3, k1=0.6),
           xt.Sextupole(length=0.2, k2=1.5), xt.Octupole(length=0.2, k3=2.5),
           xt.Multipole(knl=[0.0, 0.02], ksl=[0.0, 0.0]),
           xt.Bend(length=1.0, angle=0.1, edge_entry_angle=0.05, edge_exit_angle=0.05,
                   edge_entry_fint=0.5, edge_entry_hgap=0.02, k1=0.2),
           xt.UniformSolenoid(length=0.4, ks=0.3),
           xt.Cavity(voltage=1e6, frequency=325e6, phase=math.radians(60.0)),
           xt.Marker(), xt.LimitRect(min_x=-0.01, max_x=0.01, min_y=-0.02, max_y=0.02),
           xt.LimitEllipse(a=0.01, b=0.02), _new(xt.XYShift, dx=1e-3, dy=2e-3),
           xt.FirstOrderTaylorMap(length=0.1)]
    names = [f"e{i}" for i in range(len(els))]
    lat, rep = _round_trip_kinds(els, names, ref)
    kinds = [p.element.kind for p in lat.flatten()]
    assert kinds == ["Drift", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Bend",
                     "Solenoid", "RFCavity", "Marker", "Collimator", "Collimator", "Patch",
                     "Taylor"]
    q = lat.elements["e1"]
    assert q.multipole.Bn[1] == pytest.approx(0.6 * b)
    bd = lat.elements["e5"]
    assert bd.bend.angle == pytest.approx(0.1)
    assert bd.bend.e1 == pytest.approx(0.05)
    assert bd.multipole.Bn[1] == pytest.approx(0.2 * b)
    cav = lat.elements["e7"]
    assert cav.rf.phase_rad == pytest.approx(-math.pi / 6)
    assert cav.rf.voltage_V == pytest.approx(1e6)
    sol = lat.elements["e6"]
    assert sol.solenoid.Bsol_T == pytest.approx(0.3 * b)


def test_from_line_reads_lag_and_phase_together():
    line = xt.Line(elements=[_new(xt.Cavity, voltage=1e6, frequency=325e6, lag=30.0,
                                  phase=math.radians(30.0))],
                   element_names=["c"])
    lat = from_line(line, proton_ref())
    assert lat.elements["c"].rf.phase_rad == pytest.approx(math.radians(60.0 - 90.0))


def test_from_line_thin_multipole_is_a_kicker():
    lat, rep = _round_trip_kinds([xt.Multipole(knl=[-1e-3], ksl=[2e-3])], ["k"])
    k = lat.elements["k"]
    assert k.kind == "Kicker"
    assert k.hkick == pytest.approx(1e-3)
    assert k.vkick == pytest.approx(2e-3)
    assert "MULTIPOLE_AS_KICKER" in rep.codes()


def test_from_line_unknown_class_is_a_marker_with_a_native_passthrough():
    lat, rep = _round_trip_kinds([xt.Elens(current=1.0)], ["ee"])
    el = lat.elements["ee"]
    assert el.kind == "Marker"
    assert el.native["xtrack"]["element"]["__class__"] == "Elens"
    assert "UNSUPPORTED_XTRACK_ELEMENT" in rep.codes()
    # and it is re-emitted verbatim
    line = to_line(lat)
    assert isinstance(line.element_dict["ee"], xt.Elens)


def test_from_line_dipole_edge_is_unsupported_but_preserved():
    lat, rep = _round_trip_kinds([xt.DipoleEdge(e1=0.05, k=0.1, fint=0.5, hgap=0.02)], ["de"])
    assert "UNSUPPORTED_XTRACK_ELEMENT" in rep.codes()
    line = to_line(lat)
    assert isinstance(line.element_dict["de"], xt.DipoleEdge)
    assert float(line.element_dict["de"].e1) == pytest.approx(0.05)


def _sliced_line(mode: str):
    """A real sliced line (slices cannot be constructed by hand: their ``_parent`` must
    live in the same xobject buffer)."""
    line = xt.Line(elements={"q": xt.Quadrupole(length=0.6, k1=0.5),
                             "d": xt.Drift(length=1.0),
                             "b": xt.Bend(length=1.0, angle=0.1)},
                   element_names=["q", "d", "b"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Teapot(2, mode=mode))])
    return line


def test_from_line_merges_thick_slices():
    line = _sliced_line("thick")
    from lattix.fidelity import FidelityReport

    rep = FidelityReport()
    lat = from_line(line, proton_ref(), report=rep)
    flat = [p.element for p in lat.flatten()]
    assert "SLICE_MERGED" in rep.codes()
    quads = [e for e in flat if e.kind == "Quadrupole"]
    assert len(quads) == 2
    assert all(e.length == pytest.approx(0.3) for e in quads)
    assert all(e.multipole.Bn[1] == pytest.approx(0.5 * proton_ref().brho_signed)
               for e in quads)
    assert sum(e.length for e in flat) == pytest.approx(2.6, abs=1e-12)


def test_from_line_thin_slices_become_integrated_multipoles():
    line = _sliced_line("thin")
    b = proton_ref().brho_signed
    lat = from_line(line, proton_ref())
    flat = [p.element for p in lat.flatten()]
    thin = [e for e in flat if e.kind == "Multipole" and e.multipole.BnL.get(1)]
    assert len(thin) == 2                                # two Teapot kicks in the quad
    assert all(e.length == 0.0 for e in thin)
    assert sum(e.multipole.BnL[1] for e in thin) == pytest.approx(0.5 * 0.6 * b)
    assert sum(e.length for e in flat) == pytest.approx(2.6, abs=1e-12)


def test_from_line_bend_with_a_separate_k0_is_lossy():
    line = xt.Line(elements=[xt.Bend(length=1.0, angle=0.1, k0=0.05)], element_names=["b"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    from lattix.fidelity import FidelityReport

    rep = FidelityReport()
    lat = from_line(line, proton_ref(), report=rep)
    assert "BEND_K0_NE_H" in rep.codes()
    assert lat.elements["b"].native["xtrack"]["k0"] == pytest.approx(0.05)
    assert float(to_line(lat).element_dict["b"].k0) == pytest.approx(0.05)


def test_from_line_reads_misalignment_attributes():
    line = xt.Line(elements=[xt.Quadrupole(length=0.3, k1=0.5, shift_x=1e-3,
                                           rot_s_rad_no_frame=0.02)],
                   element_names=["q"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    lat = from_line(line, proton_ref())
    sh = lat.elements["q"].shift
    assert sh.x_offset == pytest.approx(1e-3)
    assert sh.tilt == pytest.approx(0.02)


# ------------------------------------------------------- ledger completeness
def test_every_element_gets_exactly_one_ledger_entry_in_each_direction():
    """I-14: one entry per source element, both ways (the CONST_P0 rigidity note that
    accompanies an accelerating element is the documented second row, as in MAD-X)."""
    from collections import Counter

    from lattix.fidelity import FidelityReport

    lat = all_kinds_lattice()
    rep = FidelityReport()
    line = to_line(lat, report=rep)
    names = [p.element.name for p in lat.flatten()]
    seen = Counter(e.element for e in rep.entries)
    assert set(names) <= set(seen), f"no ledger entry for {set(names) - set(seen)}"
    # the only names the ledger mentions that flatten() does not are superposition
    # children, which get their own entry on top of the parent's SUPERPOSITION_FLATTENED
    assert set(seen) - set(names) == {"sup_child"}
    # the accelerating elements additionally carry the rigidity note (as in MAD-X); the
    # field map carries its FM_AS_CAVITY rule row *and* CONST_P0, since its rule row is
    # about the map, not about p0
    rigidity = {"CONST_P0_DELTA_RIGIDITY", "CONST_P0_LOCAL_RIGIDITY", "CONST_P0_START_RIGIDITY",
                "CONST_P0_PHASE_SLIP"}
    for name in seen:
        codes = [e.code for e in rep.entries if e.element == name and e.code not in rigidity]
        assert len(codes) == (2 if name == "fm" else 1), f"{name}: {codes}"
    assert sorted(e.code for e in rep.entries if e.element == "fm") == \
        ["CONST_P0", "CONST_P0_DELTA_RIGIDITY", "CONST_P0_PHASE_SLIP", "FM_AS_CAVITY"]

    rep_back = FidelityReport()
    back = from_line(line, report=rep_back)
    back_names = [p.element.name for p in back.flatten()]
    # lattice-level notes (PHASE_SLIP_RESTORED, ENERGY_MODE_RESTORED) carry no element
    back_seen = Counter(e.element for e in rep_back.entries if e.element is not None)
    assert set(back_names) == set(back_seen)
    for name, n in back_seen.items():
        assert n == 1, f"{name} got {n} entries"


def test_slice_merge_records_one_entry_per_slice():
    from lattix.fidelity import FidelityReport

    line = xt.Line(elements={"q": xt.Quadrupole(length=0.6, k1=0.5)}, element_names=["q"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Teapot(2, mode="thick"))])
    rep = FidelityReport()
    lat = from_line(line, proton_ref(), report=rep)
    assert len(rep.entries) == len(lat.flatten())
    assert {e.code for e in rep.entries} <= {"SLICE_MERGED", "OK"}


def test_known_classes_matches_what_from_line_actually_maps():
    """Every class in KNOWN_CLASSES must come back as something other than the
    ``UNSUPPORTED_XTRACK_ELEMENT`` fallback, and a class outside it must not."""
    from lattix.fidelity import FidelityReport
    from lattix.formats.xtrack import KNOWN_CLASSES

    samples = {
        "Drift": xt.Drift(length=1.0), "Quadrupole": xt.Quadrupole(length=0.3, k1=0.5),
        "Sextupole": xt.Sextupole(length=0.2, k2=1.0),
        "Octupole": xt.Octupole(length=0.2, k3=1.0),
        "Multipole": xt.Multipole(knl=[0.0, 0.01]),
        "Bend": xt.Bend(length=1.0, angle=0.1),
        "RBend": xt.RBend(length_straight=1.0, angle=0.1),
        "UniformSolenoid": xt.UniformSolenoid(length=0.4, ks=0.3),
        "Solenoid": xt.Solenoid(length=0.4, ks=0.3),
        "Cavity": xt.Cavity(voltage=1e6, frequency=1e8),
        "Marker": xt.Marker(),
        "LimitRect": xt.LimitRect(min_x=-0.01, max_x=0.01, min_y=-0.01, max_y=0.01),
        "LimitEllipse": xt.LimitEllipse(a=0.01, b=0.01),
        "XYShift": _new(xt.XYShift, dx=1e-3, dy=0.0),
        "SRotation": _new(xt.SRotation, angle=1.0),
        "XRotation": _new(xt.XRotation, angle=1.0),
        "YRotation": _new(xt.YRotation, angle=1.0),
        "ZetaShift": _new(xt.ZetaShift, dzeta=1e-3),
        "FirstOrderTaylorMap": xt.FirstOrderTaylorMap(length=0.1),
        "ReferenceEnergyIncrease": xt.ReferenceEnergyIncrease(Delta_p0c=1e6),
    }
    assert set(samples) <= KNOWN_CLASSES
    for cname, el in samples.items():
        line = xt.Line(elements=[el], element_names=["e"])
        line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
        rep = FidelityReport()
        from_line(line, proton_ref(), report=rep)
        assert "UNSUPPORTED_XTRACK_ELEMENT" not in rep.codes(), cname

    line = xt.Line(elements=[xt.Elens(current=1.0)], element_names=["e"])
    line.particle_ref = xt.Particles(mass0=938272088.16, q0=1, kinetic_energy0=8e8)
    rep = FidelityReport()
    from_line(line, proton_ref(), report=rep)
    assert "UNSUPPORTED_XTRACK_ELEMENT" in rep.codes()
    assert "Elens" not in KNOWN_CLASSES
