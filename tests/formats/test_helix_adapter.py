"""HELIX ⇄ IR adapter (``lattix.formats.helix``), Phase 1 task 1.6.

Anti-cancellation: the round trip is judged by HELIX's *own* matrix engine
(``linac_gen.tracking.matrix_tracking.get_element_matrix`` with the reference
advanced exactly as ``lattix/oracles/helix.py`` does), not by comparing IR
objects to themselves — a symmetric bug in ``from_helix``/``to_helix`` would
still have to reproduce every 6x6 map, every element class and the final
reference energy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import lattix.formats.helix as helix_fmt
from lattix.fidelity import TranslationError
from lattix.formats.base import FORMATS, check_rules_coverage
from lattix.formats.helix import RULES, from_helix, read_helix_deck, to_helix
from lattix.ir import (
    ALL_KINDS,
    RFP,
    Drift,
    FieldMap,
    Kicker,
    Lattice,
    MagneticMultipoleP,
    Octupole,
    Patch,
    ReferenceParticle,
    RFCavity,
    Sextupole,
    species,
)
from lattix.ir.walk import propagate
from lattix.testing import needs

pytestmark = [pytest.mark.oracle_helix, needs("helix")]

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
HELIX_EXAMPLES = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/examples/pipii")

MATRIX_TOL = 1e-10
LENGTH_TOL = 1e-9        # mm
ENERGY_RTOL = 1e-9

# (deck, species, kinetic energy [eV]).  ``dtl_section`` accelerates protons
# through 8 GAPs at phi = -30 deg; the same deck DECELERATES H- (HELIX's
# advance_ref uses the signed charge), so H- is injected at 20 MeV to keep the
# reference above rest.
PUBLIC_CASES = [
    ("fodo_cell.dat", "proton", 2.1e6),
    ("fodo_cell.dat", "h-", 2.1e6),
    ("bend_line.dat", "h-", 2.1e6),
    ("bend_line.dat", "proton", 100e6),
    ("dtl_section.dat", "proton", 2.1e6),
    ("dtl_section.dat", "h-", 20e6),
    ("solenoid_channel.dat", "h-", 2.1e6),
    ("mebt_line.dat", "h-", 2.1e6),
    ("mebt_line.dat", "proton", 2.1e6),
]
PIPII_CASES = [
    (HELIX_EXAMPLES / "mebt" / "mebt.dat", "h-", 2.1e6),
    (HELIX_EXAMPLES / "mebt+hwr" / "mebt+hwr.dat", "h-", 2.1e6),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mods():
    return helix_fmt._helix()


def _helix_species(name: str):
    m = _mods()
    return {"proton": m.PROTON, "h-": m.H_MINUS, "deuteron": m.DEUTERON}[name]


def helix_walk(hlat, species_name: str, kinetic_energy_eV: float,
               frequency_Hz: float = 352.21e6):
    """Per-element 6x6 maps with the reference advanced as in lattix/oracles/helix.py."""
    m = _mods()
    from linac_gen.core.reference import ReferenceParticle as HRef
    from linac_gen.tracking.matrix_tracking import get_element_matrix

    ref = HRef(_helix_species(species_name), w_kin=kinetic_energy_eV / 1e6,
               frequency=frequency_Hz / 1e6)
    classes, lengths, mats, energies = [], [], [], []
    for e in hlat.elements:
        cls = type(e).__name__
        classes.append(cls)
        lengths.append(float(e.length))
        if cls == "Freq":
            if e.frequency_mhz:
                ref.frequency = float(e.frequency_mhz)
            mats.append(np.eye(6))
            energies.append(ref.w_kin)
            continue
        if cls == "SetBeamEnergy":
            ref.w_kin = float(e.energy_MeV)
            mats.append(np.eye(6))
            energies.append(ref.w_kin)
            continue
        if isinstance(e, m.FieldMapElement):
            e.reset_run_state()
        mats.append(np.asarray(get_element_matrix(e, ref), dtype=float))
        if isinstance(e, m.FieldMapElement):
            e.advance_ref(ref)
        else:
            ref.s += e.length
            if e.length > 0 and ref.wavelength > 0:
                ref.phi_s += 360.0 * e.length / (ref.beta * ref.wavelength)
            if isinstance(e, m.ThinKickElement):
                e.advance_ref(ref)
        energies.append(ref.w_kin)
    return classes, np.array(lengths), np.array(mats), np.array(energies), ref.w_kin


def assert_helix_roundtrip(hlat, species_name: str, kinetic_energy_eV: float) -> dict:
    lat, rep = from_helix(hlat, species=species_name, kinetic_energy_eV=kinetic_energy_eV)
    h2, rep2 = to_helix(lat)
    c1, l1, m1, _, w1 = helix_walk(hlat, species_name, kinetic_energy_eV)
    c2, l2, m2, _, w2 = helix_walk(h2, species_name, kinetic_energy_eV)
    assert c1 == c2, f"element class sequence changed: {c1[:20]} != {c2[:20]}"
    d_len = float(np.max(np.abs(l1 - l2))) if len(l1) else 0.0
    d_mat = float(np.max(np.abs(m1 - m2))) if len(m1) else 0.0
    d_w = abs(w1 - w2) / max(abs(w1), 1e-30)
    assert d_len <= LENGTH_TOL, f"lengths differ by {d_len} mm"
    assert d_mat <= MATRIX_TOL, f"per-element matrices differ by {d_mat}"
    assert d_w <= ENERGY_RTOL, f"final reference energy differs by {d_w} (rel)"
    return {"n": len(c1), "d_len_mm": d_len, "d_matrix": d_mat, "d_energy_rel": d_w,
            "report": rep, "report_back": rep2, "lattice": lat, "helix": h2}


def _synthetic_helix_lattice():
    """One HELIX lattice touching every element family this adapter maps."""
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Freq("FREQ", frequency_mhz=352.21))
    L.add(m.Edge("E1", pole_rotation=5.0, rho=2000.0, gap=40.0, k1=0.45, k2=2.8))
    L.add(m.Dipole("B1", angle=30.0, rho=2000.0, field_index=0.5, aperture=30.0))
    L.add(m.Edge("E2", pole_rotation=7.0, rho=2000.0, gap=40.0, k1=0.5, k2=2.8))
    L.add(m.Dipole("B2", angle=-12.0, rho=1500.0, aperture=25.0))
    L.add(m.Edge("LONE", pole_rotation=3.0, rho=1000.0, gap=20.0))
    L.add(m.ThinLens("TL", fx=2.0, fy=-3.0))
    L.add(m.MatrixElement("MX", matrix=np.eye(6), length=100.0))
    L.add(m.Multipole("MP", knl=[0.0, 0.02, 0.3], ksl=[0.0, 0.0, 0.1], tilt_deg=10.0,
                      dx=0.5, dy=-0.25))
    L.add(m.Foil("FOIL", material="C", thickness_ug_cm2=600.0))
    L.add(m.Aperture("AP", dx=15.0, dy=12.0, aperture_type=0))
    L.add(m.Aperture("AP2", dx=15.0, aperture_type=1))
    L.add(m.Marker("BPM1", is_bpm=True, diag_family=3, x_target_mm=0.1, accuracy_mm=0.5))
    L.add(m.Marker("MARK1", snapshot=True))
    L.add(m.Marker("PLAIN"))
    L.add(m.SpaceChargeComp("SCC", factor=0.9))
    L.add(m.ScGridDirective("SCG", extent_sigma=20.0))
    L.add(m.SetBeamEnergy("SBE", k=0, energy_MeV=5.0))
    L.add(m.SetBeamE0P0("SBP", k=1, dE_MeV=0.25, dphi_deg=0.0, ke=1, kp=0))
    L.add(m.COMMAND_CLASSES["ADJUST"]("ADJ", target="QUAD", param_idx=2, link_group=1,
                                      vmin=-30, vmax=30, start_step=0.5, kn=0))
    L.add(m.COMMAND_CLASSES["ADJUST_BEAM_TWISS"]("ABT", 1, 1, 1, 0, 0, 0, 0))
    L.add(m.COMMAND_CLASSES["SET_SIZE"]("SS", k=1, x_mm=4, y_mm=0, phi_or_z=0, k2=0))
    L.add(m.SetSyncPhase("SSP"))
    L.add(m.NCells("NC", mode=1, n_cells=5, beta_g=0.3, eot_v_per_m=2.0e6, theta_s_deg=-25.0,
                   aperture_mm=20.0, frequency_mhz=352.21, sync_phase=True))
    L.add(m.Steerer("ST", bx_l=0.001, by_l=-0.002))
    L.add(m.Steerer("STE", bx_l=100.0, by_l=-50.0, elec=True))
    L.add(m.Drift("D", length=100.0, aperture=20.0, x_shift=1.0, y_shift=-2.0, dx=0.5,
                  tilt_deg=3.0, pitch_deg=1.0, yaw_deg=-1.0))
    L.add(m.Quadrupole("Q", length=50.0, gradient=5.0, aperture=20.0, skew_angle=7.0,
                       g3=1.0, g4=2.0, g5=3.0, g6=4.0, gfr=15.0))
    L.add(m.RFGap("G", voltage=0.5, phase=-30.0, frequency=352.21, ttf=0.8, aperture=15.0))
    L.add(m.Solenoid("S", length=150.0, field=0.25, aperture=30.0))
    return L


# ---------------------------------------------------------------------------
# (d) RULES coverage / registry hygiene
# ---------------------------------------------------------------------------
def test_rules_cover_every_ir_kind():
    assert check_rules_coverage(helix_fmt) == set()
    assert set(RULES) == set(ALL_KINDS)


def test_helix_is_an_adapter_not_a_registered_format():
    assert "helix" not in FORMATS
    assert all(spec.module != "lattix.formats.helix" for spec in FORMATS.values())


# ---------------------------------------------------------------------------
# (a) HELIX round trip through the IR
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deck,sp,ke", PUBLIC_CASES,
                         ids=[f"{d}-{s}" for d, s, _ in PUBLIC_CASES])
def test_roundtrip_public_deck(deck, sp, ke):
    from linac_gen.io.tracewin_parser import parse_tracewin

    path = DATA / deck
    assert path.exists(), path
    h1, _meta = parse_tracewin(str(path))
    res = assert_helix_roundtrip(h1, sp, ke)
    assert res["n"] == len(h1.elements)
    # exactly one ledger entry per source element (bends fold 3 cards into 1)
    assert 0 < len(res["report"].entries) <= res["n"] + 1


@pytest.mark.slow
@pytest.mark.parametrize("deck,sp,ke", PIPII_CASES, ids=["mebt", "mebt+hwr"])
def test_roundtrip_pipii_deck(deck, sp, ke):
    if not Path(deck).exists():
        pytest.skip(f"HELIX example deck absent: {deck}")
    from linac_gen.io.tracewin_parser import parse_tracewin

    h1, _meta = parse_tracewin(str(deck))
    res = assert_helix_roundtrip(h1, sp, ke)
    assert res["n"] == len(h1.elements)
    # these decks exercise FIELD_MAP + SET_SYNC_PHASE (p_flag = 1)
    kinds = {p.element.kind for p in res["lattice"].flatten()}
    assert "FieldMap" in kinds
    assert any(e.card == "SET_SYNC_PHASE" for e in res["lattice"].elements.values()
               if e.kind == "Directive")


def test_roundtrip_every_element_family():
    L = _synthetic_helix_lattice()
    res = assert_helix_roundtrip(L, "h-", 5e6)
    kinds = [p.element.kind for p in res["lattice"].flatten()]
    for expected in ("Freq", "Bend", "Taylor", "Multipole", "Foil", "Collimator", "Instrument",
                     "Marker", "Directive", "ReferenceChange", "NCells", "Kicker", "Drift",
                     "Quadrupole", "RFCavity", "Solenoid"):
        assert expected in kinds, f"{expected} missing from {sorted(set(kinds))}"


def test_ir_walk_reproduces_the_helix_reference_energies():
    """`propagate()` on the converted IR must give HELIX's own per-element energies."""
    from linac_gen.io.tracewin_parser import parse_tracewin

    h1, _ = parse_tracewin(str(DATA / "dtl_section.dat"))
    lat, _rep = from_helix(h1, species="h-", kinetic_energy_eV=20e6)
    _c, _l, _m, energies_MeV, _w = helix_walk(h1, "h-", 20e6)
    ir = [p.ref_out.kinetic_energy_eV for p in propagate(lat)]
    assert len(ir) == len(energies_MeV)
    assert np.max(np.abs(np.array(ir) - energies_MeV * 1e6)) <= 1e-6   # eV


def test_lattice_is_json_serialisable():
    L = _synthetic_helix_lattice()
    lat, _rep = from_helix(L, species="proton", kinetic_energy_eV=5e6)
    import json

    d = lat.to_dict()
    json.dumps(d)                       # must not raise
    assert Lattice.from_dict(json.loads(json.dumps(d))).total_length == pytest.approx(
        lat.total_length)


def test_read_helix_deck():
    lat, rep = read_helix_deck(DATA / "mebt_line.dat", species="h-", kinetic_energy_eV=2.1e6)
    assert rep.source_format == "helix"
    assert rep.source_file.endswith("mebt_line.dat")
    kinds = [p.element.kind for p in lat.flatten()]
    assert kinds.count("RFCavity") == 2
    assert kinds.count("Quadrupole") == 8
    assert "Kicker" in kinds and "Collimator" in kinds


# ---------------------------------------------------------------------------
# (b) H- vs proton
# ---------------------------------------------------------------------------
def test_hminus_gap_phase_is_shifted_by_pi_and_keeps_helix_gain_sign():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice
    from linac_gen.core.reference import ReferenceParticle as HRef

    raw_phase_deg, volt_MV, w0_MeV = -30.0, 0.577, 20.0
    ir_phase, ir_gain, helix_gain = {}, {}, {}
    for sp in ("proton", "h-"):
        L = HLattice()
        L.add(m.RFGap("G", voltage=volt_MV, phase=raw_phase_deg, frequency=352.21))
        lat, _rep = from_helix(L, species=sp, kinetic_energy_eV=w0_MeV * 1e6)
        p = propagate(lat)[0]
        ir_phase[sp] = p.element.rf.phase_rad
        ir_gain[sp] = p.ref_out.kinetic_energy_eV - p.ref_in.kinetic_energy_eV
        href = HRef({"proton": m.PROTON, "h-": m.H_MINUS}[sp], w_kin=w0_MeV, frequency=352.21)
        L.elements[0].advance_ref(href)
        helix_gain[sp] = (href.w_kin - w0_MeV) * 1e6

    # proton keeps the deck phase; H- is shifted by exactly pi
    assert ir_phase["proton"] == pytest.approx(math.radians(raw_phase_deg), abs=1e-12)
    delta = abs(ir_phase["h-"] - ir_phase["proton"])
    assert delta == pytest.approx(math.pi, abs=1e-12)
    # the IR walk agrees with HELIX's own advance_ref, sign included
    for sp in ("proton", "h-"):
        assert ir_gain[sp] == pytest.approx(helix_gain[sp], rel=1e-12)
    assert ir_gain["proton"] > 0 and ir_gain["h-"] < 0
    assert ir_gain["h-"] == pytest.approx(-ir_gain["proton"], rel=1e-12)


def test_hminus_rigidity_conversions_flip_sign():
    """Kicker / multipole normalisation goes through the SIGNED rigidity (PLAN §4.1)."""
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    kicks = {}
    for sp in ("proton", "h-"):
        L = HLattice()
        L.add(m.Steerer("ST", bx_l=0.0, by_l=0.002))
        lat, _rep = from_helix(L, species=sp, kinetic_energy_eV=20e6)
        kicks[sp] = lat.elements["ST"].hkick
    assert kicks["proton"] > 0 > kicks["h-"]
    # magnitudes differ only by the H- / proton mass ratio (Brho = pc/(c|q|))
    assert kicks["h-"] == pytest.approx(-kicks["proton"], rel=2e-3)


# ---------------------------------------------------------------------------
# (c) every LOSSY / DROPPED / EQUIVALENT path is recorded; strict raises
# ---------------------------------------------------------------------------
def test_gap_records_sync_mode_assumed():
    """HELIX thin GAPs have no SET_SYNC_PHASE mode (tracewin_parser.py:701)."""
    from linac_gen.io.tracewin_parser import parse_tracewin

    h1, _ = parse_tracewin(str(DATA / "dtl_section.dat"))
    _lat, rep = from_helix(h1, species="proton", kinetic_energy_eV=2.1e6)
    assert rep.codes()["SYNC_MODE_ASSUMED"] == 8
    assert rep.ok                     # EQUIVALENT never blocks strict mode
    rep.raise_if(True)


def test_error_study_recorded_never_dropped():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Drift("D", length=100.0))
    L.errors.append({"kind": "ERROR_QUAD_NCPL_STAT", "sigma": 0.1})
    L.error_ratios = [0.0, 0.5, 1.0]
    L.error_cutoff = 3.0
    lat, rep = from_helix(L, species="proton", kinetic_energy_eV=2.1e6)
    assert "helix_errors" in lat.meta
    assert lat.meta["helix_errors"]["error_ratios"] == [0.0, 0.5, 1.0]
    assert "ERROR_STUDY_NOT_CONVERTED" in rep.codes()
    with pytest.raises(TranslationError):
        rep.raise_if(True)
    # and to_helix says so too, rather than silently forgetting it
    _h2, rep2 = to_helix(lat)
    assert "ERROR_STUDY_NOT_CONVERTED" in rep2.codes()


def test_lone_edge_and_unmodelled_aperture_are_lossy():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Edge("LONE", pole_rotation=3.0, rho=1000.0, gap=20.0))
    L.add(m.Aperture("PEPPER", dx=15.0, aperture_type=2))
    lat, rep = from_helix(L, species="proton", kinetic_energy_eV=2.1e6)
    codes = rep.codes()
    assert codes["EDGE_WITHOUT_BEND"] == 1
    assert codes["APERTURE_TYPE_UNMODELLED"] == 1
    with pytest.raises(TranslationError):
        rep.raise_if(True)
    # both survive the trip back
    h2, _rep2 = to_helix(lat)
    assert [type(e).__name__ for e in h2.elements] == ["Edge", "Aperture"]


def test_to_helix_downgrades_are_recorded_and_strict_raises():
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=100e6,
                            rf_frequency_Hz=352.21e6)
    els = [
        Patch(name="P1", x_offset=1e-3),
        RFCavity(name="CAV", length=0.3, rf=RFP(frequency_Hz=352.21e6, voltage_V=1e6,
                                                phase_rad=-0.5)),
        FieldMap(name="FM", length=0.2),
        Sextupole(name="SX", length=0.1, multipole=MagneticMultipoleP(Bn={2: 12.0})),
        Octupole(name="OC", length=0.1, multipole=MagneticMultipoleP(Bn={3: 30.0})),
    ]
    lat = Lattice.from_sequence("synth", els, ref)
    h, rep = to_helix(lat)
    codes = rep.codes()
    assert codes["PATCH_NOT_SUPPORTED"] == 1
    assert codes["THICK_CAVITY_SPLIT"] == 1        # DRIFT L/2 + GAP + DRIFT L/2
    assert codes["FM_FILES_MISSING"] == 1
    assert codes["THICK_POLE_AS_QUAD_HIGHER_ORDER"] == 2
    assert [type(e).__name__ for e in h.elements] == [
        "Marker", "Drift", "RFGap", "Drift", "Drift", "Quadrupole", "Quadrupole"]
    assert h.elements[1].length == pytest.approx(150.0)   # mm
    with pytest.raises(TranslationError):
        rep.raise_if(True)


def test_bend_edge_conflict_is_reported():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Edge("E1", pole_rotation=5.0, rho=2000.0, gap=40.0))
    L.add(m.Dipole("B1", angle=30.0, rho=2000.0, e1=9.0, e2=9.0))
    L.add(m.Edge("E2", pole_rotation=7.0, rho=2000.0, gap=40.0))
    lat, rep = from_helix(L, species="proton", kinetic_energy_eV=100e6)
    assert "BEND_EDGE_CONFLICT" in rep.codes()
    bend = lat.elements["B1"]
    # the EDGE cards win over the Dipole's own e1/e2
    assert bend.bend.e1 == pytest.approx(math.radians(5.0))
    assert bend.bend.e2 == pytest.approx(math.radians(7.0))


# ---------------------------------------------------------------------------
# (e) unit conversions and clustering rules
# ---------------------------------------------------------------------------
def test_edge_dipole_edge_becomes_one_bend():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Edge("E1", pole_rotation=5.0, rho=2000.0, gap=40.0, k1=0.45, k2=2.8))
    L.add(m.Dipole("B1", angle=-30.0, rho=2000.0, field_index=0.5, aperture=30.0, hv=1))
    L.add(m.Edge("E2", pole_rotation=7.0, rho=2000.0, gap=40.0, k1=0.5, k2=2.9))
    L.add(m.Drift("D", length=10.0))
    lat, rep = from_helix(L, species="proton", kinetic_energy_eV=100e6)
    placed = lat.flatten()
    assert [p.element.kind for p in placed] == ["Bend", "Drift"]
    b = placed[0].element
    assert b.bend.angle == pytest.approx(math.radians(-30.0))
    assert b.length == pytest.approx(2.0 * math.radians(30.0))          # |rho|*|theta| in m
    # HELIX Edge.pole_rotation = sign(theta) * e (madx_parser.py:445-449)
    assert b.bend.e1 == pytest.approx(-math.radians(5.0))
    assert b.bend.e2 == pytest.approx(-math.radians(7.0))
    assert b.bend.hgap == pytest.approx(0.020)                          # 40 mm gap -> 0.02 m
    assert b.bend.edge_int1 == pytest.approx(0.45)
    assert b.bend.edge_int2 == pytest.approx(0.5)
    assert b.bend.fringe_k2 == pytest.approx(2.8)
    assert b.bend.tilt_ref == pytest.approx(math.pi / 2)                # hv = 1 -> vertical
    # N = -k1*rho^2  =>  Bn1 = -N*Brho_signed/rho^2
    brho = lat.reference.brho_signed
    assert b.multipole.Bn[1] == pytest.approx(-0.5 * brho / 4.0)
    assert rep.counts["EXACT"] >= 1


def test_lone_dipole_keeps_zero_edges_and_no_edge_cards_on_the_way_back():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Dipole("B1", angle=45.0, rho=2000.0))
    lat, _rep = from_helix(L, species="proton", kinetic_energy_eV=100e6)
    b = lat.elements["B1"]
    assert b.bend.e1 == 0.0 and b.bend.e2 == 0.0
    h2, _rep2 = to_helix(lat)
    assert [type(e).__name__ for e in h2.elements] == ["Dipole"]


def test_unit_conversions_spot_checks():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Drift("D", length=250.0, aperture=20.0, aperture_y=12.0,
                  dx=1.0, dy=-2.0, dz=3.0, tilt_deg=90.0, pitch_deg=180.0, yaw_deg=-90.0))
    L.add(m.Quadrupole("Q", length=50.0, gradient=5.0, aperture=20.0, skew_angle=45.0,
                       g3=1.5, g4=2.5, g5=3.5, g6=4.5, gfr=15.0))
    L.add(m.Solenoid("S", length=150.0, field=0.25, aperture=30.0))
    L.add(m.RFGap("G", voltage=0.577, phase=-30.0, frequency=352.21, ttf=0.8, aperture=15.0))
    L.add(m.Foil("F", material="C", thickness_ug_cm2=600.0))
    L.add(m.Steerer("ST", bx_l=0.001, by_l=-0.002))
    L.add(m.Aperture("AP", dx=15.0, dy=12.0, aperture_type=0))
    L.add(m.Freq("FQ", frequency_mhz=162.5))
    lat, _rep = from_helix(L, species="proton", kinetic_energy_eV=100e6)
    e = lat.elements

    d = e["D"]
    assert d.length == pytest.approx(0.250)                       # mm -> m
    assert d.aperture.half_x == pytest.approx(0.020)
    assert d.aperture.half_y == pytest.approx(0.012)
    assert d.shift.x_offset == pytest.approx(1e-3)
    assert d.shift.z_offset == pytest.approx(3e-3)
    assert d.shift.tilt == pytest.approx(math.pi / 2)             # deg -> rad
    assert d.shift.x_rot == pytest.approx(math.pi)                # pitch
    assert d.shift.y_rot == pytest.approx(-math.pi / 2)           # yaw

    q = e["Q"]
    assert q.multipole.Bn[1] == pytest.approx(5.0)                # lab gradient stays T/m
    assert q.multipole.tilt[1] == pytest.approx(math.pi / 4)      # skew deg -> rad
    assert [q.multipole.Bn[n] for n in (2, 3, 4, 5)] == [1.5, 2.5, 3.5, 4.5]
    assert q.native["helix"]["gfr"] == pytest.approx(15.0)

    assert e["S"].solenoid.Bsol_T == pytest.approx(0.25)
    assert e["S"].length == pytest.approx(0.150)

    g = e["G"]
    assert g.length == 0.0
    assert g.rf.voltage_V == pytest.approx(0.577 * 0.8 * 1e6)     # MV * ttf -> V (effective)
    assert g.rf.ttf == pytest.approx(0.8)
    assert g.rf.frequency_Hz == pytest.approx(352.21e6)           # MHz -> Hz
    assert g.rf.phase_rad == pytest.approx(math.radians(-30.0))

    assert e["F"].thickness_kg_per_m2 == pytest.approx(600.0 * 1e-5)   # ug/cm2 -> kg/m2

    # the kick uses the LOCAL signed rigidity at the steerer's entrance (the GAP
    # upstream has already accelerated the reference), not the lattice entry Brho
    by_name = {p.element.name: p for p in propagate(lat)}
    brho = by_name["ST"].ref_in.brho_signed
    assert brho != lat.reference.brho_signed
    st = e["ST"]
    assert st.hkick == pytest.approx(-0.002 / brho)               # By*L -> horizontal kick
    assert st.vkick == pytest.approx(0.001 / brho)                # Bx*L -> vertical kick

    ap = e["AP"]
    assert ap.aperture.shape == "RECTANGULAR"
    assert ap.aperture.half_x == pytest.approx(0.015)
    assert ap.aperture.half_y == pytest.approx(0.012)

    assert e["FQ"].frequency_Hz == pytest.approx(162.5e6)


def test_electric_steerer_uses_the_electric_rigidity():
    m = _mods()
    from linac_gen.core.lattice import Lattice as HLattice

    L = HLattice()
    L.add(m.Steerer("STE", bx_l=100.0, by_l=-50.0, elec=True))
    lat, _rep = from_helix(L, species="proton", kinetic_energy_eV=100e6)
    k = lat.elements["STE"]
    r = lat.reference
    erho = r.beta * 299_792_458.0 * r.brho_signed
    assert k.electric is True
    assert k.hkick == pytest.approx(100.0 / erho)     # elec: bx_l is the HORIZONTAL kick
    assert k.vkick == pytest.approx(-50.0 / erho)
    h2, _ = to_helix(lat)
    assert isinstance(h2.elements[0], m.Steerer) and h2.elements[0].elec is True
    assert h2.elements[0].bx_l == pytest.approx(100.0)


def test_kicker_from_a_foreign_ir_becomes_an_integrated_field():
    """A Kicker with no HELIX native block still converts through the local Brho."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=100e6,
                            rf_frequency_Hz=352.21e6)
    lat = Lattice.from_sequence("k", [Drift(name="D", length=1.0),
                                      Kicker(name="K", hkick=1e-3, vkick=-2e-3)], ref)
    h, rep = to_helix(lat)
    assert rep.ok
    st = h.elements[1]
    assert st.by_l == pytest.approx(1e-3 * ref.brho_signed)
    assert st.bx_l == pytest.approx(-2e-3 * ref.brho_signed)
