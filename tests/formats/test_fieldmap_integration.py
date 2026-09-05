"""``integrate_map`` and the TraceWin reader's field-map post-pass.

The hand-computed cases use maps written by the test; the last block cross-checks the whole
integration against HELIX's own ``advance_ref`` on the real (undistributed) PIP-II maps.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix.formats.tracewin import read
from lattix.ir.elements import RFP, FieldMap, Superposition
from lattix.ir.fieldmap import (
    FieldMapData,
    annotate_field_maps,
    clear_cache,
    integrate_map,
)
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate
from lattix.testing import needs
from tests.formats.test_fieldmap_data import write_1d

HELIX_EXAMPLES = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/examples")
FIELDS = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/Fields")


def ref_at(ke_eV: float, freq_Hz: float | None = None, sp: str = "proton") -> ReferenceParticle:
    return ReferenceParticle(species=species(sp), kinetic_energy_eV=ke_eV, rf_frequency_Hz=freq_Hz)


def uniform_map(tmp_path: Path, name: str, value: float, length_m: float, n: int = 50,
                suffix: str = ".edz") -> FieldMapData:
    write_1d(tmp_path / f"{name}{suffix}", [value] * (n + 1), zmax_m=length_m)
    geom = 100 if suffix == ".edz" else 10
    return FieldMapData.load(geom, tmp_path, name)


def fieldmap(length: float, *, ke: float = 1.0, kb: float = 1.0, phase_rad: float = 0.0,
             freq: float | None = None, sync: bool = True, geom: int = 100) -> FieldMap:
    return FieldMap(name="fm", length=length, geom=geom, files=["fm"], ke=ke, kb=kb,
                    rf=RFP(frequency_Hz=freq, phase_rad=phase_rad, phase_is_sync=sync))


# ------------------------------------------------------------------ electric
def test_uniform_gap_on_crest_gives_the_field_integral(tmp_path):
    """1 MV/m over 0.4 m at φs = 0 with no RF clock: dE = q·ke·∫Ez dz, exactly."""
    data = uniform_map(tmp_path, "u", 1.0, 0.4)
    s = integrate_map(fieldmap(0.4, ke=2.0), ref_at(5e6), data)
    assert s.kind == "rf"
    assert s.int_Ez_V == pytest.approx(2.0 * 1e6 * 0.4)
    assert s.int_abs_Ez_V == pytest.approx(8e5)
    assert s.ttf == pytest.approx(1.0)                      # no RF clock → no transit-time loss
    assert s.v_eff_V == pytest.approx(8e5) and s.v_c_V == pytest.approx(8e5)
    assert s.dE_ref_eV == pytest.approx(8e5)
    assert s.phase_sync_rad == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("phase_deg", [0.0, -30.0, -90.0, 120.0])
def test_gain_is_v_c_times_cos_phase_sync(tmp_path, phase_deg):
    """The writers' contract: dE_ref = v_c·cos(φs) identically, at any phase."""
    data = uniform_map(tmp_path, "u", 1.0, 0.3)
    s = integrate_map(fieldmap(0.3, phase_rad=math.radians(phase_deg), freq=162.5e6),
                      ref_at(2.1e6, 162.5e6), data)
    assert s.dE_ref_eV == pytest.approx(s.v_c_V * math.cos(s.phase_sync_rad), rel=1e-12, abs=1e-9)
    assert math.degrees(s.phase_sync_rad) == pytest.approx(phase_deg, abs=0.01)


def test_transit_time_factor_of_a_uniform_gap_is_the_sinc(tmp_path):
    """T = |∫Ez e^{iωz/βc} dz| / ∫|Ez| dz = |sinc(kL/2)| for a flat field."""
    length, freq = 0.03, 100e6
    ke_eV = ref_at(0.0).species.mass_eV * (1 / math.sqrt(1 - 0.05**2) - 1)   # β = 0.05
    ref = ref_at(ke_eV, freq)
    data = uniform_map(tmp_path, "u", 1.0, length)
    s = integrate_map(fieldmap(length, freq=freq), ref, data)
    k = 2 * math.pi / (ref.beta * 299_792_458.0 / freq)
    assert s.beta_in == pytest.approx(0.05, rel=1e-9)
    assert s.ttf == pytest.approx(abs(math.sin(k * length / 2) / (k * length / 2)), rel=1e-4)
    assert s.v_eff_V == pytest.approx(s.int_abs_Ez_V * s.ttf, rel=1e-12)


def test_sync_phase_gain_is_species_independent(tmp_path):
    """SET_SYNC_PHASE makes θ a synchronous phase: an H⁻ deck gains what a proton deck does."""
    data = uniform_map(tmp_path, "u", 1.0, 0.2)
    fm = fieldmap(0.2, phase_rad=math.radians(-25.0), freq=162.5e6)
    p = integrate_map(fm, ref_at(2.1e6, 162.5e6, "proton"), data)
    h = integrate_map(fm, ref_at(2.1e6, 162.5e6, "h-"), data)
    assert p.dE_ref_eV > 0 and h.dE_ref_eV > 0
    assert h.dE_ref_eV == pytest.approx(p.dE_ref_eV, rel=2e-3)     # only the masses differ
    # the sign flip lives in ψ, exactly as HELIX's scan fit puts it there
    assert abs(abs(h.sync_offset_deg - p.sync_offset_deg) - 180.0) < 1.0
    for s in (p, h):
        assert math.degrees(s.phase_sync_rad) == pytest.approx(-25.0, abs=0.01)


def test_relative_phase_map_ignores_the_running_rf_phase(tmp_path):
    """Without SET_SYNC_PHASE the deck phase is *relative*: the RF phase when the reference particle
    enters the map, whatever the bunch clock says.  LightWin's Envelope3D reproduces the ADS design
    deck that way (20 → 502.24 MeV, its own regression value; lattix 502.22 MeV), HELIX adds the
    running phase and ends at 22.6 MeV — measured 2026-09-05, docs/oracles.md."""
    data = uniform_map(tmp_path, "u", 1.0, 0.02)
    fm = fieldmap(0.02, phase_rad=0.0, freq=162.5e6, sync=False)
    ref = ref_at(2.1e6, 162.5e6)
    a = integrate_map(fm, ref, data, entrance_phase_deg=0.0)
    b = integrate_map(fm, ref, data, entrance_phase_deg=90.0)
    assert a.dE_ref_eV > 0                                    # on crest at the entrance: accelerating
    assert b.dE_ref_eV == pytest.approx(a.dE_ref_eV, rel=1e-12)   # the bunch clock does not matter
    assert a.dE_ref_eV == pytest.approx(a.v_c_V * math.cos(a.phase_sync_rad), rel=1e-12)
    # a quarter period later *in the card* the same map decelerates: the card phase is what counts
    c = integrate_map(fieldmap(0.02, phase_rad=math.pi / 2, freq=162.5e6, sync=False), ref, data)
    assert c.dE_ref_eV < 0
    assert math.degrees(c.phase_rf_rad - a.phase_rf_rad) == pytest.approx(90.0)
    # a synchronous-phase map ignores the running phase entirely (TraceWin SET_SYNC_PHASE)
    sync = fieldmap(0.02, phase_rad=0.0, freq=162.5e6, sync=True)
    assert integrate_map(sync, ref, data, entrance_phase_deg=90.0).dE_ref_eV == \
        pytest.approx(integrate_map(sync, ref, data, entrance_phase_deg=0.0).dE_ref_eV)


def test_step_count_follows_the_maps_own_grid(tmp_path):
    data = uniform_map(tmp_path, "u", 1.0, 0.4, n=120)
    assert integrate_map(fieldmap(0.4), ref_at(5e6), data).n_steps == 120     # HELIX: N_z − 1
    small = uniform_map(tmp_path, "s", 1.0, 0.4, n=8)
    assert integrate_map(fieldmap(0.4), ref_at(5e6), small).n_steps == 50     # floor


# ------------------------------------------------------------------ magnetic
def test_static_solenoid_map_hard_edge_preserves_both_integrals(tmp_path):
    """Uniform 1 T over the whole 0.4 m map: L_eff = 0.4 m, B_eff = kb·1 T."""
    write_1d(tmp_path / "s.bsz", [1.0] * 61, zmax_m=0.4)
    data = FieldMapData.load(10, tmp_path, "s")
    s = integrate_map(fieldmap(0.4, kb=2.0, geom=10), ref_at(5e6), data)
    assert s.kind == "solenoid" and s.dE_ref_eV == 0.0
    assert s.int_Bz_Tm == pytest.approx(0.8) and s.int_Bz2_T2m == pytest.approx(1.6)
    assert s.L_eff_m == pytest.approx(0.4) and s.B_eff_T == pytest.approx(2.0)
    assert s.L_eff_m == pytest.approx(s.int_Bz_Tm**2 / s.int_Bz2_T2m)
    assert s.B_eff_T * s.L_eff_m == pytest.approx(s.int_Bz_Tm)


def test_triangular_solenoid_profile_shortens_the_hard_edge(tmp_path):
    z = np.linspace(0.0, 1.0, 101)
    write_1d(tmp_path / "t.bsz", 1.0 - abs(2 * z - 1.0), zmax_m=1.0)      # triangle, peak 1 T
    data = FieldMapData.load(10, tmp_path, "t")
    s = integrate_map(fieldmap(1.0, geom=10), ref_at(5e6), data)
    assert s.int_Bz_Tm == pytest.approx(0.5, rel=1e-6)                    # ½·base·height
    assert s.int_Bz2_T2m == pytest.approx(1 / 3, rel=1e-3)
    assert s.L_eff_m == pytest.approx(0.75, rel=1e-3)                     # (1/2)²/(1/3)
    assert s.B_eff_T == pytest.approx(2 / 3, rel=1e-3)


def test_negative_kb_keeps_the_sign_of_b_eff(tmp_path):
    write_1d(tmp_path / "s.bsz", [1.0] * 61, zmax_m=0.3)
    data = FieldMapData.load(10, tmp_path, "s")
    s = integrate_map(fieldmap(0.3, kb=-1.8, geom=10), ref_at(5e6), data)
    assert s.int_Bz_Tm < 0 and s.int_Bz2_T2m > 0
    assert s.B_eff_T == pytest.approx(-1.8) and s.L_eff_m == pytest.approx(0.3)


def test_quadrupole_map_geom_digit_9_integrates_the_gradient(tmp_path):
    write_1d(tmp_path / "q.bsz", [5.0] * 51, zmax_m=0.2)
    data = FieldMapData.load(90, tmp_path, "q")               # stat_B digit 9
    s = integrate_map(fieldmap(0.2, kb=1.0, geom=90), ref_at(5e6), data)
    assert s.kind == "quad"
    assert s.int_Gz_Tm_per_m == pytest.approx(1.0)            # 5 T/m × 0.2 m
    assert s.int_Gz2_T2m_per_m2 == pytest.approx(5.0)
    assert s.int_Bz_Tm == 0.0 and s.dE_ref_eV == 0.0


# ------------------------------------------------------------------ reader post-pass
def _deck(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "deck.dat"
    p.write_text(body)
    return p


def test_reader_fills_dE_ref_voltage_ttf_and_map_summary(tmp_path):
    write_1d(tmp_path / "cav.edz", [0.0, 1.0, 2.0, 1.0, 0.0], zmax_m=0.2)
    deck = _deck(tmp_path, "FREQ 162.5\nSET_SYNC_PHASE\n"
                           "FIELD_MAP 100 200 -30 20 0 1.5 0 0 cav\nEND\n")
    lat, rep = read(deck, species="h-", kinetic_energy_eV=2.1e6)
    fm = next(e for e in lat.elements.values() if isinstance(e, FieldMap))
    s = fm.meta["map_summary"]
    assert rep.ok and [e.code for e in rep.entries if e.element == fm.name] == ["FM_INTEGRATED"]
    assert s["kind"] == "rf" and s["n_steps"] == 50
    assert fm.rf.dE_ref_eV == pytest.approx(s["dE_ref_eV"])
    assert fm.rf.voltage_V == pytest.approx(s["v_eff_V"]) and fm.rf.ttf == pytest.approx(s["ttf"])
    assert fm.rf.dE_ref_eV == pytest.approx(s["v_c_V"] * math.cos(s["phase_sync_rad"]), rel=1e-9)
    assert math.degrees(s["phase_sync_rad"]) == pytest.approx(-30.0, abs=0.01)
    # the walk now moves the reference across the map
    assert propagate(lat)[-1].ref_out.kinetic_energy_eV == \
        pytest.approx(2.1e6 + fm.rf.dE_ref_eV)
    assert set(fm.meta["field_files_sha256"]) == {str(tmp_path / "cav.edz")}


def test_reader_static_map_records_hard_edge_numbers(tmp_path):
    write_1d(tmp_path / "sol.bsz", [1.0] * 61, zmax_m=0.3)
    lat, rep = read(_deck(tmp_path, "FIELD_MAP 10 300 0 16 -1.8 0 0 0 sol\nEND\n"), species="h-")
    fm = next(e for e in lat.elements.values() if isinstance(e, FieldMap))
    s = fm.meta["map_summary"]
    assert s["kind"] == "solenoid" and s["B_eff_T"] == pytest.approx(-1.8)
    assert s["L_eff_m"] == pytest.approx(0.3) and s["int_Bz_Tm"] == pytest.approx(-0.54)
    assert fm.rf.dE_ref_eV == 0.0 and fm.rf.voltage_V == 0.0


def test_reader_keeps_fm_files_missing_and_never_integrates(tmp_path):
    lat, rep = read(_deck(tmp_path, "FIELD_MAP 100 200 -30 20 0 1 0 0 nofile\nEND\n"),
                    species="h-")
    fm = next(e for e in lat.elements.values() if isinstance(e, FieldMap))
    assert "FM_FILES_MISSING" in rep.codes() and "map_summary" not in fm.meta
    assert fm.rf.dE_ref_eV is None


def test_reader_reports_a_map_it_cannot_decode(tmp_path):
    (tmp_path / "bad.edz").write_text("not a field map\n")
    lat, rep = read(_deck(tmp_path, "FIELD_MAP 100 200 -30 20 0 1 0 0 bad\nEND\n"), species="h-")
    fm = next(e for e in lat.elements.values() if isinstance(e, FieldMap))
    entry = next(e for e in rep.entries if e.element == fm.name)
    assert entry.code == "FM_NOT_INTEGRATED" and entry.cls.value == "EXACT"
    assert fm.rf.dE_ref_eV is None and rep.ok                    # the card itself is not lossy
    assert any("not integrated" in w for w in lat.warnings)


def test_field_maps_option_turns_the_pass_off(tmp_path):
    write_1d(tmp_path / "cav.edz", [0.0, 1.0, 0.0], zmax_m=0.2)
    deck = _deck(tmp_path, "FREQ 162.5\nFIELD_MAP 100 200 -30 20 0 1 0 0 cav\nEND\n")
    lat, _ = read(deck, species="h-", field_maps=False)
    fm = next(e for e in lat.elements.values() if isinstance(e, FieldMap))
    assert fm.rf.dE_ref_eV is None and "map_summary" not in fm.meta


def test_superposition_cluster_sums_its_children_in_order(tmp_path):
    write_1d(tmp_path / "a.edz", [1.0] * 51, zmax_m=0.2)
    deck = _deck(tmp_path, "FREQ 162.5\nSUPERPOSE_MAP 0\n"
                           "FIELD_MAP 100 200 0 20 0 1 0 0 a\n"
                           "SUPERPOSE_MAP 100\nFIELD_MAP 100 200 0 20 0 1 0 0 a\n"
                           "DRIFT 10 20\nEND\n")
    lat, rep = read(deck, species="proton", kinetic_energy_eV=5e6)
    cluster = next(e for e in lat.elements.values() if isinstance(e, Superposition))
    kids = [lat.elements[n] for _z, n in cluster.children]
    assert all(k.rf.dE_ref_eV is not None for k in kids)
    assert cluster.rf.dE_ref_eV == pytest.approx(sum(k.rf.dE_ref_eV for k in kids))
    # the second child is integrated at the energy the first one leaves behind
    assert kids[1].meta["map_summary"]["beta_in"] > kids[0].meta["map_summary"]["beta_in"]
    assert cluster.meta["map_summary"]["kind"] == "cluster"
    assert propagate(lat)[-1].ref_out.kinetic_energy_eV == \
        pytest.approx(5e6 + cluster.rf.dE_ref_eV)
    assert [e.code for e in rep.entries if e.element == cluster.name] == ["FM_INTEGRATED"]


def test_annotate_is_idempotent_and_reusable_on_any_lattice(tmp_path):
    write_1d(tmp_path / "cav.edz", [1.0] * 21, zmax_m=0.2)
    lat, _ = read(_deck(tmp_path, "FREQ 162.5\nFIELD_MAP 100 200 -30 20 0 1 0 0 cav\nEND\n"),
                  species="proton", kinetic_energy_eV=5e6)
    fm = next(e for e in lat.elements.values() if isinstance(e, FieldMap))
    first = fm.rf.dE_ref_eV
    out = annotate_field_maps(lat)
    assert [n for n, _ in out] == [fm.name]
    assert fm.rf.dE_ref_eV == pytest.approx(first)


# ------------------------------------------------------------------ HELIX cross-check
DECKS = [(HELIX_EXAMPLES / "pipii" / "mebt" / "mebt.dat", 4),
         (HELIX_EXAMPLES / "pipii" / "mebt+hwr" / "mebt+hwr.dat", 20)]


def _helix_gains(deck: Path, ke_MeV: float, freq_MHz: float) -> list[float]:
    """HELIX's own ``advance_ref`` per FIELD_MAP, in eV (the reference lattix must match)."""
    from lattix.oracles.helix import _import_helix

    _import_helix()
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle as HRef
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    from linac_gen.io.tracewin_parser import parse_tracewin

    hlat, _meta = parse_tracewin(str(deck))
    ref = HRef(H_MINUS, w_kin=ke_MeV, frequency=freq_MHz)
    out = []
    for e in hlat.elements:
        cls = type(e).__name__
        if cls == "Freq":
            ref.frequency = float(e.frequency_mhz)
            continue
        if cls == "SetBeamEnergy":
            ref.w_kin = float(e.energy_MeV)
            continue
        if isinstance(e, FieldMapElement):
            w0 = ref.w_kin
            e.reset_run_state()
            e.advance_ref(ref)
            out.append((ref.w_kin - w0) * 1e6)
            continue
        ref.s += e.length
        if e.length > 0 and ref.wavelength > 0:
            ref.phi_s += 360.0 * e.length / (ref.beta * ref.wavelength)
        if isinstance(e, ThinKickElement):
            e.advance_ref(ref)
    return out


@pytest.mark.oracle_helix
@needs("helix")
@pytest.mark.parametrize("deck, n_maps", DECKS, ids=lambda v: getattr(v, "name", v))
def test_dE_ref_matches_helix_advance_ref_on_the_pipii_maps(deck, n_maps):
    """PLAN §6 task 3.1 acceptance: every real PIP-II map agrees with HELIX to ≤ 1e-6
    relative (the bunchers sit at φs = −90°, where the gain is ~1e-2 eV and only the
    absolute agreement is meaningful)."""
    if not deck.is_file() or not FIELDS.is_dir():
        pytest.skip("PIP-II decks / ANL-CEA maps not available")
    clear_cache()
    lat, _rep = read(deck, species="h-", kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    mine = [e.rf.dE_ref_eV for e in lat.elements.values() if isinstance(e, FieldMap)]
    theirs = _helix_gains(deck, 2.1, 162.5)
    assert len(mine) == len(theirs) == n_maps
    for got, want in zip(mine, theirs, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-5)
    placed = propagate(lat)
    assert placed[-1].ref_out.kinetic_energy_eV == \
        pytest.approx(2.1e6 + sum(theirs), rel=1e-12)
