"""IMPACT-Z oracle: what ``ImpactZexe`` can and cannot measure (PLAN §6 task 3.3).

The adapter turns a deck into a 13-particle finite-difference probe (``flagdist = 19``
+ ``particle.in``) with a ``-2`` phase-space dump at every element boundary, so it
delivers **per-element 6×6 maps** as well as the reference energy profile from
``fort.18``.  These tests pin

* the basis conventions (PLAN §5.1: never trust a remembered longitudinal sign);
* the maps against closed-form MAD-X thick-element maps and against cpymad itself;
* the reference energy through IMPACT-Z's ideal-cavity model against
  :func:`lattix.ir.walk.propagate`;
* what is *not* available (Twiss, dispersion, floor coordinates).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import lattix.oracles.impactz as _impactz_adapter  # noqa: F401  (registers "impactz")
from lattix.formats.impactz import Writer
from lattix.ir.elements import (
    RFP,
    Bend,
    BendP,
    Drift,
    MagneticMultipoleP,
    Marker,
    Quadrupole,
    RFCavity,
    Solenoid,
    SolenoidP,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate
from lattix.oracles import get_oracle
from lattix.oracles.base import Basis, BeamSpec, Probe
from lattix.oracles.basis import drift_common
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, PHI_S_DEG, V_VOLT
from lattix.oracles.impactz import ImpactzOracle, transform_to_common
from lattix.testing import needs

pytestmark = [pytest.mark.oracle_impactz, needs("impactz")]

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
MASS = species("proton").mass_eV


def ref(ke: float = KE_EV, freq: float = FREQ_HZ) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke,
                             rf_frequency_Hz=freq)


def deck(tmp_path: Path, elements, *, reference=None, name="t", **kw) -> Path:
    lat = Lattice.from_sequence(name, list(elements), reference or ref())
    out = tmp_path / "ImpactZ.in"
    Writer().write(lat, out, **kw)
    return out


def run(tmp_path: Path, elements, *, reference=None, write_kw=None, **kw):
    d = deck(tmp_path, elements, reference=reference, **(write_kw or {}))
    return get_oracle("impactz").run(d, workdir=tmp_path / "run", **kw)


def brho(r: ReferenceParticle) -> float:
    return r.brho_signed


# ---------------------------------------------------------------- availability
def test_available_and_registered():
    o = get_oracle("impactz")
    assert isinstance(o, ImpactzOracle)
    ok, why = o.available()
    assert ok, why
    assert "IMPACT-Z" in why and o.formats == ("impactz",)


def test_only_reads_impactz_decks(tmp_path):
    with pytest.raises(ValueError, match="ImpactZ.in decks only"):
        get_oracle("impactz").run(tmp_path / "x", fmt="madx")


# ------------------------------------------------------------------- fingerprint
def test_drift_matches_the_common_basis_analytic_map(tmp_path):
    """PLAN §5.1 basis fingerprint: a 1 m drift must give the same R56 as every
    other engine (docs/oracles.md pins +0.9955387)."""
    r = run(tmp_path, [Drift(name="d", length=1.0)])
    assert r.basis is Basis.COMMON
    expect = drift_common(1.0, KE_EV, MASS)
    assert r.R_elem[0] == pytest.approx(expect, abs=3e-9)
    assert r.R_elem[0][4, 5] == pytest.approx(0.9955387, abs=1e-7)
    assert r.R_elem[0][0, 1] == pytest.approx(1.0, abs=1e-11)


def test_native_basis_is_the_measured_one(tmp_path):
    """(x/Scxl, γβx, y/Scxl, γβy, ω·Δt [rad, late-positive], γ_ref − γ)."""
    r = run(tmp_path, [Drift(name="d", length=1.0)], basis="native")
    assert r.basis is Basis.IMPACTZ
    xl = r.meta["scxl_m"]
    assert xl == pytest.approx(299_792_458.0 / (2 * math.pi * FREQ_HZ))
    beta, gamma = _beta_gamma(KE_EV)
    # native R12 = L/(Scxl·γβ) — x is in units of Scxl, px in units of mc
    assert r.R_elem[0][0, 1] == pytest.approx(1.0 / (xl * beta * gamma), rel=1e-11)
    T = transform_to_common(KE_EV, MASS, FREQ_HZ)
    assert np.diag(T)[0] == pytest.approx(xl)
    assert np.diag(T)[4] < 0 and np.diag(T)[5] < 0        # late-positive φ, q6 = γ0 − γ
    common = T @ r.R_elem[0] @ np.linalg.inv(T)
    assert common == pytest.approx(drift_common(1.0, KE_EV, MASS), abs=3e-9)


def test_basis_module_agrees_with_the_adapter(tmp_path):
    """``lattix.oracles.basis``'s Basis.IMPACTZ branch was corrected from this adapter's
    measurement (2026-09-03); the two transforms must agree and the adapter must not warn."""
    from lattix.oracles.basis import transform_matrix

    module = transform_matrix(Basis.IMPACTZ, KE_EV, MASS, FREQ_HZ)
    adapter = transform_to_common(KE_EV, MASS, FREQ_HZ)
    assert np.allclose(np.diag(module), np.diag(adapter), rtol=1e-12, atol=0)
    r = run(tmp_path, [Drift(name="d", length=1.0)])
    assert not any("lattix.oracles.basis" in w for w in r.warnings)


def _beta_gamma(ke: float) -> tuple[float, float]:
    g = 1.0 + ke / MASS
    return math.sqrt(1.0 - 1.0 / (g * g)), g


# --------------------------------------------------------------- element maps
def test_quadrupole_map_matches_the_analytic_thick_quad(tmp_path):
    r0 = ref()
    g = 5.0
    r = run(tmp_path, [Quadrupole(name="q", length=0.3,
                                  multipole=MagneticMultipoleP(Bn={1: g}))],
            reference=r0)
    k = math.sqrt(g / brho(r0))
    L = 0.3
    expect = np.eye(6)
    expect[0, 0] = expect[1, 1] = math.cos(k * L)
    expect[0, 1] = math.sin(k * L) / k
    expect[1, 0] = -k * math.sin(k * L)
    expect[2, 2] = expect[3, 3] = math.cosh(k * L)
    expect[2, 3] = math.sinh(k * L) / k
    expect[3, 2] = k * math.sinh(k * L)
    expect[4, 5] = L / r0.gamma ** 2
    assert r.R_elem[0][:4, :4] == pytest.approx(expect[:4, :4], abs=1e-12)
    assert r.R_elem[0][4, 5] == pytest.approx(expect[4, 5], abs=3e-9)


def test_map_steps_matter(tmp_path):
    """``bmpstp`` is the number of map integration steps: a single one costs 1.2e-4 on
    a 0.3 m quadrupole, twenty bring it to the analytic map (4e-14)."""
    els = [Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 5.0}))]
    kw = {"steps_per_m": 0.1}          # one integration segment
    coarse = run(tmp_path / "a", els, write_kw={"map_steps": 1, **kw})
    fine = run(tmp_path / "b", els, write_kw={"map_steps": 20, **kw})
    d = np.abs(coarse.R_elem[0][:4, :4] - fine.R_elem[0][:4, :4]).max()
    assert 1e-5 < d < 1e-2


def test_solenoid_map_matches_the_analytic_map(tmp_path):
    r0 = ref()
    B, L = 0.5, 0.4
    r = run(tmp_path, [Solenoid(name="s", length=L, solenoid=SolenoidP(Bsol_T=B))],
            reference=r0)
    K = B / brho(r0) / 2.0
    c, s = math.cos(K * L), math.sin(K * L)
    expect = np.array([[c * c, s * c / K, s * c, s * s / K],
                       [-K * s * c, c * c, -K * s * s, s * c],
                       [-s * c, -s * s / K, c * c, s * c / K],
                       [K * s * s, -s * c, -K * s * c, c * c]])
    assert r.R_elem[0][:4, :4] == pytest.approx(expect, abs=1e-12)


def test_sector_bend_matches_the_madx_analytic_map(tmp_path):
    r0 = ref()
    L, ang = 1.0, 0.1
    r = run(tmp_path, [Bend(name="b", length=L, bend=BendP(angle=ang))], reference=r0)
    h = ang / L
    expect = np.eye(6)
    expect[0, 0] = expect[1, 1] = math.cos(ang)
    expect[0, 1] = math.sin(ang) / h
    expect[1, 0] = -h * math.sin(ang)
    expect[2, 3] = L
    expect[0, 5] = (1 - math.cos(ang)) / h
    expect[1, 5] = math.sin(ang)
    expect[4, 0] = -math.sin(ang)
    expect[4, 1] = -(1 - math.cos(ang)) / h
    expect[4, 5] = -(ang - math.sin(ang)) / h + L / r0.gamma ** 2
    assert r.R_elem[0] == pytest.approx(expect, abs=1e-8)


@needs("madx")
def test_quadrupole_transverse_2x2_vs_cpymad(tmp_path):
    """The same IR lattice through both writers, compared element by element.
    MEASURED 2026-09-03: transverse 4×4 3.6e-15, R56 1.4e-9 (IMPACT-Z's low-β
    cancellation, see DEFAULT_STEPS)."""
    from lattix.formats.madx import Writer as MadxWriter
    from lattix.oracles.compare import compare_pair

    r0 = ref(ke=8e8, freq=352.21e6)
    els = [Drift(name="d0", length=0.25),
           Quadrupole(name="qf", length=0.3, multipole=MagneticMultipoleP(Bn={1: 2.0})),
           Drift(name="d1", length=0.5),
           Quadrupole(name="qd", length=0.3, multipole=MagneticMultipoleP(Bn={1: -2.0})),
           Drift(name="d2", length=0.25)]
    lat = Lattice.from_sequence("fodo", els, r0)
    izd = tmp_path / "iz"
    izd.mkdir()
    Writer().write(lat, izd / "ImpactZ.in")
    mad = tmp_path / "fodo.madx"
    MadxWriter().write(lat, mad)

    a = get_oracle("impactz").run(izd / "ImpactZ.in", workdir=tmp_path / "izrun")
    b = get_oracle("madx").run(mad, fmt="madx",
                               beam=BeamSpec("proton", 8e8, 352.21e6),
                               workdir=tmp_path / "madrun")
    assert a.s_out == pytest.approx(b.s_out, abs=1e-12)
    bc = b.to_common()
    for i, name in enumerate(a.names):
        assert a.R_elem[i][:4, :4] == pytest.approx(bc.R_elem[i][:4, :4], abs=1e-9), name
    pc = compare_pair(a, b)
    assert pc.n_shared == 5
    assert pc.blocks["T4x4"] < 1e-12
    assert pc.blocks["R56"] < 1e-8
    assert pc.max_rcum_abs < 1e-8


@needs("madx")
def test_fodo_madx_deck_vs_cpymad_end_to_end(tmp_path):
    """``fodo.madx`` (800 MeV proton, FODO + 2 sbends, 6.6 m) translated to IMPACT-Z and
    compared with cpymad on the same IR.  MEASURED 2026-09-03: transverse 4×4 6.0e-14,
    dispersion 3.6e-15, path row 4.8e-15, R56 8.7e-10 over 8 shared boundaries."""
    from lattix.formats.madx import Reader as MadxReader
    from lattix.formats.madx import Writer as MadxWriter
    from lattix.oracles.compare import compare_pair

    lat, _ = MadxReader().read(DATA / "helix" / "fodo.madx")
    lat.reference = lat.reference.model_copy(update={"rf_frequency_Hz": 352.21e6})
    izd = tmp_path / "iz"
    izd.mkdir()
    Writer().write(lat, izd / "ImpactZ.in")
    mad = tmp_path / "fodo.madx"
    MadxWriter().write(lat, mad)

    a = get_oracle("impactz").run(izd / "ImpactZ.in", workdir=tmp_path / "izrun")
    b = get_oracle("madx").run(
        mad, fmt="madx",
        beam=BeamSpec("proton", lat.reference.kinetic_energy_eV, 352.21e6),
        workdir=tmp_path / "madrun")
    pc = compare_pair(a, b)
    assert pc.n_shared == 8
    assert pc.length_a == pytest.approx(6.6) and pc.length_b == pytest.approx(6.6)
    assert pc.blocks["T4x4"] < 1e-11
    assert pc.blocks["disp"] < 1e-11
    assert pc.blocks["path"] < 1e-11
    assert pc.blocks["R56"] < 1e-8
    assert pc.max_rcum_abs < 1e-8
    assert pc.energy_rel == 0.0


# ------------------------------------------------------- reference energy / RF
def test_fodo_deck_runs_and_fort18_holds_the_kinetic_energy(tmp_path):
    """(a) the written FODO deck completes and fort.18 carries the right energy."""
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "helix" / "fodo.madx")
    lat.reference = lat.reference.model_copy(update={"rf_frequency_Hz": 352.21e6})
    d = tmp_path / "iz"
    d.mkdir()
    Writer().write(lat, d / "ImpactZ.in")
    r = get_oracle("impactz").run(d / "ImpactZ.in", workdir=tmp_path / "run")
    f18 = Path(r.meta["workdir"]) / "fort.18"
    assert f18.is_file()
    rows = np.loadtxt(f18)
    ke = lat.reference.kinetic_energy_eV
    assert rows[0, 3] * 1e6 == pytest.approx(ke, rel=1e-12)
    assert rows[-1, 3] * 1e6 == pytest.approx(ke, rel=1e-12)     # no RF in the deck
    assert rows[-1, 0] == pytest.approx(6.6, abs=1e-9)           # z at the end [m]
    assert r.total_length == pytest.approx(6.6)
    assert r.ref_kinetic_eV_out[-1] == pytest.approx(ke, rel=1e-12)


def test_accelerating_linac_reference_energies_match_propagate(tmp_path):
    """(b) IMPACT-Z's ideal-cavity model integrates dγ/dz = (E0/mc²)·cos φs, so its
    reference energies reproduce the IR walk far better than the 1e-4 the task asks."""
    r0 = ref(ke=2.1e6)
    els = []
    for i in range(4):
        els += [Drift(name=f"d{i}", length=0.4),
                RFCavity(name=f"c{i}", length=0.0,
                         rf=RFP(voltage_V=1.5e6, phase_rad=math.radians(-20.0),
                                frequency_Hz=FREQ_HZ)),
                Drift(name=f"e{i}", length=0.4),
                Quadrupole(name=f"q{i}", length=0.2,
                           multipole=MagneticMultipoleP(Bn={1: 3.0 if i % 2 else -3.0}))]
    lat = Lattice.from_sequence("linac", els, r0)
    d = tmp_path / "iz"
    d.mkdir()
    Writer().write(lat, d / "ImpactZ.in")
    r = get_oracle("impactz").run(d / "ImpactZ.in", workdir=tmp_path / "run")

    walked = propagate(lat)
    by_name = {p.element.name: p for p in walked}
    for name, ke in zip(r.names, r.ref_kinetic_eV_out, strict=True):
        want = by_name[name].ref_out.kinetic_energy_eV
        assert ke == pytest.approx(want, rel=1e-9), name
    assert r.ref_kinetic_eV_out[-1] - r.ref_kinetic_eV_in[0] == pytest.approx(
        4 * 1.5e6 * math.cos(math.radians(-20.0)), rel=1e-9)


def test_rfdata_cavity_model_reaches_the_equivalent_tier(tmp_path):
    """``rf_model="rfdata"`` writes a raised-cosine on-axis profile whose amplitude and
    phase are corrected by the complex form factor.  IMPACT-Z re-integrates it with a
    varying β, so the gain lands within the Equivalent tier rather than exactly:
    MEASURED 1.1e-3 for a 2 mm gap, 1.1e-4 for a 0.2 mm gap at 2.1 MeV."""
    r0 = ref()
    els = [Drift(name="d1", length=0.5),
           RFCavity(name="c", length=0.0,
                    rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(PHI_S_DEG),
                           frequency_Hz=FREQ_HZ)),
           Drift(name="d2", length=0.5)]
    r = run(tmp_path, els, reference=r0,
            write_kw={"rf_model": "rfdata", "thin_gap_length_m": 2e-4})
    i = r.names.index("c")
    gain = r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i]
    assert gain == pytest.approx(V_VOLT * math.cos(math.radians(PHI_S_DEG)), rel=5e-4)
    assert (Path(r.meta["workdir"]) / "rfdata1.in").is_file()


def test_h_minus_gain_is_charge_independent(tmp_path):
    """IMPACT-Z's ideal cavity integrates dγ/dz = (E0/mc²)·cos φ with no charge factor,
    which is exactly the IR's species-independent ``dE = V·cos φ_s`` rule."""
    rh = ReferenceParticle(species=species("h-"), kinetic_energy_eV=KE_EV,
                           rf_frequency_Hz=FREQ_HZ)
    els = [Drift(name="d1", length=0.5),
           RFCavity(name="c", length=0.0,
                    rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(PHI_S_DEG),
                           frequency_Hz=FREQ_HZ)),
           Drift(name="d2", length=0.5)]
    r = run(tmp_path, els, reference=rh)
    assert r.charge == -1
    i = r.names.index("c")
    gain = r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i]
    assert gain == pytest.approx(V_VOLT * math.cos(math.radians(PHI_S_DEG)), rel=1e-12)


def test_cavity_longitudinal_and_damping(tmp_path):
    """The cavity's linearisation matches the value docs/oracles.md pins for Bmad's
    lcavity and HELIX, and det(2×2) tracks p_in/p_out (invariant I-5)."""
    r0 = ref()
    els = [Drift(name="d1", length=0.5),
           RFCavity(name="c", length=0.0,
                    rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(PHI_S_DEG),
                           frequency_Hz=FREQ_HZ)),
           Drift(name="d2", length=0.5)]
    r = run(tmp_path, els, reference=r0)
    i = r.names.index("c")
    gain = r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i]
    assert gain == pytest.approx(V_VOLT * math.cos(math.radians(PHI_S_DEG)), rel=1e-12)
    assert r.R_elem[i][5, 4] == pytest.approx(-4.3046, abs=2e-3)   # bunching, cf. Bmad
    p_in = math.sqrt((1 + r.ref_kinetic_eV_in[i] / MASS) ** 2 - 1)
    p_out = math.sqrt((1 + r.ref_kinetic_eV_out[i] / MASS) ** 2 - 1)
    assert np.linalg.det(r.R_elem[i][:2, :2]) == pytest.approx(p_in / p_out, rel=1e-9)


def test_determinant_product_over_the_line(tmp_path):
    r0 = ref()
    els = [Drift(name="d1", length=0.5),
           RFCavity(name="c1", rf=RFP(voltage_V=5e5, phase_rad=0.0, frequency_Hz=FREQ_HZ)),
           Drift(name="d2", length=0.5),
           RFCavity(name="c2", rf=RFP(voltage_V=5e5, phase_rad=0.0, frequency_Hz=FREQ_HZ)),
           Drift(name="d3", length=0.5)]
    r = run(tmp_path, els, reference=r0)
    prod = float(np.prod([np.linalg.det(m[:2, :2]) for m in r.R_elem]))
    p0 = math.sqrt((1 + r.ref_kinetic_eV_in[0] / MASS) ** 2 - 1)
    p1 = math.sqrt((1 + r.ref_kinetic_eV_out[-1] / MASS) ** 2 - 1)
    assert prod == pytest.approx(p0 / p1, rel=1e-8)


# -------------------------------------------------------------- other outputs
def test_maps_false_reports_identity_and_says_so(tmp_path):
    r = run(tmp_path, [Drift(name="d", length=1.0),
                       RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=0.0,
                                                 frequency_Hz=FREQ_HZ))],
            maps=False)
    assert np.allclose(r.R_elem, np.eye(6))
    assert r.basis is Basis.COMMON
    assert r.meta["maps"].startswith("not requested")
    assert any("R_elem is the identity" in w for w in r.warnings)
    assert r.ref_kinetic_eV_out[-1] == pytest.approx(KE_EV + 1e6, rel=1e-9)


def test_no_twiss_dispersion_or_survey(tmp_path):
    r = run(tmp_path, [Drift(name="d", length=1.0)])
    assert r.twiss == {} and r.disp == {} and r.survey is None
    assert "not available" in r.meta["twiss"]


def test_envelopes_are_available_on_request(tmp_path):
    r = run(tmp_path, [Drift(name="d", length=1.0)], envelopes=True)
    env = r.meta["envelopes"]
    assert set("xyz") <= set(env)
    assert env["x"].shape[1] >= 7
    assert "deg" in env["units"]["z"]


def test_probe_tracks_in_the_common_basis(tmp_path):
    p = Probe.default(n=8)
    r = run(tmp_path, [Drift(name="d", length=1.0)], probe=p)
    assert r.probe_out is not None and r.probe_out.shape == (8, 6)
    # IMPACT-Z tracks the exact drift (x += L·px/γβ_z), so a 1e-4 probe departs from
    # the linear map by ~1e-8 at β = 0.067 — that difference is the model, not an error.
    expect = p.coords @ drift_common(1.0, KE_EV, MASS).T
    assert r.probe_out[:, :4] == pytest.approx(expect[:, :4], abs=5e-8)
    assert r.probe_out[:, 4] == pytest.approx(expect[:, 4], abs=5e-8)


def test_element_names_come_from_the_lattix_tags(tmp_path):
    r = run(tmp_path, [Drift(name="alpha", length=0.5), Marker(name="beta"),
                       Drift(name="gamma", length=0.5)])
    assert r.names == ["alpha", "beta", "gamma"]
    assert r.length == pytest.approx([0.5, 0.0, 0.5])


def test_vendored_example_needs_its_field_files(tmp_path):
    """The vendored examples reference ``rfdataN.in`` files that are not redistributed
    with the decks, so the adapter must fail loudly rather than silently truncate."""
    src = DATA / "impactz" / "Example3" / "ImpactZ.in"
    with pytest.raises(RuntimeError, match="ImpactZexe failed"):
        get_oracle("impactz").run(src, workdir=tmp_path / "run")


def test_longer_line_with_every_supported_element(tmp_path):
    """A 25-element line exercising every card the writer emits."""
    from lattix.ir.elements import Collimator, Kicker

    r0 = ref()
    els = []
    for i in range(3):
        els += [Drift(name=f"d{i}a", length=0.3),
                Quadrupole(name=f"qf{i}", length=0.2,
                           multipole=MagneticMultipoleP(Bn={1: 4.0})),
                Drift(name=f"d{i}b", length=0.3),
                Solenoid(name=f"s{i}", length=0.2, solenoid=SolenoidP(Bsol_T=0.2)),
                Kicker(name=f"k{i}", hkick=1e-4),
                Collimator(name=f"c{i}"),
                Bend(name=f"b{i}", length=0.5,
                     bend=BendP(angle=0.02, e1=0.01, e2=0.01, hgap=0.02, edge_int1=0.45)),
                Marker(name=f"m{i}")]
    lat = Lattice.from_sequence("line", els, r0)
    d = tmp_path / "iz"
    d.mkdir()
    Writer().write(lat, d / "ImpactZ.in")
    r = get_oracle("impactz").run(d / "ImpactZ.in", workdir=tmp_path / "run")
    assert len(r.names) == len(els)
    assert r.total_length == pytest.approx(lat.total_length)
    assert np.isfinite(r.R_elem).all()
    kick = r.names.index("k0")
    kr = r.R_elem[kick]
    assert kr[:4, :4] == pytest.approx(np.eye(4), abs=1e-9)
    # ``kick_BPM`` adds ``dpx·γβ_z`` of the *individual* particle (BPM.f90:241), so the
    # kick scales with 1/p just like a real corrector: the map is identity except for
    # a momentum-dependent column.
    assert abs(kr[1, 5]) > 1e-6
