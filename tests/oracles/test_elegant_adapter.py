"""Elegant adapter: per-element matrices vs elegant's own concatenation,
Twiss/floor alignment, reference energy through ``RFCA change_p0=1``, probe
tracking, rpn definitions, and the *measured* longitudinal conventions of
elegant's (x, x', y, y', s, δ) basis (PLAN §5.1 fingerprints).

Both SDDS readers are exercised: ``pysdds`` (skipped in interpreters without
it) and the ``sdds2stream`` text fallback.
"""
from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pytest

from lattix.oracles import get_oracle
from lattix.oracles.base import C_LIGHT, SPECIES, Basis, BeamSpec, Probe
from lattix.oracles.basis import drift_common, transform_matrix
from lattix.oracles.elegant import ME_ELEGANT_EV, ElegantOracle, SddsReader
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, write_decks
from lattix.testing import needs

pytestmark = [pytest.mark.oracle_elegant, needs("elegant")]

MASS = SPECIES["proton"][0]
BEAM = BeamSpec(species="proton", kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ)   # 2.1 MeV proton

FODO = """\
d1: drif, l=0.25
qf: kquad, l=0.1, k1=2.0
d2: drif, l=0.5
qd: kquad, l=0.1, k1=-2.0
fp: line=(d1,qf,d2,qd,d1)
"""
FODO_LENGTHS = [0.25, 0.1, 0.5, 0.1, 0.25]
FODO_QUAD = FODO.replace("kquad", "quad")     # matrix-tracked quads: tracking == linear map for δ = 0


@pytest.fixture(params=["pysdds", "text"])
def oracle(request) -> ElegantOracle:
    if request.param == "pysdds" and importlib.util.find_spec("pysdds") is None:
        pytest.skip("pysdds not importable in this interpreter "
                    "(the sdds2stream text fallback is tested)")
    return ElegantOracle(sdds_backend=request.param)


def _beta_gamma(pc: float) -> tuple[float, float]:
    g = math.sqrt(1.0 + pc * pc)
    return pc / g, g


def _pc(beam: BeamSpec) -> float:
    return beam.beta * beam.gamma


# ---------------------------------------------------------------------------
def test_available_and_registry():
    o = get_oracle("elegant")
    assert isinstance(o, ElegantOracle)
    ok, msg = o.available()
    assert ok, msg
    assert "elegant" in msg and re.search(r"\d{4}\.\d", msg), msg   # "<path> elegant 2026.3.0 (...)"
    auto = "pysdds" if importlib.util.find_spec("pysdds") else "text"
    assert o.reader().backend == auto
    assert SddsReader(o.exe().parent, "text").backend == "text"
    with pytest.raises(ValueError):
        SddsReader(o.exe().parent, "hdf5")


# ---------------------------------------------------------------------------
def test_fodo_matrices_twiss_floor(oracle, tmp_path):
    deck = tmp_path / "fodo.lte"
    deck.write_text(FODO)
    res = oracle.run(deck, beam=BEAM, workdir=tmp_path / "ind", use_beamline="fp", velocity_term=False)

    assert res.basis is Basis.ELEGANT
    assert res.meta["sdds_backend"] == oracle.sdds_backend
    assert [n.lower() for n in res.names] == ["d1", "qf", "d2", "qd", "d1"]   # elegant upper-cases
    np.testing.assert_allclose(res.length, FODO_LENGTHS, atol=1e-12)
    assert res.s_out[-1] == pytest.approx(sum(FODO_LENGTHS), abs=1e-12)
    assert res.mass_eV == MASS and res.charge == +1
    np.testing.assert_allclose(res.ref_kinetic_eV_in, KE_EV, rtol=1e-9)
    np.testing.assert_allclose(res.ref_kinetic_eV_out, KE_EV, rtol=1e-9)

    # per-element maps are the element's own (drift: R12 = R34 = L, R56 = 0 in elegant's path-length s)
    for i in (0, 2, 4):
        D = np.eye(6)
        D[0, 1] = D[2, 3] = FODO_LENGTHS[i]
        np.testing.assert_allclose(res.R_elem[i], D, atol=1e-14)
    k, L = 2.0, 0.1
    w = math.sqrt(k) * L
    assert res.R_elem[1][0, 0] == pytest.approx(math.cos(w), abs=1e-12)          # thick quad, focusing plane
    assert res.R_elem[1][1, 0] == pytest.approx(-math.sqrt(k) * math.sin(w), abs=1e-12)
    assert res.R_elem[1][2, 2] == pytest.approx(math.cosh(w), abs=1e-12)
    for R in res.R_elem:
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    # product of the individual maps == elegant's own start->exit concatenation (individual_matrices=0)
    cum = oracle.run(deck, beam=BEAM, workdir=tmp_path / "cum", use_beamline="fp", velocity_term=False,
                     matrix_mode="cumulative")
    assert cum.names == res.names and cum.meta["matrix_mode"] == "cumulative"
    np.testing.assert_allclose(res.R_cum, cum.R_elem, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(cum.R_elem[0], res.R_elem[0], atol=1e-14)

    # floor coordinates: straight line along Z
    assert res.survey.shape == (5, 4)
    np.testing.assert_allclose(res.survey[-1], [0.0, 0.0, sum(FODO_LENGTHS), 0.0], atol=1e-12)
    np.testing.assert_allclose(res.survey[:, 2], np.cumsum(FODO_LENGTHS), atol=1e-12)

    # twiss_output rows line up with the matrix rows: propagate the BeamSpec optics through R_cum
    b0 = BEAM.betx
    for i, Rc in enumerate(res.R_cum):
        for (a, b), bet, alf in (((0, 1), "betx", "alfx"), ((2, 3), "bety", "alfy")):
            beta_i = Rc[a, a] ** 2 * b0 + Rc[a, b] ** 2 / b0
            alpha_i = -(Rc[a, a] * Rc[b, a] * b0 + Rc[a, b] * Rc[b, b] / b0)
            assert res.twiss[bet][i] == pytest.approx(beta_i, rel=1e-9)
            assert res.twiss[alf][i] == pytest.approx(alpha_i, abs=1e-9)
    for key in ("dx", "dpx", "dy", "dpy"):
        np.testing.assert_allclose(res.disp[key], 0.0, atol=1e-14)
    assert res.twiss["betx"][-1] != pytest.approx(b0)   # optics actually moved
    assert Path(res.meta["command_file"]).read_text().count("&run_setup") == 1


# ---------------------------------------------------------------------------
def test_drift_fingerprint(oracle, tmp_path, record_property):
    """1 m drift from ``fingerprint.DECKS['elegant']``: transverse map exact;
    elegant's s is the *path length*, so R56 = 0 at any energy — the common-basis
    map therefore lacks the analytic velocity term L/γ² (reported, not patched)."""
    path, fmt = write_decks("elegant", tmp_path)["drift"]
    raw = oracle.run(path, fmt=fmt, beam=BEAM, workdir=tmp_path / "raw", velocity_term=False)
    res = oracle.run(path, fmt=fmt, beam=BEAM, workdir=tmp_path / "run")
    assert res.n == 1 and res.length[0] == pytest.approx(1.0, abs=1e-12)
    Rn = raw.R_elem[0]                   # elegant's own path-length map
    Rc = res.to_common().R_elem[0]       # adapter default: arrival-time (common) basis
    exp = drift_common(1.0, KE_EV, MASS)

    np.testing.assert_allclose(Rn[:4, :4], exp[:4, :4], atol=1e-14)
    assert Rn[4, 4] == 1.0 and Rn[5, 5] == 1.0
    assert np.count_nonzero(Rn - np.diag(np.diag(Rn))) == 2        # only R12 and R34
    R56_native, R56_common, R56_expected = float(Rn[4, 5]), float(Rc[4, 5]), float(exp[4, 5])
    record_property("R56_native", R56_native)
    record_property("R56_common", R56_common)
    record_property("R56_expected", R56_expected)
    print(f"\nelegant drift fingerprint: R56_native={R56_native!r} R56_common={R56_common!r} "
          f"expected(common)={R56_expected!r} (= L/γ²)")
    # measured 2026-09-03 with elegant 2026.3.0: path-length s -> no R56 in a drift (raw),
    # and the adapter's default velocity term restores L/γ² in the common basis
    assert R56_native == 0.0
    assert R56_expected == pytest.approx(1.0 / BEAM.gamma ** 2, rel=1e-12)
    assert R56_common == pytest.approx(R56_expected, rel=1e-12)
    assert raw.to_common().R_elem[0][4, 5] == 0.0


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("species", ["proton", "h-", "electron"])
def test_cavity_fingerprint(oracle, tmp_path, record_property, species):
    """Thin ``RFCA change_p0=1`` from ``fingerprint.DECKS['elegant']``: the
    reference energy follows p0, and the gain sign follows the particle's charge
    relative to the electron (elegant: +V·sin(PHASE) for negative particles,
    −V·sin(PHASE) for positive ones).  R65 of elegant's matrix uses a phase slip
    ω·Δs/c (no 1/β), R66 = β_in p_in / (β_out p_out)."""
    beam = BeamSpec(species=species, kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ)
    path, fmt = write_decks("elegant", tmp_path)["cavity"]
    text = path.read_text()
    phase = math.radians(float(re.search(r"phase=([-+0-9.eE]+)", text).group(1)))
    volt = float(re.search(r"volt=([-+0-9.eE]+)", text).group(1))
    res = oracle.run(path, fmt=fmt, beam=beam, workdir=tmp_path / "run")

    assert [n.lower() for n in res.names] == ["d1", "c", "d2"]
    assert res.mass_eV == beam.mass_eV and res.charge == beam.charge
    j = 1
    gain = float(res.ref_kinetic_eV_out[j] - res.ref_kinetic_eV_in[j])
    expected_gain = -beam.charge * volt * math.sin(phase)
    assert res.ref_kinetic_eV_in[0] == pytest.approx(KE_EV, rel=1e-9)
    assert res.ref_kinetic_eV_out[0] == pytest.approx(KE_EV, rel=1e-9)
    assert res.ref_kinetic_eV_in[j] == pytest.approx(res.ref_kinetic_eV_out[0], rel=1e-12)
    assert res.ref_kinetic_eV_in[2] == pytest.approx(res.ref_kinetic_eV_out[j], rel=1e-12)
    assert res.ref_kinetic_eV_out[2] == pytest.approx(res.ref_kinetic_eV_out[j], rel=1e-12)
    assert gain == pytest.approx(expected_gain, rel=1e-7), (gain, expected_gain)

    R = res.R_elem[j]
    b_in, g_in = _beta_gamma(_pc(beam))
    pc_out = math.sqrt((1.0 + res.ref_kinetic_eV_out[j] / beam.mass_eV) ** 2 - 1.0)
    b_out, g_out = _beta_gamma(pc_out)
    omega = 2 * math.pi * FREQ_HZ
    R65_expected = (-beam.charge * volt * math.cos(phase) * (omega / C_LIGHT)
                    / (b_out ** 2 * g_out * beam.mass_eV))
    R66_expected = (b_in * _pc(beam)) / (b_out * pc_out)
    assert R[5, 4] == pytest.approx(R65_expected, rel=1e-6)
    assert R[5, 5] == pytest.approx(R66_expected, rel=1e-9)
    assert R[1, 1] == pytest.approx(_pc(beam) / pc_out, rel=1e-9)   # damping: x' -> x'·p_in/p_out
    assert R[3, 3] == pytest.approx(_pc(beam) / pc_out, rel=1e-9)
    Rc = res.to_common().R_elem[j]
    assert Rc[5, 4] == pytest.approx(-R[5, 4], rel=1e-12)                 # basis z-sign -1 flips the row
    record_property("gain_eV", gain)
    record_property("R65_native", float(R[5, 4]))
    record_property("R66_native", float(R[5, 5]))
    print(f"\nelegant cavity fingerprint ({species}, phase={math.degrees(phase):g}°): "
          f"gain={gain:.6f} eV (expected {expected_gain:.6f}), R65_native={R[5, 4]!r}, "
          f"R66_native={R[5, 5]!r}, R65_common={Rc[5, 4]!r}; "
          f"time-based R65 would be R65_native/β_in={R[5, 4] / b_in!r}")
    cp = res.meta["change_particle"]
    assert 'name = "custom"' in cp
    assert f"mass_ratio = {beam.mass_eV / ME_ELEGANT_EV:.15g}" in cp
    assert f"charge_ratio = {-beam.charge:d}" in cp


# ---------------------------------------------------------------------------
def test_probe_tracking_fodo(oracle, tmp_path):
    """Tracked probe vs the linear maps: transverse coordinates follow R_cum
    (matrix-tracked QUADs: exact for on-momentum particles; off-momentum ones
    show the physical chromatic second-order terms ~ kL·x·δ); the arrival-time
    based longitudinal coordinate carries the L/γ² velocity term that elegant's
    matrices omit; δ is untouched without RF.  (KQUAD tracking with the default
    kick count differs from its matrix by ~1e-5 relative — integrator, not
    plumbing — so the probe deck uses QUAD.)"""
    deck = tmp_path / "fodo_quad.lte"
    deck.write_text(FODO_QUAD)
    probe = Probe.default()
    half = probe.coords.shape[0] // 2
    probe.coords[:half, 5] = 0.0                                           # first half on-momentum
    res = oracle.run(deck, beam=BEAM, probe=probe, workdir=tmp_path / "run", use_beamline="fp")

    assert res.probe_out is not None and res.probe_out.shape == probe.coords.shape
    assert not np.isnan(res.probe_out).any()
    T = transform_matrix(Basis.ELEGANT, KE_EV, MASS)
    native_in = probe.coords @ np.linalg.inv(T).T
    lin = native_in @ res.R_cum[-1].T
    np.testing.assert_allclose(res.probe_out[:half, :4], lin[:half, :4], atol=1e-12)
    np.testing.assert_allclose(res.probe_out[:, :4], lin[:, :4], atol=5e-8)
    np.testing.assert_allclose(res.probe_out[:, 5], native_in[:, 5], atol=1e-12)

    com = res.to_common()
    z_in, d_in = probe.coords[:, 4], probe.coords[:, 5]
    z_exp = z_in + (sum(FODO_LENGTHS) / BEAM.gamma ** 2) * d_in          # time of flight, ahead-positive
    np.testing.assert_allclose(com.probe_out[:, 4], z_exp, atol=1e-7)
    assert np.max(np.abs(z_exp - z_in)) > 1e-5                             # the term is far above tolerance
    raw = res.meta["probe_raw"]
    assert raw["p_ref"] == pytest.approx(_pc(BEAM), rel=1e-12)
    assert raw["pCentral_end"] == pytest.approx(_pc(BEAM), rel=1e-12)
    assert raw["t_ref"] == pytest.approx(sum(FODO_LENGTHS) / (BEAM.beta * C_LIGHT), rel=1e-9)
    assert (Path(res.meta["workdir"]) / "probe.sdds").is_file()


def test_probe_tracking_cavity(oracle, tmp_path):
    """Through the p0-following cavity the tracked δ is measured against the
    *new* reference momentum and t_ref includes the post-cavity velocity."""
    path, fmt = write_decks("elegant", tmp_path)["cavity"]
    probe = Probe(coords=np.zeros((3, 6)))
    probe.coords[1, 5] = 1e-6                                              # δ only
    probe.coords[2, 0] = 1e-4                                              # x only
    res = oracle.run(path, fmt=fmt, beam=BEAM, probe=probe, workdir=tmp_path / "run")
    out = res.probe_out
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[0], 0.0, atol=1e-15)                    # reference-like particle stays put
    # δ_out = R66 δ_in + (time-based R65) · s_cav, where the drift ahead of the
    # cavity shifts the arrival by s_cav = -(L/γ²) δ_in (late-positive) — a
    # coupling elegant's matrices lack (drift R56 = 0) and whose RFCA R65
    # carries an extra factor β_in (phase slip ω·Δs/c instead of ω·Δs/(βc)).
    Rc = res.R_elem[1]
    s_cav = -(0.5 / BEAM.gamma ** 2) * 1e-6
    d_exp = Rc[5, 5] * 1e-6 + (Rc[5, 4] / BEAM.beta) * s_cav
    assert out[1, 5] == pytest.approx(d_exp, rel=1e-4)
    assert out[1, 5] != pytest.approx(Rc[5, 5] * 1e-6, rel=0.1)            # the coupling is not small here
    assert out[2, 0] == pytest.approx(1e-4, rel=1e-9) and abs(out[2, 5]) < 1e-12
    raw = res.meta["probe_raw"]
    pc_end = math.sqrt((1.0 + res.ref_kinetic_eV_out[-1] / MASS) ** 2 - 1.0)
    assert raw["p_ref"] == pytest.approx(pc_end, rel=1e-12)
    assert raw["pCentral_end"] == pytest.approx(pc_end, rel=1e-12)
    b0, b1 = BEAM.beta, _beta_gamma(pc_end)[0]
    assert raw["t_ref"] == pytest.approx(0.5 / (b0 * C_LIGHT) + 0.5 / (b1 * C_LIGHT), rel=1e-9)


# ---------------------------------------------------------------------------
def test_rpn_definitions(tmp_path, monkeypatch):
    """The conda package ships no defns.rpn: the adapter writes one so decks
    using ``pi``, ``mev``, ``chs``/``abs``, degree trig and ``max2`` evaluate."""
    monkeypatch.delenv("RPN_DEFNS", raising=False)
    deck = tmp_path / "rpn.lte"
    deck.write_text('d1: drif, l="pi 2 /"\nd2: drif, l="1 chs abs"\nd3: drif, l="30 dsin"\n'
                    'd4: drif, l="mev 2 *"\nd5: drif, l="1 3 max2"\nd6: drif, l="c_mks 1e8 /"\n'
                    "fp: line=(d1,d2,d3,d4,d5,d6)\n")
    res = ElegantOracle().run(deck, beam=BEAM, workdir=tmp_path / "run")
    np.testing.assert_allclose(res.length, [math.pi / 2, 1.0, 0.5, 2 * 0.51099906, 3.0, 2.99792458],
                               rtol=1e-12)
    assert res.meta["rpn_defns"] == str(tmp_path / "run" / "defns.rpn")

    own = tmp_path / "mine.rpn"
    own.write_text("1 atan 4 * sto pi pop\n")
    monkeypatch.setenv("RPN_DEFNS", str(own))
    deck2 = tmp_path / "pi.lte"
    deck2.write_text('d: drif, l="pi"\nfp: line=(d)\n')
    res2 = ElegantOracle().run(deck2, beam=BEAM, workdir=tmp_path / "run2")
    assert res2.meta["rpn_defns"] == str(own.resolve())
    assert res2.length[0] == pytest.approx(math.pi, rel=1e-12)


def test_errors_and_defaults(tmp_path):
    deck = tmp_path / "bad.lte"
    deck.write_text("d: drif, l=1.0\nq: nosuchelement, l=0.1\nfp: line=(d,q)\n")
    with pytest.raises(RuntimeError, match="elegant failed"):
        ElegantOracle().run(deck, beam=BEAM, workdir=tmp_path / "run")
    with pytest.raises(ValueError, match="lte"):
        ElegantOracle().run(deck, fmt="madx", beam=BEAM)
    with pytest.raises(ValueError, match="matrix_mode"):
        ElegantOracle().run(deck, beam=BEAM, matrix_mode="both")
    with pytest.raises(FileNotFoundError):
        ElegantOracle().run(tmp_path / "missing.lte", beam=BEAM)

    good = tmp_path / "good.lte"
    good.write_text("d: drif, l=1.0\nfp: line=(d)\n")
    res = ElegantOracle().run(good, workdir=tmp_path / "default")          # no beam -> BeamSpec() + warning
    assert any("BeamSpec" in w for w in res.warnings)
    assert res.ref_kinetic_eV_in[0] == pytest.approx(BeamSpec().kinetic_energy_eV, rel=1e-9)
    assert any("No USE statement" in w for w in res.warnings)              # elegant's own warning is kept
