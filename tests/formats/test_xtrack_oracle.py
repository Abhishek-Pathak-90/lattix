"""Engine-backed checks for the xtrack adapter (PLAN §6 task 3.5).

Nothing here is round-trip-only: every assertion runs real xtrack on the line lattix
built and compares it against an independent source of truth —

(a) ``fodo.madx`` → IR → :func:`to_line` → per-element maps vs **cpymad** on the source
    deck (``compare_pair``, Exact tier);
(b) the **kicker sign**: an IR ``Kicker(hkick=+1e-3)`` must deflect ``px`` by ``+1e-3``;
(c) the **cavity phase sign**: φs = −30° must *bunch* (R65 < 0 in the common basis) and
    gain ``+V·cos 30°`` of reference energy in xtrack's own tracking;
(d) ``psb.seq`` (the PS Booster ring, 526 IR elements over 157 m) read through
    ``lattix.formats.madx`` → :func:`to_line` vs ``xt.Line.from_madx_sequence`` on the
    same deck — the biggest MAD-X → xtrack cross-check available here.

Environment note (measured 2026-09-03): env ``lattix`` ships xtrack 0.112.0 **without**
``xpart`` and without ``xsuite``, so ``Line.build_particles`` and prebuilt kernels are
unavailable there; the guards below skip rather than fail, and the base env
(xtrack 0.103.5 + xpart + cpymad) runs everything.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")

from lattix.formats.madx import Reader as MadxReader  # noqa: E402
from lattix.formats.xtrack import Writer, to_line  # noqa: E402
from lattix.ir.elements import (  # noqa: E402
    RFP,
    Drift,
    Kicker,
    MagneticMultipoleP,
    Quadrupole,
    RFCavity,
)
from lattix.ir.lattice import Lattice  # noqa: E402
from lattix.ir.reference import ReferenceParticle, species  # noqa: E402
from lattix.oracles import get_oracle  # noqa: E402
from lattix.oracles.base import Basis, BeamSpec  # noqa: E402
from lattix.oracles.basis import transform_matrix  # noqa: E402
from lattix.oracles.compare import compare_pair  # noqa: E402
from lattix.testing import needs, require  # noqa: E402

require("xtrack")
pytestmark = pytest.mark.oracle_xtrack

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
M_P = 938_272_088.16
TOL = 1e-8


def _reason(e: Exception, limit: int = 160) -> str:
    text = f"{type(e).__name__}: {e}".replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + " …"


def _try_track() -> str:
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
    line.particle_ref = xt.Particles(mass0=M_P, q0=1, kinetic_energy0=8e8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        line.build_tracker()
    line.track(xt.Particles(mass0=M_P, q0=1, kinetic_energy0=8e8, x=[0.0]))
    return ""


def _why_no_tracking() -> str:
    """Empty when this xtrack build can actually track a one-element line.

    xtrack 0.112 refuses to build a kernel unless the ``xsuite`` package (prebuilt
    kernels) is importable *or* just-in-time compilation is explicitly allowed; env
    ``lattix`` has neither by default, so this turns JIT on — exactly what xtrack's own
    error message asks for — before deciding to skip.
    """
    try:
        return _try_track()
    except Exception as first:                                # noqa: BLE001
        try:
            import xobjects as xo

            xo.settings.allow_kernel_compilation = True
            return _try_track()
        except Exception:                                     # noqa: BLE001
            return _reason(first)


def _why_no_oracle() -> str:
    """The xtrack *oracle* additionally needs ``Line.build_particles`` (xpart)."""
    why = _why_no_tracking()
    if why:
        return why
    try:
        line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
        line.particle_ref = xt.Particles(mass0=M_P, q0=1, kinetic_energy0=8e8)
        line.build_tracker()
        line.build_particles(x=[0.0])
    except Exception as e:                                    # noqa: BLE001
        return f"Line.build_particles unavailable ({_reason(e, 80)})"
    return ""


_NO_TRACKING = _why_no_tracking()
_NO_ORACLE = _why_no_oracle()
needs_tracking = pytest.mark.skipif(bool(_NO_TRACKING),
                                    reason=f"this xtrack build cannot track: {_NO_TRACKING}")
needs_oracle = pytest.mark.skipif(bool(_NO_ORACLE),
                                  reason=f"the xtrack oracle cannot run here: {_NO_ORACLE}")


def proton_ref(ke: float = 8e8, freq: float | None = None) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke,
                             rf_frequency_Hz=freq)


def track(line, **coords):
    """Track ``xt.Particles`` built directly (``build_particles`` needs xpart)."""
    pref = line.particle_ref
    mass = float(np.atleast_1d(pref.mass0)[0])
    q0 = int(round(float(np.atleast_1d(pref.q0)[0])))
    ke = float(np.atleast_1d(pref.kinetic_energy0)[0])
    p = xt.Particles(mass0=mass, q0=q0, kinetic_energy0=ke, **coords)
    line.track(p)
    p.sort(interleave_lost_particles=True)
    return p


def build(line):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        line.build_tracker()
    return line


_COORDS = ("x", "px", "y", "py", "zeta", "delta")


def fd_map(line, h: float = 1e-6) -> np.ndarray:
    """Native-basis 6x6 by central differences (the xtrack oracle's method, with
    ``xt.Particles`` instead of ``build_particles``)."""
    seed = {c: np.zeros(12) for c in _COORDS}
    for k, c in enumerate(_COORDS):
        seed[c][2 * k] = h
        seed[c][2 * k + 1] = -h
    pref = line.particle_ref
    mass = float(np.atleast_1d(pref.mass0)[0])
    q0 = int(round(float(np.atleast_1d(pref.q0)[0])))
    ke = float(np.atleast_1d(pref.kinetic_energy0)[0])
    p = xt.Particles(mass0=mass, q0=q0, kinetic_energy0=ke,
                     **{c: seed[c].copy() for c in _COORDS})
    ins = np.stack([np.asarray(getattr(p, c), dtype=float) for c in _COORDS], axis=1)
    line.track(p)
    p.sort(interleave_lost_particles=True)
    outs = np.stack([np.asarray(getattr(p, c), dtype=float) for c in _COORDS], axis=1)
    R = np.empty((6, 6))
    for k in range(6):
        R[:, k] = (outs[2 * k] - outs[2 * k + 1]) / (ins[2 * k, k] - ins[2 * k + 1, k])
    return R


# ---------------------------------------------------------------- (a) fodo
@needs_oracle
@needs("madx")
@pytest.mark.oracle_madx
@pytest.mark.parametrize("deck", ["fodo.madx", "transport.madx"])
def test_written_line_matches_cpymad_on_the_source_deck(deck, tmp_path):
    src = DATA / "helix" / deck
    lat, rep_in = MadxReader().read(src)
    out = tmp_path / "line.json"
    rep_out = Writer().write(lat, out, strict=True)
    assert rep_in.ok and rep_out.ok

    a = get_oracle("madx").run(src, workdir=tmp_path / "a")
    b = get_oracle("xtrack").run(out, workdir=tmp_path / "b")
    cmp = compare_pair(a, b)

    assert cmp.n_shared >= 4
    assert cmp.length_a == pytest.approx(cmp.length_b, abs=1e-12)
    assert cmp.length_b == pytest.approx(lat.total_length, abs=1e-12)
    assert cmp.blocks["T4x4"] < TOL
    assert cmp.blocks["disp"] < TOL
    assert cmp.blocks["path"] < TOL
    assert cmp.notes == []


@needs_oracle
@needs("madx")
@pytest.mark.oracle_madx
def test_written_line_matches_xtracks_own_madx_import(tmp_path):
    """Same deck through both loaders: lattix's IR path and xtrack's ``from_madx_sequence``."""
    src = DATA / "helix" / "fodo.madx"
    lat, _ = MadxReader().read(src)
    out = tmp_path / "line.json"
    Writer().write(lat, out, strict=True)
    o = get_oracle("xtrack")
    a = o.run(src, fmt="madx", workdir=tmp_path / "a")
    b = o.run(out, workdir=tmp_path / "b")
    cmp = compare_pair(a, b)
    assert cmp.n_shared >= 4
    assert cmp.max_rcum_abs < 1e-12
    assert cmp.notes == []


# -------------------------------------------------------------- (b) kicker
@needs_tracking
def test_kicker_sign_is_positive_in_xtrack():
    """I-10: an IR ``Kicker(hkick=+1e-3)`` deflects the reference by ``px = +1e-3``."""
    ref = proton_ref()
    lat = Lattice.from_sequence("k", [Kicker(name="cor", hkick=1e-3, vkick=-2e-3)], ref)
    line = build(to_line(lat))
    p = track(line, x=[0.0])
    assert float(p.px[0]) == pytest.approx(1e-3, rel=1e-12)
    assert float(p.py[0]) == pytest.approx(-2e-3, rel=1e-12)


@needs_tracking
def test_kicker_magnitude_matches_the_integrated_field():
    """A kick from ∫B·dl through the signed rigidity, for a proton and for H-."""
    from lattix.ir.normalize import kick_from_bl

    for sp in ("proton", "h-"):
        ref = ReferenceParticle(species=species(sp), kinetic_energy_eV=8e8)
        kick = kick_from_bl(0.01, ref)                 # 0.01 T.m
        lat = Lattice.from_sequence("k", [Kicker(name="cor", hkick=kick)], ref)
        p = track(build(to_line(lat)), x=[0.0])
        assert float(p.px[0]) == pytest.approx(kick, rel=1e-12)
    assert kick_from_bl(0.01, ReferenceParticle(species=species("h-"),
                                                kinetic_energy_eV=8e8)) < 0


@needs_tracking
def test_thick_kicker_kicks_at_the_centre_like_madx():
    ref = proton_ref()
    lat = Lattice.from_sequence("k", [Kicker(name="cor", length=0.5, hkick=1e-3)], ref)
    p = track(build(to_line(lat)), x=[0.0])
    assert float(p.px[0]) == pytest.approx(1e-3, rel=1e-12)
    assert float(p.x[0]) == pytest.approx(0.5 / 2 * 1e-3, rel=1e-9)


# -------------------------------------------------------------- (c) cavity
@needs_tracking
@pytest.mark.parametrize("phi_deg", [-30.0, 0.0, +30.0, -90.0])
def test_cavity_reference_gain_is_v_cos_phi(phi_deg):
    """I-6: the reference gain through the written cavity is exactly ``V·cos φs``."""
    ke, V = 2.1e6, 1e6
    ref = proton_ref(ke, freq=162.5e6)
    lat = Lattice.from_sequence(
        "c", [RFCavity(name="cav", rf=RFP(voltage_V=V, phase_rad=math.radians(phi_deg),
                                          frequency_Hz=162.5e6))], ref)
    line = build(to_line(lat))
    p = track(line, x=[0.0], zeta=[0.0], delta=[0.0])
    p0c = float(np.atleast_1d(line.particle_ref.p0c)[0])
    E0 = float(np.atleast_1d(line.particle_ref.energy0)[0])
    dE = math.hypot(p0c * (1.0 + float(p.delta[0])), M_P) - E0
    assert dE == pytest.approx(V * math.cos(math.radians(phi_deg)), rel=1e-10, abs=1e-3)


@needs_tracking
def test_cavity_at_minus_30_degrees_bunches():
    """I-7: at φs = −30° a late particle gains more, so R65 < 0 in the common basis
    (z ahead-positive) and the reference gains +V·cos 30°."""
    ke, V = 2.1e6, 1e6
    ref = proton_ref(ke, freq=162.5e6)
    lat = Lattice.from_sequence(
        "c", [RFCavity(name="cav", rf=RFP(voltage_V=V, phase_rad=-math.pi / 6,
                                          frequency_Hz=162.5e6))], ref)
    line = build(to_line(lat))
    R_native = fd_map(line, h=1e-6)
    T = transform_matrix(Basis.XTRACK, ke, M_P)
    R = T @ R_native @ np.linalg.inv(T)
    assert R[5, 4] < 0.0
    # the Phase-0 fingerprint for this exact element (docs/oracles.md): xtrack -5.117
    assert R[5, 4] == pytest.approx(-5.117, abs=5e-3)

    p = track(line, x=[0.0], zeta=[0.0], delta=[0.0])
    p0c = float(np.atleast_1d(line.particle_ref.p0c)[0])
    E0 = float(np.atleast_1d(line.particle_ref.energy0)[0])
    dE = math.hypot(p0c * (1.0 + float(p.delta[0])), M_P) - E0
    assert dE == pytest.approx(V * math.cos(math.pi / 6), rel=1e-10)
    assert dE == pytest.approx(866_025.4038, abs=1e-3)

    # a particle arriving late (zeta < 0) gains more than the reference
    late = track(line, zeta=[-1e-3], delta=[0.0])
    early = track(line, zeta=[+1e-3], delta=[0.0])
    assert float(late.delta[0]) > float(early.delta[0])


@needs_tracking
def test_p0c_is_constant_across_the_cavity():
    """xtrack keeps p0c fixed through RF — the reason for EQUIVALENT:CONST_P0."""
    ref = proton_ref(2.1e6, freq=162.5e6)
    lat = Lattice.from_sequence(
        "c", [RFCavity(name="cav", rf=RFP(voltage_V=1e6, phase_rad=0.0,
                                          frequency_Hz=162.5e6))], ref)
    line = build(to_line(lat))
    before = float(np.atleast_1d(line.particle_ref.p0c)[0])
    p = track(line, x=[0.0])
    assert float(np.atleast_1d(p.p0c)[0]) == pytest.approx(before)
    assert float(p.delta[0]) > 0.0


@needs_tracking
def test_reference_change_moves_p0c():
    """A ReferenceChange becomes ``xt.ReferenceEnergyIncrease``, which *does* move p0c."""
    ref = proton_ref(1e8)
    from lattix.ir.elements import ReferenceChange

    lat = Lattice.from_sequence("r", [ReferenceChange(name="rc", energy_eV=1.2e8)], ref)
    line = build(to_line(lat))
    p = track(line, x=[0.0])
    expect = ref.advanced(dE_eV=2e7).pc_eV
    assert float(np.atleast_1d(p.p0c)[0]) == pytest.approx(expect, rel=1e-12)


# ------------------------------------------------------- (d) psb.seq ring
@needs_oracle
@needs("madx")
@pytest.mark.oracle_madx
@pytest.mark.slow
def test_psb_ring_matches_xtracks_own_madx_import(tmp_path):
    """The PS Booster ring (157 m, 526 IR elements) through lattix's MAD-X reader +
    ``to_line`` against ``cpymad`` + ``xt.Line.from_madx_sequence`` on the same deck."""
    src = DATA / "xtrack" / "psb.seq"
    seq = "psb1"
    # psb.seq defines no BEAM, so drive it from a wrapper deck (both sides see the same one)
    wrap = tmp_path / "psb_wrap.madx"
    wrap.write_text(f'beam, particle=proton, energy={(M_P + 160e6) / 1e9:.15g};\n'
                    f'call, file="{src}";\nuse, sequence={seq};\n')

    lat, rep_in = MadxReader().read(wrap, sequence=seq)
    assert rep_in.ok
    assert len(lat.flatten()) > 500
    out = tmp_path / "psb1.json"
    rep = Writer().write(lat, out)
    assert rep.counts.get("DROPPED", 0) == 0

    o = get_oracle("xtrack")
    a = o.run(wrap, fmt="madx", sequence=seq, workdir=tmp_path / "a")
    b = o.run(out, workdir=tmp_path / "b")
    assert a.s_out[-1] == pytest.approx(b.s_out[-1], abs=1e-12)

    cmp = compare_pair(a, b)
    assert cmp.n_shared > 250
    assert cmp.max_rcum_abs < 1e-10
    assert cmp.blocks["T4x4"] < 1e-10
    assert cmp.blocks["disp"] < 1e-10
    assert cmp.blocks["path"] < 1e-10
    assert cmp.notes == []

    # and the comparison is not vacuous: the ring's one-turn map is far from the identity
    R = np.eye(6)
    for M in a.R_elem:
        R = M @ R
    assert np.abs(R - np.eye(6)).max() > 1.0


# ----------------------------------------------------------- dual regime
def _linac() -> Lattice:
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=1e8,
                            rf_frequency_Hz=325e6)
    b = ref.brho_signed
    els = [
        Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * b})),
        Drift(name="d1", length=0.3),
        RFCavity(name="c1", rf=RFP(voltage_V=5e7, phase_rad=0.0, frequency_Hz=325e6)),
        Drift(name="d2", length=0.3),
        Quadrupole(name="q2", length=0.2, multipole=MagneticMultipoleP(Bn={1: -5.0 * b})),
    ]
    return Lattice.from_sequence("linac", els, ref)


@needs_tracking
@pytest.mark.parametrize("mode", ["local", "constant"])
def test_energy_mode_keeps_the_downstream_gradient_or_the_optics(mode):
    """``local`` preserves the lab gradient G = k1·Bρ(ref_in) (I-3); ``constant``
    preserves the normalized k1 as MAD-X's own constant-p0 twiss composes it."""
    from lattix.ir.walk import propagate

    lat = _linac()
    line = to_line(lat, energy_mode=mode)
    placed = propagate(lat)
    q2 = placed[4]
    k1 = float(line.element_dict["q2"].k1)
    if mode == "local":
        assert k1 * q2.ref_in.brho_signed == pytest.approx(q2.element.multipole.Bn[1],
                                                           rel=1e-12)
    else:
        assert k1 * lat.reference.brho_signed == pytest.approx(q2.element.multipole.Bn[1],
                                                               rel=1e-12)
    build(line)
    p = track(line, x=[1e-4])
    assert np.isfinite(float(p.px[0]))


@needs_oracle
@needs("madx")
@pytest.mark.oracle_madx
def test_hminus_deck_flips_the_focusing_in_both_engines(tmp_path):
    """I-8: the signed-rigidity policy must give the same sign in MAD-X and in xtrack."""
    src = tmp_path / "hminus.madx"
    m_hm = species("h-").mass_eV
    src.write_text(
        f"beam, particle=ion, mass={m_hm / 1e9:.15g}, charge=-1, "
        f"energy={(m_hm + 8e8) / 1e9:.15g};\n"
        "qf: quadrupole, l=0.3, k1=0.6;\n"
        "s1: sequence, l=1.0; qf, at=0.15; endsequence;\n"
        "use, sequence=s1;\n")
    lat, _ = MadxReader().read(src)
    assert lat.reference.species.charge == -1
    assert lat.elements["qf"].multipole.Bn[1] == pytest.approx(
        0.6 * lat.reference.brho_signed)
    out = tmp_path / "hminus.json"
    Writer().write(lat, out, strict=True)
    assert float(to_line(lat).element_dict["qf"].k1) == pytest.approx(0.6, rel=1e-12)

    a = get_oracle("madx").run(src, workdir=tmp_path / "a")
    b = get_oracle("xtrack").run(out, workdir=tmp_path / "b")
    cmp = compare_pair(a, b)
    assert cmp.blocks["T4x4"] < TOL


@needs_oracle
def test_oracle_reads_the_written_json_and_finds_the_particle_ref(tmp_path):
    lat = Lattice.from_sequence(
        "d", [Drift(name="d1", length=1.0)],
        ReferenceParticle(species=species("h-"), kinetic_energy_eV=2.1e6))
    out = tmp_path / "d.json"
    Writer().write(lat, out)
    res = get_oracle("xtrack").run(out, workdir=tmp_path / "w")
    assert res.charge == -1
    assert res.mass_eV == pytest.approx(species("h-").mass_eV)
    assert res.ref_kinetic_eV_in[0] == pytest.approx(2.1e6)
    # a 1 m drift: R56 = L/gamma^2 in the common basis
    T = transform_matrix(Basis.XTRACK, 2.1e6, species("h-").mass_eV)
    R = T @ res.R_elem[0] @ np.linalg.inv(T)
    gamma = 1.0 + 2.1e6 / species("h-").mass_eV
    assert R[4, 5] == pytest.approx(1.0 / gamma ** 2, rel=1e-8)


@needs_oracle
def test_beam_spec_is_not_needed_because_lattix_writes_particle_ref(tmp_path):
    """``from_madx_sequence``/``xt.load`` leave ``particle_ref`` unset (docs/oracles.md);
    a deck lattix writes always carries it."""
    lat = Lattice.from_sequence("d", [Drift(name="d1", length=1.0)], proton_ref())
    out = tmp_path / "d.json"
    Writer().write(lat, out)
    line = xt.Line.from_json(str(out))
    assert line.particle_ref is not None
    _ = BeamSpec  # the oracle's escape hatch is not needed here
