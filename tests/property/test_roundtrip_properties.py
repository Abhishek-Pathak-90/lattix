"""Property-based round trips (PLAN §5.3): random physically-sane IR lattices are written
to a format and read back; the IR must be recovered up to the format's declared lossy
fields, and the walk invariants (length, exit energy) must survive.

Needs `hypothesis` (present in the `lattix` conda env and in CI's fast job)."""
from __future__ import annotations

import math

import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from lattix.ir import (  # noqa: E402
    RFP,
    Bend,
    BendP,
    Drift,
    Kicker,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    Solenoid,
    propagate,
    species,
)

REF = ReferenceParticle(species=species("proton"), kinetic_energy_eV=50e6, rf_frequency_Hz=325e6)

_len = st.floats(0.05, 3.0)
_name = st.integers(0, 10**6).map(lambda i: f"e{i}")


@st.composite
def st_element(draw):
    kind = draw(st.sampled_from(["drift", "quad", "sol", "bend", "cav", "kick"]))
    n = draw(_name)
    if kind == "drift":
        return Drift(name=n, length=draw(_len))
    if kind == "quad":
        q = Quadrupole(name=n, length=draw(_len))
        q.gradient = draw(st.floats(-30.0, 30.0))
        q.multipole.tilt[1] = draw(st.sampled_from([0.0, math.pi / 4, -math.pi / 8]))
        return q
    if kind == "sol":
        s = Solenoid(name=n, length=draw(_len))
        s.solenoid.Bsol_T = draw(st.floats(-6.0, 6.0))
        return s
    if kind == "bend":
        return Bend(name=n, length=draw(_len),
                    bend=BendP(angle=draw(st.floats(-0.5, 0.5).filter(lambda a: abs(a) > 1e-3)),
                               e1=draw(st.floats(-0.2, 0.2)), e2=draw(st.floats(-0.2, 0.2)),
                               edge_int1=draw(st.sampled_from([0.0, 0.45, 0.5])), hgap=draw(st.floats(0.0, 0.05))))
    if kind == "cav":
        return RFCavity(name=n, rf=RFP(voltage_V=draw(st.floats(1e4, 5e6)),
                                       phase_rad=draw(st.floats(-1.2, 1.2)), frequency_Hz=325e6))
    return Kicker(name=n, hkick=draw(st.floats(-2e-3, 2e-3)), vkick=draw(st.floats(-2e-3, 2e-3)))


@st.composite
def st_lattice(draw):
    els = draw(st.lists(st_element(), min_size=1, max_size=25))
    # MAD-X cannot hold a zero-length sequence (it aborts): every lattice gets ≥ 1 thick element
    hyp.assume(any(e.length > 0 for e in els))
    return Lattice.from_sequence("prop", els, REF)


def _physical(lat):
    """Physical elements with consecutive drifts coalesced (MAD-X sequence mode merges
    adjacent drifts into one implicit drift — an EQUIVALENT rewrite, not a loss)."""
    out = []
    for p in propagate(lat):
        if p.element.kind not in ("Drift", "Quadrupole", "Solenoid", "Bend", "RFCavity", "Kicker"):
            continue
        if p.element.kind == "Drift" and out and out[-1].element.kind == "Drift":
            merged = out[-1].element.model_copy(update={"length": out[-1].element.length + p.element.length})
            out[-1] = p.model_copy(update={"element": merged, "s_in": out[-1].s_in})
            continue
        out.append(p)
    return out


def _assert_same_physics(a, b, tol=1e-9):
    pa, pb = _physical(a), _physical(b)
    assert [p.element.kind for p in pa] == [p.element.kind for p in pb]
    for x, y in zip(pa, pb, strict=True):
        ex, ey = x.element, y.element
        assert ex.length == pytest.approx(ey.length, abs=tol)
        if isinstance(ex, Quadrupole):
            assert ex.gradient == pytest.approx(ey.gradient, rel=1e-9, abs=1e-12)
        elif isinstance(ex, Solenoid):
            assert ex.solenoid.Bsol_T == pytest.approx(ey.solenoid.Bsol_T, rel=1e-9, abs=1e-12)
        elif isinstance(ex, Bend):
            assert ex.bend.angle == pytest.approx(ey.bend.angle, rel=1e-9, abs=1e-12)
            assert ex.bend.e1 == pytest.approx(ey.bend.e1, abs=1e-9)
            assert ex.bend.e2 == pytest.approx(ey.bend.e2, abs=1e-9)
        elif isinstance(ex, RFCavity):
            assert ex.rf.voltage_V == pytest.approx(ey.rf.voltage_V, rel=1e-9)
            assert math.cos(ex.rf.phase_rad) == pytest.approx(math.cos(ey.rf.phase_rad), abs=1e-9)
            assert math.sin(ex.rf.phase_rad) == pytest.approx(math.sin(ey.rf.phase_rad), abs=1e-9)
        elif isinstance(ex, Kicker):
            assert ex.hkick == pytest.approx(ey.hkick, rel=1e-9, abs=1e-15)
            assert ex.vkick == pytest.approx(ey.vkick, rel=1e-9, abs=1e-15)
    assert pa[-1].s_out == pytest.approx(pb[-1].s_out, abs=tol)
    assert pa[-1].ref_out.kinetic_energy_eV == pytest.approx(pb[-1].ref_out.kinetic_energy_eV, rel=1e-9)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(lat=st_lattice())
def test_tracewin_roundtrip_preserves_physics(tmp_path_factory, lat):
    from lattix.formats import read, write

    d = tmp_path_factory.mktemp("tw")
    p = d / "x.dat"
    rep = write(lat, p)
    assert rep.ok, rep.summary()
    back, rep2 = read(p, species="proton", kinetic_energy_eV=REF.kinetic_energy_eV, frequency_Hz=325e6)
    assert rep2.ok, rep2.summary()
    _assert_same_physics(lat, back)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(lat=st_lattice())
def test_madx_roundtrip_preserves_physics(tmp_path_factory, lat):
    """MAD-X keeps p0: quads downstream of cavities re-read with the constant rigidity, so
    the writer runs in constant mode and the reader sees the same Bρ everywhere."""
    from lattix.formats import read, write

    d = tmp_path_factory.mktemp("mx")
    p = d / "x.madx"
    rep = write(lat, p, energy_mode="constant")
    assert not [e for e in rep.problems()], rep.summary()
    back, rep2 = read(p)
    # constant-p0 comparison: rebuild the IR walk with a constant reference for the original
    _assert_same_physics(lat.model_copy(update={"reference": REF}), back)
