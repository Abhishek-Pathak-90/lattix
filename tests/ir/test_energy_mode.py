"""The constant-p0 energy modes (lattix.ir.energy_mode): the probe momentum walk, the map
rescaling, and the write -> read round trip of every constant-p0 writer in every mode on a
deck that has both an RF gain and an explicit reference change."""
from __future__ import annotations

import math

import pytest

from lattix import read, write
from lattix.ir.elements import (
    RFP,
    Bend,
    BendP,
    Drift,
    Kicker,
    MagneticMultipoleP,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    Taylor,
)
from lattix.ir.energy_mode import (
    ENERGY_MODES,
    mode_ratio,
    probe_momentum_ratio,
    rigidity_for,
    scale_taylor,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate


def _deck() -> Lattice:
    """2 MeV proton, a 1 MV on-crest gap (RF-driven gain), a +0.5 MeV reference change (not
    RF-driven), and a quad / kicker / bend / map after each."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.0e6, rf_frequency_Hz=162.5e6)
    brho = ref.brho_signed
    lens = Taylor(name="lens")
    lens.matrix[1][0] = 0.7
    lens.matrix[3][2] = 0.7
    lens.matrix[0][1] = 0.02
    els = [
        Quadrupole(name="q0", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho})),
        RFCavity(name="cav", length=0.0, rf=RFP(voltage_V=1.0e6, phase_rad=0.0, frequency_Hz=162.5e6)),
        Drift(name="d1", length=0.1),
        Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho})),
        Kicker(name="cor", hkick=1e-3, vkick=-2e-3),
        Bend(name="b1", length=0.5, bend=BendP(angle=0.05)),
        lens,
        ReferenceChange(name="jump", dE_ref_eV=0.5e6),
        Quadrupole(name="q2", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho})),
        Kicker(name="cor2", hkick=2e-3),
    ]
    return Lattice.from_sequence("em", els, ref)


def _names(lat: Lattice) -> dict[str, str]:
    """IR name -> element name after a round trip (writers may lower-case)."""
    return {n.lower(): n for n in lat.elements}


def test_probe_momentum_follows_rf_gains_only():
    lat = _deck()
    placed = propagate(lat)
    ratios = dict(zip([p.element.name for p in placed], probe_momentum_ratio(placed, lat.reference), strict=True))
    p_start = lat.reference.pc_eV
    p_after_gap = lat.reference.advanced(dE_eV=1.0e6).pc_eV
    assert ratios["q0"] == 1.0
    assert ratios["cav"] == 1.0
    assert ratios["d1"] == pytest.approx(p_after_gap / p_start)
    assert ratios["q1"] == pytest.approx(p_after_gap / p_start)
    # the explicit reference change is invisible to a constant-p0 engine's particle
    assert ratios["q2"] == pytest.approx(p_after_gap / p_start)
    local_q2 = [p for p in placed if p.element.name == "q2"][0].ref_in.pc_eV
    assert local_q2 > p_after_gap                       # the IR reference did jump


def test_mode_ratio_and_rigidity():
    brho_start, brho_local, probe = 1.0, 1.3, 1.2
    assert mode_ratio("local", brho_local, brho_start, probe) == 1.0
    assert mode_ratio("constant", brho_local, brho_start, probe) == pytest.approx(1.3)
    assert mode_ratio("delta", brho_local, brho_start, probe) == pytest.approx(1.2)
    assert rigidity_for("local", brho_local, brho_start, probe) == pytest.approx(1.3)
    assert rigidity_for("constant", brho_local, brho_start, probe) == pytest.approx(1.0)
    assert rigidity_for("delta", brho_local, brho_start, probe) == pytest.approx(1.3 / 1.2)


def test_scale_taylor_is_a_similarity_transform_on_the_momentum_rows():
    m = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    m[1][0], m[0][1], m[5][4], m[2][5] = 0.7, 0.02, 0.3, 0.1
    o = [0.0, 1e-3, 0.0, 0.0, 0.0, 2e-3]
    m2, o2 = scale_taylor(m, o, 1.25)
    assert m2[1][0] == pytest.approx(0.7 * 1.25)          # px per x
    assert m2[0][1] == pytest.approx(0.02 / 1.25)         # x per px
    assert m2[5][4] == pytest.approx(0.3 * 1.25)          # delta per z
    assert m2[2][5] == pytest.approx(0.1 / 1.25)          # y per delta
    assert m2[0][0] == 1.0 and m2[1][1] == 1.0
    assert o2[1] == pytest.approx(1e-3 * 1.25) and o2[5] == pytest.approx(2e-3 * 1.25) and o2[0] == 0.0
    m3, o3 = scale_taylor(m2, o2, 1.25, inverse=True)
    import numpy as np

    assert np.allclose(m3, m, atol=1e-15) and np.allclose(o3, o, atol=1e-18)


@pytest.mark.parametrize("fmt", ["madx", "mad8", "xtrack"])
@pytest.mark.parametrize("mode", ENERGY_MODES)
def test_constant_p0_writers_round_trip_every_mode(tmp_path, fmt, mode):
    if fmt == "xtrack":
        pytest.importorskip("xtrack")
    lat = _deck()
    suffix = {"madx": ".madx", "mad8": ".lat", "xtrack": ".json"}[fmt]
    out = tmp_path / f"em_{mode}{suffix}"
    rep = write(lat, out, fmt, energy_mode=mode)
    # the reference change travels as a tag (EQUIVALENT, not lost); MAD8 alone drops the map
    lost = {e.code for e in rep.entries if e.cls.value in ("LOSSY", "DROPPED")}
    assert lost <= ({"TAYLOR_DROPPED"} if fmt == "mad8" else set()), rep.codes()
    back, rep2 = read(out, fmt, **({"species": "proton"} if fmt == "mad8" else {}))
    names = _names(back)
    for q in ("q0", "q1", "q2"):
        assert back.elements[names[q]].multipole.Bn[1] == pytest.approx(lat.elements[q].multipole.Bn[1], rel=1e-11), \
            f"{fmt} {mode}: gradient of {q}"
    assert back.elements[names["cor"]].hkick == pytest.approx(1e-3, rel=1e-11)
    assert back.elements[names["cor"]].vkick == pytest.approx(-2e-3, rel=1e-11)
    assert back.elements[names["cor2"]].hkick == pytest.approx(2e-3, rel=1e-11)
    assert back.elements[names["b1"]].bend.angle == pytest.approx(0.05, rel=1e-12)
    assert "k0" not in (back.elements[names["b1"]].native.get(fmt) or {})
    if fmt != "mad8":                                   # MAD8 has no matrix element
        lens = back.elements[names["lens"]]
        assert lens.matrix[1][0] == pytest.approx(0.7, rel=1e-11)
        assert lens.matrix[0][1] == pytest.approx(0.02, rel=1e-11)
    if mode == "delta":
        assert "CONST_P0_DELTA_RIGIDITY" in rep.codes()


def test_delta_mode_writes_start_rigidity_after_rf_and_local_after_a_reference_change(tmp_path):
    """The numbers in the deck: after the gap k1 = G/Bρ_start (the engine's delta does the rest);
    after the reference change the local rigidity comes back in (nothing in the deck moves the
    engine's particle there)."""
    lat = _deck()
    out = tmp_path / "em.madx"
    write(lat, out, "madx", energy_mode="delta")
    text = out.read_text()
    placed = {p.element.name: p for p in propagate(lat)}
    brho_start = lat.reference.brho_signed
    p_start = lat.reference.pc_eV
    p_probe = lat.reference.advanced(dE_eV=1.0e6).pc_eV

    def k1_of(name: str) -> float:
        line = [ln for ln in text.splitlines() if ln.startswith(f"{name}:")][0]
        return float(line.split("k1=")[1].split(",")[0].split(";")[0])

    g = lat.elements["q1"].multipole.Bn[1]
    assert k1_of("q1") == pytest.approx(g / brho_start, rel=1e-12)                      # pure RF gain
    brho_q2 = placed["q2"].ref_in.brho_signed
    assert k1_of("q2") == pytest.approx(g / brho_q2 * (p_probe / p_start), rel=1e-12)   # RF gain + jump
    # the kicker and the bend's k0 after the gap carry the same ratio
    cor = [ln for ln in text.splitlines() if ln.startswith("cor:")][0]
    assert float(cor.split("hkick=")[1].split(",")[0]) == pytest.approx(1e-3 * p_probe / p_start, rel=1e-12)
    b1 = [ln for ln in text.splitlines() if ln.startswith("b1:")][0]
    assert float(b1.split("k0=")[1].split(",")[0].split(";")[0]) == pytest.approx(0.05 / 0.5 * p_probe / p_start,
                                                                                 rel=1e-12)


def test_local_and_delta_coincide_without_rf_gain(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.0e6)
    brho = ref.brho_signed
    els = [Quadrupole(name="q0", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho})),
           ReferenceChange(name="jump", dE_ref_eV=0.5e6),
           Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho}))]
    lat = Lattice.from_sequence("nr", els, ref)
    texts = {}
    for mode in ("local", "delta"):
        out = tmp_path / f"nr_{mode}.madx"
        write(lat, out, "madx", energy_mode=mode)
        texts[mode] = [ln for ln in out.read_text().splitlines() if ln.startswith("q")]
    assert texts["local"] == texts["delta"]


def test_constant_and_delta_coincide_with_rf_gain_only(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.0e6, rf_frequency_Hz=162.5e6)
    brho = ref.brho_signed
    els = [RFCavity(name="cav", length=0.0, rf=RFP(voltage_V=1.0e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
           Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho}))]
    lat = Lattice.from_sequence("ro", els, ref)
    texts = {}
    for mode in ("constant", "delta"):
        out = tmp_path / f"ro_{mode}.madx"
        write(lat, out, "madx", energy_mode=mode)
        texts[mode] = [ln for ln in out.read_text().splitlines() if ln.startswith("q1")]
    assert texts["constant"] == texts["delta"]


def _two_gap_deck() -> Lattice:
    """Two on-crest gaps 0.3 m apart: the second one is reached earlier than a constant-beta0
    clock predicts, by (1/beta1 - 1/beta0)·0.3 m/c."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.0e6, rf_frequency_Hz=162.5e6)
    els = [RFCavity(name="g1", length=0.0, rf=RFP(voltage_V=1.0e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
           Drift(name="d1", length=0.3),
           RFCavity(name="g2", length=0.0, rf=RFP(voltage_V=1.0e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
           Drift(name="d2", length=0.1),
           RFCavity(name="g3", length=0.2, rf=RFP(voltage_V=1.0e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6))]
    return Lattice.from_sequence("slip", els, ref)


def test_phase_slip_turns_is_the_early_arrival_at_the_kick():
    from lattix.ir.energy_mode import phase_slip_turns
    from lattix.ir.reference import C_LIGHT

    lat = _two_gap_deck()
    placed = {p.element.name: p for p in propagate(lat)}
    start = lat.reference
    beta0 = start.beta
    assert phase_slip_turns(placed["g1"].ref_in, 0.0, 0.0, 162.5e6, start) == 0.0
    beta1 = placed["d1"].ref_in.beta                     # after the first gap
    dt = 0.3 / (beta1 * C_LIGHT) - 0.3 / (beta0 * C_LIGHT)
    assert phase_slip_turns(placed["g2"].ref_in, 0.3, 0.0, 162.5e6, start) == pytest.approx(162.5e6 * dt, rel=1e-12)
    assert 162.5e6 * dt < -0.05                          # tens of degrees, not a rounding effect
    # a thick cavity's kick sits at its centre: half a length of drift at the entrance velocity
    beta2 = placed["g3"].ref_in.beta
    s3 = placed["g3"].s_in
    t3 = placed["g3"].ref_in.time_s + 0.1 / (beta2 * C_LIGHT)
    assert phase_slip_turns(placed["g3"].ref_in, s3, 0.2, 162.5e6, start) == pytest.approx(
        162.5e6 * (t3 - (s3 + 0.1) / (beta0 * C_LIGHT)), rel=1e-12)


@pytest.mark.parametrize("fmt", ["madx", "mad8", "xtrack"])
def test_delta_mode_moves_downstream_cavity_phases_and_the_reader_moves_them_back(tmp_path, fmt):
    if fmt == "xtrack":
        pytest.importorskip("xtrack")
    from lattix.ir.energy_mode import phase_slip_turns

    lat = _two_gap_deck()
    suffix = {"madx": ".madx", "mad8": ".lat", "xtrack": ".json"}[fmt]
    out = tmp_path / f"slip{suffix}"
    rep = write(lat, out, fmt, energy_mode="delta")
    assert rep.codes()["CONST_P0_PHASE_SLIP"] == 2          # g2 and g3, not g1
    placed = {p.element.name: p for p in propagate(lat)}
    slip2 = phase_slip_turns(placed["g2"].ref_in, placed["g2"].s_in, 0.0, 162.5e6, lat.reference)
    if fmt in ("madx", "mad8"):
        from lattix.ir.rf import madx_lag

        text = out.read_text()
        lag_line = [ln for ln in text.splitlines() if ln.lower().startswith("g2:")][0]
        lag = float(lag_line.lower().split("lag=")[1].split(",")[0].split(";")[0].strip())
        assert lag == pytest.approx(madx_lag(-math.pi / 6) - slip2, abs=1e-12)
    back, rep2 = read(out, fmt, **({"species": "proton"} if fmt == "mad8" else {}))
    names = {n.lower(): n for n in back.elements}
    for g in ("g1", "g2", "g3"):
        assert back.elements[names[g]].rf.phase_rad == pytest.approx(-math.pi / 6, abs=1e-12), f"{fmt}: {g}"
    assert "PHASE_SLIP_RESTORED" in rep2.codes()
