"""DYNAC as an engine (Phase 5.8, local only): the FODO gate against cpymad, TraceWin decks against HELIX,
the charge-signed buncher, a thick cavity through FIELD + CAVNUM, the tw2dyn lockstep and the shipped SNS
deck through the engine.  Runs where a built ``dynac`` is at hand (``LATTIX_DYNAC_EXE``; marker
``oracle_dynac``); nothing from DYNAC's ``datafiles/`` enters the repository."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lattix import read, write
from lattix.crossval import _beam
from lattix.ir.elements import RFP, Drift, RFCavity
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.oracles import get_oracle
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, V_VOLT, write_decks

pytestmark = pytest.mark.oracle_dynac
DATA = Path(__file__).resolve().parents[1] / "data" / "public"
DYNAC_CLONE = Path("/Users/abhishekpathak/Desktop/Projects/particle_tracking_codes/tier3_peers/dynac")
TW2DYN = DYNAC_CLONE / "build" / "converters" / "tw2dyn"
SNS_DECK = DYNAC_CLONE / "datafiles" / "sns" / "mebt_dtl1.in"
PRECISION = 5e-5                      # WRBEAM dumps carry 6 significant digits: ~1e-5 per fitted map


@pytest.fixture(scope="module")
def dynac():
    o = get_oracle("dynac")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"DYNAC unavailable: {why}")
    return o


@pytest.mark.oracle_madx
def test_fodo_matches_cpymad_on_every_block(dynac, tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.dynac.in"
    rep = write(lat, out, "dynac")
    assert rep.ok
    rd = dynac.run(out, fmt="dynac", beam=_beam(lat), workdir=tmp_path / "dy")
    rm = get_oracle("madx").run(DATA / "helix" / "fodo.madx", fmt="madx", beam=_beam(lat), workdir=tmp_path / "madx")
    pc = compare_pair(rm, rd)
    assert pc.n_shared == 8
    assert pc.blocks["T4x4"] < PRECISION and pc.blocks["disp"] < PRECISION and pc.blocks["path"] < PRECISION
    assert pc.blocks["R56"] < 1e-5


@pytest.mark.oracle_helix
@pytest.mark.parametrize("deck, sp, tol, e_tol", [("fodo_cell.dat", "proton", PRECISION, 2e-5),
                                                   ("solenoid_channel.dat", "proton", PRECISION, 2e-5),
                                                   ("mebt_line.dat", "h-", 3e-2, 2e-5)])
def test_tracewin_decks_against_helix(dynac, tmp_path, deck, sp, tol, e_tol):
    """Quads and solenoids at the dump precision; the MEBT's thin gaps are DYNAC bunchers whose RF
    defocusing uses the mid-gap velocity (TraceWin/HELIX: the entrance one): Equivalent tier."""
    helix = get_oracle("helix")
    if not helix.available()[0]:
        pytest.skip("HELIX not available")
    lat, _ = read(DATA / "helix" / deck, "tracewin", species=sp, kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    out = tmp_path / (Path(deck).stem + ".dynac.in")
    write(lat, out, "dynac")
    rd = dynac.run(out, fmt="dynac", beam=_beam(lat), workdir=tmp_path / "dy")
    rh = helix.run(DATA / "helix" / deck, fmt="tracewin", beam=_beam(lat), workdir=tmp_path / "hx")
    pc = compare_pair(rh, rd)
    assert pc.n_shared >= 7
    assert pc.blocks["T4x4"] < tol and pc.blocks["disp"] < tol
    assert pc.energy_rel < e_tol


def test_negative_charge_gains_at_the_same_phase(dynac, tmp_path):
    """A DYNAC buncher gains q·V·cos φ: the writer adds 180° for a negative species, so a proton and an H⁻
    gain the same +V·cos 30° at the IR's φs = −30° (fingerprint deck)."""
    deck, fmt = write_decks("dynac", tmp_path)["cavity"]
    from lattix.oracles.base import BeamSpec

    r = dynac.run(deck, fmt=fmt, beam=BeamSpec(species="proton", kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ),
                  workdir=tmp_path / "p")
    j = r.names.index("c")
    gain = V_VOLT * math.cos(math.radians(30.0))
    assert r.ref_kinetic_eV_out[j] - r.ref_kinetic_eV_in[j] == pytest.approx(gain, rel=2e-5)
    assert r.to_common().R_elem[j][5, 4] < 0                       # bunching
    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                                frequency_Hz=FREQ_HZ)),
                                      Drift(name="d2", length=0.5)], ref)
    out = tmp_path / "hminus.dynac.in"
    write(lat, out, "dynac")
    rh = dynac.run(out, fmt="dynac", beam=_beam(lat), workdir=tmp_path / "h")
    k = rh.names.index("c")
    assert rh.ref_kinetic_eV_out[k] - rh.ref_kinetic_eV_in[k] == pytest.approx(gain, rel=2e-5)
    assert rh.to_common().R_elem[k][5, 4] < 0 and rh.charge == -1


def test_thick_cavity_is_a_centred_buncher(dynac, tmp_path):
    """A 0.2 m, 1 MV cavity at −30°: a BUNCHER at its centre between half drifts — the exact gain, the
    length kept (MEASURED: CAVNUM on a generated profile gains 1.6× lattix's calibration at 2.1 MeV)."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    lat = Lattice.from_sequence("t", [Drift(name="d1", length=0.3),
                                      RFCavity(name="c", length=0.2,
                                               rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                      frequency_Hz=FREQ_HZ)),
                                      Drift(name="d2", length=0.3)], ref)
    out = tmp_path / "thick.dynac.in"
    rep = write(lat, out, "dynac")
    assert "THICK_CAVITY_AS_BUNCHER" in rep.codes() and not (tmp_path / "thick.fields.txt").exists()
    r = dynac.run(out, fmt="dynac", beam=_beam(lat), workdir=tmp_path / "t")
    j = r.names.index("c")
    assert r.total_length == pytest.approx(0.8)
    gain = r.ref_kinetic_eV_out[j] - r.ref_kinetic_eV_in[j]
    assert gain == pytest.approx(V_VOLT * math.cos(math.radians(30.0)), rel=2e-5)
    assert r.to_common().R_elem[j][5, 4] < 0


@pytest.mark.skipif(not TW2DYN.is_file(), reason="tw2dyn (DYNAC's own TraceWin converter) not built")
def test_tw2dyn_lockstep_on_a_tracewin_deck(dynac, tmp_path):
    """DYNAC's own ``tw2dyn`` on HELIX's MEBT deck against lattix's TraceWin reader → DYNAC writer: the
    drifts, quadrupole fields and buncher voltages/phases agree to 1e-6 (tw2dyn writes the raw TraceWin
    phase, which is the buncher's charge-signed phase for this H⁻ deck).  tw2dyn knows DRIFT, QUAD, GAP,
    FREQ and END only, reads ten GAP parameters and writes no title or beam block: the deck is filtered
    and padded for it, and lattix reads its output with the beam given as options."""
    import re

    src = DATA / "helix" / "mebt_line.dat"
    wd = tmp_path / "tw2dyn"
    wd.mkdir()
    kept = []
    for ln in src.read_text().splitlines():
        t = ln.split()
        if t and re.match(r"^(DRIFT|QUAD|GAP|FREQ|END)$", t[0]):
            kept.append(" ".join(t + ["0"] * (11 - len(t)) if t[0] == "GAP" else t))
    (wd / "mebt.dat").write_text("\n".join(kept) + "\n")
    proc = subprocess.run([str(TW2DYN), "mebt.dat", "mebt_tw2dyn.in"], cwd=wd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-500:]
    opts = {"species": "h-", "kinetic_energy_eV": 2.1e6, "frequency_Hz": 162.5e6}
    ref_lat, _ = read(src, "tracewin", **opts)
    theirs, _ = read(wd / "mebt_tw2dyn.in", "dynac", **opts)
    out = tmp_path / "mebt_lattix.dynac.in"
    write(ref_lat, out, "dynac")
    ours, _ = read(out, "dynac")

    def summary(lat):
        quads = [(round(p.s_in, 9), p.element.length, p.element.multipole.Bn.get(1, 0.0))
                 for p in lat.flatten() if p.element.kind == "Quadrupole"]
        gaps = [(round(p.s_in, 9), p.element.rf.voltage_V, math.degrees(p.element.rf.phase_rad),
                 p.element.rf.frequency_Hz)
                for p in lat.flatten() if p.element.kind == "RFCavity"]
        return quads, gaps, lat.total_length

    q1, g1, L1 = summary(theirs)
    q2, g2, L2 = summary(ours)
    assert len(q1) == len(q2) == 8 and len(g1) == len(g2) == 2
    assert L1 == pytest.approx(L2, rel=1e-9)
    for a, b in zip(q1, q2, strict=True):
        assert a[0] == pytest.approx(b[0], abs=1e-9) and a[1] == pytest.approx(b[1], rel=1e-9)
        assert a[2] == pytest.approx(b[2], rel=1e-6)                       # tw2dyn's single-precision field
    for a, b in zip(g1, g2, strict=True):
        assert a[0] == pytest.approx(b[0], abs=1e-9) and a[1] == pytest.approx(b[1], rel=1e-6)
        assert (a[2] - b[2] + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-6)
        assert a[3] == pytest.approx(b[3]) == pytest.approx(352.21e6)         # the deck's FREQ through NEWF
    # and through the engine: the same maps at the dump precision
    ra = dynac.run(wd / "mebt_tw2dyn.in", fmt="dynac", beam=_beam(ref_lat), workdir=tmp_path / "a")
    rb = dynac.run(out, fmt="dynac", beam=_beam(ref_lat), workdir=tmp_path / "b")
    pc = compare_pair(ra, rb)
    assert pc.n_shared >= 10 and pc.blocks["T4x4"] < PRECISION and pc.energy_rel < 2e-5


@pytest.mark.skipif(not SNS_DECK.is_file(), reason="DYNAC clone with the SNS example not present")
def test_shipped_sns_deck_round_trip_through_the_engine(dynac, tmp_path):
    """The SNS MEBT + DTL tank 1 deck as shipped and as lattix re-writes it from the IR: the MEBT
    (quadrupoles, drifts, four bunchers) reproduces DYNAC's maps at the dump precision; the DTL's CAVSC
    cells become thin gaps between half-cell drifts (EQUIVALENT ``CAVSC_AS_GAP``): the end energy agrees
    to 0.3 % while the transverse map diverges cell by cell (MEASURED: 2.8e-4 after the first cell, 0.36
    at the tank end — DYNAC's gap dynamics against its thin buncher)."""
    lat, rep = read(SNS_DECK, "dynac")
    assert rep.codes() == {"CAVSC_AS_GAP": 60}
    out = tmp_path / "sns.dynac.in"
    write(lat, out, "dynac")
    beam = _beam(lat)
    ra = dynac.run(SNS_DECK, fmt="dynac", beam=beam, workdir=tmp_path / "orig")
    rb = dynac.run(out, fmt="dynac", beam=beam, workdir=tmp_path / "lattix")
    assert ra.total_length == pytest.approx(rb.total_length, rel=1e-6)
    assert rb.ref_kinetic_eV_out[-1] == pytest.approx(ra.ref_kinetic_eV_out[-1], rel=5e-3)
    A, B = ra.to_common(), rb.to_common()
    last_a = {round(s, 6): i for i, s in enumerate(A.s_out)}
    last_b = {round(s, 6): j for j, s in enumerate(B.s_out)}
    common = sorted(set(last_a) & set(last_b))
    assert len(common) > 200
    mebt = [s for s in common if s < 3.68]                  # up to the last quadrupole before DTL tank 1
    i, j = last_a[mebt[-1]], last_b[mebt[-1]]
    Ra, Rb = A.R_cum[i], B.R_cum[j]
    assert np.max(np.abs(Ra[:4, :4] - Rb[:4, :4])) / max(1.0, np.max(np.abs(Ra[:4, :4]))) < 1e-4
    assert A.ref_kinetic_eV_out[i] == pytest.approx(2.5e6, rel=1e-6)
