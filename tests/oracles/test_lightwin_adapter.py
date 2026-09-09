"""LightWin as an engine (Phase 5.2): units, the field-map cavity against HELIX and lattix's own
integration, the vendored ADS deck, the ``.dat`` lockstep against the lattix reader and the audit
of what LightWin cannot model.  Runs where LightWin is installed (conda env ``lightwin`` or
``LATTIX_LIGHTWIN_PYTHON``; marker ``oracle_lightwin``)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix import read
from lattix.ir.walk import propagate
from lattix.oracles import get_oracle
from lattix.oracles.base import BeamSpec
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, write_decks
from lattix.testing import helix_path

pytestmark = pytest.mark.oracle_lightwin
DATA = Path(__file__).resolve().parents[1] / "data" / "public"
M_P = 938_272_088.16
C = 299_792_458.0


@pytest.fixture(scope="module")
def lightwin():
    o = get_oracle("lightwin")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"LightWin unavailable: {why}")
    return o


def _beam(ke=KE_EV, freq=FREQ_HZ, species="proton"):
    return BeamSpec(species=species, kinetic_energy_eV=ke, frequency_Hz=freq)


def test_drift_and_quadrupole_in_the_si_tracewin_basis(lightwin, tmp_path):
    deck = tmp_path / "dq.dat"
    deck.write_text("FREQ 162.5\nDRIFT 1000 30\nQUAD 200 5.0 30 0 0 0 0 0 0\nDRIFT 500 30\nEND\n")
    r = lightwin.run(deck, fmt="tracewin", beam=_beam(), workdir=tmp_path / "wd")
    assert r.names == ["DR1", "QP1", "DR2"] and r.basis.value == "tracewin"
    assert r.length == pytest.approx([1.0, 0.2, 0.5])
    gamma = 1 + KE_EV / M_P
    R = r.R_elem[0]
    assert R[0, 1] == pytest.approx(1.0) and R[4, 5] == pytest.approx(1.0 / gamma**2, rel=1e-9)   # (z [m], dp/p)
    pc = math.sqrt((KE_EV + M_P) ** 2 - M_P**2)
    k = 5.0 / (pc / C)                                  # G/Bρ; focusing in x for a proton and G > 0
    Q = r.R_elem[1]
    assert Q[0, 0] == pytest.approx(math.cos(math.sqrt(k) * 0.2), rel=1e-9)
    assert Q[1, 0] == pytest.approx(-math.sqrt(k) * math.sin(math.sqrt(k) * 0.2), rel=1e-9)
    assert Q[2, 2] == pytest.approx(math.cosh(math.sqrt(k) * 0.2), rel=1e-9)
    assert r.ref_kinetic_eV_out == pytest.approx([KE_EV] * 3)
    assert r.meta["dropped"] == [] and r.meta["substituted"] == []


def test_field_map_cavity_matches_lattix_and_helix(lightwin, tmp_path):
    """The fingerprint cavity: 2 cm of constant Ez at φs = −30° (SET_SYNC_PHASE)."""
    deck, fmt = write_decks("lightwin", tmp_path)["cavity"]
    r = lightwin.run(deck, fmt=fmt, beam=_beam(), workdir=tmp_path / "wd")
    j = r.names.index("FM1")
    gain = r.ref_kinetic_eV_out[j] - r.ref_kinetic_eV_in[j]
    p = r.meta["params"][j]
    assert p["phi_s"] == pytest.approx(math.radians(-30.0), abs=1e-6)   # SET_SYNC_PHASE honoured (solver: 2e-8)
    # V_cav·cos φs is TraceWin's definition; the RK-integrated gain differs from it by 0.2 %
    assert gain == pytest.approx(p["v_cav_mv"] * 1e6 * math.cos(math.radians(30.0)), rel=5e-3)
    assert gain == pytest.approx(0.87e6, rel=2e-2)                          # ~1 MV effective
    lat, _ = read(deck, "tracewin", species="proton", kinetic_energy_eV=KE_EV)
    fm = [q for q in propagate(lat) if q.element.kind == "FieldMap"][0]
    assert fm.ref_out.kinetic_energy_eV - fm.ref_in.kinetic_energy_eV == pytest.approx(gain, rel=2e-3)
    helix = get_oracle("helix")
    if not helix.available()[0]:
        pytest.skip("HELIX not available for the engine-vs-engine half")
    rh = helix.run(deck, fmt=fmt, beam=_beam(), workdir=tmp_path / "wd_h")
    pc = compare_pair(rh, r)
    assert pc.n_shared >= 3
    # Equivalent tier: the two field-map models differ in the transverse RF focusing they carry
    assert pc.blocks["T4x4"] < 2e-2 and pc.energy_rel < 5e-3


def test_ads_example_energy_follows_lattix_integration(lightwin, tmp_path):
    """The vendored ADS deck (627 elements, 142 1-D maps with *relative* phases): LightWin's own
    regression value at the exit is 502.24 MeV and lattix's field-map integration reaches it; HELIX
    adds the running bunch phase to relative phases and ends at 22.6 MeV (docs/oracles.md)."""
    deck = DATA / "lightwin" / "example.dat"
    r = lightwin.run(deck, fmt="tracewin", beam=_beam(20e6, 352.2e6), workdir=tmp_path / "wd")
    assert len(r.names) == 627 and r.meta["dropped"] == [] and r.meta["substituted"] == []
    assert r.ref_kinetic_eV_out[-1] == pytest.approx(502.24092e6, rel=1e-4)
    lat, _ = read(deck, "tracewin", species="proton", kinetic_energy_eV=20e6)
    w_lattix = [q.ref_out.kinetic_energy_eV for q in propagate(lat) if q.element.kind == "FieldMap"]
    w_lw = [w for w, k in zip(r.ref_kinetic_eV_out, r.meta["kinds"], strict=True) if k.startswith("FieldMap")]
    assert len(w_lattix) == len(w_lw) == 142
    assert np.allclose(w_lattix, w_lw, rtol=2e-4)


@pytest.mark.parametrize("rel, species, ke", [
    ("lightwin/example.dat", "proton", 20e6),
    ("helix/fodo_cell.dat", "proton", 2.1e6),
    ("helix/mebt_line.dat", "h-", 2.1e6),
    ("helix/bend_line.dat", "proton", 2.1e6),
])
def test_dat_lockstep_with_the_lattix_reader(lightwin, tmp_path, rel, species, ke):
    """LightWin's parser and lattix's reader agree on the deck: total length, quadrupole gradients,
    field-map amplitudes and phases, and every keyword LightWin skips is one lattix reads."""
    deck = DATA / rel
    table = lightwin.run(deck, fmt="tracewin", beam=_beam(ke, 162.5e6, species),
                         workdir=tmp_path / "wd", parse_only=True)
    rows = table["elements"]
    lat, _ = read(deck, "tracewin", species=species, kinetic_energy_eV=ke)
    placed = list(propagate(lat))
    assert sum(r["length_m"] for r in rows) == pytest.approx(sum(p.element.length for p in placed), abs=1e-9)
    grads = [r["grad"] for r in rows if r["kind"] == "Quad"]
    assert grads == pytest.approx([p.element.multipole.Bn[1] for p in placed if p.element.kind == "Quadrupole"])
    fms = [r for r in rows if r["kind"].startswith("FieldMap")]
    ir_fms = [p.element for p in placed if p.element.kind == "FieldMap"]
    assert len(fms) == len(ir_fms)
    assert [r["k_e"] for r in fms] == pytest.approx([e.ke for e in ir_fms])
    assert [r["phi_ref"] for r in fms] == pytest.approx([e.rf.phase_rad for e in ir_fms], abs=1e-12)
    skipped = [d["keyword"] for d in table["dropped"]]
    assert set(skipped) <= {"GAP", "TITLE", "PARTRAN_STEP", "NCELLS", "DTL_CEL", "SET_SYNC_PHASE"}
    assert skipped.count("GAP") == sum(1 for p in placed if p.element.kind == "RFCavity")


def test_reports_what_it_cannot_model(lightwin, tmp_path):
    """MEBT: two GAP cards (skipped by LightWin), a THIN_STEERING and an APERTURE (propagated as
    drifts), TITLE/PARTRAN_STEP (commands, harmless) — all audited, and the battery makes such a
    pair report-only."""
    r = lightwin.run(DATA / "helix" / "mebt_line.dat", fmt="tracewin", beam=_beam(2.1e6, 162.5e6, "h-"),
                     workdir=tmp_path / "wd")
    assert [d["keyword"] for d in r.meta["dropped"]] == ["GAP", "GAP"]
    assert {d["keyword"] for d in r.meta["ignored_commands"]} >= {"TITLE", "PARTRAN_STEP"}
    assert "TS1" in r.meta["substituted"] and "AP1" in r.meta["substituted"]
    assert any("GAP" in w for w in r.warnings)
    assert r.charge == -1 and r.mass_eV == pytest.approx(M_P + 2 * 510_998.95)


def test_battery_falls_back_to_lightwin_without_helix(lightwin, monkeypatch):
    from lattix import crossval

    helix = get_oracle("helix")
    monkeypatch.setattr(type(helix), "available", lambda self: (False, "not here"))
    assert crossval._pick_engine("tracewin") == "lightwin"
    monkeypatch.setattr(type(helix), "available", lambda self: (True, "here"))
    assert crossval._pick_engine("tracewin") == "helix"
    assert crossval._pick_engine("madx") == "madx"


HELIX_EXAMPLES = helix_path('examples')


@pytest.mark.oracle_helix
def test_hwr_section_gate_against_helix_and_tracewin(lightwin, tmp_path):
    """PLAN_PHASE5 §5.2 gate on the private PIP-II ``mebt+hwr.dat`` (never in the repo).  The deck
    lattix writes with its eight static solenoid maps as hard-edge solenoids (``static_maps=
    "hard_edge"``) is read by LightWin and by lattix alike (structural lockstep), and the reference
    energy after each of the twelve RF maps (four bunchers, eight HWR cavities) agrees with HELIX
    (measured 1.2e-4) and with the CEA TraceWin export of the same line
    (``Tracewin_code/mebtplushwr/energy.txt``, run at its own 2.1227 MeV input energy) within 0.5 %.
    H⁻ is emulated as a positive particle in mirrored fields (LightWin's synchronous phase is
    charge-blind).  LightWin 0.16.5 has no solenoid model
    (``SolenoidEnvelope3DParameters`` raises NotImplementedError; the worker writes drifts and
    audits them), so the transverse block of a solenoid-focused linac cannot be compared —
    measured 2026-09-05, docs/oracles.md."""
    from lattix import write

    deck = HELIX_EXAMPLES / "pipii" / "mebt+hwr" / "mebt+hwr.dat"
    if not deck.is_file():
        pytest.skip("private PIP-II example deck not available")
    helix = get_oracle("helix")
    if not helix.available()[0]:
        pytest.skip("HELIX not available")
    lat, rep = read(deck, "tracewin", species="h-", kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    out = tmp_path / "mebt+hwr.dat"
    rep_w = write(lat, out, "tracewin", static_maps="hard_edge")
    assert rep_w.codes().get("FM_SOL_HARDEDGE") == 8
    assert "FIELD_MAP 10 " not in out.read_text(encoding="latin-1")
    back, _ = read(out, "tracewin", species="h-", kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    beam = _beam(2.1e6, 162.5e6, "h-")
    rl = lightwin.run(out, fmt="tracewin", beam=beam, workdir=tmp_path / "lw")
    rh = helix.run(out, fmt="tracewin", beam=beam, workdir=tmp_path / "helix")
    assert rl.meta["dropped"] == [] and rl.meta["solenoids_as_drifts"] == 8 and rl.meta["negative_charge_emulated"]
    # structural lockstep between LightWin's parser and lattix's reader on the written deck
    table = lightwin.run(out, fmt="tracewin", beam=beam, workdir=tmp_path / "lw_parse", parse_only=True)
    placed = list(propagate(back))
    total = sum(p.element.length for p in placed)
    assert sum(r["length_m"] for r in table["elements"]) == pytest.approx(total, abs=1e-10)
    assert [r["grad"] for r in table["elements"] if r["kind"] == "Quad"] == \
        pytest.approx([p.element.multipole.Bn[1] for p in placed if p.element.kind == "Quadrupole"])
    # the reference energy after every cavity: LightWin vs HELIX, by position
    w_lw = [(round(float(s), 5), w) for s, w, k in zip(rl.s_out, rl.ref_kinetic_eV_out, rl.meta["kinds"], strict=True)
            if k.startswith("FieldMap")]
    w_h = {round(float(s), 5): w for s, w in zip(rh.s_out, rh.ref_kinetic_eV_out, strict=True)}
    assert len(w_lw) == 12                                     # 4 MEBT bunchers + 8 HWR cavities
    for s, w in w_lw:
        assert w == pytest.approx(w_h[s], rel=5e-3), s
    assert rl.ref_kinetic_eV_out[-1] == pytest.approx(rh.ref_kinetic_eV_out[-1], rel=1e-3)
    # … and vs TraceWin itself (the CEA export of this line, at the export's own input energy)
    export = HELIX_EXAMPLES.parent / "Tracewin_code" / "mebtplushwr" / "energy.txt"
    if export.is_file():
        cea = np.loadtxt(export, skiprows=1)
        w0 = float(cea[0, 1]) * 1e6
        rl2 = lightwin.run(out, fmt="tracewin", beam=_beam(w0, 162.5e6, "h-"), workdir=tmp_path / "lw_cea")
        for s, w, k in zip(rl2.s_out, rl2.ref_kinetic_eV_out, rl2.meta["kinds"], strict=True):
            if not k.startswith("FieldMap"):
                continue
            w_tw = float(cea[np.searchsorted(cea[:, 0], float(s) + 1e-3), 1]) * 1e6
            assert w == pytest.approx(w_tw, rel=5e-3), (s, w, w_tw)
