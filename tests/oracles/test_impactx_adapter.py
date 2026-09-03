"""ImpactX oracle (PLAN §6 task 3.2): basis fingerprint, sign convention, reference
energy following, and self-consistency against the analytic maps.

Runs only where ImpactX is importable — in-process, or through the worker under
``LATTIX_IMPACTX_PYTHON`` / conda env ``LATTIX_IMPACTX_ENV`` (default ``lattix``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lattix.oracles import SPECIES, Basis, BeamSpec, Probe, get_oracle
from lattix.oracles.basis import drift_common, momentum_eV
from lattix.oracles.fingerprint import DECKS, FOLLOWS_P0, KE_EV, PHI_S_DEG, V_VOLT, fingerprint
from lattix.testing import require

pytestmark = pytest.mark.oracle_impactx
require("impactx")

MASS = SPECIES["proton"][0]
BEAM = BeamSpec(species="proton", kinetic_energy_eV=KE_EV, frequency_Hz=162.5e6)
DATA = Path(__file__).resolve().parents[1] / "data" / "public"
#: the shared ``tests/oracles/goldens/fingerprints.json`` is owned by the Phase-0 task,
#: so the ImpactX fingerprint is pinned in this format's own golden folder instead
GOLDEN_FP = Path(__file__).resolve().parents[1] / "golden" / "impactx" / "fingerprint.json"

DRIFT_IN = """\
beam.kin_energy = 2.1
beam.particle = proton
lattice.elements = d1
d1.type = drift
d1.ds = 1.0
"""

FODO_IN = """\
beam.kin_energy = 2.1
beam.particle = proton
lattice.elements = qf d1 qd d1
lattice.nslice = 1
qf.type = quad
qf.ds = 0.3
qf.k = 2.0
qd.type = quad
qd.ds = 0.3
qd.k = -2.0
d1.type = drift
d1.ds = 0.5
"""

LINAC_IN = """\
beam.kin_energy = 2.1
beam.particle = proton
lattice.elements = d0 c1 d0 c2 d0 c3 d0
d0.type = drift
d0.ds = 0.2
c1.type = shortrf
c1.V = 0.000319736677436836
c1.freq = 162500000
c1.phase = -30
c2.type = shortrf
c2.V = 0.000319736677436836
c2.freq = 162500000
c2.phase = -30
c3.type = shortrf
c3.V = 0.000319736677436836
c3.freq = 162500000
c3.phase = -30
"""


@pytest.fixture
def impactx_decks(tmp_path_factory):
    """Add an ``impactx`` entry to the shared fingerprint DECKS table for this test only
    (PLAN asks for the fingerprint machinery without editing ``fingerprint.py``)."""
    had = "impactx" in DECKS
    old_decks, old_p0 = DECKS.get("impactx"), FOLLOWS_P0.get("impactx")
    DECKS["impactx"] = DECKS["madx"]          # ImpactX reads MAD-X through lattix
    FOLLOWS_P0["impactx"] = True              # ShortRF pushes the reference particle
    try:
        yield
    finally:
        if had:
            DECKS["impactx"], FOLLOWS_P0["impactx"] = old_decks, old_p0
        else:
            DECKS.pop("impactx", None)
            FOLLOWS_P0["impactx"] = old_p0 if old_p0 is not None else False


def write(tmp_path: Path, body: str, name: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


# ------------------------------------------------------------------ availability
def test_available_and_registered():
    o = get_oracle("impactx")
    assert o.name == "impactx"
    assert o.formats == ("impactx", "madx")
    ok, why = o.available()
    assert ok, why
    assert "impactx" in why


# ------------------------------------------------------------------ fingerprint
def test_basis_fingerprint(impactx_decks, tmp_path):
    fp = fingerprint("impactx", tmp_path / "fp")
    d, c = fp["drift"], fp["cavity"]

    # (t, pt) as MAD-X delivers them after the adapter's sign conjugation
    assert fp["basis"] == Basis.IMPACTX.value
    assert d["R56_native"] == pytest.approx(223.14839568543672, rel=1e-12), (
        "ImpactX drift R56 in (t, pt) changed: L/(beta^2 gamma^2) at 2.1 MeV")
    assert d["R56_common"] == pytest.approx(d["R56_expected"], rel=1e-12)
    assert d["max_abs_err_common"] < 1e-8, (
        f"drift map deviates from analytic: {d}")

    # a cavity at phi_s = -30 deg must bunch and must gain +V cos 30
    assert c["R65_common"] < 0
    assert c["follows_p0"] is True
    assert c["gain_eV"] == pytest.approx(V_VOLT * math.cos(math.radians(PHI_S_DEG)), rel=1e-6)
    assert c["R65_common"] == pytest.approx(-4.3046, rel=2e-3), (
        "should match the p0-following engines (bmad/helix -4.305, tracewin -4.273)")

    # golden: an engine upgrade that flips a convention must fail loudly (PLAN 5.1)
    g = json.loads(GOLDEN_FP.read_text())["impactx"]
    assert np.sign(d["R56_native"]) == np.sign(g["drift"]["R56_native"])
    assert np.sign(c["R65_native"]) == np.sign(g["cavity"]["R65_native"])
    assert d["R56_native"] == pytest.approx(g["drift"]["R56_native"], rel=1e-12)
    assert c["R65_native"] == pytest.approx(g["cavity"]["R65_native"], rel=1e-9)
    assert c["R66_native"] == pytest.approx(g["cavity"]["R66_native"], rel=1e-9)


def test_drift_matches_the_analytic_common_map(tmp_path):
    deck = write(tmp_path, DRIFT_IN, "drift.impactx.in")
    r = get_oracle("impactx").run(deck, beam=BEAM, workdir=tmp_path / "wd")
    assert r.basis is Basis.IMPACTX
    assert r.names == ["d1"]
    assert r.total_length == pytest.approx(1.0)
    got = r.to_common().R_elem[0]
    want = drift_common(1.0, KE_EV, r.mass_eV)
    assert np.max(np.abs(got - want)) < 1e-12, f"\n{got}\n!=\n{want}"


def test_sign_convention_is_pinned(tmp_path):
    """ImpactX's raw (t, pt) are both flipped w.r.t. MAD-X's (T, pt): the two flips
    cancel in R56 but not in the dispersion column, so the adapter conjugates with
    S = diag(1,1,1,1,-1,-1).  This test pins that measurement."""
    body = """\
beam.kin_energy = 2.1
beam.particle = proton
lattice.elements = b
b.type = sbend
b.ds = 1.0
b.rc = 10.0
"""
    deck = write(tmp_path, body, "bend.impactx.in")
    r = get_oracle("impactx").run(deck, beam=BEAM, workdir=tmp_path / "wd")
    raw = np.asarray(r.meta["R_raw_impactx"], dtype=float)[0]
    out = r.R_elem[0]
    assert abs(raw[0, 5]) > 1e-3, "a sector bend must have dispersion"
    assert out[0, 5] == pytest.approx(-raw[0, 5]), "dispersion column flips"
    assert out[4, 0] == pytest.approx(-raw[4, 0]), "path-length row flips"
    assert out[4, 5] == pytest.approx(raw[4, 5]), "R56 is invariant under the double flip"
    assert out[0, 1] == pytest.approx(raw[0, 1]), "the transverse block is untouched"
    assert "diag(1,1,1,1,-1,-1)" in r.meta["sign_convention"]


# ------------------------------------------------------------------ physics
def test_quad_map_is_the_analytic_one(tmp_path):
    deck = write(tmp_path, FODO_IN, "fodo.impactx.in")
    r = get_oracle("impactx").run(deck, beam=BEAM, workdir=tmp_path / "wd")
    assert r.names == ["qf", "d1", "qd", "d1"]
    k, ds = 2.0, 0.3
    w = math.sqrt(k)
    R = r.R_elem[0]
    assert R[0, 0] == pytest.approx(math.cos(w * ds), rel=1e-12)
    assert R[0, 1] == pytest.approx(math.sin(w * ds) / w, rel=1e-12)
    assert R[1, 0] == pytest.approx(-w * math.sin(w * ds), rel=1e-12)
    assert np.linalg.det(R[0:2, 0:2]) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_reference_energy_follows_the_cavities(tmp_path):
    """ImpactX pushes the reference particle through ShortRF: the per-element reference
    energies must match the IR walk to 1e-9 (invariant I-18)."""
    from lattix.formats.impactx import Reader
    from lattix.ir.walk import propagate

    deck = write(tmp_path, LINAC_IN, "linac.impactx.in")
    r = get_oracle("impactx").run(deck, beam=BEAM, workdir=tmp_path / "wd")
    lat, _ = Reader().read(deck)
    lat.reference = lat.reference.model_copy(update={"kinetic_energy_eV": KE_EV})
    placed = propagate(lat)
    assert len(placed) == r.n
    for p, ein, eout in zip(placed, r.ref_kinetic_eV_in, r.ref_kinetic_eV_out,
                            strict=True):
        assert p.ref_in.kinetic_energy_eV == pytest.approx(ein, rel=1e-9)
        assert p.ref_out.kinetic_energy_eV == pytest.approx(eout, rel=1e-9)
    gain = float(r.ref_kinetic_eV_out[-1] - r.ref_kinetic_eV_in[0])
    assert gain == pytest.approx(3 * 3e5 * math.cos(math.radians(-30.0)), rel=1e-8)


def test_cavity_damping_matches_the_momentum_ratio(tmp_path):
    """Invariant I-5: det of each transverse 2x2 equals p_in/p_out."""
    deck = write(tmp_path, LINAC_IN, "linac.impactx.in")
    r = get_oracle("impactx").run(deck, beam=BEAM, workdir=tmp_path / "wd")
    for i in range(r.n):
        p_in = momentum_eV(r.ref_kinetic_eV_in[i], r.mass_eV)
        p_out = momentum_eV(r.ref_kinetic_eV_out[i], r.mass_eV)
        det = np.linalg.det(r.R_elem[i][0:2, 0:2])
        assert det == pytest.approx(p_in / p_out, rel=1e-10), r.names[i]


def test_probe_tracking_through_a_drift(tmp_path):
    deck = write(tmp_path, DRIFT_IN, "drift.impactx.in")
    probe = Probe.default(16)
    r = get_oracle("impactx").run(deck, beam=BEAM, probe=probe, workdir=tmp_path / "wd")
    got = r.to_common().probe_out
    want = probe.coords @ drift_common(1.0, KE_EV, r.mass_eV).T
    assert np.max(np.abs(got - want)) < 1e-12


def test_nslice_does_not_change_the_linear_map(tmp_path):
    deck = write(tmp_path, FODO_IN, "fodo.impactx.in")
    o = get_oracle("impactx")
    a = o.run(deck, beam=BEAM, nslice=1, workdir=tmp_path / "a")
    b = o.run(deck, beam=BEAM, nslice=8, workdir=tmp_path / "b")
    assert np.max(np.abs(a.R_cum[-1] - b.R_cum[-1])) < 1e-12


def test_finite_difference_step_is_not_critical(tmp_path):
    deck = write(tmp_path, FODO_IN, "fodo.impactx.in")
    o = get_oracle("impactx")
    a = o.run(deck, beam=BEAM, h=1e-7, workdir=tmp_path / "a")
    b = o.run(deck, beam=BEAM, h=1e-5, workdir=tmp_path / "b")
    assert np.max(np.abs(a.R_cum[-1] - b.R_cum[-1])) < 1e-10


# ------------------------------------------------------------------ MAD-X leg
#: ImpactX 26.01's own MADXParser is regex-per-line and only accepts a fixed number of
#: key=value pairs per type (``SBEND`` needs exactly five, ``TITLE`` aborts, a ``MONITOR``
#: used in a LINE must be defined) — this deck is written the way it accepts.
NATIVE_MADX = """\
BEAM,PARTICLE=PROTON,ENERGY=1.738272;
D1: DRIFT,L=0.5;
QF: QUADRUPOLE,L=0.3,K1=0.6;
QD: QUADRUPOLE,L=0.3,K1=-0.6;
FODO: LINE=(QF,D1,QD,D1);
USE, SEQUENCE = FODO;
"""


@pytest.mark.oracle_madx
def test_madx_deck_through_lattix_and_through_impactx_own_parser(tmp_path):
    """Two independent routes into ImpactX from the same MAD-X deck: lattix's reader +
    ImpactX writer, and ImpactX's own ``KnownElementsList.load_file``.

    The BeamSpec energy must equal the deck's (1.738 272 GeV total = 799.999 911 84 MeV
    kinetic with lattix's proton mass); otherwise lattix re-normalises k1 with the
    overridden rigidity while ImpactX's parser passes k1 through unchanged."""
    pytest.importorskip("cpymad")
    deck = tmp_path / "native.madx"
    deck.write_text(NATIVE_MADX)
    beam = BeamSpec(species="proton", kinetic_energy_eV=799_999_911.84)
    o = get_oracle("impactx")
    mine = o.run(deck, fmt="madx", beam=beam, workdir=tmp_path / "mine")
    theirs = o.run(deck, fmt="madx", beam=beam, native_madx=True, workdir=tmp_path / "theirs")
    assert theirs.names == mine.names
    assert mine.total_length == pytest.approx(theirs.total_length, abs=1e-12)
    assert np.max(np.abs(mine.R_cum[-1] - theirs.R_cum[-1])) < 1e-12, (
        f"lattix -> ImpactX differs from ImpactX's own MAD-X import:\n"
        f"{mine.R_cum[-1]}\n{theirs.R_cum[-1]}")


def test_native_madx_needs_a_beam(tmp_path):
    deck = tmp_path / "native.madx"
    deck.write_text(NATIVE_MADX)
    with pytest.raises(ValueError, match="native_madx"):
        get_oracle("impactx").run(deck, fmt="madx", native_madx=True)


def test_unknown_format_is_rejected(tmp_path):
    deck = write(tmp_path, DRIFT_IN, "drift.impactx.in")
    with pytest.raises(ValueError):
        get_oracle("impactx").run(deck, fmt="elegant")
    with pytest.raises(FileNotFoundError):
        get_oracle("impactx").run(tmp_path / "missing.in")


def test_meta_records_the_engine_and_the_ledger(tmp_path):
    deck = write(tmp_path, FODO_IN, "fodo.impactx.in")
    r = get_oracle("impactx").run(deck, beam=BEAM, workdir=tmp_path / "wd")
    assert r.meta["impactx_version"]
    assert r.meta["mode"] in ("in-process", "subprocess")
    assert r.meta["source_format"] == "impactx"
    assert r.meta["fidelity"]["target_format"] == "impactx"
    assert r.charge == 1
    assert r.mass_eV == pytest.approx(MASS, rel=1e-12)
