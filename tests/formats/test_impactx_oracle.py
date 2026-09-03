"""ImpactX translation, checked with engines on both ends (PLAN §5.2, anti-cancellation).

``fodo.madx`` → (lattix MAD-X reader) → IR → (lattix ImpactX writer, both flavours) →
ImpactX, compared per element against **cpymad on the source deck**.  Nothing here is a
round trip: the two engines never see each other's files.

Tolerance: PLAN §5.2's Exact tier is 1e-8 on the transverse 4×4 and 1e-8 on dispersion.
ImpactX's ``Drift``/``Quad``/``Sbend``/``DipEdge`` pushes are their analytic linear maps
and its default symplectic integrator does not touch them, so ``nslice`` is irrelevant to
the linear map (asserted) and the measured agreement is ~2e-15 — no ``*_exact`` element
is needed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix.formats.impactx import Reader, Writer
from lattix.formats.madx import Reader as MadxReader
from lattix.ir.walk import propagate
from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.compare import compare_pair
from lattix.testing import require

pytestmark = [pytest.mark.oracle_impactx, pytest.mark.oracle_madx]
require("impactx")
require("madx")

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
FODO = DATA / "fodo.madx"
#: the deck says ENERGY = 1.738272 GeV total, i.e. 799.999 911 84 MeV kinetic with
#: lattix's proton mass (PLAN Phase-1 note: not 800 MeV)
FODO_KE = 799_999_911.84
BEAM = BeamSpec(species="proton", kinetic_energy_eV=FODO_KE)

#: PLAN §5.2 Exact tier
TOL_T4X4 = 1e-8
TOL_DISP = 1e-8


@pytest.fixture(scope="module")
def madx_result():
    return get_oracle("madx").run(FODO, fmt="madx", beam=BEAM)


def _translate(tmp_path: Path, flavor: str, **opts) -> Path:
    lat, rep = MadxReader().read(FODO)
    suffix = ".impactx.py" if flavor == "python" else ".impactx.in"
    out = tmp_path / f"fodo{suffix}"
    rep_out = Writer().write(lat, out, flavor=flavor, strict=True, **opts)
    assert rep_out.ok, rep_out.summary()
    return out


# ------------------------------------------------------------------ A-style gate
@pytest.mark.parametrize("nslice", [1, 5])
def test_fodo_madx_to_impactx_inputs_matches_cpymad(tmp_path, madx_result, nslice):
    deck = _translate(tmp_path, "inputs", nslice=nslice)
    got = get_oracle("impactx").run(deck, fmt="impactx", beam=BEAM, workdir=tmp_path / "wd")
    cmp = compare_pair(got, madx_result)
    assert cmp.n_shared >= 4, cmp.row()
    assert cmp.length_a == pytest.approx(cmp.length_b, abs=1e-12)
    assert cmp.blocks["T4x4"] < TOL_T4X4, cmp.row()
    assert cmp.blocks["disp"] < TOL_DISP, cmp.row()
    assert cmp.blocks["R56"] < TOL_DISP, cmp.row()
    assert cmp.blocks["path"] < TOL_DISP, cmp.row()
    assert cmp.energy_rel < 1e-12, cmp.row()


def test_fodo_madx_to_impactx_python_deck_builds_the_same_lattice(tmp_path, madx_result):
    """The Python flavour is not parsed back, so it is checked by *executing* the same
    element list the oracle builds and by asserting the script's own calls match it."""
    py = _translate(tmp_path, "python")
    text = py.read_text()
    inp = _translate(tmp_path, "inputs")
    lat, _ = MadxReader().read(FODO)
    from lattix.formats.impactx.writer import to_elements

    emits, _ = to_elements(lat, flavor="python")
    for e in emits:
        assert e.python_call() in text
    # the inputs deck the oracle runs has the same element sequence
    names = inp.read_text().split("lattice.elements = ", 1)[1].splitlines()[0].split()
    assert names == [e.name for e in emits]

    got = get_oracle("impactx").run(inp, fmt="impactx", beam=BEAM, workdir=tmp_path / "wd")
    assert got.names == names
    assert compare_pair(got, madx_result).blocks["T4x4"] < TOL_T4X4


def test_translated_fodo_has_no_downgrades(tmp_path):
    lat, _ = MadxReader().read(FODO)
    for flavor in ("python", "inputs"):
        rep = Writer().write(lat, tmp_path / f"f.{flavor}", flavor=flavor)
        assert rep.ok, rep.summary()
        assert set(rep.counts) <= {"EXACT"}, rep.summary()


# ------------------------------------------------------------ accelerating linac
def linac_deck(tmp_path: Path) -> tuple[Path, object]:
    """Three thin 300 kV gaps at phi_s = -30 deg, proton at 2.1 MeV."""
    import math as _m

    from lattix.ir.elements import RFP, Drift, MagneticMultipoleP, Quadrupole, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6,
                            rf_frequency_Hz=162.5e6)
    b = ref.brho_signed
    els = []
    for i in range(3):
        sign = 1.0 if i % 2 == 0 else -1.0
        els += [
            Quadrupole(name=f"q{i + 1}", length=0.1,
                       multipole=MagneticMultipoleP(Bn={1: sign * 5.0 * b})),
            Drift(name=f"da{i + 1}", length=0.2),
            RFCavity(name=f"cav{i + 1}", length=0.0,
                     rf=RFP(voltage_V=3e5, phase_rad=-_m.pi / 6, frequency_Hz=162.5e6)),
            Drift(name=f"db{i + 1}", length=0.2),
        ]
    lat = Lattice.from_sequence("linac", els, ref)
    deck = tmp_path / "linac.impactx.in"
    rep = Writer().write(lat, deck, flavor="inputs", strict=True)
    assert rep.ok, rep.summary()
    return deck, lat


def test_accelerating_linac_reference_energies(tmp_path):
    """ImpactX follows the reference energy, so the IR walk and the engine must agree to
    1e-9 element by element (invariant I-18) — EXACT for accelerating lattices."""
    deck, lat = linac_deck(tmp_path)
    r = get_oracle("impactx").run(deck, fmt="impactx", beam=None, workdir=tmp_path / "wd")
    placed = propagate(lat)
    assert len(placed) == r.n
    for p, ein, eout in zip(placed, r.ref_kinetic_eV_in, r.ref_kinetic_eV_out, strict=True):
        assert p.ref_in.kinetic_energy_eV == pytest.approx(ein, rel=1e-9), p.name
        assert p.ref_out.kinetic_energy_eV == pytest.approx(eout, rel=1e-9), p.name
    total = float(r.ref_kinetic_eV_out[-1] - r.ref_kinetic_eV_in[0])
    assert total == pytest.approx(3 * 3e5 * math.cos(math.radians(-30.0)), rel=1e-9)


def test_shortrf_phase_rule_is_pinned(tmp_path):
    """A ShortRF written from phi_s = -30 deg must give +V·cos 30 (and -V·cos 30 at
    phi_s = 150 deg, zero at ±90): the IR's cos convention with 0 = crest maps to
    ImpactX's `phase` in degrees unchanged."""
    from lattix.ir.elements import RFP, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    o = get_oracle("impactx")
    for phi_deg, expected in ((0.0, 1.0), (-30.0, math.cos(math.radians(30.0))),
                              (-90.0, 0.0), (90.0, 0.0), (150.0, -math.cos(math.radians(30)))):
        lat = Lattice.from_sequence("c", [
            RFCavity(name="c", length=0.0,
                     rf=RFP(voltage_V=1e6, phase_rad=math.radians(phi_deg),
                            frequency_Hz=162.5e6))], ref)
        deck = tmp_path / f"c{phi_deg:+.0f}.impactx.in"
        Writer().write(lat, deck, flavor="inputs", strict=True)
        assert f"c.phase = {phi_deg:g}" in deck.read_text()
        r = o.run(deck, fmt="impactx", beam=None, workdir=tmp_path / f"wd{phi_deg:+.0f}")
        gain = float(r.ref_kinetic_eV_out[0] - r.ref_kinetic_eV_in[0])
        assert gain == pytest.approx(expected * 1e6, abs=1e-3), f"phi_s = {phi_deg} deg"


def test_bunching_sign(tmp_path):
    """Invariant I-7: at phi_s = -30 deg a late particle gains more, so R65 < 0 in the
    common basis (z ahead-positive, delta = dp/p)."""
    deck, _ = linac_deck(tmp_path)
    r = get_oracle("impactx").run(deck, fmt="impactx", beam=None,
                                  workdir=tmp_path / "wd").to_common()
    cav = [i for i, n in enumerate(r.names) if n.startswith("cav")]
    assert cav
    got = [r.R_elem[i][5, 4] for i in cav]
    for name, v in zip([r.names[i] for i in cav], got, strict=True):
        assert v < 0, f"{name} does not bunch (R65_common = {v})"
    # MEASURED (impactx 26.01): the bunching weakens as the reference energy grows
    assert got == pytest.approx([-1.6225949559554524, -1.3793370191631427,
                                 -1.1914389007963715], rel=1e-6)


# ------------------------------------------------------------------- coverage
def test_every_writable_kind_survives_the_engine(tmp_path):
    """Everything the writer can emit must actually build inside ImpactX (a typo in a
    keyword would otherwise only surface at run time)."""
    from tests.formats.test_impactx_writer import all_kinds_lattice

    lat = all_kinds_lattice()
    deck = tmp_path / "all.impactx.in"
    rep = Writer().write(lat, deck, flavor="inputs")
    assert not rep.ok, "the all-kinds lattice is expected to have downgrades"
    lat2, _ = Reader().read(deck)
    written = deck.read_text().split("lattice.elements = ", 1)[1].splitlines()[0].split()
    r = get_oracle("impactx").run(deck, fmt="impactx", beam=None, workdir=tmp_path / "wd")
    assert r.names == written
    assert np.all(np.isfinite(r.R_elem))
    # the reader folds the two dipedges back into their Bend, so it sees fewer elements
    assert len(lat2.flatten()) == len(written) - 2
    assert r.total_length == pytest.approx(lat2.total_length, abs=1e-12)


# --------------------------------------------- ImpactX's own vendored corpus
IMPACTX_DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "impactx"


def test_impactx_own_fodo_pair_is_a_three_way_anchor(tmp_path):
    """ImpactX ships the same FODO cell twice — ``input_fodo.in`` and ``fodo.madx``.
    Read each with its own lattix reader, run both through ImpactX, and compare with
    cpymad on the MAD-X deck: three routes, one machine (PLAN §5.2 anti-cancellation)."""
    inputs = IMPACTX_DATA / "input_fodo.in"
    madx = IMPACTX_DATA / "fodo.madx"
    if not (inputs.exists() and madx.exists()):
        pytest.skip("ImpactX sample decks not vendored")
    beam = BeamSpec(species="electron", kinetic_energy_eV=2.0e9)
    o = get_oracle("impactx")
    a = o.run(inputs, fmt="impactx", beam=beam, workdir=tmp_path / "a")
    b = o.run(madx, fmt="madx", beam=beam, workdir=tmp_path / "b")
    m = get_oracle("madx").run(madx, fmt="madx", beam=beam)
    for x, y in ((a, b), (a, m), (b, m)):
        c = compare_pair(x, y)
        assert c.n_shared >= 6, c.row()
        assert c.blocks["T4x4"] < TOL_T4X4, c.row()
        assert c.blocks["disp"] < TOL_DISP, c.row()


@pytest.mark.parametrize("name", ["fodo", "kicker", "hvkicker", "solenoid"])
def test_vendored_impactx_madx_decks_match_cpymad(tmp_path, name):
    """The vendored ImpactX MAD-X samples whose elements the IR maps exactly."""
    deck = IMPACTX_DATA / f"{name}.madx"
    if not deck.exists():
        pytest.skip(f"{deck} not vendored")
    src, _ = MadxReader().read(deck)
    beam = BeamSpec(species=src.reference.species.name,
                    kinetic_energy_eV=src.reference.kinetic_energy_eV)
    a = get_oracle("impactx").run(deck, fmt="madx", beam=beam, workdir=tmp_path / "wd")
    b = get_oracle("madx").run(deck, fmt="madx", beam=beam)
    c = compare_pair(a, b)
    assert c.length_a == pytest.approx(c.length_b, abs=1e-12)
    assert c.blocks["T4x4"] < TOL_T4X4, c.row()
    assert c.blocks["disp"] < TOL_DISP, c.row()


@pytest.mark.parametrize(("name", "n_edges"), [("chicane", 4), ("dogleg", 2)])
def test_vendored_bend_decks_fold_their_dipedges(tmp_path, name, n_edges):
    """ImpactX's chicane/dogleg samples state the DIPEDGE curvature (H) and the SBEND
    ``angle/L`` with 1.4e-6 absolute disagreement (their own file says "TODO make this
    work with inline calculations"), so lattix's MAD-X reader refuses to fold them
    (now folded: ``DIPEDGE_FOLDED``; the 1e-4 curvature tolerance absorbs ImpactX's rounded H)
    so the edge focusing survives the translation and strict mode passes."""
    from lattix.fidelity import TranslationError

    deck = IMPACTX_DATA / f"{name}.madx"
    if not deck.exists():
        pytest.skip(f"{deck} not vendored")
    src, rin = MadxReader().read(deck)
    # the MAD-X reader folds DIPEDGEs whose curvature matches the bend to 1e-4 (ImpactX
    # states H to 1.4e-6): folded, and reported as EQUIVALENT rather than dropped
    assert rin.codes().get("DIPEDGE_FOLDED") == n_edges
    assert rin.codes().get("DIPEDGE_UNFOLDED") is None
    out = tmp_path / f"{name}.impactx.in"
    rout = Writer().write(src, out, flavor="inputs")
    assert rout.codes().get("FOREIGN_DIRECTIVE") is None     # nothing left over once the edges fold
    Writer().write(src, tmp_path / f"{name}_strict.in", flavor="inputs", strict=True)   # and strict is happy
    assert TranslationError  # (kept importable for the docstring's history)
    # what is left still builds and tracks in ImpactX
    r = get_oracle("impactx").run(out, fmt="impactx", beam=None, workdir=tmp_path / "wd")
    assert np.all(np.isfinite(r.R_elem))
    assert r.total_length == pytest.approx(src.total_length, abs=1e-12)
