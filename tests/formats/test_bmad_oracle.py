"""Bmad writer pinned against real engines (PLAN §5.6 gate A2, §6 task 2.2).

Three independent checks, none of them a round trip:

1. ``fodo.madx`` → IR → ``.bmad`` compared by **Tao** against **MAD-X** running the
   original deck (Exact tier: transverse 4×4, dispersion and path-length blocks
   below 1e-8, equal length, survey end point below 1e-9);
2. an **accelerating** IR linac (three thin gaps, proton at 2.1 MeV) written to
   Bmad: Tao's per-element reference energies must equal the IR's own
   :func:`lattix.ir.walk.propagate` walk, and the transverse maps must equal the
   HELIX oracle running the same IR exported to TraceWin;
3. a cross-check against **Bmad's own converter** ``bmad_to_mad_sad_elegant
   -madx``: the k1/angle/e1/e2/l it reads out of our deck must equal the numbers
   we wrote.

Measured 2026-09-03 (Bmad 20260828.0, Tao, MAD-X 5.09.03, HELIX linac_gen 1.9.1).
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lattix.formats.base import read
from lattix.formats.bmad import Writer
from lattix.ir.walk import propagate
from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.basis import rescale_to_constant_p0
from lattix.oracles.compare import compare_pair
from lattix.testing import needs
from tests.formats.test_bmad_writer import linac_lattice

pytestmark = pytest.mark.oracle_bmad

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
#: Bmad's own MAD-X/MAD-8/Elegant converter, shipped in the ``bmad`` conda env.
BMAD_TO_MAD = Path("/Users/abhishekpathak/anaconda3/envs/bmad/bin/bmad_to_mad_sad_elegant")


# ---------------------------------------------------------------- gate A2
@needs("bmad")
@needs("madx")
@pytest.mark.oracle_madx
def test_fodo_madx_to_bmad_matches_madx(tmp_path):
    lat, _ = read(DATA / "fodo.madx")
    out = tmp_path / "fodo.bmad"
    rep = Writer().write(lat, out, strict=True)
    assert rep.ok

    beam = BeamSpec(species="proton", kinetic_energy_eV=lat.reference.kinetic_energy_eV,
                    betx=10.0, bety=10.0)
    rm = get_oracle("madx").run(DATA / "fodo.madx", beam=beam)
    rb = get_oracle("bmad").run(out, beam=beam)
    c = compare_pair(rm, rb)

    assert c.n_shared >= 8, c.row()
    assert c.length_a == pytest.approx(c.length_b, abs=1e-12)
    assert c.length_a == pytest.approx(6.6, abs=1e-12)
    assert c.blocks["T4x4"] < 1e-8, c.row()
    assert c.blocks["disp"] < 1e-8, c.row()
    assert c.blocks["path"] < 1e-8, c.row()
    assert c.survey_end_abs is not None and c.survey_end_abs < 1e-9, c.row()
    # MAD-X keeps p0 constant and this deck has no cavity, so the energies must agree too
    assert c.energy_rel < 1e-12, c.row()


# ---------------------------------------------------------------- linac
@needs("bmad")
def test_linac_reference_energy_follows_the_ir_walk(tmp_path):
    """Bmad's ``lcavity`` moves p0, so Bmad is an EXACT target for a linac."""
    lat = linac_lattice()
    out = tmp_path / "linac.bmad"
    Writer().write(lat, out)

    beam = BeamSpec(species="proton", kinetic_energy_eV=lat.reference.kinetic_energy_eV,
                    betx=10.0, bety=10.0, frequency_Hz=162.5e6)
    rb = get_oracle("bmad").run(out, beam=beam)
    tao = {n.lower(): rb.ref_kinetic_eV_out[i] for i, n in enumerate(rb.names)}

    worst = 0.0
    for p in propagate(lat):
        ir = p.ref_out.kinetic_energy_eV
        worst = max(worst, abs(ir - tao[p.element.name.lower()]) / abs(ir))
    assert worst < 1e-9, f"reference energy vs Tao: {worst:.3e}"
    # three 300 kV gaps at -30 deg
    total = 3 * 3e5 * math.cos(-math.pi / 6)
    assert rb.ref_kinetic_eV_out[-1] == pytest.approx(2.1e6 + total, rel=1e-12)


@needs("bmad")
@needs("helix")
def test_linac_transverse_maps_match_helix(tmp_path):
    """Same IR through Bmad and through TraceWin/HELIX.

    Every element agrees to 1e-7 **except** the thin gaps: a zero-length Bmad
    ``lcavity`` applies no transverse (pondermotive) RF kick while a TraceWin/HELIX
    ``GAP`` does, so ``R21 = R43`` differ by exactly that term.  The writer records
    it as ``LOSSY:THIN_CAVITY_NO_RF_FOCUSING``; everything else in the cavity map
    (adiabatic damping, the whole longitudinal block, the energy gain) agrees to
    1e-9.  Measured 2026-09-03: non-cavity max |ΔR| = 1.7e-12, cavity R21 =
    0.7617 / 0.6511 / 0.5649.
    """
    from lattix.formats.tracewin import Writer as TraceWinWriter

    lat = linac_lattice()
    bmad_deck = tmp_path / "linac.bmad"
    tw_deck = tmp_path / "linac.dat"
    rep = Writer().write(lat, bmad_deck)
    TraceWinWriter().write(lat, tw_deck)
    assert "THIN_CAVITY_NO_RF_FOCUSING" in rep.codes()

    beam = BeamSpec(species="proton", kinetic_energy_eV=2.1e6, betx=10.0, bety=10.0,
                    frequency_Hz=162.5e6)
    rb = get_oracle("bmad").run(bmad_deck, beam=beam).to_common()
    rh = get_oracle("helix").run(tw_deck, beam=beam).to_common()
    ib = {n.lower(): i for i, n in enumerate(rb.names)}
    ih = list(rh.names)

    pairs = [("q1", "QUAD_001"), ("da1", "DRIFT_001"), ("db1", "DRIFT_002"),
             ("q2", "QUAD_002"), ("da2", "DRIFT_003"), ("db2", "DRIFT_004"),
             ("q3", "QUAD_003"), ("da3", "DRIFT_005"), ("db3", "DRIFT_006")]
    worst = max(float(np.max(np.abs(rb.R_elem[ib[nb]] - rh.R_elem[ih.index(nh)])))
                for nb, nh in pairs)
    assert worst < 1e-7, f"non-cavity per-element max |dR| = {worst:.3e}"

    for nb, nh, expect in (("cav1", "GAP_001", 0.761657299),
                           ("cav2", "GAP_002", 0.651070730),
                           ("cav3", "GAP_003", 0.564853991)):
        a = rb.R_elem[ib[nb]].copy()
        b = rh.R_elem[ih.index(nh)].copy()
        assert b[1, 0] == pytest.approx(expect, rel=1e-6)      # HELIX's radial RF kick
        assert b[3, 2] == pytest.approx(expect, rel=1e-6)
        assert a[1, 0] == 0.0 and a[3, 2] == 0.0               # Bmad's thin gap has none
        b[1, 0], b[3, 2] = a[1, 0], a[3, 2]
        assert float(np.max(np.abs(a - b))) < 1e-7, f"{nb} vs {nh} outside the RF-kick term"

    # the reference energies (and therefore the adiabatic damping) agree exactly
    assert float(np.max(np.abs(rb.ref_kinetic_eV_out[[ib[n] for n, _ in pairs]]
                               - rh.ref_kinetic_eV_out[[ih.index(m) for _, m in pairs]])
                        / rb.ref_kinetic_eV_out[[ib[n] for n, _ in pairs]])) < 1e-9


@needs("bmad")
@needs("helix")
def test_linac_damping_matches_the_momentum_ratio(tmp_path):
    """PLAN I-5: on a p0-following engine, det(2x2) = p_in/p_out per element."""
    lat = linac_lattice()
    out = tmp_path / "linac.bmad"
    Writer().write(lat, out)
    rb = get_oracle("bmad").run(out, beam=BeamSpec(species="proton", kinetic_energy_eV=2.1e6,
                                                   betx=10.0, bety=10.0)).to_common()
    mass = rb.mass_eV
    for i, name in enumerate(rb.names):
        if not name.lower().startswith("cav"):
            continue
        p_in = math.sqrt((rb.ref_kinetic_eV_in[i] + mass) ** 2 - mass ** 2)
        p_out = math.sqrt((rb.ref_kinetic_eV_out[i] + mass) ** 2 - mass ** 2)
        det = float(np.linalg.det(rb.R_elem[i][:2, :2]))
        assert det == pytest.approx(p_in / p_out, rel=1e-9), name
        # the rescaled map is then the identity in x (a thin gap has no focusing)
        r = rescale_to_constant_p0(rb.R_elem[i], p_in, p_out)
        assert float(np.max(np.abs(r[:4, :4] - np.eye(4)))) < 1e-9


# ---------------------------------------------------------------- converters
_MADX_DEF = re.compile(r"^\s*([A-Za-z][\w.$]*)\s*:\s*(\w+)\s*,?\s*(.*?);\s*$",
                       re.MULTILINE | re.DOTALL)


def _parse_madx_defs(text: str) -> dict[str, dict[str, float]]:
    """Numeric attributes of every ``name: type, a=1, b=2;`` in a MAD-X deck."""
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"\n\s+", " ", text)
    out: dict[str, dict[str, float]] = {}
    for name, base, body in _MADX_DEF.findall(text):
        attrs = {"__type__": base.lower()}
        for part in body.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                attrs[k.strip().lower()] = float(v.strip())
            except ValueError:
                continue
        out[name.lower()] = attrs
    return out


@needs("bmad")
@pytest.mark.oracle_madx
@pytest.mark.skipif(not BMAD_TO_MAD.is_file(), reason=f"{BMAD_TO_MAD} not installed")
def test_cross_check_against_bmad_own_madx_converter(tmp_path):
    """Bmad's own reader must see exactly the numbers our writer put in the deck."""
    lat, _ = read(DATA / "fodo.madx")
    deck = tmp_path / "fodo.bmad"
    Writer().write(lat, deck)

    proc = subprocess.run([str(BMAD_TO_MAD), "-madx", "-force", deck.name],
                          cwd=tmp_path, capture_output=True, text=True, check=False)
    produced = tmp_path / "fodo.madx"
    assert produced.is_file(), f"converter failed:\n{proc.stdout}\n{proc.stderr}"
    defs = _parse_madx_defs(produced.read_text())

    walk = {p.element.name: p for p in propagate(lat)}
    checked = 0
    for name, p in walk.items():
        got = defs.get(name.lower())
        if got is None:
            continue
        el = p.element
        brho = p.ref_in.brho_signed
        if el.kind == "Quadrupole":
            assert got["k1"] == pytest.approx(el.multipole.Bn[1] / brho, abs=1e-12), name
            assert got["l"] == pytest.approx(el.length, abs=1e-12), name
            checked += 1
        elif el.kind == "Bend":
            assert got["angle"] == pytest.approx(el.bend.angle, abs=1e-12), name
            assert got["e1"] == pytest.approx(el.bend.e1, abs=1e-12), name
            assert got["e2"] == pytest.approx(el.bend.e2, abs=1e-12), name
            assert got["l"] == pytest.approx(el.length, abs=1e-12), name
            checked += 1
        elif el.kind == "Drift":
            assert got["l"] == pytest.approx(el.length, abs=1e-12), name
            checked += 1
    assert checked >= 5, f"nothing cross-checked; parsed {sorted(defs)}"

    # and the converted deck must still be the same machine to MAD-X itself
    from cpymad.madx import Madx  # noqa: F401

    stripped = re.sub(r"//.*", "!", produced.read_text()).split("initial: beta0")[0]
    round_trip = tmp_path / "converted.madx"
    round_trip.write_text(stripped)
    beam = BeamSpec(species="proton", kinetic_energy_eV=lat.reference.kinetic_energy_eV,
                    betx=10.0, bety=10.0)
    a = get_oracle("madx").run(DATA / "fodo.madx", beam=beam)
    b = get_oracle("madx").run(round_trip, beam=beam)
    c = compare_pair(a, b)
    assert c.blocks["T4x4"] < 1e-12, c.row()
    assert c.length_a == pytest.approx(c.length_b, abs=1e-12)


@needs("bmad")
@pytest.mark.skipif(not BMAD_TO_MAD.is_file(), reason=f"{BMAD_TO_MAD} not installed")
def test_bmad_own_converter_flips_the_skew_multipole_sign(tmp_path):
    """A finding, pinned so an upstream fix is noticed.

    ``bmad_to_mad_sad_elegant -madx`` writes ``ksl = -n!*a_n`` (see Bmad's
    ``write_lattice_in_mad_format.f90``), but MAD-X's own ``ksl`` has the *same*
    sign as Bmad's ``a_n``/``k{n}sl``: measured with cpymad and Tao, a Bmad
    ``ab_multipole, a1 = 0.03`` and a MAD-X ``multipole, ksl = {0, 0.03}`` give
    an identical map, so the converted deck's skew quadrupole is reversed.  lattix
    writes ``k{n}sl = ksl`` (no flip), which is the mapping the engines agree on.
    """
    deck = tmp_path / "skew.bmad"
    deck.write_text("parameter[particle] = proton\nparameter[e_tot] = 1738272000\n"
                    "parameter[geometry] = open\nbeginning[beta_a] = 10\n"
                    "beginning[beta_b] = 10\n"
                    "m1: ab_multipole, a1 = 0.03\nd: drift, l = 0.1\n"
                    "lat: line = (d, m1, d)\nuse, lat\n")
    subprocess.run([str(BMAD_TO_MAD), "-madx", "-force", deck.name],
                   cwd=tmp_path, capture_output=True, text=True, check=False)
    produced = tmp_path / "skew.madx"
    assert produced.is_file()
    ksl = re.search(r"ksl\s*=\s*\{([^}]*)\}", produced.read_text())
    assert ksl is not None
    values = [float(v) for v in ksl.group(1).split(",")]
    assert values[1] == pytest.approx(-0.03), "upstream converter no longer flips ksl"

    # lattix keeps the sign the engines agree on
    lat, _ = read(deck)
    assert lat.elements["m1"].multipole.BsL[1] == \
        pytest.approx(0.03 * lat.reference.brho_signed)
    ours = tmp_path / "ours.bmad"
    Writer().write(lat, ours)
    assert "k1sl = 0.03" in ours.read_text()


@needs("bmad")
def test_written_deck_is_accepted_by_tao_for_every_golden(tmp_path):
    """Every golden must be a lattice Bmad actually loads, not merely plausible text."""
    golden = Path(__file__).resolve().parents[1] / "golden" / "bmad"
    oracle = get_oracle("bmad")
    for deck in sorted(golden.glob("*.bmad")):
        res = oracle.run(deck, beam=BeamSpec(species="proton", kinetic_energy_eV=1e8,
                                             betx=10.0, bety=10.0),
                         workdir=tmp_path / deck.stem)
        assert len(res.names) > 0, deck.name
        assert np.all(np.isfinite(res.R_elem)), deck.name
