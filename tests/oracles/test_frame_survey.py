"""The floor frames against the engines that survey.  MAD-X through cpymad and xtrack pin the
positions and the three MAD-X angles at every element exit on the bend decks, the rings and the
patches deck (yaw, pitch, roll, translation); these tests are what decide the sign conventions
of ``lattix.ir.frames`` and of the MAD-X reader's frame cards."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix.formats import read
from lattix.ir import frame_survey
from lattix.oracles import BeamSpec, get_oracle

PUBLIC = Path(__file__).resolve().parents[1] / "data" / "public"

#: (deck, beam for the engine or None when the deck carries its own, read options)
DECKS = [
    ("lattix/rect_bends.madx", None, {"frequency_Hz": 352.21e6}),
    ("lattix/vertical_bends.madx", None, {"frequency_Hz": 352.21e6}),
    ("lattix/patches.madx", None, {"frequency_Hz": 352.21e6}),
    ("xtrack/psb.seq", BeamSpec("proton", 160e6), {"species": "proton", "kinetic_energy_eV": 160e6}),
    ("xtrack/elena.seq", BeamSpec("proton", 5.3e6), {"species": "proton", "kinetic_energy_eV": 5.3e6}),
]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _check(engine: str, rel: str, beam, opts: dict, workdir: Path) -> None:
    ok, why = get_oracle(engine).available()
    if not ok:
        pytest.skip(why)
    lat, _ = read(PUBLIC / rel, "madx", **opts)
    frames = frame_survey(lat.flatten())
    res = get_oracle(engine).run(PUBLIC / rel, fmt="madx", beam=beam, workdir=workdir)
    sv = np.asarray(res.meta["survey6"], dtype=float)
    assert sv.shape == (len(res.s_out), 6)
    by_key: dict[tuple[str, float], list] = {}
    by_s: dict[float, list] = {}
    for f in frames:
        by_key.setdefault((f.name.lower(), round(f.s_out, 9)), []).append(f)
        by_s.setdefault(round(f.s_out, 9), []).append(f)
    checked = 0
    for name, s, row in zip(res.names, res.s_out, sv, strict=True):
        cands = by_key.get((name.lower(), round(float(s), 9))) or by_s.get(round(float(s), 9))
        assert cands, f"{rel}: no lattix element {name!r} ending at s={s}"
        f = cands[-1]
        np.testing.assert_allclose(f.exit.V, row[:3], atol=1e-9, err_msg=f"{rel} {engine} {name}: position")
        theta, phi, psi = f.angles("exit")
        assert _wrap(theta - row[3]) == pytest.approx(0.0, abs=1e-10), (rel, engine, name, "theta", theta, row[3])
        assert phi == pytest.approx(row[4], abs=1e-10), (rel, engine, name, "phi", phi, row[4])
        assert _wrap(psi - row[5]) == pytest.approx(0.0, abs=1e-10), (rel, engine, name, "psi", psi, row[5])
        checked += 1
    assert checked >= 5


@pytest.mark.oracle_madx
@pytest.mark.parametrize("rel,beam,opts", DECKS, ids=[d[0] for d in DECKS])
def test_frames_match_madx_survey(rel, beam, opts, tmp_path):
    _check("madx", rel, beam, opts, tmp_path)


@pytest.mark.oracle_xtrack
@pytest.mark.parametrize("rel,beam,opts", DECKS, ids=[d[0] for d in DECKS])
def test_frames_match_xtrack_survey(rel, beam, opts, tmp_path):
    _check("xtrack", rel, beam, opts, tmp_path)


BMAD_DECKS = ["helix/fodo.bmad", "lattix/misaligned.bmad"]


def _assert_frame(frame, angles, row, label) -> None:
    np.testing.assert_allclose(frame.V, row[:3], atol=1e-9, err_msg=f"{label}: position")
    theta, phi, psi = angles
    assert _wrap(theta - row[3]) == pytest.approx(0.0, abs=1e-10), (label, "theta", theta, row[3])
    assert phi == pytest.approx(row[4], abs=1e-10), (label, "phi", phi, row[4])
    assert _wrap(psi - row[5]) == pytest.approx(0.0, abs=1e-10), (label, "psi", psi, row[5])


@pytest.mark.oracle_bmad
@pytest.mark.parametrize("rel", BMAD_DECKS)
def test_frames_match_bmad_floor_positions(rel, tmp_path):
    """Bmad's floor positions at every exit and centre, and the *Actual* position of a misaligned
    body at its centre, against the exit, centre and body frames."""
    ok, why = get_oracle("bmad").available()
    if not ok:
        pytest.skip(why)
    lat, _ = read(PUBLIC / rel, "bmad")
    frames = frame_survey(lat.flatten())
    res = get_oracle("bmad").run(PUBLIC / rel, fmt="bmad", workdir=tmp_path)
    exits = np.asarray(res.meta["survey6"], dtype=float)
    centres = np.asarray(res.meta["centre6"], dtype=float)
    bodies = np.asarray(res.meta["body6"], dtype=float)
    by_key = {}
    for f in frames:
        by_key.setdefault((f.name.lower(), round(f.s_out, 9)), []).append(f)
    checked = shifted = rolled = 0
    for name, s, e_row, c_row, b_row in zip(res.names, res.s_out, exits, centres, bodies, strict=True):
        cands = by_key.get((name.lower(), round(float(s), 9)))
        if not cands:
            continue                                   # Bmad's own markers (BEGINNING, END)
        f = cands[-1]
        _assert_frame(f.exit, f.angles("exit"), e_row, f"{rel} {name} exit")
        _assert_frame(f.centre, f.angles("centre"), c_row, f"{rel} {name} centre")
        _assert_frame(f.body, f.angles("body"), b_row, f"{rel} {name} body")
        checked += 1
        shifted += f.shifted
        rolled += f.roll != 0.0
    assert checked >= 5
    if rel == "lattix/misaligned.bmad":
        assert shifted >= 2 and rolled >= 2        # the offset quad and the rolled bend; the two skew quads
