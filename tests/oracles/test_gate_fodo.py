"""Phase-0 gate: the same FODO+bend lattice through every available engine.

MAD-X vs HELIX: transverse 4×4 and dispersion agree; the bend path-length row
is a documented HELIX gap (docs/oracles.md) and is reported, not asserted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.compare import compare_pair

FODO_MADX = Path(__file__).parents[1] / "data" / "public" / "helix" / "fodo.madx"
_HELIX_FODO = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/examples/madx/fodo.madx")
BEAM = BeamSpec("proton", 800e6, 352.21e6)


def _deck() -> Path:
    if FODO_MADX.is_file():
        return FODO_MADX
    if _HELIX_FODO.is_file():
        return _HELIX_FODO
    pytest.skip("fodo.madx not vendored yet and HELIX checkout absent")


@pytest.mark.oracle_madx
@pytest.mark.oracle_helix
def test_madx_vs_helix_transverse_and_dispersion(tmp_path):
    for e in ("madx", "helix"):
        ok, why = get_oracle(e).available()
        if not ok:
            pytest.skip(why)
    deck = _deck()
    a = get_oracle("madx").run(deck, fmt="madx", beam=BEAM, workdir=tmp_path / "madx")
    b = get_oracle("helix").run(deck, fmt="madx", beam=BEAM)
    pc = compare_pair(a, b)
    assert pc.n_shared >= 4
    assert pc.length_a == pytest.approx(6.6) and pc.length_b == pytest.approx(6.6)
    assert pc.blocks["T4x4"] < 1e-6
    assert pc.blocks["disp"] < 1e-7
    assert pc.blocks["E_row"] == 0.0 and pc.blocks["z_col"] == 0.0
    # known HELIX gap: dipole path-length terms (R51, R52) and dispersive R56
    assert pc.blocks["path"] > 0.1, "HELIX now models bend path length — update docs/oracles.md"
