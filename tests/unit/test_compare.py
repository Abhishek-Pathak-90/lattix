"""Engine-free checks of the cross-engine comparison logic."""
from __future__ import annotations

import numpy as np

from lattix.oracles.base import SPECIES, Basis, OracleResult
from lattix.oracles.basis import drift_common
from lattix.oracles.compare import BLOCKS, compare_all, compare_pair, shared_boundaries

MP = SPECIES["proton"][0]
KE = 800e6


def _res(engine, names, lengths, mats):
    n = len(names)
    return OracleResult(engine=engine, basis=Basis.COMMON, names=names, length=np.array(lengths),
                        s_out=np.cumsum(lengths), R_elem=np.array(mats),
                        ref_kinetic_eV_in=np.full(n, KE), ref_kinetic_eV_out=np.full(n, KE),
                        mass_eV=MP, charge=1)


def _thin(kick):
    K = np.eye(6)
    K[1, 0] = kick
    return K


def test_shared_boundaries_skip_thin_kick_endings_and_take_last_row():
    d = drift_common(1.0, KE, MP)
    a = _res("a", ["d1", "d2"], [1.0, 1.0], [d, d])
    # b splits an edge kick out at s=1 (thin) and has a marker at the end
    b = _res("b", ["d1", "edge", "d2", "m"], [1.0, 0.0, 1.0, 0.0], [d, _thin(0.1), d, np.eye(6)])
    pairs = shared_boundaries(a, b)
    assert pairs == [(1, 3)]          # s=1 excluded (ends in a thin kick in b); s=2 -> b's marker row


def test_compare_pair_identical_lattices_is_exact():
    d = drift_common(1.0, KE, MP)
    a = _res("a", ["d"], [1.0], [d])
    b = _res("b", ["d"], [1.0], [d])
    pc = compare_pair(a, b)
    assert pc.n_shared == 1 and pc.max_rcum_abs == 0.0 and all(v == 0.0 for v in pc.blocks.values())
    assert set(pc.blocks) == set(BLOCKS)


def test_compare_pair_flags_block_difference():
    d = drift_common(1.0, KE, MP)
    d2 = d.copy()
    d2[4, 0] = 0.3      # path-length term only
    pc = compare_pair(_res("a", ["d"], [1.0], [d]), _res("b", ["d"], [1.0], [d2]))
    assert pc.blocks["path"] == 0.3 and pc.blocks["T4x4"] == 0.0 and pc.blocks["disp"] == 0.0


def test_compare_all_pairs():
    d = drift_common(1.0, KE, MP)
    rs = {k: _res(k, ["d"], [1.0], [d]) for k in ("a", "b", "c")}
    assert len(compare_all(rs)) == 3
    assert "no shared" in compare_pair(rs["a"], _res("z", ["d"], [0.5], [d])).notes[0]
