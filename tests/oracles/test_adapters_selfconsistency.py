"""Adapter self-consistency on tiny decks (each engine against itself)."""
from __future__ import annotations

import numpy as np
import pytest

from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.fingerprint import _SUFFIX, DECKS

BEAM = BeamSpec("proton", 2.1e6, 162.5e6)


def _write(tmp_path, engine, kind):
    fmt, text = DECKS[engine][kind]
    p = tmp_path / f"{kind}{_SUFFIX[fmt]}"
    p.write_text(text)
    return p, fmt


@pytest.mark.parametrize("engine", [pytest.param(e, marks=getattr(pytest.mark, f"oracle_{e}"), id=e)
                                    for e in ("madx", "helix", "tracewin")])
def test_cavity_deck_structure(engine, tmp_path):
    try:
        ok, why = get_oracle(engine).available()
    except KeyError as e:
        pytest.skip(str(e))
    if not ok:
        pytest.skip(why)
    deck, fmt = _write(tmp_path, engine, "cavity")
    r = get_oracle(engine).run(deck, fmt=fmt, beam=BEAM, workdir=tmp_path / "wd")
    assert r.total_length == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(r.s_out) >= -1e-12)
    # energies: monotonic bookkeeping, one gain step for p0-following engines
    assert np.allclose(r.ref_kinetic_eV_in[1:], r.ref_kinetic_eV_out[:-1])
    gain = r.ref_kinetic_eV_out[-1] - r.ref_kinetic_eV_in[0]
    if engine == "madx":
        assert gain == 0.0
    else:
        assert gain == pytest.approx(0.8660254e6, rel=1e-5)
    # cumulative map equals the product of per-element maps (by construction) and the
    # transverse blocks of pure drifts are unit-determinant
    for i in range(r.n):
        if r.length[i] > 0 and abs(r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i]) < 1e-9:
            assert np.linalg.det(r.R_elem[i][:2, :2]) == pytest.approx(1.0, abs=1e-6)
