"""Basis fingerprints per engine (PLAN §5.1): analytic drift, bunching cavity,
and a golden file that pins the native-basis signs against engine upgrades."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lattix.oracles import get_oracle
from lattix.oracles.fingerprint import DECKS, fingerprint

GOLDEN = Path(__file__).parent / "goldens" / "fingerprints.json"
# analytic-drift tolerance per engine: file-based engines are limited by their output precision
DRIFT_TOL = {"tracewin": 1e-6}   # Transfer_matrix1.dat is written with 7 significant digits
# collection-time markers so `-m "not oracle_tracewin"` etc. deselect correctly
ENGINES = [pytest.param(e, marks=getattr(pytest.mark, f"oracle_{e}"), id=e) for e in DECKS]


def _load_golden() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


@pytest.mark.parametrize("engine", ENGINES)
def test_fingerprint(engine, tmp_path, request):
    try:
        ok, why = get_oracle(engine).available()
    except KeyError as e:
        pytest.skip(f"no adapter registered for {engine!r}: {e}")
    if not ok:
        pytest.skip(f"engine {engine!r} unavailable: {why}")
    fp = fingerprint(engine, tmp_path)

    d = fp["drift"]
    tol = DRIFT_TOL.get(engine, 1e-8)
    assert d["max_abs_err_common"] < tol, (
        f"{engine}: drift map in common basis deviates from analytic; "
        f"R56_native={d['R56_native']:.6e} R56_common={d['R56_common']:.6e} "
        f"expected={d['R56_expected']:.6e} (fix basis._Z_SIGN / scaling, not the test)")

    c = fp["cavity"]
    assert c["R65_common"] < 0, (
        f"{engine}: cavity at φs=-30° must bunch (R65_common<0); "
        f"R65_native={c['R65_native']:.6e} R65_common={c['R65_common']:.6e}")
    if c["follows_p0"]:
        assert c["gain_eV"] == pytest.approx(c["gain_expected_eV"], rel=1e-6), (
            f"{engine}: reference gain {c['gain_eV']:.6e} eV, expected {c['gain_expected_eV']:.6e}")
    else:
        assert c["gain_eV"] == pytest.approx(0.0, abs=1e-6)

    golden = _load_golden()
    if engine in golden:
        g = golden[engine]
        for key in ("R56_native",):
            assert np.sign(d[key]) == np.sign(g["drift"][key]), \
                f"{engine}: drift {key} sign changed vs golden — engine convention flipped?"
        assert np.sign(c["R65_native"]) == np.sign(g["cavity"]["R65_native"]), \
            f"{engine}: cavity R65 sign changed vs golden — engine convention flipped?"
    else:
        golden[engine] = {"drift": d, "cavity": c, "basis": fp["basis"]}
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
