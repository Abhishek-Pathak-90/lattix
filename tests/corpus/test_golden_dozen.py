"""Re-measure the golden dozen and compare with golden.yaml (structure + MAD-X optics)."""
from __future__ import annotations

import pytest
import yaml

from tests.corpus.goldens import ENTRIES, GOLDEN, measure, resolve

pytestmark = pytest.mark.corpus

_golden = yaml.safe_load(GOLDEN.read_text()) if GOLDEN.exists() else {}


@pytest.mark.parametrize("key,public,mid,fmt", ENTRIES, ids=[e[0] for e in ENTRIES])
def test_golden(key, public, mid, fmt):
    if key not in _golden:
        pytest.skip("no golden recorded yet (run python -m tests.corpus.goldens --write)")
    p = resolve(public, mid)
    if p is None:
        pytest.skip("deck not available on this machine")
    got = measure(key, p, fmt)
    exp = _golden[key]
    for field, ev in exp.items():
        if field.endswith("_error"):
            continue
        gv = got.get(field)
        if isinstance(ev, float):
            assert gv == pytest.approx(ev, rel=1e-9, abs=1e-9), field
        else:
            assert gv == ev, field
