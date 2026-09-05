"""Cross-format battery as tests: every (deck, src → dst) pair, both ways (see lattix.crossval).

The engine-free checks (IR round trip modulo the ledger, write→read→write fixed point) run for
every pair; the engine comparison runs only with ``--engines`` semantics, i.e. when the
``crossval`` *and* the relevant ``oracle_*`` markers are selected and both engines exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lattix import crossval

pytestmark = pytest.mark.crossval

_CASES = [(rel, fmt, opts, dst) for rel, fmt, opts in crossval.DECKS
          for dst in crossval.readable_formats() if dst != fmt]
_IDS = [f"{Path(rel).name}:{fmt}->{dst}" for rel, fmt, opts, dst in _CASES]


@pytest.fixture(scope="session")
def engine_cache() -> dict:
    return {}


def _skip_if_reader_missing(r) -> None:
    """A source format whose reader needs an optional package that is not installed here (cpymad on
    the macOS runner) is a skip, not a failure — the same rule the conftest applies to raised
    MissingDependencyErrors."""
    if r.error.startswith("MissingDependencyError"):
        pytest.skip(r.error)


@pytest.mark.parametrize("rel,fmt,opts,dst", _CASES, ids=_IDS)
def test_ir_roundtrip_and_fixed_point(rel, fmt, opts, dst, tmp_path):
    r = crossval.run_case(crossval.PUBLIC / rel, fmt, dst, tmp_path, engines=False, read_options=opts)
    _skip_if_reader_missing(r)
    assert not r.error, r.error
    assert r.ir_ok, "IR round trip differs beyond the ledger:\n  " + "\n  ".join(r.ir_problems)
    assert r.fixed_ok, f"write→read→write is not a fixed point: {r.fixed_note}"


_ENGINE_CASES = [(rel, fmt, opts, dst) for rel, fmt, opts, dst in _CASES
                 if crossval.ENGINE_FOR_FORMAT.get(fmt) and crossval.ENGINE_FOR_FORMAT.get(dst)]
_ENGINE_IDS = [f"{Path(rel).name}:{fmt}->{dst}" for rel, fmt, opts, dst in _ENGINE_CASES]


def _marks(fmt: str, dst: str):
    return [getattr(pytest.mark, f"oracle_{crossval.ENGINE_FOR_FORMAT[f]}") for f in (fmt, dst)]


@pytest.mark.slow
@pytest.mark.parametrize("rel,fmt,opts,dst", [pytest.param(*c, marks=_marks(c[1], c[3])) for c in _ENGINE_CASES],
                         ids=_ENGINE_IDS)
def test_engines_agree(rel, fmt, opts, dst, tmp_path, engine_cache):
    from lattix.oracles import get_oracle

    for f in (fmt, dst):
        ok, why = get_oracle(crossval.ENGINE_FOR_FORMAT[f]).available()
        if not ok:
            pytest.skip(why)
    r = crossval.run_case(crossval.PUBLIC / rel, fmt, dst, tmp_path, engines=True, engine_cache=engine_cache,
                          read_options=opts)
    _skip_if_reader_missing(r)
    assert not r.error, r.error
    if r.engine_ok is None:
        pytest.skip(r.engine_note)
    assert r.engine_ok, f"tier {r.tier}: {r.engine_note}"


# --- derived sources: decks lattix itself wrote, so every format is also a *source* ------------

@pytest.mark.slow
def test_derived_sources_round_trip(tmp_path):
    """Every base deck written to every format, then each of those as a source to every target."""
    results = crossval.run_matrix([], workdir=tmp_path, engines=False, derived=True)
    assert results, "no derived sources were produced"
    bad = [r for r in results if r.error or r.ir_ok is False or r.fixed_ok is False]
    msg = "\n".join(f"{Path(r.deck).name} {r.src}->{r.dst}: {r.error or (r.ir_problems[:1] or [r.fixed_note])[0]}"
                    for r in bad[:20])
    assert not bad, f"{len(bad)} of {len(results)} derived cases fail:\n{msg}"


@pytest.mark.slow
@pytest.mark.oracle_madx
@pytest.mark.oracle_xtrack
@pytest.mark.oracle_bmad
@pytest.mark.oracle_elegant
@pytest.mark.oracle_impactx
@pytest.mark.oracle_impactz
@pytest.mark.oracle_flame
def test_derived_sources_engines_agree(tmp_path, engine_cache):
    """The derived sources through the engines: Elegant, ImpactX, IMPACT-Z, xtrack and FLAME decks
    written by lattix run in their own engines against every other engine."""
    results = crossval.run_matrix([], workdir=tmp_path, engines=True, derived=True)
    ran = [r for r in results if r.engine_ok is not None]
    if not ran:
        pytest.skip("no engine pair available")
    bad = [r for r in ran if r.engine_ok is False or r.error]
    msg = "\n".join(f"{Path(r.deck).name} {r.src}->{r.dst} [{r.tier}]: {r.error or r.engine_note}" for r in bad[:20])
    assert not bad, f"{len(bad)} of {len(ran)} derived engine cases fail:\n{msg}"
