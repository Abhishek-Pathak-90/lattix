"""Corpus smoke for the TraceWin reader: every deck in the private manifest reads without an
exception, the fidelity codes are collected into a histogram, and the whole pass stays within the
nightly time budget (PLAN §5.4; ``pytest -m corpus -s`` prints the histogram)."""

from __future__ import annotations

import collections
import time
import warnings

import pytest

from lattix.corpus import iter_decks
from lattix.formats.tracewin import read
from lattix.testing import corpus_dir

pytestmark = pytest.mark.corpus

TIME_BUDGET_S = 180.0


@pytest.fixture(scope="module")
def entries():
    root = corpus_dir()
    if root is None or not (root / "manifest.yaml").is_file():
        pytest.skip("LATTIX_CORPUS_DIR is not set or has no manifest.yaml")
    decks = list(iter_decks(root, fmt="tracewin"))
    if not decks:
        pytest.skip("no TraceWin decks in the corpus manifest")
    return decks


def test_every_tracewin_deck_reads(entries):
    hist: collections.Counter = collections.Counter()
    per_source: collections.Counter = collections.Counter()
    failures = []
    t0 = time.perf_counter()
    for e in entries:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lat, rep = read(e["path"])
        except Exception as exc:  # noqa: BLE001 — every failure is reported at the end
            failures.append((e["id"], f"{type(exc).__name__}: {exc}"))
            continue
        per_source[e["source"]] += 1
        assert lat.use is not None and lat.total_length >= 0.0
        for code, n in rep.codes().items():
            hist[code] += n
    elapsed = time.perf_counter() - t0
    print(f"\n{len(entries)} TraceWin decks in {elapsed:.1f} s; per source {dict(per_source)}")
    for code, n in hist.most_common():
        print(f"  {code:32s} {n}")
    assert not failures, failures[:10]
    assert elapsed < TIME_BUDGET_S, f"corpus pass took {elapsed:.0f} s (> {TIME_BUDGET_S:.0f} s)"
