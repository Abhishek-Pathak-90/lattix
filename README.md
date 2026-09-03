# lattix — accelerator lattice translator

Translate a lattice deck from one accelerator code's format to another's
(TraceWin `.dat`, MAD-X, MAD8 flat, Elegant `.lte`, Bmad, PALS; later ImpactX,
IMPACT-Z, FLAME, xtrack) through a code-neutral intermediate representation, with
a fidelity report on every conversion and validation against the real engines.

**Status: Phase 0 complete** — the engine harness, corpus tooling and CI scaffold
exist; the IR, readers and writers arrive in Phase 1 (see `PLAN.md`, `docs/oracles.md`,
`docs/corpus.md`).

## What works today

```bash
# which engines this machine can run (MAD-X via cpymad, xtrack, Bmad/Tao, elegant, HELIX, TraceWin)
PYTHONPATH=. python3 -m lattix.cli oracles

# measure each engine's longitudinal conventions (1 m drift, thin cavity at φs = −30°)
PYTHONPATH=. python3 -m lattix.cli fingerprint

# run the SAME lattice through several engines and compare cumulative maps, block by block
PYTHONPATH=. python3 -m lattix.cli validate \
    --deck madx=tests/data/public/helix/fodo.madx --deck bmad=tests/data/public/helix/fodo.bmad \
    --oracles madx,xtrack,bmad,helix --ke 800e6 --freq 352.21e6

# corpus manifest of the private decks (never committed): sha256 + provenance + format sniff
LATTIX_CORPUS_DIR=~/lattix_corpus PYTHONPATH=. python3 -m lattix.corpus build-manifest --sources corpus/sources.yaml
```

Result of the Phase-0 gate (FODO + bends, 800 MeV proton): MAD-X, xtrack and Bmad agree
to ≤ 6e-9 on cumulative maps; HELIX agrees in the transverse block (1.3e-7) and dispersion
(7e-9) and differs only in the bend path-length row it does not model.

## Environments

* base env: `cpymad`, `xtrack`, HELIX (`HELIX_ROOT`, default local checkout), TraceWin
  (`TRACEWIN_EXE`, default `TraceWin.app`; the local build is a 20-element trial).
* `environment-ci.yml` → env `lattix`: elegant, impactx, impact-z, cpymad, xtrack, pysdds, pals-schema, ruff.
* `environment-bmad.yml` (or the existing env `bmad`): bmad, pytao — used out-of-process by the Bmad adapter.

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests/unit tests/corpus -q          # engine-free (~7 s)
PYTHONPATH=. python3 -m pytest tests/oracles -q                    # every engine available here
PYTHONPATH=. python3 -m pytest tests -m "not oracle_tracewin" -q   # what CI runs
```

Markers: `oracle_madx`, `oracle_xtrack`, `oracle_bmad`, `oracle_elegant`, `oracle_impactx`,
`oracle_impactz`, `oracle_flame`, `oracle_helix`, `oracle_tracewin` (local only), `corpus`, `slow`.
