# Contributing

Thank you for looking. lattix is small enough that the whole of it can be understood in an
afternoon, and the rules below are what keep it that way.

## Setting up

```bash
git clone https://github.com/Accel-Toolkit/lattix.git
cd lattix
pip install -e ".[dev]"
pytest -m "not corpus and not slow and not crossval"     # the engine-free suite, about ninety seconds
```

The engine-free suite is what continuous integration runs on every push, and it must pass before
anything else is worth discussing. Tests that need a simulation code carry an `oracle_<engine>`
marker and skip when that engine is not installed, so a clean machine passes the whole selection.

## Engines

The engines are found through environment variables, never through a fixed path. Set the ones you
have: `HELIX_ROOT`, `TRACEWIN_EXE`, `LATTIX_DYNAC_EXE`, `LATTIX_SYNERGIA_ROOT`, `LATTIX_CODES_DIR`
(a local mirror of the upstream repositories some tests compare against), and the ones listed in
`docs/oracles.md`. `lattix oracles` reports which engines the current machine can run.

`environment-ci.yml` builds a conda environment with every freely installable engine, and
`environment-bmad.yml` builds the Bmad and Tao environment the Bmad adapter runs out of process.

## The ledger is the contract

Every element that a writer cannot carry across exactly must record a ledger entry with a named
code, a class (`EQUIVALENT`, `LOSSY` or `DROPPED`) and a message that says what changed and why.
Nothing is ever dropped silently. New codes are picked up by `python -m lattix.fidelity_catalog
--markdown`, which regenerates the catalogue block in `docs/fidelity.md`; a test fails if the
documentation and the code disagree.

## Golden files and the battery

Writer tests compare against golden files under `tests/golden/`. When an intentional change moves
a golden, regenerate it with `LATTIX_UPDATE_GOLDEN=1 pytest <that test>` and commit the result
with an explanation of why the output changed. The version banner is neutralised, so a release
never moves a golden on its own.

The cross-format battery, `pytest tests/crossval -m crossval`, writes every registered deck to
every format that can hold it, reads it back, and runs both ends through their engines. It is the
acceptance gate for a new format or a changed convention, and it takes a while.

## Engine bugs go upstream

When the battery shows an engine disagreeing with the others for a reason that lives in the
engine, lattix records the measurement in `docs/oracles.md`, caps the pair's tier with a rule in
`lattix/crossval.py::engine_verdict` that names the limit, and the bug is reported to the engine's
maintainers. lattix does not compensate for another code's bug in its own writer.

## Style

`ruff check lattix tests` must pass; lines are at most 120 characters. Prefer a short, specific
docstring that says why over a comment that says what. Measured conventions are cited with the
date and the engine version they were measured on.

## Private data

Nothing from a private accelerator project enters the repository or continuous integration. Public
sample decks live in `tests/data/public/` with their upstream licence beside them; anything else
is referenced through `LATTIX_CORPUS_DIR` and never committed.
