# lattix — accelerator lattice translator

Translate a lattice deck from one accelerator code's format to another's
(TraceWin `.dat`, MAD-X, MAD8 flat, Elegant `.lte`, Bmad, PALS; later ImpactX,
IMPACT-Z, IMPACT-T, FLAME, xtrack, SciBmad, Cheetah LatticeJSON, PyORBIT3 linac XML, Ocelot lattice modules, MAD-NG write-only; Astra, GPT, CSRtrack, Merlin++, SLICKTRACK, SAD, SXF and Accelerator Toolkit through Bmad's converters) through a code-neutral intermediate representation, with
a fidelity report on every conversion and validation against the real engines.

**Status: Phase 5 in progress (0.1.0)** — SciBmad, the xsuite completion (every xtrack element class, knobs, environments, MAD-NG output) LightWin as a second TraceWin-semantics engine, Cheetah (LatticeJSON format + engine), PyORBIT3 (linac XML format + engine), IMPACT-T (`ImpactT.in` format + engine), Ocelot (lattice-module format + engine, electron machines) and the Bmad bridge (Astra, GPT, CSRtrack, Merlin++, SLICKTRACK, SAD, SXF, AT through Bmad's own converters) have landed; documentation, CI and a local release build are in place; HELIX GUI/MCP integration is deferred. Phase 3 delivered — readers and writers for TraceWin, MAD-X, MAD8 flat, Elegant, Bmad,
PALS, ImpactX, IMPACT-Z, FLAME, xtrack and SciBmad, TraceWin field maps integrated into equivalent cavities and
hard-edge magnets, the HELIX adapter, the fidelity report and the CLI — each validated against its real
engine (see `PLAN.md`, `docs/oracles.md`, `docs/corpus.md`). Still open: the HELIX GUI/MCP integration (deferred), the nightly
self-hosted runner registration, and a PyPI upload (the 0.1.0 wheel and sdist are built locally).

## What works today

```bash
# translate a deck (fidelity summary on stderr; --strict fails on the first LOSSY/DROPPED element)
PYTHONPATH=. python3 -m lattix.cli convert examples/mebt.dat mebt.madx --read-option species=h- \
    --read-option kinetic_energy_eV=2.1e6 --write-option energy_mode=constant --report mebt.fidelity.json
PYTHONPATH=. python3 -m lattix.cli inspect mebt.dat --read-option species=h- --elements
PYTHONPATH=. python3 -m lattix.cli convert BTL2025v0703.lat btl.bmad          # MAD8 -> Bmad
PYTHONPATH=. python3 -m lattix.cli convert fodo.madx fodo.lte                  # MAD-X -> Elegant (species-aware RF phase)
PYTHONPATH=. python3 -m lattix.cli convert fodo.madx fodo.pals.yaml            # -> PALS (loads in ImpactX, validates with pals-schema)
PYTHONPATH=. python3 -m lattix.cli convert mebt+hwr.dat mebt_hwr.bmad --read-option species=h- \
    --read-option kinetic_energy_eV=2.1e6      # field maps -> lcavities with HELIX-equivalent gains
PYTHONPATH=. python3 -m lattix.cli convert fodo.madx ImpactZ.in --to impactz   # -> IMPACT-Z deck
PYTHONPATH=. python3 -m lattix.cli convert fodo.madx fodo.cheetah.json         # -> Cheetah LatticeJSON
PYTHONPATH=. python3 -m lattix.cli convert mebt.dat mebt.pyorbit.xml --read-option species=h- \
    --read-option kinetic_energy_eV=2.1e6      # -> PyORBIT3 linac XML (one <Cavity> per gap)
PYTHONPATH=. python3 -m lattix.cli convert fodo.madx fodo.madng --to madng     # -> MAD-NG (through xtrack)

# which engines this machine can run (MAD-X via cpymad, xtrack, Bmad/Tao, elegant, ImpactX, IMPACT-Z, FLAME,
# SciBmad, LightWin, Cheetah, PyORBIT3, HELIX, TraceWin)
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

## Documentation

* [docs/index.md](docs/index.md) — map of the documentation
* [docs/tutorial.md](docs/tutorial.md) — install, convert, read the fidelity summary, strict mode, Python API
* [docs/conventions.md](docs/conventions.md) — units, rigidity, energy walk, RF phase per format, bends, bases
* [docs/fidelity.md](docs/fidelity.md) — the ledger and the catalogue of every fidelity code
* [docs/crossval.md](docs/crossval.md) — the cross-format battery (every pair, both ways, engines on both ends)
* [docs/oracles.md](docs/oracles.md) — engines, adapters and every measured convention
* [docs/formats/](docs/formats/) — one page per format
* [CITATION.cff](CITATION.cff) — how to cite
