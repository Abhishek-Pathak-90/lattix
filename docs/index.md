# lattix documentation

lattix translates accelerator lattice files between codes through one intermediate
representation, records what each translation changed, and validates the result by running
the real engines.

| Page | What it covers |
|---|---|
| [tutorial.md](tutorial.md) | Install, convert a deck, read the fidelity summary, use strict mode, call the Python API |
| [conventions.md](conventions.md) | Units, rigidity, energy walk, RF phase per format, bends, kickers, coordinate bases, field maps, names |
| [fidelity.md](fidelity.md) | The ledger: EXACT / EQUIVALENT / LOSSY / DROPPED, strict mode, allow-lists, and the catalogue of every code |
| [crossval.md](crossval.md) | The cross-format battery: every format pair on every deck, IR round trip modulo the ledger, fixed point, engines on both ends |
| [oracles.md](oracles.md) | The engines lattix runs, how each adapter works, and every measured convention with its number |
| [corpus.md](corpus.md) | The private and public deck corpus, the manifest, the golden dozen |
| formats/ | One page per format: [TraceWin](formats/tracewin.md), [MAD-X](formats/madx.md), [MAD8](formats/mad8.md), [Elegant](formats/elegant.md), [Bmad](formats/bmad.md), [PALS](formats/pals.md), [ImpactX](formats/impactx.md), [IMPACT-Z](formats/impactz.md), [FLAME](formats/flame.md), [xtrack](formats/xtrack.md), [MAD-NG](formats/madng.md), [SciBmad](formats/scibmad.md), [Cheetah](formats/cheetah.md), [PyORBIT3](formats/pyorbit.md), [IMPACT-T](formats/impactt.md), [Ocelot](formats/ocelot.md), [Bmad bridge (Astra, GPT, CSRtrack, Merlin++, SLICKTRACK, SAD, SXF, AT)](formats/bmad_bridge.md), [lattix JSON](formats/lattix.md), [HELIX adapter](formats/helix.md) |

The design and phase plan is in [../PLAN.md](../PLAN.md).  `tests/docs/test_docs_consistency.py`
keeps these pages in step with the source: the fidelity catalogue is regenerated from the
code, every registered format must have a page, every element kind and phase conversion
must appear in the conventions, and the version must agree across `pyproject.toml`,
`lattix/_version.py` and `CITATION.cff`.
