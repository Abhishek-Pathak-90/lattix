# Corpus notes (Phase 0)

`corpus/sources.yaml` points at the private decks on this Mac; `python -m lattix.corpus
build-manifest --sources corpus/sources.yaml` writes `$LATTIX_CORPUS_DIR/manifest.yaml`
(955 entries on 2026-09-03: 816 TraceWin decks, 56 field maps, 38 TFS, 29 TraceWin
outputs, 9 MAD-X, 4 MAD8, 3 Elegant; 89 redistributable).  `tests/data/public/` holds the
49 vendored public samples with their licences.  `tests/corpus/golden.yaml` pins
structure/optics numbers for the anchor decks (`python -m tests.corpus.goldens --write`).

What the first measurement pass revealed (inputs to Phases 1–2):

| Deck | Finding | Consequence |
|---|---|---|
| `BAL2025V0213.FLAT` (MAD8) | HELIX's MAD8 resolver stops at `SQRT` in a parameter expression | lattix's `Expr` must cover the MAD8 function set (SQRT, ABS, SIN, COS, …) — same whitelist as MAD-X |
| `BTL …/elegant_lattice.lte` | 592 elements parse but total length 0.0 with 224 warnings: lengths are RPN expressions HELIX defaults to 0 | the Elegant reader needs a full RPN evaluator with `% … sto` variables; this deck is the regression anchor |
| `BTL2025v0703.lat` | 1125 elements after LINE expansion, 307.969918 m | matches the HELIX lockstep anchor (307 969.918 mm) |
| `booster_pip2_20250722.seq` | not self-contained (`fmag`/`dmag` classes live in the driver `run_beam_dynamics.madx`) | MAD-X reader must follow `CALL` chains; use the driver as the entry point |
| PIP-II `latf_5/6/7.dat` | begin with >800 lines of stacked TraceWin save headers | sniffer counts code lines, not header lines |
| `mebt+hwr.dat`, `btl.dat` | 1 and 37 parse warnings under HELIX (field-map / downgrade notes) | expected warning classes to be recorded per entry in the manifest |
