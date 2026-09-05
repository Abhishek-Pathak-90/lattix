# Cross-format battery

`lattix crossval` (library `lattix/crossval.py`, tests `tests/crossval/`) runs **every ordered pair
of formats on every public deck** and checks that a translation reads back to the same physics,
that writing is stable, and — where two engines exist — that both engines agree on the maps.
It is the "both ways" proof for each code: a deck of format A goes to B, B is read again, and
B is also run in B's engine while A runs in A's.

## The three checks

**IR round trip** — read(A) → write(B) → read(B).  The battery computes, for the source and for
the re-read lattice, the integrated physics at every element exit: length, bend angle per
plane, ∫Bₙ·dl for every multipole order (normal and skew, thick and thin combined, rotated by
tilts and body rolls), ∫Bsol·dl, kicks, RF gain and voltage, and the reference kinetic energy.
Positions are compared *after everything that ends at that position* (thin elements share
positions), and both profiles must agree to 1e-9 relative at every position the source has —
unless the writer's ledger declared a LOSSY/DROPPED entry for that element.  Such entries
neutralise exactly the quantities they name (`lattix.crossval.AFFECTS`; a code not listed there
neutralises the whole element), so **a difference the ledger does not explain is a defect**.
Models that legitimately move a boundary a little (a thin gap written as a short cavity)
are listed in `FUZZY_S`; a species the target cannot name suspends the strength comparison.

**Fixed point** — write(B) → read(B) → write(B) again must give the same deck.  Comments,
float formatting (12 significant digits), spacing, case and JSON key order are ignored;
element order, values and names are not.

**Engines** — when `ENGINE_FOR_FORMAT` has an adapter for both formats, the source deck runs
in the source engine and the translation in the target engine; cumulative maps are compared in
the common basis at shared boundaries (`lattix.oracles.compare`).  The tier comes from the
ledger: all EXACT → `|ΔR̂|` ≤ 1e-7 and the reference energy to 1e-9 (p0-following engines);
any EQUIVALENT → 2 % and 0.5 %; LOSSY/DROPPED → reported, not asserted.

## Running it

```
$ lattix crossval                                   # all decks × all pairs, no engines (~4 min)
$ lattix crossval --engines --src tracewin           # engine pairs for one source format
$ lattix crossval --decks fodo,mebt --markdown report.md --json report.json
$ pytest tests/crossval -m "crossval and not slow"   # the same as tests (230 cases)
$ pytest tests/crossval -m "crossval and slow"       # engine pairs (needs the engines)
```

`--markdown` writes one row per case: ledger counts, tier, IR result with the worst
relative difference, fixed-point result, engine metric, and the first problem.

## Derived sources

`lattix crossval --derived` (and the `slow` tests) also use decks lattix itself wrote from five base
decks as sources, so Elegant, ImpactX, IMPACT-Z, xtrack and lattix JSON — which have no public deck
of their own — are exercised as sources too, and against their engines.  With HELIX available this
gives 56 engine pairs on 5 decks (~10 min); SciBmad adds one engine on both ends of every pair where
`julia` with SciBmad is installed (each run starts Julia: ~8 s a case).

## What it found (2026-09-04)

The first runs turned up defects in almost every writer that no per-format test had caught:
TraceWin and Elegant decks lost the reference particle (now carried by a
`lattix: reference` header tag both readers parse); MAD-X decks written with local rigidities
read back with the wrong gradients after cavities (now an `energy_mode` tag the reader
honours); thick collimators and instruments lost their length in TraceWin and ImpactX;
ImpactX multipole keys were written in a case its parser does not read; a bare TraceWin
`BEND` next to an edged one changed its edges on re-read; Bmad literals came back as deferred
MAD-X expressions; a deck variable named `raddeg` was silently replaced by the MAD-X and Bmad
built-in constant; PALS could not carry custom ion species; xtrack JSON lost instruments,
directives, frequency markers, shared definitions and the root line name; MAD8 decks that end
statements with `;` (PyORBIT's) did not parse; zero PALS edge integrals and zero Elegant
collimator limits changed meaning on re-read.  The engine pairs then showed that no code but
TraceWin/HELIX applies the thin-gap RF defocusing (conventions §5.1 — lattix now writes it as a
thin lens), that Elegant's `EMATRIX` needs its diagonal, that FLAME's `tmatrix` is in millimetres,
that ImpactX reads `k_normal` while its Python API takes `K_normal`, and that IMPACT-Z's map
integrator cannot hold a sextupole.  All of these are fixed and pinned by the battery.

The derived-source runs then found the deeper ones (details in `oracles.md`): constant-p0
engines double-corrected the energy of every magnet after a cavity and met every downstream
cavity off its synchronous phase — the `delta` energy mode (conventions §3) fixes both, taking a
DTL from `T4x4 = 11` to `3.5e-10` against xtrack; the xtrack oracle measured maps around the
wrong particle; FLAME maps came back in millimetres; ImpactX re-read a split thick cavity as a
thin one; a reference change vanished through MAD-X/MAD8.  The battery itself learned to carry a
derived deck's own tier, to ignore provenance in the JSON fixed point, and to treat HELIX pairs on
negative-angle bends as report-only (HELIX's bend body has the wrong sign there; TraceWin itself
does not).  What remains is documented engine behaviour: fringe-integral bends between MAD-X and
Bmad, HELIX's bend path-length row and negative-bend body, MAD-X `twiss` at large δ, IMPACT-Z's
short-cavity focusing versus the thin-gap formula; ion species cannot yet be handed to the
engines.
