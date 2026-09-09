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
$ pytest tests/crossval -m "crossval and not slow"   # the same as tests (26 decks × 18 targets)
$ pytest tests/crossval -m "crossval and slow"       # engine pairs (needs the engines)
```

`--markdown` writes one row per case: ledger counts, tier, IR result with the worst
relative difference, fixed-point result, engine metric, and the first problem.

## Derived sources

`lattix crossval --derived` (and the `slow` tests) also use decks lattix itself wrote from five base
decks as sources, so Elegant, ImpactX, IMPACT-Z, xtrack and lattix JSON — which have no public deck
of their own — are exercised as sources too, and against their engines.  With HELIX available this
gives 56 engine pairs on 5 decks (~10 min); SciBmad adds one engine on both ends of every pair where
`julia` with SciBmad is installed (each run starts Julia: ~8 s a case).  LightWin's ADS linac
(`lightwin/example.dat`, 627 elements, 142 one-dimensional RF maps with relative phases) is the
public field-map deck of the battery: it exercises every `FM_TO_CAVITY`/`FM_AS_CAVITY` conversion
against LightWin's envelope.  A writer-only format (MAD-NG)
has no fixed point and is not a battery pair; its tests live with the xtrack ones.  Cheetah's LatticeJSON
is a battery format with Cheetah as its engine, and PyORBIT3's linac XML one with PyORBIT3 as its
engine (its thin gaps carry PyORBIT's own focusing, so pairs with thin cavities are Equivalent tier).  IMPACT-T's
`ImpactT.in` is a battery format with IMPACT-T as its engine: its solenoid tables and RF profiles are
Equivalent tier, and decks with bends are report-only (the engine's dipole has no pole-face focusing).  A derived
IMPACT-T or IMPACT-Z deck gets a directory of its own under `derived/`: their `rfdataN` and `1TN.T7` files are
numbered from 1 per deck and would otherwise overwrite each other.  Ocelot's lattice module is a battery
format with Ocelot as its engine: every public deck is a proton or H⁻ one, so its engine pairs are report only
(`OCELOT_ELECTRON_ONLY`, Ocelot's maps divide by the electron mass); the electron gates live in
`tests/oracles/test_ocelot_adapter.py`.  DYNAC's deck is a battery format with DYNAC as its engine where a
built binary is at hand (never in CI): its `.fields.txt` side file gets a directory of its own like IMPACT-T's,
RF pairs are Equivalent tier (the buncher's mid-gap kick, CAVNUM's own crest) and the exact tier is held at the
6-digit dump precision (`ENGINE_PRECISION`).  Synergia's lattice JSON is a battery format with Synergia as its
engine where the clone is built (never in CI); it keeps one design momentum and scales every strength by
`p_design/p_bunch` itself (the `"constant"` energy mode), and its solenoid body is wrong upstream
(`ff_solenoid` swaps `ks`/`ksl`), so pairs with solenoids are report-only.  OPAL-T has no engine at all: its cases are fixed points and IR
round trips only.  TraceWin decks
are checked with HELIX where it exists and with LightWin's `Envelope3D` otherwise (CI); a pair whose
LightWin run skipped or drift-substituted elements is report-only.

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
negative-angle bends as report-only while HELIX's bend body had the wrong sign there (fixed in
HELIX d3f281a on 2026-09-06; the oracle reports whether its tree carries the fix, and the transverse
block and the dispersion are held to the deck's tier again — `psb.seq` 2.3e-13 / 1.7e-13, at the
equivalent tolerance its fringe-integral bends set).  What
remains is documented engine behaviour: fringe-integral bends between MAD-X and Bmad, a HELIX
before f0c37e5 without a path-length row (R51/R52) or momentum compaction in R56 (5.0e2 / 1.1e3 on
`psb.seq`; the oracle reports `dipole_path_row` and those blocks are compared only on a tree that has
them, 1.9e-16 on `fodo.madx` since 2026-09-08), MAD-X `twiss` at large δ, IMPACT-Z's short-cavity
focusing versus the thin-gap formula; ion species cannot yet be handed to the engines.

Third round (2026-09-06, the PIP-II BTL and two lattix bend decks — `docs/oracles.md` "Bend faces"):
the battery now carries rectangular, negative-angle and vertical bends; IMPACT-Z and PyORBIT were right
on negative bends all along (their vertical bends are written horizontal, LOSSY and named in the
verdict); Synergia's pole-face focusing follows the sign of the charge and its `sbend` has no tilt
(report only); DYNAC's and Synergia's bend maps are the equivalent tier; the IMPACT-T pole-face file of
a negative bend is the proper mirror and its oracle no longer drops the line after a missed dump;
a constant-p0 engine on a line whose momentum grows more than twofold is not run (report only, reason
given). The IR round trip never compares `e1`/`e2` — pole faces are checked by the engines only.
