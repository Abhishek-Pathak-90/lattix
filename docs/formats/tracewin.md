# TraceWin `.dat`

TraceWin (CEA) describes a linac as one card per line: `NAME: CARD value …` with positional
operands in mm, degrees, MHz and volts, plus command cards (`FREQ`, `SET_SYNC_PHASE`,
`LATTICE`, `ADJUST_*`, `ERROR_*`) that change the state for the cards that follow.  lattix
registers the suffix `.dat` (and sniffs `.lat` files written in the TraceWin dialect); both a
reader and a writer exist, ported from HELIX's `tracewin_parser.py` / `tracewin_writer.py`
and then pinned against the TraceWin binary.

## Reading

`lattix/formats/tracewin/reader.py` keeps the HELIX label grammar (`NAME : CARD`, `NAME:CARD`,
standalone `NAME :`), latin-1 decoding, `;` comments including the `;@LG`, `; HELIX_FOIL` and
`; HELIX_SC_GRID` comment cards, and the `FREQ` / `FIELD_MAP_PATH` / `SET_SYNC_PHASE` state
machine.  A trailing `;` comment on an element card travels with the element (`meta["comment"]`)
and the writer puts it back on the card.  A comment that names the card, either a bare token
(`DRIFT 100 25.4 ; D1`) or a position followed by a name and type words as decks converted from MAD
flat files carry them (`; 4.898 HKV MONITOR`, `; 2.450 BA1011 RBEND`), names an unlabelled element
(duplicates get `_2`, `_3` …; an explicit `NAME:` label always wins) and stores the type words in
`meta["tags"]`; prose comments name nothing.  A zero-length `DRIFT` so named is read as a `Marker`,
the survey marker of a device the deck only points at, and is written back as the same `DRIFT`
card.  Every card becomes an IR element in deck order:

* `DRIFT`, `QUAD` (with `G3..G6`), `SOLENOID`, `THIN_STEERING`, `APERTURE`, `MARKER`,
  `GAP`, `FIELD_MAP`, `NCELLS`, `RFQ_CELL`, `DTL_CEL` and the diagnostic family map to
  `Drift`, `Quadrupole`, `Solenoid`, `Kicker`, `Collimator`, `Marker`, `RFCavity`,
  `FieldMap`, `NCells`, `RFQCell` and `Instrument`;
* `EDGE` + `BEND` + `EDGE` are clustered into one `Bend`; the field index becomes a lab
  gradient through the local rigidity; the edge angle is read as `β = sign(θ)·e`;
* `SUPERPOSE_MAP` clusters become a `Superposition`;
* `FREQ` becomes a `Freq` element and is also resolved into every following RF element;
  `SET_SYNC_PHASE` binds to the next RF card including a thin `GAP` (TraceWin semantics)
  and sets `phase_is_sync`; `SET_BEAM_ENERGY` / `SET_BEAM_E0_P0` become `ReferenceChange`;
* every other command (`LATTICE`, `LATTICE_END`, `ADJUST_*`, `SET_*`, `MIN_*`, `DIAG_*`
  targets, `PARTRAN_STEP`, `TITLE`) is an inline zero-length `Directive` whose `role`
  classifies it, re-emitted verbatim by the writer; `ERROR_*` cards go to `lattice.errors`.

Raw (non-synchronous) RF phases are converted to the IR's species-independent synchronous
phase with `phase_from_tracewin_deg` (a π shift for negative species); the species is a
reader option and EQUIVALENT `SPECIES_ASSUMED` is recorded when it was defaulted.  A TraceWin deck
carries no beam: `species`, `kinetic_energy_eV` and `frequency_Hz` are reader options, taken from the
deck's `; lattix: reference` tag when present (`REFERENCE_FROM_TAG`), else from a HELIX project file
`<stem>.lgproj` next to the deck (`REFERENCE_FROM_PROJECT`); without any of them a 2.1 MeV proton is
assumed and flagged (`BEAM_ASSUMED`, also a lattice warning) — a deck of a different energy read that
way is not wrong element by element, but every normalised strength and any engine run is.  Unknown
cards are kept as directives with DROPPED `UNKNOWN_CARD`; a `FIELD_MAP` whose files are
missing is kept with LOSSY `FM_FILES_MISSING`; the DIAG_POSITION `1e50` sentinels and the
no-op hardware cards (`BPM :`) are recognised.

## Writing

`lattix/formats/tracewin/writer.py` emits one card per IR element with HELIX's default
elision and the `LATTICE n` counts recomputed from the cards actually written:

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Solenoid, Marker | `DRIFT L R`, `QUAD L G R θ G3..G6`, `SOLENOID L B R`, `MARKER` | EXACT |
| Sextupole, Octupole | `QUAD` card with `G = 0` and the gradient in `G3`/`G4` | EQUIVALENT `THICK_MULTIPOLE_AS_QUAD_CARD` |
| Multipole (thin) | comment | DROPPED `THIN_MULTIPOLE_UNSUPPORTED` |
| Bend | `EDGE β ρ gap K1 K2` + `BEND θ ρ N R HV` + `EDGE`; `HV = 1` for `tilt_ref = ±π/2` | EXACT; other tilts LOSSY `BEND_TILT_UNSUPPORTED` |
| RFCavity | thin: `GAP E0TL φ R` (preceded by `SET_SYNC_PHASE` and `FREQ` when needed); thick: `DRIFT L/2` + `GAP` + `DRIFT L/2` | EXACT / EQUIVALENT `THICK_CAVITY_AS_GAP` |
| FieldMap | `FIELD_MAP geom L φ f kb ke ki ka file` re-emitted from the source provenance | EXACT; no files DROPPED `FIELDMAP_NO_SOURCE` |
| NCells, RFQCell | re-emitted from the native operands | EXACT; missing DROPPED `NCELLS_PARAMS_MISSING` / `RFQ_PARAMS_MISSING` |
| Kicker | `THIN_STEERING Bx By`; a thick kicker is `DRIFT` + `THIN_STEERING` + `DRIFT`; a zero-kick thick kicker is a plain `DRIFT` | EXACT / EQUIVALENT `THICK_KICKER_SPLIT` |
| Collimator | `APERTURE dx dy type` (rectangular 0, elliptical 1) | EQUIVALENT `THICK_COLLIMATOR_AS_THIN` when it had a length |
| Instrument | the diagnostic/hardware card of its family, else a marker | EQUIVALENT `INSTRUMENT_AS_MARKER` |
| Foil | `; HELIX_FOIL` comment card | LOSSY `FOIL_AS_COMMENT` |
| Taylor, Patch | comment | DROPPED `TAYLOR_UNSUPPORTED`, `PATCH_UNSUPPORTED` |
| ReferenceChange, Freq, Directive | `SET_BEAM_ENERGY`, `FREQ`, the original card | EXACT |
| Superposition | `SUPERPOSE_MAP` cluster of `FieldMap` children | LOSSY `SUPERPOSE_CHILD_UNSUPPORTED` for other children |

The PIP-II TraceWin export contains no `THIN_STEERING` cards for zero-kick correctors and
its `LATTICE n` counts exclude `DIAG_*`, `APERTURE` and `THIN_STEERING`; the writer follows
both conventions.

## Units and conventions

* Deck units mm / deg / MHz / MeV / V; the IR is m / rad / Hz / eV / V.
* Bend: ρ > 0 with the sign in θ; `HV = 1` with angle θ is MAD tilt +π/2 with the same θ, and
  tilt −π/2 is written as `HV = 1` with −θ; `β = sign(θ)·e`, `gap = 2·hgap`, `K1 = fint`,
  `K2` default 2.80.  Measured on the licensed binary for all four sign combinations
  (HELIX and TraceWin agree to 1e-6).  The PIP-II export writes its two negative-angle
  vertical bends with the edge sign reversed; the MAD8 anchor test pins those two.
* Longitudinal basis `(z, dp/p)`, z ahead-positive; the reference momentum follows the
  cavities (`FOLLOWS_P0`).
* Thin `GAP` gain: `q·E0TL·cos φ_rf` before `SET_SYNC_PHASE`, `E0TL·cos φ_s` after it.
* Kicker field signs are pinned by tracking through the binary (invariant I-10), see
  [../oracles.md](../oracles.md).

* **Field-map phases.**  `SET_SYNC_PHASE` makes the card phase a synchronous phase (calibrated
  by integrating the map); otherwise, with `P = 0`, it is the RF phase when the reference particle
  *enters* the map — independent of the bunch clock.  HELIX adds the running bunch phase there;
  LightWin and TraceWin (by the ADS design deck that only accelerates this way, 20 → 502.24 MeV)
  do not, and neither does lattix since 2026-09-05 (`docs/oracles.md`, Phase 5.2).

* Write option `static_maps="hard_edge"`: a static solenoid/quadrupole map becomes the hard-edge
  magnet of `lattix.ir.fieldmap.replacement_for` (∫B and ∫B² preserved, drift padding; the
  `FM_SOL_HARDEDGE` / `FM_QUAD_HARDEDGE` rows, Equivalent tier) — for engines without static maps (LightWin).

## Known limits

* Only rectangular (0) and elliptical (1) `APERTURE` types are modelled; 2–6 stay in
  `native` (LOSSY `APERTURE_TYPE_UNMODELLED`).
* Only `DRIFT x_shift y_shift` misalignments have an IR form; `ERROR_*` studies are kept in
  `lattice.errors` and re-emitted by this writer only.
* Thick sextupoles/octupoles are TraceWin `QUAD` cards with higher-order gradients: the
  optics is the same, the card type is not.
* The trial TraceWin build on the development Mac runs at most 20 elements; the oracle
  adapter uses a 120 s timeout with one retry.

## Oracle

`lattix/oracles/tracewin.py` runs `TraceWin project.ini hide dat_file=… path_cal=…` (the
LightWin runner pattern) and reads `Transfer_matrix1.dat`; marker `oracle_tracewin`, needs
`TRACEWIN_EXE`, local only.  HELIX (`oracle_helix`, `HELIX_ROOT`) is the second, in-process
oracle for this format and agrees with TraceWin to 1e-6 on the vertical-bend decks.

LightWin's `Envelope3D` (`lattix/oracles/lightwin.py`, marker `oracle_lightwin`, conda env
`lightwin`) is a second TraceWin-semantics engine, free of the trial's 20-element limit: DRIFT,
QUAD, SOLENOID, BEND and 1-D FIELD_MAP cards, `SET_SYNC_PHASE` and relative phases; GAP/NCELLS are
skipped and EDGE/THIN_STEERING/APERTURE become drifts (audited in the result, report-only in the
battery).  Measured against HELIX on the field-map fingerprint (0.17 % on the gain) and against
lattix's own integration on the ADS example deck (2e-4 per cavity).

