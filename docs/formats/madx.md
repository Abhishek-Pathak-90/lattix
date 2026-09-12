# MAD-X

MAD-X (CERN) decks are sequences of element definitions, `sequence … endsequence` blocks or
`line` definitions, a `beam` statement and deferred `:=` expressions.  lattix registers
`.madx`, `.seq`, `.mad` and `.str`; the reader drives a real MAD-X process through `cpymad`
and the writer emits sequence or line decks.

## Reading

`lattix/formats/madx/reader.py` hands the deck to MAD-X (`Madx.call`), so `call` chains,
macros, `if`, variables and `:=` behave exactly as MAD-X defines them, and builds the IR from
`sequence.expanded_elements` — MAD-X's own expansion including its implicit drifts, so the
element boundaries are MAD-X's.  `beam` gives the `ReferenceParticle`; `dipedge` elements
adjacent to a bend are folded into `e1`/`e2`/`fint`/`hgap` (EQUIVALENT `DIPEDGE_FOLDED`, with a
curvature tolerance of 1e-4; orphans are LOSSY `DIPEDGE_ORPHAN`); `ealign` misalignments become
`BodyShiftP`; `yrotation`, `xrotation`, `srotation` and `translation` become `Patch` elements with
the conventions MAD-X's own `survey` uses (measured with cpymad 5.09.03, [conventions §11](../conventions.md));
`changeref` is ignored by that survey and stays DROPPED; `efcomp` field errors are LOSSY
`EFCOMP_DROPPED`; unknown types are DROPPED `UNSUPPORTED_MADX_TYPE`.  Without cpymad the reader
raises `MissingDependencyError`.  The writer emits a `Patch` as the matching card, or as several
cards at one position when it combines offsets and rotations (EQUIVALENT `PATCH_AS_CARDS`).

## Writing

`lattix/formats/madx/writer.py` writes a `beam` statement (mass and energy in GeV, signed
charge), one definition per element with the `! lattix: name="…" type="…"` provenance tag,
and the root line as a `sequence` (`refer=centre`, `at=` positions) or, when the lattice has
zero length, as a `line` (EQUIVALENT `ZERO_LENGTH_LINE_MODE`).

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Solenoid | `drift`, `quadrupole k1 tilt`, `sextupole k2`, `octupole k3`, `solenoid ks` | EXACT; a drift's aperture is LOSSY `APERTURE_DROPPED` |
| Multipole | `multipole knl={…} ksl={…}` | EXACT |
| Bend | `sbend l angle e1 e2 fint fintx hgap k1 tilt` (a rectangular source gets `e += angle/2`) | EXACT |
| RFCavity | `rfcavity volt lag freq` | EQUIVALENT `CONST_P0` + `CONST_P0_DELTA_RIGIDITY` (or `…_LOCAL_RIGIDITY` / `…_START_RIGIDITY`) |
| FieldMap | thick `rfcavity` with the map's self-consistent voltage and synchronous phase | EQUIVALENT `FM_TO_CAVITY` |
| NCells, RFQCell | drift of the same length | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `kicker hkick vkick` / `hkicker` / `vkicker` | EXACT |
| Collimator | `rcollimator` / `ecollimator` with `apertype`, `aperture` | EXACT |
| Marker, Instrument | `marker`, `instrument` / `monitor` | EXACT |
| Taylor | `matrix` element | EXACT |
| Foil, Patch | marker | LOSSY `FOIL_TO_MARKER`, `PATCH_DROPPED` |
| ReferenceChange | marker + lattix tag carrying the jump (restored on read) | EQUIVALENT `REFCHANGE_AS_TAG` |
| Freq | comment (the frequency is already in each cavity) | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` (MAD-X-native directives are re-emitted) |
| Superposition | consecutive elements | LOSSY `SUPERPOSITION_FLATTENED` |

A MAD-X sequence cannot contain overlapping elements: the writer sorts entries by position,
shortens the *preceding* drift when a source drift is negative (the PIP-II BTL has one,
−0.204288 m; LOSSY `NEGATIVE_DRIFT_DROPPED` / EQUIVALENT `DRIFT_SHORTENED_BY_OVERLAP`) and shifts
genuine thick-element collisions downstream (LOSSY `OVERLAP_SHIFTED`); markers inside an
overlap are moved out (`MARKER_MOVED_OUT_OF_OVERLAP`).

## Units and conventions

* m, rad, `volt` in MV, `freq` in MHz, `lag` in turns: `lag = φ/2π + 0.25`, gain
  `V·sin(2π·lag)`, species-independent (measured with cpymad 5.09.03 for charge ±1).
* Normalized strengths use the signed rigidity `Bρ_signed = sign(q)·pc/(|q|c)` of MAD-X's
  own reference orbit at each element (`energy_mode=delta`, default: the start rigidity across
  the RF gains MAD-X carries as `pt`, the local rigidity across reference changes it cannot
  apply — kicks, `matrix` maps and a bend's `k0` are rescaled alike), at each element's
  entrance (`energy_mode=local`) or at the lattice start (`energy_mode=constant`); see
  `conventions.md` §3.
* Longitudinal basis `(T, pt)` with T ahead-positive (measured: a drift's R56 is positive);
  p0 is constant through RF, so accelerating lattices are optically correct per section
  only — compare them against Elegant, Bmad or TraceWin rather than MAD-X.
* `apertype=ellipse` / `rectangle` with `aperture={…}` on every element type except `drift`.

## Known limits

- The reader's `frequency_Hz` option sets the machine RF clock an RF-free MAD-X deck lacks, which the RF-based
  writers (IMPACT-Z, IMPACT-T, DYNAC) need to stay exact. MAD-X keeps p0 constant: on a line whose momentum
  grows more than twofold (`P0_RATIO_LIMIT`) the battery and `lattix validate` do not run it and say why
  (its own `twiss` fails at p ×5.6 on the LightWin ADS deck).
* MAD-X's own `twiss` fails on an open accelerating line with real `rfcavity` elements
  ("error with deltap"); the MAD-X leg of the field-map gate checks loading and reporting.
* Deferred expressions are re-emitted only for MAD-X → MAD-X translations; other targets
  receive evaluated numbers.
* The reader needs `cpymad`; there is no pure-Python fallback grammar yet.

## Oracle

`lattix/oracles/cpymad.py`: `twiss … sectormap` per-element matrices, `survey`, `track onepass`;
marker `oracle_madx`.  Fingerprint golden in `tests/oracles/goldens/`.
