# ImpactX

ImpactX (LBNL) takes an AMReX `inputs` file (`key = value`, `#` comments, `lattice.elements`
naming the sequence, `<name>.type` per element) or a Python script using the same element
classes.  lattix registers `.impactx.in` (reader and writer) and `.impactx.py` (writer;
scripts are not executed to read them).

## Reading

`lattix/formats/impactx/reader.py` parses the ParmParse grammar: `beam.kin_energy` (MeV),
`beam.particle` (`proton`, `electron`, `positron`, `Hminus` → `SPECIES_FROM_IMPACTX`),
`beam.charge` and the distribution keys (kept verbatim in `meta["impactx_beam"]` so a
re-write reproduces the deck), `lattice.elements`, `lattice.nslice`, `lattice.periods`,
`lattice.reverse`, and each element's parameters as defined in `InitElement.cpp`.  The
sequence is flattened (`line` sub-lattices, `periods`, `reverse` expanded in place,
EQUIVALENT `LINE_FLATTENED`); `dipedge` elements against an `sbend`/`cfbend` are folded into
the bend (`DIPEDGE_FOLDED`).  Normalized strengths use the signed rigidity at the element's
first occurrence, propagated through `shortrf` gains as ImpactX does.  A Python deck is
reported DROPPED `IMPACTX_PYTHON_NOT_READ`; ImpactX's own MAD-X subset is read by the MAD-X
reader, which is a superset of it.

## Writing

`lattix/formats/impactx/writer.py` writes the beam block, `lattice.elements` and one block
per element (names up to 64 characters).

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Solenoid | `drift ds`, `quad ds k rotation`, `sol ds ks` | EXACT |
| Sextupole, Octupole | drift + thin `multipole` (order 3 / 4) + drift | LOSSY `THICK_TO_THIN_MULTIPOLE` |
| Multipole | one `multipole` per order (`K_normal`, `K_skew`) | EXACT |
| Bend | `dipedge` + `sbend ds rc` + `dipedge` | EXACT |
| RFCavity | thin: `shortrf V freq phase`; thick: drift + `shortrf` + drift | EXACT / EQUIVALENT `THICK_CAVITY_AS_SHORTRF` |
| FieldMap | drift + `shortrf` + drift with the map's voltage | LOSSY `FM_TO_CAVITY` |
| NCells, RFQCell | drift | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `kicker`; thick: drift + kick + drift | EXACT / EQUIVALENT `THICK_KICKER_SPLIT` |
| Collimator | `aperture` (thin); thick: drift + aperture + drift | EQUIVALENT `THICK_COLLIMATOR_SPLIT` |
| Marker, Instrument | zero-length drift / `beam_monitor` with `instrument=beam_monitor` | EQUIVALENT `MARKER_AS_ZERO_DRIFT`, `INSTRUMENT_AS_MARKER` |
| Taylor | `linear_map` | LOSSY `TAYLOR_OFFSET_DROPPED` when the map has a constant term |
| Patch | pure roll or pure z shift only | LOSSY `PATCH_DROPPED` otherwise |
| Foil, ReferenceChange | marker | LOSSY `FOIL_TO_MARKER`, `REFCHANGE_DROPPED` |
| Freq | comment | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | consecutive elements | LOSSY `SUPERPOSITION_FLATTENED` |

## Units and conventions

* m; `quad k` in 1/m² with the signed rigidity; `rotation` in degrees; `shortrf V` is
  dimensionless (`voltage_V / mass_eV`), `freq` in Hz, `phase` in degrees with 0 = crest
  (the IR convention, no shift — measured gain 866 025.404 eV at −30°).
* Native basis `(x, px, y, py, t, pt)` with t late-positive and `pt = −ΔE/p0c`; the adapter
  returns `S·R·S` in MAD-X's `(T, pt)`; the two sign flips cancel in the longitudinal block
  but not in the dispersion column.
* p0 follows `shortrf` (`FOLLOWS_P0`).

## Known limits

* ImpactX has no thick multipole, thick aperture or field-map element; `Marker` has no
  `inputs` type.
* `load_inputs_file` segfaults after the first call in one process (ParmParse state), so
  the oracle builds the lattice through the Python API instead.

## Oracle

`lattix/oracles/impactx.py`: in-process or a `python -I` worker in env `lattix`
(`LATTIX_IMPACTX_ENV` / `LATTIX_IMPACTX_PYTHON`), per-element maps from 13-probe central
differences re-seeded at every element; marker `oracle_impactx`.  `fodo.madx` agrees with
cpymad to 1.8e-15 and with ImpactX's own MAD-X loader to 0.0.
