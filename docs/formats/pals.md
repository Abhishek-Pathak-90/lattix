# PALS

The Particle Accelerator Lattice Standard (PALS, LBNL/Cornell) is a YAML/JSON document:
a `PALS:` root, named elements with `kind` and parameter groups (`MagneticMultipoleP`,
`BendP`, `RFP`, `SolenoidP`, `ApertureP`, `BodyShiftP`, …), `BeamLine`s and `Lattice`
branches.  lattix registers `.pals.yaml`, `.pals.yml` and `.pals.json`; the IR was designed
to mirror PALS names, so the reader and writer are mostly renames plus the rigidity and
phase conversions.  Written against the standard at commit `a2b1083` (2026-09-01) and
`pals-schema` 0.3.0.

## Reading

`lattix/formats/pals/reader.py` parses the root (`version`, `authors`, `notes`,
`extension_labels`, `include`, `facility`), element definitions, `BeamLine.line` items (bare
names, in-place definitions, `inherit`, `repeat` — negative means reversed order, not
direction reversal — `direction: -1` for true reversal, `placement`, `zero_point`,
`periodic`), `Lattice.branches` with a `BeginningEle` first, `use:` (default: the last
`Lattice`).  The names that differ from the IR:

| IR | PALS |
|---|---|
| `BendP.angle` | `angle_ref` |
| `BendP.edge_int1` (fint) and `hgap` | `edge1_int` = the product `fint·hgap` in m (LOSSY `PALS_EDGE_INT_NEEDS_HGAP` when it cannot be split) |
| `RFP.n_cell`, `RFP.phase_rad` | `num_cells`; `phase` in rad/2π plus `zero_phase` |
| `BodyShiftP.tilt` | `z_rot` |
| `ApertureP.x_limits`, `aperture_at` | `x_min`/`x_max` (or `x_center`/`x_width`), `location` |
| `Kicker.hkick`/`vkick` | `MagneticMultipoleP.Kn0L`/`Ks0L` |

Normalized strengths (`Kn{n}`, `Ks{n}`, `Ksol`, `Bn0_ref`) are converted in a second pass
after the line is expanded, with the rigidity at each element.  Unknown kinds are DROPPED
`UNSUPPORTED_PALS_KIND`; a missing `BeginningEle` gives `PALS_REFERENCE_ASSUMED`; PyYAML's
reading of the standard's own `1.0e9` as a string is handled.

## Writing

`lattix/formats/pals/writer.py` emits the `PALS:` root with a `BeginningEle` carrying
`ReferenceP` (species, `pc_ref`, `time_ref`), the elements with lab fields (`Bn1`, `Bsol`) and
`RFP` with `phase` in turns and `zero_phase: ACCELERATING`, and a `use:`.  Flavor `flat`
(no sublines, no `repeat`, Drift + Quadrupole only, no `BeginningEle`) targets ImpactX's
`pals_to_impactx`.

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Multipole, Bend, Solenoid, Kicker, Collimator, Marker, Instrument, Taylor, Patch, ReferenceChange, Freq | the PALS kind of the same name (`Kicker` → `Kicker` with `Kn0L`/`Ks0L`) | EXACT |
| RFCavity | `RFCavity` with `RFP` (`voltage` or `gradient`, `phase`, `frequency`, `num_cells`, `dE_ref`) | EXACT |
| FieldMap | `RFCavity` with the map's voltage and phase | EQUIVALENT `FM_TO_CAVITY` |
| NCells, RFQCell | drift | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Foil | `Foil` with `dE_ref` only | LOSSY `FOIL_MATERIAL_DROPPED` |
| Directive | a PALS note | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | `UnionEle` of shifted children | EQUIVALENT `SUPERPOSITION_AS_UNIONELE` |

## Units and conventions

* SI + eV like the IR; `phase = φ/2π` (`pals_phase`) with `zero_phase = ACCELERATING` so 0 is
  the crest; `dE_ref` written explicitly.
* Both `Bn1` and `Kn1` are legal in PALS; lattix writes the lab field, Bmad's `write pals`
  writes `Kn1` (and `kind: Bend` with `g_ref` plus a redundant `Kn0`).

## Known limits

* `pals-schema` 0.3.0 is an older draft: its untagged union turns unknown kinds into
  placeholders silently, so lattix validates structure itself and treats schema failures as
  expected (`xfail`) when the standard moves.
* ImpactX 26.08 `pals_to_impactx` reads only the `flat` flavor.
* No field-map extension exists in the standard yet.

## Oracle

Two independent implementations serve as oracles: Tao's `write pals` (env `bmad`) for the
reader and ImpactX's `KnownElementsList.load_file` for the writer's `flat` flavor; markers
`oracle_bmad` and `oracle_impactx`.
