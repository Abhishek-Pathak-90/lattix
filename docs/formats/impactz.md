# IMPACT-Z

IMPACT-Z (LBNL) reads a positional `ImpactZ.in`: eleven header records (processor grid,
particle count and integrator flags, space-charge mesh, distribution, charge states,
Twiss/scale parameters, and `current energy mass charge frequency phase`), then one line per
element `length nseg mapstp type value1 … /` with numeric type codes (0 drift, 1 quadrupole,
3 solenoid, 4 dipole, 5 multipole, 101–110 RF structures, negative codes for diagnostics
and thin elements, −99 end).  lattix registers the file name `impactz.in` with a reader and
a writer, both checked against the sources at `src/Contrl/AccSimulator.f90` and the
conda-forge `impact-z` 2.7.7 binary.

## Reading

`lattix/formats/impactz/reader.py` reads the header into the `ReferenceParticle` (kinetic
energy, mass and charge in eV / e, RF frequency), keeps the run settings in `meta`
(EQUIVALENT `SIM_SETTINGS_KEPT_IN_META`), and maps the element codes: 0 → `Drift` (radius as
aperture), 1 → `Quadrupole` (gradient T/m; a file id loads an `rfdataN.in` profile,
`IMPACTZ_QUAD_GRADIENT_PROFILE`), 3 → `Solenoid`, 4 → `Bend` (angle, k1, half gap, e1, e2,
pole-face curvatures LOSSY `IMPACTZ_POLE_FACE_CURVATURE`, one fringe integral), 5 →
`Multipole`, 101/103 → `NCells` (`NCELLS_AS_CCL`), 104 → `RFCavity` (an ideal cavity when
`Param(5)` is negative, else `IMPACTZ_RFDATA_REFERENCE`), −21 → `Kicker`, −13 → `Collimator`,
−55 → thin `Multipole`, −2/−8 and other diagnostics → `Instrument`, unknown codes DROPPED
`UNSUPPORTED_IMPACTZ_TYPE`.  Per-element `dx dy rot_x rot_y rot_z` columns become
`BodyShiftP` (`flagerr = 1` required for IMPACT-Z to apply them).  Names live in the
`! lattix: name=… kind=…` comment the writer adds (`IMPACTZ_NAME_TAG`).

The reference gain of an rfdata cavity is not integrated on read: the element comes back as a
`FieldMap` with the Fourier profile in `meta["impactz_rfdata"]` and no `dE_ref` (LOSSY
`IMPACTZ_RF_GAIN_UNKNOWN`), so a lattice with such cavities has no energy profile until the maps are
re-integrated.

## Writing

`lattix/formats/impactz/writer.py` writes the header from the reference particle and
`meta`, then one card per element ending with `/`.

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Solenoid | type 0, type 3 | EXACT |
| Quadrupole | type 1 with the roll in `rot_z` | LOSSY `IMPACTZ_SKEW_QUAD` when tilted (IMPACT-Z applies `rot_z` only as an error) |
| Sextupole, Octupole | type 5 (order id, gradient) | LOSSY `IMPACTZ_MULTIPOLE_LINEAR_MAP` (the linear-map integrator misreads the order id as the gradient; use `flagmap = 2`) |
| Multipole | −55 thin-lens kick, normal components | EQUIVALENT `THIN_MULTIPOLE_AS_KICK`; skew LOSSY `IMPACTZ_SKEW_MULTIPOLE` |
| Bend | type 4 (angle, k1, half gap, e1, e2, fint) | LOSSY `IMPACTZ_NO_REF_TILT` for tilted bends, `IMPACTZ_SINGLE_FINT` when `fintx ≠ fint` |
| RFCavity | type 104 ideal cavity (negative `Param(5)`): gradient V/m, synchronous phase deg | EXACT; thin gaps EQUIVALENT `THIN_GAP_AS_SHORT_CAVITY` |
| FieldMap | type 104 with an `rfdataN.in` on-axis Ez profile when the map provides one, else a drift | EQUIVALENT `FM_AS_RFDATA` / LOSSY `FM_TO_DRIFT` |
| NCells | type 101/103 when voltage, frequency and length are known, else a drift | LOSSY `NCELLS_TO_DRIFT` |
| RFQCell | drift | LOSSY `RFQ_TO_DRIFT` |
| Kicker | −21 kick (+ drift for a thick kicker) | LOSSY `IMPACTZ_THICK_KICKER_SPLIT` |
| Collimator | −13 rectangular slit (ellipses by bounding box) | LOSSY `IMPACTZ_ELLIPSE_AS_SLIT` |
| Marker, Instrument | zero-length drift | LOSSY `INSTRUMENT_TO_MARKER` |
| Foil, Taylor, Directive | drift / comment | DROPPED `FOIL_DROPPED`, `TAYLOR_DROPPED`, `FOREIGN_DIRECTIVE` |
| Patch, ReferenceChange | drift / nothing | LOSSY `PATCH_DROPPED`, `REFCHANGE_DROPPED` |
| Superposition | drift of the same length | LOSSY `SUPERPOSITION_TO_DRIFT` |

## Units and conventions

* Lengths in m, gradients in T/m (lab fields, no rigidity involved), kinetic energy and mass
  in eV, frequency in Hz.
* **Ideal cavity sentinel** (measured, `BeamBunch.f90:323-437`): any element other than
  types 0/1/4 with a negative `Param(5)` is an ideal RF cavity with gradient V/m and
  synchronous phase in degrees, gain `E0·L·cos φs` with no charge factor — the IR rule
  verbatim (thin-cavity gain to 1.2e-16).  A solenoid with a negative `dx` would become a
  cavity; the writer zeroes it (`IMPACTZ_SOLENOID_DX_SENTINEL`).
* Native basis `(x/Scxl, γβx, y/Scxl, γβy, ω·Δt, γ_ref − γ)` with `Scxl = c/(2πf)`, late-positive
  phase; p0 follows.  The adapter's transform is `d = (Scxl, 1/βγ, Scxl, 1/βγ, −β·Scxl, −1/β²γ)`.
* No Twiss, dispersion or survey output: the oracle reads a 13-particle probe dumped at every
  boundary and the reference energy from `fort.18`.

## Known limits

* Type-5 multipoles produce NaN under `flagmap = 1`; misalignments need `flagerr = 1`.
* IMPACT-Z has no reference tilt, second fringe integral, skew multipole, patch or foil.
* The `rfdata` file limit and the reference-energy coupling of type 104 are recorded as
  `IMPACTZ_RFDATA_LIMIT` / `IMPACTZ_RFDATA_REFERENCE` when hit.

## Oracle

`lattix/oracles/impactz.py` runs `ImpactZexe` (env `lattix`, or `IMPACTZ_EXE`) with
`flagdist = 19` probe particles and zero-length `-2` dumps at every boundary; marker
`oracle_impactz`.  `fodo.madx` agrees with cpymad to 6e-14 on the transverse blocks.
