# Cheetah (LatticeJSON)

`lattix/formats/cheetah/` reads and writes the LatticeJSON files of [Cheetah](https://github.com/desy-ml/cheetah)
(`cheetah.latticejson.save_cheetah_model` / `load_cheetah_model`, version tag `cheetah-0.8`), the
differentiable beam-dynamics code from DESY.  Cheetah is GPL-3: lattix touches only the JSON and never
imports it; the engine runs in its own environment through `lattix/oracles/cheetah.py`.  Every
convention below was measured on Cheetah 0.8.4 / torch 2.14 on 2026-09-05 (`docs/oracles.md`).

## Reading

The reader takes the element types Cheetah's own `latticejson` writes: `Drift`, `Quadrupole` (`k1`,
`tilt`, `misalignment`), `Sextupole` (`k2`), `Dipole` (`angle`, `k1`, sector-referenced `dipole_e1/e2`,
`gap` = 2·hgap, `fringe_integral`, `tilt`), `RBend` (`rbend_e1/e2` + angle/2), `Solenoid` (`k` =
B/(2Bρ)), `Cavity` (`voltage`, `phase`, `frequency`, `cavity_type`), the three correctors, `Aperture`,
`BPM`, `Screen`, `Marker`, `CustomTransferMap` (a `Taylor`, transformed from Cheetah's basis into the
IR's at the entry energy) and nested `lattices` (IR lines, root = `root`).  `TransverseDeflectingCavity`,
`Undulator` and `SpaceChargeKick` have no IR kind (DROPPED `UNSUPPORTED_CHEETAH_ELEMENT`, kept as a
drift of their length).  A LatticeJSON carries no beam: the reference comes from the `lattix:
reference` tag lattix leaves in `info` (EQUIVALENT `REFERENCE_FROM_TAG`) or from `read(...,
species=, kinetic_energy_eV=)`.  Normalized strengths become fields with the signed rigidity at each
element's entrance (the energy follows the cavities).  The `metadata.lattix` block lattix writes
restores the exact IR kinds, names, phases and gains; the apertures and body drifts written beside an
element are folded back onto it.

## Writing

| IR kind | Cheetah | Ledger |
|---|---|---|
| Drift | `Drift` | EXACT |
| Quadrupole, Sextupole | `Quadrupole(k1 = Bn1/Bρ_signed, tilt, misalignment)`, `Sextupole(k2)`; a skew component is a tilt | EXACT (other orders LOSSY `MULTIPOLE_ORDERS_DROPPED`) |
| Octupole | `Drift` | LOSSY `OCTUPOLE_TO_DRIFT` |
| Multipole (thin) | `CombinedCorrector(0 m)` for the dipole terms | EQUIVALENT `MULTIPOLE_AS_CORRECTOR`, higher orders LOSSY `MULTIPOLE_ORDERS_DROPPED` |
| Bend | `Dipole(arc length, angle, k1, dipole_e1/e2, tilt, gap = 2·hgap, fringe_integral[_exit])` | EXACT |
| Solenoid | `Solenoid(k = Bsol/(2 Bρ_signed))` | EXACT |
| RFCavity | `Cavity(voltage = −V/q, phase = −φ [deg], frequency, cavity_type)` | EXACT; a zero-length cavity EQUIVALENT `CHEETAH_ZERO_LENGTH_CAVITY` |
| FieldMap | the `lattix.ir.fieldmap` replacement ladder | EQUIVALENT / LOSSY `FM_*` |
| NCells, RFQCell | `Drift` | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `CombinedCorrector(horizontal_angle = hkick, vertical_angle = vkick)` | EXACT (`EKICK_AS_MAGNETIC` for electric) |
| Collimator | `Aperture(x_max, y_max, shape)` + `Drift` body | EXACT |
| Marker | `Marker` | EXACT |
| Instrument | `BPM` / `Marker` (`Drift` when it has a length) | EXACT / EQUIVALENT `INSTRUMENT_AS_MARKER`, `MONITOR_AS_DRIFT` |
| Foil | `Marker` | LOSSY `FOIL_TO_MARKER` |
| Taylor | `CustomTransferMap` (7×7, transformed into Cheetah's basis at the entry energy) | EXACT (EQUIVALENT `TAYLOR_BASIS_CHEETAH` notes the transform; a foreign basis is re-used verbatim) |
| Patch, ReferenceChange, Directive | `Marker` with the data in the metadata | DROPPED `PATCH_DROPPED`, `REFCHANGE_DROPPED`, `FOREIGN_DIRECTIVE` |
| Freq | `Marker` | EXACT (the frequency is per cavity) |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Element apertures become `Aperture` elements at the element's ends (EQUIVALENT `APERTURE_AS_ELEMENT`);
names become Python identifiers (`Segment` registers them as torch modules; the original is in
`metadata.lattix.name`, EXACT `NAMES_SANITIZED`); a thin gap keeps lattix's thin-gap RF-focusing lens
next to the cavity (`RF_FOCUSING_AS_MATRIX`).

## Units and conventions

* m, rad, V, Hz, degrees for the cavity phase.  First-order maps are 7×7 (augmented) in
  `(x, px/p0, y, py/p0, τ, ΔE/(p0 c))` with `τ = c·Δt` **late-positive**: a 1 m drift at 2.1 MeV gives
  `R56 = −L/(β²γ²)` (`Basis.CHEETAH`; `tests/oracles/goldens/fingerprints.json`).
* **Cavity**: `ΔE = −voltage·q·cos(phase)`, so `voltage = −V/q` keeps the IR's species-independent
  `V·cos φ` (protons get a negative voltage, electrons and H⁻ a positive one); the slope
  `r65 ∝ sin(phase)` means a late particle gains more for `phase > 0`, the opposite of the IR's
  `φ < 0` bunching convention — `phase = −φ`, exactly what Cheetah's own Bmad converter does
  (`phase = −phi0`) and its Elegant converter (`phase − 90°`).  The matrix is Rosenzweig–Serafini's and
  divides by the length: a zero-length cavity is `inf` in Cheetah; the oracle tracks 1 µm.
* `k1 > 0` focuses `x` for every species and the solenoid rotation ignores the charge: both are
  normalized with the *signed* rigidity (`k1 = Bn1/Bρ_signed`, `k = Bsol/(2 Bρ_signed)`), measured
  identical to HELIX for protons and H⁻ (5e-15).
* `Dipole.gap` is the full gap (`HGAP = gap/2`), `fringe_integral` enters the linear edge map, the
  face angles are sector-referenced and `RBend` adds `angle/2` itself; corrector angles are
  species-agnostic kicks (`px += angle`).
* The reference energy follows the cavities; Cheetah's named species carry CODATA-2022 masses
  (1.3 eV above lattix's proton), so the oracle hands Cheetah the kinetic energy.

## Known limits

* No octupole, thin multipole, foil, patch or explicit reference change in Cheetah.
* A zero-length cavity cannot be evaluated by Cheetah itself (`inf`); the file keeps the exact IR
  length, the oracle substitutes 1 µm and Cheetah's cavity focusing at that length is negligible, so
  the transverse effect of a thin gap is the lattix lens (Equivalent tier against HELIX: 6.5e-3 on
  the MEBT).
* Apertures are separate zero-length elements; `Screen` parameters are not modelled (an
  `Instrument` of family SCREEN is a `Marker`).

## Oracle

`lattix/oracles/cheetah.py` + `cheetah_worker.py` (conda env `cheetah`: torch CPU +
`cheetah-accelerator`; marker `oracle_cheetah`): per-element `first_order_transfer_map(E, species)`
with the energy advanced by each cavity's gain.  Measured: `fodo.madx` vs cpymad 1.8e-15 on the
transverse block and every longitudinal block within 8e-9; `fodo_cell.dat` and `solenoid_channel.dat`
vs HELIX 3e-14 / 5e-15; `mebt_line.dat` vs HELIX 6.5e-3 (Equivalent tier, the thin gaps).  The oracle also
runs a `.bmad` deck through Cheetah's own Bmad converter (`fmt="bmad"`): lattix's Bmad and LatticeJSON
writers agree through it below 1e-8 on `fodo.madx`.
