# Synergia (lattice JSON)

`lattix/formats/synergia/` reads and writes the lattice archives of [Synergia 3](https://github.com/fnalacceleratormodeling/synergia2)
(Fermilab; `Lattice.as_json()` / `Lattice.load_from_json()`, a cereal JSON archive): a name, a reference
particle (charge, mass and total energy in GeV) and a flat list of elements carrying MAD-X's type names
and attributes.  Synergia is open source under Fermilab's DOE licence; lattix writes and reads the JSON,
and the engine runs through `lattix/oracles/synergia.py` where a built install is at hand (the clone's
pixi build locally: `pixi install -e cpu`, `pixi run -e cpu cmake/build/install` after
`git submodule update --init --recursive`; `LATTIX_SYNERGIA_PYTHON` or `LATTIX_SYNERGIA_ROOT`).
Every convention below was measured on Synergia 3 (clone `17e691d` of 2026-03-27, built 2026-09-05;
`docs/oracles.md` Phase 5.9).

## Reading

The archive's `value0` holds the lattice: `reference_particle_value` (charge, `four_momentum` mass / total
energy in GeV — the species by mass and charge, or an ion of that mass with EQUIVALENT
`SPECIES_ASSUMED`) and `elements`, each with `stype` (MAD-X's names), `lazy_double_attributes`
(`{"key", "value": {"value0": "<number>"}}`), `lazy_vector_attributes` (`knl`, `ksl`) and
`string_attributes`.  Types: `drift`, `quadrupole` (`k1`, `k1s`, `tilt`, `hoffset`, `voffset`),
`sextupole`/`octupole` (`k2`/`k3`, skew, `tilt`), `multipole` (`knl`, `ksl`, `tilt`), `sbend` and `rbend`
(`angle`, `e1`, `e2`, `fint`, `fintx`, `hgap`, `k1`, `k2`, `tilt`; an `rbend`'s faces are stored
sector-referenced, EQUIVALENT `RBEND_AS_SECTOR`), `solenoid` (`ks`), `rfcavity` (`volt` MV, `lag` turns,
`freq` MHz, `harmon`), `kicker`/`hkicker`/`vkicker`, `rcollimator` (`xsize`, `ysize`), `monitor`,
`hmonitor`, `vmonitor`, `instrument`, `marker`, `matrix` (`rm11 … rm66`, `kick1 … kick6`, transformed from
Synergia's basis into the IR's at the entry energy).  `nllens`, `elens`, `foil`, `dipedge` and `generic`
have no IR kind (DROPPED `UNSUPPORTED_SYNERGIA_ELEMENT`, a drift of their length with the attributes in
`native['synergia']`).

Synergia keeps one design momentum, so its normalized strengths are MAD-X's: a native archive is read
like a native MAD-X deck (every strength with the start rigidity, phases straight from `lag`); a lattix
archive carries a top-level `lattix` block (species, RF clock, `energy_mode`) and the writer's
`lattix` string attribute per element (kinds, phases, families, …), and the reader undoes the writer's
normalization the way the MAD-X reader does (`restore_energy_mode`, EQUIVALENT `ENERGY_MODE_RESTORED`;
no phase slip to undo).

## Writing

| IR kind | Synergia | Ledger |
|---|---|---|
| Drift | `drift(l)` | EXACT |
| Quadrupole, Sextupole, Octupole | `quadrupole(l, k1 = Bn1/Bρ, k1s, tilt, hoffset, voffset)`, `sextupole(k2, k2s)`, `octupole(k3, k3s)` | EXACT (other orders LOSSY `MULTIPOLE_ORDERS_DROPPED`) |
| Multipole (thin) | `multipole(knl, ksl, tilt)` (+ a `drift` body for a length) | EXACT (`THICK_MULTIPOLE_SPLIT`) |
| Bend | `sbend(l, angle, e1, e2, fint, fintx, hgap, k1, k2, tilt)` with sector faces | EXACT; after acceleration LOSSY `CONST_P0_BEND_UNDERBENT` (no `k0`) |
| Solenoid | `solenoid(l, ks = Bsol/Bρ)` | EXACT |
| RFCavity | `rfcavity(l, volt [MV], lag [turns] = φ/2π + ¼, freq [MHz], harmon)` | EXACT; `CONST_P0` and `CONST_P0_START_RIGIDITY` per accelerating element |
| FieldMap | the `lattix.ir.fieldmap` replacement ladder | EQUIVALENT / LOSSY `FM_*` |
| NCells, RFQCell | `drift` | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `kicker(l, hkick·r, vkick·r)` | EXACT (`EKICK_AS_MAGNETIC`) |
| Collimator | `rcollimator(l, xsize, ysize)` (an ellipse in the tag) | EXACT |
| Marker | `marker` | EXACT |
| Instrument | `monitor` / `hmonitor` / `vmonitor` / `instrument(l)` | EXACT |
| Foil | `marker` | LOSSY `FOIL_TO_MARKER` |
| Taylor | `matrix(rm11 … rm66, kick1 … kick6)` in Synergia's basis at the entry energy; a thick one is followed by a drift with the drift divided out | EXACT (`TAYLOR_BASIS_SYNERGIA`, `TAYLOR_THIN_PLUS_DRIFT`) |
| Patch | `marker` with the offsets in the tag | LOSSY `PATCH_DROPPED` |
| ReferenceChange | `marker` with the change in the tag | EQUIVALENT `REFCHANGE_AS_TAG` |
| Freq | `marker` (the frequency is per cavity) | EXACT |
| Directive | `marker` with the card in the tag | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Element apertures become Synergia's `aperture_type` and radius/size attributes, which its aperture
operation applies in tracking (EQUIVALENT `APERTURE_AS_ATTRIBUTE`); a quadrupole's transverse offsets are
its `hoffset`/`voffset`, the other misalignment components LOSSY `MISALIGN_DROPPED`.  The `energy_mode`
option (`"constant"` default — measured, see below — `"local"`, `"delta"`) is the MAD-X writer's
(`lattix/ir/energy_mode.py`).

## Units and conventions

* m, rad, MV (`volt`), MHz (`freq`), turns (`lag`), GeV (the reference's mass and total energy).
  Particle coordinates `(x, xp = px/p_ref, y, yp, cdt, dpop = Δp/p_ref)` with `cdt` **late-positive**: a
  1 m drift at 2.1 MeV gives `R56 = −L/(βγ²) = −14.905` (`Basis.SYNERGIA`, `z = −β·cdt`, `δ = dpop`;
  `tests/oracles/goldens/fingerprints.json`).
* **Cavity**: `ff_rfcavity` gains `E1 = E0 + volt·sin(2π·lag)` (MAD-X's rule) on the *bunch* reference
  particle (`new_pref_b`) while the design reference keeps its momentum, so the IR's cos convention is
  `lag = φ/2π + ¼` — 866 025.4037844 eV at −30° for a proton and an H⁻ (4e-15), bunching.  **No phase
  slip**: the bunch's time is measured against its own accelerated reference (two 1 MV gaps at −30° both
  gain V·cos 30° with the plain lag; the MAD-X writer's slip correction would make the second one lose
  935 keV).
* **Strengths**: every element scales its normalized strength by `p_design/p_bunch` before acting
  (`brho_l/brho_b` in `ff_solenoid`/`ff_quadrupole`), so `k1 = G/Bρ_design` is right on an accelerated
  bunch — the writer's default `energy_mode="constant"` (`CONST_P0_START_RIGIDITY` per accelerating
  element); MEASURED: a quadrupole after two 1 MV gaps (2.1 → 3.83 MeV) written that way shows the lab
  gradient at the local momentum to 8e-8, `"local"` comes out weaker by `p_design/p_local`.  Kicks and
  thick maps scale by `Bρ_local/Bρ_start` like the MAD-X writer's.
* **Bends after acceleration** are under-bent: the field comes from the design momentum and `sbend`
  reads no `k0` — a 0.1 rad bend after a 1 MV crest gap at 2.1 MeV maps with `θ_eff = 0.0823`
  (LOSSY `CONST_P0_BEND_UNDERBENT`).
* Pole faces are MAD-X's (sector-referenced `e1/e2`, `fint`/`hgap`); `ks = Bsol/Bρ_signed`;
  `hkick`/`vkick` deflect the reference; `knl`/`ksl` are MAD's integrated strengths.
* **Solenoid bug (clone `17e691d`)**: `ff_solenoid.h` passes `(ksl, ks)` to a body that takes `(ks, ksl)`
  — the momenta rotate by `ks` regardless of the length and the displacement is divided by `ks·L`
  (`R12 = sin(ks)/(ks·L) = 3.33` for `ks = 0.1`, `L = 0.3 m`).  The writer keeps `ks = B/Bρ` (the code's
  intent); the battery treats Synergia pairs with solenoids as report-only.
* The JSON is a cereal archive: every element carries `format` (1 = MAD-X), `type` (the
  `element_type` enum index), `ancestors`, `length_attribute_name`, `bend_angle_attribute_name`,
  `revision` and four `markers`; the lattice `updated` flags and an empty `tree` (a static lattice);
  numbers are strings in `value0` (an `mx_expr`; lattix writes plain numbers and evaluates only numeric
  strings).  Apertures are Synergia's own `aperture_type` (`circular`, `elliptical`, `rectangular`) with
  `circular_aperture_radius`, `elliptical_aperture_horizontal/vertical_radius`,
  `rectangular_aperture_width/height` (full sizes).

## Known limits

* Synergia's `foil`, `nllens`, `elens` and `dipedge` have no IR kinds; the IR's foil, patch and
  reference change have no Synergia element (markers with tags).
* One design momentum: bends downstream of acceleration are under-bent (no `k0`), a
  `ReferenceChange` cannot be applied.
* A thick `Taylor` is a thin `matrix` followed by a drift with the drift divided out (Synergia refuses a
  matrix with a length: EQUIVALENT `TAYLOR_THIN_PLUS_DRIFT`).
* The solenoid body bug above; quadrupoles are a second-order Yoshida integrator (4.5e-8 on the FODO).
* The oracle needs a local build (no conda-forge package; ~25 min with pixi): marker `oracle_synergia`,
  never in CI.

## Oracle

`lattix/oracles/synergia.py` + `synergia_worker.py`: one single-element lattice per element with the
design reference, a fresh 13-particle probe bunch whose reference carries the energy reached so far and
whose `cdt` is carried along, one `Propagator` pass (`Independent_stepper_elements(1)`), the map fitted
from the offsets.  Measured: the fingerprint drift exact to 1.0e-8 and the 1 MV gap to 4e-15; HELIX's
`fodo.madx` vs cpymad 4.5e-8 on T4×4, dispersion and path; lattix's archive loaded by Synergia, re-read
and re-written gives the same maps to 1e-12; the battery's Synergia cases pass.
