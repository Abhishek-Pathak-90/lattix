# Ocelot (lattice module)

`lattix/formats/ocelot/` reads and writes the Python lattice modules of
[Ocelot](https://github.com/ocelot-collab/ocelot) (DESY): element constructors, a `cell` tuple and
`lattice = MagneticLattice(cell)`.  Ocelot is GPL-3: lattix writes and parses the module as text and never
imports Ocelot; the engine runs in its own environment through `lattix/oracles/ocelot.py`.  Every
convention below was measured on Ocelot 25.06.0 (`ocelot-desy` 25.6.0) on 2026-09-05 (`docs/oracles.md`,
Phase 5.7).

## Reading

The reader walks the module's AST and never runs it: element constructors with positional or keyword
arguments (literals, arithmetic on literals and on earlier numeric assignments, `pi`, `np.pi`, `inf`,
`sqrt(...)` and friends), attribute assignments (`q.dx = 1e-3`, `q.tilt = ...`, `tws0.E = ...`), sequences
(`cell = part + 2 * (c,) + [u, s]`, nested tuples and lists) and the `MagneticLattice(cell)` call.  The
element types Ocelot ships: `Drift`, `Quadrupole` (`k1`, `k2`, `tilt`), `Sextupole` (`k2`), `Octupole`
(`k3`), `Multipole` (`kn`, MAD's `knl`), `SBend`/`Bend` (sector faces, `gap` = 2·hgap, `fint`, `fintx`,
`tilt`, `k1`, `k2`), `RBend` (Ocelot adds `angle/2` to both faces itself: `rect`), `Solenoid` (`k` =
B/(2Bρ)), `Cavity` and `TWCavity` (`v` GV, `phi` deg, `freq`), `Hcor`/`Vcor`, `Marker`, `Monitor`,
`Aperture` (`xmax`, `ymax`, `dx`, `dy`, `type`), `Matrix` (`r11 … r66`, `b1 … b6`, transformed from Ocelot's
basis into the IR's at the entry energy).  `Undulator`, `TDCavity` and every other type have no IR kind
(DROPPED `UNSUPPORTED_OCELOT_ELEMENT`, a drift of their length with the parameters in `native['ocelot']`).
A definition placed several times is one IR element placed several times.

An Ocelot module carries no beam: the reference comes from the `# lattix: reference` tag lattix writes
(EQUIVALENT `REFERENCE_FROM_TAG`), from `read(..., species=, kinetic_energy_eV=)`, or — a native Ocelot
file — from `tws0.E`, an electron of that total energy (EQUIVALENT `REFERENCE_FROM_TWISS`).  Normalized
strengths become fields with the signed rigidity at each element's entrance (the energy follows the
cavities).  The `# lattix: name=… kind=…` tags restore the exact IR kinds, names, families, phases and
gains; the apertures, body drifts and zero-length `Vcor` written beside an element are folded back onto
it, and a surrogate cavity (`L=0 pad=…`) becomes the thin gap again with its length returned to the
neighbouring drifts (a drift of that length stands beside the gap when there is none: EQUIVALENT
`THIN_GAP_PAD_DRIFT`).  A `Cavity` tagged `kind=NCells` comes back as an `NCells` with the cavity's
voltage, phase and frequency (EQUIVALENT `NCELLS_FROM_CAVITY`); a `Matrix` with `delta_e` keeps only its
map (LOSSY `OCELOT_MATRIX_DELTA_E_DROPPED`) unless it is lattix's `ReferenceChange`.

A module that builds its lattice with loops or functions cannot be read this way: `read(...,
use_ocelot=True)` runs it in the Ocelot environment (GPL, out of process, through the oracle's worker)
and reads the resulting sequence (EQUIVALENT `OCELOT_EXECUTED`).

## Writing

| IR kind | Ocelot | Ledger |
|---|---|---|
| Drift | `Drift(l)` | EXACT |
| Quadrupole, Sextupole, Octupole | `Quadrupole(l, k1 = Bn1/Bρ_signed, k2, tilt)`, `Sextupole(l, k2)`, `Octupole(l, k3)`; a skew component is a tilt | EXACT (other orders LOSSY `MULTIPOLE_ORDERS_DROPPED`) |
| Multipole (thin) | `Multipole(kn = [BnL_n/Bρ_signed, …])` — MAD's `knl`, `kn[0]` a design bend | EXACT; skew orders LOSSY `OCELOT_SKEW_MULTIPOLE_DROPPED`; a thick one gets a `Drift` body (EQUIVALENT `THICK_MULTIPOLE_SPLIT`) |
| Bend | `SBend(l, angle, k1, k2, e1, e2, tilt, gap = 2·hgap, fint, fintx)` (sector faces; `rect` in the tag) | EXACT |
| Solenoid | `Solenoid(l, k = Bsol/(2 Bρ_signed))` | EXACT |
| RFCavity | `Cavity(l, v = V·1e-9, phi = −φs [deg], freq)`; a thin gap is a short surrogate cavity (`lattix.ir.rf.thin_gap_surrogate_length`, `thin_gap_length_m=` to override) that takes its length from the neighbouring drifts | EXACT; thin gap EQUIVALENT `THIN_GAP_AS_SHORT_CAVITY` (LOSSY `THIN_GAP_ADDS_LENGTH` when no drift can give the length) |
| FieldMap | the `lattix.ir.fieldmap` replacement ladder (`from=FieldMap` in the tag) | EQUIVALENT / LOSSY `FM_*` |
| NCells | `Cavity(l, v = the train's voltage, phi, freq)` | EQUIVALENT `NCELLS_AS_CAVITY` |
| RFQCell | `Drift` | LOSSY `RFQCELL_TO_DRIFT` |
| Kicker | `Hcor(l, angle = hkick)` + a zero-length `Vcor(angle = vkick)` | EXACT (EQUIVALENT `KICKER_SPLIT_HV` for both planes; LOSSY `EKICK_AS_MAGNETIC` for electric) |
| Collimator | `Aperture(xmax, ymax, dx, dy, type)` + `Drift` body | EXACT |
| Marker | `Marker()` | EXACT |
| Instrument | `Monitor(l)` (family in the tag) | EQUIVALENT `INSTRUMENT_AS_MONITOR` |
| Foil | `Marker` (material and thickness in the tag) | LOSSY `FOIL_TO_MARKER` |
| Taylor | `Matrix(l, r11 … r66, b1 … b6)` in Ocelot's basis at the entry energy | EXACT (EQUIVALENT `TAYLOR_BASIS_OCELOT` notes the transform) |
| Patch, Directive | `Marker` with the data in the tag | DROPPED `PATCH_DROPPED`, `FOREIGN_DIRECTIVE` |
| ReferenceChange | `Matrix(l = 0, delta_e = dE_ref [GeV])` (Ocelot advances the energy by `delta_e`) | EQUIVALENT `REFCHANGE_AS_MATRIX` |
| Freq | `Marker` (the frequency is per cavity) | EXACT |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Element apertures become `Aperture` elements at the element's ends (EQUIVALENT `APERTURE_AS_ELEMENT`;
Ocelot apertures carry `dx`/`dy`, so offset limits are exact); misalignments are `var.dx = …` /
`var.dy = …` attribute lines (the other components LOSSY `MISALIGN_DROPPED`).  Variable names are Python
identifiers (`bi4.bsw1l1.1` → `bi4_bsw1l1_1`, names Ocelot exports get an `e_` prefix; the `eid` and the
tag keep the original).  A definition placed several times is written once and listed several times in
`cell`; a drift that a surrogate cavity must shorten is cloned first when it is placed elsewhere too.
A reference that is not an electron or positron records LOSSY `OCELOT_ELECTRON_ONLY` (see below); a
thin gap does *not* get lattix's thin-gap RF-focusing lens (Ocelot's cavity has a model of its own).

## Units and conventions

* m, rad, GV (`Cavity.v`), Hz, degrees (`Cavity.phi`); `tws0.E` is the reference's **total** energy in
  GeV.  First-order maps in MAD-X's set `(x, px/p0, y, py/p0, τ, ΔE/(p0 c))` with `τ = c·Δt`
  **late-positive**: a 1 m drift gives `R56 = −L/(β²γ²)` (`Basis.OCELOT`, `_Z_SIGN = −1`;
  `tests/oracles/goldens/fingerprints.json`: drift map exact to 0.0 in the common basis).
* **Cavity**: `ΔE = v·cos(phi)` GeV for any species (no charge factor: Ocelot has no charge); the slope
  `R65 ∝ +sin(phi)` with `τ` late-positive means a late particle gains more for `phi > 0`, the opposite of
  the IR's `φ < 0` bunching convention — `phi = −φs` (measured 866 025.4 eV and `R65_common = −0.51` at
  `phi = +30°` for the 1 MV fingerprint gap).  The matrix (Rosenzweig–Serafini edges plus body) divides
  by the length: a zero-length `Cavity` raises `ZeroDivisionError` in Ocelot itself, so a thin gap is a
  short surrogate cavity.  Its transverse focusing scales with `V/(E·l)`: a 1 MV gap at 2.1 MeV gives
  `R21 = −1.48` over 2 cm and `−0.79` over the 37 mm surrogate — there is no length-free thin limit in
  this model, the surrogate length is a recorded choice.
* `k1 > 0` focuses `x` for every species and the solenoid rotation ignores the charge: `k1 =
  Bn1/Bρ_signed`, `k = Bsol/(2 Bρ_signed)` (`R11 = cos²(kL)`); `Multipole.kn` is MAD's `knl`
  (`R21 = −kn[1]`, the kick `Σ kn[n]·(x+iy)^n/n!`), and `kn[0]` is a *design* bend (`R26 = kn[0]`, the
  reference is not kicked); `Hcor`/`Vcor` kick `px += angle`.
* `SBend`: sector-referenced `e1`/`e2`, `gap` is the full gap (`hgap = gap/2`), `fint`/`fintx`, `tilt`;
  `RBend` adds `angle/2` to both faces itself.  Ocelot's survey follows MAD8 (a positive angle bends
  towards negative x).
* **Electron only.**  Every Ocelot map computes `γ = E/m_e` with Ocelot's electron mass (CODATA 1998,
  `510 998.867 eV`, 9e-8 below lattix's).  The oracle hands Ocelot the total energy that reproduces
  lattix's γ and reports lattix's kinetic energies advanced by exactly each transformation's ΔE, so
  drifts and magnets agree with the other engines for *any* species at the same γ; the cavity model and
  the longitudinal coupling are an electron's.  Writing another species records LOSSY
  `OCELOT_ELECTRON_ONLY` (normalized strengths stay exact) and the battery treats such pairs as report
  only.

## Known limits

* No foil, patch, RFQ cell or skew thin multipole in Ocelot; an `Instrument` is a `Monitor`; a
  traveling-wave cavity is written as a `Cavity` (the type in the tag).
* Thin gaps are surrogate cavities of a recorded length (`THIN_GAP_AS_SHORT_CAVITY`); a line without a
  drift beside the gap grows by that length (`THIN_GAP_ADDS_LENGTH`).
* Non-electron references are report only (`OCELOT_ELECTRON_ONLY`).
* A module that builds its lattice with code needs `use_ocelot=True` (runs it in the Ocelot
  environment).

## Oracle

`lattix/oracles/ocelot.py` + `ocelot_worker.py` (environment `ocelot`: `ocelot-desy` 25.6.0, a venv from
env `lattix`'s python 3.11 here, `LATTIX_OCELOT_PYTHON` elsewhere; marker `oracle_ocelot`): every
element's transformations `elem.R(E)` composed as `MagneticLattice.transfer_maps` does, the energy advanced
by each transformation's `get_delta_e()`.  Measured: HELIX's `fodo.madx` with a 1 GeV electron vs cpymad
1.8e-15 on the transverse block and every longitudinal block below 1e-15; the fingerprint gap gains
866 025.4037 eV (1e-15) and bunches; the XFEL S2E Elegant deck (`XFEL_elegant_TD1_S2E.lte`, 4126
elements, 2.16 km, 130 MeV → 17.5 GeV) read by lattix's Elegant reader and written as an Ocelot module
against Ocelot's own `ElegantLatticeConverter` on the same file: 3284 shared boundaries, T4×4 2.9e-12,
reference energy 2.4e-15 (`tests/oracles/test_ocelot_adapter.py`, local: the deck lives in the Ocelot
clone).  The oracle also reads an Elegant `.lte` through that converter (`fmt="elegant"`).
