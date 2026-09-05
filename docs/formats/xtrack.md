# xtrack

xtrack (CERN) lattices are Python `Line` objects serialised as JSON (`Line.to_json`).  lattix
registers `.json` (sniffed for the xtrack shape; `.lattix.json` is lattix's own IR
serialisation) and converts in both directions through `lattix.formats.xtrack.to_line` /
`from_line`, measured on xtrack 0.103.5 and 0.112.0.

## Reading

`from_line` walks `line.elements` with a table per xtrack class: `Drift`, `Quadrupole`,
`Sextupole`, `Octupole`, `Bend` (`h`, `k0`, `edge_entry_angle`, `edge_exit_angle`,
`rot_s_rad`), `Multipole` (`knl`, `ksl` — a lone dipole term is a `Kicker`), `Solenoid` /
`UniformSolenoid`, `Cavity` (`lag` in degrees plus `phase` in radians, whose effects add),
`LimitRect` / `LimitEllipse` (`Collimator`), `Marker`, `XYShift` / `SRotation` / `XRotation` /
`YRotation` (`Patch`), `ZetaShift` (EQUIVALENT `ZETASHIFT_AS_REFCHANGE`),
`ReferenceEnergyIncrease` (`ReferenceChange`), `FirstOrderTaylorMap` (`Taylor`, basis tag
`TAYLOR_BASIS_XTRACK`); everything else is DROPPED `UNSUPPORTED_XTRACK_ELEMENT`.  Thin slices
are merged back where the provenance says so (`SLICE_MERGED`).  `line.metadata["lattix"]`,
written by `to_line`, restores the exact IR kinds and names; a foreign line (`xt.load`,
`from_madx_sequence`) falls back to xtrack's own conventions.  `particle_ref` gives the
species and energy; a line without one records `RF_FREQUENCY_MISSING` on cavities.

## Writing

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Solenoid | the xtrack class of the same name; `tilt` → `rot_s_rad`, explicit skew `Bs[n]` → `k1s`/`k2s`/`k3s` | EXACT |
| Multipole | `Multipole(knl, ksl)` | EXACT |
| Bend | sector `Bend(length, angle, edge_entry_angle, edge_exit_angle)`; a rectangular source already has θ/2 in e1/e2 | EXACT |
| RFCavity | `Cavity(voltage, frequency, lag = φ° + 90)` | EQUIVALENT `CONST_P0` + `CONST_P0_DELTA_RIGIDITY` (or `…_LOCAL_RIGIDITY` / `…_START_RIGIDITY`) |
| FieldMap | thick `Cavity` with the map's voltage (a drift when none is known) | EQUIVALENT `FM_AS_CAVITY` |
| NCells, RFQCell | `Drift` | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `Multipole(knl=[−hkick], ksl=[+vkick])`, `isthick=True` for a body length | EXACT |
| Collimator | `LimitRect` / `LimitEllipse` | EXACT |
| Marker | `Marker` | EXACT |
| Instrument | `Drift` of the same length (marker when thin) | EQUIVALENT `MONITOR_AS_DRIFT` |
| Taylor | `FirstOrderTaylorMap` used verbatim in xtrack's basis | EQUIVALENT `TAYLOR_BASIS_XTRACK` |
| Patch | `XYShift`, `SRotation`, `XRotation`, `YRotation` | EXACT |
| ReferenceChange | `ReferenceEnergyIncrease` | EQUIVALENT `REFCHANGE_AS_P0C` |
| Foil | `Marker` | LOSSY `FOIL_TO_MARKER` |
| Freq | nothing (frequency is per cavity) | EXACT |
| Directive | `Marker` | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | consecutive elements | LOSSY `SUPERPOSITION_FLATTENED` |

## Units and conventions

* m, rad, V, Hz; `Cavity.phase = φ + π/2` ≡ `lag = φ° + 90` (measured: `Cavity(1 MV, lag=60°)`
  on a 2.1 MeV proton gains +V·cos 30° to 5e-14 in both releases; `lag` is deprecated in
  0.112, so the writer sets `phase`).
* `rot_s_rad` is exactly MAD-X `tilt` (2.2e-16); `(k1, k1s)` is *not* the same thing in
  xtrack (7.6e-4 on a thick quadrupole), so only explicit skew components become `k1s`.
* `shift_x`/`shift_y`/`shift_s` are MAD-X `dx`/`dy`/`ds`, `rot_s_rad_no_frame` is `dpsi`;
  drifts and markers have none (EQUIVALENT `SHIFT_NO_OP`).
* `XYShift(dx)` maps `x → x − dx`, `SRotation` rotates the frame by +angle, `ZetaShift(dzeta)`
  maps `ζ → ζ − dzeta` — the IR `Patch` convention used by `lattix.ir.lattice.survey`.
* `Bend.h` cannot be assigned; `k0` reads back as the string `'from_h'`.
* Longitudinal basis `(ζ, δ)`, ζ ahead-positive; p0 constant through RF.

## Known limits

* Thick kickers need `isthick=True` or 7.5 m of the PSB vanish.
* xtrack 0.112 `Line.from_madx_sequence` is broken (`'AttrDict' object has no attribute
  'Line'`); the adapter falls back to `MadLoader(classes=xt, allow_thick=True)`.
* Instruments have no xtrack element.

## Oracle

`lattix/oracles/xtrack.py`: `twiss(method='4d')` R-matrices by finite differences
(`h = 1e-5` with the read-back perturbation as denominator), `survey`, tracking; marker
`oracle_xtrack`.  The PSB `psb.seq` through lattix vs `Line.from_madx_sequence` agrees on
every block to 0.0 over 304 boundaries.
