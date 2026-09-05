# xtrack

xtrack (CERN) lattices are Python `Line`/`Environment` objects serialised as JSON (`Line.to_json`,
`Environment.to_json`).  lattix registers `.json` (sniffed for the xtrack shape; `.lattix.json` is
lattix's own IR serialisation) and converts in both directions through
`lattix.formats.xtrack.to_line` / `from_line` (and `to_environment` for nested lines), measured on
xtrack 0.103.5 and 0.112.0.

## Reading

`from_line` walks `line.elements` with a table per xtrack class.  Every one of the 102
`BeamElement` classes xtrack 0.112 ships is in that table (a test enumerates `xt.__dict__`):

* **Magnets**: `Drift`, `Quadrupole`, `Sextupole`, `Octupole`, `Bend` (`h`, `k0`,
  `edge_entry_angle`, `edge_exit_angle`, `rot_s_rad`), `RBend` (arc length from
  `length_straight`; face-referenced `e1`/`e2` become the IR's sector-referenced angles, `rect`
  set), `Magnet` (EQUIVALENT `MAGNET_AS_BEND` when it has curvature, otherwise a thick multipole
  magnet, `MAGNET_AS_MULTIPOLE_MAGNET`), `Multipole` (`knl`, `ksl` — a lone dipole term is a
  `Kicker`), `SimpleThinQuadrupole`/`SimpleThinBend`, `Wedge` (LOSSY `WEDGE_AS_THIN_DIPOLE`),
  `Solenoid`/`UniformSolenoid`, `VariableSolenoid` (mean `ks`, LOSSY `VARIABLE_SOLENOID_MEAN`).
* **Edges and misalignments**: `DipoleEdge`/`MagnetEdge` next to a bend are folded into it
  (EQUIVALENT `DIPEDGE_FOLDED`); a lone one is a thin matrix (`DIPEDGE_AS_MATRIX`, re-emitted
  verbatim); `MultipoleEdge` is DROPPED `EDGE_DROPPED`.  A `Misalignment` pair around a magnet is
  the magnet's `BodyShiftP`; an unpaired one is a `Patch` (EQUIVALENT `MISALIGNMENT_AS_PATCH`).
* **RF**: `Cavity` (`lag` in degrees plus `phase` in radians, whose effects add), `RFMultipole`
  (RFCavity, LOSSY `RF_MULTIPOLE_TERMS_DROPPED` when it has multipole terms, else EQUIVALENT
  `RF_MULTIPOLE_AS_CAVITY`); `CrabCavity`, `ACDipole`, `Elens`, `Wire`, `Exciter`, `NonLinearLens`,
  `ElectronCooler`, `LineSegmentMap` and the other beam–beam/space-charge classes are DROPPED with
  their own codes (`CRAB_CAVITY_DROPPED`, `ELENS_DROPPED`, `NON_LATTICE_ELEMENT`, …).
* **Apertures and monitors**: `LimitRect`/`LimitEllipse` (`Collimator`), `LimitPolygon`,
  `LimitRectEllipse`, `LimitRacetrack` (bounding box, LOSSY `APERTURE_SHAPE`),
  `LongitudinalLimitRect` (DROPPED `LONGITUDINAL_APERTURE_DROPPED`); `BeamPositionMonitor` and the
  other monitors are `Instrument`s (EQUIVALENT `MONITOR_AS_INSTRUMENT`); `Marker`.
* **Frame and reference**: `XYShift`/`SRotation`/`XRotation`/`YRotation`/`Rotation`/`Translation`
  (`Patch`), `ZetaShift` (EQUIVALENT `ZETASHIFT_AS_REFCHANGE`), `TimeDelay`,
  `ReferenceEnergyIncrease` and `ReferenceEnergyChange` (`ReferenceChange`, `REFCHANGE_AS_P0C`),
  `FirstOrderTaylorMap` (`Taylor`, basis tag `TAYLOR_BASIS_XTRACK`), `SecondOrderTaylorMap`
  (linear part, LOSSY `TAYLOR_ORDER_TRUNCATED`, native payload kept for xtrack → xtrack).
* Thick and thin **slices** are merged back where the provenance says so (`SLICE_MERGED`).

`line.metadata["lattix"]`, written by `to_line`, restores the exact IR kinds and names; a foreign
line (`xt.load`, `from_madx_sequence`) falls back to xtrack's own conventions.  `particle_ref` gives
the species and energy; a line without one records `RF_FREQUENCY_MISSING` on cavities.

**Knobs.**  `line.vars` and the `_var_manager` expressions of the JSON become IR `variables` and
`Expr`s on the element attributes they drive (`k1`, `knl[n]`, `ksl[n]`, `voltage`, `lag`, `angle`,
`length`, `hkick`/`vkick` …; EXACT `KNOBS_READ`), so MAD-X `:=` knobs survive MAD-X → xtrack →
MAD-X (the PS Booster's 128 knobs come back identical).

**Environments.**  An `Environment` JSON with several lines is read as nested IR `lines` (the root
is `metadata.lattix.use`, else the line named on the command line, else the longest); sub-lines
used reversed are expanded (`REVERSED_LINE_EXPANDED`).

## Writing

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Solenoid | the xtrack class of the same name; `tilt` → `rot_s_rad`, explicit skew `Bs[n]` → `k1s`/`k2s`/`k3s` | EXACT |
| Multipole | `Multipole(knl, ksl)` | EXACT |
| Bend | sector `Bend(length, angle, edge_entry_angle, edge_exit_angle, edge_entry_fint, edge_entry_hgap)`; a rectangular source already has θ/2 in e1/e2.  `rbend=True` writes rectangular sources as `RBend(length_straight, angle, face-referenced e1/e2)` (EQUIVALENT `RBEND_WRITTEN`); `bend_model=` / `edge_model=` set xtrack's `model` and `edge_entry_model`/`edge_exit_model` (EQUIVALENT `BEND_MODEL_OPTION`) | EXACT |
| RFCavity | `Cavity(voltage, frequency, lag = φ° + 90)` | EQUIVALENT `CONST_P0` + `CONST_P0_DELTA_RIGIDITY` (or `…_LOCAL_RIGIDITY` / `…_START_RIGIDITY`) |
| FieldMap | thick `Cavity` with the integrated map's `V_c` at its synchronous phase (dE_ref = V_c·cos φs; the card phase of a relative-phase map is not φs), a drift when nothing is known | EQUIVALENT `FM_AS_CAVITY` |
| NCells, RFQCell | `Drift` | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `Multipole(knl=[−hkick], ksl=[+vkick])`, `isthick=True` for a body length | EXACT |
| Collimator | `LimitRect` / `LimitEllipse` | EXACT |
| Marker | `Marker` | EXACT |
| Instrument | `Drift` of the same length (marker when thin) | EQUIVALENT `MONITOR_AS_DRIFT` |
| Taylor | `FirstOrderTaylorMap` used verbatim in xtrack's basis; a native `DipoleEdge` / `SecondOrderTaylorMap` payload is re-emitted as is | EQUIVALENT `TAYLOR_BASIS_XTRACK` |
| Patch | `XYShift`, `SRotation`, `XRotation`, `YRotation` | EXACT |
| ReferenceChange | `ReferenceEnergyIncrease` | EQUIVALENT `REFCHANGE_AS_P0C` |
| Foil | `Marker` | LOSSY `FOIL_TO_MARKER` |
| Freq | nothing (frequency is per cavity) | EXACT |
| Directive | `Marker` | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | consecutive elements | LOSSY `SUPERPOSITION_FLATTENED` |

Variables become `line.vars` (dependent ones as xdeps expressions) and every element attribute
whose IR expression still gives the written number becomes an `element_refs` expression (EXACT
`KNOBS_WRITTEN`; an expression xdeps cannot evaluate keeps its value, EQUIVALENT
`KNOB_EXPRESSION_DROPPED`).  A lattice with nested lines is written as an `Environment` JSON
(`document="environment"`, automatic for nested sources) whose `metadata.lattix.use` names the
root; `document="line"` flattens.  `lattix convert --to madng` renders the same `Line` through
xtrack's MAD-NG writer (`docs/formats/madng.md`).

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
* `Bend.h` cannot be assigned; `k0` reads back as the string `'from_h'`.  `RBend.edge_entry_angle`
  is relative to the rectangular face (xtrack's own MAD-X loader convention; a hand-built
  `RBend(length_straight=0.8, angle=0.08, e1=0.01, e2=0.02)` matches cpymad's `rbend` to 5.1e-10).
* Longitudinal basis `(ζ, δ)`, ζ ahead-positive; p0 constant through RF (the `delta` energy mode
  normalizes every strength with the momentum xtrack's own particle has, `docs/conventions.md` §3).

## Known limits

* Thick kickers need `isthick=True` or 7.5 m of the PSB vanish.
* xtrack 0.112 `Line.from_madx_sequence` is broken (`'AttrDict' object has no attribute
  'Line'`); the adapter falls back to `MadLoader(classes=xt, allow_thick=True)`.
* Instruments have no xtrack element; beam–beam, space-charge, cooler, wire and exciter elements
  have no IR kind and are dropped with a named code.
* `Magnet` edge and body models beyond `Bend`'s (`k2`, `k3`, `knl/ksl` in one thick element) are
  kept as a thick multipole magnet; second-order Taylor terms are not tracked by the IR.
* Knob expressions that need functions xdeps lacks keep their numbers.

## Oracle

`lattix/oracles/xtrack.py`: `twiss(method='4d')` R-matrices by finite differences centred on the
tracked reference orbit (`h = 1e-5` with the read-back perturbation as denominator), `survey`,
tracking; marker `oracle_xtrack`.  The PSB `psb.seq` through lattix vs `Line.from_madx_sequence`
agrees on every block to 0.0 over 304 boundaries; the `rbend=True` output matches cpymad to
5.1e-10 on the transverse block.
