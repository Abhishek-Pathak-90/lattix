# HELIX adapter

HELIX is the private (GPL-3) linac design package whose readers lattix descends from.
`lattix/formats/helix.py` is an *adapter*, not a file format: it converts between the IR and
HELIX's in-memory `linac_gen.Lattice` so that HELIX can import and export every lattix
format and so that HELIX's matrix tracking can serve as an oracle.  It is deliberately
absent from the `FORMATS` registry, never a dependency, and imported at call time from
`HELIX_ROOT`.

## Reading

`from_helix(lattice)` maps HELIX element classes to IR kinds: `Drift`, `Quad` (with `g3..g6`),
`Dipole` + adjacent `Edge` (EQUIVALENT `BEND_EDGES_FROM_DIPOLE` when the edges come from the
dipole itself; LOSSY `BEND_EDGE_CONFLICT` when both are present — the EDGE cards win),
`Solenoid`, `RFGap` (thin gaps read as a raw RF phase: EQUIVALENT `SYNC_MODE_ASSUMED`),
`FieldMap` (`FM_FILES_UNKNOWN` for maps built in memory), `NCells`, `RFQCell`, `Steerer`,
`Aperture` (types other than rectangle/ellipse LOSSY `APERTURE_TYPE_UNMODELLED`), markers and
diagnostics, `Foil`, `ThinLens` (EQUIVALENT `THINLENS_AS_TAYLOR`, kept in HELIX's basis:
`MATRIX_BASIS_HELIX`), `VaneRFQ` (LOSSY `VANE_RFQ_GEOMETRY_NOT_CONVERTED`), lattice commands
(`SET_BEAM_E0_P0` phase shifts LOSSY `REF_PHASE_SHIFT_NOT_CONVERTED`), `ERROR_*` studies
(LOSSY `ERROR_STUDY_NOT_CONVERTED`, kept in `meta["helix_errors"]`).  Unknown classes are
DROPPED `HELIX_ELEMENT_UNKNOWN`.

## Writing

`to_helix(lattice)` builds a `linac_gen.Lattice` from the IR through a `RULES` table over
all 22 kinds:

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Solenoid, Marker, Instrument, Kicker, Collimator, Freq, ReferenceChange | the HELIX element of the same role | EXACT |
| Sextupole, Octupole | HELIX `Quad` with `g3`/`g4` | EQUIVALENT `THICK_POLE_AS_QUAD_HIGHER_ORDER` |
| Bend | `Dipole` + two `Edge`s; horizontal or vertical only | LOSSY `BEND_TILT_UNSUPPORTED` otherwise; a zero-angle bend becomes a drift (`BEND_ZERO_ANGLE`) |
| RFCavity | thin `RFGap` (synchronous phase emitted as the equivalent raw phase: EQUIVALENT `SYNC_PHASE_AS_RAW`); thick: drift + gap + drift | EQUIVALENT `THICK_CAVITY_SPLIT` |
| FieldMap | HELIX `FieldMap` when files and geometry are known, else a drift | LOSSY `FM_FILES_MISSING` |
| NCells, RFQCell | rebuilt from the TraceWin operands, else a drift | DROPPED `NCELLS_PARAMS_MISSING`, LOSSY `RFQ_CELL_NOT_REBUILT` |
| Taylor | `ThinLens` from the 6×6 map | EQUIVALENT `THINLENS_AS_TAYLOR` |
| Patch | marker | LOSSY `PATCH_NOT_SUPPORTED` |
| Directive | marker | LOSSY `DIRECTIVE_DROPPED` |
| Superposition | rebuilt `SUPERPOSE_MAP` cluster, else a drift | LOSSY `SUPERPOSE_NOT_REBUILT` |

## Units and conventions

* HELIX works in mm / deg / MeV / MHz / T / T·m; the IR is SI + eV.
* HELIX's thin `RFGap.advance_ref` uses the signed charge (`dW = q·V·T·cos φ`), so an H⁻
  deck's raw phase differs from the IR synchronous phase by π; the adapter applies
  `tracewin_phase_deg` / `phase_from_tracewin_deg` with `sync=False`.
* HELIX ignores `SET_SYNC_PHASE` on thin gaps (`tracewin_parser.py:701`).
* Longitudinal basis `(Δφ deg, ΔW MeV)`, late-positive; p0 follows.

## Known limits

* HELIX's `Dipole` has no reference tilt: only horizontal and vertical bends survive.
* Field errors, per-seed error studies and the RFQ vane geometry are not converted.
* Anything in this page depends on the HELIX version checked out at `HELIX_ROOT`; the
  adapter is exercised only when that directory exists.

## Oracle

`lattix/oracles/helix.py` calls HELIX's `get_element_matrix` / `compute_transfer_matrix`
in-process; marker `oracle_helix`, needs `HELIX_ROOT`.  Lockstep test:
`tests/formats/test_tracewin_vs_helix_lockstep.py` reads the same TraceWin deck through both
parsers and compares per-element matrices to 1e-10.
