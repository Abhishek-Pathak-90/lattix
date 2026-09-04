# FLAME (GLPS)

FLAME (FRIB) reads GLPS decks: `name: type, key = value, …;` element definitions,
`name: LINE = (…);`, `USE: name;`, global assignments (`IonEk`, `IonEs`, `IonZ`,
`IonChargeStates`, `NCharge`, `SampleFreq`, `sim_type`), vectors and strings.  lattix
registers `.flame.lat` and `.lat` (sniffed) with a reader and a writer, checked against
FLAME 1.9.2 built from the clone.

## Reading

`lattix/formats/flame/reader.py` is a hand-port of FLAME's `src/glps.y` / `glps.l`:
identifiers may contain `:` (EPICS names), numbers have no leading sign, `#` comments,
`2*cell` repeats and `-cell` reverses, built-in functions (`sin … deg2rad rad2deg file …`).
The element table (`src/moment.cpp`, `sphinx_doc/element.rst`):

| FLAME | IR |
|---|---|
| `source` | `Marker` at the start of the line (the beam state is kept in `meta`) |
| `marker`, `bpm` | `Marker`, `Instrument(BPM)` |
| `drift L` | `Drift` |
| `orbtrim theta_x theta_y` (or `realpara = 1` with `tm_xkick`/`tm_ykick` in T·m, `xyrotate`) | `Kicker` |
| `quadrupole L B2`, `sextupole L B3`, `solenoid L B` | `Quadrupole`, `Sextupole`, `Solenoid` (lab fields) |
| `sbend L phi phi1 phi2 dphi1 dphi2 K bg ver` | `Bend` (+ `Bn[1]`) |
| `rfcavity L cavtype f phi scl_fac syncflag` | `RFCavity` (LOSSY `FLAME_CAVTYPE_VOLTAGE_UNKNOWN`: the gain is a tabulated TTF model) |
| `stripper` | `Foil` (EQUIVALENT `FLAME_STRIPPER_MODEL`) |
| `tmatrix matrix` (7×7) | `Taylor` |
| `edipole`, `equad` | marker + DROPPED `ELECTROSTATIC_UNSUPPORTED` |

The reference particle comes from `IonEk` (MeV/u), `IonEs` (eV/u) and the charge states
(`FLAME_NO_IONEK`, `FLAME_NO_IONES`, `FLAME_NO_CHARGE_STATE`, `MULTI_CHARGE_STATE_FIRST` when
several are given).

## Writing

`lattix/formats/flame/writer.py` writes the globals (`sim_type = "MomentMatrix"`, energies
per nucleon with A = 1 for species without a mass number — EQUIVALENT `FLAME_PER_NUCLEON`),
a unit `BaryCenter0`/`S0` state and a prepended `source` element (EQUIVALENT
`FLAME_SOURCE_ADDED`), then the elements and the `LINE`/`USE`.

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Solenoid, Marker | `drift`, `quadrupole B2 roll`, `sextupole B3`, `solenoid B`, `marker` | EXACT |
| Octupole | drift | LOSSY `OCTUPOLE_TO_DRIFT` |
| Multipole | `orbtrim` for the dipole term | LOSSY `MULTIPOLE_TO_ORBTRIM`, `MULTIPOLE_ORDERS_DROPPED` |
| Bend | `sbend L phi phi1 phi2 K ver` | LOSSY `BEND_FRINGE_DROPPED` (no fint/hgap), `BEND_TILT_DROPPED` for tilts other than 0/±π/2 |
| RFCavity | `rfcavity` only when the element carries a FLAME `cavtype`; else a drift | LOSSY `RFCAVITY_NEEDS_CAVTYPE` |
| FieldMap, NCells, RFQCell, Superposition | drift | LOSSY `FM_TO_DRIFT`, `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT`, `SUPERPOSITION_FLATTENED` |
| Kicker | `orbtrim theta_x theta_y` (+ a following drift for the body) | LOSSY `THIN_TYPE_LENGTH_PADDED` when thick |
| Collimator | `marker` with `aper` (+ drift) | LOSSY `COLLIMATOR_TO_MARKER` (FLAME never reads `aper`) |
| Instrument | `bpm` (+ drift) | EXACT / `THIN_TYPE_LENGTH_PADDED` |
| Foil | `stripper` | EQUIVALENT `FOIL_AS_STRIPPER` |
| Taylor | `tmatrix` (+ drift) | EXACT / `THIN_TYPE_LENGTH_PADDED` |
| Patch, ReferenceChange | marker | LOSSY `PATCH_DROPPED`, `REFCHANGE_DROPPED` |
| Freq | comment | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` |

A deck with cavities but no `Eng_Data_Dir` is LOSSY `FLAME_NO_ENG_DATA_DIR` (FLAME falls back to
its built-in tables).

## Units and conventions

* Lengths in m, `phi`/`phi1`/`phi2` in degrees, `B2` in T/m, `B3` in T/m², `B` in T;
  `sbend K` is normalized (1/m², `Kx = K + 1/ρ²`, `Ky = −K`); `roll` **is** MAD-X `tilt`
  (9 digits); `orbtrim theta_x → +x′`.
* `rfcavity phi` is a synchronous phase when `syncflag ≥ 1`, but the gain is a tabulated TTF
  polynomial (3.2 % off `V·cos φ` at −35°).
* Native basis `(x mm, x′, y mm, y′, φ rad late-positive w.r.t. SampleFreq, ΔEk MeV/u)`,
  per nucleon and per charge state; p0 follows.

## Known limits

* No thin multipole, octupole, field map, fringe integrals, reference tilt, patch.
* `aper` is read nowhere in FLAME's source; collimation is not simulated.
* PyPI ships only manylinux x86_64 wheels; on macOS FLAME is built from the clone.

## Oracle

`lattix/oracles/flame.py`: `Machine.propagate` per-element `transmat` in-process or through a
`-I` worker (`LATTIX_FLAME_ENV` / `LATTIX_FLAME_PYTHON`); marker `oracle_flame`.  `fodo.madx`
agrees with cpymad to 3.6e-15; the FRIB `LS1`, `LS1FS1` and `ALL_lattice` decks
read → write → FLAME bit-exact.
