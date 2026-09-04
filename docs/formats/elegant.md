# Elegant `.lte`

Elegant (APS) lattice files are Fortran-namelist-like element definitions
(`NAME: TYPE, ATTR=value, …`), `LINE=(…)` beam lines with `n*(…)` repetition, RPN `%` store
lines and `NAME[ATTR] = value` overrides.  lattix registers `.lte` with a reader and a
writer; both were pinned against elegant 2026.3.0.

## Reading

`lattix/formats/elegant/reader.py` ports HELIX's tokenizer, RPN store lines, element
templates, overrides and line expansion, and applies the conventions measured on the engine:

* `RBEN`/`RBEND` `L` is the **chord**: the reader converts to the arc and adds θ/2 to both
  pole faces, exactly what elegant's own `&save_lattice` does (EQUIVALENT `RBEN_CHORD_TO_ARC`);
* `FINT` defaults to **0.5**; `FINT1`/`FINT2` (−1 = use `FINT`) only on `CSBEND`/`CSRCSBEND`;
* `RFCA` phase is charge-signed (`phase_from_elegant_deg`); an `.lte` names no species, so
  `species` is a reader option and EQUIVALENT `SPECIES_ASSUMED` is recorded whenever the deck
  has RF; `FREQ` defaults to 500 MHz (EQUIVALENT `RFCA_DEFAULT_FREQ`) and `CHANGE_P0` to 0
  (`RFCA_NO_P0_CHANGE`);
* `!` starts a comment, `&` continues, `;` is *not* a separator, a trailing `,` continues
  only when the next line does not start a statement; `#include` files are followed when
  present (`ELEGANT_INCLUDE`, else `INCLUDE_NOT_FOLLOWED`);
* `KQUAD` and `QUAD` have the same linear matrix; per-type misalignment attributes
  (`DX DY DZ TILT`, `PITCH`, `YAW`, `ETILT`) become `BodyShiftP` where the type accepts them;
  `WATCH`, `MONI`, `HMON`, `VMON` are instruments; `MAXAMP` is read as a collimator; unknown
  types are DROPPED `UNSUPPORTED_ELEGANT_TYPE`.

## Writing

`lattix/formats/elegant/writer.py` writes upper-case definitions, one `LINE` for the root
(sub-lines kept) and comments for directives.

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Solenoid | `DRIF`, `KQUAD`, `SEXT`, octupole via `MULT`, `SOLE` | EXACT |
| Multipole | one `MULT` per order at the same position | EQUIVALENT `MULT_SPLIT_BY_ORDER`; skew terms LOSSY `SKEW_MULTIPOLE_DROPPED` |
| Bend | `CSBEND L ANGLE E1 E2 FINT HGAP K1 TILT` (`RBEN` only for a rectangular source) | EXACT |
| RFCavity | `RFCA VOLT PHASE FREQ CHANGE_P0=1` | EQUIVALENT `RFCA_CHANGE_P0`, `ELEGANT_PHASE_FOR_SPECIES`; LOSSY `NCELL_DROPPED` when `n_cell` was set |
| FieldMap | thick `RFCA` with the map's voltage and synchronous phase | EQUIVALENT `FM_TO_CAVITY` |
| NCells, RFQCell | drift | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `KICKER` / `HKICK` / `VKICK` (electric variants when the source was electric) | EXACT |
| Collimator | `RCOL` / `ECOL` | EXACT |
| Marker, Instrument | `MARK`, `MONI`/`HMON`/`VMON`/`WATCH` by family | EXACT; families without a type EQUIVALENT `INSTRUMENT_AS_MARKER` |
| Taylor | `EMATRIX` | EXACT |
| Foil, Patch, ReferenceChange | `MARK` | LOSSY `FOIL_TO_MARKER`, `PATCH_DROPPED`, `REFCHANGE_DROPPED` |
| Freq | comment | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | consecutive elements | LOSSY `SUPERPOSITION_FLATTENED` |

Definitions not referenced by the root line are still written, normalized with the
lattice-start rigidity (EQUIVALENT `DEFINITION_NOT_IN_LINE`).

## Units and conventions

* m, rad, `VOLT` in V, `FREQ` in Hz, `PHASE` in degrees: crest at +90° for negative species
  and −90° for positive ones — measured by tracking: proton `PHASE=240` and H⁻ `PHASE=60`
  both give +V·cos 30°.
* `CHANGE_P0=1` makes elegant's reference momentum follow the gain (its default is 0).
* Longitudinal basis `(s, δ)` with s the path length (late-positive); the adapter adds the
  `−L/γ²` drift term to compare with the common basis.
* Normalized strengths with the signed rigidity at each element's entrance.

## Known limits

* `RFCA` has no cell count and `MULT` no skew term.
* The species is not stored in an `.lte`: reading a deck written for protons as H⁻ changes
  the RF phase and the sign of every normalized strength.
* Space-charge and CSR flavours (`CSRCSBEN`, `RFCW`) are read as their linear equivalents.

## Oracle

`lattix/oracles/elegant.py` generates an `.ele` (`&run_setup`, `&twiss_output matched=0`,
`&matrix_output individual_matrices=1`, `&floor_coordinates`, `&track`) and parses the SDDS
output with `pysdds`; marker `oracle_elegant`, binary from `PATH` or `ELEGANT_EXE`.  Gate A2
(elegant leg) agrees with cpymad to 2.8e-10 on the transverse blocks.
