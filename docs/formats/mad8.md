# MAD8 flat files

The MAD8 `SAVELINE` / hand-written flat dialect (`.lat`, `.FLAT`) is what the PIP-II BTL and
BAL optics decks use: `name: TYPE, attr=expr, …` definitions, `name: LINE=(…)` with nesting,
reflection and repetition, lazily resolved `:=` parameters and `NAME[ATTR]` references.
lattix registers `.lat` and `.flat` (a `.lat` in the TraceWin or FLAME dialect is sniffed and
routed to that reader) with a reader and a writer.

## Reading

`lattix/formats/mad8/reader.py` ports HELIX's `mad8_parser.py` (`_logical_lines`, the lazy
`_Mad8File` resolver with memoisation and cycle detection, `_expand`, `_root_line`,
`_declare_periods`) onto the IR:

* `!` comments, `&` continuations, definitions anywhere in the file, class inheritance
  (`QF2: QF, K1=…`), the whole `lattix.ir.expr` function whitelist (the BAL deck's
  `P0 := SQRT(E0*(2*MASS+E0))` is where HELIX's resolver stopped);
* root line: the `USE` target, else the unreferenced `LINE` with the largest expansion
  (EQUIVALENT `AMBIGUOUS_ROOT_LINE`);
* rigidity resolution order: `brho=` argument → a `BRHO := …` parameter (EQUIVALENT
  `RIGIDITY_FROM_BRHO`) → a `BEAM` statement → a hard error, never a silent default; a
  disagreeing `BEAM` adds `RIGIDITY_CONFLICT`;
* species from `BEAM` (`PARTICLE`/`MASS`/`CHARGE`), else the `species=` option (default `h-`,
  the PIP-II lineage);
* `RBEND L` is the **arc** (MAD8 has no `OPTION, RBARC`), pole faces are chord-referenced so
  the IR gets `e1 += angle/2` with `BendP.rect = True`; vertical bends keep `TILT = ±π/2` as
  `tilt_ref` (EXACT); `KICKER`/`HKICKER`/`VKICKER` become one thick `Kicker` keeping both the
  deflection and the body length; `HMONITOR`/`VMONITOR` are BPMs, plain `MONITOR` a generic
  instrument with the MAD8 type kept in `native`;
* with `auto_periods=True` FODO-type cells are bracketed with `LATTICE` / `LATTICE_END`
  directives where HELIX's `_declare_periods` puts them.

## Writing

`lattix/formats/mad8/writer.py` is the MAD-X writer in MAD8 dialect: `&` continuations,
16-character names (`MAX_NAME_LEN = 16`, renames recorded as `MAD8_NAME`), `LINE` mode, one
rigidity per deck.

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Multipole, Solenoid, Kicker, Collimator, Marker, Instrument | the MAD8 element of the same name | EXACT |
| Bend | `SBEND L ANGLE E1 E2 FINT HGAP K1 TILT` | EXACT; a different exit integral is LOSSY `FINTX_DROPPED` |
| RFCavity, FieldMap | `RFCAVITY VOLT LAG FREQ` (map → thick cavity) | EQUIVALENT `CONST_P0` + `CONST_P0_START_RIGIDITY`, `FM_TO_CAVITY` |
| NCells, RFQCell | drift | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Taylor | marker or drift of the same length | LOSSY `TAYLOR_DROPPED` |
| Foil, Patch, ReferenceChange | marker | LOSSY `FOIL_TO_MARKER`, `PATCH_DROPPED`, `REFCHANGE_DROPPED` |
| Freq | comment | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | consecutive elements | LOSSY `SUPERPOSITION_FLATTENED` |

## Units and conventions

* m, rad, MV, MHz, `LAG` in turns as in MAD-X; normalized strengths with the signed
  rigidity at the lattice start (MAD8 has one `BRHO`).
* `BRHO` decks: the legacy PIP-II conversion header `variable mad2tw -4.8828922` is the
  external anchor for the H⁻ sign.
* The BAL deck's `BRHO := P0/C*1.0E11` is 1000× too large (never referenced) — read the
  rigidity from `BEAM` there.

## Known limits

* MAD8 itself is not installed here; there is no `oracle_mad8`.  The dialect is validated
  through the lockstep anchors (BTL MAD8 vs the PIP-II TraceWin export: lengths 4e-10 m,
  gradients 1.4e-10, angles 7e-12 over 873 magnets) and by running the MAD-X translation
  through cpymad.
* No `FINTX`, no `MATRIX` element, no reference-energy change.

## Oracle

None native.  `tests/formats/test_mad8_anchor.py` (structure) and the Phase-2 gate A4
(`BTL2025v0703.lat` → `.dat` / `.madx`, HELIX vs cpymad to 1e-7 over 308 m) stand in;
marker `corpus` (private decks).
