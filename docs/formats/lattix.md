# lattix JSON (`.lattix.json`)

The package's own lossless serialisation of the intermediate representation: the pydantic
models of `lattix.ir` dumped as JSON, with every element definition, line, variable,
command, error study, reference particle and provenance field.  lattix registers
`.lattix.json` (a plain `.json` is sniffed: the xtrack shape goes to the xtrack reader, the
IR shape here).

## Reading

`lattix/formats/lattix_json.py` validates the document with the IR models (`extra="forbid"`,
so a stale or foreign key is an error, not a silent drop) and returns the `Lattice` as
written, including `native` passthrough blocks of other formats and expressions in their
symbolic form.  Every element is EXACT.

## Writing

Every kind is written as itself: the all-kinds test lattice gives 22 EXACT entries and
nothing else.  `write ∘ read` is a fixed point (invariant I-13), which is what makes this
format the reference for the round-trip and property tests: a lattice can be parked here
between two conversions without losing anything.

| IR kind | Emitted | Ledger |
|---|---|---|
| all 22 kinds | the model itself | EXACT |

## Units and conventions

SI + eV, radians, the IR conventions of [../conventions.md](../conventions.md) — there is no
boundary conversion.  Numbers are written with full double precision.

## Known limits

* Not a lattice language: nothing reads it but lattix.
* Field-map *files* are referenced by path and checksum, not embedded.

## Oracle

None needed; the round-trip tests (`tests/property/test_roundtrip_properties.py`,
`tests/roundtrip/`) exercise it on every generated lattice.
