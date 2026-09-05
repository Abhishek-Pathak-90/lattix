# Bmad bridge: Astra, GPT, CSRtrack, Merlin++, SLICKTRACK, SAD, SXF, Accelerator Toolkit

`lattix/formats/bmad_bridge/` reaches the formats Bmad itself converts to and from.  A write
goes IR → Bmad (lattix's own writer, `docs/formats/bmad.md`) → the converter that ships with
Bmad; a read goes the converter → Bmad → IR (lattix's Bmad reader).  Every element keeps the
Bmad writer's or reader's ledger and carries `EQUIVALENT VIA_BMAD`; whatever the converter
printed is recorded as `BMAD_CONVERTER_NOTE`, or `BMAD_CONVERTER_LOSS` when it announced
something it could not translate.  There is no engine on the far side: the tests are
structural, and the intermediate Bmad file (`bmad_copy=…`) is checked to re-read to the IR.
Measured with Bmad 20260828 (conda-forge, env `bmad`), see
[oracles.md](../oracles.md#phase-56-measurements-bmad-bridge-bmad-20260828-2026-09-05).

| format | suffix | direction | Bmad tool | note |
|---|---|---|---|---|
| `astra` | `.astra` | write | `bmad_to_astra` (namelist written by lattix, `fieldmap_dimension` option, no particle files) | dipoles as corner coordinates, quadrupoles, cavities need field maps |
| `gpt` | `.gpt` | write | `bmad_to_gpt` (namelist, `fieldmap_dimension`) | **bends are not translated by Bmad** (`TRANSLATION TO GPT FOR BEND NOT YET IMPLEMENTED`, a `BMAD_CONVERTER_LOSS`) |
| `csrtrack` | `.csrtrk.in`, `.csrtrack` | write | `bmad_to_csrtrack` (namelist + a CSRtrack template, `template=` option) | markers `<name>_m1/_m2` around each element; the beam part is a one-particle placeholder |
| `merlin` | `.tfs`, `.merlin.tfs` | write | `bmad_to_merlin` | a TFS table: name, type, s, length, tilt, angle/e1/e2, ks, k0l…k3l, RF |
| `slicktrack` | `.slick` | write | `bmad_to_slicktrack` (`no_split=True` keeps bends and quads whole) | |
| `sad` | `.sad` | write and read | Tao `write sad`; `sad_to_bmad.py` (+ `sad_to_bmad.params`) | SAD carries `MOMENTUM` but no species: pass `species=` when reading |
| `sxf` | `.sxf` | read | `sxf_to_bmad.py` | no beam energy in SXF: pass `species=` and `kinetic_energy_eV=` |
| `at` | `.at` | read | `accelerator_toolkit_to_bmad.py` | Bmad's own note: "only good for simple lattices" |

Not bridged: OPAL-T and XSIF — the Tao of Bmad 20260828 accepts no `write opal` or `write xsif`
(`UNKNOWN "WHAT"`, `BAD OUT_TYPE: XSIF`); PTC flat files — the shipped `ptc_flat_file_to_bmad`
reads a flat file and writes nothing.  MAD-X, MAD8, Elegant, PALS and SciBmad have lattix's own
writers (`bmad_to_mad_sad_elegant` and Tao's `write madx|mad8|elegant|pals|scibmad` are not
needed).

## Reading

```
$ LATTIX_BMAD_UTIL_DIR=/path/to/bmad-ecosystem/util_programs \
  lattix convert ring.sad ring.dat --from sad --read-option species=proton
```

`sad_to_bmad.py`, `sxf_to_bmad.py` and `accelerator_toolkit_to_bmad.py` are scripts of a Bmad
*source* tree (`util_programs/`), not part of the conda package: `LATTIX_BMAD_UTIL_DIR` (or
`ACC_ROOT_DIR`, `BMAD_DIST`, `DIST_BASE_DIR`) must point at one, and the scripts run with the
`bmad` environment's python (`LATTIX_BMAD_PYTHON`, else env `bmad`).  The converter's Bmad file is
read by lattix's Bmad reader, whose read options (`species=…`, energies) pass through; the
lattice's `meta["bmad_bridge"]` records the tool and the Bmad file (`keep_workdir=…` keeps it).
A SAD lattice is read without `sad_to_bmad_postprocess` (SAD's `fshift` patches are not
inserted; Bmad's `DOC` explains when they matter).

## Writing

```
$ lattix convert mebt.dat mebt.astra --to astra
$ lattix convert fodo.madx fodo.gpt --to gpt --write-option fieldmap_dimension=2
```

The binaries are found next to the `bmad` environment's python (`LATTIX_BMAD_BIN` overrides,
then `PATH`).  Options: `bmad_copy=` keeps the intermediate Bmad file, `keep_workdir=` the
converter's working directory (with its `convert.log`), `fieldmap_dimension=` (Astra, GPT: 3),
`template=` (CSRtrack: the run parameters around `INSERT_LATTICE_AND_BUNCH_PARAMS_HERE`),
`no_split=` (SLICKTRACK), and any option of lattix's Bmad writer (`line_mode=…`).  Strict mode
raises on the Bmad writer's losses and on a `BMAD_CONVERTER_LOSS`.

## Units and conventions

Bmad's: the IR is written to Bmad in SI and eV (`docs/formats/bmad.md`), and each converter
applies its own target's units (Astra fields in T and MV/m with `D1…D4` corner coordinates,
SAD `K1` integrated, Merlin's TFS integrated `k0l…k3l`).  lattix does not re-derive them.

## Known limits

* No engine runs the produced file, so nothing beyond structure is checked here; the converter's
  notes are the fidelity statement (GPT: bends untranslated; Astra and GPT: cavities without field
  maps are placeholders; CSRtrack: Bmad also computes a bunch, which needs Twiss the lattice does
  not carry — its `NON-POSITIVE BETA` error is recorded and the lattice section is complete).
* SAD and SXF carry no species (SAD) or no energy at all (SXF): give them on read.
* The readers depend on scripts outside the conda package; `available("sad")` explains what is
  missing.  Accelerator Toolkit input has no generator here and is wired but unexercised.

## Oracle

None.  The Bmad file written on the way is the checkpoint: `bmad_copy=` re-read by
`lattix.formats.bmad` reproduces the IR (the tests compare the two profiles), and the Bmad
oracle (Tao) can run that file.
