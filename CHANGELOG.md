# Changelog

All notable changes to lattix are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/). Release candidates rehearse a version on TestPyPI
and are listed under the version they rehearse.

## [Unreleased]

### Added
- SciBmad (Beamlines.jl) as a format and an engine, with a pure Python reader for the Julia
  lattice subset and a writer whose conventions were measured against BeamTracking.
- The xsuite completion: every xtrack element class, knobs and environments, and a MAD-NG writer.
- LightWin as a second engine with TraceWin semantics.
- Cheetah LatticeJSON, PyORBIT3 linac XML, IMPACT-T, Ocelot lattice modules, DYNAC decks,
  Synergia lattice JSON and OPAL-T decks, each with a reader, a writer and, where the code is
  freely runnable, an engine adapter.
- The Bmad bridge: Astra, GPT, CSRtrack, Merlin++, SLICKTRACK, SAD, SXF and Accelerator Toolkit
  through the converters that ship with Bmad. These need a Bmad installation.
- The cross-format battery (`lattix crossval`): every registered deck written to every format that
  can hold it, read back, and run through the engines on both ends.
- The delta energy mode for constant-momentum targets (MAD-X, MAD8, xtrack, SciBmad), with the
  phase slip a fixed reference velocity implies written into every cavity and undone on reading.
- `lattix ui`, a browser workbench: a beam-line synoptic and floor plan, per-element inspection,
  before-and-after alignment of a translation, and engine validation with the battery's verdict.
- Two public bend decks with pole faces, a negative-angle bend and vertical bends, asserted in
  continuous integration across MAD-X, Bmad, Elegant, xtrack, ImpactX and IMPACT-Z.
- The engine verdict explains itself: every pair names the ledger codes that changed the optics
  and the measured engine limits that cap its tier.
- A version-aware HELIX oracle: it reads the checkout's git ancestry and reports whether the dipole
  fixes it depends on are present, so the battery holds HELIX to the strict tier on a tree that has
  them and to a report-only rule on one that does not.
- A constant-momentum limit: engines that keep one reference momentum are not run on a line that
  gains more than a factor of two, and the verdict says so.
- Packaging for PyPI: the Julia worker ships in the wheel, the metadata carries classifiers,
  keywords and project URLs, and the licence is declared as a PEP 639 expression.

### Changed
- Bmad pitch planes: `x_pitch` is a rotation about y (the IR `y_rot`) and `y_pitch` is `-x_rot`,
  matching Elegant, PALS and MAD8. A Bmad round trip hid the earlier plane swap.
- The SciBmad reader honours a reference carried on a leading `Marker`, the form HELIX's examples
  and Bmad's own converter write, with explicit read options winning field by field.
- The HELIX oracle routes MAD-X, MAD8 and Elegant decks through lattix's reader and TraceWin
  writer, because HELIX's own MAD-X parser does not follow `call, file=`.
- IMPACT-T pole faces on a negative bend follow the mirror of the positive bend; the oracle resumes
  after a missed dump and reports a non-finite tail without a map instead of dropping it.
- Engines are found through environment variables only (`HELIX_ROOT`, `TRACEWIN_EXE`,
  `LATTIX_DYNAC_EXE`, `LATTIX_SYNERGIA_ROOT`); the package no longer carries any machine's paths.
- The sample decks are found through `LATTIX_PUBLIC_DECKS` when the package is installed from a
  wheel; a checkout needs nothing.
- The README groups formats by heritage rather than by what each code can simulate, after the
  ImpactX maintainers pointed out that ImpactX runs rings as well as linacs.

### Fixed
- Twelve golden files embedded the version banner; every golden comparison now neutralises it.
- One HELIX comparison test errored on a machine with a HELIX checkout but no scipy; it skips.

### Notes
- The source distribution contains the package, the licence, the README and the citation file.
  Tests and documentation are in the repository.
- Reading MAD-X decks needs `cpymad`, provided by the `oracles` extra.

## [0.1.0] - 2026-09-03

### Added
- Readers and writers for TraceWin, MAD-X, MAD8 flat, Elegant, Bmad, PALS, ImpactX, IMPACT-Z,
  FLAME and xtrack, through a code-neutral intermediate representation that mirrors PALS.
- The fidelity ledger: every element of every conversion marked exact, equivalent, lossy or
  dropped, with a named code.
- TraceWin field maps integrated into equivalent cavities and hard-edge magnets.
- The HELIX adapter, the engine oracles (MAD-X through cpymad, xtrack, Bmad through Tao, Elegant,
  ImpactX, IMPACT-Z, FLAME, HELIX, TraceWin) and the `lattix validate` comparison of transfer
  maps block by block.
- The command line: `convert`, `inspect`, `oracles`, `fingerprint`, `validate`, `report`.
