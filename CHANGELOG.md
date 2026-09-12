# Changelog

All notable changes to lattix are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/). Release candidates rehearse a version on TestPyPI
and are listed under the version they rehearse.

## [Unreleased]

### Added
- TraceWin: a trailing comment on an element card travels with the element and is written back;
  a comment that names the card (`; 4.898 HKV MONITOR`, `; D1`) names an unlabelled element and
  keeps its type words as tags, and a zero-length `DRIFT` so named is read as a `Marker`.
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
- Floor frames (`lattix.ir.frames`): the position and full orientation of every element at its
  entrance, centre and exit, the misaligned body frame (offsets and pitches about the centre, a
  skew magnet rolled by its own tilt), MAD-X survey angles kept continuous along the line, a start
  pose as MAD-X `SURVEY` takes it, and Superposition children in place. Pinned against MAD-X's
  survey, xtrack's and Bmad's floor positions (`Reference` and `Actual`) to 1e-9 m and 1e-10 rad.
- `lattix survey`: the floor coordinates and survey angles of every element as a table, CSV or
  JSON; the workbench serves the same at `GET /api/session/<sid>/survey`.
- MAD-X `yrotation`, `xrotation`, `srotation` and `translation` read as `Patch` elements and a
  `Patch` writes back as those cards (several at one position for a combined patch,
  EQUIVALENT `PATCH_AS_CARDS`), with cpymad's survey as the arbiter of the signs; `changeref`
  stays unsupported because MAD-X's own survey ignores it.
- Workbench plugins: an installed package adds tabs to `lattix ui` through the
  `lattix.ui.plugins` entry-point group, served under `/plugins/<name>/` behind the same token and
  kept in step with the page over `postMessage`; `lattix ui --no-plugins` turns them off.
- Two public decks for the survey oracles: `lattix/patches.madx` and `lattix/misaligned.bmad`.

### Changed
- `survey()` keeps its shape and its numbers on every public deck, but its `theta` is now the
  azimuth of the exit direction rather than the sum of bend angles, which only agreed while the
  bend plane was horizontal.
- The MAD-X oracle aligns table rows by element name, so a thick element followed by a
  zero-length frame card no longer reports the card's survey row as its own.
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
