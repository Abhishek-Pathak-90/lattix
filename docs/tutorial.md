# Tutorial

Ten minutes from install to a validated translation.  Every command below was run on the
public sample decks in `tests/data/public/helix/` and the output is shown as printed.

## 1. Install

```
$ pip install lattix                 # or: pip install -e .  in a clone
$ pip install "lattix[oracles]"      # adds cpymad, xtrack and friends
$ lattix --version
```

The package needs only numpy, pydantic, lark and pyyaml.  Readers that drive an engine need it
installed: the MAD-X reader needs `cpymad` (the `oracles` extra provides it), the xtrack adapter
needs `xtrack`, PALS validation uses `pals-schema`, and the eight formats reached through Bmad's
converters need a Bmad installation.  The oracle adapters find their engines on
`PATH` or through `TRACEWIN_EXE`, `ELEGANT_EXE`, `HELIX_ROOT`, `LATTIX_BMAD_ENV` and friends
(see [oracles.md](oracles.md)).  `environment-ci.yml` builds a conda environment with
every free engine.

## 2. Look at a deck

```
$ lattix inspect fodo_cell.dat
fidelity tracewin→?: EXACT=24 EQUIVALENT=0 LOSSY=0 DROPPED=0
fodo_cell.dat: 24 placed elements, 24 definitions, 1 lines, L = 1.600000 m
reference: proton 2.1 MeV -> 2.1 MeV; RF clock 352210000.0
  Drift            12
  Quadrupole       8
  Directive        2
  Freq             1
  Instrument       1
```

`inspect` reads the deck into the intermediate representation, walks the reference
particle through it and prints the reader's ledger: here every card was read exactly.
Add `--elements` for one line per placed element with its exit position.

`survey` gives the floor coordinates: where every element sits in the global frame, with
MAD-X's survey angles (the line starts at the origin along +Z, a positive horizontal bend turns
toward −X).

```
$ lattix survey bend_line.dat
# lattix 0.2.0 survey of bend_line.dat (tracewin): 21 rows
# frames: exit; misalignments applied to the body frame; superposition children not expanded
# start pose (MAD-X SURVEY x0 y0 z0 theta0 phi0 psi0; m, rad): 0 0 0 0 0 0
# units: m and rad; theta, phi, psi are MAD-X survey angles, theta continuous along the line
 i               name        kind  parent        s_in       s_out           L             X  Y           Z         theta  phi  psi        angle  tilt_ref
 3         DRIFT_0001       Drift                   0         0.1         0.1             0  0         0.1             0    0    0            0         0
 4          QUAD_0001  Quadrupole                 0.1        0.16        0.06             0  0        0.16             0    0    0            0         0
…
```

`--at all` tabulates the entrance, centre, exit and misaligned-body frames side by side, `--csv`
and `--json` write the table for survey and alignment work, and `--x0 … --psi0` set the start
pose the way MAD-X `SURVEY` does.  Misaligned elements get a body frame that differs from the
reference orbit's; `--no-shift` ignores the misalignments.

## 3. Convert

```
$ lattix convert fodo_cell.dat --to madx fodo_cell.madx
fidelity tracewin→madx: EXACT=35 EQUIVALENT=0 LOSSY=12 DROPPED=1
  LOSSY      APERTURE_DROPPED             ×12   MAD-X 'drift' has no apertype/aperture attribute
  DROPPED    FOREIGN_DIRECTIVE            ×1    format-specific directive written as a comment only
```

The target format comes from the suffix when `--to` is omitted (`.madx`, `.lte`, `.bmad`,
`.dat`, `.pals.yaml`, …).  The summary lists every non-exact code once with its
multiplicity.  Here the twelve TraceWin drifts carried an aperture radius that a MAD-X
`drift` cannot hold, and the deck's `PARTRAN_STEP` card has no MAD-X meaning; both are
written as comments in the output:

```
! lattix 0.1.0 from IR

beam, particle=proton, mass=0.93827208816, charge=1, energy=0.94037208816;

! --- elements
quad_0001: quadrupole, l=0.05, k1=23.8648547049148, apertype=ellipse, aperture={0.02,0.02};   ! lattix: name="QUAD_0001" type="QUAD"
...
fodo_cell: sequence, l=1.6, refer=centre;
  ! lattix directive: TITLE FODO Quadrupole Channel
  ! lattix directive: PARTRAN_STEP 100 50
  quad_0001, at=0.075;
  ...
endsequence;

use, sequence=fodo_cell;
```

The `! lattix: name="…" type="…"` tag keeps the original name and card type; a reader that
sees it restores them, so a round trip through MAD-X gives the TraceWin deck back.

Other targets from the same source:

```
$ lattix convert fodo_cell.dat --to elegant fodo_cell.lte
fidelity tracewin→elegant: EXACT=47 EQUIVALENT=1 LOSSY=0 DROPPED=1
  DROPPED    FOREIGN_DIRECTIVE            ×1    format-specific directive written as a comment only
  EQUIVALENT INSTRUMENT_AS_MARKER         ×1    elegant has no diagnostic type for family 'DIAG_PHASE'; the zero-length diagnostic is written as MARK (same optics, the family label survives only in provenance)

$ lattix convert fodo_cell.dat --to bmad fodo_cell.bmad
fidelity tracewin→bmad: EXACT=47 EQUIVALENT=0 LOSSY=0 DROPPED=1
  DROPPED    FOREIGN_DIRECTIVE            ×1    format-specific directive written as a comment only
```

Elegant and Bmad keep the drift apertures, so those two are exact apart from the
TraceWin-only tracking card.

## 4. Strict mode and reports

`--strict` turns the first LOSSY or DROPPED entry into a failure (exit status 1) and
leaves no new file behind:

```
$ lattix convert fodo_cell.dat --to madx --strict fodo_cell.madx
lattix: error: DROPPED:FOREIGN_DIRECTIVE on Directive 'PARTRAN_STEP_0001': format-specific directive written as a comment only
```

A deck that MAD-X can hold completely passes:

```
$ lattix convert fodo.madx --to tracewin --strict fodo.dat
fidelity madx→tracewin: EXACT=18 EQUIVALENT=0 LOSSY=0 DROPPED=0
```

`--report ledger.json` saves the full ledger next to the output; `lattix report` prints the
ledger for a translation without keeping the result:

```
$ lattix report fodo.madx --to tracewin
fidelity madx→tracewin: EXACT=18 EQUIVALENT=0 LOSSY=0 DROPPED=0
```

What each code means is in [fidelity.md](fidelity.md).

## 5. Options

Readers and writers take `KEY=VALUE` options:

```
$ lattix convert linac.dat --to elegant linac.lte --read-option species=h-
$ lattix convert linac.dat --to madx linac.madx --write-option energy_mode=constant
```

`species` matters for formats whose deck does not name the particle (TraceWin, Elegant,
MAD8 flat files): the RF phase and the sign of every normalized strength depend on it.
`energy_mode` chooses how an accelerating lattice is normalized for a constant-p0 target
(MAD-X, MAD8, xtrack): `local` (default) uses each element's own entrance rigidity,
`constant` the lattice start.  Both are recorded in the ledger.

## 6. Python API

```python
import lattix

lat, report_in = lattix.read("fodo_cell.dat")           # format from the suffix
print(report_in.summary())

report = lattix.write(lat, "fodo_cell.lte", strict=False)
print(report.counts)                                     # {'EXACT': 47, 'EQUIVALENT': 1, 'DROPPED': 1}
report.raise_if(strict=True)                             # TranslationError on the first problem

lattix.translate("fodo_cell.dat", "fodo_cell.bmad", strict=True)
```

`lat` is a `lattix.ir.lattice.Lattice`; `lat.flatten()` gives the placed elements with
their positions, and `lattix.ir.walk.propagate(lat)` adds the reference energy at both ends
of each element.  `lattix.FORMATS` is the registry of formats.

## 7. Validate with the engines

```
$ lattix oracles                                         # which adapters can run here
$ lattix fingerprint --oracles madx,bmad                 # each engine's longitudinal conventions
$ lattix validate --deck madx=fodo.madx --deck bmad=fodo.bmad --deck elegant=fodo.lte \
      --oracles madx,bmad,elegant --tol 1e-8 --json fodo.validate.json
$ lattix validate --deck tracewin=btl.dat --deck madx=btl.madx --oracles helix,madx \
      --species h- --ke 800e6 --tier lossy      # the verdict: blocks the pair is held to, caveats
```

`validate` runs the same lattice through several engines, transforms every map into the
common basis and compares the cumulative maps block by block at the boundaries the engines
agree on; `--tol` makes it fail when the largest normalized transverse difference exceeds
the tolerance.  How the bases are aligned and which numbers were measured is in
[conventions.md](conventions.md) and [oracles.md](oracles.md).

## 8. The browser UI

```
$ lattix ui --root tests/data/public
```

opens a page on `127.0.0.1` that reads a deck (a sample, a path under the root, an uploaded folder with
its field maps), draws the beam line, translates it to any format with the writer's options and draws
the written deck read back, aligned element by element with the source: click an element for its
parameters in SI and native units, the reference energy at its ends, the ledger entries and the
statement in the written deck; the *Validation* tab runs the engines on both decks.  See `docs/ui.md`.
