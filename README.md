<p align="center">
  <img src="docs/assets/readme-hero.png" alt="lattix: twenty-eight accelerator lattice formats connected through one intermediate representation" width="880">
</p>

<p align="center">
  <img src="docs/assets/readme-badges.png" alt="BSD 3-Clause, Python 3.11 and above, 28 formats, 17 engines, 588 ledger codes, version 0.1.0" width="760">
</p>

# lattix

Every accelerator code speaks its own dialect. A lattice written for TraceWin cannot be read by
MAD-X, and one written for Elegant means nothing to IMPACT-Z, so the same machine gets rebuilt by
hand for every code and quietly loses detail each time.

lattix reads a deck into one code-neutral representation and writes it back out in another code's
language. It tells you exactly what changed on the way, and it checks its own work against the real
simulation codes rather than against itself.

## See it work

Translate a TraceWin achromatic bend line into an Elegant lattice:

```console
$ lattix convert bend_line.dat bend_line.lte --read-option species=h- --read-option kinetic_energy_eV=2.1e6

fidelity tracewin→elegant: EXACT=41 EQUIVALENT=3 LOSSY=0 DROPPED=1
  EQUIVALENT INSTRUMENT_AS_MARKER  ×3   elegant has no diagnostic type for family 'DIAG_SIZE'; the
                                        zero-length diagnostic is written as MARK (same optics, the
                                        family label survives only in provenance)
  DROPPED    FOREIGN_DIRECTIVE     ×1   format-specific directive written as a comment only
```

That summary is the point of the tool. Forty-one elements came through untouched, three diagnostics
became markers because Elegant has no equivalent type, and one TraceWin directive had nowhere to go.
Nothing was lost silently. The written file carries the reference particle so it can be read back:

```
! lattix 0.1.0 from IR
! lattix: reference species="h-" mass_eV=939294086.06 charge=-1 kinetic_energy_eV=2100000 rf_frequency_Hz=352210000

DRIFT_0001: DRIF, L=0.1
QUAD_0001: KQUAD, L=0.06, K1=-14.3111295671334
```

Add `--strict` and the conversion stops at the first element it cannot carry across, which is what
you want in a script. Add `--report out.json` and the whole ledger lands in a file.

## Twenty-eight formats

<p align="center">
  <img src="docs/assets/readme-formats.png" alt="The formats grouped into linac codes, ring codes, the modern Python stack, and the eight reached through Bmad's converters" width="900">
</p>

Twenty of them work in both directions. Six can only be written and two can only be read, which is a
limit of the formats themselves rather than of the translator.

| group | formats |
|---|---|
| linac | TraceWin, DYNAC, IMPACT-Z, IMPACT-T, ImpactX, FLAME, OPAL-T |
| ring and general | MAD-X, MAD8, Elegant, Bmad, PALS, MAD-NG |
| modern stack | Xtrack, SciBmad, Cheetah, PyORBIT3, Ocelot, Synergia, lattix JSON |
| through Bmad's converters | SAD, Astra, GPT, CSRtrack, Merlin++, SLICKTRACK, SXF, Accelerator Toolkit |

The last group is worth understanding before you rely on it. lattix does not write those formats
itself. It writes a Bmad deck and then calls the converters that ship with Bmad, so those eight need
a Bmad installation present. The other twenty need nothing beyond lattix.

## The ledger

<p align="center">
  <img src="docs/assets/readme-ledger.png" alt="A fidelity ledger: each element marked exact, equivalent, lossy or dropped, with a named code" width="900">
</p>

Every element in every conversion is marked exact, equivalent, lossy or dropped, and each mark carries a
named code saying precisely what happened. There are 588 of them, all catalogued in
[docs/fidelity.md](docs/fidelity.md) with the source line that raises each one.

The distinction matters in practice. `RBEN_CHORD_TO_ARC` is equivalent, meaning the optics survive and
only the bookkeeping differs. `IMPACTZ_NO_REF_TILT` is lossy, meaning a vertical bend was written
horizontal because that code has no other option, and your results will differ. You can see which one
you got before you run anything.

## Checked against the real codes

<p align="center">
  <img src="docs/assets/readme-validation.png" alt="Thirteen engines compared against MAD-X on one bend line, agreement plotted on a logarithmic scale" width="900">
</p>

lattix does not take its own word for it. The same lattice is written out for as many of the
seventeen supported engines as can accept it, each engine is run, and their element-by-element
transfer maps are compared. The plot above is one bend line with pole faces and a negative bend,
measured against MAD-X.

Most engines land between 1e-11 and 1e-9, which is arithmetic noise. Two sit higher for reasons that
are understood and documented: DYNAC prints six digits in its dump files, and IMPACT-T has no
pole-face focusing at all. Those are labelled rather than hidden.

A harder case, a proton synchrotron ring with thirty-two dipoles, agrees with HELIX to two parts in
ten to the thirteenth on the transverse block.

## What it will not do

Being clear about this is more useful than a longer feature list.

- **IMPACT-T bends are reported, not asserted.** Its dipole turns the whole bunch by the reference
  angle with no pole-face focusing, so bend lines are compared and shown, never held to a tolerance.
- **HELIX dipoles have no path-length row.** The transverse block and the dispersion are compared;
  the longitudinal row is not, because that code does not model it.
- **Codes that hold one reference momentum cannot follow strong acceleration.** MAD-X, Xtrack and
  SciBmad are not run on a line where the beam gains more than a factor of two, rather than reporting
  a comparison nobody should trust.
- **Ion species cannot yet be handed to most engines**, so those decks translate but do not validate.

Full detail, with the measurement behind each one, is in [docs/oracles.md](docs/oracles.md).

## Install

```bash
pip install -e .                 # numpy, pydantic, lark and pyyaml, nothing else
pip install -e ".[oracles]"      # adds cpymad, xtrack and friends for validation
```

Requires Python 3.11 or newer. The optional engine extras stay optional on purpose: several
simulation codes are GPL, and lattix keeps them at arm's length by running them as separate
processes, which is what lets the translator itself stay BSD licensed.

Two conda environments cover the engines that are not pip installable. `environment-ci.yml` builds
the env used by continuous integration, with Elegant, ImpactX, IMPACT-Z and cpymad.
`environment-bmad.yml` builds the Bmad and Tao environment used out of process by the Bmad adapter
and by the eight bridged formats.

## Other things it does

```bash
lattix ui --root tests/data/public     # a browser workbench: translate, then compare the beam lines
lattix inspect mebt.dat --elements     # what the reader made of a deck
lattix oracles                         # which engines this machine can actually run
lattix fingerprint                     # measure each engine's longitudinal conventions
lattix validate --deck madx=fodo.madx --deck bmad=fodo.bmad --oracles madx,xtrack,bmad --ke 800e6
```

## Tests

```bash
pytest -m "not corpus and not slow and not crossval"     # engine free, about ninety seconds
pytest -m oracle_bmad                                    # one engine
pytest tests/crossval -m crossval                        # the full cross-format battery
```

The battery is the acceptance gate. It writes every registered deck to every format that can hold
it, reads it back, and runs both ends through their engines, so a regression in one writer shows up
as a failing pair rather than as a surprise months later.

## Documentation

- [docs/tutorial.md](docs/tutorial.md), install, convert, read a fidelity summary, use the Python API
- [docs/conventions.md](docs/conventions.md), units, rigidity, the energy walk, RF phase per format, bends
- [docs/fidelity.md](docs/fidelity.md), the ledger and the catalogue of every code
- [docs/crossval.md](docs/crossval.md), the cross-format battery
- [docs/oracles.md](docs/oracles.md), every engine, adapter and measured convention
- [docs/formats/](docs/formats/), one page per format
- [CITATION.cff](CITATION.cff), how to cite this work

## Licence

BSD 3-Clause. Copyright 2026 Abhishek Pathak.
