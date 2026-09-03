# PALS examples vendored for the lattix test-suite

- Upstream: <https://github.com/pals-project/pals> (the PALS standard repository)
- Commit: `a2b108342c84e03c80f606d88ff446059b757b9e` (main, 2026-09-01), fetched 2026-09-03
  from `raw.githubusercontent.com/pals-project/pals/main/…`
- License: **CC-BY-4.0** (`Attribution 4.0 International`), full text in
  `../LICENSES/PALS.txt` (upstream `LICENSE`).  Attribution: the PALS project
  (Particle Accelerator Lattice Standard, pals-project/pals); the files are copied
  verbatim, including their own header comments.
- Standard text used while writing `lattix/formats/pals` (same commit, not vendored —
  cited by URL in the module docstrings): `source/fundamentals.md`,
  `source/lattice-element-kinds.md`, `source/beamlines.md`,
  `source/lattice-construction.md`, `source/extensions.md`,
  `source/parameters/{aperture,bend,bodyshift,floor,magneticmultipole,meta,patch,
  reference,referencechange,rf,solenoid,taylor,tracking}.md`.

| File | Upstream path | What it exercises |
|---|---|---|
| `fodo.pals.yaml` | `examples/fodo.pals.yaml` | `PALS:` root, `facility` list, named element references in a `line`, an inline element definition (`drift2`), `inherit:` with an override (`quad2`), `repeat: 3`, `Lattice`/`branches`, `use:` — 3 cells × 5 items = 15 placed elements, 9.0 m |
| `iota.pals.yaml` | `examples/iota.pals.yaml` | FNAL IOTA ring: `extension_labels` + an `ImpactX` extension block, `Kn1` (normalized) quadrupoles needing the reference momentum, bends given by `radius_ref` + `length` with `edge1_int`/`edge2_int`, an inline `BeginningEle` carrying `ReferenceP` and `TwissP`, `periodic: true`, and `repeat: -1` (reversed element order, no direction reversal) |
| `bend_angle_radius.pals.yaml` | `examples/unit_tests/elements/bend_angle_radius.pals.yaml` | a `Bend` given by `angle_ref` + `radius_ref` only — the arc length (1.0 m) is derived |
| `rf_voltage.pals.yaml` | `examples/unit_tests/elements/rf_voltage.pals.yaml` | `RFP.voltage` only; `gradient = voltage / L_active` with `L_active` defaulting to `length` |
| `rf_gradient.pals.yaml` | `examples/unit_tests/elements/rf_gradient.pals.yaml` | `RFP.gradient` only; `voltage = gradient · L_active` |
| `drift_quad_bend.pals.yaml` | `examples/unit_tests/elements/drift_quad_bend.pals.yaml` | one branch with `BeginningEle` + `Drift` + `Quadrupole` (`Bn1`) + `Bend` (`angle_ref`) + `Marker` |

Also relevant and vendored elsewhere: `../impactx/fodo.pals.yaml` (ImpactX
`examples/pals/fodo.pals.yaml`, BSD-3-Clause-LBNL).  Note that that file has **no `PALS:`
root node** — it is a bare one-key `BeamLine` document, the only shape ImpactX's
`KnownElementsList.load_file` and `pals-schema`'s `BeamLine` model accept.  The lattix
PALS reader accepts both shapes; the writer emits the standard shape by default and the
bare-`BeamLine` shape with `flavor="impactx"`.
