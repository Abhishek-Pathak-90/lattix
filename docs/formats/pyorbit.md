# PyORBIT3 (linac XML)

`lattix/formats/pyorbit/` reads and writes the linac lattice XML of
[PyORBIT3](https://github.com/PyORBIT-Collaboration/PyORBIT3) (MIT), the file
`orbit.py_linac.linac_parsers.SNS_LinacLatticeFactory` builds a linac from — the format of the SNS and
ESS models shipped with the code.  The engine runs in its own environment through
`lattix/oracles/pyorbit.py`.  Every convention below was measured on the meson build of PyORBIT3
commit `22b45fa` (2026-05-14) on 2026-09-05 (`docs/oracles.md`).

## Reading

Sequences are the root's children with a `length` attribute (`read(..., sequences=[...])` picks
some); each `accElement` has a `type`, a `length` and a `pos` (its **centre**).  The reader converts
`QUAD` (`field` = lab gradient in T/m, circular `aperture` = full diameter), `BEND` (`theta`,
sector-referenced `ea1/ea2`, rectangular or elliptical `aperture_x/y`; `kls/poles` multipole content
is LOSSY `MULTIPOLE_ORDERS_DROPPED`), `SOLENOID` (`B` = B₀/Bρ in 1/m, turned into a lab field with
the entrance rigidity), `RFGAP` (`E0TL` in GeV, `phase` in degrees, the frequency from the gap's
`<Cavity>`; the `TTFs` polynomials are kept as native passthrough, EQUIVALENT
`PYORBIT_TTF_POLYNOMIALS`), `DCH`/`DCV` (`B·effLength` kicks; the DCH/DCV pair lattix writes for one
kicker becomes one Kicker again) and `MARKER`.  Any other type is DROPPED
`UNSUPPORTED_PYORBIT_ELEMENT` with a drift or marker keeping its place.  Drifts are implicit and are
created from the gaps between elements (`<seq>_drift_<n>`); a thin node inside a thick magnet splits
it, because PyORBIT applies the node between the magnet's parts (EQUIVALENT
`PYORBIT_THIN_NODE_SPLITS_MAGNET`, the SNS MEBT correctors sit inside the quads); overlapping
elements are pushed apart (LOSSY `PYORBIT_OVERLAP`).  The XML carries no beam: the reference comes
from the `lattix: reference` comment lattix writes (EQUIVALENT `REFERENCE_FROM_TAG`) or from
`read(..., species=, kinetic_energy_eV=)`, and the reference's RF frequency falls back to
`bpmFrequency`.  A gap whose cavity the sequence does not define is LOSSY `RF_FREQUENCY_UNKNOWN`.  The
`<x>_in`/`<x>_out` markers lattix writes around a thick cavity or kicker fold back into one thick
element (EQUIVALENT `PYORBIT_END_MARKERS_FOLDED`).

## Writing

| IR kind | PyORBIT | Ledger |
|---|---|---|
| Drift | nothing (the factory fills the gaps between elements) | EXACT |
| Quadrupole | `QUAD field = Bn1` [T/m], circular `aperture` | EXACT; a skew term LOSSY `SKEW_COMPONENT_DROPPED`, a tilt LOSSY `PYORBIT_QUAD_TILT_DROPPED`, other aperture shapes LOSSY `APERTURE_SHAPE` |
| Sextupole, Octupole | nothing (the length becomes drift) | LOSSY `PYORBIT_NO_MULTIPOLE` |
| Multipole (thin) | `DCH`/`DCV` for the dipole terms | LOSSY `PYORBIT_MULTIPOLE_AS_CORRECTOR`, higher orders LOSSY `MULTIPOLE_ORDERS_DROPPED` |
| Bend | `BEND theta ea1 ea2` (sector-referenced), `aperture_x/y`; a zero-angle bend is a drift | EXACT; fint with a gap LOSSY `BEND_FRINGE_DROPPED`, a tilt LOSSY `PYORBIT_BEND_TILT_DROPPED`, a gradient LOSSY `MULTIPOLE_ORDERS_DROPPED`, zero angle EQUIVALENT `ZERO_ANGLE_BEND_AS_DRIFT` |
| Solenoid | `SOLENOID B = Bsol/Bρ` [1/m] | EXACT |
| RFCavity | `RFGAP E0TL = V·10⁻⁹, phase` (+180° for a negative species), one `<Cavity>` per gap; a thick cavity is a gap at its centre between `<name>_in`/`<name>_out` markers; without a frequency (a 0 Hz gap gives NaNs in PyORBIT) a `MARKER` | EXACT / EQUIVALENT `THICK_CAVITY_AS_GAP`; no frequency: LOSSY `PYORBIT_GAP_NEEDS_FREQUENCY` (gain lost) or EQUIVALENT `PYORBIT_IDLE_GAP_AS_MARKER` (no voltage either) |
| FieldMap | the `lattix.ir.fieldmap` replacement ladder | EQUIVALENT / LOSSY `FM_*` |
| NCells, RFQCell | nothing (drift) | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `DCH` + `DCV` (`B·effLength = ∓kick·Bρ`); a thick kicker kicks at its centre between end markers | EXACT / EQUIVALENT `THICK_KICKER_SPLIT`; LOSSY `EKICK_AS_MAGNETIC`, `KICKER_TILT_DROPPED` |
| Collimator, Marker, Instrument, Foil | `MARKER` | LOSSY `COLLIMATOR_TO_MARKER`, EXACT, EQUIVALENT `INSTRUMENT_AS_MARKER`, LOSSY `FOIL_TO_MARKER` |
| Taylor, Patch, ReferenceChange, Directive | nothing | DROPPED `TAYLOR_DROPPED`, `PATCH_DROPPED`, `REFCHANGE_DROPPED`, `FOREIGN_DIRECTIVE` |
| Freq | nothing (the frequency is per cavity) | EXACT |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Misalignments are LOSSY `MISALIGN_DROPPED`.  The document is `<lattix>` with one `<seq>` per
lattice (`bpmFrequency`, `length`, the `<Cavities>` list, then the elements with `pos` = centre) and
lattix's reference tag in a comment.  Lengths are written on a picometre grid and positions on a
half-picometre grid, cumulated from the written lengths; the factory's overlap check
(`pos₁ − L₁/2 − (pos₀ + L₀/2) < 0`) has no tolerance, so a node PyORBIT's own arithmetic would
see as overlapping its neighbour is pushed 1 pm past it (far below the factory's 10 µm
`zeroDistance`, so it never becomes a drift there), and the reader treats gaps below 1 nm as
touching, which makes write → read → write a fixed point — also for the vendored SNS and ESS
decks and for every battery deck.  A thin gap carries PyORBIT's own transverse
gap focusing (`BaseRfGap`), not lattix's thin-gap lens.

## Units and conventions

* m, rad, T/m (`QUAD field`), 1/m (`SOLENOID B`), T·m (`DCH/DCV B·effLength`), GeV (`E0TL`), degrees
  (`phase`), Hz (`<Cavity frequency>`, `bpmFrequency`).  Coordinates are
  `(x [m], x′, y [m], y′, z [m] ahead-positive, dE [GeV])`: a 1 m drift at 2.1 MeV gives
  `R56 = +237.30 = L/γ² · 10⁹/(β²γ mc²)` (`Basis.PYORBIT`).
* `QUAD field` is the lab gradient with the charge in `bunch.B_Rho()` — a positive field focuses `x`
  for a proton and defocuses it for H⁻ — so the writer emits `Bn1` unchanged.  `SOLENOID B` is
  `B₀/Bρ` (TEAPOT's `soln`), the tracker applies the charge.  `DCH`: `x′ −= B·effLength/Bρ_signed`,
  `DCV`: `y′ += B·effLength/Bρ_signed`.
* `RFGAP`: `ΔE = q·E0TL·cos(phase)`, so H⁻ needs `phase + 180°` for the IR's species-independent
  `V·cos φ`; the slope bunches at φs = −30° like TraceWin's `GAP`.  **One `<Cavity>` per gap**:
  PyORBIT derives the later gaps of a shared cavity from the time of flight through the first, which
  cost 2.6 % of the MEBT energy when lattix shared cavities by frequency.
* `BEND theta` has MAD-X's sign (`R16 > 0` for `theta > 0`), `ea1/ea2` are sector-referenced; single
  sector bends of 1°–45° agree with MAD-X to 1e-11 on the transverse block.
* `aperture` is a full diameter for `aprt_type` 1 (circle), `aperture_x/y` full sizes for 3
  (rectangle) and elliptical otherwise.

## Known limits

* No sextupole, octupole, thin multipole, fringe-field integral, bend tilt, misalignment, matrix,
  patch, foil or collimator in the linac XML.
* Correctors and gaps are thin: a thick kicker or cavity acts at its centre (Equivalent tier).
* The transit-time factor is folded into `E0TL`; PyORBIT's velocity-dependent `TTFs` polynomials
  survive only as passthrough (constant `T = 1` polynomials are written otherwise).
* Ring lattices (PyORBIT's MAD8 `.lat` path) are not this format; lattix's MAD8 reader covers them.

## Oracle

`lattix/oracles/pyorbit.py` + `pyorbit_worker.py` (conda env `pyorbit`: python 3.10, meson build of
PyORBIT3 with `USE_MPI=none`; `libfftw3` is preloaded on macOS; marker `oracle_pyorbit`):
`LinacTrMatricesController` matrices at every node entrance (cumulated from the first node; the
per-node map is `R[k+1]·R[k]⁻¹` with an extra node at the exit), fitted from a 13-particle symmetric
probe bunch tracked at two amplitudes and Richardson-extrapolated (the tracker's third-order terms
otherwise leave 2e-7 relative on a 5 m chicane).  Measured against HELIX on the transverse block:
`fodo_cell.dat` 1.2e-11, `solenoid_channel.dat` 9e-13, `bend_line.dat` 2.9e-11, `csr_chicane.dat`
1.1e-8; `mebt_line.dat` (H⁻, thin gaps with PyORBIT's own gap focusing) 4.3e-3 with the energy
exact.  PyORBIT on the SNS and ESS MEBT originals and on lattix's rewrites of them: 7.7e-10 /
6.6e-11 with identical final energies.
