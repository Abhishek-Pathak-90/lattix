# SciBmad (Beamlines.jl)

`lattix/formats/scibmad/` reads and writes the Julia lattice files of the SciBmad ecosystem
(Beamlines.jl for the lattice, BeamTracking.jl for the tracking); `lattix/oracles/scibmad.py` runs
them through Julia as an engine.  Every convention was measured on SciBmad 0.5.2 (Julia 1.10) on
2026-09-05; the numbers are in `docs/oracles.md`.

## Reading

The reader understands the subset the writer produces (below) plus Bmad's `bmad_to_scibmad` output (its
`function map_x(v, q) … end` maps, `PhaseReference.*` and `SaganCavity(num_cells = …)` spellings;
second-order map terms are dropped with `TAYLOR_ORDER_TRUNCATED`).  Unknown element kinds become a
drift or marker of the same length (`UNSUPPORTED_SCIBMAD_KIND`); `Kn0 ≠ g_ref` is kept as a native
`k0` (`BEND_K0_NE_G`) unless the energy-mode tag explains it.  Reference energy comes from
`pc_ref`, `E_ref` or `p_over_q_ref`, or from the `# lattix: reference` tag.

## Writing

The file the writer produces:

```julia
# lattix 0.1.0 from madx (SciBmad / Beamlines.jl)
# lattix: energy_mode=delta
# lattix: reference species="proton" mass_eV=938272088.16 charge=1 kinetic_energy_eV=799999911.84
using Beamlines

@elements begin
  qf = Quadrupole(L = 0.3, Kn1 = 0.6)
  drift_0 = Drift(L = 0.7)
  b1 = SBend(L = 1, g_ref = 0.1, Kn0 = 0.1, e1 = 0.05, e2 = 0.05)
  m1 = Marker()
end

fodo = Beamline([qf, drift_0, b1, drift_0, m1];
    pc_ref = 1463295949.06973, species_ref = Species("proton"))
```

`using Beamlines` is what Bmad's own `bmad_to_scibmad` writes; an environment that only has the
umbrella package needs `using SciBmad` instead (`Writer(...).write(lat, path, using="SciBmad")`),
and the oracle strips the line before loading the file.

### Element table

| IR kind | SciBmad | Ledger |
|---|---|---|
| Drift | `Drift(L)` | EXACT |
| Quadrupole, Sextupole, Octupole | `Quadrupole(L, Kn1, Ks1, tilt1, …)` etc.; any extra order as `Kn<n>` | EXACT (every order is kept) |
| Multipole (thin) | `Multipole(Kn<n>L, Ks<n>L)` | EXACT |
| Bend | `SBend(L, g_ref, Kn0, e1, e2, edge1_int, edge2_int, tilt_ref)`; `hgap`, `rect`, `fringe_k2` in the tag | EXACT; `g_ref` alone is a curved frame, so the design field is `Kn0 = g·r` |
| Solenoid | `Solenoid(L, Ksol)` | EXACT |
| RFCavity | `RFCavity(L, voltage = −V, phi0 = φ [rad], rf_frequency, traveling_wave)`; `n_cell`, `L_active_m`, `dE_ref_eV` in the tag | EXACT in the file; `CONST_P0` + energy-mode rows when it accelerates; thin gaps get the `LineElement` RF lens |
| FieldMap | the degradation ladder of `lattix.ir.fieldmap` (cavity / hard-edge solenoid / quadrupole / drift) | LOSSY `FM_TO_*` |
| NCells, RFQCell | `Drift(L)` | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `Kicker(Kn0L = −hkick·r, Ks0L = +vkick·r)` | EXACT (`EKICK_AS_MAGNETIC` for electric) |
| Collimator | `Drift`/`Marker` + `x1_limit … y2_limit`, `aperture_shape` (tag `kind="Collimator"`) | EXACT |
| Marker, Instrument | `Marker()` / `Drift(L)` (tag `kind="Instrument" family=…`) | EXACT |
| Foil | `Marker()` + tag | LOSSY `FOIL_TO_MARKER` |
| Taylor | `LineElement(L, transport_map = f)` with a generated `function f(v, q, p=nothing)` | EXACT (linear) |
| Patch | `Patch(dx, dy, dz, dx_rot, dy_rot, dz_rot, dt)` | EXACT (`e_tot_offset` in the tag) |
| ReferenceChange | `Marker()` + tag (`REFCHANGE_AS_TAG`, restored on read) | EQUIVALENT |
| Freq | `Marker()` + tag | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Apertures (`x1_limit … y2_limit`, `aperture_shape = ApertureShape.Rectangular|Elliptical`) and
alignments (`x_offset, y_offset, z_offset, x_rot, y_rot, tilt`) are written on any element with the
IR's own names.

## Units and conventions

Everything below was measured, not assumed:

* **One reference momentum per beamline.**  A `Patch(dE_ref = …)` anywhere but the first element is
  refused and beamlines cannot nest, so SciBmad is a constant-p0 engine like MAD-X and xtrack: the
  writer uses `energy_mode="delta"` by default (`conventions.md` §3), which normalizes every
  strength with the momentum the engine's own particle has, rescales kicks, maps and the bend
  field `Kn0` by the same ratio `r`, and moves each downstream cavity's `phi0` by the slip of the
  constant-velocity clock.  The `# lattix: energy_mode=…` tag lets the reader undo all of it.
* **RF gain is `−V·cos(phi0)`** with `phi0` in radians and the default `zero_phase =
  PhaseRef.Accelerating`, for protons and electrons alike.  The writer negates the voltage
  (`GAIN_SIGN = −1`) so `phi0` keeps the IR's synchronous phase (0 = crest, gain `+V cos φ`); the
  fingerprint test fails loudly if a later SciBmad flips this.  `AboveTransition`/`BelowTransition`
  put `phi0 = 0` at the zero crossing (the reader shifts by +90°, `ZERO_PHASE_SHIFTED`).
* **`g_ref` is the coordinate curvature only**: without `Kn0` an `SBend` is a curved drift
  (`R16 = 0`).  `e1`/`e2` are sector-referenced like MAD's; `edge1_int`/`edge2_int` are `fint·hgap`
  and are stored but not tracked by BeamTracking 0.5 (the oracle zeroes them and says so).
* **Zero-length cavities cannot be tracked** in 0.5 (`thin_pure_rf` is missing); the file keeps
  `L = 0` and the oracle tracks 1 µm instead.  The thin-gap transverse RF kick is the
  `LineElement` lens `lattix.formats.base.with_rf_focusing` writes for every non-TraceWin target.
* Kicks: `Kn0L = −hkick`, `Ks0L = +vkick` (`px += hkick` measured).  H⁻ is `Species("#1H-")`
  (`m_p + 2 m_e`); plain `"H-"` is the isotope-averaged anion.  A species outside the named table
  is written with the eight-argument `Species(name, charge, mass, …)` constructor
  (`SPECIES_NOT_REPRESENTABLE`).

## Known limits

* SciBmad 0.5.2 keeps one reference momentum per beamline: accelerating lattices rely on the delta
  energy mode (right for the engine's own particle, documented in the ledger), and a
  `ReferenceChange` is a tag on a marker, not an element the engine applies.
* Zero-length cavities are untrackable in 0.5.2 (the oracle substitutes 1 µm); `edge1_int`/`edge2_int`
  are stored but not tracked (zeroed by the oracle with a warning), so bends with fringe integrals
  are report-only against SciBmad in the battery.
* No foil, NCells or RFQ-cell physics; field maps go through the degradation ladder.
* Bmad's `bmad_to_scibmad` output uses spellings (`SaganCavity(num_cells = …)`,
  `PhaseReference.*`) that SciBmad 0.5.2 itself does not accept; the reader keeps them as text and
  reads the rest.

## Oracle

`lattix/oracles/scibmad_worker.jl` loads the file into a fresh module, takes the last (or the
named) `Beamline`, tracks the reference particle element by element and measures each element's
6×6 map by central finite differences around that orbit (SciBmad puts the RF gain into `pz`).
Basis `bmad`, `FOLLOWS_P0 = False` in the battery.  Measured against the other engines:
`fodo.madx` vs cpymad `8e-11`, `mebt_line.dat` vs HELIX `1.2e-10`, `dtl_section.dat` vs HELIX
`1.6e-11` on the transverse block.
