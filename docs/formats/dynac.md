# DYNAC (V6 deck)

`lattix/formats/dynac/` reads and writes the decks of [DYNAC](http://dynac.web.cern.ch/dynac/dynac.html)
V6R16 (CERN/CEA/ESS; a title line, then unnamed type codes each followed by their parameter lines; cm, kG,
MV, MeV, degrees).  DYNAC is freeware under its own EULA (free to use and pass on, not to sell): lattix
writes and reads the deck text; the engine runs locally through `lattix/oracles/dynac.py` when a built
`dynac` binary is at hand (`LATTIX_DYNAC_EXE`), never in CI, and nothing from its `datafiles/` enters the
repository.  Every convention below was measured on DYNAC V6R16 built with gfortran 16 on 2026-09-05
(`docs/oracles.md`, Phase 5.8).

## Reading

`lattix/formats/dynac/cards.py` tokenizes a deck the way DYNAC's list-directed input does: each card's
entries are gathered read by read, a read short of numbers continuing on the next line (the SNS example
splits its 16-entry `CAVSC` lines in two), `;` lines being comments.  The beam block gives the reference
particle: `GEBEAM` (the master frequency) + `INPUT` (rest mass `UEM × ATM`, charge, kinetic energy) or
`RDBEAM`; lattix's `; lattix: reference` tag or `read(species=, kinetic_energy_eV=, frequency_Hz=)` win
over it (a rest mass that is none of lattix's named species becomes an ion of that mass, EQUIVALENT
`SPECIES_ASSUMED`; a title-less converter output such as `tw2dyn`'s is read too).

| DYNAC card | IR | Notes |
|---|---|---|
| `DRIFT`, `FDRIFT` | Drift | cm → m |
| `QUADRUPO L B_tip R` | Quadrupole | `Bn1 = 10·B_tip/R` T/m (lab field: a positive pole-tip field focuses a positive charge in x), aperture `R`; a preceding `TWQA` roll becomes the tilt |
| `SEXTUPO`, `QUADSXT`, `QUAFK`, `SOQUAD` | Sextupole, Quadrupole, Solenoid | strengths (`IMKS = 0`: `k` in cm⁻ⁿ normalized with the signed rigidity at the entrance) or fields; `SOQUAD`'s quadrupole part LOSSY `SOQUAD_QUAD_DROPPED` |
| `SOLENO IMKS L ARG` | Solenoid | `B` kG or `k = B/(2Bρ)` cm⁻¹ |
| `BMAGNET` (4 lines) | Bend | arc = ρ·angle, TRANSPORT pole faces (= MAD's e1/e2), `EK1` = fint, `APB` = hgap, `XN = −k1ρ²`, `EK2` kept as `fringe_k2`; pole-face curvature LOSSY `BEND_POLE_CURVATURE_DROPPED`; a `ZROT ±a` pair around it is the tilt |
| `STEER FLD NVF` | Kicker | kick = `∫B·dl / Bρ_signed` (magnetic) or `FLD/(Eρ)` (electric, `NVF ≥ 2`); the writer's `STEER` pairs tagged `kind=Multipole` come back as thin dipole terms |
| `BUNCHER V φ h R` | RFCavity (thin) | `V` MV; the phase is charge-signed (gain `q·V·cos φ`): a negative species gets `φ − 180°`; frequency = `h ×` the master frequency (`NEWF` updates it) |
| `CAVSC` (16 entries) | Drift + RFCavity (thin) + Drift | a DTL cell: a gap of `E0·T·L` at the cell middle (EQUIVALENT `CAVSC_AS_GAP`; MEASURED 0.3 % from DYNAC's TTF-derivative model for a βλ cell, 0.6–1.2 % per SNS DTL gap) |
| `FIELD` + `CAVNUM`/`CAVMC`, `HARM` + `CAVNUM` | RFCavity (thick) | the `z, E_z` block (or the Fourier series) integrated at the entrance energy for the voltage (`gain_from_profile`); lattix's own decks carry the calibrated voltage in the tag |
| `NEWF f` | Freq | Hz |
| `NREF` | ReferenceChange | `IREWF` 1: dW MeV, 2: absolute, 0: percent |
| `ALINER` | (misalignment) / Patch | a `±` pair around an element is its transverse offset; a lone one is a patch (angle entries LOSSY `ALINER_ANGLES_AS_PATCH`) |
| `ZROT`, `CHANGREF` | (bend tilt) / Patch | a lone `ZROT` is a patch tilt; `CHANGREF` a yaw patch (EQUIVALENT `CHANGREF_AS_PATCH`) |
| `REJECT` | (aperture) / Collimator | a non-default window is the aperture of the element that follows (DYNAC's defaults `1000 4000 100 100 400` clear it) |
| `EMIT`, `EMITL` | Marker / Instrument | the family from the tag |
| `STRIPPER` | Foil | material from `Z`, thickness g/cm² → kg/m² |
| `RFQPTQ`, `RFQCL`, `EGUN`, `FSOLE`, `QUAELEC`, `EDFLEC` | Drift / Marker | LOSSY `RFQ_NOT_TRANSLATED`, `EGUN_NOT_TRANSLATED`, `FSOLE_NOT_TRANSLATED`, `EQUAD_TO_DRIFT`, `EDFLEC_TO_DRIFT` (the survey kept where a length is known) |
| everything else (`SCDYNAC`, `EMITGR`, `TOF`, `REFCOG`, …) | Directive (`format="dynac"`) | re-emitted verbatim by the DYNAC writer (EXACT `DYNAC_DIRECTIVE_KEPT`), dropped by the others |

Normalized strengths become fields with the signed rigidity at each element's entrance, the energy
following the bunchers and cavities (cavities first, in order).  The `; lattix: name=… kind=…` tags
restore names, kinds, families, thick-cavity lengths and traveling-wave types.

## Writing

The deck: a title, `; lattix: reference` and `; lattix: lattice` tags, a `GEBEAM`/`INPUT` block for the
reference (a placeholder ellipsoid — DYNAC needs a bunch, lattix carries no emittance; `beam_extent=`),
`EMIPRT 2`, one tagged card group per element, `STOP`.

| IR kind | DYNAC | Ledger |
|---|---|---|
| Drift | `DRIFT L` | EXACT |
| Quadrupole | `QUADRUPO L B_tip R` (`B_tip = Bn1·R/10` kG at `R` = the aperture or `quad_radius_m`); a roll is a `TWQA` pair | EXACT (`QUAD_TILT_AS_TWQA`; other orders LOSSY `MULTIPOLE_ORDERS_DROPPED`) |
| Sextupole | `SEXTUPO 1 B_tip L R` | EXACT (a drift at first order in DYNAC) |
| Octupole | `DRIFT` | LOSSY `OCTUPOLE_TO_DRIFT` |
| Multipole (thin) | `STEER` for the dipole terms | EQUIVALENT `MULTIPOLE_AS_STEER`, higher orders LOSSY |
| Bend | `BMAGNET 1 / ANGL RMO BAIM XN 0 / e1 0 fint K2 hgap / e2 0 fintx K2 hgap` inside `ZROT ±tilt`; `BAIM = −|Bρ|/ρ` for a negative species | EXACT (`BEND_FRINGE_DROPPED` when a gap has no integral) |
| Solenoid | `SOLENO 1 L B[kG]` | EXACT |
| RFCavity | `BUNCHER V[MV] φ h R` (`φ + 180°` for a negative species); a thick one is the buncher at its centre between half-length drifts | EXACT; thick EQUIVALENT `THICK_CAVITY_AS_BUNCHER` |
| NCells | `FIELD` (a π/2π cell-train profile calibrated to the train's voltage, in `<deck>.fields.txt`) + `CAVNUM` | EQUIVALENT `NCELLS_AS_CAVNUM` (`DYNAC_NCELLS_PARAMS` for the unmapped columns) |
| FieldMap | `FIELD` (the 1-D `E_z` map itself, scaled by `ke`) + `CAVNUM` at the synchronous phase; else the hard-edge ladder | EQUIVALENT `FM_AS_CAVNUM` / `FM_*` |
| RFQCell | `DRIFT` | LOSSY `RFQCELL_TO_DRIFT` |
| Kicker | `STEER FLD[T·m] = kick·Bρ_signed, NVF` (+ `DRIFT` for the length) | EXACT (`KICKER_THIN_AT_ENTRANCE`; `EKICK_AS_MAGNETIC`) |
| Collimator | `REJECT` window, `DRIFT`, `REJECT` defaults | EQUIVALENT `COLLIMATOR_AS_REJECT` |
| Marker, Instrument | `EMIT` | EXACT / EQUIVALENT `INSTRUMENT_AS_EMIT` |
| Foil | `STRIPPER Z A g/cm² A_projectile` | EQUIVALENT `FOIL_AS_STRIPPER` |
| Taylor | `DRIFT` | LOSSY `TAYLOR_DROPPED` |
| Patch | `ALINER` for the offsets, `ZROT` for the tilt | EQUIVALENT `PATCH_AS_ALINER` (`PATCH_ROTATION_DROPPED`) |
| ReferenceChange | `NREF 0 dW 0 1` | EQUIVALENT `REFCHANGE_AS_NREF` |
| Freq | `NEWF f` | EXACT |
| Directive | a DYNAC one verbatim; a foreign one a comment | EXACT `DYNAC_DIRECTIVE_KEPT` / DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Element apertures become `REJECT` windows around the element (EQUIVALENT `APERTURE_AS_REJECT`; an
ellipse is a circle of the larger half axis, LOSSY `APERTURE_SHAPE`); transverse misalignments a pair of
opposite `ALINER` beam shifts (EQUIVALENT `MISALIGN_AS_ALINER`).

## Units and conventions

* cm, kG, MV, MeV, degrees, Hz (`NEWF`, `GEBEAM`); particle coordinates `(x cm, x′ rad, y cm, y′ rad,
  φ rad late-positive w.r.t. the master frequency, W MeV)` — a 1 m drift at 2.1 MeV gives `R56 =
  −12.10 rad/MeV` (`Basis.DYNAC`, `_Z_SIGN = −1`, `d[4] = −βλ/2π`, `d[5] = 10⁶/(β²γmc²)`);
  `WRBEAM` dumps carry six significant digits, so maps fitted from them are good to ~1e-5.
* **Rest mass** is `UEM × ATM` (`INPUT`/`RDBEAM`): lattix writes `UEM = m/A` with `A = round(m/u)`.
* **Quadrupole**: `B_tip` is the lab pole-tip field (a positive value focuses a positive charge in x, an
  H⁻ is defocused by the same card): `Bn1 = 10·B_tip/R` T/m, whatever the charge.  `SOLENO 1 L B`
  takes the lab field; the rotation follows the charge (`k = B/(2Bρ_signed)`).  `TWQA 0 a` is MAD's
  `tilt = a` (R13 = −0.109 for +10° in both).
* **Bend** (`BMAGNET`): TRANSPORT conventions — a positive `ANGL` bends towards −x (MAD-X's survey too),
  `PENT1/2` are MAD's `e1/e2`, `EK1` the fringe integral (MAD `fint`) with `APB` the half gap, `XN =
  −k1ρ²`.  With `BAIM = 0` the field is derived from the reference and a negative species is bent the
  wrong way (its rigidity is taken positive): lattix writes `BAIM = −|Bρ|/ρ` kG for `q < 0`, which
  reproduces the proton geometry (R16, R51 signs).  A 10° sector and a 10° wedge (faces 5°, fint 0.45,
  gap 6 cm, n = 0.5) agree with MAD-X to 1e-5 in every block.  `ZROT +90` around the magnet bends
  down, like MAD-X `tilt = +π/2` (R36 > 0 in both): `ZROT a` = MAD's tilt.  A negative `ANGL` mirrors
  the map's x′ column and a negative `RMO` displaces the reference (both wrong), so a **bend to the
  left** is the positive magnet turned by `ZROT 180` — with its pole faces **negated**: MAD-X's
  `(angle < 0, e1, e2)` is `(angle > 0, tilt = π, −e1, −e2)` to 1e-17 (its face angles are measured
  from the x axis), and the IR follows MAD-X (tag `neg=1`; the PSB's rectangular left bends then come
  out drift-like in x, as in MAD-X).
* **Buncher**: the particles gain `q·V·cos φ` (the phase is charge-signed like TraceWin's raw phase: a
  negative species needs `φ + 180°`); a late particle gains more for `φ < 0` (native `R65 = +0.5
  MeV/rad` for 1 MV at −30°: bunching in the common basis).  The transverse RF kick uses the mid-gap
  velocity (`R21 = 2.32 /m` for 1 MV at 2.1 MeV; TraceWin/HELIX's entrance-velocity formula gives
  3.02 /m): Equivalent tier on thin gaps.  The reference bookkeeping after an `RDBEAM` gives the
  "reference" half a buncher's gain (`dynac.long`); `REFCOG 1` keeps it on the physics — the oracle uses
  the on-axis probe particle's absolute energy (`WRBEAM IREC = 1`) and `REFCOG 1`.
* **CAVSC** (a DTL cell): gains `E0·T·L·cos φ` within 0.3 % for a βλ cell, for a proton and an H⁻
  alike (the phase is *not* charge-signed); a `TP = 0` entry makes DYNAC produce NaN.
* **FIELD + CAVNUM**: the file holds `f [Hz]`, then `z [m] E_z [V/m]` pairs ending in `0. 0.`; DYNAC
  counts cells from the zero crossings and dies on long runs of zeros (lattix writes the nonzero span
  and pads with drifts).  `CAVNUM DPHASE` is relative to DYNAC's own crest: a 20 cm raised cosine gains
  11 216 eV at crest against 11 208 from an RK4 integration of the same field (7e-4), 9 732 vs 9 725 at
  −30°; but a strongly accelerating low-β cavity (1 MV on 2.1 MeV over 6 cm) gains 1.59× lattix's
  calibration, so thick cavities are written as centred bunchers (exact gain) and `CAVNUM` is kept for
  cell trains and field maps.  `HARM` + `CAVNUM` without a `FIELD` diverges.
* **ALINER** shifts the beam permanently (`+0.1` moves the beam to +x by 1 mm); `NREF 0 dW 0 1` changes
  the reference energy only; `CHANGREF` yaws the reference direction.

## Known limits

- **Bends (measured 2026-09-06):** the maps fitted from the 6-digit dumps drift by ~1e-6 per BMAGNET (9.5e-5 over
  the 4-bend chicane vs Elegant, 1.7e-3 over the 36 PIP-II BTL bends vs HELIX): bend decks are the equivalent
  tier in the battery; the 5e-5 exact floor stays for RF-only decks.
* No octupole, matrix element, RFQ cell, foil model of lattix's, pitch/yaw patch or continuous aperture
  in DYNAC; sextupoles act at second order only (`SECORD`).
* Thick cavities are thin bunchers at their centre (their length kept by drifts); cell trains and field
  maps go through `CAVNUM`, whose crest is DYNAC's own (Equivalent tier).
* Thin gaps use DYNAC's mid-gap RF defocusing (Equivalent tier against TraceWin/HELIX); the reader's
  `CAVSC` gap drops the TTF derivatives (0.3–1 %).
* A misaligned element is a pair of `ALINER` cards; DYNAC's own `RANDALI`/`MMODE` error cards are
  directives, not applied.
* Local only: the EULA keeps DYNAC and its example decks out of the repository and CI.

## Oracle

`lattix/oracles/dynac.py` (marker `oracle_dynac`; `LATTIX_DYNAC_EXE`, `dynac` on the PATH or the local
clone's build): DYNAC prints no matrices, so the adapter runs **one DYNAC job per optics card** — a fresh
`RDBEAM` probe (an on-axis particle and ± offsets in each coordinate) at the energy reached so far, the
card, a `WRBEAM` dump — and fits each card's 6×6 map from the twelve offset particles; the energies are
the on-axis particle's.  A single run with dumps after every card fits maps from *propagated* offsets,
whose conditioning collapses in strongly focusing lines (1.6e5 on the 5 m `csr_chicane.dat`) beyond
what six-digit dumps resolve — hence the jobs.  The persistent states travel along (`NEWF`, `SECORD`, an
active `TWQA`, the `FIELD`/`HARM` block a cavity reads, `NREF` offsets); `ALINER` (an affine shift) has
no first-order map and is skipped; a `TOF 0` deck is flagged (the phases cannot follow the time of flight
job by job); `REJECT`/`CHASE`/`COMPRES` are dropped and every lens gets a 10 m radius with its field
scaled along, so the probe never hits an aperture.  Measured: the fingerprint drift exact to 5e-7 and the
1 MV gap gaining 866 030 eV (6-digit dumps); HELIX's `fodo.madx` vs cpymad 3.5e-5 on T4×4 and below 2e-6
on every other block; `fodo_cell.dat` 3.2e-5, `solenoid_channel.dat` 3.9e-6 and `bend_line.dat` 8e-5 vs
HELIX; `mebt_line.dat` vs HELIX 6.4e-3 on T4×4 (the bunchers' mid-gap RF kick) and 2.4e-6 on the energy;
the synthetic `csr_chicane.dat` (an exponentially unstable line, growth 1e5) 2.4e-2 — the dump precision
amplified; the SNS MEBT + DTL tank 1 example as shipped against lattix's rewrite: the MEBT identical, the
tank's end energy to 0.3 % (7.548 vs 7.525 MeV; DYNAC's `CAVSC` gap dynamics against the thin buncher);
`tw2dyn` (DYNAC's own TraceWin converter, built from `converters/tw2dyn.f`) on the MEBT deck against
lattix's writer: every quadrupole, drift and buncher identical and the same DYNAC maps.
