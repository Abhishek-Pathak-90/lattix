# IMPACT-T (`ImpactT.in`)

IMPACT-T is LBNL's time-domain (t-code) space-charge tracker.  Its deck is a nine-record header
followed by beamline cards, each `L nseg mapstp type zedge v1 v2 … /`, positioned by an absolute
`zedge` and read from `ImpactT.in` in the working directory; field profiles come from `rfdataN`
files (Fourier coefficients) and solenoid tables from `1TN.T7`.  lattix reads and writes the format
(`lattix.formats.impactt`, format name `impactt`, file name `impactt.in` in any case) and runs
the engine as an oracle (`lattix/oracles/impactt.py`, conda-forge `impact-t` 3.1.5,
`ImpactTexe`).  Everything below was measured on that binary; see
[oracles.md](../oracles.md#phase-55-measurements-impact-t-315-2026-09-05).

## Reading

* **Header** (`Header`): `npcol nprow`; `dt ntstep nbunch`; `dim np flagmap flagerr flagdiag
  flagimg zimage`; mesh `nx ny nz flagbc xrad yrad perdlen`; `flagdist rstartflg flagsbstp nemission
  temission`; three `distparam` rows; `current kinetic_energy mass charge frequency phase`.  The
  reference particle comes from the last record (species by mass and charge, a custom species
  when no known one matches: Sample1's electron with mass 511 005 eV); every simulation setting
  is kept in `meta["impactt_header"]` (`EQUIVALENT SIM_SETTINGS_KEPT_IN_META`) and the beam current
  is `LOSSY IMPACTT_BEAM_CURRENT` (the IR carries no current).
* **Timeline**: cards are ordered by their nominal start (`zedge`, or `zedge + pad` for a lattix
  tag) on a picometre grid; the first card sets the origin (`meta["impactt_s_offset"]`, Sample1
  starts below zero); empty space between cards becomes an implicit drift (`gap_n`,
  `native["impactt"]["implicit"]`, nothing is written for it on a rewrite); a card starting
  before the previous one ends is placed at the previous exit (`LOSSY IMPACTT_OVERLAP`,
  `native["impactt"]["pushed"]`; a rewrite keeps every card's own `zedge`,
  `IMPACTT_NATIVE_POSITION`) — except a drift card, which IMPACT-T ignores anyway: it keeps only its
  uncovered part and moves nothing (`EQUIVALENT IMPACTT_DRIFT_OVERLAP`, the trace of a negative drift
  upstream in the source, which the writer does not emit: `NEGATIVE_DRIFT_DROPPED`).  Zero-length cards inside a thick card split it
  (`IMPACTT_THIN_INSIDE_THICK`) — except RF cards, which are one field: such a card goes to the
  RF card's entrance or exit, whichever is nearer.  A drift card tagged `kind=Kicker` or
  `kind=Collimator` holding one `-1`/`-11` card of the same name is one thick element again.
* **Element types**: `0` drift; `1` quadrupole (lab gradient; a fringe length or profile file
  and an RF quadrupole are recorded as `IMPACTT_QUAD_FRINGE_DROPPED`, `IMPACTT_QUAD_PROFILE`,
  `IMPACTT_RF_QUADRUPOLE`); `3` solenoid — the tracked field (`getfldt_Sol`) is an `(r, z)` table
  `1T<id>.T7`, so a lattix-written card (tag `L= Bsol= pad=`) is a hard-edge solenoid again
  (`EXACT IMPACTT_NAME_TAG`; the card's radius is its aperture), any other table gives the
  equivalent hard edge with `L_eff = (∫Bz)²/∫Bz²`, `B_eff = ∫Bz²/∫Bz` (`IMPACTT_SOLENOID_TABLE`),
  and no table is `IMPACTT_SOLENOID_NO_TABLE`; `4` dipole — the angle from `By·L/Bρ`, the
  pole-face angles from the file's face lines (`z = k·x + b`), `Bx` dropped, the Enge fringe
  model noted (`IMPACTT_BEND_BX_DROPPED`, `IMPACTT_BEND_FRINGE_MODEL`, `IMPACTT_BEND_NO_FILE`);
  `5` multipole (sextupole and octupole as thick elements, a decapole as a thin one,
  `IMPACTT_DECAPOLE_AS_THIN`); `104` cavity with an `rfdataN` Fourier profile — the reader
  integrates the reference through the profile: `V` is the largest gain over the driven phase
  and the synchronous phase follows from the gain and its slope (`IMPACTT_RF_DRIVEN_PHASE`); the
  other RF types (`101`–`103`, `105`, `110`–`113`) pass through with their columns and data file
  (`LOSSY IMPACTT_RF_GAIN_UNKNOWN`, `IMPACTT_SOLRF_BZ_DROPPED`).  Controls: `-1` steer becomes a
  `Kicker` (`hkick = dpx/γβ`; a centroid shift `IMPACTT_CENTROID_SHIFT`), `-2` an `Instrument`
  (`PHASE_DUMP`), `-11` a `Collimator`; the other negative types are `Directive`s
  (`IMPACTT_RUN_CONTROL`); `-99` ends the line.  Anything else is `DROPPED
  UNSUPPORTED_IMPACTT_ELEMENT`.
* **lattix tags** `! lattix: name=… kind=… [L=… Bsol=… pad=…]` restore names, kinds and the
  elements that had to change shape: a thin gap written as a short cavity (`L=0 pad=…`) is a thin
  gap again at its nominal position and the drifts around it get their length back; a thin
  multipole written as short type 1/5 cards is one thin `Multipole` again; a padded solenoid
  table is the nominal hard edge.
* **Clock**: the deck's time origin is the reference at z = 0.  The walk's clock starts at the
  nominal start with the field-free flight up to it, and every cavity's exit time is integrated
  through its profile, so the driven phases downstream stay consistent with IMPACT-T's own
  reference (which leaves each cavity a little earlier or later than the IR's point gain says).

## Writing

The writer positions every card at the element's `s_in` (plus the deck's offset) and ends the
line with `-99`.  Rules (`Writer.RULES`, one per IR kind — see [fidelity.md](../fidelity.md) for
every code):

* **Drift, Quadrupole, Sextupole, Octupole**: types 0, 1, 5 (a skew quadrupole is a rolled normal
  one, `IMPACTT_SKEW_AS_ROTATION`; other orders in a quadrupole are dropped).
* **Solenoid**: type 3 with a generated `(r, z)` table (`1TN.T7`, `IMPACTT_SOLENOID_TABLE`).  The
  table is a flat top whose ends are three intervals of width `h` (option `solenoid_ramp_m`,
  default 0.2 mm) straddling the nominal ends with `Bz/B = 0, −0.2318182, 1.2318182, 1` and
  `Br/(B·r/h) = 0, −¼, −¼, 0`: under IMPACT-T's bilinear interpolation its paraxial map equals
  the hard-edge solenoid's to `O(h²)` (3e-8 in the model; a plain `0 → ½ → 1` ramp is `O(h)`,
  9e-6), and IMPACT-T tracks it to 9e-7 at a 1 ps step (`tools/impactt_solenoid_table.py`).  The
  table interval needs at least ten integration steps; its radial extent is the element's
  aperture radius (1 m without one).
* **Bend**: type 4 with `By = Bρ·θ/L` and the 22-value pole-face file (`k1 = tan e1`,
  `k4 = tan(|θ| − e2)`, 1 nm Enge zones); `Bx`, tilt, fringe integrals and multipole terms are
  recorded (`IMPACTT_BEND_TILT_DROPPED`, `BEND_FRINGE_DROPPED`, `MULTIPOLE_ORDERS_DROPPED`); a
  zero-angle bend is a drift.  **IMPACT-T's dipole bends the whole bunch by the reference angle
  with no pole-face focusing (`R21 = R26 = 0`)** — a translated bend is geometrically right and
  optically the engine's own model.
* **RFCavity**: type 104 with a generated `rfdataN` (`IMPACTT_RF_PROFILE`): one raised-cosine
  bump no wider than `βλ/2` (or `L_active`), or `n_cell` half-wave cells for a multi-cell cavity.
  The driven phase `theta0` and the field scale are calibrated by integrating the reference
  through the profile at lattix's time of flight (`rfprofile.calibrate`): the largest gain over
  `theta0` equals `V` and the reference gains `V·cos φs` on the branch where a later particle
  gains more at a negative phase (`IMPACTT_RF_DRIVEN_PHASE`; a driven source phase
  `IMPACTT_RF_PHASE_NOT_SYNC`).  A thin gap becomes a short cavity centred on the gap whose length
  the neighbouring drifts give up (`THIN_GAP_AS_SHORT_CAVITY`; free space between cards is used
  next, and only then the line grows, `LOSSY THIN_GAP_ADDS_LENGTH`).  Its length balances the two
  ways a field integration departs from the thin-gap model — the kick is smeared over the bump
  (∝ length) and a strong short bump adds ponderomotive focusing (∝ V²/length):
  `0.4·sqrt(qVλ/(2π mc² βγ³ |sin φ|))` within `[1 mm, βλ/2]` (5 mm for the MEBT's 80 kV gaps,
  16–20 mm for a DTL's 577 kV gaps; the option `thin_gap_length_m` fixes it).  No voltage or
  frequency: a drift (`IMPACTT_CAVITY_TO_DRIFT`).
* **FieldMap**: the replacement ladder of [conventions.md](../conventions.md) (cavity, hard-edge
  solenoid or quadrupole), each part written by its own rule.  **NCells**: a cell train profile
  (`NCELLS_AS_PROFILE`, its cell parameters `IMPACTT_NCELLS_PARAMS`).
* **Multipole (thin)**: dipole terms become a `-1` steer at the position (`IMPACTT_MULTIPOLE_AS_KICK`,
  `dpx = −BnL/Bρ·γβ`, `dpy = +BsL/Bρ·γβ`); higher orders become 1 mm type 1/5 cards tagged
  `L=0 pad=…` (`IMPACTT_THIN_MULTIPOLE_AS_THICK`; the reader folds them back into one thin
  multipole).
* **Kicker**: a `-1` card at the centre (`dpx = hkick·γβ`); a thick kicker is a tagged drift
  around it (`THICK_KICKER_SPLIT`; an electric kicker `EKICK_AS_MAGNETIC`).  **Collimator**: a
  `-11` card at the centre (rectangular flag 1, round 11; `THICK_COLLIMATOR_AT_CENTRE`).
* **Marker, Instrument, Foil, Taylor, Patch, ReferenceChange, Freq, Directive**: zero-length
  tagged drift cards or nothing, each with its ledger entry (`INSTRUMENT_AS_MARKER`,
  `FOIL_TO_MARKER`, `TAYLOR_DROPPED`, `PATCH_DROPPED`, `REFCHANGE_DROPPED`, `FOREIGN_DIRECTIVE`).
* **Header**: `dt` (option `dt_s`, default 1 ps), `ntstep` from the length at the initial
  velocity, `np`, mesh, `flagdist 2`, the reference `kinetic_energy mass charge` and the RF
  frequency of the lattice (`LOSSY IMPACTT_NO_FREQUENCY` when there is none: 1 GHz).  A deck that
  came from IMPACT-T keeps its own header from `meta["impactt_header"]` and its data files keep
  their ids.
* **Origin**: IMPACT-T evaluates no field at z < 0.  A fresh deck whose first field region would
  start below zero (a solenoid table ramps before its nominal end) is moved so that it starts at
  zero (`EQUIVALENT IMPACTT_DECK_OFFSET`); the beam is expected at z = 0 at t = 0 and the driven
  phases include the field-free flight to the nominal start.  A deck that came from IMPACT-T keeps
  its own origin.
* Misalignments are dropped (`IMPACTT_MISALIGNMENT_DROPPED`) unless `errors=True` writes the
  `dx dy rot_x rot_y rot_z` columns; a tilted quadrupole then needs them
  (`IMPACTT_TILT_NEEDS_ERRORS`).

`write(lat, "ImpactT.in", "impactt", dt_s=…, n_particles=…, current_A=…, solenoid_ramp_m=…,
thin_gap_length_m=…, harmonics=…, errors=…)`; `write_rfdata=False` skips the data files.

## Units and conventions

| Quantity | IMPACT-T | IR |
|---|---|---|
| positions, lengths | m, absolute `zedge` | `s_in` + `meta["impactt_s_offset"]` |
| time | s, the reference at z = 0 at t = 0 | the walk's clock, started with the flight to the nominal start |
| quadrupole | lab gradient T/m | `Bn[1]` |
| solenoid | `Bz0` T = scale of the table | `Bsol_T` (the table's plateau is 1) |
| dipole | `By` T, faces `z = k·x + b` rel. to `zedge` | `angle = By·L/Bρ_signed`, `e1 = atan k1`, `e2 = |θ| − atan k4` |
| cavity | `scale` V/m × `Ez(u)`, `theta0` deg driven on the absolute time | `voltage_V` = largest gain over `theta0`, `phase_rad` synchronous |
| steer | `dpx` in γβ units | `hkick = dpx/γβ` |
| energies | eV (header), `fort.18` reference | `kinetic_energy_eV` |

## Known limits

- **Pole faces of a negative bend (2026-09-06):** the face lines are the x → −x mirror of the positive bend with
  the faces negated: `k1 = tan(e1)`, `k4 = s·tan(|θ| − s·e2)` (the exit slope was `s·tan(|θ| − e2)` before).
  The oracle resumes its final chunk after a missed dump, reports the tail after the last written dump with
  a NaN map instead of dropping it, and keeps only the uncovered part of a drift card that overlaps the
  previous element.
* The dipole model (no pole-face focusing) makes bend decks report-only in the battery.
* Thin gaps are short cavities: their transverse RF focusing is a field integration's, not the
  thin-gap formula's (Equivalent tier: `mebt_line` 8.1e-3 against HELIX with the reference energy
  1.5e-7; a DTL section's 577 kV gaps, 27 % of the energy each, are beyond the thin-gap model —
  5 on the cumulative map — and report-only).
* A solenoid table needs ≥ 10 integration steps per table interval (at `h/5` the sampled edge
  kick is off by 5e-5); the oracle sizes its steps accordingly.
* IMPACT-T triggers its controls (`-1`, `-2`, `-4`, `-11`, `-99`) on the bunch centroid and walks
  each control list in order: a probe that runs away (rings under the dipole model) can drag the
  centroid past a control inside a dipole loop, and every later control of that kind is then
  skipped.  The oracle keeps 1 cm field-free clearances around dipoles, dumps only where a drift
  can hold them, and resumes a chunk from the last dump that was written (`meta["missing_dumps"]`).
* Passthrough RF types (`101`–`103`, `105`, `110`–`113`) carry no voltage in the IR: the walk's
  reference energy is not updated through them.
* Space charge, the mesh, image charges and emission settings are simulation settings kept in
  `meta["impactt_header"]`, not physics the IR checks.

## Oracle

`lattix/oracles/impactt.py` runs `ImpactTexe` on an instrumented copy of the deck: a 13-particle
probe read from `partcl.data` (`flagdist 16`, `nemission −1`), launched a little before the first
control (a padded table, a dipole's probe lead) with the driven phases delayed accordingly; `-4`
cards giving every element its own step `dt = L/(N βc)` (the reference lands exactly on the
boundaries: quads 4e-8, drifts 5e-13); `-2` dumps at every boundary (fixed-time snapshots of
`x, γβx, y, γβy, z, γβz`, drifted to the reference plane), central-difference Jacobians and
`R_elem = J_i·J_{i−1}⁻¹`; positions relative to the deck's nominal start.  A control fires on the
integration step nearest to it (the cards sit exactly on the boundary).  IMPACT-T keeps every
negative-type card of a run in one array of 200 (`Nbpmmax`), so long decks are chunked at 186 such
cards, each chunk's deck trimmed to the cards its bunch can reach (`Ndriftmax`, `Nquadmax` …) and
its `theta0` shifted by `360·f·t`.  A dipole starts its own loop when the *first* particle reaches
it and re-bases the frame one step past the exit face: the step switch and the dump go 1 cm before
the face and 1 cm plus one step past it; an element shorter than that overshoot carries the
clearance to the next.  `meta` reports the workdir, steps, chunks, bends, missing dumps and the
`bend_model` note.
