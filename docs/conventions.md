# Conventions

This page is the physics contract of lattix: the units and sign conventions of the
intermediate representation (IR), the rule that propagates the reference energy, and the
per-format conversions the readers and writers apply.  Every convention marked *measured*
was pinned against the named engine on 2026-09-03 (see [oracles.md](oracles.md) for the
raw numbers); nothing here is recalled from a manual alone.

## 1. The IR in one paragraph

An IR `Lattice` holds element *definitions*, nested `Line`s (with `repeat` and `reverse`),
the `use` root, a `ReferenceParticle`, symbolic variables, and side channels (`commands`,
`errors`, `meta`).  `flatten()` expands the root line into `Placed` views carrying
`s_in`/`s_out` and the reference particle at both ends; `propagate()` walks the energy.
There are 22 element kinds, named after the PALS standard:

`Drift`, `Quadrupole`, `Sextupole`, `Octupole`, `Multipole`, `Bend`, `Solenoid`, `RFCavity`,
`FieldMap`, `NCells`, `RFQCell`, `Kicker`, `Collimator`, `Marker`, `Instrument`, `Foil`,
`Taylor`, `Patch`, `ReferenceChange`, `Freq`, `Directive`, `Superposition`.

Parameter groups mirror PALS: `MagneticMultipoleP` (`Bn`, `Bs`, `BnL`, `BsL`, `tilt`),
`BendP` (`angle`, `e1`, `e2`, `edge_int1/2`, `hgap`, `tilt_ref`, rectangular flags),
`RFP` (`voltage_V`, `gradient`, `phase_rad`, `frequency_Hz`, `harmon`, `n_cell`, `dE_ref_eV`,
`phase_is_sync`), `SolenoidP` (`Bsol_T`), `ApertureP`, `BodyShiftP`.  Every element also
carries `native` (per-format passthrough, re-emitted only to the same format) and
`provenance` (source file, line, original name and type).

## 2. Units

The IR is SI plus electron-volts everywhere: m, rad, T, T/m, T·m^(1−n), V, V/m, Hz, s, eV.
**Phases are radians, cosine convention, 0 = crest.**  The boundary conversions are:

| Format | Length | Angle | Field / strength | RF voltage | RF frequency | RF phase | Energy |
|---|---|---|---|---|---|---|---|
| TraceWin `.dat` | mm | deg | `QUAD G` T/m, `BEND ρ` mm, `SOLENOID B` T, `THIN_STEERING Bx By` T·m | `GAP E0TL` V | `FREQ` MHz | deg | MeV (run input) |
| MAD-X / MAD8 | m | rad | normalized `k1`, `ks`, `knl/ksl` at the BEAM rigidity | `volt` MV | `freq` MHz | `lag` turns | `beam energy` GeV |
| Elegant `.lte` | m | rad | normalized `K1`, `KS`, `KNL` | `RFCA VOLT` V | `FREQ` Hz | `PHASE` deg | eV (run setup) |
| Bmad | m | rad | normalized `k1`, `ks`, `k{n}l`, `k{n}sl` | `voltage` V | `rf_frequency` Hz | `phi0` turns | `parameter[e_tot]` eV |
| PALS | m | rad | `Bn`/`Kn` (both allowed) | `voltage` V | `frequency` Hz | `phase` turns | eV |
| ImpactX | m | rad (`DipEdge psi`), deg (rotations) | normalized `k` 1/m² | `ShortRF V` = voltage_V / mass_eV (dimensionless) | Hz | deg | MeV (`beam.kin_energy`) |
| IMPACT-Z | m | rad | lab gradient T/m (type 1) | ideal cavity gradient V/m | Hz | deg | eV (`fort.18` reference) |
| FLAME GLPS | m | deg (`sbend phi`) | normalized `K` 1/m² | cavity `scl_fac` × tabulated TTF | `SampleFreq` Hz | deg | MeV/u (`IonEk`) |
| xtrack | m | rad | normalized `k1`, `knl/ksl` | `voltage` V | `frequency` Hz | `lag` deg (0.103) / `phase` rad (0.112) | eV |
| PyORBIT3 linac XML | m | rad | `QUAD field` T/m (lab), `SOLENOID B` = B₀/Bρ 1/m, `DCH/DCV B·effLength` T·m | `RFGAP E0TL` GeV | `<Cavity frequency>` Hz | deg | GeV (`bunch.getSyncParticle().kinEnergy()`) |
| IMPACT-T `ImpactT.in` | m, absolute `zedge` | rad (pole faces as lines `z = k·x + b`) | lab gradient T/m (type 1), `Bz0` T scaling an `(r, z)` table (type 3), `By` T (type 4) | type 104 `scale` V/m × Fourier `Ez` (`rfdataN`, period = card length) | Hz (card and header) | `theta0` deg, driven on the absolute time | eV (header; `fort.18`) |
| Ocelot module | m | rad | normalized `k1 = Bn1/Bρ_signed`, `k = Bsol/(2Bρ_signed)`, `kn = [BnL_n/Bρ_signed]` (MAD's `knl`) | `Cavity.v` GV | `freq` Hz | `phi` deg (`phi = −φs`) | GeV total (`tws0.E`) |
| DYNAC deck | cm | deg (`BMAGNET ANGL`, pole faces, `ZROT`, `TWQA`) | lab pole-tip field kG at radius `R` (`QUADRUPO`), `SOLENO` kG, `STEER` ∫B·dl T·m, `BMAGNET XN = −k1ρ²` | `BUNCHER V` MV; `CAVSC E0` MV/m × T × L; `FIELD` V/m | Hz (`GEBEAM`, `NEWF`) | deg, charge-signed for `BUNCHER` (`+180°` for a negative species), crest-relative for `CAVNUM` | MeV (`INPUT`; rest mass `UEM × ATM`) |
| Synergia lattice JSON | m | rad | MAD-X's normalized `k1`, `k1s`, `knl`/`ksl`, `ks` — all with the design (start) rigidity, Synergia scales by `p_design/p_bunch` itself | `volt` MV | `freq` MHz | `lag` turns (`lag = φ/2π + ¼`, no slip) | GeV total (`four_momentum.energy`; one design momentum, `energy_mode=constant`) |
| OPAL-T deck | m (`ELEMEDGE` path length) | rad | `K1`, `K2`, `K3`, `KN`, `KS` over the **BEAM's initial** `P0/c` (unsigned) | `VOLT` MV/m peak of a normalised map; `DESIGNENERGY` MeV = the crest energy | `FREQ` MHz | `LAG` rad relative to the autophased crest | GeV (`BEAM PC`); `DESIGNENERGY` MeV kinetic per bend/kicker/cavity |
| HELIX (adapter) | mm | deg | T, T/m | MV | MHz | deg | MeV |

## 3. Reference particle and rigidity

`Species(name, mass_eV, charge)`.  Protons use the CODATA-2018 mass 938 272 088.16 eV; H⁻ is
m_p + 2 m_e with charge −1.  Bmad's own proton mass is 938 272 089.43 eV (1.35e-9 larger,
*measured*), so the Bmad adapter reports Bmad's `mass_of()` rather than the IR's.

Canonical strengths are **lab-frame fields** (`Bn1` = G in T/m, `Bsol_T`, `BnL`, kicker angles
in rad).  Normalized views are computed with the *signed* rigidity at the element **entrance**:

```
brho_abs    = pc / (|q| c)
brho_signed = sign(q) · brho_abs
k1  = G / brho_signed          (lattix.ir.normalize.k1_from_gradient)
ks  = Bsol / brho_signed       (ks_from_field)
knl = BnL / brho_signed        (kn_from_bn: MAD's k_n L, no factorial)
kick = ∫B·dl / brho_signed     (kick_from_bl)
```

All MAD-family normalized strengths (MAD-X, MAD8, Elegant, Bmad, xtrack, ImpactX, FLAME `K`)
use `brho_signed`, so an H⁻ deck's `k1` has the opposite sign of a proton deck's for the same
magnet.  HELIX's importers were inconsistent here (signed for MAD-X, unsigned for Elegant); the
IR has one rule and the invariant tests pin it with cpymad and Bmad running a `charge = −1`
beam.

Constant-p0 codes (MAD-X, MAD8, xtrack, SciBmad) know one rigidity per deck.  Their reference particle
keeps `p0` but *does* pick up the RF gains the deck contains as `δ = Δp/p0` (measured: MAD-X
`twiss` carries the gain in the orbit's `pt`, xtrack in `delta`), while an explicit reference
change (TraceWin `SET_BEAM_ENERGY`, a `ReferenceChange`) cannot reach it at all.  Writing an
accelerating lattice to them therefore has three `energy_mode`s (`lattix.ir.energy_mode`),
all tagged in the deck (`! lattix: energy_mode=…`) so the reader undoes them, all in the ledger:

* `delta` (default): `Bρ_used = Bρ_start · p_local/p_probe` with `p_probe` the momentum after
  the RF gains only — the start rigidity across pure RF acceleration (Bmad's own MAD-X
  converter renormalizes `k1·p0c/p0c_start` the same way), the local rigidity across reference
  changes.  Kicks, `Taylor` maps and a bend's `k0` are rescaled by the same ratio
  `r = Bρ_local/Bρ_used` (`CONST_P0_DELTA_RIGIDITY`).  Measured: with the local rigidity instead,
  a `k1 = 4.08` quad after a 1 MV on-crest gap at 2 MeV reads back as `k_eff = 3.33` in both
  MAD-X and xtrack — the chromatic weakening `k1/(1+δ)` corrected it a second time.
* `local`: `Bρ_used = Bρ_local` at each element's entrance, the section-wise deck one writes by
  hand (`CONST_P0_LOCAL_RIGIDITY`); right only when the engine's particle does not gain energy.
* `constant`: `Bρ_used = Bρ_start` everywhere (`CONST_P0_START_RIGIDITY`).

The reference change itself travels as a lattix tag on the marker that replaces it
(`REFCHANGE_AS_TAG`, restored on read); MAD-X's `twiss` linearises about the accelerated orbit
to second order only, so at large `δ` (a DTL: `Δp/p ≈ 0.3`) its maps are several percent off
whatever the mode — compare such lines against Elegant, Bmad, ImpactX or TraceWin.

## 4. Energy walk

`lattix.ir.walk.propagate` advances `(s, t, E_kin, RF clock)` element by element:

| Kind | Reference energy gain |
|---|---|
| `RFCavity` | `dE = voltage_V · cos(phase_rad)` (thick cavities: `gradient · L_active`); `rf.dE_ref_eV` overrides when the source provided it |
| `FieldMap`, `NCells`, `RFQCell`, `Superposition` | `rf.dE_ref_eV` when known (field maps: from `integrate_map`); otherwise 0 with a walk warning |
| `ReferenceChange` | explicit `dE_ref_eV` (TraceWin `SET_BEAM_ENERGY`, `SET_BEAM_E0_P0`, xtrack `ZetaShift` → recorded) |
| `Freq` | switches the RF clock; every RF element with its own `frequency_Hz` also moves the clock (HELIX `RFGap.advance_ref` semantics) |
| everything else | `ref_out = ref_in` |

The gain is **species-independent** by construction: the IR phase is the synchronous phase of
the reference particle and the charge sign is already folded in.  Invariant I-6 checks
`Σ dE_ref = E_end − E_start` on every accelerating lattice against a p0-following engine.

## 5. RF phase per format (`lattix.ir.rf`)

| Format | Conversion (φ = IR synchronous phase, rad) | Function pair | Measured evidence |
|---|---|---|---|
| MAD-X, xtrack (`lag`) | `lag = φ/2π + 0.25`; gain = `V·sin(2π·lag)`, species-independent | `madx_lag`, `phase_from_madx_lag` | cpymad 5.09.03: charge ±1 give the same gain; xtrack 0.103/0.112 `Cavity(lag=60°)` = +V·cos 30° to 5e-14 |
| Bmad `lcavity` | `phi0 = φ/2π`, 0 = crest, species-independent | `bmad_phi0`, `phase_from_bmad_phi0` | ΔE = V·cos(2π·phi0) reproduced exactly with `l = 0, cavity_type = traveling_wave` |
| Elegant `RFCA` | `PHASE = φ° + 90` for negative species, `φ° − 90` for positive | `elegant_phase_deg`, `phase_from_elegant_deg` | elegant 2026.3.0 tracking: proton `PHASE=240` and H⁻ `PHASE=60` both give +V·cos 30° |
| TraceWin `GAP`/`FIELD_MAP` | deg; after `SET_SYNC_PHASE` the deck phase *is* the synchronous phase; otherwise it is the raw RF phase with gain `q·E0TL·cos φ_rf`, so negative species get a π shift | `tracewin_phase_deg`, `phase_from_tracewin_deg` | TraceWin binary and HELIX `RFGap.advance_ref` (signed charge) |
| PALS `RFP.phase` | turns, `zero_phase = ACCELERATING` → 0 = crest | `pals_phase` | pals-schema 0.3.0 / Bmad `write pals` |
| ImpactX `ShortRF.phase` | deg, 0 = crest, no shift | (writer) | ImpactX 26.08: cavity gain 866 025.404 eV at −30° |
| IMPACT-Z ideal cavity | any type other than 0/1/4 with negative `Param(5)`: gradient V/m, synchronous phase deg, gain `E0·L·cos φs` with no charge factor | (writer) | `BeamBunch.f90:323-437`; gain to 1.2e-16 |
| FLAME `rfcavity phi` | synchronous when `syncflag ≥ 1`, but the gain is a tabulated TTF polynomial, not V·cos φ | reader records `FLAME_CAVTYPE_VOLTAGE_UNKNOWN` | 3.2 % off V·cos φ at −35° |
| PyORBIT3 `RFGAP phase` | deg; `ΔE = q·E0TL·cos(phase)`, so a negative species gets `phase + 180°` (the TraceWin rule without `SET_SYNC_PHASE`) | (writer/reader) | PyORBIT3 `22b45fa`: a proton at −30° and H⁻ at 150° both gain +V·cos 30°; the slope bunches |
| IMPACT-T type 104 `theta0` | a driven phase on the absolute time (`scale·Ez·cos(2πf·t + θ0)`); lattix integrates the reference through the profile: `V` is the largest gain over θ0, the reference gains `V·cos φs`, the branch from the slope (a later particle gains more at φs < 0) | (writer/reader, `rfprofile.calibrate`) | IMPACT-T 3.1.5: proton and H⁻ thin gaps gain 866 025.7 eV at −30° (4e-7 of V·cos 30°), a 0.2 m cavity at 325 MHz 6.7e-7; the slope bunches |
| Ocelot `Cavity.phi` | deg; gain `v·cos(phi)` GeV for any species (Ocelot has no charge); `R65 ∝ +sin(phi)` with τ late-positive, so the IR's bunching at φs < 0 needs `phi = −φs` | (writer/reader) | Ocelot 25.06.0: 866 025.4 eV and `R65_common = −0.51` at `phi = +30°` for the 1 MV fingerprint gap (a surrogate cavity: Ocelot's matrix divides by the length) |
| DYNAC `BUNCHER PDP` | deg; gain `q·V·cos φ` (the TraceWin raw-phase rule: a negative species needs `φ + 180°`); a late particle gains more for `φ < 0`; `CAVSC`'s phase is not charge-signed; `CAVNUM DPHASE` is relative to DYNAC's own crest | (writer/reader) | DYNAC V6R16: +0.866 MeV for a proton at −30°, −0.433 for an H⁻ at −30° and +0.866 at +150°; `CAVSC` 43.5 keV at 0° for both species |
| Synergia `rfcavity lag` | turns; `ff_rfcavity`: `E1 = E0 + volt·sin(2π·lag)` on the bunch reference — MAD-X's rule, so `lag = φ/2π + ¼`; no phase slip (the bunch's time is measured against its own accelerated reference) | (writer/reader) | Synergia 3 (2026-09-05): 866 025.4037844 eV at −30° (4e-15); a second gap at −30° gains the same with the plain lag |
| OPAL-T `RFCAVITY LAG` | rad; `CavityAutophaser` sets `φ = LAG + crest phase` for `E ∝ cos(ωt + φ)`, so `LAG = φ` (a late particle gains more at a negative `LAG`, the crest found for the actual charge) | (writer/reader) | source-derived (OPAL-X 078ff1e), no engine |

`energy_gain_eV(voltage_V, phase_rad)` is the one formula the walk uses.  Codes whose cavity model
needs a length (IMPACT-T's field profile, Ocelot's Rosenzweig–Serafini matrix) get a thin gap as the
short surrogate cavity of `thin_gap_surrogate_length(voltage_V, phase_rad, frequency_Hz, ref)` —
`0.4·sqrt(qVλ/(2π mc² βγ³ |sin φ|))` clamped to `[1 mm, βλ/2]`, the length at which the surrogate's
own RF defocusing balances the thin-gap formula — taken from the neighbouring drifts and recorded as
`THIN_GAP_AS_SHORT_CAVITY`.  The phase-slope
sign (a late particle at φs < 0 gains more) is not derivable from these formulas because the
time coordinate sign differs per engine; every writer is pinned by the cavity fingerprint in
§8 instead.

### 5.1 Thin-gap RF defocusing

TraceWin's `GAP` (and HELIX's `RFGap`) applies, after the energy kick and the adiabatic damping,
a round thin lens `Δx' = k·x`, `Δy' = k·y` with

```
k = −π · V_eff · sin φ / (m c² · (βγ)_out³ · λ)        [1/m]
```

(`thin_gap_defocusing` in `lattix.ir.rf`), φ the IR synchronous phase, `(βγ)_out` the reference
particle after the gap and λ the gap's RF wavelength (HELIX `rf_gap.py::kick_matrix`, lockstep with TraceWin to 1e-6).  No other code's thin
cavity has this term: a zero-length Bmad `lcavity`, an Elegant `RFCA`, MAD-X/xtrack cavities, ImpactX
`ShortRF`, IMPACT-Z's ideal cavity and FLAME all gave `R21 = R43 = 0` where HELIX gives 1.04 for an
80 kV gap at −85° on a 2.1 MeV proton (measured 2026-09-04).  Writers for targets with a
first-order matrix element (MAD-X `matrix`, Elegant `EMATRIX`, Bmad `taylor`, xtrack
`FirstOrderTaylorMap`, ImpactX `linear_map`, FLAME `tmatrix`, PALS `Taylor`) therefore follow every
thin cavity with a `Taylor` lens named `<cavity>_rfdefocus` carrying `k` on R21 and R43
(EQUIVALENT `THIN_GAP_RF_FOCUSING_AS_MATRIX`); with it Bmad, ImpactX and Elegant reproduce HELIX's
MEBT transverse maps to 1e-10 or better, and constant-p0 MAD-X/xtrack to 2e-2 (they lack the
damping).  MAD8 and IMPACT-Z have no matrix element: the kick is dropped and recorded as LOSSY
`THIN_CAVITY_NO_RF_FOCUSING`.  A re-read deck keeps its lenses (they are recognised by name), so
the round trip is a fixed point.  Thick cavities are left to each engine's own model.

## 6. Bends and pole faces

The IR bend stores arc `length`, `angle`, `e1`, `e2` (with rectangular flags), `edge_int1/2`
(fint), `hgap` and `tilt_ref` independently.

| Format | Rule | Measured evidence |
|---|---|---|
| TraceWin | `BEND θ[deg] ρ[mm] N R HV` + `EDGE β ρ gap K1 K2` on both sides; ρ > 0 with the sign in θ; **`β = sign(θ)·e`**, so a rectangular bend has β = \|θ\|/2 whatever the direction; `HV=1` with angle θ ≡ MAD tilt +π/2 with the same θ, and tilt −π/2 is written as `HV=1` with −θ; `gap = 2·hgap`, `K1 = fint`, `K2` default 2.80 | five-card decks through the TraceWin binary, HELIX and MAD-X agree to 1e-6 on all four sign combinations |
| MAD-X | `sbend l angle e1 e2 fint fintx hgap k1 tilt`; a rectangular source becomes `sbend` with `e += angle/2`; MAD-X folds the edges into the bend map | A1/A4 gates |
| MAD8 | `RBEND L` is the **arc** | BTL lockstep, 873 magnets to 4e-10 m |
| Elegant | `RBEN L` is the **chord** (elegant itself rewrites it to `SBEN, L = L·(θ/2)/sin(θ/2)` with `E1 = E2 = θ/2`); `FINT` default **0.5**; `FINT1/FINT2` only on `CSBEND`/`CSRCSBEND` | elegant 2026.3.0 `&save_lattice` |
| Bmad | `rbend` deck `l` is the chord; Bmad stores the arc and adds θ/2 to e1/e2 itself | Bmad 20260828 |
| ImpactX | `Sbend(ds, rc)` + `DipEdge` on each side | fodo.madx vs cpymad 1.8e-15 |
| FLAME | `sbend K` is normalized (`Kx = K + 1/ρ²`, `Ky = −K`); `roll` **is** MAD-X `tilt` | 9-digit agreement |
| xtrack | `rot_s_rad` is MAD-X `tilt`; `Bend.h` cannot be assigned (length + angle are passed) and `k0` reads back as `'from_h'` | xtrack 0.103.5 / 0.112.0 |
| PyORBIT3 | `BEND theta` with MAD-X's sign, `ea1/ea2` sector-referenced; no fringe integral, no tilt | sector bends of 1°–45° vs MAD-X 1e-11 |
| IMPACT-T | type 4 `By = Bρ_signed·θ/L` with the pole faces as lines in `rfdataN` (`k1 = tan e1`, `k4 = tan(|θ| − e2)`); the tracked dipole bends the whole bunch by the reference angle — no pole-face focusing, `R21 = R26 = 0` — so bend decks are report-only | fodo.madx bends 1e-2 vs cpymad (quads and drifts 1.2e-9) |
| Ocelot | `SBend(l, angle, e1, e2, gap = 2·hgap, fint, fintx, tilt, k1, k2)` with sector-referenced faces; `RBend` adds `angle/2` to both faces itself; survey follows MAD8 | fodo.madx (1 GeV e⁻) vs cpymad 1.8e-15; XFEL S2E `.lte` vs Ocelot's own converter 2.9e-12 |
| DYNAC | `BMAGNET ANGL RMO BAIM XN` with TRANSPORT's conventions: a positive angle bends towards −x, `PENT1/2` = e1/e2, `EK1` = fint with `APB` = hgap, `XN = −k1ρ²`; `BAIM = −|Bρ|/ρ` written for a negative species (the derived field would bend it the wrong way); `ZROT a` around it = MAD's tilt; a bend to the left is the right magnet inside `ZROT ±180` with `−e1/−e2` (MAD-X's own identity) | sector and wedge vs MAD-X 1e-5 (dump precision); `ZROT +90` bends down like `tilt = +π/2` |
| Synergia | `sbend(l, angle, e1, e2, fint, fintx, hgap, k1, k2, tilt)` with MAD-X's sector faces; an `rbend` is read with `angle/2` added to its faces; the field comes from the design momentum and there is no `k0`: after acceleration the bunch is under-bent by `p_design/p_local` (LOSSY `CONST_P0_BEND_UNDERBENT`) | fodo.madx vs cpymad 4.5e-8; `θ_eff = 0.0823` for a 0.1 rad bend after a 1 MV gap at 2.1 MeV |
| OPAL-T | `SBEND L ANGLE E1 E2 HGAP GAP FINT DESIGNENERGY` with MAD's faces, OPAL's default `1DPROFILE1` fringe profile, a required `DESIGNENERGY` [MeV kinetic]; a positive `ANGLE` bends towards −x (`compute3DLattice`) | source-derived, no engine |

Vertical bends are `tilt_ref = ±π/2`.  The PIP-II TraceWin export writes its two negative-angle
vertical bends with the edge sign reversed; the MAD8 anchor test pins exactly those two.

## 7. Kickers, multipoles, solenoids, apertures

* **Kicker**: `hkick`/`vkick` in rad with the `∫B·dl` view through `brho_signed`.  TraceWin
  `THIN_STEERING Bx By`; a thick kicker is written `DRIFT L/2 + THIN_STEERING + DRIFT L/2`
  and a zero-kick thick kicker becomes a plain `DRIFT` (the PIP-II export convention).  xtrack:
  `Multipole(knl=[−hkick], ksl=[+vkick])`; thick kickers need `isthick=True`.  FLAME
  `orbtrim theta_x → +x′`.
* **Multipole**: `BnL`/`BsL` are integrated lab fields.  Bmad `k{n}l`/`k{n}sl` ≡ MAD-X
  `knl`/`ksl` exactly, with `b_n = k_nl/n!`.  Bmad's `bmad_to_mad_sad_elegant -madx` writes
  `ksl = −n!·a_n` (upstream bug, skew sign reversed); lattix follows the engines, not the
  converter.  Thick sextupoles/octupoles go to TraceWin as thin kicks between drifts and to
  IMPACT-Z as split elements, both recorded.
* **Solenoid**: `Bsol_T` lab field; `ks = Bsol/brho_signed`, so an H⁻ solenoid rotates the
  opposite way to a proton one (invariant I-8, checked with Bmad species and MAD-X
  `charge = −1`).  IMPACT-Z solenoids with a negative `dx` would turn into cavities
  (`Param(5)` sentinel); the writer zeroes the offset and records it.
* **Apertures**: `ApertureP` half-sizes and shape; TraceWin `APERTURE dx dy type` (0
  rectangular, 1 elliptical exact; 2–6 kept in `native`), MAD-X `apertype/aperture`, Elegant
  `RCOL/ECOL`, Bmad `x1_limit … aperture_type`.  MAD-X drifts carry no aperture attribute
  (`LOSSY APERTURE_DROPPED`).

## 8. Longitudinal coordinates and the common basis

Engines are compared in one basis, `(x, px/p0, y, py/p0, z, δ)` with **z ahead-positive** and
the local `(β, γ, p0)` at every boundary.  The adapters' native pairs and the sign they need
(`lattix.oracles.basis._Z_SIGN`) are:

| Engine | Native longitudinal pair | z sign | p0 through RF |
|---|---|---|---|
| MAD-X (cpymad) | `(T, pt)` | +1 (*measured*: drift R56 > 0) | constant |
| xtrack | `(ζ, δ)` | +1 | constant |
| Bmad | `(z, pz)` | +1 | `lcavity` follows |
| Elegant | `(s = path length, δ)` | −1, and the adapter adds the `−L/γ²` drift term | `RFCA change_p0=1` follows |
| TraceWin | `(z, dp/p)` | +1 | follows |
| HELIX | `(Δφ deg, ΔW MeV)`, late-positive | −1 | follows |
| ImpactX | `(t, pt = −ΔE/p0c)`, t late-positive; the adapter returns `S·R·S` in MAD-X's `(T, pt)` | +1 after the transform | follows (`ShortRF`) |
| IMPACT-Z | `(x/Scxl, γβx, y/Scxl, γβy, ω·Δt, γ_ref − γ)`, `Scxl = c/2πf` | −1 | follows |
| FLAME | `(x mm, x′, y mm, y′, φ rad late-positive w.r.t. SampleFreq, ΔEk MeV/u)` | −1 | follows |
| PyORBIT3 | `(x m, x′, y m, y′, z m ahead-positive, dE GeV)` | +1 (`d[5] = 10⁹/(β²γ mc²)`) | follows |
| IMPACT-T | fixed-time dumps `(x m, γβx, y m, γβy, z m ahead-positive, γβz)`; the adapter drifts every particle to the reference plane and returns the common basis | +1 | follows |
| Ocelot | `(x, px, y, py, τ = c·Δt late-positive [m], ΔE/(p0 c))` — MAD-X's set, every map with γ = E/m_e (electron only) | −1 | follows (`Cavity` gains `v·cos(phi)`) |
| DYNAC | `(x cm, x′, y cm, y′, φ rad late-positive w.r.t. the master frequency, ΔW MeV)` from `WRBEAM` dumps (6 digits) | −1 (`d[4] = −βλ/2π`, `d[5] = 10⁶/(β²γmc²)`) | follows (the on-axis probe particle) |
| Synergia | `(x, xp = px/p_ref, y, yp, cdt, dpop = Δp/p_ref)` (`Bunch::cdt`, `Bunch::dpop`); a 1 m drift gives `R56 = −L/(βγ²)` | −1 (`d[4] = −β`, `d[5] = 1`) | follows (the bunch reference; strengths scaled by `p_design/p_bunch`) |

Each adapter is fingerprinted before use: a 1 m drift must give R56 = +0.9955387 for a
2.1 MeV proton and a thin 1 MV cavity at φs = −30° must gain 866 025.4 eV in the
p0-following engines (MAD-X and xtrack keep p0 and report 0).  The goldens live in
`tests/oracles/goldens/`; an engine upgrade that flips a convention fails there, not in a
physics test.  Cumulative maps are compared only at boundaries that both engines agree on
(thin kicks are attributed differently), block by block: transverse 4×4, dispersion column,
path-length row, R56, energy row.  P0-following maps are rescaled to constant p0
(`basis.rescale_to_constant_p0`) before the transverse comparison, and `Π det(2×2) =
p_in/p_out` is asserted separately.

## 9. Field maps to targets without maps

`lattix.ir.fieldmap.integrate_map` integrates the reference particle through the map
(midpoint sub-steps, β updated after every step, `SET_SYNC_PHASE` fixed-point calibration)
and returns the self-consistent complex voltage `v_c` and the synchronous phase, so that
`dE_ref = v_c · cos φ_sync` identically.  Writers without a field-map element emit that pair as
a thin cavity (`LOSSY FM_TO_CAVITY`), which is why ±90° bunchers convert cleanly.  Static
solenoid and quadrupole maps degrade to hard-edge elements that preserve `∫B` and `∫B²`:
`L_eff = (∫B)²/∫B²`, `B_eff = ∫B²/∫B` (`FM_SOL_HARDEDGE`, `FM_QUAD_HARDEDGE`).  On the PIP-II
MEBT+HWR deck every map agrees with HELIX to ≤ 2e-13 relative on the cavities.

## 10. Names and overlaps

Names are sanitized per format and uniquified (`_2`, `_3`, …); the original name and type are
kept in `provenance` and in an adjacent `! lattix: name="…" type="…"` tag that readers parse
back.  Known limits: MAD8 16 characters, Bmad 40, ImpactX 64.  A MAD-X sequence cannot hold
negative drifts (the BTL contains one, −0.204288 m): the MAD-X writer orders entries by
position, shortens the preceding drift and keeps every element where the source put it;
genuine thick-element collisions are shifted and recorded (`LOSSY OVERLAP_SHIFTED`).
Zero-length sequences fall back to line mode (`ZERO_LENGTH_LINE_MODE`).

## 11. Survey frames

`lattix.ir.frames.frame_survey` walks the expanded line with a full frame — a position `V` and
a 3×3 orientation `W` whose columns are the local x, y and s axes in global coordinates — and
returns, for every placed element, the frame at its entrance, its centre and its exit, and the
body frame after its misalignment.  `survey()` is the `(X, Y, Z, theta)` view of the exit frames.

* **Global frame.**  The line starts at the origin with the beam along +Z, X horizontal and Y up
  (right-handed).  A positive horizontal bend turns the beam toward −X and its azimuth `theta`
  decreases: MAD-X `survey`'s frame.  The angles are MAD-X's, `W = Θ(theta)·Φ(phi)·Ψ(psi)` with
  `theta = atan2(W₀₂, W₂₂)`, `phi = asin(W₁₂)`, `psi = atan2(W₁₀, W₁₁)`, and `theta` is kept
  continuous along the line (MAD-X `proxim`: a ring ends at −2π).  `site_frame(x0, y0, z0,
  theta0, phi0, psi0)` is the start pose of MAD-X `SURVEY`.
* **Bend.**  The frame moves along the arc in the bend's plane, `T(tilt_ref)` conjugating the
  rotation, so a `tilt_ref = +π/2` bend with a positive angle bends toward −Y (`phi = −angle`);
  the centre frame is the arc midpoint with the tangent turned by half the angle.  A reversed
  placement negates the angle.
* **Patch.**  The frame jumps by the offsets in the local frame, then rotates by
  `Ry(y_rot)·Rx(x_rot)·Rs(tilt)` — yaw, pitch, roll in that order; a positive `x_rot` tips the
  beam toward −Y.  The centre of a patch is its entrance.
* **Body frame.**  The centre frame rolled by the element's own field roll — a skew quadrupole
  is a rolled quadrupole (`multipole.tilt[1]`; `[2]` and `[3]` for sextupoles and octupoles) —
  then moved by `Element.shift` about the element centre, in the centre frame: translate by the
  offsets, then rotate by `Ry(y_rot)·Rx(x_rot)·Rs(tilt)`.  *Measured* against Bmad's floor
  positions (Tao `ele_floor`, `Reference` at the exit and the centre, `Actual` at the centre)
  on `lattix/misaligned.bmad`: positions to 1e-9 m and all three angles to 1e-10 for an offset
  and pitched quadrupole, a rolled bend, a skew quadrupole and a combined patch
  (`tests/oracles/test_frame_survey.py`).  The reference orbit never moves because of a shift;
  only the body frame does.
* **Superposition children** (`expand_children=True`) are placed along the container's straight
  axis at `s_in + z_offset` and carry the container's index.
* **MAD-X cards** (measured with cpymad 5.09.03 on 2026-09-11, `tests/oracles/test_frame_survey.py`):
  `yrotation, angle=a` adds `a` to theta (`Patch.y_rot = a`), `xrotation, angle=a` subtracts `a`
  from phi (`Patch.x_rot = a`), `srotation, angle=a` adds `a` to psi (`Patch.tilt = a`), and
  `translation, dx, dy, ds` moves in the local frame (`Patch` offsets).  MAD-X's own `survey`
  ignores `changeref`, so lattix keeps it unsupported rather than guessing its order of operations.

`survey_table` and `survey_csv` flatten the frames into rows (`X, Y, Z, theta, phi, psi` per
frame, metres and radians, 12 significant digits).

### Field-map phases (TraceWin `FIELD_MAP`, measured 2026-09-05)

| Card | Meaning in the IR | Integration |
|---|---|---|
| `SET_SYNC_PHASE` + `FIELD_MAP … φ …` | `rf.phase_is_sync = True`, φ is the synchronous phase | the RF phase is calibrated so the integrated gain has that synchronous phase |
| `FIELD_MAP … φ … P=0` | relative phase: the RF phase when the reference particle enters the map | `Ez(z)·cos(ω t(z) + φ)` with `t = 0` at the entrance — no bunch-clock term (LightWin/TraceWin; HELIX adds the running phase: a HELIX bug) |
| `FIELD_MAP … φ … P=1` | treated as synchronous (HELIX's reading; no reference deck) | as the first row |

