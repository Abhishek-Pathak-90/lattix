# Oracle engines — verified facts (Phase 0, 2026-09-03)

Everything below was **measured on this machine** (macOS arm64, Rosetta present), not
recalled.  Re-run `lattix fingerprint` / `pytest tests/oracles` after any engine upgrade;
`tests/oracles/goldens/fingerprints.json` pins the native-basis signs.

## Engines

| Engine | How it runs here | Version | Basis (native) | p0 through RF | Notes |
|---|---|---|---|---|---|
| MAD-X | `cpymad` in the base env and in env `lattix` | 5.09.03 (2024-04-25) via cpymad 1.19.0 | (x, px, y, py, T, pt) | constant | `twiss, sectormap` gives per-element maps; **T is ahead-positive** (drift R56 = +L/(β²γ²) in native units → +L/γ² in the common basis after z = +β·T, δ = pt/β). Table names carry `:N` occurrence suffixes. |
| HELIX | in-process, `HELIX_ROOT` (default: local `HELIX_v3` checkout) | linac_gen 1.9.1 | (x mm, x′ mrad, y mm, y′ mrad, Δφ deg, ΔW MeV) | follows | Δφ is *late-positive* (z = −βλ/360·Δφ); ΔW → δ = ΔW/(β²γmc²). **HELIX dipoles omit the path-length row (R51, R52) and the dispersive part of R56** — transverse 4×4 and dispersion agree with MAD-X to 1e-7 / 7e-9 on `examples/madx/fodo.madx`, `path` block differs by 0.7. Worth filing upstream. Also: HELIX's matrix path (`compute_transfer_matrix`, mirrored by the oracle) ignores `SET_BEAM_E0_P0`, and `SET_SYNC_PHASE` binds only to the next FIELD_MAP/NCELLS (never a thin GAP: `tracewin_parser.py:701-711`). |
| TraceWin | `TRACEWIN_EXE` (default `TraceWin.app/Contents/MacOS/TraceWin`, x86_64 under Rosetta), batch mode `TraceWin project.ini hide dat_file= path_cal= energy1= freq1= current1= nbr_part1=` with LightWin's `generic_project.ini` (MIT) as the project file | licensed **trial** build: *"Number of element limited to 20"* | `Transfer_matrix1.dat`: (x, x′, y, y′, z, dp/p), lengths m, **z ahead-positive** (R56 = +L/γ² measured), dp/p relative to the *exit* p0 (cavity R66 = p_in/p_out) | follows | Rejects HELIX's `TITLE` card ("Unknown element or command") — HELIX `.dat` dialect is not portable as-is. Outputs also `tracewin.out` (`gama-1` per element exit → reference energy), `.beta`, `.par`, `Density_Env.dat`. 20-element cap → fingerprints and small decks only; PIP-II decks rely on the CEA reference exports in `HELIX_v3/Tracewin_code/`. Stalled once for 600 s in a full-suite run (passes in 16 s alone) → adapter timeout 120 s. |
| Bmad / Tao | conda env `bmad` (`pytao` 1.2.4, bmad 20260828.0, py3.13/numpy2) via an out-of-process worker | 20260828.0 | (x, px, y, py, z, pz) | follows (`lcavity`) | Converters shipped: `bmad_to_mad_sad_elegant`, `madx_to_bmad.py`, `elegant_to_bmad.py`, `sad_to_bmad`. `bmad_to_mad_sad_elegant` needs positive initial Twiss (`beginning[beta_a]`) or `-force`. |
| elegant | conda env `lattix` (`conda-forge elegant 2026.3.0`, osx-arm64) + `pysdds` 0.6 (text fallback via `sdds2stream`) | 2026.3.0 (2026-07-02) | (x, x′, y, y′, s, δ) with **s = path length**: its matrices carry no velocity-bunching term (drift R56 = 0); the adapter adds −L/γ² per element so maps land in the arrival-time common basis | follows with `change_p0=1` | **RFCA phase convention follows the charge sign relative to the electron**: negative species crest at +90°, positive species (protons) at −90° — `phase = 60` *decelerates* protons, `phase = 240` gives +V·cos30°. Its RFCA *matrix* phase slip is ultra-relativistic (R65 = β·true value; tracking is physical). Constants are CODATA-86 (m_e = 0.51099906 MeV; built-in `proton` = 938.2866 MeV, 1.5e-5 off) → adapter uses `change_particle name=custom, mass_ratio, charge_ratio`. conda-forge build ships no `defns.rpn` (adapter embeds one and passes `-rpnDefns=<abs path>`). |
| xtrack | base env 0.103.5; env `lattix` 0.112.0 | — | (x, px, y, py, ζ, δ) | constant | Native Lark MAD-X parser rejects some MAD-X (e.g. `sequence, l=…, refer=centre` header in HELIX's fodo.madx) → load through cpymad `Line.from_madx_sequence`. |
| LightWin | env `lightwin` (python 3.12, LightWin 0.16.5, MIT) | `lattix/oracles/lightwin.py` + `lightwin_worker.py` | (x, x', y, y', z [m], dp/p) — TraceWin's, SI | follows the field maps | Envelope3D only: DRIFT, QUAD, SOLENOID, BEND, FIELD_MAP (1-D); EDGE/THIN_STEERING/APERTURE/DIAG_* propagated as drifts, GAP/NCELLS/DTL_CEL skipped — both audited in `meta` and report-only in the battery. |
| Cheetah | env `cheetah` (torch 2.14 CPU, cheetah-accelerator 0.8.4, GPL-3) | `lattix/oracles/cheetah.py` + `cheetah_worker.py` | (x, px, y, py, τ = cΔt late-positive [m], ΔE/(p0 c)) | follows the cavities | 7×7 first-order maps per element; cavity phase runs the other way (`phase = −φ`), k1/k charge-blind (signed rigidity in the writer); zero-length cavity is inf (1 µm substituted). |
| IMPACT-T | conda-forge `impact-t` 3.1.5 (`ImpactTexe`, BSD) in env `lattix` | 3.1.5 | fixed-time `(x, γβx, y, γβy, z, γβz)` dumps drifted to the reference plane → the common basis | follows | `lattix/oracles/impactt.py`: `-2` dumps and `-4` step changes at every boundary, each element its own `dt = L/(N βc)` so the reference lands on the boundaries; 13-particle probes from `partcl.data` (`flagdist 16`); `R_elem = J_i·J_{i−1}⁻¹`; ≤ 90 controls per run (chunked, `theta0` shifted by `360·f·t`); dipoles re-base the frame one step past the face; **the dipole model has no pole-face focusing (report only)**. |
| PyORBIT3 | env `pyorbit` (python 3.10, meson build of PyORBIT3 `22b45fa` 2026-05-14 with `USE_MPI=none`, MIT; `libfftw3` preloaded on macOS) | `lattix/oracles/pyorbit.py` + `pyorbit_worker.py` | (x [m], x′, y [m], y′, z [m] ahead-positive, dE [GeV]) | follows the gaps | `LinacTrMatricesController` maps at every node entrance, cumulated from the first node (per-node map `R[k+1]·R[k]⁻¹`, one extra node at the exit); 13-particle symmetric probes tracked at two amplitudes and Richardson-extrapolated; one `<Cavity>` per gap. |
| ImpactX / IMPACT-Z | env `lattix` (`impactx` 26.08, `impact-z` 2.7.7, both osx-arm64 builds) | — | Phase 3 | follows | `ImpactZexe` on PATH in the env; `impactx` importable. |
| SciBmad | `julia` (juliaup 1.10) with `SciBmad` 0.5.2 (Beamlines.jl + BeamTracking.jl) through `lattix/oracles/scibmad_worker.jl`; `LATTIX_JULIA` or PATH | 0.5.2 (2026-09-05) | (x, px, y, py, z, pz), z ahead-positive (drift R56 = +L/γ² measured) | constant (one reference momentum per `Beamline`; nesting and `Patch(dE_ref)` past the first element are refused) | `RFCavity` gain is **−V·cos(phi0)** with `phi0` in radians for protons and electrons alike (writer negates the voltage: `GAIN_SIGN`); a zero-length cavity and `edge1_int/edge2_int` cannot be tracked (worker substitutes 1 µm / 0 and says so); `g_ref` alone is a curved frame — the dipole field is `Kn0`; `LineElement(transport_map=f)` with `f(v, q, p=nothing)` works as a thin lens; `using Beamlines` fails in an environment that only has `SciBmad` (worker strips `using` lines). `Species("#1H-")` is H⁻ (m_p + 2 m_e); `Species("H-")` is the isotope-averaged anion. Startup ~13 s, a run ~8 s. |

Not installable as-is: `lightwin` (needs Python ≥ 3.12; the TraceWin runner pattern was
reused instead), `flame-code` (Phase 3, pip).

## Measured longitudinal fingerprints (proton, 2.1 MeV, 162.5 MHz; drift 1 m; thin cavity 1 MV at φs = −30°)

| Engine | drift R56 native | R56 common (expect +0.9955387) | cavity R65 common | reference gain (eV) |
|---|---|---|---|---|
| madx | +2.2315e+02 (T, pt) | +0.9955387 | −6.077 | 0 (constant p0) |
| xtrack | +0.9955387 (ζ, δ) | +0.9955387 (1e-10) | −5.117 | 0 (constant p0; exact energy→δ, so its cavity slope differs from MAD-X's linear pt map at 2.1 MeV) |
| bmad | +0.9955387 (z, pz) | +0.9955387 (6e-12) | −4.305 (lcavity, l = 1 mm) | 866 025.4 (`lcavity` follows p0; det(2×2) = p_in/p_out) |
| helix | −6.9326e+02 (deg, MeV) | +0.9955387 | −4.305 | 866 025.4 |
| tracewin | +0.9955387 (z, dp/p) | +0.9955387 (3e-8, 7-digit file) | −4.273 | 866 025.1 |
| elegant | 0 (path length; −0.9955387 after the adapter's −L/γ² term) | +0.9955387 | −0.288 (matrix; ≈ β·4.30 — ultra-relativistic phase slip) | 866 025.4 (`change_p0=1`, proton phase = φs − 90°) |

Why the cavity R65 differs between constant-p0 and p0-following engines at this energy:
δ = ΔW/(β²γmc²) is normalised with the *exit* β²γ in p0-following codes and the *entry*
β²γ in MAD-X; the ratio (β_in²γ_in)/(β_out²γ_out) = 0.708 here (41 % energy gain in one
gap) reproduces HELIX's −4.305 from MAD-X's −6.077 exactly.  Rescaling momentum rows by
p_out/p_in (`basis.rescale_to_constant_p0`) makes transverse blocks comparable but the
longitudinal cavity row is a genuinely different linearisation at low energy — compare
accelerating linacs against Elegant/Bmad/TraceWin/HELIX, not MAD-X/xtrack (PLAN D3/D4).
HELIX vs TraceWin differ by 0.7 % on R65 (gap model detail; Equivalent tier).

## Adapter details learned the hard way

* **xtrack** finite differences: `Particles` stores δ through an energy round-trip that loses
  ~1e-14 absolute, so `h = 1e-7` gives R66 = 0.99999988; the adapter uses `h = 1e-5` with the
  read-back perturbation as denominator (errors ≲ 1e-10).  `twiss`/`survey` rows are element
  entrances plus `_end_point` (exit(i) = row i+1).  `from_madx_sequence`/`xt.load` do not set
  `particle_ref`.
* **Bmad**: `lat_list("*","ele.l")` (lower-case); `ele_twiss` is empty unless `beginning[…]`
  is set; `mat6` changes once `particle_start` is set (read matrices before probing); the
  `.digested` cache is keyed on a 1-second mtime (fresh temp dir per run); the worker runs
  as `python -I bmad_worker.py` so `lattix/oracles/pytao.py` cannot shadow the real `pytao`;
  H⁻ is `#1H-`.  **Bmad's proton mass (938 272 089.43 eV) differs from CODATA-2018
  (938 272 088.16 eV) by 1.35e-9** — enough to break a 1e-9 fingerprint, so the adapter reports
  Bmad's own `mass_of()`.
* **MAD-X**: `sequence.beam` is only attached at `USE`; table names carry `:N`; `sectormap`
  writes a file named `sectormap` into the cwd (adapters `chdir` to a temp dir).

## Phase-0 gate (2026-09-03): `examples/madx/fodo.madx` (800 MeV proton, FODO + 2 sbends, 6.6 m)

| pair | shared boundaries | max |ΔR_cum| | transverse 4×4 | dispersion | path row |
|---|---|---|---|---|---|
| madx vs xtrack | 8 | 3.7e-9 | 2.4e-9 | 6.1e-10 | 5.4e-10 |
| madx vs bmad | 8 | 2.4e-9 | 2.8e-14 | 4.9e-15 | 4.4e-15 |
| xtrack vs bmad | 8 | 6.1e-9 | 2.4e-9 | 6.1e-10 | 5.4e-10 |
| madx vs helix | 4 | 7.1e-1 | 1.3e-7 | 7.4e-9 | 7.1e-1 (HELIX dipole gap) |

`lattix validate --deck madx=tests/data/public/helix/fodo.madx --deck bmad=tests/data/public/helix/fodo.bmad --oracles madx,xtrack,bmad,helix --ke 800e6 --freq 352.21e6`

## Comparison rules that fell out of the measurements

* Compare cumulative maps only at **unambiguous boundaries**: engines attribute thin kicks
  differently (MAD-X folds bend edges into the bend map; HELIX/TraceWin emit EDGE rows),
  so a boundary whose downstream-most row is a thin non-identity kick in either engine is
  skipped (`compare.shared_boundaries`).
* Report per-block metrics (transverse 4×4, dispersion column, path-length row, R56,
  energy row) — a single max-abs number hides which physics disagrees.

## Phase 1 measurements (2026-09-03)

* **MAD-X twiss carries a cavity's energy gain in the orbit `pt`** (50 MV on a 100 MeV proton →
  pt = 0.1125) and linearises downstream magnets about it, although p0 stays constant. So
  `energy_mode="constant"` is what composes with MAD-X when the cavities are in the deck;
  `"local"` is right when the energy change is *not* modelled by MAD-X (ReferenceChange,
  FM_TO_DRIFT, extracted sections).
* MAD-X `rbend` expanded `l` is the chord; the node length is the arc `L·(θ/2)/sin(θ/2)`
  (`rbarc=true`). `(k1, k1s)` ≡ `hypot(k1,k1s)` rotated by `−atan2(k1s,k1)/2`. Element names
  fail above 41 characters. `apertype` is illegal on `drift`, `matrix`, `translation`.
* HELIX writes `.dat` numbers with `%.10g`; that alone costs 4e-8 on a converted k1 (gate A1),
  so lattix writes `%.15g`.
* `fodo.madx` declares `energy = 0.938272 + 0.800 GeV` with a rounded proton mass: its kinetic
  energy is 799.99991 MeV, not 800 MeV — a 1e-7 rigidity trap for anyone re-reading a converted
  deck at "800 MeV".
* The PIP-II MEBT deck has four buncher FIELD_MAPs (ke = 0.068/0.045 at −90°); until Phase 3's
  `FM_TO_CAVITY`, MAD-X sees drifts there and the comparison stops at the first map.
* HELIX↔IR↔HELIX is bit-exact on every deck tried (mebt 427, mebt+hwr 483 incl. 24 real field
  maps); HELIX's particle masses differ from CODATA-2018 at 1e-8, visible in ∫B·dl→kick.

## Phase 2 measurements (2026-09-03)

* **Elegant**: `RBEN L` is the CHORD (`RBEN, L=1, ANGLE=0.1` → elegant's own `SBEN, L=1.0004168`,
  MAD-X `rbarc` semantics) — cheetah, ocelot and Bmad's `elegant_to_bmad.py` all pass L through
  unchanged (error L·θ²/24). `FINT` default is **0.5** (not 0.45 as PLAN §4.3 said); `FINT1/FINT2`
  only on CSBEND/CSRCSBEND. `RFCA`: FREQ default 500 MHz, CHANGE_P0 default 0. `#` is a comment
  character, `;` is not a separator, a trailing `,` is not a continuation (only `&`). Per-type
  misalignment attributes differ (DRIF/KICKER/WATCH accept none). KQUAD's linear matrix is
  bit-identical to QUAD's. Charge-sign phase rule pinned by tracking: proton `PHASE=240`, H⁻
  `PHASE=60` both give +V·cos30°. A2 elegant leg: T4x4 2.8e-10.
* **Bmad**: `lcavity` with `l = 0` (or any l < 1 mm) and the default `standing_wave` is FATAL
  ("infinite pondermotive kick") → thin cavities are written `l = 0, cavity_type = traveling_wave`,
  which reproduces ΔE = V·cos(2π·phi0) exactly. **A thin Bmad cavity has no transverse RF kick**
  (R21 = R43 = 0 where HELIX/TraceWin give 0.76/0.65/0.56 for 300 kV gaps at −30°). `rbend` deck
  `l` is the chord; Bmad stores the arc and adds angle/2 to e1/e2 itself. `k{n}l`/`k{n}sl` ≡ MAD-X
  knl/ksl exactly; `b_n = k_nl/n!`. Name limit 40. Default particle is positron. `k0l` needs
  `k0l_status = straight_reference`. **Upstream bug**: `bmad_to_mad_sad_elegant -madx` writes
  `ksl = −n!·a_n`, reversing skew multipoles (both engines agree the sign should be +). A2 Bmad leg:
  T4x4 2.8e-14; A3 Bmad leg on the real MEBT: 9.9e-14.
* **MAD8**: BTL lockstep MAD8 vs TraceWin export (873 magnets): lengths 4e-10 m, gradients
  1.4e-10, angles 7e-12 — passes where HELIX's own anchor is xfail (HELIX's Dipole lacks a
  reference tilt). The only systematic difference is `tilt_ref` sign on 4 vertical bends (the
  TraceWin export keeps only `hv=1`). `BAL2025V0213.FLAT` parses (HELIX stops at SQRT) and has a
  deck bug: `BRHO := P0/C*1.0E11` is 1000× too large (never referenced). The BTL deck contains a
  genuine negative drift (`DBV3NT = −0.204288 m`): MAD-X aborts on it in both sequence and line
  mode; the MAD-X writer now resolves overlaps (shift the overlapping element downstream, shorten
  the next drift, LOSSY `OVERLAP_SHIFTED`). RBEND `L` is the arc in MAD8.
* **PALS**: `pals-schema` 0.3.0 loads every document lattix writes but is an older draft (untagged
  union → unknown kinds silently become placeholders; PyYAML reads the standard's own `1.0e9` as a
  string). ImpactX 26.08 `pals_to_impactx` needs the `PALS:` root, a final `use:`, no sublines or
  `repeat:`, only Drift+Quadrupole, and raises on `BeginningEle` (→ writer flavor `flat`). Bmad
  `write pals` emits `kind: Bend`, `g_ref` + a redundant `Kn0`, always `Kn1` (never `Bn1`), no
  `use:`, signed Fortran exponents. Field names in the standard text: `angle_ref`, `edge1_int` =
  fint·hgap product, `num_cells`, `z_rot`, `x_min/x_max`, `dtime_ref`.
* **TraceWin export convention**: the PIP-II BTL export has no THIN_STEERING cards — zero-kick
  correctors are plain drifts; `LATTICE n` counts exclude DIAG_*, APERTURE and THIN_STEERING, so
  the TraceWin writer recounts every `LATTICE n` from the cards it emits.

## Vertical bends and edge angles in TraceWin (measured with the licensed binary, 2026-09-03)

Five-card decks (EDGE/BEND/EDGE, ρ = 25.24 m, |θ| = 2.3835°, |β| = 1.1918°, HV=1) through
TraceWin, HELIX and MAD-X (`sbend` with `e1 = e2 = angle/2`, tilt −π/2):

| cards | R21 | R43 | R36 | matches MAD-X |
|---|---|---|---|---|
| θ = −2.38°, β = −1.19° | +0.00165 | −0.00329 | −0.0218 | **no** (edge focusing reversed) |
| θ = −2.38°, β = +1.19° | −0.00165 | 0 | −0.0218 | MAD-X angle +0.0416, tilt −π/2 |
| θ = +2.38°, β = +1.19° | −0.00165 | 0 | +0.0218 | MAD-X angle −0.0416, tilt −π/2 (≡ +0.0416, +π/2) |
| θ = +2.38°, β = −1.19° | +0.00165 | −0.00329 | +0.0218 | no |

Rules encoded in `formats/tracewin`: `HV=1` with angle θ ≡ MAD tilt +π/2 with the same θ (a
tilt of −π/2 is written as HV=1 with −θ); the EDGE angle carries the sign of the bend angle,
**β = sign(θ)·e** (HELIX's MAD-X importer rule), so a rectangular bend has β = |θ|/2 whatever the
bend direction.  HELIX and TraceWin agree to 1e-6 on all four combinations.

**Finding for the PIP-II export**: `btl_2025v0703.dat` writes the two negative-angle vertical
bends (BVDD, ORB1) with β = e, i.e. the reversed edge focusing (R21 = +0.00165 instead of
−0.00165, R43 = −0.0033 instead of 0); the two positive ones (BVDU, ORB2) are right.  The MAD8
lockstep anchor pins exactly those two as edge-sign flips.

## MAD8 negative drifts in MAD-X

The BTL's `DBV3NT = −0.204288 m` makes the following corrector start inside the preceding
drift.  MAD-X aborts on any overlap; the writer orders sequence entries by position, shortens
the *preceding* drift (a drift is only a gap in a MAD-X sequence) and keeps every element where
the source put it — gate A4 then agrees HELIX vs MAD-X to 1e-7 through the whole 308 m line.
Only genuine thick-element collisions are shifted (LOSSY `OVERLAP_SHIFTED`).

## Phase 3 engines (measured 2026-09-03)

| Engine | How it runs | Native basis | p0 through RF | Measured facts |
|---|---|---|---|---|
| **ImpactX 26.08** | env `lattix`, in-process or `python -I lattix/oracles/impactx.py` worker; per-element maps by 13-probe central differences re-seeded at every element (`el.push` applies ONE slice) | (x, px, y, py, t, pt) with **t late-positive and pt = −ΔE/p0c** — the adapter returns S·R·S in MAD-X's (T, pt); the two sign flips cancel in the longitudinal block (drift R56 +223.148 native, +0.9955387 common, identical to MAD-X) but not in the dispersion column | follows (`ShortRF` pushes the reference; cavity R65 −4.3046, gain 866 025.404 eV) | `ShortRF.phase` is the synchronous phase in degrees, 0 = crest (IR convention, no shift); `V` is dimensionless = voltage_V/mass_eV; `load_inputs_file` segfaults after the first call (ParmParse state) → the oracle builds through the Python API; Marker has no `inputs` type. fodo.madx vs cpymad 1.8e-15; vs ImpactX's own MAD-X loader 0.0. |
| **IMPACT-Z 2.7.7** | env `lattix` `ImpactZexe`; per-element maps from a 13-particle `particle.in` probe (`flagdist=19`) with zero-length `-2` dumps at every boundary; `fort.18` gives the reference energy | (x/Scxl, γβx, y/Scxl, γβy, ω·Δt [rad, late-positive], γ_ref−γ) with Scxl = c/(2πf); `basis.py` corrected to `d = (Scxl, 1/βγ, Scxl, 1/βγ, −β·Scxl, −1/β²γ)` | follows | **any element other than types 0/1/4 with a negative `Param(5)` is an ideal RF cavity** (`BeamBunch.f90:323-437`): gradient V/m, synchronous phase deg, gain E0·L·cos φs with no charge factor — the IR rule verbatim, used as `rf_model="ideal"` (thin cavity gain to 1.2e-16, R65 −4.30458); a solenoid with negative dx becomes a cavity (writer zeroes it, LOSSY); type 5 multipoles produce NaN under `flagmap=1`; misalignments need header `flagerr=1`; no Twiss/dispersion/survey outputs. fodo.madx vs cpymad: T4x4 6e-14. |
| **FLAME 1.9.2** (built from the clone against Homebrew Boost; PyPI ships only manylinux x86_64 wheels) | env `lattix`, in-process or `-I` worker; `Machine.propagate` per-element `transmat` (7×7×n_charge_states, state index LAST) | (x mm, x′ rad, y mm, y′ rad, φ rad late-positive w.r.t. **SampleFreq** (default 80.5 MHz), ΔEk MeV/u); `_Z_SIGN = −1` confirmed (drift 4.4e-16) | follows (per nucleon: `IonEk`, `IonEs`, charge states) | `sbend K` is normalized (1/m², `Kx = K + 1/ρ²`, `Ky = −K`); `roll` **is** MAD-X `tilt` (9 digits); `orbtrim theta_x → +x′`; `rfcavity phi` is a synchronous phase when `syncflag` ≥ 1 but the gain is a tabulated TTF polynomial, not V·cos φ (3.2 % off at −35°) → the reader records `FLAME_CAVTYPE_VOLTAGE_UNKNOWN`; `aper` is read nowhere in FLAME's source. fodo.madx vs cpymad 3.6e-15; LS1/LS1FS1/ALL_lattice read→write→FLAME bit-exact. `FrontEnd.lat` ships an undefined `v` (FLAME rejects it). |
| **xtrack** (0.103.5 base, 0.112.0 env) | `lattix.formats.xtrack.to_line`/`from_line` + `Line` JSON | (x, px, y, py, ζ, δ) | constant | `rot_s_rad` is MAD-X `tilt` (2.2e-16); kicker = `Multipole(knl=[−hkick], ksl=[+vkick])`; `Cavity.phase = φ + π/2` round-trips on both releases (lag deprecated in 0.112); `Bend.h` cannot be assigned (pass length+angle) and `k0` reads back as the string `'from_h'`; thick kickers need `isthick=True` or 7.5 m of the PSB vanish; PSB `psb.seq` through lattix vs `Line.from_madx_sequence`: every block exactly 0.0 over 304 boundaries. |

Environment note: a helper's `conda install -n lattix boost bison` (for the FLAME build) downgraded elegant/impactx/hdf5 and broke 28 tests; the env was repaired by removing those packages and pinning `elegant=2026.3.0 impactx=26.08 impact-z=2.7.7 gsl=2.7` (the elegant binary links `libgsl.25`). Never `conda install` into `lattix` without pinning those four.

## Field maps (Phase 3, 2026-09-03)

`lattix.ir.fieldmap.integrate_map` ports HELIX `field_map.py::advance_ref` in SI (midpoint sub-steps
over the card length, β updated after every step, SET_SYNC_PHASE fixed-point calibration).  On
`mebt+hwr.dat` every one of the 20 RF/static maps agrees with HELIX to ≤ 2e-13 relative on the
cavities (the −90° bunchers gain milli-eV, where the relative figure is cancellation noise: worst
absolute 3.2e-6 eV); `mebt+hwr+ssr1+ssr2.dat` (100 maps) ends at 169.820410859731 MeV vs HELIX
…734.6.  The summary carries `v_c` (self-consistent complex voltage magnitude) and its synchronous
phase so that `dE_ref = v_c·cos φ_sync` identically — that pair is what writers emit
(`FM_TO_CAVITY`), which is why ±90° bunchers convert where `dE/cos φ` would blow up.  Static
solenoid/quad maps degrade to hard-edge elements preserving ∫B and ∫B² (`L_eff = (∫B)²/∫B²`,
`B_eff = ∫B²/∫B`).  With the HWR cavities as real `rfcavity`s, MAD-X's own `twiss` fails on the open
accelerating line ("error with deltap") — the constant-p0 limit of PLAN §8, now reachable; the
MAD-X gate therefore checks loading and reporting, the physics gate runs on Bmad/elegant.

## Cross-format battery findings (2026-09-04)

* **Fringe-integral model, MAD-X vs Bmad.** With `fint·hgap ≠ 0` the two engines disagree on a
  bend's transverse map at O(ψ²) of the fringe correction: the ELENA main bend (60°, e1 = e2 =
  0.287 rad, fint 0.424, hgap 0.038 m) gives `max|ΔR̂|` = 8.6e-4 between cpymad and Tao, the same
  bend with `hgap = 0` agrees to 2e-15, and the PSB bends (fint 0.5, hgap 0.035, e = 0.098) add
  1.5e-6 each.  The parameters translate exactly; the engines model the correction differently,
  so the battery rates lattices with fringe-integral bends as Equivalent tier between engines.
* **ImpactX inputs keys are case sensitive**: the inputs file reads `k_normal`/`k_skew`
  (`InitElement.cpp`) while the Python API takes `K_normal`/`K_skew`; the writer emits the
  lower-case keys and the oracle maps them when it rebuilds elements through the API.
* **FLAME rejects a second definition of a global** (`IonEs already defined`): the adapter
  appends beam globals only when the deck does not define them.
* Custom ion species (FLAME `ion_A238_Q33`) are not yet expressible in `BeamSpec`, so the engine
  comparison skips such decks (the IR round trip still runs); Phase 5 backlog.
* **HELIX bend maps have no path-length coupling**: on the FODO's `b1` HELIX gives
  `R51 = R52 = 0` and `R56 = 0.29135` where MAD-X gives `−0.10008, −0.04996, 0.28969`; the
  transverse and dispersion blocks agree to 5e-15.  The battery excludes the `path` and `R56`
  blocks from HELIX comparisons of lattices with bends.
* **Thin-gap RF defocusing** (TraceWin/HELIX) is absent from every other code's thin cavity
  (T4x4 differed by 4.2 on the MEBT line): see conventions §5.1 for the lens lattix now writes;
  after it Bmad 2.2e-13, ImpactX 2.4e-13, Elegant 1.3e-10, MAD-X 1.6e-2 and xtrack 1.5e-2
  (constant p0: no damping) against HELIX on `mebt_line.dat`.
* **Elegant `EMATRIX` defaults every `R_ij` to 0** (a lens written with `R21` alone zeroes x and y):
  the writer emits the full first-order matrix.  **FLAME `tmatrix` is in (mm, rad)**: a map in
  metres is rescaled (`R21/1000`, `1000·R12`).
* **Elegant `KQUAD` rejects `K2`**; a quadrupole with higher-order components loses them in every
  target but TraceWin and PALS (LOSSY `QUAD_HIGHER_ORDER_DROPPED`).
* **IMPACT-Z**: type-5 multipoles under the linear-map integrator give NaN maps; the writer now
  selects the Lorentz integrator (`flagmap = 2`) whenever a sextupole or octupole is written.

### Second round (2026-09-04): constant-p0 engines, TraceWin's bend sign, derived sources

* **Constant-p0 engines carry the RF gain in their orbit.**  Measured with a 1 MV on-crest gap at
  2 MeV followed by a quad written with `k1 = 5 /m²` at the start rigidity: MAD-X `twiss`
  (`sectormap`) and a tracked xtrack particle both see `k_eff = 5.00` — the chromatic factor
  `1/(1+δ)` of the orbit's own `δ` — while a deck written with the *local* rigidity
  (`k1 = 4.08`) reads back as `3.33`: the old `energy_mode=local` default corrected the energy
  twice.  The new default `delta` (conventions §3, Bmad's `bmad_to_mad` convention) normalizes
  with the engine's own orbit momentum and rescales kicks, maps and a bend's `k0` alike.
* **… and they keep their clock at the start velocity.**  xtrack: `ΔE = V sin(lag − 2π ζ/(β0 λ))`,
  MAD-X: `ΔE = V sin(2π lag − 2π f t/c)` (both measured; both kick at the centre of a thick
  cavity), so a particle arriving `Δt` after the `s/(β0 c)` clock sees its phase advanced by
  `2π f Δt`.  After the first gap of `dtl_section.dat` the reference is already tens of degrees
  early at the next one.  `delta` mode therefore writes every downstream cavity at
  `φ − 2π f Δt_design` (`CONST_P0_PHASE_SLIP`) and the reader walks the design back in.
  Result on the DTL (66 % kinetic-energy gain over 20 elements), HELIX vs xtrack on the
  transverse block: `T4x4 = 3.5e-10` (it was 11 before either fix, 0.45 with the rigidity fix
  alone).  MAD-X stays at 3.7 on the same deck: its `twiss` maps are second-order expansions
  about the orbit and `δ` reaches 0.3 there — an engine limit, not a translation one
  (`mebt_line.dat`, δ ≈ 0.03, is at 1e-3).
* **The xtrack oracle now measures maps around the tracked reference orbit** (it seeded every
  element's finite differences at `δ = 0` before, which is a different particle after a cavity).
* **A `ReferenceChange` travels through MAD-X/MAD8 as a tag** on its marker
  (`REFCHANGE_AS_TAG`, restored by the reader): the energy jump the engine cannot apply no
  longer disappears from the round trip; xtrack's `ReferenceEnergyIncrease` step is resolved
  into the jump on read.
* **TraceWin's bend sign, measured with the licensed binary**: `BEND −11.25 ρ>0` gives the
  positive bend's transverse block (`R12 = +1.607`, `R21 = −0.0237`) with `R16`, `R26` flipped,
  and an `EDGE` angle acts on its own sign (`EDGE −5.625` on that bend focuses, `+5.625`
  cancels the body as an rbend should) — so the writer's `β = sign(θ)·e` rule is right and
  the real BTL/fnalscl decks (`BEND −1.494 69052.8 …`) follow the same convention.
  **HELIX's `Dipole._body_matrix` evaluates the sector map at the signed angle with ρ > 0**,
  giving `R12 = −1.607`, `R21 = +0.0237` and `R26` alone flipped for the same card (the y block
  and the edges are right).  The battery rates HELIX pairs on decks with negative-angle bends
  report-only (`psb.seq` was 2.6e4 off for this reason alone); the fix belongs in HELIX.
* **FLAME's `tmatrix` is read into SI now** (the reader used to keep millimetres with a
  `basis="flame"` flag every other writer ignored: an RF-defocusing lens read from FLAME came
  out 1000× too weak in Elegant/Bmad/xtrack, 4.2 on the MEBT line).
* **SciBmad 0.5.2 (Phase 5, 2026-09-05)**, measured before writing a line of the format: see the
  engines table for the conventions.  With the delta energy mode, the `LineElement` RF lens and the
  worker's 1 µm substitution for thin gaps, lattix-written SciBmad decks agree with HELIX on the
  transverse block to `1.2e-10` (`mebt_line.dat`) and `1.6e-11` (`dtl_section.dat`, 66 % energy
  gain) and with cpymad to `8e-11` on `fodo.madx`; Bmad's own `bmad_to_scibmad` reference output
  parses (its `SaganCavity(num_cells=…)` and `PhaseReference.*` spellings differ from 0.5.2's API
  and are kept as native text).  Fringe integrals are stored but untracked there, so bends with
  `fint` are report-only against SciBmad.
* **Derived sources keep the tier of the write that made them**: the MEBT line written to FLAME
  has no cavity model (`RFCAVITY_NEEDS_CAVTYPE`), to IMPACT-Z a 1 mm short cavity whose own
  field integration gives a different transverse kick than TraceWin's thin-gap formula
  (2.7e-2 vs Bmad, 4.3 vs HELIX through a TraceWin `GAP`), so cases starting from those decks
  are report-only rather than false failures.  ImpactX's `drift + ShortRF + drift` triple is
  read back as the thick cavity it was (`THICK_CAVITY_RESTORED`), so it no longer gains a
  thin-gap lens on the second write.

## Phase 5.1 measurements (xtrack 0.112.0, 2026-09-05)

* **Every element class is read.**  102 `BeamElement` subclasses in `xt.__dict__`, all in the reader
  table (`tests/formats/test_xtrack_convert.py::test_known_classes_matches_what_from_line_actually_maps`).
* **`RBend` face angles are face-referenced.**  A hand-built
  `xt.RBend(length_straight=0.8, angle=0.08, edge_entry_angle=0.01, edge_exit_angle=0.02, k1=0.1)`
  matches cpymad's `rbend, l=0.8, angle=0.08, e1=0.01, e2=0.02, k1=0.1` (rbarc on) to **5.1e-10** on
  the transverse block, i.e. the same convention as xtrack's own MAD-X loader (`e1` copied as is,
  `l` → `length_straight`).  The IR keeps sector-referenced angles (`e1 + θ/2`, `rect=True`), so the
  reader adds the wedge (`θ/2 ∓ rbend_angle_diff/2`) and the `rbend=True` writer subtracts it; the
  default sector `Bend` output matches cpymad to 5.0e-10 on the same deck.
* **Knobs.**  `psb.seq` through the MAD-X reader carries 128 deferred expressions; they come out of
  xtrack as `element_refs` expressions and of MAD-NG (`--to madng`) as `k2 =\ (k2bi4bsw1l11)`.  A
  MAD-X deck written with plain `=` (HELIX's `fodo.madx`, `k1=kfocus`) has no knobs to keep: MAD-X
  evaluates `=` at definition time, only `:=` is deferred.
* **MAD-NG output** is xtrack's `mad_writer.to_madng_sequence` on the lattix `Line`; no MAD-NG
  engine here, the ledger says `VIA_XTRACK` for every element.

## Phase 5.2 measurements (LightWin 0.16.5, 2026-09-05)

* **Basis and units.**  A 1 m drift at 2.1 MeV gives `R12 = 1`, `R56 = +0.9955387 = L/γ²` in
  `(x, x', y, y', z [m], dp/p)`: TraceWin's basis in SI, `z` ahead-positive (`Basis.TRACEWIN`).
  A 5 T/m, 0.2 m quadrupole focuses `x` for a proton (`R11 = cos(√k L)`, `k = G/Bρ`, 1e-9).
* **Field-map cavity vs HELIX** (fingerprint deck: 2 cm of constant Ez, `SET_SYNC_PHASE`, φs = −30°):
  gain 873.6 keV (LightWin, `v_cav = 1.0068 MV`, `phi_s = −30.000°`) vs 872.0 keV (HELIX) vs
  873.6 keV (lattix `integrate_map`); `R22` 0.83916 vs 0.84058; LightWin carries a transverse RF
  focusing term (`R21 = −0.0112`) HELIX's map model does not: Equivalent tier, not Exact.
* **Relative field-map phases: HELIX and TraceWin/LightWin differ.**  A `FIELD_MAP … φ0 … 0`
  card without `SET_SYNC_PHASE` carries a *relative* phase.  Scanning φ0 on a 2 cm constant map
  behind a 0.5 m drift (2.1 MeV, 162.5 MHz), HELIX's gain-vs-φ0 curve is the LightWin curve shifted
  by 20.4°, and on the ADS spoke map behind the same drift (20 MeV, 352.2 MHz) by 40°: exactly
  `2πf·L/(βc) mod 360°` of the drift, i.e. HELIX adds the running bunch phase at the map entrance
  to a relative phase; LightWin takes φ0 as the RF phase when the reference particle enters.  On
  LightWin's vendored ADS deck (`tests/data/public/lightwin/example.dat`, 142 maps, 20 MeV in) the
  arrival convention reaches **502.24 MeV** (LightWin's own regression value; lattix
  `integrate_map` with the same convention: 502.22 MeV, every cavity within 2e-4), HELIX's
  convention 22.55 MeV — the deck accelerates only with the arrival convention, which is therefore
  what TraceWin (the deck's author) does.  lattix now uses it (`lattix.ir.fieldmap._ADD_RUNNING_PHASE
  = False`); synchronous-phase decks (PIP-II: `SET_SYNC_PHASE` everywhere) are unaffected.
  **HELIX bug to fix in HELIX** (`linac_gen` field-map propagation), like the negative-bend one.
* **The TraceWin trial + LightWin's `generic_project.ini` do not integrate 1-D field maps
  usefully**: on the same decks the licensed binary reports ≈ 0 keV for the spoke map at every
  φ0 and half the amplitude on the synthetic map (the effective map length looks halved) — the
  batch settings in the binary `.ini` are not under our control; the TraceWin oracle is not a
  field-map reference until a native project file is built (`docs/oracles.md` open item).
* **What LightWin cannot model** (`tests/oracles/test_lightwin_adapter.py`): GAP cards are skipped
  outright (LightWin's `Dummy` instruction), EDGE/THIN_STEERING/APERTURE/DIAG_* become drifts of
  their length (`DriftEnvelope3DParameters`), TITLE/PARTRAN_STEP are ignored commands.  The
  adapter lists them in `meta["dropped"]` / `meta["substituted"]` / `meta["ignored_commands"]`
  and the battery makes any such pair report-only.  In the battery LightWin is the fallback engine
  for TraceWin decks when HELIX is absent (CI), `lattix.crossval.ENGINE_CANDIDATES`.
* **LightWin has no solenoid model** (`SolenoidEnvelope3DParameters` raises `NotImplementedError`,
  and geometry-10 static maps are refused outright): the worker writes SOLENOID cards as drifts of
  the same length (`meta["solenoids_as_drifts"]`, report-only in the battery), so a solenoid-focused
  linac is compared on its reference energy only.  The TraceWin writer's `static_maps="hard_edge"`
  option (`FM_SOL_HARDEDGE`/`FM_QUAD_HARDEDGE`) removes the static maps for it.
* **LightWin's synchronous phase is charge-blind**: run with `q_adim = −1`, the fingerprint cavity
  *decelerates* an H⁻ by 860 keV while reporting `phi_s = −30°` (HELIX/lattix: +872 keV).  The
  worker therefore emulates a negative particle as a positive one of the same mass in mirrored
  fields (QUAD gradients, THIN_STEERING fields and static `kb` negated, relative RF phases +180° —
  TraceWin's rule in `lattix.ir.rf`): +873.5 keV, quads identical to HELIX (5e-15) for both signs.
* **PIP-II MEBT+HWR gate** (private deck, `static_maps="hard_edge"`, H⁻ 2.1 MeV): the energy after
  each of the 12 RF maps (4 bunchers, 8 HWR cavities) agrees between LightWin and HELIX to 1.2e-4 (10.2211 vs 10.2223 MeV at the
  exit) and, run at the CEA export's own 2.1227 MeV input, with TraceWin's `energy.txt` within 0.5 %
  (10.262 MeV).
* **The ADS deck in the battery** (`lightwin/example.dat`, 142 relative-phase 1-D maps): every
  `FM_TO_CAVITY`/`FM_AS_CAVITY` conversion now carries the integrated `(V_c, φs)` pair (the card
  phase of a relative-phase map is not its synchronous phase — xtrack and ImpactX used it and
  decelerated), all 11 conversions round-trip through the IR and are write→read→write fixed points,
  and the reference energy of the derived cavities through **Elegant, Bmad and ImpactX agrees with
  LightWin's envelope to 5.9e-5** (502.2232 vs 502.2414 MeV at the exit; IMPACT-Z's `rfdata`
  Fourier cavities 0.95 %).  The transverse block does not: every code models the RF focusing of a
  cavity differently (LightWin integrates the map, Elegant/Bmad have end-field models, xtrack's
  `Cavity` has none), and over 142 cavities at 20–500 MeV the cumulated 4×4 map differs by O(10);
  such pairs assert the energy and report the map.  Backlog: give derived cavities the thin-gap RF
  focusing lens thin GAPs already get (`RF_FOCUSING_AS_MATRIX`), engine by engine.

## Phase 5.3 measurements (Cheetah 0.8.4, 2026-09-05)

* **Basis.**  A 1 m drift at 2.1 MeV: `R12 = 1`, `R56 = −223.1485 = −L/(β²γ²)` in
  `(x, px, y, py, τ, ΔE/(p0 c))`: MAD-X's canonical pair with `τ = c·Δt` *late*-positive
  (`Basis.CHEETAH`, `_Z_SIGN = −1`; the MAD-X transform alone gave `R56_common = −0.9955`).  With the
  kinetic energy handed to Cheetah (its proton mass is CODATA 2022, 1.3 eV above lattix's) the
  fingerprint drift matches the analytic map to 1e-8.
* **Cavity.**  `ΔE = −voltage·q·cos(phase)`: +866 keV for an electron and −866 keV for a proton at
  `voltage = 1 MV, phase = −30°`; the slope `r65 = k·sin(phase)·V_eff/(β(E+ΔE))` is negative at
  `phase = −30°`, i.e. a late particle gains *less* — Cheetah's phase is the negative of the IR's
  (its own converters: Bmad `phase = −phi0`, Elegant `phase − 90°`).  The writer emits
  `voltage = −V/q`, `phase = −φ`; the fingerprint then bunches (`R65_common < 0`) and gains
  `V·cos 30°` to 1e-9 for protons and H⁻ alike.  A zero-length cavity is `inf` (the matrix divides
  by the length); at 1 µm the cavity's own focusing is ~1e-5/m, so the lattix thin-gap lens carries it.
* **Magnets.**  `k1 > 0` focuses `x` for proton, electron and H⁻ alike (no charge in the map), the
  solenoid `k = B/(2Bρ)` rotation ignores the charge, correctors kick `px += angle`: the writer
  normalizes with the signed rigidity and the H⁻ FODO/solenoid decks agree with HELIX to 5e-15.
  `Dipole.gap` is the full gap and `fringe_integral` enters the linear edge map (`R43` moves from
  −0.009983 to −0.009533 with `gap = 0.05, fint = 0.45`); face angles are sector-referenced (RBend adds
  `angle/2`); the fringe model equals MAD-X's (`fodo.madx` 1.8e-15 on the transverse block, 4e-10 on
  dispersion and path length, 7e-9 on R56).
* **Gates.**  `fodo.madx` vs cpymad and vs Bmad exact on every block; `fodo_cell.dat` 3e-14 and
  `solenoid_channel.dat` 5e-15 vs HELIX; `mebt_line.dat` (H⁻, two thin gaps) 6.5e-3 vs HELIX on the
  transverse block with the energy exact — Equivalent tier, as the plan expected
  (`CHEETAH_ZERO_LENGTH_CAVITY`).
* **Lockstep through a third party.**  The `.bmad` lattix writes for `fodo.madx`, read by Cheetah's
  own `converters.bmad` (its namelist parser rejects `title, "…"`, which the worker drops), gives the
  same Cheetah maps as lattix's LatticeJSON of the same lattice (T4×4 and dispersion below 1e-8):
  two lattix writers checked against each other by someone else's reader.

## Phase 5.4 measurements (PyORBIT3 meson build of `22b45fa`, 2026-09-05)

* **Basis.**  A 1 m drift at 2.1 MeV: `R56 = +237.30` in `(x [m], x′, y [m], y′, z [m], dE [GeV])`
  = `L/γ² · 10⁹/(β²γ mc²)`: z is ahead-positive and the energy coordinate is absolute
  (`Basis.PYORBIT`: `d[4] = +1`, `d[5] = 10⁹/(β²γ mc²)`).  The reference energy follows the gaps
  (`kinEnergy` of the synchronous particle at each `LinacTrMatrixGenNode`).
* **Matrix generation.**  `LinacTrMatricesController.addTrMatrixGenNodes` puts a node at the
  *entrance* of every element holding the least-squares map from the first node to itself; the
  per-element map is `R[k+1]·R[k]⁻¹`, with one more node at the exit of the last element.  Symmetric
  probes make the fit clean to second order, but the tracker's third-order terms leave an O(ε²)
  residue: 5.0e-5 (2e-7 relative) on the 5.3 m chicane at ±1e-5, 4.5e-6 at ±3e-6, 5.1e-7 at ±1e-6.
  Tracking twice (ε and ε/2) and Richardson-extrapolating brings the chicane to 1.1e-8 against MAD-X
  and HELIX, independent of ε (the same at 3ε).  An energy probe as large as the transverse one
  (1e-5 GeV) costs 1e-5 on R56 at 2.1 MeV; ±1e-7 GeV is used.
* **Gaps.**  `ΔE = q·E0TL·cos(phase)` (E0TL in GeV, phase in degrees): a proton gains 866 025.4 eV
  at `E0TL = 1e-3, phase = −30`, H⁻ needs `phase + 180°` for the same gain, and the slope bunches at
  −30° (`R65_common < 0`) — TraceWin's GAP convention.  PyORBIT derives the phase of every later gap
  of a `<Cavity>` from the time of flight through its first gap: sharing one cavity between the
  MEBT's gaps lost 2.6 % of the energy, so lattix writes one `<Cavity>` per gap.  `BaseRfGap`
  carries PyORBIT's own transverse gap focusing: the MEBT's thin gaps are Equivalent tier (4.3e-3 vs
  HELIX on the transverse block, energy exact), as the plan expected.
* **Magnets.**  `QUAD field` is the lab gradient with the charge in `bunch.B_Rho()` (`fodo_cell.dat`
  1.2e-11 vs HELIX; the H⁻ SNS MEBT quads defocus with the signed rigidity); `SOLENOID B` is `B₀/Bρ`
  in 1/m — TEAPOT's `soln` normalized strength — with the charge applied by the tracker
  (`solenoid_channel.dat` 9e-13); `BEND theta` has MAD-X's sign with sector-referenced `ea1/ea2`
  (single sector bends of 1°–45° vs MAD-X: 1e-11 transverse, 9e-10 dispersion; `bend_line.dat`
  2.9e-11 vs HELIX); `DCH B·effLength` kicks `x′ −= B·L/Bρ_signed`, `DCV` `y′ += B·L/Bρ_signed`.  The
  SNS MEBT correctors sit *inside* the quads and PyORBIT applies them between the magnet's parts;
  the reader splits the quad the same way.
* **File rules.**  `pos` is the element centre; drifts are implicit (`SNS_LinacLatticeFactory` fills
  the gaps) and its overlap check is strict (`dist = pos₁ − L₁/2 − (pos₀ + L₀/2) < 0`, no
  tolerance), so the writer keeps lengths on a picometre grid, cumulates positions from the written
  lengths, and pushes a node 1 pm when PyORBIT's own double arithmetic on the written decimals would
  see an overlap; the reader parses the decimals exactly and treats gaps below 1 nm as touching.
  Rounding positions to 12 decimals instead flipped one SIS18 position by 1 pm on the second
  write (a value 3e-14 from a rounding boundary).  `aperture` is a full diameter.  A fringe
  integral without a gap (`fint·hgap = 0`) is inert in every engine and is not a loss.
* **Gates.**  SNS MEBT (the vendored `<MEBT>` sequence of `sns_linac.xml`) and ESS MEBT: PyORBIT on
  the original and on lattix's read → write agree to 7.7e-10 / 6.6e-11 with identical final energies
  (thin correctors inside quads, per-gap cavities and the TTF passthrough all survive); both and the
  solenoid test deck are text fixed points.  `.dat` → XML vs HELIX: `fodo_cell` 1.2e-11,
  `solenoid_channel` 9e-13, `bend_line` 2.9e-11, `csr_chicane` 1.1e-8 (Exact), `mebt_line` 4.3e-3
  (Equivalent).

## Phase 5.5 measurements (IMPACT-T 3.1.5, 2026-09-05)

* **Deck rules.**  `flagdist 16` reads `partcl.data` (nine columns: `x, γβx, y, γβy, z, γβz,
  q/m, weight, id`; `nemission −1`); `-2` writes `fort.<mapstp>` as a fixed-time snapshot with
  the same columns; `-1` steers (`Param2` = time, `Param3..8` = `dx, dpx, dy, dpy, dz, dpz` in
  γβ units); `-4` changes `dt` from the push of the iteration starting nearest; `-11` collimates
  (`xmin xmax ymin ymax`, flag ≤ 10 rectangular); `-99` stops when the bunch passes `Param3`.
  Every negative-type card of a run lives in one array of `Nbpmmax = 200` (`NumConst.f90`; PSB's
  267 of them crashed the binary), drifts in 1400, quadrupoles in 400, dipoles in 100.  A control
  fires on the integration step nearest to its position (a card ±0.25 step from a boundary acts
  at the landing step, ±0.75 one step off), so the oracle puts every card exactly on a boundary
  and gives each element its own `dt = L/(N βc)`: quads agree with the analytic map to 4e-8,
  drifts to 5e-13.
* **Centroid triggers, in-order lists.**  The trigger position `distance` is the bunch centroid
  `<z>` (`globalrange`), and each control list is walked in order: a control the centroid passes
  inside a dipole loop (no triggers there) is skipped and blocks every later control of its kind.
  ELENA and PSB showed it — under IMPACT-T's dipole model (no dispersion or edge focusing) the
  momentum probes run away (4 cm at 24 m) and drag the centroid 1.2 mm behind the reference, and
  PSB's 3 µm drift after a bend got a `dt` of 6e-21 s (its length minus the dipole's overshoot).
  The oracle now keeps 1 cm field-free clearances before and after a dipole, dumps before one
  only where a drift holds the clearance, lets an element shorter than the overshoot carry it to
  the next, chunks at 186 BPM-type cards with each chunk's deck trimmed to what its bunch can
  reach (which also made the ring runs 10× faster), and resumes a chunk from the last dump that
  was written (`meta["missing_dumps"]`, a warning).  Both rings complete as report-only cases.
* **No field at z < 0.**  A solenoid table starting at −0.3 mm (the first element of a deck)
  delivered 55 % of its entrance kick (`R11` 0.884 vs 0.789 for 0.4 m at 0.5 T); the same table
  after a 0.3 m drift, or the whole deck moved by +1 m, the full kick.  The writer moves a fresh
  deck whose first field region would start below zero (`IMPACTT_DECK_OFFSET`, a deck from
  IMPACT-T keeps its origin), and the deck's clock is the reference at z = 0: the reader starts
  the walk with the field-free flight to the nominal start, the oracle launches the probes
  before the first control and delays the driven phases by the flight to z = 0.
* **Solenoid tables.**  `getfldt_Sol` has no analytic field: type 3 needs `1T<id>.T7`
  (`Rmin Rmax NrIntv`, `Zmin Zmax NzIntv` in cm, then `Br Bz` rows with r fastest), scaled by
  `Param2`; file id 0 stops the run and particles beyond `Rmax` are lost.  A paraxial lab-frame
  integration of the bilinear table reproduces IMPACT-T's own map to 1e-9, which let the writer's
  edge be designed offline: three intervals straddling each nominal end with
  `Bz/B = 0, −0.2318182, 1.2318182, 1` and `Br/(B·r/h) = 0, −¼, −¼, 0` give the hard-edge map to
  `O(h²)` (3e-8 at `h = 0.2 mm` for 0.4 m, 0.5 T, 2.1 MeV protons; a plain `0 → ½ → 1` ramp with
  ∫Bz and ∫Bz² compensated is `O(h)`: 8.8e-6, reproduced by the model to three digits).
  Tracked: 9.2e-7 at a 1 ps step, 6.2e-7 as `dt → 0`, 4.9e-5 when the interval has only five
  steps — the oracle uses at least ten (`solenoid_channel.dat` 9.3e-7 vs HELIX).
* **Cavities.**  A type-104 profile is `scale·Ez(u)·cos(2πf·t + θ0)` on the absolute time, so
  `theta0` is calibrated at lattix's time of flight by integrating the reference through the
  profile (RK4; `V` = the largest gain over `theta0`, `φs` from the gain and its slope, to
  machine precision).  The first-order transit-time calibration was 5.3e-4 off on a thin gap and
  *reflected* the reference of a 0.2 m, 1 MV cavity at 325 MHz (a bump spanning 3 βλ has a
  vanishing transit factor, the calibrated field was 1.4 GV/m): the bump is now no wider than
  βλ/2.  IMPACT-T's reference leaves each cavity a little earlier or later than the IR's point
  gain says, and the writer carries that to the cavities downstream (without it the MEBT's energy
  was 2.4e-4 off, with it 1.5e-7).  Measured gains at φs = −30°: a 1 MV thin gap 866 025.7 eV for
  the proton and for H⁻ (4e-7 of V·cos 30°; 3.3e-6 as a 1 mm surrogate), 0.2 m/1 MV/325 MHz
  6.7e-7, 0.1 m/300 kV 2.6e-7; the slope bunches (`R65 < 0`).
* **Thin gaps.**  A thin gap becomes a short cavity whose length the neighbouring drifts give up
  (tag `L=0 pad=…`; the reader restores the gap and the drifts).  No length reproduces the
  thin-gap formula's transverse focusing in a field integration: the kick is smeared over the
  bump (the MEBT's 80 kV gaps vs HELIX: 6e-3 at 1–4 mm, 1.6e-2 at 12 mm, 3.6e-2 at βλ/2) while a
  strong short bump adds ponderomotive focusing (a 577 kV DTL gap: `R21` −9.1 vs 2.7 at 1 mm,
  1.9 at 10 mm, 2.63–2.86 at 20–60 mm).  The length now balances the two,
  `0.4·sqrt(qVλ/(2π mc² βγ³ |sin φ|))` within `[1 mm, βλ/2]` (5 mm for the MEBT, 16–20 mm for the
  DTL): `mebt_line.dat` vs HELIX 8.1e-3 transverse and 1.5e-7 on the energy (Equivalent), the DTL
  section 5 on the cumulative map (its gaps take 27 % of the energy each — beyond the thin-gap
  model, and report-only in the battery).
* **Dipoles.**  Type 4 needs the 22-value pole-face file (`csr_flag, γ_in, k1, b1 … k4, b4, z01,
  z02, c1 … c8, zcsr1, zcsr2`; without it the run is NaN).  The tracked dipole bends the whole
  bunch by the reference angle: `R21 = R26 = 0` (fodo.madx bends 1e-2 vs cpymad, its quads and
  drifts 1.2e-9), so bend decks are report-only in the battery.  The bend loop starts when the
  *first* particle reaches the face and re-bases the frame one step past the exit
  (`δ = (n+1)·dt·βc − arc` with `dt = arc/((n+1e-4) βc)`), which is where the oracle's step switch
  and dump go (plus the 1 cm clearance).
* **Gates.**  `fodo_cell.dat` 2.3e-8 vs HELIX (Exact), `solenoid_channel.dat` 9.3e-7,
  `mebt_line.dat` 8.1e-3 / energy 1.5e-7 (Equivalent), `bend_line.dat` report-only; `fodo.madx`
  quads and drifts 1.2e-9 vs cpymad; `Sample1` (SolRF, CCL, three overlapping cards) reads,
  rewrites bit-exact with its data files under their original ids, and the all-kinds lattice is a
  text fixed point for protons and H⁻; the battery's IMPACT-T pairs (every source format,
  the derived decks, PSB and ELENA as report-only rings) are all within tier.
