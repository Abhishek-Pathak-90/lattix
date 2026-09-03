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
| ImpactX / IMPACT-Z | env `lattix` (`impactx` 26.08, `impact-z` 2.7.7, both osx-arm64 builds) | — | Phase 3 | follows | `ImpactZexe` on PATH in the env; `impactx` importable. |

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
