# Phase 5 — the remaining codes (Cheetah, xsuite, Ocelot, LightWin, IMPACT-T, PyORBIT3, DYNAC, Synergia2, OPAL-X, the Bmad bridge)

> Same format as PLAN.md (phases → tasks → gates). Every fact below was read out of the clones under
> `particle_tracking_codes/` on 2026-09-04 (commit dates given) or checked against PyPI/conda-forge that
> day; anything not verified is marked *verify*.  Phases 0–4 of PLAN.md are complete (0.1.0).

## 0. Findings per code

| Code | Clone / release | Install here | Licence | Native lattice form | What it gives lattix | Verdict |
|---|---|---|---|---|---|---|
| **xtrack / xsuite** | clone 0.104.1 (2026-05-18); env `lattix` runs **0.112.0** | present | Apache-2.0 | `Line`/`Environment` JSON, MAD-X in/out, **MAD-NG out** (`mad_writer.to_madng_sequence`) | element gaps to close (below), xdeps knobs, multi-line `Environment`, MAD-NG export for free | **P1** |
| **LightWin** | 0.16.5 (2026-04-13) | `pip install lightwin`, **needs Python ≥ 3.12** → own env | MIT | TraceWin `.dat` (reader + writer) | (a) a second, independent TraceWin parser for lockstep tests; (b) **free TraceWin-semantics envelope** (`Envelope3D`: drift, quad, 1-D RF field maps geometries 100/1100/70/7700, thin lens; `Envelope1D`: + bends, superposed maps) — an oracle for the SC linac that is not limited to 20 elements like the trial TraceWin | **P1** |
| **Cheetah** | clone 0.8.2-dev (2026-04-29); PyPI 0.8.4 | `pip install cheetah-accelerator` + **torch** (base env has torch 2.10, env `lattix` does not) → own env | GPL-3 → runtime-optional adapter, never vendored | Python `Segment` objects; `to_lattice_json`/`from_lattice_json`; readers for Bmad, Elegant, Ocelot, NX tables | differentiable optics on the same lattice; per-element `first_order_transfer_map(energy, species)` (7×7 augmented) as an oracle; `Species` with proton/electron/deuteron or custom mass/charge | **P2** |
| **PyORBIT3** | clone 2026-05-14 | no PyPI/conda package: conda env (py 3.10, fftw, meson, ninja) + `pip install .` (C++ build) | MIT | **linac XML** (`sns_linac_lattice_factory`); rings via its MAD8/MAD-X/SAD parsers into TEAPOT | Booster/injection peer with foil physics; transport matrices from `LinacTrMatrixGenNode.getTransportMatrix()` and `TEAPOT_MATRIX_Lattice.getRingMatrix()`; SNS and ESS linac XML examples (MIT, vendorable) | **P2** |
| **IMPACT-T** | clone 2025-10-03 | conda-forge `impact-t 3.1.5` (nompi build) | BSD | `ImpactT.in`: same header family as IMPACT-Z, elements positioned by absolute `zedge`, RF/solenoid field maps as `rfdata` Fourier coefficients | time-domain sibling of IMPACT-Z: ~70 % of the IMPACT-Z writer reusable; `lume-impact 0.12.1` (PyPI) already parses both input formats and runs the engines | **P2** |
| **Ocelot** | clone 25.07.1 (2025-07-17); PyPI 26.6.1 | `pip install ocelot-collab` | GPL-3 → runtime-optional | Python lattice file (`from ocelot import *` + element constructors + `cell = (…)`); adaptors: MAD-X seq, MAD8 twiss, Elegant (in/out), **Astra (out)**, Genesis 2/4, CSRtrack, longlist, TFS, openPMD | electron-machine ecosystem (XFEL/FLASH); `lattice_transfer_map(lattice, E_GeV)` R matrices | **P3** (electron only) |
| **Bmad util programs** | env `bmad` (20260828) | present | Bmad licence | Tao `write` → SCIBMAD, PALS, ELEGANT, MAD-8, MAD-X, SAD, **OPAL-T**; programs `bmad_to_astra`, `bmad_to_gpt`, `bmad_to_csrtrack`, `bmad_to_merlin`, `bmad_to_slicktrack`; readers `elegant_to_bmad`, `mad_to_bmad`, `sad_to_bmad`, `sxf_to_bmad`, `ptc_flat_file_to_bmad`, `slicktrack_to_bmad`, `accelerator_toolkit_to_bmad` | a **bridge**: IR → `.bmad` → Bmad tool → Astra / GPT / OPAL-T / SAD / CSRtrack / Merlin / SlickTrack, and AT / SAD / SXF / PTC-flat → `.bmad` → IR, at the cost of one ledger entry `VIA_BMAD` | **P3** (cheap) |
| **DYNAC** | clone v6 (2016-10-25) | CMake + **gfortran** (not installed; `brew install gcc`) | custom EULA, **not open source**: run locally, never redistribute decks or output | 59-card deck (`dynac.F:160-172`), cm / kG / MV units; ships `converters/tw2dyn.f` (TraceWin → DYNAC for DRIFT, QUAD, DTL_CEL, FREQ, GAP, FIELD_MAP; EDGE/BEND "not yet implemented") | ion-linac cross-check; `FIRORD` card prints first-order matrices ("TRANSVERSE CANONICAL MATRIX (cm, radian)") → oracle; ESS full linac (RFQ, MEBT, DTL, spoke, SCL at 0 / 62.5 mA) and SNS MEBT+DTL1 decks with `dynac.print.ref` | **P3** |
| **Synergia2** | clone 2026-03-27 | pixi (osx-arm64 supported; pixi not installed); the PyPI `synergia` 0.1.3 is an unrelated package | Fermilab licence | MAD-X / MAD8 readers (`madx_reader`, `mad8_reader.py`), native **JSON** (`Lattice.as_json` / `load_from_json`); `examples/mi/mi20_ra_08182020.lat` (Main Injector, MAD8-style), Booster examples reference `booster_init_lattice.json` | Fermilab Booster/MI peer; `Lattice_simulator.get_linear_one_turn_map`, `tune_linear_lattice`; libFF elements: drift, quadrupole, sextupole, octupole, sbend, rbend, dipedge, rfcavity, kicker, multipole, solenoid, marker, matrix, constfoc, elens, foil, nllens | **P3** |
| **OPAL-X** (= OPAL 2) | clone 2026-04-14 | source build: MPI, H5Hut, HDF5, GSL, Boost ≥ 1.66, optional Trilinos/AMReX — no conda package, no `mpirun` here | GPL-3 | MAD-like `.in`: RFCAVITY (`VOLT` [MV], `FREQ` [MHz], `LAG` [rad], `FMAPFN`), TRAVELINGWAVE, VARIABLE_RF_CAVITY, SBEND/RBEND/SBEND3D, QUADRUPOLE, SEXTUPOLE, OCTUPOLE, MULTIPOLE(T), SOLENOID, DRIFT, MARKER, MONITOR, KICKER/HKICKER/VKICKER, C/E/R/FLEXIBLE COLLIMATOR, SLIT, PEPPERPOT, DEGRADER, STRIPPER, SEPTUM, PROBE, OUTPUTPLANE, UNDULATOR, VACUUM, SOURCE, CYCLOTRON, RINGDEFINITION, SCALINGFFAMAGNET; OPAL-T places elements by `ELEMEDGE` | LEBT/RFQ/photoinjector community format; no example decks in the clone (regression tests live elsewhere); Tao's `write opal` is a reference writer | **P4** (writer first, no engine) |
| xfields, xobjects, fbpic | — | — | — | no lattice concept | nothing to translate | n/a |

Toolchain on this Mac (2026-09-04): cmake 4.3.2, Boost 1.92 (Homebrew), **no gfortran, no mpirun, no pixi**; torch 2.10 in the base env only.

## 1. What the survey pinned down (conversion facts to encode)

### 1.1 xsuite (xtrack 0.112.0, measured by introspection)
* Element classes lattix does **not** map yet: `RBend` (same fields as `Bend` plus straight length), `Magnet` (unified thick magnet: `k0..k3`, `k0s..k3s`, `knl/ksl`, `angle`, `h`, edges), `Misalignment` (`dx dy ds theta phi psi anchor` — a first-class misalignment element), `MultipoleEdge`, `MagnetEdge`, `DipoleEdge` (`r21 r43 hgap k e1 fint model side`), `Wedge`, `RFMultipole` (`voltage frequency lag knl ksl pn ps`), `CrabCavity`, `ACDipole`, `Elens`, `Wire`, `Exciter`, `NonLinearLens`, `LimitPolygon`, `LimitRectEllipse`, `LimitRacetrack`, `LongitudinalLimitRect`, `SecondOrderTaylorMap` (`R`, `T`, `k`), `LineSegmentMap`, `VariableSolenoid` (`ks_profile`), the thick/thin slice classes.
* `Bend`/`RBend` carry `edge_entry_fint`, `edge_entry_hgap`, `edge_entry_model` (`linear | full | dipole-only | suppressed`), `model` (`adaptive | full | bend-kick-bend | rot-kick-rot | mat-kick-mat | …`); lattix already writes fint/hgap.
* Expressions: `line.to_dict()` carries `_var_management_data`; `line.element_refs['q'].k1._expr` is `vars['kq']`; `env.new('q', xt.Quadrupole, k1='kq', …)` accepts expression strings — so MAD-X `:=` knobs can round-trip TraceWin/MAD-X → IR `Expr` → xtrack and back.
* `to_madx_sequence` and `to_madng_sequence` emit knobs as `k1 := (kq)` (MAD-X) and `k1 =\ (kq)` inside Lua `bline`/`sequence` chunks (MAD-NG); `Line.to_madng` needs `pymadng`.
* xtrack ≥ 0.112 refuses kernels without `xsuite` unless JIT is allowed (done in 0.1.0: `allow_jit`).

### 1.2 LightWin
* `.dat` I/O: `tracewin_utils.dat_files.dat_filecontent_from_file` / `export_dat_filecontent`; elements `Aperture, Diagnostic, Bend, Drift, Edge, Quad, ThinSteering, Solenoid, FieldMap (geometries 100, 1100, 70, 7700), SuperposedFieldMap, DummyElement`.
* Transfer matrices: `envelope_3d/transfer_matrices_p.py`: `drift, quad, field_map_rk4, thin_lense` (no bend, solenoid or thin GAP transversely); `envelope_1d/transfer_matrices.py`: `z_drift, z_field_map_rk4, z_field_map_leapfrog, z_thin_lense, z_bend, z_superposed_field_maps_rk4` (longitudinal only).
* TraceWin runner: `beam_calculation/tracewin/tracewin.py` (`hide`, `dat_file`, `path_cal`), the pattern lattix already copied.

### 1.3 Cheetah (0.8.x)
* Coordinates `(x, px, y, py, tau, p)` with `tau = ct − s/β0` (docstring) → **late-positive**; fingerprint before use (expect `_Z_SIGN = −1` like Elegant/HELIX).
* `Cavity(length, voltage [V], phase [deg], frequency [Hz], cavity_type standing_wave|traveling_wave)`; gain `ΔE = −voltage · q · cos(phase)` (`effective_voltage = −voltage × num_elementary_charges`): a positive `voltage` accelerates electrons, so the IR's species-independent `V·cos φ` maps to `voltage = −V/sign(q)`; the R-matrix is a Rosenzweig–Serafini form (Equivalent tier against thin-gap codes).
* `Dipole(length, angle, k1, dipole_e1, dipole_e2, tilt, gap, gap_exit, fringe_integral, fringe_integral_exit, fringe_at, fringe_type="linear_edge", tracking_method linear|second_order|drift_kick_drift)`, `RBend`, `Quadrupole(length, k1, tilt, misalignment)`, `Sextupole(length, k2)`, `Solenoid(length, k)`, `HorizontalCorrector/VerticalCorrector(length, angle)`, `CombinedCorrector`, `TransverseDeflectingCavity`, `Undulator`, `Aperture(x_max, y_max, shape, is_active)`, `BPM`, `Screen`, `Marker`, `CustomTransferMap(predefined_transfer_map 7×7)`, `SpaceChargeKick`, nested `Segment`.
* `Species(name | charge, mass_eV)`: proton, electron, deuteron known; H⁻ as a custom species.
* Converters: `Segment.from_bmad / from_elegant / from_ocelot / from_nx_tables`, `to_lattice_json` / `from_lattice_json`.

### 1.4 PyORBIT3 linac XML (from `sns_linac.xml` and the factory)
* `<seq name length bpmFrequency>` → `<accElement type name length pos>` with `<parameters …/>`: `QUAD field[T/m] aperture aprt_type`; `DCH/DCV B[T] effLength`; `BEND theta[rad] ea1 ea2 poles kls skews aperture_x aperture_y`; `SOLENOID B`; `RFGAP E0L E0TL[GeV] phase[deg] mode cavity EzFile` + `<TTFs beta_min beta_max>` polynomials; `<Cavity name frequency ampl pos>`; `THICK_KICK`, `VACWIN material_index`, `MARKER`. Gap phases are converted to radians on load; E0TL is in GeV.
* Ring input: MAD8 (`TEAPOT_Lattice.readMAD`), MAD-X (`readMADX`), SAD.

### 1.5 IMPACT-T `ImpactT.in` (from `examples/Sample1`)
* Header: `npcol nprow`; `dt nstep nbunch`; `dim np flagmap flagerr flagdiag`; mesh; `flagdist restart …`; distribution rows; `I/A Ek/eV Mc2/eV Q/e freq/Hz phs/rad`.
* Elements: `L/m N/A N/A type zedge v1 … v23 /`, laid out by **absolute starting edge**: `<0` bpm (8), `0` drift (zedge radius), `1` quad (zedge gradient fileID radius misalignments), `2` constfoc, `3` solenoid (zedge Bz0 fileID …), `4` dipole (zedge Bx By fileID …), `5` multipole (typeID 2 sex / 3 oct / 4 dec), `101` DTL, `102` CCDTL, `103` CCL, `104` SC cavity (zedge scale freq theta0 fileID radius …), `105` SolRF (… Bz0), `110–113` EMfld variants; `rfdata` files hold Fourier coefficients (`utilities/RFcoeflcls.f90`).

### 1.6 DYNAC (`converters/tw2dyn.f`)
* Units cm and kG: `DRIFT L/10`; `QUADRUPO L/10, B_tip = 0.1·G·(R/20), R/20`; `DTL_CEL` → half `QUADRUPO`, negative `DRIFT`, `CAVSC` (cell index, βr, cell length, TTF, TTF′, TTF″, E0, φs, freq), second half quad; `FREQ` → `NEWF` (Hz); `GAP` → `BUNCHER` (E0TL in MV, phase, 1, aperture cm); `FIELD_MAP` → `FIELD` (file name, scale 1e6 V/m) + `CAVNUM`; `EDGE`/`BEND`/`NCELLS`/`MULTIPOLE`/`LATTICE` not converted.
* Matrices: `FIRORD`/`SECORD` cards print first/second-order matrices ("TRANSVERSE CANONICAL MATRIX (cm, radian)", `dynac.F:14269`).

### 1.7 Ocelot
* Electron-only physics: cavity map and optics divide by `m_e_GeV`; `Cavity(l, v [GV], phi [deg], freq [Hz], vx_up…)`, gain `V·cos φ` with no charge sign; `Bend(l, angle, k1, k2, e1, e2, tilt, gap = 2·HGAP, fint, fintx)`, `Quadrupole(l, k1, k2, tilt)`, `Solenoid(l, k)`, `Hcor/Vcor(l, angle)`, `Aperture(xmax, ymax, dx, dy, type)`, `Matrix(l, delta_e, r, t)`, `Multipole(kn)`, `TWCavity`, `TDCavity`, `Undulator`, `Monitor`, `Marker`, `XYQuadrupole`, `Pulse`.
* `lattice_transfer_map(lattice, energy_GeV)` and `lattice.transfer_maps(energy)` give per-element `(B, R, T)`.

### 1.8 OPAL-T
* Attribute units from the sources: RF voltage [MV], RF frequency [MHz], phase lag [rad]; elements positioned by `ELEMEDGE`; field maps by `FMAPFN` (T7 / 1DDynamic / 3D formats, *verify* the exact map formats in the OPAL manual).

## 2. Priorities and rationale (PIP-II linac + BTL + Booster)

1. **P1 — xsuite completion** (engines present, CI green): closes the reader gaps above, adds knob round-trips through xdeps, multi-line `Environment` JSON, `Misalignment` ⇄ `BodyShiftP`, and a MAD-NG writer by delegation to xtrack. Booster work stays on MAD-X/xtrack; MAD-NG comes for free.
2. **P1 — LightWin**: the cheapest large gain in *testing*: a second TraceWin parser to lockstep every public and private deck, and a free envelope oracle for quad + 1-D-map sections (HWR/SSR/LB650/HB650), which the 20-element trial TraceWin cannot cover.
3. **P2 — Cheetah**: differentiable optics on the PIP-II lattice (matching, tolerance studies with autograd, ML surrogates); an adapter plus an oracle, all GPL code loaded at call time only.
4. **P2 — PyORBIT3 linac XML**: the SNS/ESS lineage of linac description, foil/injection physics for BTL → Booster, and an independent transport-matrix oracle for linacs *and* rings.
5. **P2 — IMPACT-T**: completes the IMPACT family with little new code; needed for LEBT/RFQ/source studies where time-domain space charge matters.
6. **P3 — Bmad bridge**: Astra, GPT, OPAL-T, SAD, CSRtrack, Merlin, SlickTrack outputs and AT/SAD/SXF/PTC inputs in a few days, each labelled `VIA_BMAD` in the ledger.
7. **P3 — Ocelot, DYNAC, Synergia2**: peers with narrower value (electron-only; EULA-restricted; MAD-X already covers Synergia's input). Do them when a study needs them; the adapters are small.
8. **P4 — OPAL-T writer**: no engine here; writer validated structurally and against Tao's `write opal`; reader later.

## 3. Tasks

### 5.1 xsuite completion (1 week)
- [ ] Reader: `RBend`, `Magnet`, `Misalignment` → `BodyShiftP` (and `ds`/`anchor` → Patch when it cannot be absorbed), `DipoleEdge`/`MagnetEdge`/`MultipoleEdge` folded into the adjacent magnet (the MAD-X `dipedge` rule, `DIPEDGE_FOLDED`), `Wedge` → Bend with `k`/`k1`, `LimitPolygon/RectEllipse/Racetrack` → `Collimator` (bounding shape, LOSSY `APERTURE_SHAPE`), `LongitudinalLimitRect` → DROPPED, `RFMultipole` → RFCavity + LOSSY `RF_MULTIPOLE_TERMS_DROPPED`, `CrabCavity/ACDipole/Elens/Wire/Exciter/NonLinearLens/LineSegmentMap` → DROPPED with their own codes, `SecondOrderTaylorMap` → `Taylor` (order 2 when the IR carries `T`, else `TAYLOR_ORDER_TRUNCATED`), `VariableSolenoid` → hard-edge solenoid preserving ∫B and ∫B² (`FM_SOL_HARDEDGE` rule).
- [ ] Expressions: `_var_management_data` → IR `variables` + `Expr` on element attributes; writer emits `env[name] = value` and attribute expression strings; MAD-X `:=` knobs survive TraceWin → MAD-X → xtrack → MAD-X (gate: `psb.seq` knobs identical after the round trip).
- [ ] `Environment` JSON with several lines → IR `lines` (root = `env.lines` selected by name); `xt.load` for both shapes.
- [ ] Writer: `Bend.edge_entry_model`/`model` options exposed as write options; `RBend` for rectangular sources (default stays sector `Bend`).
- [ ] MAD-NG writer: `lattix convert --to madng` delegating to `xtrack.mad_writer.to_madng_sequence`; ledger `VIA_XTRACK`; reader deferred (needs a Lua parser or MAD-NG itself).
- [ ] Gate: every class in `xt.__dict__` that is a `BeamElement` appears in the reader table (test), PSB and `fodo.madx` blocks still 0.0 vs `from_madx_sequence`, knob round trip exact.

### 5.2 LightWin adapter and oracle (1 week)
- [ ] Env `lightwin` (conda python 3.12 + `pip install lightwin[cython]`); CI job on ubuntu (macOS if the wheel resolves).
- [ ] `lattix/oracles/lightwin.py`: run `Envelope3D` (and `Envelope1D` for the longitudinal block) through a `-I` worker; extract per-element transfer matrices from `SimulationOutput`; basis fingerprint (1 m drift, 1-D map cavity at −30°); mark `oracle_lightwin`.
- [ ] Lockstep test: LightWin's `.dat` parser vs lattix reader on every public deck and, nightly, the 724 private decks (lengths, element counts, quad gradients, field-map phases).
- [ ] Gate: `mebt+hwr.dat` HWR section, lattix→TraceWin writer output read by LightWin and by lattix agree to 1e-10 structurally; LightWin envelope vs HELIX transverse blocks within the Equivalent tier (2 %) and the reference energy within 0.5 %.

### 5.3 Cheetah adapter (1 week)
- [ ] Env `cheetah` (python 3.11, torch CPU, `cheetah-accelerator`); GPL: `lattix/formats/cheetah.py` is an adapter (imported at call time, absent from `FORMATS`, like HELIX).
- [ ] `to_segment(lattice, species)` / `from_segment(segment)`: table over the 22 kinds — Drift, Quadrupole (k1, tilt, misalignment), Sextupole, Bend/RBend (e1/e2, gap, fint), Solenoid (k = ks), RFCavity → `Cavity(voltage = −V/sign q, phase° , frequency)`, thin cavity as `length = 0` if supported (*verify*), Kicker → `HorizontalCorrector`+`VerticalCorrector` (or `CombinedCorrector`), Collimator → `Aperture`, Marker/Instrument → `Marker`/`BPM`/`Screen`, Taylor → `CustomTransferMap` (7×7 with the IR 6×6 embedded), FieldMap → `Cavity` via `FM_TO_CAVITY`, NCells/RFQCell → Drift (LOSSY), Patch/ReferenceChange/Freq/Directive → DROPPED with codes, Superposition → flattened.
- [ ] Oracle `lattix/oracles/cheetah.py`: `Segment.first_order_transfer_map(energy, species)` per element (7×7 → 6×6 + kick), fingerprint (τ late-positive expected), p0 follows cavities.
- [ ] Gate: `fodo.madx` vs cpymad T4×4 1e-8; MEBT quads + gaps vs HELIX Equivalent tier (Cheetah's cavity matrix differs from the thin-gap model by design; record `CHEETAH_CAVITY_MODEL`).

### 5.4 PyORBIT3 linac XML (1.5 weeks incl. build)
- [ ] Env `pyorbit` (conda py 3.10 + meson build of the clone); `lattix/formats/pyorbit_xml/` reader + writer with the §1.4 schema: sequences from `Line`s (or one sequence), `pos` = centre positions, QUAD field T/m, BEND theta/ea1/ea2/kls, RFGAP `E0TL` (GeV) + `phase` (deg) + `<Cavity>` grouping by `RFP.frequency`, DCH/DCV from Kicker, SOLENOID, VACWIN from Foil (LOSSY), MARKER for Instrument/Marker, apertures.
- [ ] Oracle: `LinacTrMatrixGenNode` at every node boundary (linac), `TEAPOT_MATRIX_Lattice.getRingMatrix` for MAD8 rings; fingerprint; species via `bunch.mass()`/charge.
- [ ] Gate: SNS MEBT (vendored, MIT) read → write fixed point; lattix `.dat` → XML → PyORBIT matrices vs HELIX within Equivalent tier (gap model), quads Exact.

### 5.5 IMPACT-T (1.5 weeks)
- [ ] `lattix/formats/impactt/` sharing `impactz` header/element machinery: `zedge` positioning (from `s_in`), types 0/1/3/4/5/104/105/110, `rfdata` Fourier coefficients generated from 1-D `Ez(z)` maps (port `RFcoeflcls.f90`), diagnostics as `bpm` (<0) types; reader inverse.
- [ ] Engine: conda-forge `impact-t` (pin) + `lume-impact` as the runner; oracle: bpm dumps (type −2) at element boundaries → probe maps as for IMPACT-Z; basis `(x, px/(mc), y, py/(mc), t, γ)` (*verify* against `fort.18`); marker `oracle_impactt`.
- [ ] Gate: `fodo.madx` T4×4 vs cpymad 1e-8 (quads/drifts), MEBT gap section vs IMPACT-Z within 1e-6 (same Fourier map), `Sample1` (SolRF) reads and re-writes bit-exact.

### 5.6 Bmad bridge (3 days)
- [ ] `lattix convert --to astra|gpt|opal|sad|csrtrack|merlin|slicktrack` = IR → `.bmad` in a temp dir → `bmad_to_astra` / `bmad_to_gpt` (namelist inputs generated by lattix) / Tao `write opal|sad` in env `bmad`; ledger `EQUIVALENT VIA_BMAD` per element plus Bmad's own conversion notes captured from stdout.
- [ ] `lattix convert --from at|sad|sxf|ptc` = converter → `.bmad` → IR (`VIA_BMAD` on read).
- [ ] Tests: structural (element counts, lengths, total angle) and Bmad-consistency (the produced `.bmad` re-read equals the IR); no target engine here.

### 5.7 Ocelot (1 week, electron machines)
- [ ] Writer for the Python lattice file (plain text, no dependency): element constructors with the §1.7 argument names, `v` in GV, `phi` in deg, `gap = 2·hgap`; reader by AST of literal constructor calls, falling back to importing ocelot for expressions; adapter `to_magnetic_lattice`/`from_magnetic_lattice` at call time (GPL).
- [ ] Guard: writing a non-electron reference records `LOSSY OCELOT_ELECTRON_ONLY` (its cavity and energy terms assume m_e); normalized transverse strengths stay exact.
- [ ] Oracle: `lattice_transfer_map` per element (electron beams); FLASH/XFEL `.lte` lockstep through Ocelot's own Elegant adaptor.

### 5.8 DYNAC (1.5 weeks; needs `brew install gcc`)
- [ ] Writer for the tw2dyn subset and beyond: DRIFT, QUADRUPO, SOLENO, BMAGNET (+ pole faces, *verify* card layout in the V6 user guide PDF under `help/`), CAVSC (thin gap with TTF polynomial), BUNCHER, CAVNUM + FIELD (1-D Ez maps), NEWF/HARM, STEER, RFQPTQ (from RFQCell when parameters exist), STRIPPER (Foil), CHANGREF/NREF; INPUT/GEBEAM from the reference particle. Reader for the same cards.
- [ ] Lockstep: our TraceWin reader → DYNAC writer vs `tw2dyn` on the SNS MEBT+DTL1 deck (same cards, same numbers).
- [ ] Oracle: build DYNAC, run with `FIRORD`, parse the printed first-order matrices; marker `oracle_dynac`, local only (EULA): nothing from `datafiles/` enters the repo.

### 5.9 Synergia2 (1 week, when a Booster study needs it)
- [ ] pixi install of the clone (osx-arm64); reader/writer for the JSON lattice (schema from `as_json` of the MAD-X-loaded MI lattice); MAD-X path already exists.
- [ ] Oracle: `Lattice_simulator.get_linear_one_turn_map` / `tune_linear_lattice` on the Booster MAD-X (`ring_code/MAD-X_BOOSTER`) vs cpymad/xtrack one-turn maps.

### 5.10 OPAL-T writer (1 week)
- [ ] `lattix/formats/opal/` writer: `ELEMEDGE` positions from `s_in`, RFCAVITY `VOLT` MV / `FREQ` MHz / `LAG` rad (phase convention pinned against Tao's `write opal` output, then against OPAL itself when an engine exists), QUADRUPOLE `K1`, SBEND/RBEND, SOLENOID `KS`, KICKER, collimators, MONITOR/MARKER; field maps via `FMAPFN` for FieldMap elements (1DDynamic/T7 formats, *verify*).
- [ ] Tests: structural + Tao `write opal` comparison on `fodo.bmad`; reader and engine deferred (OPAL build needs MPI + H5Hut; consider the OPAL Docker image later).

### 5.11 Cross-cutting
- [ ] CI: new envs `lightwin` (py 3.12), `cheetah` (torch CPU), `pyorbit` (meson build) as separate jobs; `impact-t` added to `environment-ci.yml` (pinned); GPL packages are never imported at module level and never vendored; Bmad bridge tests run in the existing bmad env.
- [ ] Docs: one page per new format under `docs/formats/`, `docs/oracles.md` sections for each new engine with measured fingerprints, conventions table rows.
- [ ] Ledger codes: every new writer covers all 22 kinds (`check_rules_coverage`); new codes listed by the catalogue automatically.

## 4. Gates (Phase 5 acceptance)

| # | Deck → target | Oracles | Gate |
|---|---|---|---|
| B1 | `psb.seq` (knobs) → xtrack → MAD-X | xtrack vs cpymad | all blocks 0.0 at 304 boundaries **and** every `:=` knob preserved by name |
| B2 | `mebt+hwr.dat` HWR section → `.dat` (LightWin parse) | LightWin `Envelope3D` vs HELIX | T4×4 within 2 %, reference energy 0.5 %, structure exact |
| B3 | `fodo.madx` → Cheetah `Segment` | Cheetah vs cpymad | T4×4 1e-8, drift fingerprint recorded |
| B4 | SNS MEBT XML → IR → `.dat` | PyORBIT `LinacTrMatrixGenNode` vs HELIX | quads Exact 1e-8, gaps Equivalent 2 % |
| B5 | `mebt.dat` → `ImpactT.in` | IMPACT-T probe vs IMPACT-Z | 1e-6 on the transverse blocks |
| B6 | `fodo.bmad` → Astra, GPT, OPAL-T (via Bmad) | structural + Bmad re-read | element counts, lengths, Σangle exact; `VIA_BMAD` entries present |

## 5. Risks
* GPL packages (Cheetah, Ocelot, OPAL) stay optional adapters; a wheel failing to install must skip, never fail (the `MissingDependencyError` path).
* Cheetah's cavity model is Rosenzweig–Serafini, not the thin-gap model: Equivalent tier only.
* IMPACT-T is time-domain: element-boundary maps come from bpm dumps and are Equivalent tier where space charge or the time step blur the boundary.
* DYNAC's EULA forbids redistribution: local corpus and nightly only.
* OPAL has no engine here; the writer is validated against Bmad's writer, which is itself a model.
* Ocelot is electron-only: record it, do not silently scale.

## 6. Order and effort (~10 weeks)
5.1 xsuite (1) → 5.2 LightWin (1) → 5.3 Cheetah (1) → 5.4 PyORBIT3 (1.5) → 5.5 IMPACT-T (1.5) → 5.6 Bmad bridge (0.6) → 5.7 Ocelot (1) → 5.8 DYNAC (1.5) → 5.9 Synergia2 (1) → 5.10 OPAL-T (1).
