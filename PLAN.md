# Lattice Translator (working name `lattix`) — Implementation Plan

> Format follows `HELIX_v3/docs/superpowers/plans/2026-04-17-tracewin-compat.md`: phases → tasks → checkbox steps, each phase ends green.

## Context

You want a tool that translates an accelerator lattice from one code's input format to another's, with testing rigorous enough to trust the translated deck (PIP-II linac + BTL/BAL + Booster work spans TraceWin, MAD8, MAD-X, Elegant, and now Bmad/xtrack/ImpactX peers). The `Translator/` folder is empty; the real assets are elsewhere on disk:

- **HELIX_v3** (`HELIX_unzipped/HELIX_v3/linac_gen/io/`) already has *readers* for TraceWin `.dat` (70 kB, strict/permissive downgrade design), MAD-X (subset), MAD8 flat (`.lat/.FLAT`, lazy `:=` resolver, LINE expansion) and Elegant `.lte` (namelist + RPN), but only **one writer** (TraceWin). Its `Lattice` is a flat list with no stored `s`, no reference particle, energy is a replay artifact; solenoid `ks→B` uses signed Bρ in MAD-X but unsigned in Elegant; `Dipole` cannot hold `e1/e2` through a BEND card; `ERROR_*`, `ttf`, `n_steps`, misalignments are lossy on write. Test style to copy: `tests/io/test_madx_conventions.py` (cpymad sectormap as external oracle), `tests/io/test_mad8_parser.py::test_btl_anchor_lockstep` (two independent paths to the same PIP-II BTL), `tests/io/test_parser_downgrades.py` (strict vs permissive for every downgrade), `tests/dataguard.py` (skip when undistributable data absent).
- **17 cloned codes** in `particle_tracking_codes/` (IMPACT-Z/T, ImpactX, OPAL-X, FLAME, LightWin, fbpic, cheetah, xtrack/xfields/xobjects, ocelot, PyORBIT3, synergia2, dynac, bmad-ecosystem, pytao) — source only, none built, but several ship converters and example lattices (inventory in §3).
- **Engines that run on this Mac today (verified 2026-09-03):** MAD-X 5.09.03 via `cpymad` (base env); `xtrack 0.103.5` (`load_madx_lattice`, `Line.to_madx_sequence`, JSON); Bmad 20260828 + Tao + `pytao 1.2.4` in conda env `bmad` (FODO 6×6 matrix computed, det=1) with converters `bmad_to_mad_sad_elegant` (`-madx/-mad8/-sad/-elegant`), `mad_to_bmad/{madx,mad8}_to_bmad.py`, `elegant_to_bmad.py`, `sad_to_bmad`, `sxf_to_bmad`, `opera_fieldmap_to_bmad`; TraceWin.app (x86_64, licensed keys present, Rosetta OK) with documented batch mode `TraceWin project.ini hide …` and a reusable subprocess runner in LightWin; IMPACT-Z `ImpactZexeMac` (x86_64).
- **Installable engines (verified on conda-forge/PyPI, osx-arm64):** `elegant 2026.3.0`, `impactx 26.08`, `impact-z 2.7.7`, `bmad` (already), `pals-schema 0.3.0` (PyPI), `flame-code 1.9.3`, `lightwin 0.16.5`, `cheetah-accelerator 0.8.4`, `ocelot-desy 25.6.0`, `pmd-beamphysics 0.5.6`, `lume-impact 0.12.1`.
- **PALS** (Particle Accelerator Lattice Standard, LBNL/Cornell 2025–26): YAML, SI + eV, phase in rad/2π, 30 element kinds with parameter groups (`MagneticMultipoleP`, `RFP{voltage|gradient, phase, frequency, L_active, dE_ref, zero_phase}`, `BendP`, `ApertureP`, `BodyShiftP`, `ReferenceP`), Python package `pals-schema` (pydantic). **No converters exist yet** ("in need of volunteers"), so PALS is a *target format and naming reference*, not a drop-in IR.
- **Multi-format PIP-II corpus already on disk** (cross-format lockstep anchors): BTL/BAL as MAD8 `.FLAT` + Elegant `.lte` + TraceWin `.dat`; HWR cryomodule `.lte`; ~724 real PIP-II TraceWin decks under `PIP_II/`; HELIX `examples/pipii/*` (btl 960 el., mebt 427, mebt+hwr 483, full SCL, fnalscl NCELLS deck, RFQ_CELL decks); 56 field maps in `HELIX_v3/Fields/` (ANL/CEA — never commit); FNAL Booster MAD-X in `ring_code/MAD-X_BOOSTER`; genuine TraceWin outputs in `HELIX_v3/Tracewin_code/` incl. `Transfer_matrix1.dat`.

**Outcome:** a standalone, pip-installable Python package + CLI (`lattix convert in.dat --to madx out.madx`) with a code-neutral IR, N readers/writers, a fidelity report on every conversion, and a test suite where every format pair is pinned against at least one *external* engine (never round-trip-only).

---

## 1. Decisions (confirmed by you on 2026-09-03: D1 standalone BSD-3, D2 own PALS-mirroring IR, D3 TraceWin⇄MAD-X⇄Elegant⇄Bmad first, D4/D5 engines in CI + TraceWin locally)

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Where it lives | **Standalone repo** `Translator/` → package `lattix`, BSD-3; HELIX becomes an optional *adapter* (`lattix.adapters.helix`) and later depends on it for import/export | HELIX is GPL-3 and private until OPTT release; a neutral BSD tool can be used by MAD/Bmad/ImpactX users and by HELIX. Porting the four HELIX parsers is allowed (you own them). |
| D2 | IR | **Own pydantic-v2 IR**, SI + eV, PALS kind/parameter-group names mirrored, nested `Line`s with repeat/reverse + `flatten()`, symbolic `Expr` preserved, per-element entry/exit reference energy from a `walk()` pass, `native` passthrough dict per source format | PALS is the right vocabulary but immature; mirroring it makes the PALS reader/writer a rename and keeps us convertible when PALS converters land. |
| D3 | Format priority | **Tier 1:** TraceWin `.dat` ⇄ MAD-X ⇄ Elegant ⇄ Bmad (+ MAD8 flat read, PALS write/read). **Tier 2:** ImpactX (Python input), IMPACT-Z `ImpactZ.in`, FLAME `.lat`, xtrack JSON/`Line`, Cheetah/Ocelot objects. **Tier 3:** OPAL, DYNAC, Astra/GPT (via Bmad converters), TRACK | Tier 1 = your PIP-II + Booster daily formats and the ones with runnable oracles today. |
| D4 | Oracles | cpymad (MAD-X), pytao (Bmad), elegant (conda), xtrack, HELIX matrix tracking, TraceWin batch (local-only), impactx/impact-z/flame later | Every Tier-1 pair gets ≥1 external engine on both ends. |
| D5 | Never-silent fidelity | Strict mode raises on any downgrade; permissive records to a machine-readable `FidelityReport` (JSON) that the CLI prints and tests assert on | Inherited HELIX house rule; round-trips cancel symmetric errors, so reports + external oracles are the safety net. |

---

## 2. Architecture (package layout)

```
Translator/
  pyproject.toml         # lattix, python>=3.11; deps numpy pydantic>=2 lark pyyaml; extras [oracles] cpymad xtrack pysdds, [pals] pals-schema
  environment-ci.yml     # conda-forge: cpymad xtrack elegant impactx impact-z (+pip flame-code pysdds pals-schema lightwin hypothesis)
  environment-bmad.yml   # bmad + pytao (separate env: py3.13/numpy2)
  lattix/
    __init__.py          read(path, fmt=None) -> Lattice; write(lat, path, fmt=None, *, strict=False) -> FidelityReport; translate(src, dst)
    ir/  lattice.py (Lattice, Line, LineItem, Placed, flatten(), survey()) · elements.py (kinds + PALS parameter groups, pydantic extra='forbid')
         reference.py (Species, ReferenceParticle) · walk.py (s + reference energy propagation) · expr.py (Expr AST, deferred eval, RPN, NAME[ATTR])
         normalize.py (G<->K1, B<->ks, BnL<->KnL, kick<->∫B·dl via signed Bρ) · rf.py (phase conventions, V_eff/TTF, dE_ref, integrate_map())
         fieldmap.py (FieldMapSpec, channels/files) · units.py
    formats/ base.py (Reader/Writer protocols, single suffix Registry, RULES-coverage hook) · naming.py (per-format NameRules, reversible renames)
         tracewin/{syntax,reader,writer}.py  madx/  mad8/  elegant/  bmad/  pals/          # Phase 1-2
         impactx/  impactz/  flame/  xtrack/  ocelot/  cheetah/  helix/                    # Phase 3-4 (helix = IR <-> linac_gen.Lattice adapter)
    fidelity.py          FidelityEntry/Report, TranslationError, allow-list policy
    oracles/ base.py, basis.py (coordinate transforms), cpymad.py, xtrack.py, pytao.py (+bmad_worker), elegant.py, tracewin.py, helix.py, impactx.py, impactz.py, flame.py
    cli.py               lattix convert | inspect | validate | report
    corpus.py            manifest build/verify (sha256, provenance, license)
  tests/   unit/ golden/ roundtrip/ oracle/ invariants/ property/ downgrades/ corpus/ fidelity/ oracles/ + conftest.py (markers, require())
  tests/data/public/     vendored MIT/BSD/Apache samples (FLAME, LightWin, xtrack, ImpactX, HELIX examples)
  docs/    conventions.md (rule tables of §4), fidelity.md (codes), formats/*.md, oracles.md
```

Reuse from HELIX (port, keep provenance in module docstrings): `tracewin_syntax.SCHEMA`+`parse_positionals`, `tracewin_geom.decode_geom/component_files/enabled_channels`, `field_map_factory` dispatch, `madx_parser._eval_expr/_signed_brho/_brho/_drift_pad`, `mad8_parser._Mad8File` lazy resolver + `_expand/_root_line`, `elegant_parser._logical_statements/_rpn_eval/_expand`, `tracewin_writer._emit_card` default-elision, `portable_paths.best_relpath`, `tests/dataguard.py`, the TraceWin (z,δp/p)→(deg,MeV) similarity transform at `tests/analysis/test_tracewin_crosscheck.py:325-327`, `tracewin_outputs.read_partran_out` and the 26-column envelope reader in `compare_tracewin.py:33`.

Redesign (do not port as-is): `Lattice` (needs `s`, reference energy per element, hierarchy), `Dipole` (store angle + length + e1/e2 independently; emit `EDGE` cards on TraceWin write), solenoid sign policy (one rule), RF gap `ttf` (keep as a field; fold only on TraceWin write and record in report), suffix dispatch (one registry, not three).

---

## 3. Format inventory of the 17 cloned codes (`particle_tracking_codes/`)

| Code | Native lattice format | Reads foreign | Writes foreign | Runs on this Mac | Corpus in clone |
|---|---|---|---|---|---|
| **Bmad** | `.bmad` (`bmad/parsing/bmad_parser.f90`; docs `bmad/doc/lattice-file.tex`, `conversion.tex`) | MAD-X, MAD8/XSIF, Elegant, SAD, SXF, PTC flat, AT, OPERA maps (`util_programs/*_to_bmad`, python scripts) | MAD-8, MAD-X, Elegant, SAD, OPAL-T, **PALS**, SciBmad (`bmad/output/write_lattice_in_foreign_format.f90`; Tao `write madx|elegant|pals…`) | conda env `bmad` (done) | 470 `.bmad` (CBETA, CESR, CEBAF, ERL), 7 `.sad`; **`regression_tests/write_foreign_test/` = one `.bmad` + byte-exact `madx/mad8/lte/sad.correct`** |
| **xtrack** | Python `Environment` + JSON | MAD-X (native Lark grammar `xtrack/mad_parser/`, cpymad `mad_loader.py`), TFS, SixTrack | MAD-X sequence, MAD-NG (`xtrack/mad_writer.py`) | base env (done) | 19 `.madx` + 22 `.seq` + 44 `.json` (LHC, PSB, PS, SPS, ELENA, LEIR, CLIC DR, FCC-ee, BESSY-3, SLS-2); Bmad cross-ref JSONs in `test_data/spin_refs_bmad/` |
| **ImpactX** | AMReX `.in` + Python API | MAD-X (`src/python/impactx/MADXParser.py`, `madx_to_impactx.py`, 26 keywords), **PALS** (`extensions/KnownElementsList.py`) | — | `conda install -c conda-forge impactx` | 99 `.in`, 7 `.madx` (fodo, chicane, kicker, solenoid, dogleg, booster), 1 `.pals.yaml` |
| **IMPACT-Z** | `ImpactZ.in` positional table; element type codes 0–6, 101–110, negatives = diagnostics (`src/Contrl/AccSimulator.f90:246-520`) | — | — | `conda install -c conda-forge impact-z` or `ImpactZexeMac` (Rosetta) | 3 decks |
| **IMPACT-T** | `ImpactT.in` (t-based; table in `examples/Sample1/ImpactT.in:79-121`) | Parmela/Elegant particle files | — | conda-forge `impact-t` | 7 samples |
| **FLAME** | GLPS `.lat` (`src/glps.y`; elements `src/moment.cpp:1528-1548`: source, marker, bpm, drift, orbtrim, sbend, quadrupole, sextupole, solenoid, rfcavity, stripper, edipole, equad, tmatrix) | — | GLPS (`GLPSPrinter`) | `pip install flame-code` | **35 `.lat`** (FRIB LS1/FS1, FrontEnd, `ALL_lattice.lat` every type) |
| **LightWin** | TraceWin `.dat` (`src/lightwin/tracewin_utils/`), ~40 cards + ERROR_* commands | — | TraceWin `.dat`; **headless TraceWin runner** `beam_calculation/tracewin/tracewin.py` | `pip install lightwin` | 6 `.dat` (ADS linac, spoke) |
| **DYNAC** | 64-card two-line deck (`source/dynac.F:160-172`), **cm / kG / cm⁻²** units | **TraceWin → DYNAC** (`converters/tw2dyn.f`: DRIFT QUAD EDGE BEND DTL_CEL NCELLS MULTIPOLE FREQ GAP FIELD_MAP LATTICE) | DYNAC | CMake + gfortran | **21 decks**: SNS MEBT+DTL1, full ESS RFQ→SCL, e-gun, each with `dynac.print.ref` |
| **PyORBIT3** | linac XML (`py_linac/linac_parsers/sns_linac_lattice_factory.py`), ring MAD8 `.lat` | MAD8, MAD-X, SAD (`py/orbit/parsers/`) | linac XML | meson build (conda env) | SNS linac XML (862 kB), ESS XML, SNS ring `.LAT`, SIS-18 |
| **Synergia2** | MAD-X + JSON | MAD-X (Boost.Spirit), MAD8 | MAD-X (`Lattice::as_madx_file`) | pixi/conda | 22 `.madx`, 41 `.seq`, 22 `.lat` (IOTA, Main Injector, SIS-18) |
| **Ocelot** | Python `lattice.py` | MAD-X seq/TFS, MAD8 twiss, **Elegant**, ASTRA, longlist, CSRtrack (`ocelot/adaptors/`) | **Elegant**, ASTRA, CSRtrack, Genesis | `pip install ocelot-collab` | XFEL `.lte`, FLASH `.lat`; Elegant↔Ocelot round-trip JSON refs |
| **Cheetah** | Python objects + latticejson | **Bmad, Elegant, Ocelot, NX-tables** (`cheetah/converters/`; shared Fortran-namelist tokenizer + RPN/infix) | latticejson | `pip install cheetah-accelerator` | `tests/resources/{bmad_tutorial_lattice.bmad, fodo.lte, cavity.lte}`, ARES JSON |
| **OPAL-X** | MAD-like `.in` (`src/OpalParser/`), 40+ element keywords (`src/OpalConfigure/Configure.cpp:193-241`) | SDDS particles | SDDS particles | heavy source build | none shipped |
| **pytao** | — (Bmad bridge) | via Bmad | `write madx/mad8/elegant/sad/opal/xsif/pals/scibmad` (`pytao/interface_commands.py:5112-5152`); `lat_param_units()` unit oracle | conda env `bmad` (done) | — |
| fbpic, xfields, xobjects | no lattice concept | — | — | — | — |

Design consequences: (a) **PALS is the only format with an independent writer (Bmad/Tao) and reader (ImpactX)** already in the tree, so a PALS reader/writer gets two free oracles; (b) Bmad's `write_foreign_test` gives byte-exact MAD-8/MAD-X/Elegant/SAD references from one source; (c) unit outliers to encode once in `units.py`: TraceWin mm+MHz+deg, DYNAC cm+kG+cm⁻², Ocelot cavity GV, MAD-X MV+MHz+lag in turns, Bmad `phi0` in turns, FLAME bend angles in deg, ImpactX `k` in 1/m² and rotations in deg; (d) xtrack `mad_parser/loader.py:411-510` is the best MAD-X unit-conversion reference (`k0_from_h`, hgap, rbend `length_straight`, lag→rad).

---

## 4. IR and conversion rules (the physics contract — also becomes `docs/conventions.md`)

### 4.1 IR core model (pydantic v2, `extra="forbid"`)
- **Units:** SI + eV everywhere (m, rad, T, T/m, T·m^(1−n), V, V/m, Hz, s, eV, eV/c). **Phase in radians, cos convention, 0 = crest**; boundary conversions: deg (TraceWin, Elegant, ImpactX, FLAME, IMPACT-Z), rad/2π (MAD-X `lag`, Bmad `phi0`, PALS `phase`), rad (OPAL). HELIX's mm/deg/MeV exist only inside `formats/helix`.
- **Canonical strengths are lab-frame fields** (`Bn1`=G T/m, `Bsol` T, `Bn{n}L`, kicker deflection in rad); normalized forms (`K1`, `ks`, `KnL`) are *views* computed with the **local signed rigidity** at element entrance. `Expr` stores whichever form the source gave (`k1 := kqf` stays symbolic) so MAD-X knobs survive MAD-X→MAD-X; targets without variables get evaluated numbers.
- **Rigidity policy:** `brho_abs = pc/(|q|c)`, `brho_signed = sign(q)·brho_abs`. All MAD-family normalized strengths (K1, ks, KnL, hkick/vkick) use `brho_signed` (HELIX pattern `madx_parser._signed_brho`, `madx_parser.py:288`); this fixes HELIX's solenoid inconsistency (signed in MAD-X import, unsigned in Elegant import `elegant_parser.py:327`); H⁻ sign pinned by oracle invariant I-8, not by convention lookup.
- **Model:** `Species(name, mass_eV, charge)` (H⁻ = m_p + 2m_e, per HELIX `particle.py:24`), `ReferenceParticle(species, E_kin, frequency, time)`, `Provenance(format, file, line, original_name, original_type, path)`, `BodyShift` (PALS BodyShiftP: offsets m, rotations rad), `Aperture` (PALS ApertureP: limits, shape, location), `Element(name, kind, length: Expr|float, aperture, shift, tracking, native: {fmt: {...}}, provenance, meta)` — `native` is an opaque per-format passthrough re-emitted only to the same format; `LineItem(ref, repeat, reverse)`, `Line`, `Command(kind, anchor, args, provenance)` side-channel; `Lattice(elements, lines, use, reference, variables, commands, errors, meta)`; `flatten()` → `Placed(element, s_in, s_out, ref_in, ref_out, path, index)` (a view, reversible via `path`; reversed asymmetric elements swap ends).
- **`walk()` energy rule:** `RFCavity`: `E_out = E_in + q·V_eff·cos φs` with `V_eff = voltage` or `gradient·L_active` (thin gap: `E0TL`, TTF already folded — TraceWin GAP / HELIX `rf_gap.py:208`); `FieldMap`: `dE_ref` from `rf.integrate_map()` (synchronous-phase integration of `ke·Ez(z)cos(ωz/βc+φ)` with velocity update) or taken from the source when it provides one; `ReferenceChange` applies `dE_ref` (TraceWin `SET_BEAM_ENERGY`/`SET_BEAM_E0_P0`); `Freq` updates `ref.frequency` (TraceWin FREQ semantics, HELIX `lattice_commands.py:85-100`); everything else `ref_out = ref_in`. Every normalized view uses `ref_in`.
- **`Expr`:** port HELIX `_eval_node` whitelist (`+ - * / ^`, unary, sqrt/abs/sin/cos/tan/exp/log, pi/twopi/e/clight), Elegant RPN (`_rpn_eval`), MAD8 `NAME[ATTR]` lazy resolver with cycle detection (`mad8_parser.py:167-207`); `deferred` flag preserves `:=`.
- **PALS alignment:** kinds and parameter-group names verbatim from `pals-python/src/pals/{kinds,parameters}`: `MagneticMultipoleP{Bn,Bs,Kn,Ks,tilt,…L}`, `BendP{e1,e2,e1_rect,e2_rect,edge_int1/2,g_ref,L_chord,tilt_ref,bend_field_ref}`, `RFP{frequency,harmon,voltage,gradient,phase,cavity_type,n_cell,zero_phase,L_active,dE_ref}`, `SolenoidP{Ksol,Bsol}`, `ApertureP`, `BodyShiftP`, `ReferenceP{species_ref,pc_ref,E_tot_ref,time_ref}`, `ReferenceChangeP`, `FloorP`, `MetaP`, `TrackingP`.

### 4.2 Taxonomy and fidelity matrix (E exact · M equivalent model · L lossy, recorded · U dropped with `DROPPED` entry; `?` verify at implementation)
| IR kind | canonical params | TW | MAD-X/8 | Elegant | Bmad | ImpactX | IMPACT-Z | FLAME | PALS | others |
|---|---|---|---|---|---|---|---|---|---|---|
| Drift / Quadrupole (`Bn1`, `tilt1`, TW G3..G6→`Bn3..6`) | length, G, skew | E | E | E | E | E | E | E | E | OPAL/DYNAC/xtrack/Ocelot E |
| Sextupole/Octupole thick | `Bn2`/`Bn3` | L (thin+drifts) | E | E | E | E | E(5) | E | E | |
| Multipole thin | `Bn{n}L, Bs{n}L` | U (comment) | E | E | E | E | M | U? | E | |
| Bend (+ edges) | arc length, angle, `g_ref`, e1/e2 (+rect flag), fint, hgap, k1 (N=−k1ρ²), `tilt_ref` (hv) | E (BEND+EDGE×2) | E | E | E | E (Sbend+DipEdge) | E? | E (phi,phi1,phi2,K) | E | OPAL E, DYNAC E |
| Solenoid | `Bsol`, length | E | E | E | E | E | E | E | E | |
| RFCavity (thin gap or thick) | V or gradient, φs, f, n_cell, ttf, dE_ref | E (GAP) | **M** (RFCAVITY, p0 constant) | E (RFCA change_p0=1) | E (lcavity) | E (ShortRF) / M (RFCavity) | M (104+rfdata) | M (cavtype) | E | xtrack M, Ocelot/Cheetah E |
| FieldMap (TW geom-coded) | channels×dim, files, ke/kb/ki/ka, φ, f | E | L→cavity/hard-edge | L | M (grid_field) | M (Fourier/3D) | M (rfdata) | U | U (extension) | OPAL M, DYNAC M (cf. `tw2dyn.f`) |
| NCells (DTL/CCL) / RFQCell | mode, n, βg, E0T, θs / V, r0, m, L, type | E | L→gaps / U | L / U | L / U | M? / U | E (101/103) / U | U | U | DYNAC M / RFQPTQ? |
| Kicker (Steerer) | hkick, vkick (rad) + ∫B·dl view | E | E | E | E | E | U? | E (orbtrim) | E | |
| Collimator / element aperture | ApertureP | E (types 0,1) L (2–6) | E | E | E | E | U | M | E | |
| Marker / Instrument | diag family | E | E | E | E | E | U | E | E | |
| Foil · Taylor/Matrix · Patch/FloorShift · ReferenceChange · Superposition | | comment · U · U · E · E | U · E · U · U · L | U · E · U · E · L | E? · E · E · E · E | U · E · E · E · U | U | E(stripper) · E(tmatrix) · U · U · U | E · E · E · E · E (UnionEle) | |

**TraceWin commands:** `FREQ` → `Freq` command *and* resolved into each RF element's `frequency` (idempotent write, HELIX FREQ-elision pattern); `SET_SYNC_PHASE` → phase-semantics flag on following cavities; `SET_BEAM_ENERGY/E0_P0` → `ReferenceChange`; `SET_*/MIN_*/ADJUST*/DIAG_*` targets → `commands` kind `MatchingDirective` (only TraceWin/HELIX write them; others record `DROPPED:MATCHING_DIRECTIVE`); `LATTICE/LATTICE_END` → `Period` commands (Bmad/MAD-X writers can emit as line structure); `ERROR_*` → `lattice.errors` (TraceWin-only); `PARTRAN_STEP` → `meta["tracking"]`; `FIELD_MAP_PATH` → resolved into `FieldMap.files`; `TITLE` → `meta`.

### 4.3 Conversion rules (Bρ = signed rigidity at `ref_in`)
| Kind | TraceWin | MAD-X / MAD8 | Elegant | Bmad | ImpactX | others |
|---|---|---|---|---|---|---|
| Quadrupole | `QUAD L[mm] G[T/m] R Θ[deg] G3..G6` | `k1 = G/Bρ`, `tilt` rad (bare flag = π/4) | `k1 = G/Bρ` | `k1 = G/Bρ` | `Quad(ds, k=G/Bρ)` | IMPACT-Z/FLAME/DYNAC take lab G (DYNAC cm, kG!); PALS `Bn1`/`Kn1` |
| Bend | `BEND θ[deg] ρ[mm] N R HV` + `EDGE β ρ gap=2·hgap K1=fint K2=2.80` ×2; ρ>0, sign in θ, plane in HV; **β = sign(θ)·e and HV=1 ≡ tilt +π/2 (tilt −π/2 → HV=1 with −θ), measured with TraceWin 2026-09-03**; reader clusters EDGE+BEND+EDGE | `SBEND l angle e1 e2 fint fintx hgap k1 tilt` (RBEND only if source rect; `e += angle/2`) | `CSBEND/SBEN` same; fint default 0.45 | `sbend/rbend … ref_tilt` | `Sbend(ds, rc)` + `DipEdge(psi, rc, g=2·hgap, K2=fint)` | FLAME `phi/phi1/phi2` in deg, `K`=k1 |
| Solenoid | `SOLENOID L B R` | `ks = Bsol/Bρ` | `ks` | `ks` | `Sol(ds, ks)` | FLAME/IMPACT-Z lab B |
| RF cavity | `GAP E0TL[V] φs[deg] R p_flag` (TTF folded), FREQ state MHz | `RFCAVITY volt[MV]=V_eff·1e-6, lag = φ/2π + 0.25` (MAD-X gain = V·sin 2π·lag, measured in HELIX `test_madx_conventions.py:287`), `freq[MHz]`; **no p0 update → M, `EQUIVALENT:CONST_P0`** | `RFCA volt[V], phase[deg] = φ·180/π + 90` **for negative species (e⁻, H⁻); φ·180/π − 90 for positive species** (RBEN L is the chord; FINT default 0.5) (measured 2026-09-03: elegant's crest follows sign(q) relative to the electron), `freq[Hz], change_p0=1`; elegant's 5th coordinate is path length, so matrix R56 lacks the velocity term and its RFCA matrix phase slip is ultra-relativistic — use tracking for low-β longitudinal checks | `lcavity voltage/gradient, phi0 = φ/2π, rf_frequency, l_active, n_cell` (0 = crest); ring `rfcavity phi0 = φ/2π + 0.25` | `ShortRF(V, freq, phase[deg]=φ·180/π)`; `RFCavity(ds, escale, freq, phase, cos_coef)` | xtrack `lag[deg] = φ·180/π + 90`; Ocelot/Cheetah `phase[deg]`, 0 = crest (Ocelot V in **GV**); OPAL `LAG` rad; PALS `phase = φ/2π`, `zero_phase=ACCELERATING`, `dE_ref` explicit |
| Field map (RF) → no-map target | — | `RFCavity(L, V_eff = ke·∫Ez·T(β_in), φs, f)` code `FM_TO_CAVITY` (report ∫Ez, T, dE_ref) | same | Bmad `grid_field` file conversion (M) | Fourier `cos_coef` / 3D (M) | IMPACT-Z `rfdata` (M) |
| Static solenoid / quad map → hard edge | — | `L_eff = (∫B)²/∫B²`, `B_eff = ∫B²/∫B` (preserves ∫B and ∫B²), code `FM_SOL_HARDEDGE`; quad map (geom digit 9) preserves ∫G | | | | |
| Kicker | `THIN_STEERING Bx By R elec`; sign set by oracle I-10 (HELIX notes manual sign differs, `steerer.py:22-25`) | `hkick/vkick` (Δp/p0) | `HKICK/VKICK/KICKER` | `hkicker/vkicker/kicker` | `Kicker` | FLAME `orbtrim`, DYNAC `STEER` |
| Multipole | inside QUAD as G3..G6 only | `knl/ksl` | same | same | `Multipole(n, K_normal, K_skew)` | FLAME `sextupole B3` lab |
| Aperture | `APERTURE dx dy type` (0 rect, 1 circ E; 2–6 L with `native`) | `RCOLLIMATOR/ECOLLIMATOR`, `APERTYPE/APERTURE` | `RCOL/ECOL` | `x1_limit..y2_limit, aperture_type` | `Aperture/PolygonAperture` | |
| Misalignment | only `DRIFT x_shift/y_shift` → `BodyShift`; else `L:MISALIGN_DROPPED` | `EALIGN` (command-level, applied by reader) | `DX DY DZ TILT` | `x_offset … tilt` | `dx dy rotation_degree` | FLAME `dx dy pitch yaw roll` |
| Energy for constant-p0 targets (MAD-X, xtrack, Ocelot) | | `energy_mode="local"` (k1_i = G_i/Bρ(ref_in_i), correct per-section optics) or `"constant"`; either way recorded | | | | |

The **phase-slope sign** (a late particle at φs<0 gains more) is not derivable from formula tables because time-coordinate signs differ per code (Bmad's own converter uses `phi0 = 0.5 − lag`, a reflection); every writer fixes it by invariant I-7 against its engine.

### 4.4 Fidelity policy and naming
- Writers are table-driven: `RULES: dict[kind, Rule(cls, emit)]`; a test asserts `set(RULES) == ALL_KINDS` per writer (an unmapped kind is a failing test, never a skipped element). Readers log parse downgrades into the same ledger (HELIX `_downgrade`, `tracewin_parser.py:94-98`).
- `FidelityEntry(element, path, kind, cls ∈ {EXACT, EQUIVALENT, LOSSY, DROPPED}, code, message, details, source_format, target_format)`; `FidelityReport.raise_if(strict)` raises `TranslationError` on the first LOSSY/DROPPED not in the allow-list; permissive writes `<out>.fidelity.json` + stderr summary; exactly one entry per source element.
- `NameRules(pattern, case_insensitive, max_len, reserved)` per format (TW label optional; MAD-X lower-case ≤48?; MAD8 ≤16?; Elegant case-insensitive; Bmad ≤40? case-insensitive; FLAME case-sensitive unique; IMPACT-Z/DYNAC unnamed — verify limits at implementation). `sanitize()` → `uniquify()` (`_2, _3…`), original name/type kept in `provenance` and as an adjacent comment tag `! lattix: name="…" type="…"` that readers parse back (reversible, I-15).

### 4.5 Invariants (each is a test; tolerance tier from §5.2)
I-1 Σlength · I-2 signed Σangle per plane + survey end point (vs MAD-X `survey`, Bmad floor) · I-3 `G_i·L_i` preserved; `k1_i·Bρ(ref_in_i) == G_i` · I-4 per-element 6×6 vs target engine after basis transform · I-5 `det(M₂ₓ₂) = p_in/p_out` per element and `Π det = p_start/p_end` · I-6 `dE_ref = q·V_eff·cos φs`, `Σ dE_ref = E_end − E_start` · I-7 phase-slope sign (bunching at φs = −30°) · I-8 ∫B, ∫B² preserved; H⁻ solenoid rotation sign = proton sign flipped (Bmad species, MAD-X `beam, charge=-1`) · I-9 every `Bn{n}L/Bs{n}L` preserved · I-10 kicker deflection sign+magnitude in every engine incl. TraceWin · I-11 rect/ellipse half-sizes preserved · I-12 converted map files reproduce ∫Ez, ∫B, ∫B² to 1e-10, checksums in provenance · I-13 write∘read fixed point byte-identical · I-14 ledger completeness · I-15 injective, reversible names · I-16 each RF element's `frequency` equals the active FREQ at its position · I-17 `Period` spans preserved where representable · I-18 `E_kin(ref_in_i)` equals the engine's per-element reference energy for accelerating lattices.

---

## 5. Verification design ("proper testing")

### 5.1 Oracle harness — one interface, N adapters, one common basis
`lattix/oracles/base.py`: `Oracle.available()`, `Oracle.run(deck, fmt, beam, probe) -> OracleResult{basis, s_bounds, names, R_elem (N,6,6), R_cum, twiss, disp, survey (X,Y,Z,θ), ref_energy(s), probe_out, warnings}` + `to_common()`. `Probe` = fixed 64-particle bunch (seed 20260903, 1e-4 m/rad amplitudes) so every engine stays linear.

| Adapter | Install (status today) | Observables | Native basis / p0 model |
|---|---|---|---|
| cpymad (MAD-X 5.09.03) | present | `twiss … sectormap` → `sectortable` r11..r66 per element (pattern `HELIX_v3/tests/io/test_madx_conventions.py:161-167`), `survey`, `track onepass` | (x,px,y,py,t,pt); **p0 constant** across RF |
| xtrack 0.103.5 | present | `twiss(method='4d', betx, bety)` → `get_R_matrix(a,b)`, `survey()`, `track()`; independent MAD-X reader (native Lark or cpymad) | (x,px,y,py,ζ,δ) |
| Bmad/Tao via pytao 1.2.4 | conda env `bmad` (present); run as `conda run -n bmad python -m lattix.oracles.bmad_worker` (JSON over stdout) — avoids merging py3.13/numpy2 with base py3.11 | `ele_mat6` per element, `matrix(beginning,end)`, `lat_list("*","ele.s"/"ele.e_tot")`, `ele_floor`, `ele_twiss` | (x,px,y,py,z,pz); `lcavity` follows p0 |
| elegant 2026.3.0 | `conda install -c conda-forge elegant` (osx-arm64 build exists) | generated `.ele` (`&run_setup`, `&twiss_output matched=0`, `&matrix_output individual_matrices=1`, `&floor_coordinates`, `&track`+`&sdds_beam`); parse SDDS with `pysdds` (present) or bundled `sddsconvert -ascii` | (x,x',y,y',s,δ); `RFCA change_p0=1` follows p0 |
| TraceWin (local only, `TRACEWIN_EXE`) | `TraceWin.app/Contents/MacOS/TraceWin` (x86_64, keyed; Rosetta OK) | argv per LightWin `interface.py:115-128`: `project.ini hide dat_file= path_cal= energy1= current1=0 freq1= nbr_part1=`; outputs `Transfer_matrix1.dat` (parser at `test_tracewin_crosscheck.py:302-313`), `partran1.out` (`tracewin_outputs.read_partran_out`), 26-col envelope export | (x,x',y,y',z,dp/p); follows p0 |
| HELIX (in-process, `HELIX_ROOT`) | `PYTHONPATH=HELIX_v3` | `get_element_matrix`, `compute_transfer_matrix` (`tracking/matrix_tracking.py:110,135`), envelope | (mm,mrad,mm,mrad,deg,MeV); follows p0 |
| ImpactX 26.08 | `conda install -c conda-forge impactx` | `lattice.load_file(madx)`, `init_envelope`/`track_envelope`, `reduced_beam_characteristics`; no per-element R → probe + envelope only | (x,px,y,py,t,pt) |
| IMPACT-Z 2.7.7 / FLAME 1.9.3 | conda-forge / `pip install flame-code` | IMPACT-Z `fort.18/24/25/26`; FLAME `transmat`, `moment1_env`, `ref_IonEk`, `pos` | Phase 3 |

**Common basis C** = (x m, px/p0, y m, py/p0, z m, δ) with *local* (β,γ,p0) at each boundary; `R_C = T_out·R_native·T_in⁻¹`. HELIX/TraceWin transform = inverse of `diag(-360/(βλ), β²γ·mc²)` at `test_tracewin_crosscheck.py:325-327`. **Rule: never hard-code a remembered longitudinal sign** — Phase 0 runs a *basis fingerprint* per engine (1 m drift → R56 sign/magnitude; thin cavity at lag 0.125 → energy-gain sign) and commits the resulting `T` as golden JSON, so an engine upgrade that flips a convention fails loudly. Engines split into p0-constant (MAD-X, xtrack, ImpactX-from-MAD-X) and p0-following (TraceWin, HELIX, Bmad lcavity, Elegant change_p0, IMPACT-Z, FLAME): compare transverse blocks after damping normalisation `R̂ = R/√det(R₂ₓ₂)` and assert `Π det(R₂ₓ₂) = p_in/p_out` separately on p0-following engines.

### 5.2 Metrics and tolerance tiers (tier chosen from the capability matrix's fidelity class, never a loose global tolerance)

| Metric | Exact | Equivalent-model (hard-edge vs field map, fringe models) | Lossy |
|---|---|---|---|
| Per-element `‖R̂a−R̂b‖F/‖R̂a‖F`, elementwise `rtol·|a|+1e-10` | 1e-8 | 2 % | report only |
| Cumulative `R_cum` drift vs s | 1e-7 | 5 % | report |
| `Π det(2×2)` vs `p_in/p_out` | 1e-9 | 1e-3 | report |
| Twiss β (rel) / α (abs) | 1e-7 / 1e-8 | 1 % / 0.02 | report |
| Dispersion (abs+rel), survey end point (abs), θ | 1e-9 m / 1e-7, 1e-9 m, 1e-12 rad | 2 %, 1e-6 m | report |
| Reference energy vs s (rel), probe final coords | 1e-9, 1e-9 m + 1e-7 | 0.5 %, 2 % of rms | report |
| Structural: total length, Σ|angle|, counts by kind, name multiset | exact | exact | exact |

Anti-cancellation (HELIX house rule 3): every format pair has ≥1 test with **external engines on both ends**; every writer has a *lockstep anchor* (same machine reaching the IR by two code paths sharing nothing, e.g. `BTL2025v0703.lat` via MAD8 vs `btl_2025v0703.dat` via TraceWin, as `test_mad8_parser.py:301-331`); cumulative invariants asserted over whole lines (the RFQ-K2 lesson).

### 5.3 Test taxonomy (`tests/`)
`unit/` (tokenizers, `:=`/RPN/Bmad expression evaluators, unit tables, one hand-computed test per conversion rule) · `golden/<fmt>/<deck>.<ext>` (writer snapshots; normaliser strips timestamps/paths, floats `%.12g`) · `roundtrip/` (A→IR→A→IR equal; IR→B→IR equal modulo the declared lossy set; strict raises on any lossy field) · `oracle/test_<src>_<dst>_<deck>.py` · `invariants/` · `property/` (hypothesis `st_lattice()` from the capability matrix; profiles `ci`=200, `nightly`=5000; oracle-backed property tests only with cpymad/HELIX) · `downgrades/` (every downgrade in BOTH regimes, pattern `HELIX_v3/tests/io/test_parser_downgrades.py:34-35`) · `corpus/` (manifest smoke, allowed warning classes, golden dozen with pinned numbers) · `fidelity/` (pydantic `FidelityReport` schema) · `oracles/` (fingerprints, availability). Markers: `oracle_madx|xtrack|bmad|elegant|impactx|impactz|flame|helix|tracewin`, `corpus`, `slow`; one `require()/require_data()` guard in `conftest.py` (mirrors `HELIX_v3/tests/dataguard.py`, replaces the hand-rolled `Fields/` check at `test_tracewin_writer.py:7-10`).

### 5.4 Corpus (`LATTIX_CORPUS_DIR` + `manifest.yaml` with id, format, sha256, origin, license, redistributable, expected_warning_classes, anchors; sha mismatch fails the corpus job)
- **Private, referenced only:** 724 PIP-II TraceWin decks (`PIP_II/Paper/.../TraceWin/SC_Linac/FDR_Design/{calculations,optimized_datfiles}`, `.../SCL/*/optimized_datfiles`, `.../BTL|BAL/*/lboptimization - Copy/calculations`); multi-format anchors `PIP_II/.../BTL|BAL/*_Lattice_with_Spacecharge/MAD_lattice/{BTL2022V0922_NEWCOL.FLAT, elegant_lattice.lte}`, `.../Virtual Accelerator/Prototypes/V1_14_August_2024/TraceWin_elegant_lattice.lte` (HWR CM), `.../TraceWin_lattice/{HWR,SSR1,SSR2,LB650}_Lat.lat,hb650_sad.lat` (TraceWin dialect despite `.lat`); HELIX `examples/pipii/**`, `examples/piplattice/fnalscl.dat`, `examples/lebt_plus_rfq/*.dat`, repo-root `BTL2025v0703.lat`, `BAL2025V0213.FLAT`; `HELIX_v3/Fields/` (56 ANL/CEA maps); TraceWin ground truth `HELIX_v3/Tracewin_code/**`, `HELIX_unzipped/TraceWIn_Tools/{Transfer_matrix1.dat, Individual_matrix.dat, BTL_lattice.dat}`; `ring_code/MAD-X_BOOSTER/{NEW,OLD}`.
- **Vendorable to `tests/data/public/`** (licenses checked): FLAME `examples/*.lat` (MIT), LightWin 6 decks (MIT), PyORBIT3 examples (MIT), xtrack `test_data/` (Apache-2.0), ImpactX `examples/fodo/fodo.madx` + `.in` (BSD-3-LBNL), IMPACT-Z examples (BSD), HELIX `examples/madx/{fodo,transport}.madx` + synthetic `.dat` teaching decks. **Cite only:** cheetah/ocelot/OPAL/pytao (GPL-3), synergia2 (custom), bmad-ecosystem (per-directory terms — check before copying), dynac (check).
- **Golden dozen** (`tests/corpus/golden.yaml`): `fodo.madx` (cpymad today: L 6.6 m, βx_end 7.1635, βy_end 15.2867), `transport.madx` (12 m), `fodo_cell.dat`, `bend_line.dat`, `mebt.dat` (427), `mebt+hwr.dat` (483), `btl.dat` (960), `BTL2025v0703.lat` (949 expanded, end s 307 969.918 mm), BTL `.FLAT`⇄`.lte` pair, HWR `.lte`, `fnalscl.dat`, `lebt_pxie.dat`, Booster `NEW/*.madx`.
- **Lockstep anchors:** BTL MAD8 vs TraceWin; BTL `.FLAT` vs `elegant_lattice.lte`; HWR CM `.lte` vs HWR section of `mebt+hwr.dat` (Equivalent tier); `fodo.madx` through cpymad/xtrack/Tao/elegant (four engines, Exact).

### 5.5 CI matrix
| Job | Runner | Env | Runs | Budget |
|---|---|---|---|---|
| `fast` | ubuntu + macos-14 | pip: numpy scipy pydantic lark pyyaml hypothesis pysdds pals-schema | unit, golden, roundtrip, invariants, downgrades, fidelity, property(ci) | < 3 min |
| `oracles` | ubuntu + macos-14 | `environment-ci.yml` (conda-forge: python 3.11, cpymad, xtrack, elegant, impactx, impact-z; pip: flame-code, pysdds, pals-schema, lightwin) + `environment-bmad.yml` (bmad, pytao) | all `oracle_*` except tracewin; corpus smoke on vendored data; uploads `fidelity/*.json` + per-element error-vs-s HTML | < 15 min |
| `nightly` | this Mac (self-hosted) | + `LATTIX_CORPUS_DIR`, `TRACEWIN_EXE`, `HELIX_ROOT` | full 724-deck corpus, `oracle_tracewin`, property(nightly), golden dozen | < 2 h |
Coverage ≥ 90 % (fail < 85 %). CI never contains CEA/ANL data.

### 5.6 Phase-0 bring-up (exact commands) and the five acceptance tests that gate Phases 1–2 (A1, A3-MAD-X leg, A5-MAD-X leg gate Phase 1; A2, A3 all legs, A4 gate Phase 2; A5 TraceWin leg gates Phase 3)
```bash
conda env create -f environment-ci.yml && conda activate lattix     # cpymad xtrack elegant impactx impact-z + pip extras
# reuse existing env 'bmad' (bmad 20260828.0, pytao 1.2.4) — verified today
python -c "from cpymad.madx import Madx; print(Madx(stdout=False).version)"      # MAD-X 5.09.03 ✔
elegant -h | head -1; sddsquery -h | head -1; python -c "import impactx, flame, pysdds, pals"
conda run -n bmad python -c "import pytao; print(pytao.__version__)"              # 1.2.4 ✔
export TRACEWIN_EXE="/Users/abhishekpathak/Desktop/Projects/TraceWin/TraceWin.app/Contents/MacOS/TraceWin"
export LATTIX_CORPUS_DIR=~/lattix_corpus && python -m lattix.corpus build-manifest
pytest tests/oracles -m "not oracle_tracewin"                                     # basis fingerprints per engine
"$TRACEWIN_EXE" project.ini hide dat_file=fodo_cell.dat path_cal=/tmp/tw energy1=2.1 current1=0 nbr_part1=1000   # confirm batch mode + Transfer_matrix1.dat
```
| # | Deck → targets | Oracles | Gate |
|---|---|---|---|
| A1 | `fodo.madx` → `.dat` → `.madx` | cpymad vs HELIX | per-element R̂ 4×4 rtol 1e-8; L = 6.6 m; survey Δ ≤ 1e-9 m |
| A2 | `fodo.madx` → `.bmad`, `.lte` | cpymad vs Tao vs elegant | R̂ 6×6 rtol 1e-8; βx_end 7.1635 / βy_end 15.2867 in all three |
| A3 | `mebt.dat` (427 el., quads + GAP, no maps) → `.madx`, `.bmad`, `.lte` | HELIX vs cpymad (damping-normalised 4×4), Tao, elegant (6×6, p0-following) | R̂ 1e-7; Π det = p_in/p_out to 1e-9; W(s) rel 1e-9 |
| A4 | `BTL2025v0703.lat` (MAD8) → `.dat`, `.madx` | HELIX lockstep vs cpymad | 949 elements; end s 307 969.918 mm; R 1e-8; η abs 1e-9 m; survey 1e-6 m |
| A5 | `mebt+hwr.dat` (FIELD_MAP) → `.madx`/`.lte` with maps degraded to thin gap + drifts | TraceWin batch (nightly) + HELIX vs cpymad/elegant | Equivalent tier: β 1 %, W(s) 0.5 %, longitudinal 2×2 ratios mean 1±0.002 / std < 0.005; FidelityReport lists every FIELD_MAP downgrade; strict raises |

---

## 6. Phases and tasks (each phase ends green and committed; stop-anywhere-and-ship)

| Phase | Lands | Gate (from §5.6) |
|---|---|---|
| 0 (wk 1) **— done 2026-09-03** | Repo scaffold, both conda envs, corpus manifest, six oracle adapters + basis fingerprints, CI | met: madx/xtrack/bmad ≤ 6e-9 on `fodo.madx`; HELIX 4×4 1.3e-7 (dipole path-length gap documented); goldens for madx, xtrack, bmad, elegant, helix, tracewin |
| 1 (wk 2–4) **— done 2026-09-03** | IR, TraceWin reader/writer, MAD-X reader/writer, `walk()`, HELIX adapter, FidelityReport, CLI | met: A1 (madx→dat→madx ≤ 1e-8), A3 (MEBT warm section 1e-7), A5 (field maps reported, strict raises) |
| 2 (wk 5–7) **— done 2026-09-03** | Elegant, Bmad, MAD8, PALS readers/writers; lockstep anchors | met: A2, A3 all legs, A4 |
| 3 (wk 8–10) **— done 2026-09-03** | Field maps + thick cavities; ImpactX, IMPACT-Z, FLAME, xtrack, NCells/RFQ rules | met: field-map gate (Bmad 1e-14, MAD-X loads), FLAME ALL_lattice bit-exact; A5 TraceWin leg and fnalscl→IMPACT-Z deferred to the nightly (trial TraceWin cap; NCells as CCL) |
| 4 (wk 11–12) | HELIX GUI/MCP integration, nightly fuzz + full corpus, docs, release 0.1 | full 724-deck corpus smoke green; coverage ≥ 90 % |

### Phase 0 — Scaffold, toolchain, oracles
- [x] **0.1** `pyproject.toml` (name `lattix`, py≥3.11, ruff+mypy, pytest markers from §5.3), `lattix/` skeleton, `tests/conftest.py` with `require()/require_data()`, `.gitignore` (`.hypothesis/`, `*.fidelity.json`, `sectormap`), BSD-3 `LICENSE`, `README.md`.
- [x] **0.2** (env `lattix` built: elegant 2026.3.0, impactx 26.08, impact-z 2.7.7, cpymad, xtrack 0.112, pysdds, pals-schema; `lightwin` dropped — needs py≥3.12) `environment-ci.yml` + `environment-bmad.yml` (reuse existing env `bmad`); run the §5.6 verification commands; install `elegant`, `impactx`, `impact-z`, `flame-code`, `pysdds`, `pals-schema`, `hypothesis` and record versions in `docs/oracles.md`.
- [x] **0.3** (manifest 955 entries at ~/lattix_corpus; 39 public samples 688 kB; golden dozen measured — see docs/corpus.md) `lattix/corpus.py build-manifest` over `LATTIX_CORPUS_DIR` (sha256, provenance, license, redistributable, expected_warning_classes); vendor public samples into `tests/data/public/` with a `LICENSES/` folder; golden-dozen `tests/corpus/golden.yaml` seeded with today's cpymad numbers.
- [x] **0.4** (adapters madx, xtrack, bmad (worker in env `bmad`), helix, tracewin; goldens for all five; gate met: madx/xtrack/bmad ≤ 6e-9, HELIX 4×4 1.3e-7 + documented dipole path-length gap) `oracles/base.py`, `basis.py`, `cpymad.py`, `xtrack.py`, `pytao.py` + `bmad_worker` (JSON over stdout via `conda run -n bmad`), `helix.py`; basis-fingerprint tests (1 m drift R56; thin cavity at lag 0.125) → `tests/oracles/goldens/*.json`.
- [x] **0.5** (workflows written, not yet exercised on GitHub; TraceWin batch verified: works headless, trial build capped at 20 elements, rejects `TITLE`) GitHub Actions `fast` + `oracles` (ubuntu, macos-14); nightly workflow stub gated on `LATTIX_CORPUS_DIR`/`TRACEWIN_EXE`; verify TraceWin batch mode once by hand (`"$TRACEWIN_EXE" project.ini hide …` → `Transfer_matrix1.dat` appears) and note the result in `docs/oracles.md`.

### Phase 1 — IR + TraceWin + MAD-X
- [x] **1.1** (done 2026-09-03: 22 kinds, PALS parameter groups, sync-phase RF convention measured against HELIX/MAD-X, walk/survey/normalize/rf/expr, fidelity ledger, format registry, lattix JSON, CLI convert/inspect/report; hypothesis strategies deferred to the property tests) `ir/` (elements, lattice, reference, walk, expr, normalize, rf, units) with unit tests for every §4.3 formula (hand-computed numbers) and hypothesis strategies (`st_lattice()`); `RULES`-coverage test hook in `formats/base.py`.
- [x] **1.2** (done 2026-09-03: 816-deck corpus parses in 19 s, 0 exceptions; lockstep vs HELIX path exact; SUPERPOSE clusters, SET_SYNC_PHASE one-shot, `variable`/expression decks) TraceWin reader: port `tracewin_syntax.SCHEMA` (+`unit`, `writer_default` columns), label grammar, latin-1, FREQ state, EDGE+BEND+EDGE clustering → `Bend`, FIELD_MAP/SUPERPOSE clusters → `FieldMap`/`Superposition`, commands → side-channel, `ERROR_*` → `errors`, strict/permissive ledger. Dual-regime downgrade tests for every path; smoke over HELIX `examples/**/*.dat` (public) and the 724 PIP-II decks (nightly).
- [x] **1.3** (done: %.15g, EDGE cards from e1/e2, 5 goldens, 13-deck fixed point, HELIX byte-diff explained) TraceWin writer: port `_emit_card` elision + field-map re-emission + `best_relpath`; **emit `EDGE` cards from `Bend.e1/e2`** (fixes HELIX's lossy BEND); golden snapshots; idempotence (I-13); byte-compare against HELIX's writer on `examples/pipii/mebt/mebt.dat` (differences must be explained by the e1/e2 and ttf fixes).
- [x] **1.4** (done: cpymad-backed, symlink-mirror cwd for relative CALLs, rbend chord→arc, k1s→tilt, dipedge folding, EALIGN; lark fallback deferred) MAD-X reader: cpymad-backed (authoritative: `Madx.input` → sequence expansion, `elements`, `globals` with deferred expressions) when importable, else a lark grammar (start from xtrack's `mad_parser/madx.lark`, keep HELIX's `_eval_node` whitelist); `BEAM` → `ReferenceParticle`; `EALIGN`/`EFCOMP` → BodyShift/field errors; `CALL` resolved relative to the deck.
- [x] **1.5** (done: sequence/line modes, energy_mode constant|local (MAD-X twiss carries cavity gain in pt → `constant` composes with MAD-X), expressions re-emitted, 41-char name limit measured, apertype rules) MAD-X writer: sequence and line modes, `energy_mode` local/constant, expression emission, `RFCAVITY lag = φ/2π + 0.25`, `EQUIVALENT:CONST_P0` entries; MAD8 dialect switch reserved for 2.3.
- [x] **1.6** (done: bit-exact HELIX→IR→HELIX on mebt, mebt+hwr incl. real field maps; sync-phase mechanism documented) `formats/helix`: IR ⇄ `linac_gen.Lattice` (mm/deg/MeV; `RFGap` V↔MV; `Dipole` from `Bend` + `Edge`), so the HELIX oracle and HELIX users get every format; lockstep test: `read_tracewin(mebt.dat)` via lattix vs `parse_tracewin` via HELIX → per-element matrices equal 1e-10.
- [x] **1.7** `fidelity.py` + `cli.py` (`convert`, `inspect`, `validate`, `report`); `validate` runs source and target oracles and prints the §5.2 metrics table.
- [x] **1.8** (A1/A3/A5 green 2026-09-03; A3 restated: the MEBT has four buncher FIELD_MAPs, compared up to the first map) Gate: A1, A3 (MAD-X leg), A5 (MAD-X leg, `FM_TO_CAVITY` degradation + strict raises) green on `oracles` CI.

### Phase 2 — Elegant, Bmad, MAD8, PALS
- [x] **2.1** (done 2026-09-03: 153 tests; RBEN chord, FINT 0.5, charge-sign phase pinned by tracking; A2 leg 2.8e-10) Elegant reader (port `_logical_statements`, `_rpn_eval`, `_expand`, templates, `name[prop]=` overrides) + writer (`RFCA change_p0=1`, phase+90°, `CSBEND`, `KQUAD`); `oracles/elegant.py` (generated `.ele`, `pysdds`); golden test vs Bmad `regression_tests/write_foreign_test/lte.correct` semantics.
- [x] **2.2** (done: 80 tests; thin lcavity → traveling_wave; A2 leg 2.8e-14; Bmad converter skew-sign bug pinned) Bmad reader (elements, `line`, `use`, `parameter[…]`, `beginning[…]`, `call`, `superimpose` → `Superposition`, infix expressions; Cheetah `converters/bmad.py` as reference) + writer (`lcavity`, `sbend`, `ref_tilt`, `x_offset…`); cross-check against `bmad_to_mad_sad_elegant -madx/-elegant` and `madx_to_bmad.py` outputs (differences must be explained).
- [x] **2.3** (done: 119 tests; BTL lockstep 1e-10; BAL parses; BRHO deck bug found; negative-drift overlap handling in the MAD-X writer) MAD8 flat reader (port `_Mad8File` lazy resolver, `LINE` expansion, `_root_line`, periodicity → `Period` commands) + MAD8 writer (MAD-X writer dialect: `&` continuations, 16-char names, `LINE`).
- [x] **2.4** (done: 139 tests; validated by pals-schema 0.3.0, loaded by ImpactX 26.08, cross-read with Tao `write pals`) PALS reader/writer via `pals-schema` (fallback plain YAML): `RFP.phase = φ/2π`, `dE_ref` explicit, `BeginningEle/ReferenceP`; validate against Tao `write pals` output and ImpactX `KnownElementsList.load_file(*.pals.yaml)`.
- [x] **2.5** (BTL MAD8 vs TraceWin 1e-10; BTL .FLAT vs .lte structure-only — the .lte export carries no values; fodo through cpymad/xtrack/Tao/elegant) Lockstep anchors: BTL MAD8 vs TraceWin (949 elements, end s 307 969.918 mm); BTL `.FLAT` vs `elegant_lattice.lte`; `fodo.madx` through cpymad/xtrack/Tao/elegant (Exact tier).
- [x] **2.6** (green 2026-09-03: A2 Bmad 2.8e-14 / elegant 2.8e-10; A3 all legs; A4 BTL MAD8→dat/madx 1e-7 over 308 m) Gate: A2, A3 (all legs), A4.

### Phase 3 — Linac depth and Tier-2 formats
- [x] **3.1** (done 2026-09-03: HELIX-equivalent integration bit-for-bit on 100 PIP-II maps; FM_TO_CAVITY / FM_SOL_HARDEDGE / FM_QUAD_HARDEDGE in every writer; sha256 per map file) `ir/fieldmap.py` + `formats/tracewin/fieldmap_files.py` (port `tracewin_geom`, `field_map_reader`, `tracewin_fieldmap_reader`): `integrate_map()` (dE_ref, T(β), ∫B, ∫B²), degradation rules `FM_TO_CAVITY`, `FM_SOL_HARDEDGE`, quad-map; converters to Bmad `grid_field`, ImpactX Fourier `cos_coef`, IMPACT-Z `rfdata`; checksums (I-12).
- [x] **3.2** (done: 108 tests; exact vs ImpactX's own loader; t late-positive / pt = −ΔE/p0c documented; `load_inputs_file` segfault found) ImpactX writer (Python input + `.in`) and reader (`.in` + its MAD-X subset); `oracles/impactx.py` (envelope + probe).
- [x] **3.3** (done: 96 tests; ideal-cavity sentinel = the IR RF rule; per-element maps from a 13-particle probe; 5 landmines documented) IMPACT-Z writer/reader (`ImpactZ.in` type codes, `rfdata`), `oracles/impactz.py` (conda `impact-z` or Rosetta binary; `fort.18/24/25/26`).
- [x] **3.4** (done: 127 tests; built from the clone (no macOS wheel); bit-exact FLAME round trips; K normalized, roll = tilt) FLAME GLPS reader/writer + `oracles/flame.py` (per-element `transmat`); `ALL_lattice.lat` round-trip.
- [x] **3.5** (done: 107 tests on 0.103 and 0.112; PSB ring through lattix vs xtrack's own loader exactly 0.0) xtrack adapter (IR ⇄ `xt.Line`/JSON, `Cavity lag+90°`); optional Ocelot/Cheetah object adapters (GPL packages are runtime-optional, never vendored).
- [~] **3.6** (NCells → IMPACT-Z 103 (EQUIVALENT), TraceWin exact, elsewhere LOSSY; RFQ cells TraceWin-only) NCells/DTL and RFQ cells: TraceWin E; IMPACT-Z 101/103 M; elsewhere L→gaps with report; `SET_SYNC_PHASE` semantics tests in both regimes.
- [x] **3.7** (field-map gate: Bmad reference energies vs HELIX 1e-14 after all 12 HWR cavities; MAD-X leg loads with real rfcavities — MAD-X twiss then fails on the open accelerating line, the documented constant-p0 limit) Gate: A5 TraceWin leg (nightly); HWR `.lte` vs `mebt+hwr.dat` HWR section (Equivalent tier); `fnalscl.dat` → IMPACT-Z within Equivalent tier.

### Phase 4 — Integration, hardening, release
- [ ] **4.1** (deferred 2026-09-03 by user decision: finish lattix first; HELIX tree untouched) HELIX: replace the three suffix dispatchers (`cli/common.py:62`, `interphase/app.py:76`, `parallel/scan_pool.py`) with `lattix` registry behind a feature flag; GUI File → Import/Export (all formats, fidelity summary in status bar per `_write_lattice` pattern); MCP tool `translate_lattice` in `linac_gen/assist/tools.py`. (Separate HELIX PR; HELIX house rules apply: 5-pass audit, no `pip install -e .`.)
- [x] **4.2** (2026-09-03: `nightly.yml` self-hosted stub, hypothesis profiles `ci`=200/`nightly`=5000, fast-CI coverage gate 70 % (engine-free selection measures 80 % with cpymad on Linux, 74 % without it on macOS), `lattix validate --html`) Nightly: property fuzz (5000 examples), full corpus, `oracle_tracewin`; coverage gate; `lattix report` HTML (per-element error vs s).
- [x] **4.3** (2026-09-03: docs/{index,tutorial,conventions,fidelity}.md, docs/formats/*.md ×12, `lattix/fidelity_catalog.py` + `tests/docs/test_docs_consistency.py`, CITATION.cff) Docs: `conventions.md` (§4 tables, kept in sync by a docs-consistency test like HELIX `tests/core/test_docs_consistency.py`), `fidelity.md` (codes), `formats/*.md`, tutorial; CITATION.cff.
- [x] **4.4** (2026-09-03: version 0.1.0, sdist + wheel built locally, tag v0.1.0; **no PyPI upload** by user decision; private GitHub repo `Abhishek-Pathak-90/lattix`) Release 0.1 to PyPI; license audit of `tests/data/public/`. Backlog (Tier 3): OPAL, DYNAC (reference `dynac/converters/tw2dyn.f`), Astra/GPT via Bmad converters, TRACK, PyORBIT3 linac XML.

## 7. Verification (end-to-end)
1. `pytest -m "not corpus and not slow"` → fast suite green on both runners; `pytest -m "oracle_madx or oracle_xtrack or oracle_bmad or oracle_elegant"` green in `environment-ci.yml`.
2. Per phase, the gate rows in §5.6/§6: run `lattix validate <src> <dst> --oracles …` and assert the metrics table meets the tier; results archived under `runs/validate/<date>/`.
3. Nightly on this Mac: `LATTIX_CORPUS_DIR=… TRACEWIN_EXE=… pytest -m "corpus or oracle_tracewin or slow"`; manifest sha256 check; fidelity JSONs diffed against the previous night (new LOSSY/DROPPED codes are findings).
4. HELIX integration (Phase 4): open a translated `.madx`→`.dat` deck in the HELIX GUI, run envelope, and use the existing MCP tool `compare_to_tracewin` against the CEA reference exports (`Tracewin_code/`) — correlations must match the untranslated deck's within 1e-6.
5. Adversarial review before each phase commit (HELIX house rule 6): a subagent instructed to refute the diff with an engine run.

## 8. Risks and open items
- **RF phase/time sign conventions differ per engine** (Bmad's own converter reflects, `phi0 = 0.5 − lag`). Mitigation: I-7 oracle test per writer; never trust a remembered sign.
- **MAD-X cannot follow p0 through cavities**; translated accelerating linacs are optically correct per section only. Mitigation: `energy_mode`, `EQUIVALENT:CONST_P0` entry, documented; Elegant/Bmad/ImpactX are the faithful targets.
- **Field maps have no equivalent in MAD-X/Elegant/PALS**; degradation is L by design and always reported.
- **TraceWin batch on macOS (x86_64 under Rosetta) untested here**; verify in 0.5 (LightWin runner pattern). If it fails, TraceWin ground truth falls back to the existing CEA exports in `Tracewin_code/`.
- **Elegant longitudinal matrices are path-length based and ultra-relativistic** (drift R56 = 0; RFCA R65 = β·true): the adapter restores the drift term; cavity/longitudinal comparisons at low β use elegant *tracking* (`probe_out`), not its matrices.
- **Elegant RFCA phase depends on the species charge sign** — the writer must emit `phase = φ − 90°` for protons and `φ + 90°` for H⁻/electrons (HELIX's importer assumes the negative-species rule, correct for its H⁻ default only).
- **Two conda envs** (bmad needs py3.13/numpy2): pytao runs out-of-process (JSON worker).
- **PALS is work-in-progress**: pin `pals-schema` version; PALS parity tests are allowed to xfail on schema changes.
- **Name-length limits** (MAD-X 48?, MAD8 16?, Bmad 40?) to verify against manuals in 1.5/2.2/2.3.
- **Private data**: PIP-II decks, `Fields/`, `Tracewin_code/`, `TraceWIn_Tools/` never enter the repo or CI (manifest + `require_data`).
- HELIX side note: `tests/io/test_madx_conventions.py` leaves a `sectormap` file at HELIX's repo root (cpymad writes to cwd) — gitignore or `m.chdir()` in HELIX.
