# OPAL-T (input deck)

`lattix/formats/opal/` writes and reads the input decks of [OPAL](https://gitlab.psi.ch/OPAL/src) in its
OPAL-T (time-integration linac) flavour: MAD-9-style statements terminated by `;`, `//` and `/* */`
comments, `name: TYPE, attr=value, …;` element definitions placed by **`ELEMEDGE`** (the entrance path
length in metres), a `LINE`, a `BEAM`, `FIELDSOLVER`, `DISTRIBUTION`, `TRACK` and `RUN` block.  OPAL is
GPL-3: lattix handles the deck text and the 1-D field-map files it points to; **no OPAL engine runs here**
(a build needs MPI, H5hut, Boost and GSL), so every convention below was taken from the OPAL-X source
(`078ff1e`, `src/Elements/*.cpp`, `src/Classic/AbsBeamline/*.cpp`, `src/Algorithms/CavityAutophaser.cpp`)
and is marked *source-derived* rather than measured.  Bmad's own OPAL writer (`opal_interface_mod.f90`)
could not serve as a second reference: Tao's `write opal` is unreachable in the current Bmad (the dispatch
in `tao_write_cmd.f90` matches the misspelt keyword `opal_latice`), and where its code disagrees with the
OPAL-X source (it writes a quadrupole's lab gradient as `k1`) lattix follows OPAL-X.

## Reading

`lattix/formats/opal/reader.py` tokenizes the statements (a `;` outside quotes and brackets ends one;
`//` comments carry lattix's tags), evaluates `REAL x = …` variables and attribute expressions lazily with
the infix evaluator of `lattix.ir.expr` (OPAL's `PI`, `TWOPI`, `CLIGHT`, `EMASS`, `PMASS`, `DEGRAD`,
`RADDEG`), resolves an element that names another element as its type (`qd: qf, K1=-kq`), expands the
`LINE` selected by `TRACK, LINE=…` (or `read(line=…)`), including nested lines, `n*(…)` repeats and `-x`
reversal, and places the members in `LINE` order: a positive difference between an element's `ELEMEDGE`
and the previous element's end becomes a drift (EQUIVALENT `OPAL_GAP_DRIFT_INSERTED`), an overlap is a
warning.  The reference particle comes from lattix's `// lattix: reference` tag, else from the `BEAM`
(`PARTICLE` — `PROTON`, `ELECTRON`, `POSITRON`, `ANTIPROTON`, `DEUTERON`, `HMINUS` — with `MASS` [GeV] and
`CHARGE` overriding it as an ion, EQUIVALENT `SPECIES_ASSUMED`; the energy from `PC` [GeV/c], `ENERGY`
[GeV total] or `GAMMA`; `BFREQ` [MHz] as the RF clock); `read(species=, kinetic_energy_eV=, frequency_Hz=)`
win over both.

| OPAL element | IR | Notes |
|---|---|---|
| `DRIFT` | Drift | lattix's tag brings back what a writer folded into a drift (a limit-less collimator, a Taylor map, an RFQ cell, a field-free solenoid) |
| `QUADRUPOLE K1 K1S PSI` | Quadrupole | `Bn1 = K1 · P0/c` with the **BEAM's** unsigned rigidity (`OpalData::getP0()`), skew `K1S`, `PSI` the roll |
| `SEXTUPOLE K2`, `OCTUPOLE K3` | Sextupole, Octupole | `Bn = K · P0/c` |
| `MULTIPOLE KN KS` | Multipole | `BnL[n] = KN[n] · L · P0/c` (a lattix-written pair `KICKER` + `MULTIPOLE` of the same name is one thin multipole again) |
| `SBEND L ANGLE E1 E2 HGAP/GAP FINT K1 K2 PSI` | Bend | arc length, geometric angle, MAD's sector faces; `RBEND` gets `angle/2` added to its faces (EQUIVALENT `RBEND_AS_SECTOR`); `DESIGNENERGY` is not needed (the IR walk carries the energy) |
| `SOLENOID KS FMAPFN` | Solenoid | `Bsol = KS · P0/c` times the map's peak (1 for a normalised map); the length is lattix's tag, else the map's extent |
| `RFCAVITY`, `TRAVELINGWAVE VOLT FREQ LAG DESIGNENERGY FMAPFN` | RFCavity | voltage = lattix's tag, else `DESIGNENERGY − E_kin` (the crest energy, MeV), else the crest gain of the `1DDynamic` map at `VOLT` integrated by `lattix.formats.impactt.rfprofile` (LOSSY `RFCAVITY_GAIN_UNKNOWN` without a map); phase = `LAG` (EQUIVALENT `OPAL_LAG_AS_SYNC_PHASE`: OPAL adds it to the autophased crest); the map profile is kept in `native['opal']` |
| `KICKER HKICK VKICK`, `HKICKER`/`VKICKER KICK`, `K0`/`K0S` | Kicker | deflections in rad; a field `K0` [T] over `L` gives `−K0·L/Bρ_signed` |
| `RCOLLIMATOR`, `ECOLLIMATOR XSIZE YSIZE` | Collimator | half apertures |
| `MONITOR` | Instrument | family from the tag |
| `MARKER` | Marker | or the kind lattix's tag names (Patch, ReferenceChange, Freq, Foil, a thin solenoid or collimator) |
| `APERTURE="CIRCLE(d)"`, `"ELLIPSE(w,h)"`, `"RECTANGLE(w,h)"`, `"SQUARE(w)"` | (element aperture) | full sizes in m (`OpalElement::getApert`) |
| `DX DY DZ DTHETA DPHI DPSI` | (misalignment) | offsets and rotations about y, x, z |
| `SOURCE`, `DEGRADER`, `PROBE`, `SEPTUM`, `STRIPPER`, `CCOLLIMATOR`, `CYCLOTRON`, … | Drift / Marker | LOSSY `UNSUPPORTED_OPAL_ELEMENT`, attributes in `native['opal']` |

## Writing

`lattix/formats/opal/writer.py` writes one statement per placed element with its `ELEMEDGE`, the `LINE`,
a `BEAM` (`PARTICLE`, `MASS` GeV, `CHARGE`, `PC` GeV/c, `BFREQ` MHz), a field-solver-less `FIELDSOLVER`,
a placeholder `DISTRIBUTION` (lattix carries no beam), `TRACK` (`ZSTOP` = the lattice length, `DT` and
`MAXSTEPS` from the slowest reference velocity) and `RUN, METHOD="PARALLEL-T"`; `OPTION, AUTOPHASE=4`
so that every `LAG` is relative to the crest.  Names become identifiers (case-insensitive, OPAL keywords
avoided) with the original in the `// lattix:` tag.  Field maps (`1DDynamic` for cavities,
`1DMagnetoStatic` for solenoids, `lattix/formats/opal/maps.py`) are written next to the deck as
`<stem>_<name>.1dd` / `.1dms`.

| IR kind | OPAL | Class |
|---|---|---|
| Drift | `DRIFT L` (no field; kept for the LINE) | EXACT |
| Quadrupole, Sextupole, Octupole | `QUADRUPOLE K1 = Bn1/(P0/c)`, `K1S`, `PSI`; `SEXTUPOLE K2`; `OCTUPOLE K3` | EXACT (other orders LOSSY `MULTIPOLE_ORDERS_DROPPED`) |
| Multipole | dipole terms as a `KICKER` (deflections), the others as a `MULTIPOLE KN/KS` over the element's length or a 1 mm surrogate centred on a thin one | EQUIVALENT `MULTIPOLE_AS_SHORT` |
| Bend | `SBEND L ANGLE E1 E2 HGAP GAP FINT K1 K2 DESIGNENERGY PSI FMAPFN="1DPROFILE1-DEFAULT"` | EQUIVALENT `OPAL_BEND_DEFAULT_PROFILE`; `BEND_FINTX_DROPPED`, `FRINGE_K2_DROPPED`, `ZERO_ANGLE_BEND_AS_DRIFT` |
| Solenoid | `SOLENOID KS = Bsol/(P0/c)` with a generated flat-top map (raised-cosine ramps of `solenoid_ramp_m`, the map `L + ramp` long so that `∫B dz = Bsol·L`, `ELEMEDGE` moved back by half a ramp) | EQUIVALENT `OPAL_SOLENOID_MAP`; a zero-length one LOSSY `SOLENOID_TO_MARKER` |
| RFCavity | `RFCAVITY VOLT FREQ LAG DESIGNENERGY TYPE="STANDING"` with a generated `1 − cos` bump (no wider than βλ/2) or a cell train (`n_cell > 1`); a thin gap over `thin_gap_surrogate_length` centred on it | EQUIVALENT `OPAL_CAVITY_MAP`; no voltage or frequency LOSSY `OPAL_CAVITY_TO_DRIFT`; a driven phase EQUIVALENT `RF_RAW_PHASE_AS_SYNC` |
| FieldMap | the map's own on-axis `E_z` (`1DDynamic`, `VOLT` = its peak, `LAG` = the synchronous phase, `DESIGNENERGY` = the crest energy from the map summary) or `B_z` (`1DMagnetoStatic`); else the hard-edge ladder of `lattix.ir.fieldmap.replacement_for` | EQUIVALENT `FM_AS_OPAL_MAP` (`FM_STATIC_B_DROPPED` for a mixed map) |
| NCells | `RFCAVITY` with a cell-train profile of the train's voltage | EQUIVALENT `NCELLS_AS_CAVITY`; `NCELLS_TO_DRIFT`, `DYNAC_NCELLS_PARAMS` |
| RFQCell | `DRIFT` | LOSSY `RFQCELL_TO_DRIFT` |
| Kicker | `KICKER L HKICK VKICK DESIGNENERGY` (a thin one over `thin_length_m`) | EXACT; electric LOSSY `EKICK_AS_MAGNETIC` |
| Collimator | `RCOLLIMATOR` / `ECOLLIMATOR XSIZE YSIZE` | EXACT; no limits LOSSY `COLLIMATOR_TO_MARKER`; `APERTURE_OFFSET_DROPPED` |
| Marker, Instrument | `MARKER`; `MONITOR OUTFN` | EXACT; EQUIVALENT `INSTRUMENT_AS_MONITOR` |
| Foil, Taylor, Patch | `MARKER` / `DRIFT` with the data in the tag | LOSSY `FOIL_TO_MARKER`, `TAYLOR_DROPPED`, `PATCH_DROPPED` |
| ReferenceChange, Freq | `MARKER` with the change / frequency in the tag | EQUIVALENT `REFCHANGE_AS_TAG`; EXACT |
| Directive | a comment | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | children in order | LOSSY `SUPERPOSITION_FLATTENED` |

Element apertures become `APERTURE="CIRCLE(…)"` / `"ELLIPSE(…)"` / `"RECTANGLE(…)"` (EQUIVALENT
`APERTURE_AS_ATTRIBUTE`), misalignments `DX DY DZ DTHETA DPHI DPSI`.  Options: `thin_length_m` (1 mm),
`solenoid_ramp_m` (2 cm), `map_points` (201), `dt_s`, `n_particles`, `autophase`, `install_apertures`.

## Units and conventions

*Source-derived (OPAL-X 078ff1e), not measured — an OPAL build would be the first thing to run.*

* m, rad, MV/m (`VOLT`), MHz (`FREQ`), rad (`LAG`), MeV kinetic (`DESIGNENERGY`), GeV (`BEAM MASS`,
  `PC`); `ELEMEDGE` is the entrance path length; a bend rotates the reference axis (`compute3DLattice`),
  a positive `ANGLE` towards −x like MAD-X's.
* **Strengths**: `OpalQuadrupole`/`OpalSBend`/`OpalMultipole`/`OpalSolenoid` multiply `K1`, `K2/2`,
  `K3/6`, `KN[n]/n!`, `KS` by `OpalData::getP0()/c` — the **BEAM's initial momentum, unsigned** — so the
  written normalized strength is the lab field over the start `|Bρ|` at every energy and for either
  charge sign (a negative species keeps the sign a proton would get; MAD-X flips it).  `MULTIPOLE KN`
  are the MAD `K_n` (`Multipole::setNormalComponent` divides by `(n−1)!`); the dipole term is written as a
  `KICKER` because OPAL-T halves a multipole's dipole component.
* **Kickers**: `Corrector::goOnline` sets `kickField = q·p/(c·L)·(vkick, −hkick)`, so `HKICK`/`VKICK` are the
  deflections of the actual particle at `DESIGNENERGY`; a corrector needs a length longer than one time
  step (`thin_length_m`).
* **Cavities**: `E ∝ VOLT·cos(ωt + φ)` on a map normalised to its peak; with `OPTION, AUTOPHASE`
  `CavityAutophaser` sets `φ = LAG + (phase of maximum energy gain)`, so `LAG` is the synchronous phase in
  the IR's convention (a late particle gains more at a negative `LAG`, the crest is found for the actual
  charge); a positive `DESIGNENERGY` makes OPAL rescale the amplitude until the crest energy equals it,
  so lattix writes `E_kin,in + V_eff` there and the transit-time estimate `V_eff/|∫E_z e^{ikz}dz|` as
  `VOLT`.  `RFCavity::initialise` and `Solenoid::initialise` set the element length to the map's extent.
* **Field-map files** (`Fieldmap::readHeader`, `FM1DDynamic`, `FM1DMagnetoStatic`): first word
  `1DDynamic` / `1DMagnetoStatic` and the number of Fourier terms (optionally `FALSE` for an unnormalised
  map), then `zbegin zend n` in cm (`n` intervals, `n + 1` values), the frequency in MHz (dynamic maps),
  `rbegin rend nr`, and one on-axis value per line.
* Apertures `"CIRCLE(d)"` etc. are full sizes in m (`width2HalfWidth = 0.5` in `getApert`).
* The `PSI` sign (a rotation about z) and the misalignment reference point follow the source's Tait–Bryan
  handling (`OpalElement`, `compute3DLattice`); Bmad's writer uses the opposite `psi` sign — unverified
  either way.

## Known limits

* No engine: nothing here is measured against OPAL; the `LAG` convention needs `OPTION, AUTOPHASE > 0`.
* OPAL-T integrates real fields: hard-edge elements become maps (solenoids), thin ones short surrogates
  (`thin_length_m`), bends use OPAL's default fringe profile (`1DPROFILE1-DEFAULT`) rather than MAD's
  `fint`/`hgap` model.
* Taylor maps, patches, foils, directives and RFQ cells have no OPAL-T counterpart; travelling-wave
  cavities are written as standing-wave maps with a tag (OPAL's `TRAVELINGWAVE` wants its own map layout).
* The `DISTRIBUTION` is a placeholder; the deck runs only after a real one is put in.

## Oracle

None.  The battery treats OPAL as a format without an engine (fixed points, IR round trips); the
conventions are pinned to the OPAL-X source in `tests/formats/test_opal_writer.py` (strengths over the
BEAM's rigidity for a proton and an H⁻, `LAG`, `DESIGNENERGY`, the map headers and integrals) and
`tests/formats/test_opal_reader.py` (a hand-written native deck).  An OPAL build (MPI + H5hut) or the
OPAL Docker image would make the phase-slope sign and the multipole scaling measurable.
