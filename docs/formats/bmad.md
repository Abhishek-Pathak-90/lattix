# Bmad

Bmad (Cornell) lattice files define elements as `name: type, attr = value, …`, lines with
`line = (…)`, `use, name`, `parameter[…]` and `beginning[…]` settings, `call::file`,
`superimpose` and infix expressions.  lattix registers `.bmad` with a reader and a writer,
both verified against Bmad 20260828.0 through Tao.

## Reading

`lattix/formats/bmad/reader.py` is a hand-written recursive-descent parser (Cheetah's
`converters/bmad.py` was the structural reference):

* `!` comments, `;` separators, continuation on a trailing `&` or `, ( { [ =` or a next line
  starting with `, ) } ] =`; case-insensitive names with the source spelling kept in
  provenance; 40-character name limit (measured: 41 fails);
* element inheritance (`quadrupole1: q0, l = 0.6`), bare attribute flags selecting Bmad's
  defaults (multipole tilt π/(2(n+1)): π/4, π/6, π/8 measured), `name[attr] = value`
  overrides anywhere in the file;
* `superimpose` → `Superposition` (EQUIVALENT `SUPERIMPOSE_RESOLVED`), `patch`, `floor_shift`,
  `fiducial` → `Patch`, `foil` → `Foil`, `taylor` → `Taylor`, controllers (`overlay`, `group`)
  kept as directives (LOSSY `BMAD_CONTROLLER`), `grid_field` maps LOSSY `BMAD_GRID_FIELD`;
* rigidity: `k1`/`ks`/kicks are species-independent in Bmad exactly as in MAD-X, so the
  lab field is `k1·Bρ_signed`; because `lcavity` moves p0 the reader runs two passes and
  converts every strength with the rigidity at that element's own entrance;
* `rbend`: the deck `l` is the chord; the IR stores the arc and sector-referenced pole faces
  (`e1 += θ/2`) with `BendP.rect`;
* `lcavity`: `phi0` in turns, `dE = V·cos(2π·phi0)` (`phase_from_bmad_phi0`); a ring
  `rfcavity` gives the reference particle no energy (EQUIVALENT `RING_RFCAVITY`, IR phase
  `2π(0.25 − phi0)`);
* `k{n}l`/`k{n}sl` are exactly MAD-X `knl`/`ksl`; `b_n = k_nl/n!`; the species comes from
  `parameter[particle]` (`SPECIES_FROM_BMAD`), `#1H-` for H⁻.

## Writing

`lattix/formats/bmad/writer.py` writes `parameter[particle]`, `parameter[e_tot]` (eV),
`parameter[geometry] = open`, lower-case definitions with the `! lattix: name="…" type="…"`
tag, and the root line.

| IR kind | Emitted | Ledger |
|---|---|---|
| Drift, Quadrupole, Sextupole, Octupole, Multipole, Solenoid | the Bmad type of the same name (`k1`, `k2`, `k3`, `k{n}l`/`k{n}sl`, `ks`) | EXACT |
| Bend | `sbend l angle e1 e2 fint fintx hgap k1 ref_tilt` (`rbend` for a rectangular source) | EXACT |
| RFCavity | `lcavity voltage phi0 rf_frequency n_cell`; thin cavities as `l = 0, cavity_type = traveling_wave` | EXACT (thin: EQUIVALENT `THIN_CAVITY_TRAVELING_WAVE`) |
| FieldMap | thick `lcavity` with the map's voltage and phase | EQUIVALENT `FM_TO_CAVITY` |
| NCells, RFQCell | drift | LOSSY `NCELLS_TO_DRIFT`, `RFQ_TO_DRIFT` |
| Kicker | `kicker` / `hkicker` / `vkicker` | EXACT |
| Collimator | `rcollimator` / `ecollimator` with `x1_limit … y2_limit`, `aperture_type` | EXACT |
| Marker, Instrument | `marker`, `instrument` / `monitor` by family | EXACT |
| Taylor, Foil, Patch | `taylor`, `foil`, `patch` | EXACT |
| ReferenceChange | marker (Bmad's reference energy follows its own cavities) | LOSSY `REFCHANGE_DROPPED` |
| Freq | comment | EXACT |
| Directive | comment | DROPPED `FOREIGN_DIRECTIVE` |
| Superposition | `superimpose` on the children | EQUIVALENT `SUPERPOSITION_SUPERIMPOSED` |

Apertures go on every element (`x1_limit`, `x2_limit`, `y1_limit`, `y2_limit`,
`aperture_type = elliptical|rectangular`, `aperture_at = both_ends`).

## Units and conventions

* m, rad, V, Hz, `phi0` in turns with 0 = crest (`bmad_phi0`), species-independent.
* An `lcavity` with `l < 1 mm` and the default `standing_wave` is fatal in Bmad ("infinite
  pondermotive kick"); thin cavities are therefore written `l = 0,
  cavity_type = traveling_wave`, which reproduces `ΔE = V·cos(2π·phi0)` exactly but has **no
  transverse RF kick** (HELIX/TraceWin give R21 ≈ 0.6–0.8 for 300 kV gaps at −30°).
* Longitudinal basis `(z, pz)` with z ahead-positive; p0 follows `lcavity`.
* Bmad's proton mass (938 272 089.43 eV) differs from CODATA-2018 by 1.35e-9; the adapter
  reports Bmad's own `mass_of()`.
* `k0l` needs `k0l_status = straight_reference`.  Bmad's `bmad_to_mad_sad_elegant -madx`
  writes `ksl = −n!·a_n` (skew sign reversed); lattix follows the engines.

## Known limits

* Thin cavities lose the transverse RF focusing (see above).
* `grid_field` files are not converted yet; controllers are kept as directives.
* `ele_twiss` is empty unless `beginning[…]` is set — the adapter sets it.

## Oracle

`lattix/oracles/pytao.py` runs a worker (`python -I bmad_worker.py`) inside the Bmad conda
environment (`LATTIX_BMAD_ENV`, default `bmad`) and reads `ele_mat6`, `lat_list`, floor and
Twiss data over JSON; marker `oracle_bmad`.  Gate A2 (Bmad leg) agrees with cpymad to
2.8e-14, A3 on the PIP-II MEBT to 9.9e-14.
