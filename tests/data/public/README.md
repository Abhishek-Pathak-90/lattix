# Public lattice samples vendored for the lattix test-suite

Every file below was copied verbatim from a public upstream whose license permits
redistribution (checked in the clone before copying; the license text is in
`LICENSES/<source>.txt`).  Upstream clones live in
`/Users/abhishekpathak/Desktop/Projects/particle_tracking_codes/`; the commit column is
`git rev-parse HEAD` of that clone at vendoring time (2026-09-03).  The **Format** column
is what `lattix.corpus.sniff_format` must return for the file
(`tests/corpus/test_public_samples.py` enforces it).  Total size is kept under 5 MB.

Paths in the tables are relative to `tests/data/public/`; upstream paths are relative
to the clone root.

## FLAME

- Upstream: <https://github.com/frib-high-level-controls/FLAME> (`tier1_linac/FLAME`)
- Commit: `584156d13b5c0663c55e21c3d92afc1a310c1f48`
- License: MIT, Board of Trustees of Michigan State University (`LICENSES/FLAME.txt`)

| File | Format | Upstream path |
|---|---|---|
| `flame/LS1FS1_lattice.lat` | flame | `examples/LS1FS1_lattice.lat` |
| `flame/ALL_lattice.lat` | flame | `python/flame/test/ALL_lattice.lat` |
| `flame/LS1.lat` | flame | `python/flame/test/LS1.lat` |
| `flame/FrontEnd.lat` | flame | `python/flame/test/FrontEnd.lat` |
| `flame/TMtest.lat` | flame | `python/flame/test/TMtest.lat` |
| `flame/parse1.lat` | flame | `python/flame/test/parse1.lat` (two-line parser fixture; no FLAME globals, classified by the `.lat` suffix prior) |

## LightWin

- Upstream: <https://github.com/AdrienPlacais/LightWin> (`tier1_linac/LightWin`)
- Commit: `4c96aaf2a12e1dbfd514865321331c13ad57cc82`
- License: MIT, A. Plaçais et al. (`LICENSES/LightWin.txt`)

| File | Format | Upstream path |
|---|---|---|
| `lightwin/example.dat` | tracewin | `data/example/example.dat` |
| `lightwin/field_maps_1D/beta065_1D.edz` | fieldmap | `data/example/field_maps_1D/beta065_1D.edz` |
| `lightwin/field_maps_1D/Double_spoke.edz` | fieldmap | `data/example/field_maps_1D/Double_spoke.edz` |
| `lightwin/field_maps_1D/Simple_Spoke_1D.edz` | fieldmap | `data/example/field_maps_1D/Simple_Spoke_1D.edz` |
| `lightwin/instructions_test/bigger_repeat_ele.dat` | tracewin | `src/lightwin/data/instructions_test/bigger_repeat_ele.dat` |
| `lightwin/instructions_test/repeat_ele.dat` | tracewin | `src/lightwin/data/instructions_test/repeat_ele.dat` |
| `lightwin/instructions_test/set_sync_phase.dat` | tracewin | `src/lightwin/data/instructions_test/set_sync_phase.dat` |
| `lightwin/instructions_test/superpose_map.dat` | tracewin | `src/lightwin/data/instructions_test/superpose_map.dat` |

## ImpactX

- Upstream: <https://github.com/BLAST-ImpactX/impactx> (`tier1_linac/impactx`)
- Commit: `472938dfb6eea2c1dd1ddd2a55f56076e7c436c7`
- License: BSD-3-Clause-LBNL, The Regents of the University of California (`LICENSES/ImpactX.txt`, `LICENSES/ImpactX-NOTICE.txt`)

| File | Format | Upstream path |
|---|---|---|
| `impactx/fodo.madx` | madx | `examples/fodo/fodo.madx` |
| `impactx/input_fodo.in` | impactx | `examples/fodo/input_fodo.in` |
| `impactx/chicane.madx` | madx | `examples/chicane/chicane.madx` |
| `impactx/hvkicker.madx` | madx | `examples/kicker/hvkicker.madx` |
| `impactx/kicker.madx` | madx | `examples/kicker/kicker.madx` |
| `impactx/solenoid.madx` | madx | `examples/solenoid/solenoid.madx` |
| `impactx/dogleg.madx` | madx | `examples/dogleg/dogleg.madx` |
| `impactx/fodo.pals.yaml` | pals | `examples/pals/fodo.pals.yaml` |

## IMPACT-Z

- Upstream: <https://github.com/impact-lbl/IMPACT-Z> (`tier1_linac/IMPACT-Z`)
- Commit: `0c153011b4e1f6f40388d9001957e7c906e8601f`
- License: BSD (LBNL), The Regents of the University of California (`LICENSES/IMPACT-Z.txt`)

| File | Format | Upstream path |
|---|---|---|
| `impactz/Example1/ImpactZ.in` | impactz | `examples/Example1/ImpactZ.in` |
| `impactz/Example2/ImpactZ.in` | impactz | `examples/Example2/ImpactZ.in` |
| `impactz/Example3/ImpactZ.in` | impactz | `examples/Example3/ImpactZ.in` |

## xtrack

- Upstream: <https://github.com/xsuite/xtrack> (`tier2_modern/xtrack`)
- Commit: `09d157188d7da1cf2ca8f41a53617405e29de7d9`
- License: Apache-2.0 (`LICENSES/xtrack.txt`)
- Not vendored: `test_data/hllhc15_noerrors_nobb/sequence.madx` (5.2 MB, over the 2 MB cap).

| File | Format | Upstream path |
|---|---|---|
| `xtrack/psb.seq` | madx | `test_data/psb_chicane/psb.seq` |
| `xtrack/elena.seq` | madx | `test_data/elena/elena.seq` |

## PyORBIT3

- Upstream: <https://github.com/PyORBIT-Collaboration/PyORBIT3> (`tier3_peers/PyORBIT3`)
- Commit: `22b45fa5674e6184168a5e940487e9d7fb0e5f57`
- License: MIT, PyOrbit Collaboration (`LICENSES/PyORBIT3.txt`)

| File | Format | Upstream path |
|---|---|---|
| `pyorbit3/fodo.lat` | madx | `examples/Matching/fodo.lat` (MAD-X dialect despite `.lat`) |
| `pyorbit3/sis18.lat` | madx | `tests/examples/orbit_correction/sis18.lat` (MAD-X dialect despite `.lat`) |

## HELIX

- Upstream: HELIX_v3 snapshot at `/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3`
- Commit: `71ac6be19e195741c6ff71afcad20a9b24a1c9f9`
- License: HELIX examples, (c) Abhishek Pathak, redistributed with permission of the author (`LICENSES/HELIX.txt`)

| File | Format | Upstream path |
|---|---|---|
| `helix/fodo.madx` | madx | `examples/madx/fodo.madx` |
| `helix/fodo.bmad` | bmad | generated from `helix/fodo.madx` with Bmad's own converter `util_programs/mad_to_bmad/madx_to_bmad.py` (bmad-ecosystem clone), for the Phase-0 multi-engine gate |
| `helix/transport.madx` | madx | `examples/madx/transport.madx` |
| `helix/fodo_cell.dat` | tracewin | `examples/fodo_cell.dat` |
| `helix/bend_line.dat` | tracewin | `examples/bend_line.dat` |
| `helix/dtl_section.dat` | tracewin | `examples/dtl_section.dat` |
| `helix/csr_chicane.dat` | tracewin | `examples/csr_chicane.dat` |
| `helix/solenoid_channel.dat` | tracewin | `examples/solenoid_channel.dat` |
| `helix/halo_fodo.dat` | tracewin | `examples/halo_fodo.dat` |
| `helix/mebt_line.dat` | tracewin | `examples/mebt_line.dat` |
| `helix/matching_demo.dat` | tracewin | `examples/matching_demo.dat` |

## Cite-only (NOT vendored)

| Source | Why | Reference path |
|---|---|---|
| cheetah | GPL-3.0 (`tier2_modern/cheetah/LICENSE`) — incompatible with this BSD-3 repo | `tier2_modern/cheetah` @ `e78368547cdbda9937fb467278f0b88454cfb493` |
| bmad-ecosystem | no top-level LICENSE; `bmad/Copyright` states GPL-3.0, so not clearly permissive | `tier3_peers/bmad-ecosystem/regression_tests/write_foreign_test/` (`write_foreign_test.bmad` + `lte.correct`, `mad8.correct`, `madx.correct`, `sad.correct`) @ `8073acbe96e7aa9b29222a69669a8b7d9c4a5fc8` |
