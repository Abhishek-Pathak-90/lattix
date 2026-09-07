"""Cross-engine comparison of OracleResults (PLAN §5.2 metrics).

Engines disagree on element granularity (MAD-X implicit drifts, HELIX explicit
drifts, xtrack markers), so cumulative maps are compared at *shared s
boundaries* rather than row by row.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lattix.oracles.base import OracleResult
from lattix.oracles.basis import momentum_eV, rescale_to_constant_p0


@dataclass
class PairComparison:
    a: str
    b: str
    n_shared: int
    length_a: float
    length_b: float
    max_rcum_abs: float          # max |R_cum,a − R_cum,b| at shared boundaries (common basis, const-p0)
    max_rcum_rel: float          # same, relative to max |R̂_cum,a|
    final_abs: float             # at the last shared boundary
    energy_rel: float            # max relative difference of reference kinetic energy at shared boundaries
    survey_end_abs: float | None
    blocks: dict[str, float] = field(default_factory=dict)   # max abs diff per block, see BLOCKS
    notes: list[str] = field(default_factory=list)
    per_boundary: list[tuple[float, dict[str, float]]] = field(default_factory=list)  # (s, block diffs)

    def row(self) -> str:
        se = "n/a" if self.survey_end_abs is None else f"{self.survey_end_abs:.2e}"
        blk = "  ".join(f"{k}={v:.1e}" for k, v in self.blocks.items())
        return (f"{self.a:>8} vs {self.b:<8} shared={self.n_shared:4d}  "
                f"L={self.length_a:.9g}/{self.length_b:.9g}  "
                f"maxΔRcum={self.max_rcum_abs:.2e} (rel {self.max_rcum_rel:.2e})  "
                f"final={self.final_abs:.2e}  "
                f"ΔW/W={self.energy_rel:.2e}  survey_end={se}\n"
                f"{'':>21} blocks: {blk}")


#: named sub-blocks of the 6×6 map (row slice, column slice)
BLOCKS = {
    "T4x4": (slice(0, 4), slice(0, 4)),        # transverse
    "disp": (slice(0, 4), slice(5, 6)),        # R16 R26 R36 R46
    "path": (slice(4, 5), slice(0, 4)),        # R51 R52 R53 R54 (path length vs transverse)
    "R56": (slice(4, 5), slice(5, 6)),
    "R5x_z": (slice(4, 5), slice(4, 5)),       # R55
    "E_row": (slice(5, 6), slice(0, 6)),       # R61..R66 (energy kicks)
    "z_col": (slice(0, 4), slice(4, 5)),       # R15 R25 R35 R45 (transverse vs z: RF)
}


def _last_index_per_s(s: np.ndarray, tol: float) -> dict[int, int]:
    """Map a quantised s to the LAST row at that position, so zero-length
    elements (edges, markers, thin kicks) sharing an s are all included."""
    out: dict[int, int] = {}
    for i, v in enumerate(s):
        out[int(round(v / tol))] = i
    return out


def _is_thin_kick(r: OracleResult, i: int) -> bool:
    """Zero-length row whose map is not the identity (edge, gap, kicker)."""
    return r.length[i] <= 1e-12 and not np.allclose(r.R_elem[i], np.eye(6), atol=1e-12)


def shared_boundaries(a: OracleResult, b: OracleResult, tol: float = 1e-9) -> list[tuple[int, int]]:
    """Index pairs (ia, ib) whose exit positions coincide within *tol* metres.

    Only *unambiguous* boundaries are kept: the downstream-most row at that s
    in both engines must be a thick element or an identity marker.  Engines
    attribute thin kicks differently (MAD-X folds bend edges into the bend
    map, HELIX/TraceWin emit separate EDGE rows), so a boundary that ends
    in a thin kick in either engine is not comparable.
    """
    la, lb = _last_index_per_s(a.s_out, tol), _last_index_per_s(b.s_out, tol)
    return [(la[k], lb[k]) for k in sorted(la)
            if k in lb and not _is_thin_kick(a, la[k]) and not _is_thin_kick(b, lb[k])]


def compare_pair(a: OracleResult, b: OracleResult, tol_s: float = 1e-9) -> PairComparison:
    ca, cb = a.to_common(), b.to_common()
    pairs = shared_boundaries(ca, cb, tol_s)
    notes = []
    if not pairs:
        return PairComparison(a.engine, b.engine, 0, a.total_length, b.total_length,
                              np.nan, np.nan, np.nan, np.nan, None, {}, ["no shared s boundaries"])
    Ra, Rb = ca.R_cum, cb.R_cum
    pa0 = momentum_eV(ca.ref_kinetic_eV_in[0], ca.mass_eV)
    pb0 = momentum_eV(cb.ref_kinetic_eV_in[0], cb.mass_eV)
    diffs, rels = [], []
    scale = 0.0
    blocks = {k: 0.0 for k in BLOCKS}
    per_boundary: list[tuple[float, dict[str, float]]] = []
    n_nan = 0
    for ia, ib in pairs:
        ma = rescale_to_constant_p0(Ra[ia], pa0, momentum_eV(ca.ref_kinetic_eV_out[ia], ca.mass_eV))
        mb = rescale_to_constant_p0(Rb[ib], pb0, momentum_eV(cb.ref_kinetic_eV_out[ib], cb.mass_eV))
        if not (np.all(np.isfinite(ma)) and np.all(np.isfinite(mb))):
            n_nan += 1                       # an engine reported this span without a map: nothing to compare
            continue
        dm = np.abs(ma - mb)
        scale = max(scale, float(np.max(np.abs(ma))), float(np.max(np.abs(mb))))
        d = float(np.max(dm))
        diffs.append(d)
        rels.append(d / max(np.max(np.abs(ma)), 1e-300))
        here = {k: float(np.max(dm[r, c])) for k, (r, c) in BLOCKS.items()}
        for k, v in here.items():
            blocks[k] = max(blocks[k], v)
        per_boundary.append((float(ca.s_out[ia]), here))
    ea = ca.ref_kinetic_eV_out[[p[0] for p in pairs]]
    eb = cb.ref_kinetic_eV_out[[p[1] for p in pairs]]
    energy_rel = float(np.max(np.abs(ea - eb) / np.maximum(np.abs(ea), 1e-300)))
    if n_nan:
        notes.append(f"{n_nan} shared boundary(ies) without a map (NaN) skipped")
    if not diffs:
        return PairComparison(a.engine, b.engine, 0, a.total_length, b.total_length,
                              np.nan, np.nan, np.nan, np.nan, None, {}, [*notes, "no shared boundaries with maps"])
    sv = None
    if a.survey is not None and b.survey is not None:
        ia, ib = pairs[-1]
        sv = float(np.max(np.abs(a.survey[ia, :3] - b.survey[ib, :3])))
    if abs(a.total_length - b.total_length) > tol_s:
        notes.append("total length differs")
    if scale > 1e6:
        notes.append(f"cumulative maps reach {scale:.1e}: the line is unstable at this reference — check the "
                     "species and energy the decks were read with")
    return PairComparison(a.engine, b.engine, len(pairs), a.total_length, b.total_length,
                          float(np.max(diffs)), float(np.max(rels)), float(diffs[-1]),
                          energy_rel, sv, blocks, notes, per_boundary=per_boundary)


def compare_all(results: dict[str, OracleResult]) -> list[PairComparison]:
    names = list(results)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append(compare_pair(results[names[i]], results[names[j]]))
    return out
