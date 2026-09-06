"""Before/after semantics for the page: align the source elements with the elements of the written deck
read back, diff their physical contributions quantity by quantity against what the ledger explains, and
summarise the translation the way the battery does (:func:`lattix.crossval.ir_roundtrip`)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lattix.crossval import (
    AFFECTS,
    QUANTITIES,
    RoundTripSettings,
    contrib,
    ir_roundtrip,
    profile,
)
from lattix.fidelity import FidelityReport
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.walk import propagate

CLASS_RANK = {"EXACT": 0, "EQUIVALENT": 1, "LOSSY": 2, "DROPPED": 3, "DIFF": 4}
_DERIVED = [r"_rfdefocus$", r"_D[12]$", r"_gap$", r"_body$", r"_aper_(?:in|out)\d*$", r"_(?:in|out|rf)$", r"_k$"]
_UNIQ = r"_\d+$"
_KIND_CLASS = {
    "Drift": "drift", "Quadrupole": "magnet", "Sextupole": "magnet", "Octupole": "magnet", "Multipole": "magnet",
    "Bend": "bend", "Solenoid": "solenoid", "RFCavity": "rf", "FieldMap": "rf", "NCells": "rf", "RFQCell": "rf",
    "Superposition": "rf", "Kicker": "kick", "Taylor": "map", "Collimator": "thin", "Marker": "thin",
    "Instrument": "thin", "Foil": "thin", "Patch": "thin", "ReferenceChange": "thin", "Freq": "thin",
    "Directive": "thin",
}
_DEGRADES_TO = {
    "rf": {"rf", "drift", "solenoid", "magnet", "thin"}, "magnet": {"magnet", "kick", "drift", "thin"},
    "kick": {"kick", "magnet", "thin"}, "map": {"map", "drift", "thin"}, "thin": {"thin", "drift"},
    "solenoid": {"solenoid", "thin", "drift"}, "bend": {"bend", "drift"}, "drift": {"drift", "thin"},
}
#: absolute floors of the per-quantity comparison (crossval uses rtol 1e-9 on max(|a|, |b|, floor))
_FLOORS = {"length": 1e-9, "gain": 1e3, "volt": 1e3, "energy": 1.0}
_ATOL = {"length": 1e-9}
RTOL = 1e-9


@dataclass
class Aligned:
    i: int
    name: str
    kind: str
    s_in: float
    s_out: float
    occurrence: int
    dst: list[int] = field(default_factory=list)
    match: str = "none"                 # name | overlap | nearest | none
    dropped: bool = False               # the writer's ledger says DROPPED
    absent: bool = False                # nothing to align: a zero-length element without physics
    roles: dict[int, str] = field(default_factory=dict)


def _sanitize(name: str) -> str:
    """The identifier most writers make of a name (lower case, every other character an underscore)."""
    return re.sub(r"[^0-9a-z_]", "_", name.lower())


class _Resolver:
    def __init__(self, src: list[Placed]) -> None:
        self.exact = {p.name for p in src}
        self.lower = {p.name.lower(): p.name for p in src}
        self.sanitized = {_sanitize(p.name): p.name for p in src}

    def lookup(self, cand: str) -> str | None:
        if cand in self.exact:
            return cand
        return self.lower.get(cand.lower()) or self.sanitized.get(_sanitize(cand))

    def resolve(self, cand: str) -> tuple[str | None, bool]:
        """``(source name, stripped)``: the longest form of ``cand`` (≤ 3 suffix strips) that names a source
        element; an exact name beats any stripped form."""
        hit = self.lookup(cand)
        if hit is not None:
            return hit, False
        frontier, seen = [cand], []
        for _ in range(3):
            nxt = []
            for c in frontier:
                for pat in [*_DERIVED, _UNIQ]:
                    s = re.sub(pat, "", c)
                    if s and s != c and s not in seen:
                        seen.append(s)
                        nxt.append(s)
            frontier = nxt
        for c in sorted(seen, key=len, reverse=True):
            hit = self.lookup(c)
            if hit is not None:
                return hit, True
        return None, False


def _target_key(q: Placed, res: _Resolver) -> tuple[str | None, str]:
    e = q.element
    if (e.meta or {}).get("rf_focusing_of"):
        return res.resolve(str(e.meta["rf_focusing_of"]))[0], "rf_focusing"
    implicit = (e.native.get("impactt") or {}).get("implicit")
    if e.kind == "Drift" and (implicit or re.search(r"_D[12]$|_gap$|^gap_\d+$", e.name)):
        return None, "padding"
    cands = []
    if e.provenance is not None and e.provenance.original_name:
        cands.append(e.provenance.original_name)
    cands.append(e.name)
    for c in cands:
        r, stripped = res.resolve(c)
        if r is not None:
            role = "part" if stripped and re.search("|".join(_DERIVED), c) else "primary"
            if e.kind == "Drift" and re.search(r"_D[12]$|_gap$", c):
                role = "padding"
            return r, role
    return None, "added"


def _compatible(a: Placed, b: Placed) -> bool:
    ca, cb = _KIND_CLASS.get(a.element.kind, "thin"), _KIND_CLASS.get(b.element.kind, "thin")
    return cb == ca or cb in _DEGRADES_TO.get(ca, set())


def align(src: list[Placed], dst: list[Placed], rep_w: FidelityReport, rep_r: FidelityReport | None,
          settings: RoundTripSettings) -> dict:
    """Source elements → clusters of target elements by resolved name, then by s (see the module doc)."""
    res = _Resolver(src)
    keys = [_target_key(q, res) for q in dst]
    # clusters: maximal runs of consecutive targets with the same resolved key
    clusters: dict[str, list[list[int]]] = {}
    roles: dict[int, str] = {}
    j = 0
    while j < len(dst):
        key, role = keys[j]
        roles[j] = role
        if key is None:
            j += 1
            continue
        run = [j]
        k = j + 1
        while k < len(dst) and keys[k][0] == key:
            roles[k] = keys[k][1]
            run.append(k)
            k += 1
        clusters.setdefault(key, []).append(run)
        j = k
    occ: dict[str, list[int]] = {}
    for p in src:
        occ.setdefault(p.name, []).append(p.index)
    dropped_names = {e.element for e in rep_w.entries if e.cls.value == "DROPPED" and e.element}
    out: dict[int, Aligned] = {}
    counts_occ: dict[str, int] = {}
    for p in src:
        counts_occ[p.name] = counts_occ.get(p.name, 0) + 1
        out[p.index] = Aligned(p.index, p.name, p.element.kind, p.s_in, p.s_out, counts_occ[p.name])
    claimed: set[int] = set()

    def tol_s(p: Placed) -> float:
        return max(1e-9, settings.fuzzy.get(p.name, 0.0))

    def claim(i: int, run: list[int], match: str) -> None:
        a = out[i]
        a.dst = list(run)
        a.match = match
        for jj in run:
            a.roles[jj] = roles.get(jj, "primary")
            claimed.add(jj)

    for name, S in occ.items():
        C = [c for c in clusters.get(name, []) if not any(jj in claimed for jj in c)]
        if len(S) == len(C):
            for i, c in zip(S, C, strict=True):
                claim(i, c, "name")
        else:
            for i in S:
                p = src[i]
                free = [c for c in C if not any(jj in claimed for jj in c)]
                if not free:
                    break
                c = min(free, key=lambda c: abs(dst[c[0]].s_in - p.s_in))
                if abs(dst[c[0]].s_in - p.s_in) <= max(tol_s(p), 0.5 * max(p.length, dst[c[0]].length)):
                    claim(i, c, "name")
    # a thick source element absorbs the unclaimed targets lying inside its span (the writer's unnamed
    # padding drifts around a thick cavity written thin, ladder parts without the name convention)
    for p in src:
        a = out[p.index]
        if not a.dst or p.length <= 1e-12:
            continue
        tol = tol_s(p)
        extra = [q.index for q in dst if q.index not in claimed and q.s_in >= p.s_in - tol
                 and q.s_out <= p.s_out + tol and (q.element.kind == "Drift" or _compatible(p, q))]
        for jj in extra:
            a.dst.append(jj)
            a.roles[jj] = "padding" if dst[jj].element.kind == "Drift" else "part"
            claimed.add(jj)
        a.dst.sort()
    for p in src:
        a = out[p.index]
        if a.dst:
            continue
        tol = tol_s(p)
        if p.length > 1e-12:
            # a drift, or an element the target holds as one (a zero-angle bend, a thick collimator, a
            # neutralised cavity …): it may share the merged drift the reader created over several of them
            drop = settings.neutral.get(p.name, set())
            drift = p.element.kind == "Drift" or all(
                v == 0.0 for q, v in contrib(p).items() if q != "length" and not ("*" in drop or q in drop))
            js = [q.index for q in dst
                  if (q.index not in claimed or (drift and q.element.kind == "Drift"))
                  and (_compatible(p, q) or (drift and q.element.kind == "Drift"))
                  and min(p.s_out, q.s_out) - max(p.s_in, q.s_in) > tol]
            if js:
                claim(p.index, js, "overlap")
                continue
        else:
            js = [q.index for q in dst if q.index not in claimed and _compatible(p, q)
                  and abs(q.s_in - p.s_in) <= tol and q.length <= 1e-12]
            if js:
                claim(p.index, [min(js, key=lambda jj: abs(dst[jj].s_in - p.s_in))], "nearest")
                continue
        a.dropped = p.name in dropped_names
        c = contrib(p)
        # nothing to align: a zero-length element without any physical contribution (a directive, a marker,
        # a diagnostic, an empty multipole or kicker, …)
        a.absent = p.length <= 1e-12 and all(v == 0.0 for v in c.values()) and p.element.kind != "ReferenceChange"
    # a thick element without physics of its own (a drift, a limit-less collimator, a zero-angle bend, a
    # neutralised cavity …) written thin or merged: the target drifts overlapping its span carry its length
    for p in src:
        a = out[p.index]
        if p.length <= 1e-12:
            continue
        drop = settings.neutral.get(p.name, set())
        if not all(v == 0.0 for q, v in contrib(p).items() if q != "length" and not ("*" in drop or q in drop)):
            continue
        tol = tol_s(p)
        for q in dst:
            if q.element.kind == "Drift" and q.index not in a.dst \
                    and min(p.s_out, q.s_out) - max(p.s_in, q.s_in) > tol:
                a.dst.append(q.index)
                a.roles[q.index] = "padding"
                claimed.add(q.index)
                if a.match == "none":
                    a.match = "overlap"
        a.dst.sort()
    for jj, role in roles.items():
        if jj not in claimed and role not in ("padding", "rf_focusing"):
            roles[jj] = "added"
    targets = []
    src_of: dict[int, list[int]] = {}
    for a in out.values():
        for jj in a.dst:
            src_of.setdefault(jj, []).append(a.i)
    for q in dst:
        targets.append({"j": q.index, "name": q.name, "kind": q.element.kind, "s_in": q.s_in, "s_out": q.s_out,
                        "src": src_of.get(q.index, []),
                        "role": roles.get(q.index, "primary") if q.index in claimed else roles.get(q.index, "added"),
                        "source_key": keys[q.index][0]})
    counts = {"name": 0, "overlap": 0, "nearest": 0, "none": 0, "dropped": 0, "absent": 0, "padding": 0,
              "rf_focusing": 0, "added": 0}
    for a in out.values():
        counts[a.match] += 1
        counts["dropped"] += int(a.dropped)
        counts["absent"] += int(a.absent and not a.dropped)
    for t in targets:
        if t["role"] in ("padding", "rf_focusing", "added") and not t["src"]:
            counts[t["role"]] += 1
    elements = [{"i": a.i, "name": a.name, "kind": a.kind, "s_in": a.s_in, "s_out": a.s_out,
                 "occurrence": a.occurrence, "dst": a.dst, "match": a.match, "dropped": a.dropped,
                 "absent": a.absent, "roles": {str(k): v for k, v in a.roles.items()}} for a in out.values()]
    return {"elements": elements, "targets": targets, "counts": counts}


def _within(q: str, a: float, b: float, tol_extra: float = 0.0) -> bool:
    scale = max(_FLOORS.get(q, 1.0 if q not in ("length",) else 1e-9), abs(a), abs(b))
    return abs(a - b) <= RTOL * scale + _ATOL.get(q, 1e-12) + tol_extra


def _entries_for(names: set[str], *reports: FidelityReport | None) -> list[dict]:
    out = []
    for side, rep in zip(("writer", "reader"), reports, strict=False):
        if rep is None:
            continue
        for e in rep.entries:
            if e.element in names:
                out.append({"side": side, "cls": e.cls.value, "code": e.code, "message": e.message,
                            "details": dict(e.details), "element": e.element})
    return out


def element_diffs(src: list[Placed], dst: list[Placed], alignment: dict, rep_w: FidelityReport,
                  rep_r: FidelityReport | None, settings: RoundTripSettings, lat: Lattice, lat2: Lattice) -> list[dict]:
    """Per source element: every contribution quantity, the exit energy and the entrance position compared
    with the aligned cluster, classified equal | explained | unexplained | suspended, plus the ledger."""
    pr_s = profile(lat, settings.neutral)
    pr_d = profile(lat2, settings.neutral)
    csrc = [contrib(p) for p in src]
    cdst = [contrib(q) for q in dst]
    src_of: dict[int, list[int]] = {}
    for a in alignment["elements"]:
        for j in a["dst"]:
            src_of.setdefault(j, []).append(a["i"])
    diffs = []
    fuzzy_names = set(settings.fuzzy)
    for a in alignment["elements"]:
        i = a["i"]
        p = src[i]
        cluster = a["dst"]
        names = {p.name} | {dst[j].name for j in cluster}
        ledger = _entries_for(names, rep_w, rep_r)
        explained_q: set[str] = set(settings.neutral.get(p.name, set()))
        for j in cluster:
            explained_q |= settings.neutral.get(dst[j].name, set())
        lossy_codes = [e for e in ledger if e["cls"] in ("LOSSY", "DROPPED")]
        gone = a["match"] == "none" and (a["dropped"] or a.get("absent"))     # nothing to compare with
        quantities: dict[str, dict] = {}
        target = {q: sum(cdst[j][q] for j in cluster) for q in QUANTITIES if q != "energy"}
        if any(len(src_of.get(j, [])) > 1 for j in cluster):
            # a target drift shared by several source drifts (MAD-X implicit drifts): its length is the overlap
            target["length"] = sum(max(0.0, min(p.s_out, dst[j].s_out) - max(p.s_in, dst[j].s_in))
                                   if len(src_of.get(j, [])) > 1 else cdst[j]["length"] for j in cluster)
        for q in QUANTITIES:
            if q == "energy":
                continue
            va, vb = csrc[i][q], target[q]
            if q in settings.skip:
                cls, codes = "suspended", []
            elif _within(q, va, vb):
                cls, codes = "equal", []
            elif "*" in explained_q or q in explained_q:
                cls, codes = "explained", _codes_for(lossy_codes, q)
            elif q == "length" and abs(va - vb) <= _fuzzy_window(src, i, settings):
                cls, codes = "explained", _fuzzy_codes(src, i, settings, rep_w)
            elif a["match"] == "none":
                cls, codes = ("explained", _codes_for(lossy_codes, q)) if gone else ("unexplained", [])
            else:
                cls, codes = "unexplained", []
            if va or vb or cls != "equal":
                quantities[q] = {"src": va, "dst": vb, "delta": vb - va, "class": cls, "codes": codes}
        # energy at the exit (state), both sides with the same neutralised gains subtracted
        e_src = float(pr_s.energy[i])
        j_last = max(cluster) if cluster else None
        e_dst = float(pr_d.energy[j_last]) if j_last is not None else None
        if e_dst is None:
            e_cls, e_codes = ("equal", []) if gone else ("unexplained", [])
        elif _within("energy", e_src, e_dst):
            e_cls, e_codes = "equal", []
        elif settings.skip_energy:
            e_cls = "explained"
            e_codes = sorted({e["code"] for e in _entries_for(set(lat.elements), rep_w)
                              if e["code"].startswith("CONST_P0") or e["code"] == "REFCHANGE_DROPPED"})
        elif pr_s.e_dropped is not None and pr_s.e_dropped[i]:
            e_cls, e_codes = "explained", ["upstream gain neutralised"]
        else:
            e_cls, e_codes = "unexplained", []
        energy = {"src_out_eV": e_src, "dst_out_eV": e_dst, "delta": (e_dst - e_src) if e_dst is not None else None,
                  "class": e_cls, "codes": e_codes}
        # entrance position
        s_dst = min((dst[j].s_in for j in cluster), default=None)
        tol = max(1e-9, settings.fuzzy.get(p.name, 0.0), _fuzzy_window(src, i, settings))
        shared = any(len(src_of.get(j, [])) > 1 for j in cluster)
        if s_dst is None:
            p_cls = "equal" if gone else "unexplained"
        elif abs(s_dst - p.s_in) <= tol:
            p_cls = "equal"
        elif shared and s_dst <= p.s_in + tol and max(dst[j].s_out for j in cluster) >= p.s_out - tol:
            p_cls = "equal"                    # covered by a merged drift
        else:
            upstream = [d for d in diffs if d["quantities"].get("length", {}).get("class") == "explained"]
            shifted = upstream or any(src[m].name in fuzzy_names for m in range(i))
            p_cls = "explained" if shifted else "unexplained"
        position = {"src_s_in": p.s_in, "dst_s_in": s_dst, "delta": (s_dst - p.s_in) if s_dst is not None else None,
                    "tol": tol, "class": p_cls}
        worst = "EXACT"
        for e in ledger:
            if CLASS_RANK[e["cls"]] > CLASS_RANK[worst]:
                worst = e["cls"]
        unexplained = any(v["class"] == "unexplained" for v in quantities.values()) or e_cls == "unexplained" \
            or p_cls == "unexplained" or (a["match"] == "none" and not gone)
        if unexplained:
            worst = "DIFF"
        diffs.append({"i": i, "quantities": quantities, "energy": energy, "position": position, "ledger": ledger,
                      "worst": worst})
    return diffs


def _codes_for(lossy_codes: list[dict], q: str) -> list[str]:
    return sorted({e["code"] for e in lossy_codes
                   if "*" in AFFECTS.get(e["code"], {"*"}) or q in AFFECTS.get(e["code"], {"*"})})


def _fuzzy_neighbours(src: list[Placed], i: int, settings: RoundTripSettings) -> list[Placed]:
    """The fuzzy elements (a thin gap written as a short cavity, …) whose position lies within their own
    window of either end of element ``i``: their surrogate length was taken from ``i``."""
    p = src[i]
    out = []
    for q in src:
        w = settings.fuzzy.get(q.name, 0.0)
        if w and (abs(q.s_in - p.s_in) <= w or abs(q.s_in - p.s_out) <= w or abs(q.s_out - p.s_in) <= w):
            out.append(q)
    return out


def _fuzzy_window(src: list[Placed], i: int, settings: RoundTripSettings) -> float:
    return max((settings.fuzzy.get(q.name, 0.0) for q in _fuzzy_neighbours(src, i, settings)), default=0.0)


def _fuzzy_codes(src: list[Placed], i: int, settings: RoundTripSettings, rep_w: FidelityReport) -> list[str]:
    from lattix.crossval import FUZZY_S

    names = {q.name for q in _fuzzy_neighbours(src, i, settings)}
    return sorted({e.code for e in rep_w.entries if e.code in FUZZY_S and e.element in names})


def compare_translation(lat: Lattice, rep_w: FidelityReport, lat2: Lattice, rep_r: FidelityReport, *,
                        src_fmt: str, dst_fmt: str, placed: list[Placed] | None = None,
                        placed2: list[Placed] | None = None, fixed_point: dict | None = None) -> dict:
    """The page's comparison payload: the battery's round-trip verdict, the alignment, the per-element
    diffs and the summary counts."""
    src = placed if placed is not None else propagate(lat)
    dst = placed2 if placed2 is not None else propagate(lat2)
    rt = ir_roundtrip(lat, rep_w, lat2, rep_r)
    alignment = align(src, dst, rep_w, rep_r, rt.settings)
    diffs = element_diffs(src, dst, alignment, rep_w, rep_r, rt.settings, lat, lat2)
    worst_counts = {k: 0 for k in CLASS_RANK}
    for d in diffs:
        worst_counts[d["worst"]] += 1
    ref_in, ref_out = lat.reference, (src[-1].ref_out if src and src[-1].ref_out else lat.reference)
    ref2_in, ref2_out = lat2.reference, (dst[-1].ref_out if dst and dst[-1].ref_out else lat2.reference)
    return {
        "source": {"format": src_fmt, "species": ref_in.species.name, "ke_in_eV": ref_in.kinetic_energy_eV,
                   "ke_out_eV": ref_out.kinetic_energy_eV, "n_placed": len(src),
                   "total_length_m": src[-1].s_out if src else 0.0},
        "target": {"format": dst_fmt, "species": ref2_in.species.name, "ke_in_eV": ref2_in.kinetic_energy_eV,
                   "ke_out_eV": ref2_out.kinetic_energy_eV, "n_placed": len(dst),
                   "total_length_m": dst[-1].s_out if dst else 0.0},
        "ledger": {"writer": {"counts": dict(rep_w.counts), "codes": dict(rep_w.codes())},
                   "reader": {"counts": dict(rep_r.counts), "codes": dict(rep_r.codes())}},
        "tier": rt.tier,
        "settings": {"skip_energy": rt.settings.skip_energy, "skip": sorted(rt.settings.skip),
                     "fuzzy": dict(rt.settings.fuzzy), "species_loss": rt.settings.species_loss},
        "ir": {"ok": rt.diff.ok, "worst": rt.diff.worst, "problems": list(rt.diff.problems),
               "n_boundaries": rt.diff.n_boundaries},
        "fixed_point": fixed_point,
        "alignment": alignment, "diffs": diffs,
        "summary": {"n_unexplained": worst_counts["DIFF"], "worst_counts": worst_counts,
                    "delta_length_m": (dst[-1].s_out if dst else 0.0) - (src[-1].s_out if src else 0.0),
                    "delta_ke_out_eV": ref2_out.kinetic_energy_eV - ref_out.kinetic_energy_eV},
    }
