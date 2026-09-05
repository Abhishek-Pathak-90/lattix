"""IR → Cheetah LatticeJSON (see the package docstring for the measured conventions)."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage, with_rf_focusing
from lattix.ir.elements import ALL_KINDS, Element, FieldMap
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.walk import propagate

VERSION_TAG = "cheetah-0.8"
#: EXACT Cheetah element per IR kind, or the degradation the ledger records
_INFO = {
    "Drift": "Drift", "Quadrupole": "Quadrupole (k1 = Bn1/Bρ_signed, tilt, misalignment)",
    "Sextupole": "Sextupole (k2 = Bn2/Bρ_signed)", "Octupole": "Drift (Cheetah has no octupole)",
    "Multipole": "CombinedCorrector for the dipole terms (higher orders dropped)",
    "Bend": "Dipole (arc length, angle, sector-referenced e1/e2, gap = 2·hgap, fringe_integral, tilt, k1)",
    "Solenoid": "Solenoid (k = Bsol/(2 Bρ_signed))", "RFCavity": "Cavity (voltage = −V/q, phase in degrees)",
    "FieldMap": "the lattix.ir.fieldmap replacement ladder (Cavity / Solenoid / Quadrupole / Drift)",
    "NCells": "Drift", "RFQCell": "Drift", "Kicker": "CombinedCorrector (angles in rad)",
    "Collimator": "Aperture (+ Drift for the body)", "Marker": "Marker", "Instrument": "BPM / Screen / Marker",
    "Foil": "Marker", "Taylor": "CustomTransferMap (7×7 in Cheetah's MAD-X-like basis)",
    "Patch": "dropped", "ReferenceChange": "dropped", "Freq": "nothing (the frequency is per cavity)",
    "Directive": "dropped", "Superposition": "children in order",
}
assert set(_INFO) == set(ALL_KINDS)
RULES = dict(_INFO)
_SAFE = re.compile(r"[^A-Za-z0-9_]")
_THIN = 1e-12
#: IR kinds the reader gives back as they were (the others come back as what they were written as)
_RESTORABLE = frozenset({"Drift", "Quadrupole", "Sextupole", "Bend", "Solenoid", "RFCavity", "Kicker", "Collimator",
                         "Marker", "Instrument", "Foil", "Taylor", "Patch", "ReferenceChange", "Freq", "Directive",
                         "Multipole"})
_KIND_OF_TYPE = {"Drift": "Drift", "Marker": "Marker", "Cavity": "RFCavity", "Solenoid": "Solenoid",
                 "Quadrupole": "Quadrupole", "CombinedCorrector": "Kicker", "Aperture": "Collimator"}


def _num(x: float) -> float:
    x = float(x)
    return 0.0 if x == 0 else x


def similarity_diag(m, off, d_from, d_to):
    """A first-order map from the basis with ``x_common = diag(d_from)·x`` to the basis with
    ``x_common = diag(d_to)·x``: ``R'_ij = R_ij · (d_from_i/d_to_i) / (d_from_j/d_to_j)`` — exact
    on the diagonal and on zeros, so an identity stays an identity bit for bit."""
    a = [float(d_from[i]) / float(d_to[i]) for i in range(6)]
    r = [[(float(m[i][j]) if (i == j or m[i][j] == 0) else float(m[i][j]) * a[i] / a[j]) for j in range(6)]
         for i in range(6)]
    o = [(float(off[i]) if off[i] == 0 else float(off[i]) * a[i]) for i in range(6)]
    return r, o


class NameMap:
    """Cheetah names must be Python identifiers (``Segment`` registers them as torch modules)."""

    def __init__(self) -> None:
        self._by_id: dict[int, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}

    def assign(self, original: str, key: object | None = None) -> str:
        if key is not None and id(key) in self._by_id:
            return self._by_id[id(key)]
        base = _SAFE.sub("_", original) or "e"
        if base[0].isdigit():
            base = "e_" + base
        name, n = base, 2
        while name in self._used:
            name = f"{base}_{n}"
            n += 1
        self._used.add(name)
        if key is not None:
            self._by_id[id(key)] = name
        if name != original:
            self.renamed[original] = name
        return name


@dataclass
class _Ctx:
    brho: float
    ref: ReferenceParticle
    rep: FidelityReport
    names: NameMap
    elements: dict
    install_apertures: bool


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport` (Cheetah LatticeJSON)."""

    format = "cheetah"
    RULES = RULES

    def write(self, lattice: Lattice, path, *, strict: bool = False, install_apertures: bool = True,
              title: str | None = None) -> FidelityReport:
        from pathlib import Path

        rep = FidelityReport(target_format="cheetah", target_file=str(path))
        doc = self.to_document(lattice, report=rep, install_apertures=install_apertures, title=title)
        Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, **options) -> str:
        return json.dumps(self.to_document(lattice, **options), indent=1) + "\n"

    def to_document(self, lattice: Lattice, *, report: FidelityReport | None = None,
                    install_apertures: bool = True, title: str | None = None) -> dict:
        rep = report if report is not None else FidelityReport(target_format="cheetah")
        lattice, _ = with_rf_focusing(lattice, "cheetah")
        names = NameMap()
        elements: dict[str, list] = {}
        ctx = _Ctx(lattice.reference.brho_signed, lattice.reference, rep, names, elements, install_apertures)
        order: list[str] = []
        emitted: dict[int, list[str]] = {}
        for p in propagate(lattice):
            el = p.element
            ref = p.ref_in or lattice.reference
            ctx.brho, ctx.ref = ref.brho_signed, ref
            if id(el) in emitted and el.kind not in ("Superposition",):
                order += emitted[id(el)]               # a definition placed again: same names
                continue
            names_here = self._emit(el, ctx)
            emitted[id(el)] = names_here
            order += names_here
        root = _SAFE.sub("_", lattice.use or lattice.name or "root") or "root"
        source = lattice.meta.get("source_format", "IR")
        info = format_reference_tag(lattice.reference, "#")      # provenance lives under "lattix" (scrubbed keys)
        if names.renamed:
            rep.exact(None, None, code="NAMES_SANITIZED",
                      message=f"{len(names.renamed)} name(s) made Python identifiers "
                              "(originals in metadata.lattix.name)")
        return {"version": VERSION_TAG, "title": title or lattice.name or root, "info": info, "root": root,
                "lattix": {"written_by": f"lattix {__version__}", "source_format": source},
                "elements": elements, "lattices": {root: order}}

    # ------------------------------------------------------------------ elements
    def _emit(self, el: Element, ctx: _Ctx) -> list[str]:
        kind = el.kind
        out: list[str] = []
        if kind == "Superposition":
            ctx.rep.lossy("SUPERPOSITION_FLATTENED", "superposed field maps written as consecutive elements",
                          element=el.name, kind=kind)
            for child in el.children:
                out += self._emit(child, ctx)
            return out
        if kind == "FieldMap":
            return self._fieldmap(el, ctx)
        pre, post = self._apertures(el, ctx)
        handler = getattr(self, f"_k_{kind.lower()}")
        body = handler(el, ctx)                       # list of (name, type, params) or None
        return pre + body + post

    def _add(self, ctx: _Ctx, el: Element, ctype: str, params: dict, *, suffix: str = "",
             extra_meta: dict | None = None, key: object | None = None) -> str:
        name = ctx.names.assign(el.name + suffix, key if key is not None else (el if not suffix else None))
        meta = {"kind": el.kind, "name": el.name}
        degraded = el.meta.get("cheetah_degraded_from")
        if el.kind not in _RESTORABLE:
            meta["kind"] = _KIND_OF_TYPE.get(ctype, el.kind)
            degraded = degraded or el.kind
        if degraded:
            meta["degraded_from"] = degraded
        if el.provenance is not None:
            if el.provenance.original_type:
                meta["original_type"] = el.provenance.original_type
        if extra_meta:
            meta.update(extra_meta)
        params = {k: v for k, v in params.items() if v is not None}
        params["metadata"] = {"lattix": meta}
        ctx.elements[name] = [ctype, params]
        return name

    def _misalignment(self, el: Element) -> list[float] | None:
        sh = el.shift
        if sh is None or (not sh.x_offset and not sh.y_offset):
            return None
        return [_num(sh.x_offset), _num(sh.y_offset)]

    def _shift_rest(self, el: Element, ctx: _Ctx, *, tilt_used: bool) -> None:
        sh = el.shift
        if sh is None:
            return
        lost = [k for k in ("z_offset", "x_rot", "y_rot") if getattr(sh, k)]
        if not tilt_used and sh.tilt:
            lost.append("tilt")
        if lost:
            ctx.rep.lossy("MISALIGN_DROPPED", f"Cheetah elements carry only x/y offsets (and a tilt on "
                          f"magnets); dropped: {', '.join(lost)}", element=el.name, kind=el.kind)

    def _apertures(self, el: Element, ctx: _Ctx) -> tuple[list[str], list[str]]:
        ap = el.aperture
        if ap is None or el.kind == "Collimator" or not ctx.install_apertures:
            return [], []
        params = self._aperture_params(ap, el, ctx)
        if params is None:
            return [], []
        pre, post = [], []
        if ap.aperture_at in ("ENTRANCE", "BOTH_ENDS", "CONTINUOUS"):
            pre.append(self._add(ctx, el, "Aperture", dict(params), suffix="_aper_in",
                                 extra_meta={"role": "aperture_entry"}))
        if ap.aperture_at in ("EXIT", "BOTH_ENDS", "CONTINUOUS"):
            post.append(self._add(ctx, el, "Aperture", dict(params), suffix="_aper_out",
                                  extra_meta={"role": "aperture_exit"}))
        if ap.aperture_at == "CONTINUOUS":
            ctx.rep.lossy("APERTURE_CONTINUOUS_AT_ENDS", "a continuous aperture is checked only at the "
                          "element ends in Cheetah", element=el.name, kind=el.kind)
        ctx.rep.equivalent("APERTURE_AS_ELEMENT", "Cheetah apertures are elements of their own: written "
                           "at the element's ends", element=el.name, kind=el.kind)
        return pre, post

    def _aperture_params(self, ap, el: Element, ctx: _Ctx) -> dict | None:
        if ap.x_limits is None or ap.y_limits is None:
            ctx.rep.lossy("APERTURE_PARTIAL", "an aperture without both x and y limits cannot be written",
                          element=el.name, kind=el.kind)
            return None
        (x0, x1), (y0, y1) = ap.x_limits, ap.y_limits
        if abs(x0 + x1) > _THIN or abs(y0 + y1) > _THIN:
            ctx.rep.lossy("APERTURE_OFFSET_DROPPED", "Cheetah apertures are centred: the limits were "
                          "symmetrized", element=el.name, kind=el.kind)
        return {"x_max": _num(max(abs(x0), abs(x1))), "y_max": _num(max(abs(y0), abs(y1))),
                "shape": "elliptical" if ap.shape == "ELLIPTICAL" else "rectangular", "is_active": True}

    # -- kinds ------------------------------------------------------------------------------
    def _k_drift(self, el, ctx):
        ctx.rep.exact(el.name, el.kind)
        return [self._add(ctx, el, "Drift", {"length": _num(el.length)})]

    def _magnet(self, el, ctx, ctype: str, key: str, order: int):
        mp = el.multipole
        bn = float(mp.Bn.get(order, 0.0)) if mp is not None else 0.0
        bs = float(mp.Bs.get(order, 0.0)) if mp is not None else 0.0
        tilt = float((mp.tilt or {}).get(order, 0.0)) if mp is not None and isinstance(mp.tilt, dict) else 0.0
        sh_tilt = el.shift.tilt if el.shift is not None else 0.0
        if bs:
            # a skew component is the same magnet rotated: exact, and Cheetah has only a tilt
            tilt += math.atan2(bs, bn) / (order + 1)
            bn = math.copysign(math.hypot(bn, bs), 1.0)
        k = bn / ctx.brho if ctx.brho else 0.0
        params = {"length": _num(el.length), key: _num(k), "tilt": _num(tilt + sh_tilt),
                  "misalignment": self._misalignment(el)}
        if mp is not None:
            others = {n: v for n, v in {**mp.Bn, **mp.Bs}.items() if n != order and v}
            if others:
                ctx.rep.lossy("MULTIPOLE_ORDERS_DROPPED", f"Cheetah's {ctype} carries one order; dropped orders "
                              f"{sorted(others)}", element=el.name, kind=el.kind)
        self._shift_rest(el, ctx, tilt_used=True)
        ctx.rep.exact(el.name, el.kind)
        return [self._add(ctx, el, ctype, params)]

    def _k_quadrupole(self, el, ctx):
        return self._magnet(el, ctx, "Quadrupole", "k1", 1)

    def _k_sextupole(self, el, ctx):
        return self._magnet(el, ctx, "Sextupole", "k2", 2)

    def _k_octupole(self, el, ctx):
        ctx.rep.lossy("OCTUPOLE_TO_DRIFT", "Cheetah has no octupole element; written as a drift of the "
                      "same length", element=el.name, kind=el.kind)
        return [self._add(ctx, el, "Drift", {"length": _num(el.length)})]

    def _k_multipole(self, el, ctx):
        mp = el.multipole
        bnl = dict(mp.BnL) if mp is not None else {}
        bsl = dict(mp.BsL) if mp is not None else {}
        hk = -float(bnl.get(0, 0.0)) / ctx.brho if ctx.brho else 0.0     # knl[0] = −hkick (measured, xtrack)
        vk = float(bsl.get(0, 0.0)) / ctx.brho if ctx.brho else 0.0
        higher = sorted({n for n, v in {**bnl, **bsl}.items() if n > 0 and v})
        if higher:
            ctx.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "Cheetah has no thin multipole: only the dipole terms are "
                          f"written (as a corrector); dropped orders {higher}", element=el.name, kind=el.kind)
        else:
            ctx.rep.equivalent("MULTIPOLE_AS_CORRECTOR", "a thin dipole term written as a zero-length "
                               "CombinedCorrector", element=el.name, kind=el.kind)
        return [self._add(ctx, el, "CombinedCorrector", {"length": _num(el.length), "horizontal_angle": _num(hk),
                                                          "vertical_angle": _num(vk)})]

    def _k_bend(self, el, ctx):
        b = el.bend
        mp = el.multipole
        k1 = float(mp.Bn.get(1, 0.0)) / ctx.brho if (mp is not None and ctx.brho) else 0.0
        fint = float(b.edge_int1 or 0.0)
        fintx = float(b.edge_int2) if b.edge_int2 is not None else fint
        params = {"length": _num(el.length), "angle": _num(b.angle), "k1": _num(k1),
                  "dipole_e1": _num(b.e1), "dipole_e2": _num(b.e2),
                  "tilt": _num(b.tilt_ref + (el.shift.tilt if el.shift is not None else 0.0)),
                  "gap": _num(2.0 * float(b.hgap or 0.0)), "gap_exit": _num(2.0 * float(b.hgap or 0.0)),
                  "fringe_integral": _num(fint), "fringe_integral_exit": _num(fintx),
                  "fringe_at": "both", "fringe_type": "linear_edge", "misalignment": self._misalignment(el)}
        if mp is not None and any(v for n, v in {**mp.Bn, **mp.Bs}.items() if n != 1):
            ctx.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "Cheetah's Dipole carries k1 only; other multipole "
                          "components dropped", element=el.name, kind=el.kind)
        extra = {}
        if b.rect:
            extra["rect"] = True                         # e1/e2 already sector-referenced in the IR
        self._shift_rest(el, ctx, tilt_used=True)
        ctx.rep.exact(el.name, el.kind)
        return [self._add(ctx, el, "Dipole", params, extra_meta=extra or None)]

    def _k_solenoid(self, el, ctx):
        k = float(el.solenoid.Bsol_T) / (2.0 * ctx.brho) if ctx.brho else 0.0
        self._shift_rest(el, ctx, tilt_used=False)
        ctx.rep.exact(el.name, el.kind)
        return [self._add(ctx, el, "Solenoid", {"length": _num(el.length), "k": _num(k),
                                                "misalignment": self._misalignment(el)})]

    def _cavity(self, el, ctx, *, voltage: float, phase_rad: float, length: float, freq: float | None,
                ctype: str, extra_meta: dict | None = None):
        q = ctx.ref.species.charge or 1
        if not freq:
            ctx.rep.lossy("RF_FREQUENCY_UNKNOWN", "Cheetah's Cavity needs a frequency; 0 Hz written",
                          element=el.name, kind=el.kind)
        if length <= _THIN:
            ctx.rep.equivalent("CHEETAH_ZERO_LENGTH_CAVITY", "Cheetah's cavity matrix divides by the length: "
                               "a zero-length cavity evaluates to inf in Cheetah (the oracle tracks 1 µm)",
                               element=el.name, kind=el.kind)
        params = {"length": _num(length), "voltage": _num(-voltage / q),
                  "phase": _num(-math.degrees(phase_rad)), "frequency": _num(freq or 0.0),
                  "cavity_type": "traveling_wave" if ctype == "TRAVELING_WAVE" else "standing_wave"}
        return self._add(ctx, el, "Cavity", params, extra_meta=extra_meta)

    def _k_rfcavity(self, el, ctx):
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        if not volt and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-9:
            volt = rf.dE_ref_eV / math.cos(rf.phase_rad)
        extra = {"phase_rad": _num(rf.phase_rad)}
        if rf.dE_ref_eV is not None:
            extra["dE_ref_eV"] = _num(rf.dE_ref_eV)
        if rf.n_cell:
            extra["n_cell"] = int(rf.n_cell)
        if rf.L_active_m is not None:
            extra["L_active_m"] = _num(rf.L_active_m)
        self._shift_rest(el, ctx, tilt_used=False)
        ctx.rep.exact(el.name, el.kind)
        return [self._cavity(el, ctx, voltage=float(volt or 0.0), phase_rad=rf.phase_rad, length=el.length,
                             freq=rf.frequency_Hz, ctype=rf.cavity_type or "STANDING_WAVE", extra_meta=extra)]

    def _fieldmap(self, el: FieldMap, ctx: _Ctx) -> list[str]:
        r = replacement_for(el)
        out: list[str] = []
        for part in r.parts:
            body = getattr(self, f"_k_{part.kind.lower()}")(part, ctx)
            out += body
        getattr(ctx.rep, r.cls.lower())(r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, msg in r.extra:
            getattr(ctx.rep, cls.lower())(code, msg, element=el.name, kind="FieldMap")
        return out

    def _k_ncells(self, el, ctx):
        ctx.rep.lossy("NCELLS_TO_DRIFT", "NCELLS cell train replaced by a drift of the same length",
                      element=el.name, kind=el.kind)
        return [self._add(ctx, el, "Drift", {"length": _num(el.length)})]

    def _k_rfqcell(self, el, ctx):
        ctx.rep.lossy("RFQ_TO_DRIFT", "RFQ cell replaced by a drift of the same length",
                      element=el.name, kind=el.kind)
        return [self._add(ctx, el, "Drift", {"length": _num(el.length)})]

    def _k_kicker(self, el, ctx):
        if el.electric:
            ctx.rep.lossy("EKICK_AS_MAGNETIC", "electric steerer written as a magnetic corrector",
                          element=el.name, kind=el.kind)
        ctx.rep.exact(el.name, el.kind)
        return [self._add(ctx, el, "CombinedCorrector", {"length": _num(el.length), "horizontal_angle": _num(el.hkick),
                                                          "vertical_angle": _num(el.vkick)})]

    def _k_collimator(self, el, ctx):
        out: list[str] = []
        params = self._aperture_params(el.aperture, el, ctx) if el.aperture is not None else None
        if params is None:
            ctx.rep.lossy("COLLIMATOR_TO_MARKER", "a collimator without limits is a marker/drift",
                          element=el.name, kind=el.kind)
            if el.length > _THIN:
                return [self._add(ctx, el, "Drift", {"length": _num(el.length)})]
            return [self._add(ctx, el, "Marker", {})]
        out.append(self._add(ctx, el, "Aperture", params))
        ctx.rep.exact(el.name, el.kind)
        if el.length > _THIN:
            out.append(self._add(ctx, el, "Drift", {"length": _num(el.length)}, suffix="_body",
                                 extra_meta={"role": "extra"}))
        return out

    def _k_marker(self, el, ctx):
        ctx.rep.exact(el.name, el.kind)
        return [self._add(ctx, el, "Marker", {})]

    def _k_instrument(self, el, ctx):
        fam = (el.family or "MONITOR").upper()
        meta = {"family": fam}
        if el.params:
            meta["params"] = dict(el.params)
        if el.length > _THIN:
            ctx.rep.equivalent("MONITOR_AS_DRIFT", "an instrument with a length is a drift (its family is "
                               "kept in the metadata)", element=el.name, kind=el.kind)
            return [self._add(ctx, el, "Drift", {"length": _num(el.length)}, extra_meta=meta)]
        if fam == "BPM":
            ctx.rep.exact(el.name, el.kind)
            return [self._add(ctx, el, "BPM", {"is_active": False}, extra_meta=meta)]
        ctx.rep.equivalent("INSTRUMENT_AS_MARKER", f"{fam} instrument written as a Marker (family kept in "
                           "the metadata)", element=el.name, kind=el.kind)
        return [self._add(ctx, el, "Marker", {}, extra_meta=meta)]

    def _k_foil(self, el, ctx):
        ctx.rep.lossy("FOIL_TO_MARKER", "Cheetah has no foil; written as a marker", element=el.name, kind=el.kind)
        meta = {"material": el.material, "thickness_kg_per_m2": _num(el.thickness_kg_per_m2)}
        if el.dE_ref_eV is not None:
            meta["dE_ref_eV"] = _num(el.dE_ref_eV)
        return [self._add(ctx, el, "Marker", {}, extra_meta=meta)]

    def _k_taylor(self, el, ctx):
        from lattix.oracles.base import Basis
        from lattix.oracles.basis import transform_matrix

        m = [[float(x) for x in row] for row in el.matrix]
        off = [float(x) for x in el.offset]
        basis = el.basis or "common"
        if basis in ("common", "bmad", "xtrack"):
            # the IR's (z ahead, δ) basis into Cheetah's (τ late, ΔE/p0c): T R T⁻¹ at the entry energy
            import numpy as np

            src = {"common": Basis.COMMON, "bmad": Basis.BMAD, "xtrack": Basis.XTRACK}[basis]
            ke, mass = ctx.ref.kinetic_energy_eV, ctx.ref.species.mass_eV
            d_src = np.diag(transform_matrix(src, ke, mass))              # x_common = diag(d) @ x_native
            d_dst = np.diag(transform_matrix(Basis.CHEETAH, ke, mass))
            # both transforms are diagonal: R'_ij = R_ij·a_i/a_j, done entry by entry so that the
            # identity entries stay exactly 1 (a fixed point needs it)
            m, off = similarity_diag(m, off, d_src, d_dst)
            ctx.rep.equivalent("TAYLOR_BASIS_CHEETAH", f"the {basis} map was transformed into Cheetah's "
                               "(x, px, y, py, τ, ΔE/p0c) basis at the entry energy", element=el.name, kind=el.kind)
        elif basis != "cheetah":
            ctx.rep.equivalent("TAYLOR_BASIS_CHEETAH", f"the map's source basis {basis!r} is re-used verbatim in "
                               "Cheetah's (x, px, y, py, τ, ΔE/p0c) coordinates", element=el.name, kind=el.kind)
        tm = [[_num(m[i][j]) for j in range(6)] + [_num(off[i])] for i in range(6)] + [[0.0] * 6 + [1.0]]
        ctx.rep.exact(el.name, el.kind)
        # the stored map is in Cheetah's basis whatever the source basis was (transformed above)
        stored = "cheetah" if basis in ("common", "bmad", "xtrack", "cheetah") else basis
        return [self._add(ctx, el, "CustomTransferMap", {"length": _num(el.length), "predefined_transfer_map": tm},
                          extra_meta={"basis": stored})]

    def _k_patch(self, el, ctx):
        ctx.rep.dropped("PATCH_DROPPED", "Cheetah has no coordinate patch; written as a marker (its offsets "
                        "in the metadata)", element=el.name, kind=el.kind)
        meta = {k: _num(getattr(el, k)) for k in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
                if getattr(el, k, 0.0)}
        params = {"length": _num(el.length)} if el.length > _THIN else {}
        return [self._add(ctx, el, "Drift" if params else "Marker", params, extra_meta=meta or None)]

    def _k_referencechange(self, el, ctx):
        ctx.rep.dropped("REFCHANGE_DROPPED", "Cheetah's reference energy follows the cavities only; an explicit "
                        "reference change is a marker with the energy in its metadata",
                        element=el.name, kind=el.kind)
        meta = {}
        if el.dE_ref_eV is not None:
            meta["dE_ref_eV"] = _num(el.dE_ref_eV)
        if el.energy_eV is not None:
            meta["energy_eV"] = _num(el.energy_eV)
        return [self._add(ctx, el, "Marker", {}, extra_meta=meta or None)]

    def _k_freq(self, el, ctx):
        ctx.rep.exact(el.name, el.kind, message="the RF frequency is a per-cavity attribute in Cheetah")
        return [self._add(ctx, el, "Marker", {}, extra_meta={"frequency_Hz": _num(el.frequency_Hz)})]

    def _k_directive(self, el, ctx):
        ctx.rep.dropped("FOREIGN_DIRECTIVE", "format-specific directive kept only in the metadata",
                        element=el.name, kind=el.kind)
        return [self._add(ctx, el, "Marker", {}, extra_meta={"format": el.format, "card": el.card,
                                                              "args": list(el.args), "ir_role": el.role})]

    def _k_superposition(self, el, ctx):        # pragma: no cover - handled in _emit
        return self._emit(el, ctx)


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert check_rules_coverage(Writer()) == set()
