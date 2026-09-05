"""Ocelot writer: the IR as a Python lattice module (see the package docstring for the conventions).

Every element is one assignment ``var = Ctor(...)  # lattix: name=… kind=…`` and ``cell = (…)`` lists
the placements (a definition placed twice appears twice); ``lattice = MagneticLattice(cell)`` closes
the file.  The trailing tag carries what Ocelot cannot hold (the IR kind and name, a thin gap's
surrogate padding, an aperture's role, an instrument's family …) so the reader restores it.
"""
from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage, with_rf_focusing
from lattix.ir.elements import Element
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.rf import thin_gap_surrogate_length
from lattix.ir.walk import propagate

_THIN = 1e-12
#: names Ocelot's ``from ocelot import *`` brings in (never used as element variables)
_RESERVED = {"drift", "quadrupole", "sextupole", "octupole", "multipole", "sbend", "rbend", "bend", "solenoid",
             "cavity", "tdcavity", "twcavity", "hcor", "vcor", "marker", "monitor", "aperture", "matrix",
             "undulator", "magneticlattice", "twiss", "cell", "lattice", "tws0", "ocelot", "np", "pi", "inf",
             "beam", "track", "method", "particle", "navigator", "trajectory"}
_PLAIN = re.compile(r"[\w.+\-]+")


@dataclass(frozen=True)
class Rule:
    message: str
    cls: str
    code: str


@dataclass
class _Entry:
    var: str
    ctor: str
    args: list[tuple[str, str]]                       # (keyword, rendered value)
    tag: dict[str, object]                            # name, kind, … (restoration data)
    length: float = 0.0
    attrs: list[tuple[str, str]] = field(default_factory=list)     # ``var.dx = …`` lines
    synthetic: bool = False                           # a surrogate cavity that must absorb its length

    def line(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.args)
        return f"{self.var} = {self.ctor}({args})  # lattix: {_tag(self.tag)}"


def fmt(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    v = float(v)
    if v == 0:
        return "0.0"
    if math.isinf(v):
        return "np.inf" if v > 0 else "-np.inf"
    s = f"{v:.15g}"
    return s if ("." in s or "e" in s or "n" in s) else s + ".0"


def _tag(kv: dict) -> str:
    parts = []
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, bool):
            s = "1" if v else "0"
        elif isinstance(v, int | float):
            s = fmt(v)
        else:
            s = str(v)
        if not _PLAIN.fullmatch(s):
            s = '"' + s.replace('"', "'").replace("\n", " ") + '"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


def _ident(name: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_") or "e"
    if base[0].isdigit() or keyword.iskeyword(base) or base.lower() in _RESERVED:
        base = "e_" + base
    cand, n = base, 1
    while cand in used or cand.lower() in _RESERVED:
        n += 1
        cand = f"{base}_{n}"
    used.add(cand)
    return cand


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport` (an Ocelot lattice module)."""

    format = "ocelot"
    RULES: dict[str, Rule] = {
        "Drift": Rule("Drift(l)", "EXACT", "OK"),
        "Quadrupole": Rule("Quadrupole(l, k1 = Bn1/Bρ_signed, k2, tilt) (a skew component is a tilt)", "EXACT", "OK"),
        "Sextupole": Rule("Sextupole(l, k2 = Bn2/Bρ_signed, tilt)", "EXACT", "OK"),
        "Octupole": Rule("Octupole(l, k3 = Bn3/Bρ_signed, tilt)", "EXACT", "OK"),
        "Multipole": Rule("Multipole(kn = [BnL_n/Bρ_signed]) — MAD's knl, kn[0] a design bend", "EXACT", "OK"),
        "Bend": Rule("SBend(l, angle, k1, k2, e1, e2, tilt, gap = 2·hgap, fint, fintx)", "EXACT", "OK"),
        "Solenoid": Rule("Solenoid(l, k = Bsol/(2 Bρ_signed))", "EXACT", "OK"),
        "RFCavity": Rule("Cavity(l, v [GV], phi = −φs [deg], freq): Ocelot's own cavity matrix", "EXACT", "OK"),
        "FieldMap": Rule("the lattix.ir.fieldmap replacement ladder (Cavity / Solenoid / Quadrupole / Drift)",
                         "LOSSY", "FM_REPLACED"),
        "NCells": Rule("Cavity(l, v = the train's voltage, phi, freq)", "EQUIVALENT", "NCELLS_AS_CAVITY"),
        "RFQCell": Rule("Drift (Ocelot has no RFQ cell)", "LOSSY", "RFQCELL_TO_DRIFT"),
        "Kicker": Rule("Hcor(l, angle = hkick) [+ Vcor(0, angle = vkick)]", "EXACT", "OK"),
        "Collimator": Rule("Aperture(xmax, ymax, dx, dy, type) + Drift for the body", "EXACT", "OK"),
        "Marker": Rule("Marker()", "EXACT", "OK"),
        "Instrument": Rule("Monitor(l) (the family in the tag)", "EQUIVALENT", "INSTRUMENT_AS_MONITOR"),
        "Foil": Rule("Marker (Ocelot has no foil)", "LOSSY", "FOIL_TO_MARKER"),
        "Taylor": Rule("Matrix(l, r11 … r66, b1 … b6) in Ocelot's basis at the entry energy", "EXACT", "OK"),
        "Patch": Rule("Marker with the offsets in the tag (Ocelot has no patch)", "DROPPED", "PATCH_DROPPED"),
        "ReferenceChange": Rule("Matrix(l = 0, delta_e = dE_ref [GeV])", "EQUIVALENT", "REFCHANGE_AS_MATRIX"),
        "Freq": Rule("Marker (the frequency is per cavity)", "EXACT", "OK"),
        "Directive": Rule("Marker with the card in the tag", "DROPPED", "FOREIGN_DIRECTIVE"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED"),
    }

    def write(self, lattice: Lattice, path, *, strict: bool = False, **options) -> FidelityReport:
        path = Path(path)
        rep = FidelityReport(target_format="ocelot", target_file=str(path))
        text = self.render(lattice, report=rep, **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, **options) -> str:
        return self.render(lattice, **options)

    def render(self, lattice: Lattice, *, report: FidelityReport | None = None, install_apertures: bool = True,
               thin_gap_length_m: float | None = None) -> str:
        self.rep = report if report is not None else FidelityReport(target_format="ocelot")
        self.lat = lattice
        self.install_apertures = install_apertures
        self.thin_gap_length_m = thin_gap_length_m
        self.used: set[str] = set()
        lattice, _ = with_rf_focusing(lattice, "ocelot")
        self.lat = lattice
        ref = lattice.reference
        if ref.species.name.lower() not in ("electron", "positron"):
            self.rep.lossy("OCELOT_ELECTRON_ONLY",
                           f"Ocelot's maps divide by the electron mass; the reference is a {ref.species.name} of "
                           f"{ref.kinetic_energy_eV:.6g} eV — the normalized strengths are exact, the "
                           "longitudinal and cavity terms are Ocelot's electron ones", species=ref.species.name)
        order: list[_Entry] = []
        emitted: dict[tuple, list[_Entry]] = {}
        for p in propagate(lattice):
            key = (id(p.element), round((p.ref_in or ref).kinetic_energy_eV, 6))
            if key not in emitted:
                emitted[key] = self._element(p)
            order += emitted[key]
        self._absorb(order)
        defs: list[_Entry] = []
        seen: set[int] = set()
        for e in order:
            if id(e) not in seen:
                seen.add(id(e))
                defs.append(e)
        etot_GeV = (ref.kinetic_energy_eV + ref.species.mass_eV) * 1e-9
        lines = [f"# Ocelot lattice written by lattix {__version__}", format_reference_tag(ref, "#"),
                 f"# lattix: lattice {_tag({'name': lattice.name, 'use': lattice.use or lattice.name})}",
                 "import numpy as np", "from ocelot import *", "",
                 "tws0 = Twiss()", f"tws0.E = {fmt(etot_GeV)}  # total energy [GeV] of the reference", ""]
        for e in defs:
            lines.append(e.line())
            for k, v in e.attrs:
                lines.append(f"{e.var}.{k} = {v}")
        lines.append("")
        if not order:
            lines.append("cell = ()")
        else:
            lines.append("cell = (")
            row: list[str] = []
            for e in order:
                row.append(e.var)
                if len(", ".join(row)) > 88:
                    lines.append("    " + ", ".join(row) + ",")
                    row = []
            if row:
                lines.append("    " + ", ".join(row) + ",")
            lines.append(")")
        lines.append("lattice = MagneticLattice(cell)")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ helpers
    def _record(self, el: Element, rule: Rule, **details) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind, code=rule.code, message=rule.message)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind, **details)

    def _entry(self, el: Element, ctor: str, args: list[tuple[str, object]], *, length: float = 0.0,
               suffix: str = "", tag: dict | None = None, kind: str | None = None,
               synthetic: bool = False) -> _Entry:
        var = _ident(el.name + suffix, self.used)
        rendered = [(k, v if isinstance(v, str) else fmt(v)) for k, v in args if v is not None]
        rendered.append(("eid", repr(el.name + suffix)))
        full = {"name": el.name, "kind": kind or el.kind}
        if tag:
            full.update(tag)
        if el.meta.get("ocelot_from") and "from" not in full:
            full["from"] = str(el.meta["ocelot_from"])        # a re-read replacement keeps its origin
        return _Entry(var, ctor, rendered, full, length, synthetic=synthetic)

    @staticmethod
    def _brho(p: Placed) -> float:
        return p.ref_in.brho_signed if p.ref_in is not None else 0.0

    def _shift(self, el: Element, e: _Entry, *, tilt_used: bool) -> None:
        sh = el.shift
        if sh is None:
            return
        if sh.x_offset:
            e.attrs.append(("dx", fmt(sh.x_offset)))
        if sh.y_offset:
            e.attrs.append(("dy", fmt(sh.y_offset)))
        lost = [k for k in ("z_offset", "x_rot", "y_rot") if getattr(sh, k)]
        if not tilt_used and sh.tilt:
            lost.append("tilt")
        if lost:
            self.rep.lossy("MISALIGN_DROPPED", "Ocelot elements carry dx, dy and (magnets) a tilt; dropped: "
                           + ", ".join(lost), element=el.name, kind=el.kind)

    def _aperture_args(self, ap, el: Element) -> list[tuple[str, object]] | None:
        if ap.x_limits is None or ap.y_limits is None:
            self.rep.lossy("APERTURE_PARTIAL", "an aperture without both x and y limits cannot be written",
                           element=el.name, kind=el.kind)
            return None
        (x0, x1), (y0, y1) = ap.x_limits, ap.y_limits
        args: list[tuple[str, object]] = [("xmax", (x1 - x0) / 2.0), ("ymax", (y1 - y0) / 2.0)]
        if abs(x0 + x1) > _THIN:
            args.append(("dx", (x0 + x1) / 2.0))
        if abs(y0 + y1) > _THIN:
            args.append(("dy", (y0 + y1) / 2.0))
        args.append(("type", repr("ellipt" if ap.shape == "ELLIPTICAL" else "rect")))
        return args

    def _apertures(self, el: Element) -> tuple[list[_Entry], list[_Entry]]:
        ap = el.aperture
        if ap is None or el.kind == "Collimator" or not self.install_apertures:
            return [], []
        args = self._aperture_args(ap, el)
        if args is None:
            return [], []
        pre, post = [], []
        if ap.aperture_at in ("ENTRANCE", "BOTH_ENDS", "CONTINUOUS"):
            pre.append(self._entry(el, "Aperture", list(args), suffix="_aper_in", tag={"role": "aperture_in"},
                                   kind=el.kind))
        if ap.aperture_at in ("EXIT", "BOTH_ENDS", "CONTINUOUS"):
            post.append(self._entry(el, "Aperture", list(args), suffix="_aper_out", tag={"role": "aperture_out"},
                                    kind=el.kind))
        if ap.aperture_at == "CONTINUOUS":
            self.rep.lossy("APERTURE_CONTINUOUS_AT_ENDS", "a continuous aperture is checked only at the "
                           "element ends in Ocelot", element=el.name, kind=el.kind)
        self.rep.equivalent("APERTURE_AS_ELEMENT", "Ocelot apertures are elements of their own: written at "
                            "the element's ends", element=el.name, kind=el.kind)
        return pre, post

    # ------------------------------------------------------------------ elements
    def _element(self, p: Placed) -> list[_Entry]:
        el = p.element
        rule = self.RULES.get(el.kind)
        if rule is None:                                          # pragma: no cover - RULES is total
            raise KeyError(f"the Ocelot writer has no rule for kind {el.kind!r}")
        if el.kind == "Superposition":
            return self._w_superposition(el, p, rule)
        if el.kind == "FieldMap":
            return self._w_fieldmap(el, p, rule)
        pre, post = self._apertures(el)
        fn = getattr(self, f"_w_{el.kind.lower()}")
        return pre + fn(el, p, rule) + post

    def _w_drift(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "Drift", [("l", el.length)], length=el.length)]

    def _magnet(self, el, p, rule, ctor: str, key: str, order: int, extra: dict[str, int] | None = None):
        mp = el.multipole
        bn = float(mp.Bn.get(order, 0.0)) if mp is not None else 0.0
        bs = float(mp.Bs.get(order, 0.0)) if mp is not None else 0.0
        tilt = float(mp.tilt.get(order, 0.0)) if mp is not None else 0.0
        if el.shift is not None and el.shift.tilt:
            tilt += float(el.shift.tilt)
        if bs:
            tilt += math.atan2(bs, bn) / (order + 1)          # the same magnet, rotated: exact
            bn = math.hypot(bn, bs)
        brho = self._brho(p)
        args: list[tuple[str, object]] = [("l", el.length), (key, bn / brho if brho else 0.0)]
        used = {order}
        for k2, o2 in (extra or {}).items():
            v = float(mp.Bn.get(o2, 0.0)) if mp is not None else 0.0
            if v:
                args.append((k2, v / brho if brho else 0.0))
                used.add(o2)
        if tilt:
            args.append(("tilt", tilt))
        if mp is not None:
            others = sorted(n for n, v in {**mp.Bn, **mp.Bs}.items() if v and n not in used)
            if others:
                self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", f"Ocelot's {ctor} carries orders {sorted(used)}; "
                               f"dropped orders {others}", element=el.name, kind=el.kind)
        e = self._entry(el, ctor, args, length=el.length)
        self._shift(el, e, tilt_used=True)
        self._record(el, rule)
        return [e]

    def _w_quadrupole(self, el, p, rule):
        return self._magnet(el, p, rule, "Quadrupole", "k1", 1, {"k2": 2})

    def _w_sextupole(self, el, p, rule):
        return self._magnet(el, p, rule, "Sextupole", "k2", 2)

    def _w_octupole(self, el, p, rule):
        return self._magnet(el, p, rule, "Octupole", "k3", 3)

    def _w_multipole(self, el, p, rule):
        mp = el.multipole
        brho = self._brho(p)
        bnl = {n: float(v) for n, v in mp.BnL.items() if v}
        top = max([*bnl, 1])
        kn = [bnl.get(n, 0.0) / brho if brho else 0.0 for n in range(top + 1)]
        skew = sorted(n for n, v in mp.BsL.items() if v)
        if skew:
            self.rep.lossy("OCELOT_SKEW_MULTIPOLE_DROPPED", "Ocelot's Multipole has normal strengths only; "
                           f"skew orders {skew} dropped", element=el.name, kind=el.kind)
        e = self._entry(el, "Multipole", [("kn", "[" + ", ".join(fmt(v) for v in kn) + "]")])
        self._shift(el, e, tilt_used=False)
        self._record(el, rule)
        out = [e]
        if el.length > _THIN:
            self.rep.equivalent("THICK_MULTIPOLE_SPLIT", "Ocelot's Multipole is thin: the kick at the entrance, "
                                "a drift for the length", element=el.name, kind=el.kind)
            out.append(self._entry(el, "Drift", [("l", el.length)], length=el.length, suffix="_body",
                                   tag={"role": "body"}))
        return out

    def _w_bend(self, el, p, rule):
        b = el.bend
        brho = self._brho(p)
        mp = el.multipole
        k1 = float(mp.Bn.get(1, 0.0)) / brho if brho else 0.0
        k2 = float(mp.Bn.get(2, 0.0)) / brho if brho else 0.0
        tilt = float(b.tilt_ref) + (float(el.shift.tilt) if el.shift is not None else 0.0)
        fint = float(b.edge_int1 or 0.0)
        args: list[tuple[str, object]] = [("l", el.length), ("angle", b.angle)]
        if k1:
            args.append(("k1", k1))
        if k2:
            args.append(("k2", k2))
        if b.e1:
            args.append(("e1", b.e1))
        if b.e2:
            args.append(("e2", b.e2))
        if tilt:
            args.append(("tilt", tilt))
        if b.hgap:
            args.append(("gap", 2.0 * float(b.hgap)))
        if fint:
            args.append(("fint", fint))
        if b.edge_int2 is not None and abs(float(b.edge_int2) - fint) > 1e-15:
            args.append(("fintx", float(b.edge_int2)))
        if any(v for n, v in {**mp.Bn, **mp.Bs}.items() if n not in (1, 2)):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "Ocelot's SBend carries k1 and k2 only",
                           element=el.name, kind=el.kind)
        tag = {"rect": True} if b.rect else None
        e = self._entry(el, "SBend", args, length=el.length, tag=tag)
        self._shift(el, e, tilt_used=True)
        self._record(el, rule)
        return [e]

    def _w_solenoid(self, el, p, rule):
        brho = self._brho(p)
        k = float(el.solenoid.Bsol_T) / (2.0 * brho) if brho else 0.0
        e = self._entry(el, "Solenoid", [("l", el.length), ("k", k)], length=el.length)
        self._shift(el, e, tilt_used=False)
        self._record(el, rule)
        return [e]

    def _cavity(self, el, p, rule, *, V: float, freq: float, phase: float, thin: bool, kind: str,
                tag: dict | None = None, **details):
        ref = p.ref_in or self.lat.reference
        L = float(el.length)
        if thin:                      # on the picometre grid, like the padding it is absorbed by
            L = round(float(self.thin_gap_length_m or thin_gap_surrogate_length(V, phase, freq, ref)), 12)
        args: list[tuple[str, object]] = [("l", L), ("v", V * 1e-9), ("phi", -math.degrees(phase)), ("freq", freq)]
        e = self._entry(el, "Cavity", args, length=L, kind=kind, tag=tag, synthetic=thin)
        self._shift(el, e, tilt_used=False)
        self._record(el, rule, voltage_V=V, **details)
        if thin:
            e.tag["L"] = 0
            self.rep.equivalent("THIN_GAP_AS_SHORT_CAVITY", "Ocelot's Cavity divides by its length: the thin "
                                f"gap is a {L:.6g} m cavity centred on it, its length taken from the "
                                "neighbouring drifts", element=el.name, kind=el.kind, length_m=L)
        return [e]

    def _w_rfcavity(self, el, p, rule):
        rf = el.rf
        ref = p.ref_in or self.lat.reference
        V = rf.voltage_V
        if not V and rf.gradient_V_per_m is not None:
            V = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        if not V and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-9:
            V = rf.dE_ref_eV / math.cos(rf.phase_rad)
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        if not freq:
            self.rep.lossy("RF_FREQUENCY_UNKNOWN", "Ocelot's Cavity needs a frequency; 0 Hz written",
                           element=el.name, kind=el.kind)
        tag = {}
        if rf.cavity_type == "TRAVELING_WAVE":
            tag["tw"] = True
        if rf.n_cell:
            tag["n"] = int(rf.n_cell)
        return self._cavity(el, p, rule, V=float(V or 0.0), freq=float(freq), phase=rf.phase_rad,
                            thin=el.length <= _THIN and bool(V), kind="RFCavity", tag=tag or None)

    def _w_fieldmap(self, el, p, rule):
        r = replacement_for(el)
        out: list[_Entry] = []
        s = p.s_in
        for part in r.parts:
            sub = p.model_copy(update={"element": part, "s_in": s, "s_out": s + part.length})
            fn = getattr(self, f"_w_{part.kind.lower()}")
            for e in fn(part, sub, self.RULES[part.kind]):
                e.tag["from"] = "FieldMap"
                out.append(e)
            s += part.length
        getattr(self.rep, r.cls.lower())(r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, msg in r.extra:
            getattr(self.rep, cls.lower())(code, msg, element=el.name, kind="FieldMap")
        return out

    def _w_ncells(self, el, p, rule):
        rf = el.rf
        ref = p.ref_in or self.lat.reference
        V = rf.voltage_V or (rf.gradient_V_per_m or 0.0) * (rf.L_active_m or el.length)
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        if not (V and freq and el.length > _THIN):
            self.rep.lossy("NCELLS_TO_DRIFT", "an NCells without a voltage, a frequency or a length is a drift",
                           element=el.name, kind="NCells")
            return [self._entry(el, "Drift", [("l", el.length)], length=el.length, tag={"from": "NCells"},
                                kind="Drift")]
        n = int(el.params.get("n_cells") or rf.n_cell or 1)
        return self._cavity(el, p, rule, V=float(V), freq=float(freq), phase=rf.phase_rad, thin=False,
                            kind="NCells", tag={"n": n}, n_cell=n)

    def _w_rfqcell(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "Drift", [("l", el.length)], length=el.length, tag={"from": "RFQCell"},
                            kind="Drift")]

    def _w_kicker(self, el, p, rule):
        if el.electric:
            self.rep.lossy("EKICK_AS_MAGNETIC", "an electric kicker is written as a magnetic corrector of the "
                           "same deflection", element=el.name, kind="Kicker")
        out: list[_Entry] = []
        if el.hkick or not el.vkick:
            out.append(self._entry(el, "Hcor", [("l", el.length), ("angle", float(el.hkick))], length=el.length))
        if el.vkick:
            if out:
                out.append(self._entry(el, "Vcor", [("l", 0.0), ("angle", float(el.vkick))], suffix="_v",
                                       tag={"role": "vkick"}))
                self.rep.equivalent("KICKER_SPLIT_HV", "Ocelot corrects one plane per element: Hcor of the "
                                    "length then a zero-length Vcor", element=el.name, kind="Kicker")
            else:
                out.append(self._entry(el, "Vcor", [("l", el.length), ("angle", float(el.vkick))],
                                       length=el.length))
        self._shift(el, out[0], tilt_used=False)
        self._record(el, rule)
        return out

    def _w_collimator(self, el, p, rule):
        args = self._aperture_args(el.aperture, el) if el.aperture is not None else None
        if args is None:
            self.rep.lossy("COLLIMATOR_TO_MARKER", "a collimator without limits is a marker/drift",
                           element=el.name, kind=el.kind)
            if el.length > _THIN:
                return [self._entry(el, "Drift", [("l", el.length)], length=el.length)]
            return [self._entry(el, "Marker", [])]
        out = [self._entry(el, "Aperture", args)]
        self._record(el, rule)
        if el.length > _THIN:
            out.append(self._entry(el, "Drift", [("l", el.length)], length=el.length, suffix="_body",
                                   tag={"role": "body"}))
        return out

    def _w_marker(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "Marker", [])]

    def _w_instrument(self, el, p, rule):
        self._record(el, rule, family=el.family)
        tag = {"family": el.family}
        if el.params:
            tag["params"] = ";".join(f"{k}:{v}" for k, v in el.params.items())
        return [self._entry(el, "Monitor", [("l", el.length)], length=el.length, tag=tag)]

    def _w_foil(self, el, p, rule):
        self._record(el, rule)
        tag = {"material": el.material, "thick": el.thickness_kg_per_m2}
        if el.dE_ref_eV is not None:
            tag["dE"] = el.dE_ref_eV
        out = [self._entry(el, "Marker", [], tag=tag)]
        if el.length > _THIN:
            out.append(self._entry(el, "Drift", [("l", el.length)], length=el.length, suffix="_body",
                                   tag={"role": "body"}))
        return out

    def _w_taylor(self, el, p, rule):
        import numpy as np

        from lattix.formats.cheetah.writer import similarity_diag
        from lattix.oracles.base import Basis
        from lattix.oracles.basis import transform_matrix

        m = [[float(x) for x in row] for row in el.matrix]
        off = [float(x) for x in el.offset]
        basis = el.basis or "common"
        ref = p.ref_in or self.lat.reference
        stored = basis
        if basis in ("common", "bmad", "xtrack", "cheetah", "madx", "impactx"):
            src = Basis(basis)
            d_src = np.diag(transform_matrix(src, ref.kinetic_energy_eV, ref.species.mass_eV))
            d_dst = np.diag(transform_matrix(Basis.OCELOT, ref.kinetic_energy_eV, ref.species.mass_eV))
            m, off = similarity_diag(m, off, d_src, d_dst)
            stored = "ocelot"
            if basis != "ocelot":
                self.rep.equivalent("TAYLOR_BASIS_OCELOT", f"the {basis} map was transformed into Ocelot's "
                                    "(x, px, y, py, τ, ΔE/p0c) basis at the entry energy", element=el.name,
                                    kind=el.kind)
        elif basis != "ocelot":
            self.rep.equivalent("TAYLOR_BASIS_OCELOT", f"the map's source basis {basis!r} is re-used verbatim in "
                                "Ocelot's coordinates", element=el.name, kind=el.kind)
        args: list[tuple[str, object]] = [("l", el.length)]
        for i in range(6):
            for j in range(6):
                if m[i][j] != 0.0:
                    args.append((f"r{i + 1}{j + 1}", m[i][j]))
        for i in range(6):
            if off[i] != 0.0:
                args.append((f"b{i + 1}", off[i]))
        self._record(el, rule)
        return [self._entry(el, "Matrix", args, length=el.length, tag={"basis": stored})]

    def _w_patch(self, el, p, rule):
        self._record(el, rule)
        tag = {k: getattr(el, k) for k in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
               if getattr(el, k)}
        if el.length > _THIN:
            return [self._entry(el, "Drift", [("l", el.length)], length=el.length, tag=tag)]
        return [self._entry(el, "Marker", [], tag=tag)]

    def _w_referencechange(self, el, p, rule):
        ref = p.ref_in or self.lat.reference
        dE = el.dE_ref_eV
        if dE is None and el.energy_eV is not None:
            dE = float(el.energy_eV) - ref.kinetic_energy_eV
        self._record(el, rule, dE_ref_eV=dE)
        tag = {}
        if el.dE_ref_eV is not None:
            tag["dE"] = el.dE_ref_eV
        if el.energy_eV is not None:
            tag["E"] = el.energy_eV
        return [self._entry(el, "Matrix", [("l", 0.0), ("delta_e", float(dE or 0.0) * 1e-9), ("r11", 1.0),
                                          ("r22", 1.0), ("r33", 1.0), ("r44", 1.0), ("r55", 1.0), ("r66", 1.0)],
                            tag=tag or None)]

    def _w_freq(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "Marker", [], tag={"f": el.frequency_Hz})]

    def _w_directive(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "Marker", [], tag={"format": el.format, "card": el.card, "args": " ".join(el.args),
                                                   "role": el.role})]

    def _w_superposition(self, el, p, rule):
        self._record(el, rule)
        out: list[_Entry] = []
        s = p.s_in
        for _off, name in el.children:
            child = self.lat.elements.get(name)
            if child is None:
                continue
            sub = p.model_copy(update={"element": child, "s_in": s, "s_out": s + child.length})
            out += self._element(sub)
            s += child.length
        return out

    # ------------------------------------------------------------------ surrogate cavities
    def _absorb(self, order: list[_Entry]) -> None:
        """A thin gap's surrogate cavity takes its length from the drift before and after it (the tag
        ``L=0 pad=<from the drift before>`` lets the reader put the gap back).  A drift definition placed
        elsewhere too is cloned before it is shortened."""
        counts: dict[int, int] = {}
        for e in order:
            counts[id(e)] = counts.get(id(e), 0) + 1
        for i in range(len(order)):
            if order[i].synthetic and counts[id(order[i])] > 1:
                order[i] = self._clone(order[i])
        for i, e in enumerate(order):
            if not e.synthetic or "pad" in e.tag:
                continue
            before = next((j for j in range(i - 1, -1, -1) if order[j].length > 0), None)
            after = next((j for j in range(i + 1, len(order)) if order[j].length > 0), None)
            if before is not None and order[before].ctor != "Drift":
                before = None
            if after is not None and order[after].ctor != "Drift":
                after = None
            # every take on the picometre grid, the second computed from the rounded first, so that
            # take_b + take_a == L exactly and the reader's drifts come back at their original length
            take_b = round(min(0.5 * e.length, order[before].length), 12) if before is not None else 0.0
            take_a = round(min(e.length - take_b, order[after].length), 12) if after is not None else 0.0
            if take_b + take_a < e.length - 1e-15 and before is not None:
                take_b = round(min(e.length - take_a, order[before].length), 12)
            for j, take in ((before, take_b), (after, take_a)):
                if j is None or not take:
                    continue
                if counts[id(order[j])] > 1:
                    order[j] = self._clone(order[j])
                d = order[j]
                d.length = round(d.length - take, 12)
                d.args = [(k, fmt(d.length) if k == "l" else v) for k, v in d.args]
            missing = e.length - take_b - take_a
            if missing > 1e-9:
                self.rep.lossy("THIN_GAP_ADDS_LENGTH", f"the {e.length:.6g} m cavity standing in for a thin gap "
                               f"could not be absorbed by neighbouring drifts; the line grows by {missing:.6g} m",
                               element=str(e.tag["name"]), kind=str(e.tag["kind"]), added_length_m=missing)
            e.tag["pad"] = take_b

    def _clone(self, e: _Entry) -> _Entry:
        var = _ident(e.var, self.used)
        tag = dict(e.tag)
        tag["name"] = var
        args = [(k, repr(var) if k == "eid" else v) for k, v in e.args]
        return _Entry(var, e.ctor, args, tag, e.length, list(e.attrs), e.synthetic)


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert check_rules_coverage(Writer()) == set()
