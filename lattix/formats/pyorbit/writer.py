"""IR → PyORBIT3 linac XML (see the package docstring for the measured conventions)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage, with_rf_focusing
from lattix.ir.elements import ALL_KINDS, Element
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.walk import propagate

_INFO = {
    "Drift": "implicit (the factory fills the gaps between elements)", "Quadrupole": "QUAD field [T/m]",
    "Sextupole": "dropped (no PyORBIT linac element)", "Octupole": "dropped",
    "Multipole": "DCH/DCV for the dipole terms",
    "Bend": "BEND theta ea1 ea2", "Solenoid": "SOLENOID B = Bsol/Bρ [1/m]", "RFCavity": "RFGAP E0TL [GeV] phase [deg]",
    "FieldMap": "the lattix.ir.fieldmap replacement ladder", "NCells": "dropped (drift)", "RFQCell": "dropped (drift)",
    "Kicker": "DCH + DCV (B·effLength)", "Collimator": "MARKER", "Marker": "MARKER", "Instrument": "MARKER",
    "Foil": "MARKER", "Taylor": "dropped", "Patch": "dropped", "ReferenceChange": "dropped",
    "Freq": "nothing (the frequency is per cavity)", "Directive": "dropped", "Superposition": "children in order",
}
assert set(_INFO) == set(ALL_KINDS)
RULES = dict(_INFO)
_TAG = re.compile(r"[^A-Za-z0-9_]")
_THIN = 1e-12
_EFF_LENGTH_THIN = 1e-3     # effLength of a zero-length kicker (its B scales with it)


def _constant_ttfs() -> str:
    """A transit-time factor of exactly 1 at every velocity, serialized like the reader keeps the
    native ones (``xml.etree`` style, so write → read → write is a fixed point)."""
    import xml.etree.ElementTree as ET

    ttfs = ET.Element("TTFs", {"beta_max": "1.0", "beta_min": "0.0"})
    for tag, coef in (("polyT", "1.0"), ("polyS", "0.0"), ("polyTP", "0.0"), ("polySP", "0.0")):
        ET.SubElement(ttfs, tag, {"order": "0", "pcoefs": coef})
    return ET.tostring(ttfs, encoding="unicode").strip()


def _num(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0.0"
    return f"{x:.15g}"


def _tag(name: str) -> str:
    t = _TAG.sub("_", name) or "SEQ"
    return t if not t[0].isdigit() else "S_" + t


@dataclass
class _Out:
    lines: list[str] = field(default_factory=list)
    cavities: dict[float, tuple[str, float, float]] = field(default_factory=dict)   # index → (name, pos, f)
    names: set[str] = field(default_factory=set)
    cursor2: int = 0                       # running position of the placed elements, half-picometres
    prev_exit2: int = 0                    # exit of the last written node (half-pm), for the overlap push
    prev_pos: float | None = None          # the last written node's pos and length as PyORBIT parses them
    prev_len: float = 0.0
    last_pos2: int = 0                     # where the last node was actually written (half-pm)


def _dec(m: int, d: int) -> str:
    """``m·10^−d`` as the shortest exact decimal (``_dec(13689997583782, 12)`` → ``13.689997583782``)."""
    s = f"{m // 10**d}.{m % 10**d:0{d}d}".rstrip("0")
    return s + "0" if s.endswith(".") else s


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport` (PyORBIT3 linac XML)."""

    format = "pyorbit"
    RULES = RULES

    def write(self, lattice: Lattice, path, *, strict: bool = False, sequence: str | None = None) -> FidelityReport:
        from pathlib import Path

        rep = FidelityReport(target_format="pyorbit", target_file=str(path))
        text = self.dumps(lattice, report=rep, sequence=sequence)
        Path(path).write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, *, report: FidelityReport | None = None, sequence: str | None = None) -> str:
        rep = report if report is not None else FidelityReport(target_format="pyorbit")
        lattice, _ = with_rf_focusing(lattice, "pyorbit")
        placed = list(propagate(lattice))
        out = _Out()
        rep_ref = lattice.reference
        for p in placed:
            self._emit(p.element, p.ref_in or rep_ref, out, rep)
        total2 = max(out.cursor2, out.prev_exit2)          # a pushed last node may stick out by picometres
        seq = _tag(sequence or lattice.use or lattice.name or "SEQ")
        freq = rep_ref.rf_frequency_Hz or (next(iter(out.cavities.values()))[2] if out.cavities else 0.0)
        head = ['<?xml version="1.0" ?>', "<!--", format_reference_tag(rep_ref, "#"),
                f"# lattix {__version__} (PyORBIT3 linac XML)", "-->",
                "<lattix>",
                f' <{seq} bpmFrequency="{_num(freq)}" length="{_dec(5 * total2, 13)}" name={quoteattr(seq)}>']
        if out.cavities:
            head.append("  <Cavities>")
            for _, (cname, pos2, f) in sorted(out.cavities.items()):
                head.append(f'   <Cavity ampl="1.0" frequency="{_num(f)}" name={quoteattr(cname)} '
                            f'pos="{_dec(5 * pos2, 13)}"/>')
            head.append("  </Cavities>")
        return "\n".join(head + out.lines + [f" </{seq}>", "</lattix>", ""])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unique(out: _Out, name: str) -> str:
        base, n = name, 2
        while name in out.names:
            name = f"{base}_{n}"
            n += 1
        out.names.add(name)
        return name

    def _element(self, out: _Out, name: str, etype: str, l_pm: int, pos2: int, params: dict,
                 extra: str = "") -> str:
        """One ``accElement``: ``l_pm`` is the length in picometres, ``pos2`` the centre in half-picometres."""
        name = self._unique(out, name)
        # PyORBIT's factory refuses ``dist = pos1 − L1/2 − (pos0 + L0/2) < 0`` with no tolerance, so
        # lengths and positions live on an exact grid (the decimals written: pm and half-pm), a node
        # that would start before the previous node's exit is moved to it, and when PyORBIT's own
        # double arithmetic on the written numbers would still see an overlap the node is pushed
        # 1 pm further — 10⁴ times below the factory's zeroDistance, so it never becomes a drift in
        # PyORBIT, while the reader keeps the picometre and write → read → write stays a fixed point
        if pos2 - l_pm < out.prev_exit2:
            pos2 = out.prev_exit2 + l_pm
        len_s, pos_s = _dec(l_pm, 12), _dec(5 * pos2, 13)
        while out.prev_pos is not None and \
                float(pos_s) - float(len_s) / 2.0 - (out.prev_pos + out.prev_len / 2.0) < 0.0:
            pos2 += 2
            pos_s = _dec(5 * pos2, 13)
        out.prev_exit2 = pos2 + l_pm
        out.last_pos2 = pos2
        out.prev_pos, out.prev_len = float(pos_s), float(len_s)
        attrs = " ".join(f'{k}="{v}"' for k, v in params.items())
        out.lines.append(f'  <accElement length="{len_s}" name={quoteattr(name)} pos="{pos_s}" '
                         f'type="{etype}">')
        out.lines.append(f"   <parameters {attrs}/>" if attrs else "   <parameters/>")
        if extra:
            out.lines.append(extra)
        out.lines.append("  </accElement>")
        return name

    @staticmethod
    def _circle_aperture(el: Element, rep: FidelityReport) -> dict:
        ap = el.aperture
        if ap is None or ap.x_limits is None or ap.y_limits is None:
            return {}
        rx = (ap.x_limits[1] - ap.x_limits[0]) / 2.0
        ry = (ap.y_limits[1] - ap.y_limits[0]) / 2.0
        centred = abs(ap.x_limits[0] + ap.x_limits[1]) < _THIN and abs(ap.y_limits[0] + ap.y_limits[1]) < _THIN
        if ap.shape == "ELLIPTICAL" and centred and abs(rx - ry) < _THIN:
            return {"aperture": _num(2.0 * rx), "aprt_type": "1"}        # PyORBIT: the full diameter
        rep.lossy("APERTURE_SHAPE", "PyORBIT quad/solenoid apertures are circles (aperture = diameter); "
                  "this aperture was dropped", element=el.name, kind=el.kind)
        return {}

    # ------------------------------------------------------------------ elements
    def _emit(self, el: Element, ref: ReferenceParticle, out: _Out, rep: FidelityReport) -> None:
        kind = el.kind
        brho, brho_abs = ref.brho_signed, abs(ref.brho_signed)
        q = ref.species.charge or 1
        # positions come from the running sum of the *written* (picometre-rounded) lengths, never
        # from the placement's floating s: that is what the reader reconstructs
        l_pm = int(round(float(el.length) * 1e12))
        entrance2 = out.cursor2
        centre2 = entrance2 + l_pm
        exit2 = entrance2 + 2 * l_pm
        if kind == "Superposition":
            rep.lossy("SUPERPOSITION_FLATTENED", "superposed field maps written as consecutive elements",
                      element=el.name, kind=kind)
            for child in el.children:                    # the children advance the cursor
                self._emit(child, ref, out, rep)
            return
        if kind == "FieldMap":
            r = replacement_for(el)
            for part in r.parts:
                self._emit(part, ref, out, rep)
            getattr(rep, r.cls.lower())(r.code, r.message, element=el.name, kind=kind, **r.details)
            for cls, code, msg in r.extra:
                getattr(rep, cls.lower())(code, msg, element=el.name, kind=kind)
            return
        out.cursor2 = exit2
        if kind == "Drift":
            rep.exact(el.name, kind, message="implicit: the factory fills the gap")
            return
        if kind == "Quadrupole":
            mp = el.multipole
            bn1 = float(mp.Bn.get(1, 0.0)) if mp is not None else 0.0
            params = {"field": _num(bn1)}
            params.update(self._circle_aperture(el, rep))
            if mp is not None:
                if any(v for n, v in mp.Bn.items() if n != 1) or any(v for n, v in mp.Bs.items() if n != 1):
                    rep.lossy("MULTIPOLE_ORDERS_DROPPED", "PyORBIT's XML QUAD carries the gradient only; other "
                              "orders and skew components dropped", element=el.name, kind=kind)
                if mp.Bs.get(1):
                    rep.lossy("SKEW_COMPONENT_DROPPED", "a skew quadrupole component has no XML attribute",
                              element=el.name, kind=kind)
                if any(mp.tilt.values()):
                    rep.lossy("PYORBIT_QUAD_TILT_DROPPED", "the XML QUAD has no tilt: the rotated gradient is "
                              "written as a normal one", element=el.name, kind=kind)
            self._shift(el, rep)
            self._element(out, el.name, "QUAD", l_pm, centre2, params)
            rep.exact(el.name, kind)
            return
        if kind in ("Sextupole", "Octupole"):
            rep.lossy("PYORBIT_NO_MULTIPOLE", "PyORBIT's linac XML has no sextupole/octupole element; written as "
                      "a drift (implicit)", element=el.name, kind=kind)
            return
        if kind == "Multipole":
            mp = el.multipole
            bnl = dict(mp.BnL) if mp is not None else {}
            bsl = dict(mp.BsL) if mp is not None else {}
            hk = -float(bnl.get(0, 0.0)) / brho if brho else 0.0
            vk = float(bsl.get(0, 0.0)) / brho if brho else 0.0
            higher = sorted({n for n, v in {**bnl, **bsl}.items() if n > 0 and v})
            if higher:
                rep.lossy("MULTIPOLE_ORDERS_DROPPED", "PyORBIT's linac XML has no thin multipole: only the dipole "
                          f"terms are written (DCH/DCV); dropped orders {higher}", element=el.name, kind=kind)
            rep.lossy("PYORBIT_MULTIPOLE_AS_CORRECTOR", "a thin multipole's dipole terms become DCH/DCV kicks; "
                      "the multipole identity does not survive a read-back", element=el.name, kind=kind)
            self._kicker(out, el, hk, vk, el.length, centre2, brho)
            return
        if kind == "Bend":
            b = el.bend
            if b.angle == 0.0:
                # PyORBIT's Bend.initialize divides by theta; a straight "bend" is a drift here
                mp = el.multipole
                if mp is not None and any(v for v in {**mp.Bn, **mp.Bs}.values()):
                    rep.lossy("MULTIPOLE_ORDERS_DROPPED", "a zero-angle bend with a gradient is written as a "
                              "drift: the gradient is lost", element=el.name, kind=kind)
                rep.equivalent("ZERO_ANGLE_BEND_AS_DRIFT", "zero-angle bend written as a drift (PyORBIT's BEND "
                               "needs rho = L/theta)", element=el.name, kind=kind)
                self._shift(el, rep)
                return
            params = {"theta": _num(b.angle), "ea1": _num(b.e1), "ea2": _num(b.e2), "kls": "0.0", "poles": "0",
                      "skews": "0"}
            ap = el.aperture
            if ap is not None and ap.x_limits is not None and ap.y_limits is not None:
                params["aperture_x"] = _num(ap.x_limits[1] - ap.x_limits[0])
                params["aperture_y"] = _num(ap.y_limits[1] - ap.y_limits[0])
                params["aprt_type"] = "3" if ap.shape == "RECTANGULAR" else "2"
            mp = el.multipole
            if mp is not None and any(v for v in {**mp.Bn, **mp.Bs}.values()):
                rep.lossy("MULTIPOLE_ORDERS_DROPPED", "PyORBIT's BEND multipole content (kls/poles) has an "
                          "undocumented normalization; the bend's gradient was dropped", element=el.name, kind=kind)
            if (b.edge_int1 or b.edge_int2) and b.hgap:
                rep.lossy("BEND_FRINGE_DROPPED", "PyORBIT's BEND has no fringe-field integral (fint·hgap)",
                          element=el.name, kind=kind)
            if b.tilt_ref:
                rep.lossy("PYORBIT_BEND_TILT_DROPPED", "PyORBIT's BEND bends in the horizontal plane only",
                          element=el.name, kind=kind)
            self._shift(el, rep)
            self._element(out, el.name, "BEND", l_pm, centre2, params)
            rep.exact(el.name, kind)
            return
        if kind == "Solenoid":
            params = {"B": _num(float(el.solenoid.Bsol_T) / brho_abs if brho_abs else 0.0)}
            params.update(self._circle_aperture(el, rep))
            self._shift(el, rep)
            self._element(out, el.name, "SOLENOID", l_pm, centre2, params)
            rep.exact(el.name, kind)
            return
        if kind == "RFCavity":
            rf = el.rf
            volt = rf.voltage_V
            if not volt and rf.gradient_V_per_m is not None:
                volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
            if not volt and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-9:
                volt = rf.dE_ref_eV / math.cos(rf.phase_rad)
            freq = float(rf.frequency_Hz or ref.rf_frequency_Hz or 0.0)
            if not freq:
                # a 0 Hz cavity makes BaseRfGap's transverse kick divide by the wavelength (NaN
                # coordinates from the gap on, measured on the ELENA ring): a marker keeps the place
                if volt:
                    rep.lossy("PYORBIT_GAP_NEEDS_FREQUENCY", "an RFGAP without a cavity frequency cannot be "
                              "tracked by PyORBIT; written as a MARKER, its gain is lost", element=el.name, kind=kind)
                else:
                    rep.equivalent("PYORBIT_IDLE_GAP_AS_MARKER", "a cavity with neither voltage nor frequency "
                                   "is written as a MARKER", element=el.name, kind=kind)
                self._element(out, el.name, "MARKER", 0, centre2, {})
                return
            # one <Cavity> per gap: PyORBIT sets only a cavity's first gap from its ``phase`` and
            # derives the later ones from the time of flight (measured 2026-09-05: two bunchers
            # sharing one cavity lose 2.6 % of the MEBT energy)
            cav_name = f"CAV_{len(out.cavities) + 1}"
            phase_deg = math.degrees(rf.phase_rad) + (180.0 if q < 0 else 0.0)
            phase_deg = (phase_deg + 180.0) % 360.0 - 180.0
            params = {"E0L": _num(float(volt or 0.0) * 1e-9), "E0TL": _num(float(volt or 0.0) * 1e-9), "EzFile": "",
                      "cavity": cav_name, "mode": "0", "phase": _num(phase_deg)}
            params.update(self._circle_aperture(el, rep))
            if el.length > _THIN:
                rep.equivalent("THICK_CAVITY_AS_GAP", "PyORBIT RF gaps are thin: the cavity is a gap at its centre "
                               "between two markers at its ends (the length becomes drift)",
                               element=el.name, kind=kind, length=el.length)
                self._element(out, el.name + "_in", "MARKER", 0, entrance2, {})
            native = (el.native.get("pyorbit") or {}).get("ttfs_xml")
            ttf = "   " + (str(native).strip() if native else _constant_ttfs())
            self._element(out, el.name, "RFGAP", 0, centre2, params, extra=ttf)
            out.cavities[float(len(out.cavities) + 1)] = (cav_name, out.last_pos2, freq)   # where the gap went
            if el.length > _THIN:
                self._element(out, el.name + "_out", "MARKER", 0, exit2, {})
            rep.exact(el.name, kind, message="ΔE = q·E0TL·cos(phase); T folded into E0TL")
            return
        if kind in ("NCells", "RFQCell"):
            rep.lossy("NCELLS_TO_DRIFT" if kind == "NCells" else "RFQ_TO_DRIFT",
                      f"{kind} has no PyORBIT XML element; its length is drift", element=el.name, kind=kind)
            return
        if kind == "Kicker":
            if el.electric:
                rep.lossy("EKICK_AS_MAGNETIC", "electric steerer written as a magnetic corrector",
                          element=el.name, kind=kind)
            if el.length > _THIN:
                rep.equivalent("THICK_KICKER_SPLIT", "PyORBIT correctors are thin: the kick sits at the centre "
                               "between two markers at the kicker's ends", element=el.name, kind=kind)
                self._element(out, el.name + "_in", "MARKER", 0, entrance2, {})
            self._kicker(out, el, el.hkick, el.vkick, el.length, centre2, brho)
            if el.length > _THIN:
                self._element(out, el.name + "_out", "MARKER", 0, exit2, {})
            rep.exact(el.name, kind)
            return
        if kind == "Collimator":
            rep.lossy("COLLIMATOR_TO_MARKER", "PyORBIT's XML has no collimator element; a marker keeps the place",
                      element=el.name, kind=kind)
            self._element(out, el.name, "MARKER", 0, centre2, {})
            return
        if kind == "Marker":
            rep.exact(el.name, kind)
            self._element(out, el.name, "MARKER", 0, centre2, {})
            return
        if kind == "Instrument":
            rep.equivalent("INSTRUMENT_AS_MARKER", f"{el.family} instrument written as a MARKER",
                           element=el.name, kind=kind)
            self._element(out, el.name, "MARKER", 0, centre2, {})
            return
        if kind == "Foil":
            rep.lossy("FOIL_TO_MARKER", "the foil is a marker (VACWIN needs a material index lattix cannot derive)",
                      element=el.name, kind=kind)
            self._element(out, el.name, "MARKER", 0, centre2, {})
            return
        if kind == "Taylor":
            rep.lossy("TAYLOR_DROPPED", "PyORBIT's linac XML has no matrix element", element=el.name, kind=kind)
            self._element(out, el.name, "MARKER", 0, centre2, {})
            return
        if kind == "Patch":
            rep.dropped("PATCH_DROPPED", "PyORBIT's linac XML has no coordinate patch", element=el.name, kind=kind)
            return
        if kind == "ReferenceChange":
            rep.dropped("REFCHANGE_DROPPED", "PyORBIT's reference energy follows the gaps only",
                        element=el.name, kind=kind)
            return
        if kind == "Freq":
            rep.exact(el.name, kind, message="the frequency is a <Cavity> attribute")
            return
        if kind == "Directive":
            rep.dropped("FOREIGN_DIRECTIVE", "format-specific directive dropped", element=el.name, kind=kind)
            return
        raise KeyError(kind)                          # pragma: no cover - RULES covers every kind

    def _shift(self, el: Element, rep: FidelityReport) -> None:
        if el.shift is not None and not el.shift.is_zero():
            rep.lossy("MISALIGN_DROPPED", "PyORBIT's linac XML carries no misalignment", element=el.name, kind=el.kind)

    def _kicker(self, out: _Out, el: Element, hk: float, vk: float, length: float, centre2: int, brho: float) -> None:
        eff = length if length > _THIN else _EFF_LENGTH_THIN
        bh = -hk * brho / eff
        bv = vk * brho / eff
        # the reader pairs ``<name>`` (DCH) with ``<name>_V`` (DCV): reserve both names together, so a
        # repeated placement (``COR_2``) keeps its partner (``COR_2_V``) instead of ``COR_V_2``
        name, n = el.name, 2
        while name in out.names or name + "_V" in out.names:
            name = f"{el.name}_{n}"
            n += 1
        self._element(out, name, "DCH", 0, centre2, {"B": _num(bh), "effLength": _num(eff)})
        self._element(out, name + "_V", "DCV", 0, centre2, {"B": _num(bv), "effLength": _num(eff)})


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert check_rules_coverage(Writer()) == set()
