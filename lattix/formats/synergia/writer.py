"""IR → Synergia lattice JSON (``Lattice.as_json()`` layout, cereal 1.3).

MEASURED (Synergia 3 at 17e691d, 2026-09-05, docs/oracles.md Phase 5.9): Synergia keeps one *design*
reference momentum through a lattice while its *bunch* reference particle follows the RF gains, and every
magnet's normalized strength is scaled by ``p_design/p_bunch`` before it acts (``ff_solenoid``,
``ff_quadrupole``: ``brho_l/brho_b``) — so ``k = G/Bρ_design`` reproduces the lab field on an accelerated
bunch (the ``"constant"`` policy of :mod:`lattix.ir.energy_mode`, the default here), the RF phases need no
slip (the bunch's time is measured against its own reference) and a bend after acceleration is under-bent
by ``p_design/p_local`` (its ``sbend`` takes no ``k0``: recorded LOSSY).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage, with_rf_focusing
from lattix.ir.elements import Element
from lattix.ir.energy_mode import (
    check_mode,
    mode_ratio,
    probe_momentum_ratio,
    record_rigidity_mode,
    rigidity_for,
    scale_taylor,
)
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.walk import energy_gain_eV, propagate

C_LIGHT = 299_792_458.0
_THIN = 1e-12
_ACCEL_TOL_eV = 1e-6
#: Synergia's ``element_type`` enum (lattice_element.h) — the ``type`` integer beside ``stype``
TYPE_INDEX = {"generic": 0, "drift": 1, "rbend": 2, "sbend": 3, "quadrupole": 4, "multipole": 5, "rfcavity": 6,
              "hkicker": 7, "vkicker": 8, "kicker": 9, "sextupole": 10, "octupole": 11, "monitor": 12,
              "hmonitor": 13, "vmonitor": 14, "marker": 15, "instrument": 16, "rcollimator": 17, "nllens": 18,
              "solenoid": 19, "elens": 20, "foil": 21, "dipedge": 22, "matrix": 23}


@dataclass(frozen=True)
class Rule:
    message: str
    cls: str
    code: str


@dataclass
class _Entry:
    name: str
    stype: str
    doubles: dict[str, float] = field(default_factory=dict)
    vectors: dict[str, list[float]] = field(default_factory=dict)
    strings: dict[str, str] = field(default_factory=dict)
    tag: dict = field(default_factory=dict)
    length: float = 0.0

    def to_json(self) -> dict:
        strings = dict(self.strings)
        if self.tag:
            strings["lattix"] = _tag(self.tag)
        return {
            "name": self.name, "format": 1, "stype": self.stype, "type": TYPE_INDEX[self.stype], "ancestors": [],
            "string_attributes": [{"key": k, "value": v} for k, v in strings.items()],
            "lazy_double_attributes": [{"key": k, "value": {"value0": _num(v)}} for k, v in self.doubles.items()],
            "lazy_vector_attributes": [{"key": k, "value": [{"value0": _num(x)} for x in v]}
                                       for k, v in self.vectors.items()],
            "length_attribute_name": "l", "bend_angle_attribute_name": "angle", "revision": 0,
            "markers": {"value0": False, "value1": False, "value2": False, "value3": False},
        }


def _num(v: float) -> str:
    v = float(v)
    if v == 0:
        return "0"
    return repr(v)


def _tag(kv: dict) -> str:
    parts = []
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, bool):
            s = "1" if v else "0"
        elif isinstance(v, int | float):
            s = _num(v) if isinstance(v, float) else str(v)
        else:
            s = str(v)
        if " " in s or "=" in s or not s:
            s = '"' + s.replace('"', "'") + '"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport` (a Synergia lattice JSON)."""

    format = "synergia"
    RULES: dict[str, Rule] = {
        "Drift": Rule("drift(l)", "EXACT", "OK"),
        "Quadrupole": Rule("quadrupole(l, k1 = Bn1/Bρ, k1s, tilt, hoffset, voffset)", "EXACT", "OK"),
        "Sextupole": Rule("sextupole(l, k2, k2s, tilt)", "EXACT", "OK"),
        "Octupole": Rule("octupole(l, k3, k3s, tilt)", "EXACT", "OK"),
        "Multipole": Rule("multipole(knl, ksl, tilt) (thin; a drift for a length)", "EXACT", "OK"),
        "Bend": Rule("sbend(l, angle, e1, e2, fint, fintx, hgap, k1, k2, tilt) with sector faces (under-bent after "
                     "acceleration: no k0)", "EXACT", "OK"),
        "Solenoid": Rule("solenoid(l, ks = Bsol/Bρ)", "EXACT", "OK"),
        "RFCavity": Rule("rfcavity(l, volt [MV], lag = φ/2π + ¼ [turns], freq [MHz], harmon)", "EXACT", "OK"),
        "FieldMap": Rule("the lattix.ir.fieldmap replacement ladder", "LOSSY", "FM_REPLACED"),
        "NCells": Rule("drift (Synergia has no cell train)", "LOSSY", "NCELLS_TO_DRIFT"),
        "RFQCell": Rule("drift", "LOSSY", "RFQ_TO_DRIFT"),
        "Kicker": Rule("kicker(l, hkick, vkick, tilt)", "EXACT", "OK"),
        "Collimator": Rule("rcollimator(l, xsize, ysize)", "EXACT", "OK"),
        "Marker": Rule("marker", "EXACT", "OK"),
        "Instrument": Rule("monitor / hmonitor / vmonitor / instrument(l)", "EXACT", "OK"),
        "Foil": Rule("marker (Synergia's foil model has its own parameters)", "LOSSY", "FOIL_TO_MARKER"),
        "Taylor": Rule("matrix(l, rm11 … rm66, kick1 … kick6) in Synergia's (x, xp, y, yp, cdt, dpop)", "EXACT",
                       "OK"),
        "Patch": Rule("marker (no coordinate patch in Synergia)", "LOSSY", "PATCH_DROPPED"),
        "ReferenceChange": Rule("marker with the change in the tag (one design momentum)", "EQUIVALENT",
                                "REFCHANGE_AS_TAG"),
        "Freq": Rule("marker (the frequency is per cavity)", "EXACT", "OK"),
        "Directive": Rule("marker with the card in the tag", "DROPPED", "FOREIGN_DIRECTIVE"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED"),
    }

    def write(self, lattice: Lattice, path, *, strict: bool = False, **options) -> FidelityReport:
        path = Path(path)
        rep = FidelityReport(target_format="synergia", target_file=str(path))
        doc = self.to_document(lattice, report=rep, **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=4) + "\n", encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, **options) -> str:
        return json.dumps(self.to_document(lattice, **options), indent=4) + "\n"

    def to_document(self, lattice: Lattice, *, report: FidelityReport | None = None,
                    energy_mode: str = "constant", install_apertures: bool = True) -> dict:
        check_mode(energy_mode)
        self.rep = report if report is not None else FidelityReport(target_format="synergia")
        lattice, _ = with_rf_focusing(lattice, "synergia")
        self.lat = lattice
        self.energy_mode = energy_mode
        self.install_apertures = install_apertures
        self.used: set[str] = set()
        ref = lattice.reference
        placed = propagate(lattice)
        probes = probe_momentum_ratio(placed, ref)
        self.start_brho = ref.brho_signed
        entries: list[_Entry] = []
        for p, probe in zip(placed, probes, strict=True):
            self.probe = probe
            self.ratio = mode_ratio(energy_mode, (p.ref_in or ref).brho_signed, self.start_brho, probe)
            entries += self._element(p)
        sp = ref.species
        etot = ref.total_energy_eV * 1e-9
        mass = sp.mass_eV * 1e-9
        pc = ref.pc_eV * 1e-9
        doc = {"value0": {
            "name": lattice.name or "lattix",
            "has_reference_particle": True,
            "reference_particle_value": {
                "charge": int(sp.charge),
                "four_momentum": {"mass": mass, "energy": etot, "momentum": pc, "gamma": ref.gamma, "beta": ref.beta},
                "state": {f"value{i}": 0.0 for i in range(6)},
                "repetition": 0, "s": 0.0, "s_n": 0.0, "abs_time": 0.0, "abs_offset": 0.0},
            "elements": [e.to_json() for e in entries],
            "updated": {"value0": True, "value1": True, "value2": True},
            "tree": {"value0": ""}},
            "lattix": {"written_by": f"lattix {__version__}", "source_format": lattice.meta.get("source_format", "IR"),
                       "energy_mode": energy_mode, "species": sp.name, "mass_eV": sp.mass_eV, "charge": sp.charge,
                       "kinetic_energy_eV": ref.kinetic_energy_eV, "rf_frequency_Hz": ref.rf_frequency_Hz,
                       "use": lattice.use or lattice.name}}
        return doc

    # ------------------------------------------------------------------ helpers
    def _record(self, el: Element, rule: Rule, **details) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind, code=rule.code, message=rule.message)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind, **details)

    def _brho(self, p: Placed) -> float:
        """The rigidity this element's normalized strengths are written with (energy_mode)."""
        return rigidity_for(self.energy_mode, (p.ref_in or self.lat.reference).brho_signed, self.start_brho,
                            self.probe)

    def _entry(self, el: Element, stype: str, doubles: dict | None = None, *, vectors: dict | None = None,
               strings: dict | None = None, tag: dict | None = None, suffix: str = "", kind: str | None = None,
               length: float = 0.0) -> _Entry:
        full = {"kind": kind or el.kind}
        if el.name + suffix != el.name or suffix:
            full["name"] = el.name
        if tag:
            full.update(tag)
        if el.meta.get("synergia_from") and "from" not in full and not suffix:
            full["from"] = str(el.meta["synergia_from"])      # a re-read replacement keeps its origin
        return _Entry(el.name + suffix, stype, {k: float(v) for k, v in (doubles or {}).items() if v is not None},
                      {k: [float(x) for x in v] for k, v in (vectors or {}).items()}, dict(strings or {}), full, length)

    def _shift(self, el: Element, e: _Entry, *, offsets_ok: bool = False, tilt_used: bool = False) -> None:
        sh = el.shift
        if sh is None:
            return
        if offsets_ok:
            if sh.x_offset:
                e.doubles["hoffset"] = sh.x_offset
            if sh.y_offset:
                e.doubles["voffset"] = sh.y_offset
        lost = [k for k in ("z_offset", "x_rot", "y_rot") if getattr(sh, k)]
        if not offsets_ok:
            lost = [k for k in ("x_offset", "y_offset") if getattr(sh, k)] + lost
        if sh.tilt and not tilt_used:
            lost.append("tilt")
        if lost:
            self.rep.lossy("MISALIGN_DROPPED", "Synergia carries a quadrupole's hoffset/voffset and a magnet's "
                           "tilt; dropped: " + ", ".join(lost), element=el.name, kind=el.kind)

    def _aperture(self, el: Element, e: _Entry) -> None:
        ap = el.aperture
        if ap is None or not self.install_apertures:
            return
        if ap.x_limits is None or ap.y_limits is None:
            self.rep.lossy("APERTURE_PARTIAL", "an aperture without both x and y limits cannot be written",
                           element=el.name, kind=el.kind)
            return
        (x0, x1), (y0, y1) = ap.x_limits, ap.y_limits
        if abs(x0 + x1) > _THIN or abs(y0 + y1) > _THIN:
            self.rep.lossy("APERTURE_OFFSET_DROPPED", "Synergia apertures are centred on the element; the limits "
                           "were symmetrized", element=el.name, kind=el.kind)
        hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        # Synergia's aperture operations (aperture_operation.h): circular / elliptical radii, rectangular full sizes
        if ap.shape == "ELLIPTICAL" and abs(hx - hy) <= 1e-15 * max(1.0, abs(hx)):
            e.strings["aperture_type"] = "circular"
            e.doubles["circular_aperture_radius"] = hx
        elif ap.shape == "ELLIPTICAL":
            e.strings["aperture_type"] = "elliptical"
            e.doubles["elliptical_aperture_horizontal_radius"] = hx
            e.doubles["elliptical_aperture_vertical_radius"] = hy
        else:
            e.strings["aperture_type"] = "rectangular"
            e.doubles["rectangular_aperture_width"] = 2.0 * hx
            e.doubles["rectangular_aperture_height"] = 2.0 * hy
        if el.kind != "Collimator":
            self.rep.equivalent("APERTURE_AS_ATTRIBUTE", "the aperture is written as Synergia's aperture_type and "
                                "radius/size attributes (applied by its aperture operation)", element=el.name,
                                kind=el.kind)

    # ------------------------------------------------------------------ elements
    def _element(self, p: Placed) -> list[_Entry]:
        el = p.element
        rule = self.RULES.get(el.kind)
        if rule is None:                                          # pragma: no cover - RULES is total
            raise KeyError(f"the Synergia writer has no rule for kind {el.kind!r}")
        if el.kind == "Superposition":
            return self._w_superposition(el, p, rule)
        if el.kind == "FieldMap":
            return self._w_fieldmap(el, p, rule)
        fn = getattr(self, f"_w_{el.kind.lower()}")
        out = fn(el, p, rule)
        if out and el.kind != "Collimator":
            self._aperture(el, out[0])
        gain = energy_gain_eV(el, p.ref_in or self.lat.reference) if el.kind == "RFCavity" else 0.0
        if abs(gain) > _ACCEL_TOL_eV:
            self.rep.equivalent("CONST_P0", "Synergia keeps one design momentum: the reference energy does not "
                                "follow this element's gain (the bunch reference does)", element=el.name,
                                kind=el.kind, dE_eV=gain)
            record_rigidity_mode(self.rep, self.energy_mode, element=el.name, kind=el.kind, dE_eV=gain,
                                 brho=(p.ref_in or self.lat.reference).brho_signed)
        return out

    def _w_drift(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "drift", {"l": el.length}, length=el.length)]

    def _magnet(self, el, p, rule, stype: str, key: str, order: int, extra: dict[str, int] | None = None):
        mp = el.multipole
        brho = self._brho(p)
        d = {"l": el.length}
        bn = float(mp.Bn.get(order, 0.0))
        bs = float(mp.Bs.get(order, 0.0))
        d[key] = bn / brho if brho else 0.0
        if bs:
            d[key + "s"] = bs / brho if brho else 0.0
        used = {order}
        for k2, o2 in (extra or {}).items():
            v = float(mp.Bn.get(o2, 0.0))
            if v:
                d[k2] = v / brho if brho else 0.0
                used.add(o2)
        tilt = float(mp.tilt.get(order, 0.0)) + (float(el.shift.tilt) if el.shift is not None else 0.0)
        if tilt:
            d["tilt"] = tilt
        others = sorted(n for n, v in {**mp.Bn, **mp.Bs}.items() if v and n not in used)
        if others:
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", f"Synergia's {stype} carries orders {sorted(used)}; dropped "
                           f"orders {others}", element=el.name, kind=el.kind)
        e = self._entry(el, stype, d, length=el.length)
        self._shift(el, e, offsets_ok=(stype == "quadrupole"), tilt_used=True)
        self._record(el, rule)
        return [e]

    def _w_quadrupole(self, el, p, rule):
        return self._magnet(el, p, rule, "quadrupole", "k1", 1)

    def _w_sextupole(self, el, p, rule):
        return self._magnet(el, p, rule, "sextupole", "k2", 2)

    def _w_octupole(self, el, p, rule):
        return self._magnet(el, p, rule, "octupole", "k3", 3)

    def _w_multipole(self, el, p, rule):
        mp = el.multipole
        brho = self._brho(p)
        top = max([n for n, v in {**mp.BnL, **mp.BsL}.items() if v] + [0])
        knl = [float(mp.BnL.get(n, 0.0)) / brho if brho else 0.0 for n in range(top + 1)]
        ksl = [float(mp.BsL.get(n, 0.0)) / brho if brho else 0.0 for n in range(top + 1)]
        d = {}
        tilt = float(el.shift.tilt) if el.shift is not None and el.shift.tilt else 0.0
        if tilt:
            d["tilt"] = tilt
        vec = {"knl": knl}
        if any(ksl):
            vec["ksl"] = ksl
        e = self._entry(el, "multipole", d, vectors=vec)
        self._shift(el, e, tilt_used=True)
        self._record(el, rule)
        out = [e]
        if el.length > _THIN:
            self.rep.equivalent("THICK_MULTIPOLE_SPLIT", "Synergia's multipole is thin: the kick at the entrance, "
                                "a drift for the length", element=el.name, kind=el.kind)
            out.append(self._entry(el, "drift", {"l": el.length}, suffix="_body", tag={"role": "body"},
                                   length=el.length))
        return out

    def _w_bend(self, el, p, rule):
        b = el.bend
        brho = self._brho(p)
        mp = el.multipole
        d = {"l": el.length, "angle": b.angle}
        k1 = float(mp.Bn.get(1, 0.0)) / brho if brho else 0.0
        k2 = float(mp.Bn.get(2, 0.0)) / brho if brho else 0.0
        if k1:
            d["k1"] = k1
        if k2:
            d["k2"] = k2
        if b.e1:
            d["e1"] = b.e1
        if b.e2:
            d["e2"] = b.e2
        if b.edge_int1:
            d["fint"] = float(b.edge_int1)
        if b.edge_int2 is not None and abs(float(b.edge_int2) - float(b.edge_int1 or 0.0)) > 1e-15:
            d["fintx"] = float(b.edge_int2)
        if b.hgap:
            d["hgap"] = float(b.hgap)
        tilt = float(b.tilt_ref) + (float(el.shift.tilt) if el.shift is not None else 0.0)
        if tilt:
            d["tilt"] = tilt
        if abs(self.ratio - 1.0) > 1e-15 and el.length:
            # MEASURED: Synergia derives the bend field from the design momentum and its sbend takes no k0, so an
            # accelerated bunch is bent by angle·p_design/p_local (R21 = −h·sin(θ·p_design/p_local) on a 10°
            # bend after a 1 MV gap at 2.1 MeV)
            self.rep.lossy("CONST_P0_BEND_UNDERBENT", "Synergia bends the accelerated bunch by angle·p_design/p_local "
                           "(no k0 on its sbend): the geometry downstream of this bend is not the design's",
                           element=el.name, kind=el.kind, ratio=self.ratio)
        if any(v for n, v in {**mp.Bn, **mp.Bs}.items() if n not in (1, 2)):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "Synergia's sbend carries k1 and k2 only", element=el.name,
                           kind=el.kind)
        tag = {"rect": True} if b.rect else None
        e = self._entry(el, "sbend", d, tag=tag, length=el.length)
        self._shift(el, e, tilt_used=True)
        self._record(el, rule)
        return [e]

    def _w_solenoid(self, el, p, rule):
        brho = self._brho(p)
        e = self._entry(el, "solenoid", {"l": el.length, "ks": float(el.solenoid.Bsol_T) / brho if brho else 0.0},
                        length=el.length)
        self._shift(el, e)
        self._record(el, rule)
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
            self.rep.lossy("RF_FREQUENCY_UNKNOWN", "Synergia's rfcavity needs a frequency; 0 MHz written",
                           element=el.name, kind=el.kind)
        # MAD-X's gain rule (Synergia: E1 = E0 + volt·sin(2π·lag)): lag = φ/2π + 1/4 for the IR's cos convention;
        # MEASURED: no phase slip — the bunch's time is measured against its own (accelerated) reference, so a
        # second gap at −30° gains V·cos 30° with the plain lag (the MAD-X writer's slip would decelerate it)
        lag = (rf.phase_rad / (2.0 * math.pi) + 0.25) % 1.0
        d = {"l": el.length, "volt": float(V or 0.0) * 1e-6, "lag": lag, "freq": float(freq) * 1e-6}
        if rf.harmon:
            d["harmon"] = float(rf.harmon)
        tag = {"phase": rf.phase_rad}
        if rf.cavity_type == "TRAVELING_WAVE":
            tag["tw"] = True
        if rf.n_cell:
            tag["n"] = int(rf.n_cell)
        if not rf.phase_is_sync:
            tag["raw"] = True
        e = self._entry(el, "rfcavity", d, tag=tag, length=el.length)
        self._shift(el, e)
        self._record(el, rule)
        return [e]

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
        self._record(el, rule)
        return [self._entry(el, "drift", {"l": el.length}, length=el.length)]

    def _w_rfqcell(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "drift", {"l": el.length}, length=el.length)]

    def _w_kicker(self, el, p, rule):
        if el.electric:
            self.rep.lossy("EKICK_AS_MAGNETIC", "an electric kicker is written as a magnetic kicker of the same "
                           "deflection", element=el.name, kind=el.kind)
        r = self.ratio                      # the engine's kick is Δpx/p_used = kick_IR · p_local/p_used
        d = {"l": el.length, "hkick": float(el.hkick) * r, "vkick": float(el.vkick) * r}
        e = self._entry(el, "kicker", d, length=el.length)
        self._shift(el, e)
        self._record(el, rule)
        return [e]

    def _w_collimator(self, el, p, rule):
        ap = el.aperture
        d = {"l": el.length}
        if ap is not None and ap.x_limits is not None and ap.y_limits is not None:
            d["xsize"] = (ap.x_limits[1] - ap.x_limits[0]) / 2.0
            d["ysize"] = (ap.y_limits[1] - ap.y_limits[0]) / 2.0
            self._record(el, rule)
        else:
            self.rep.lossy("COLLIMATOR_TO_MARKER", "a collimator without limits is a drift", element=el.name,
                           kind=el.kind)
        e = self._entry(el, "rcollimator", d, length=el.length)
        self._aperture(el, e)                          # the operation Synergia applies (xsize/ysize are MAD-X's)
        return [e]

    def _w_marker(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "marker")]

    def _w_instrument(self, el, p, rule):
        fam = (el.family or "MONITOR").upper()
        stype = {"BPM": "monitor", "MONITOR": "monitor", "HMONITOR": "hmonitor", "VMONITOR": "vmonitor"}.get(
            fam, "instrument")
        self._record(el, rule, family=el.family)
        tag = {"family": el.family}
        if el.params:
            tag["params"] = ";".join(f"{k}:{v}" for k, v in el.params.items())
        return [self._entry(el, stype, {"l": el.length}, tag=tag, length=el.length)]

    def _w_foil(self, el, p, rule):
        self._record(el, rule)
        tag = {"material": el.material, "thick": el.thickness_kg_per_m2}
        if el.dE_ref_eV is not None:
            tag["dE"] = el.dE_ref_eV
        out = [self._entry(el, "marker", tag=tag)]
        if el.length > _THIN:
            out.append(self._entry(el, "drift", {"l": el.length}, suffix="_body", tag={"role": "body"},
                                   length=el.length))
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
        if basis in ("common", "bmad", "xtrack", "cheetah", "madx", "impactx", "ocelot", "dynac"):
            src = Basis(basis)
            d_src = np.diag(transform_matrix(src, ref.kinetic_energy_eV, ref.species.mass_eV, ref.rf_frequency_Hz))
            d_dst = np.diag(transform_matrix(Basis.SYNERGIA, ref.kinetic_energy_eV, ref.species.mass_eV))
            m, off = similarity_diag(m, off, d_src, d_dst)
            stored = "synergia"
            if basis != "synergia":
                self.rep.equivalent("TAYLOR_BASIS_SYNERGIA", f"the {basis} map was transformed into Synergia's "
                                    "(x, xp, y, yp, cdt, dpop) basis at the entry energy", element=el.name,
                                    kind=el.kind)
        if abs(self.ratio - 1.0) > 1e-15:
            m, off = scale_taylor(m, off, self.ratio)
        out: list[_Entry] = []
        if el.length > _THIN:
            # MEASURED: Synergia's matrix must have zero length — the map is written thin, followed by a drift
            # of the element's length whose map is taken out of the matrix (R_written = R · D⁻¹ in Synergia's
            # basis at the entry energy), so matrix ∘ drift is the IR map and the survey is kept
            from lattix.oracles.basis import drift_common

            T = transform_matrix(Basis.SYNERGIA, ref.kinetic_energy_eV, ref.species.mass_eV)
            D = np.linalg.inv(T) @ drift_common(el.length, ref.kinetic_energy_eV, ref.species.mass_eV) @ T
            m = (np.asarray(m) @ np.linalg.inv(D)).tolist()
            self.rep.equivalent("TAYLOR_THIN_PLUS_DRIFT", "Synergia's matrix is thin: the map (with the drift over "
                                "the element's length divided out) is followed by a drift of that length",
                                element=el.name, kind=el.kind)
        d = {"l": 0.0}
        for i in range(6):
            for j in range(6):
                if m[i][j] != 0.0:
                    d[f"rm{i + 1}{j + 1}"] = m[i][j]
            if off[i] != 0.0:
                d[f"kick{i + 1}"] = off[i]
        self._record(el, rule)
        tag = {"basis": stored}
        if el.length > _THIN:
            tag["L"] = el.length
        out.append(self._entry(el, "matrix", d, tag=tag))
        if el.length > _THIN:
            out.append(self._entry(el, "drift", {"l": el.length}, suffix="_body", tag={"role": "body"},
                                   length=el.length))
        return out

    def _w_patch(self, el, p, rule):
        self._record(el, rule)
        tag = {k: getattr(el, k) for k in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
               if getattr(el, k)}
        if el.length > _THIN:
            return [self._entry(el, "drift", {"l": el.length}, tag=tag, length=el.length)]
        return [self._entry(el, "marker", tag=tag)]

    def _w_referencechange(self, el, p, rule):
        self._record(el, rule, dE_ref_eV=el.dE_ref_eV)
        tag = {}
        if el.dE_ref_eV is not None:
            tag["dE"] = el.dE_ref_eV
        if el.energy_eV is not None:
            tag["E"] = el.energy_eV
        return [self._entry(el, "marker", tag=tag or None)]

    def _w_freq(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "marker", tag={"f": el.frequency_Hz})]

    def _w_directive(self, el, p, rule):
        self._record(el, rule)
        return [self._entry(el, "marker", tag={"format": el.format, "card": el.card, "args": " ".join(el.args),
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


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert check_rules_coverage(Writer()) == set()
