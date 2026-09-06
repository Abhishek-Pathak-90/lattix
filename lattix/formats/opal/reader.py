"""OPAL-T input deck → IR.

Statements end with ``;`` (``//`` and ``/* */`` comments, identifiers case-insensitive); ``name: TYPE,
attr=value, …`` defines an element (a defined element may serve as the type of another), ``name: LINE =
(…)`` a line, ``name: BEAM, …`` the beam, ``TRACK, LINE=…`` selects the line.  OPAL-T places every element
by its ``ELEMEDGE`` [m] (drifts carry no field), so the IR sequence is the ``LINE``'s order with drifts
inserted for the gaps; lattix's ``// lattix:`` tags restore what the writer folded (thin elements over
a surrogate length, solenoid map padding, the origin of a replaced element, patches, reference
changes).  Normalized strengths are multiplied by the BEAM's ``P0/c`` (unsigned), cavity voltages come
from ``DESIGNENERGY`` (the crest energy) or from an integration of the ``1DDynamic`` map, phases from
``LAG`` (relative to the autophased crest)."""
from __future__ import annotations

import math
import re
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.formats.mad8.reader import split_top_level
from lattix.formats.opal.maps import DYNAMIC, STATIC, parse_map
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    NCells,
    Octupole,
    Patch,
    Provenance,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.expr import ExpressionError, LazyResolver, evaluate
from lattix.ir.lattice import Lattice, Line, LineItem
from lattix.ir.reference import SPECIES, ReferenceParticle, Species, species
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.walk import energy_gain_eV

C_LIGHT = 299_792_458.0
_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')
_THIN = 1e-12
_PARTICLE = {"PROTON": "proton", "ELECTRON": "electron", "POSITRON": "positron", "ANTIPROTON": "antiproton",
             "DEUTERON": "deuteron", "HMINUS": "h-"}
_MAGNETS = {"QUADRUPOLE": (Quadrupole, 1, "K1", "K1S"), "SEXTUPOLE": (Sextupole, 2, "K2", "K2S"),
            "OCTUPOLE": (Octupole, 3, "K3", "K3S")}


def _parse_tag(text: str) -> dict[str, str]:
    return {k: (q if v.startswith('"') else v) for k, v, q in _KV.findall(text or "")}


def _param_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


class _Statement:
    __slots__ = ("text", "tag", "lineno")

    def __init__(self, text: str, lineno: int) -> None:
        self.text = text
        self.tag: dict[str, str] = {}
        self.lineno = lineno


def _strip_block_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _split_comment(line: str) -> tuple[str, str]:
    quote = False
    for i, ch in enumerate(line):
        if ch == '"':
            quote = not quote
        elif ch == "/" and not quote and line.startswith("//", i):
            return line[:i], line[i + 2:]
    return line, ""


def statements(text: str) -> tuple[list[_Statement], list[dict[str, str]]]:
    """``;``-terminated statements with their ``// lattix:`` tags, plus the header tags (comment-only lines)."""
    out: list[_Statement] = []
    headers: list[dict[str, str]] = []
    buf: list[str] = []
    depth = 0
    quote = False
    first = 0
    pending_tag: dict[str, str] | None = None
    for no, raw in enumerate(_strip_block_comments(text).splitlines(), 1):
        code, comment = _split_comment(raw)
        done_here: list[_Statement] = []
        for ch in code:
            if ch == '"':
                quote = not quote
            elif not quote and ch in "({[":
                depth += 1
            elif not quote and ch in ")}]":
                depth -= 1
            if ch == ";" and depth <= 0 and not quote:
                st = _Statement("".join(buf).strip(), first or no)
                if pending_tag:
                    st.tag = pending_tag
                    pending_tag = None
                if st.text:
                    out.append(st)
                    done_here.append(st)
                buf = []
                first = 0
            else:
                if not buf and ch.strip():
                    first = no
                buf.append(ch)
        tag_text = comment.strip()
        if tag_text.startswith("lattix:"):
            tag = _parse_tag(tag_text[len("lattix:"):])
            if tag.get("directive"):
                st = _Statement("", no)              # a dropped directive keeps its place in the sequence
                st.tag = tag
                out.append(st)
            elif done_here:
                done_here[-1].tag = tag
            elif "".join(buf).strip():
                pending_tag = tag
            else:
                headers.append(tag)
    if "".join(buf).strip():
        out.append(_Statement("".join(buf).strip(), first))
    return out, headers


def parse_attributes(body: str) -> dict[str, str | bool]:
    attrs: dict[str, str | bool] = {}
    for part in split_top_level(body):
        if not part:
            continue
        if "=" not in part:
            attrs[part.strip().upper()] = True
            continue
        key, _, val = part.partition("=")
        attrs[key.strip().upper()] = val.strip()
    return attrs


_DEF = re.compile(r"^(\w+)\s*:\s*(\w+)\s*(?:,\s*(.*))?$", re.S)
_LINE = re.compile(r"^(\w+)\s*:\s*LINE\s*=\s*\((.*)\)\s*$", re.I | re.S)
_VAR = re.compile(r"^(?:(?:REAL|BOOL|CONST|STRING)\s+)?(\w+)\s*=\s*(.*)$", re.I | re.S)
_CMD = re.compile(r"^(\w+)\s*(?:,\s*(.*))?$", re.S)


class _Def:
    __slots__ = ("name", "etype", "attrs", "tag", "lineno")

    def __init__(self, name: str, etype: str, attrs: dict, tag: dict, lineno: int) -> None:
        self.name, self.etype, self.attrs, self.tag, self.lineno = name, etype, attrs, tag, lineno


class Reader:
    format = "opal"

    def read(self, path: Path, *, strict: bool = False, species: str | Species | None = None,
             kinetic_energy_eV: float | None = None, frequency_Hz: float | None = None,
             line: str | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="opal", source_file=str(path))
        self.rep = rep
        self.path = path
        self.warnings: list[str] = []
        text = path.read_text(encoding="latin-1", errors="replace")
        stmts, headers = statements(text)
        self.defs: dict[str, _Def] = {}
        self.lines: dict[str, list[str]] = {}
        self.variables: dict[str, str] = {}
        self.beam: dict[str, str | bool] = {}
        self.options: dict[str, str | bool] = {}
        self.track: dict[str, str | bool] = {}
        self.title = ""
        self.directives: dict[str | None, list[dict]] = {}   # the element each directive precedes
        self._pending_directives: list[dict] = []
        order: list[str] = []
        for st in stmts:
            if not st.text and st.tag.get("directive"):
                self._pending_directives.append(st.tag)
                continue
            self._statement(st, order)
        if self._pending_directives:
            self.directives.setdefault(None, []).extend(self._pending_directives)
        header_lattice = next((t for t in headers if "use" in t or "name" in t), {})
        ref_tag = parse_reference_tag(text)
        ref = self._reference(ref_tag, species, kinetic_energy_eV, frequency_Hz)
        self.brho0 = ref.brho_abs
        self.ref = ref
        use = self._pick_line(line, header_lattice)
        names = self._expand(use) if use else order
        elements = self._sequence(names)
        elements = self._fold(elements)
        lat = self._assemble(str(header_lattice.get("use") or use or path.stem), elements, ref)
        lat.name = str(header_lattice.get("name") or self.title or use or path.stem)
        lat.meta["source_format"] = "opal"
        lat.meta["source_file"] = str(path)
        lat.warnings.extend(self.warnings)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ statements
    def _statement(self, st: _Statement, order: list[str]) -> None:
        t = st.text.strip()
        if not t:
            return
        m = _LINE.match(t)
        if m:
            self.lines[m.group(1).upper()] = split_top_level(m.group(2))
            return
        m = _DEF.match(t)
        if m and m.group(2).upper() != "LINE":
            name, etype, body = m.group(1), m.group(2).upper(), m.group(3) or ""
            attrs = parse_attributes(body)
            if etype == "BEAM":
                self.beam = attrs
                return
            if etype in ("FIELDSOLVER", "DISTRIBUTION", "WAKE", "GEOMETRY", "PARTICLEMATTERINTERACTION",
                         "OUTPUTPLANE"):
                return
            self.defs[name.upper()] = _Def(name, etype, attrs, st.tag, st.lineno)
            order.append(name.upper())
            if self._pending_directives:
                self.directives.setdefault(name.upper(), []).extend(self._pending_directives)
                self._pending_directives = []
            return
        m = _VAR.match(t)
        if m and ":" not in t.split("=", 1)[0]:
            self.variables[m.group(1).upper()] = m.group(2).strip()
            return
        m = _CMD.match(t)
        if m:
            key, body = m.group(1).upper(), m.group(2) or ""
            attrs = parse_attributes(body)
            if key == "TITLE":
                s = attrs.get("STRING")
                self.title = str(s).strip('"') if isinstance(s, str) else ""
            elif key == "OPTION":
                self.options.update(attrs)
            elif key == "TRACK":
                self.track.update(attrs)
            elif key == "BEAM":
                self.beam = attrs
            return
        self.warnings.append(f"line {st.lineno}: statement not understood: {t[:60]!r}")

    def _pick_line(self, requested: str | None, header: dict) -> str | None:
        for cand in (requested, header.get("use"), self.track.get("LINE") if isinstance(self.track.get("LINE"), str)
                     else None):
            if cand and str(cand).upper() in self.lines:
                return str(cand).upper()
        if self.lines:
            return list(self.lines)[-1]
        return None

    def _expand(self, name: str, depth: int = 0) -> list[str]:
        if depth > 50:
            raise ValueError(f"LINE {name!r} nests too deeply (circular?)")
        out: list[str] = []
        for entry in self.lines[name]:
            e = entry.strip()
            rep = 1
            m = re.match(r"^(-?\d+)\s*\*\s*(.*)$", e)
            if m:
                rep, e = int(m.group(1)), m.group(2).strip()
            reverse = rep < 0
            rep = abs(rep)
            if e.startswith("(") and e.endswith(")"):
                key = f"__inline_{len(self.lines)}"
                self.lines[key] = split_top_level(e[1:-1])
                items = self._expand(key, depth + 1)
            elif e.startswith("-"):
                reverse, e = not reverse, e[1:].strip()
                items = self._expand(e.upper(), depth + 1) if e.upper() in self.lines else [e.upper()]
            elif e.upper() in self.lines:
                items = self._expand(e.upper(), depth + 1)
            else:
                items = [e.upper()]
            if reverse:
                items = list(reversed(items))
            out.extend(items * rep)
        return out

    # ------------------------------------------------------------------ values
    def _resolver(self) -> LazyResolver:
        if not hasattr(self, "_res"):
            self._res = LazyResolver({k.lower(): v for k, v in self.variables.items()})
        return self._res

    def _num(self, v, default: float = 0.0) -> float:
        if v is None or v is True or v is False:
            return default if v is None else float(v)
        s = str(v).strip().strip('"')
        try:
            return float(s)
        except ValueError:
            pass
        try:
            return float(evaluate(s.lower(), {}, self._resolver()))
        except ExpressionError as e:
            self.warnings.append(f"expression {s!r} not evaluated ({e}); 0 used")
            return default

    def _arr(self, v) -> list[float]:
        if v is None or v is True:
            return []
        s = str(v).strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        return [self._num(x) for x in split_top_level(s) if x.strip()]

    @staticmethod
    def _str(v) -> str:
        return "" if v is None or v is True else str(v).strip().strip('"')

    # ------------------------------------------------------------------ reference
    def _reference(self, tag, sp_opt, ke_opt, f_opt) -> ReferenceParticle:
        beam = self.beam
        sp: Species | None = None
        ke = None
        f = None
        if tag is not None:
            sp, ke, f = tag.species, tag.kinetic_energy_eV, tag.rf_frequency_Hz
        if beam:
            pname = self._str(beam.get("PARTICLE")).upper()
            mass = self._num(beam.get("MASS"), 0.0) * 1e9 if beam.get("MASS") is not None else None
            charge = self._num(beam.get("CHARGE"), 1.0) if beam.get("CHARGE") is not None else None
            if sp is None:
                base = SPECIES.get(_PARTICLE.get(pname, "")) if pname else None
                if base is not None and (mass is None or abs(mass - base.mass_eV) <= 1e-6 * base.mass_eV) \
                        and (charge is None or int(round(charge)) == base.charge):
                    sp = base
                elif mass:
                    sp = Species(name=pname.lower() if pname else f"ion_{mass:.6g}", mass_eV=float(mass),
                                 charge=int(round(charge if charge is not None else 1.0)))
                    self.rep.equivalent("SPECIES_ASSUMED", f"the BEAM's MASS {mass:.6g} eV and CHARGE are none of "
                                        "lattix's named species; an ion of that mass", element="BEAM", kind="beam")
                elif base is not None:
                    sp = base
            if ke is None and sp is not None:
                if beam.get("PC") is not None:
                    pc = self._num(beam.get("PC")) * 1e9
                    ke = math.sqrt(pc * pc + sp.mass_eV ** 2) - sp.mass_eV
                elif beam.get("ENERGY") is not None:
                    ke = self._num(beam.get("ENERGY")) * 1e9 - sp.mass_eV
                elif beam.get("GAMMA") is not None:
                    ke = (self._num(beam.get("GAMMA")) - 1.0) * sp.mass_eV
            if f is None and beam.get("BFREQ") is not None:
                f = self._num(beam.get("BFREQ")) * 1e6
        if sp_opt is not None:
            sp = species(sp_opt)
        if ke_opt is not None:
            ke = float(ke_opt)
        if f_opt is not None:
            f = float(f_opt)
        if sp is None or ke is None:
            raise ValueError(f"{self.path.name}: no BEAM with PARTICLE/MASS and PC/ENERGY found; pass species= and "
                             "kinetic_energy_eV= to read(...)")
        return ReferenceParticle(species=sp, kinetic_energy_eV=float(ke), rf_frequency_Hz=f)

    # ------------------------------------------------------------------ elements
    def _resolved(self, key: str, depth: int = 0) -> tuple[str, dict, dict, int, str]:
        """``(OPAL type, attributes incl. inherited, tag, line number, original name)``."""
        d = self.defs[key]
        if d.etype in self.defs and depth < 20:
            etype, attrs, _tag, _no, _nm = self._resolved(d.etype, depth + 1)
            merged = {**attrs, **d.attrs}
            return etype, merged, d.tag, d.lineno, d.name
        return d.etype, dict(d.attrs), d.tag, d.lineno, d.name

    def _sequence(self, names: list[str]) -> list[Element]:
        out: list[Element] = []
        s_end = 0.0
        ref = self.ref
        for key in names:
            if key not in self.defs:
                self.warnings.append(f"LINE member {key!r} is not a defined element; skipped")
                continue
            for tag in self.directives.pop(key, []):         # written just before this element
                out.append(self._directive(tag))
            etype, attrs, tag, lineno, oname = self._resolved(key)
            el, s_in, length = self._convert(etype, attrs, tag, lineno, oname, ref)
            if el is None:
                continue
            if s_in is not None:
                gap = s_in - s_end
                if gap > 1e-9:
                    out.append(Drift(name=f"{el.name}_gap", length=gap))
                    if not tag:
                        self.rep.equivalent("OPAL_GAP_DRIFT_INSERTED", f"a drift of {gap:.6g} m fills the space "
                                            "before the element's ELEMEDGE", element=el.name, kind=el.kind)
                elif gap < -1e-9 and not tag:
                    self.warnings.append(f"{el.name}: ELEMEDGE {s_in:.6g} m overlaps the previous element ending "
                                         f"at {s_end:.6g} m; kept in LINE order")
                s_end = s_in + length
            else:
                s_end += length
            out.append(el)
            ref = ref.advanced(dE_eV=energy_gain_eV(el, ref), ds_m=el.length,
                               rf_frequency_Hz=(el.frequency_Hz if isinstance(el, Freq) else
                                                (getattr(el, "rf", None).frequency_Hz
                                                 if getattr(el, "rf", None) is not None and el.rf.frequency_Hz
                                                 else None)))
        for tags in self.directives.values():              # after the last element (or an unused one)
            out.extend(self._directive(t) for t in tags)
        self.directives = {}
        return out

    def _directive(self, tag: dict) -> Directive:
        return Directive(name=str(tag.get("name") or "directive"), format=str(tag.get("format") or "tracewin"),
                         card=str(tag.get("card") or ""), args=str(tag.get("args") or "").split(),
                         role=str(tag.get("role") or "other"))

    def _convert(self, etype: str, attrs: dict, tag: dict, lineno: int, oname: str, ref: ReferenceParticle):
        """``(element, entrance s from ELEMEDGE or None, its length)``."""
        name = str(tag.get("name") or oname)
        kind = tag.get("kind")
        prov = Provenance(format="opal", line=lineno, original_name=oname, original_type=etype)
        common = {"name": name, "provenance": prov}
        L = self._num(attrs.get("L"), 0.0)
        s = self._num(attrs.get("ELEMEDGE")) if attrs.get("ELEMEDGE") is not None else None
        thin = "L" in tag and float(tag["L"]) <= _THIN
        pad = float(tag.get("pad", 0.0) or 0.0)
        b0 = self.brho0
        el: Element | None = None
        length = None
        if etype == "DRIFT":
            el = self._drift_kind(kind, L, tag, common)
        elif etype in _MAGNETS:
            cls, order, kn, ks = _MAGNETS[etype]
            mp = MagneticMultipoleP(Bn={order: self._num(attrs.get(kn)) * b0})
            if attrs.get(ks) is not None:
                mp.Bs[order] = self._num(attrs.get(ks)) * b0
            if attrs.get("PSI") is not None:
                mp.tilt[order] = self._num(attrs.get("PSI"))
            el = cls(length=L, multipole=mp, **common)
        elif etype == "MULTIPOLE":
            knv, ksv = self._arr(attrs.get("KN")), self._arr(attrs.get("KS"))
            ell = L if L > _THIN else 1.0
            mp = MagneticMultipoleP(BnL={n: v * ell * b0 for n, v in enumerate(knv) if v},
                                    BsL={n: v * ell * b0 for n, v in enumerate(ksv) if v})
            if attrs.get("PSI") is not None:
                mp.tilt[1] = self._num(attrs.get("PSI"))
            el = Multipole(length=0.0 if thin else L, multipole=mp, **common)
            el.meta["opal_role"] = tag.get("role", "higher")
        elif etype in ("SBEND", "RBEND", "RBEND3D"):
            angle = self._num(attrs.get("ANGLE"))
            e1, e2 = self._num(attrs.get("E1")), self._num(attrs.get("E2"))
            rect = bool(tag.get("rect")) or etype != "SBEND"
            if etype != "SBEND":
                e1 += 0.5 * angle
                e2 += 0.5 * angle
                self.rep.equivalent("RBEND_AS_SECTOR", "OPAL's RBEND is read as a sector bend with angle/2 added to "
                                    "its faces", element=name, kind="Bend")
            hgap = self._num(attrs.get("HGAP")) if attrs.get("HGAP") is not None else 0.5 * self._num(attrs.get("GAP"))
            mp = MagneticMultipoleP()
            if attrs.get("K1") is not None:
                mp.Bn[1] = self._num(attrs.get("K1")) * b0
            if attrs.get("K2") is not None:
                mp.Bn[2] = self._num(attrs.get("K2")) * b0
            el = Bend(length=L, multipole=mp, **common,
                      bend=BendP(angle=angle, e1=e1, e2=e2, edge_int1=self._num(attrs.get("FINT")), hgap=hgap,
                                 tilt_ref=self._num(attrs.get("PSI")), rect=rect))
        elif etype == "SOLENOID":
            data = self._map(attrs)
            peak = 1.0 if (data is None or data.normalized) else (data.peak or 1.0)
            scale = self._num(attrs.get("KS")) * b0            # B(z) = KS · P0/c · map(z)
            B = scale * peak
            length = float(tag["L"]) if "L" in tag else (data.length_m - 2.0 * pad if data is not None else L)
            shift = pad
            if kind == "FieldMap" and data is not None and len(data.z_m) > 1:
                # lattix wrote the map from a field map: read it back as the hard-edge solenoid preserving the
                # profile's ∫B and ∫B² (what lattix.ir.fieldmap.replacement_for gives a mapless target),
                # centred in the map — not the peak field over the map's extent
                z = [float(v) for v in data.z_m]
                bz = [scale * float(v) for v in data.values]
                int1 = sum(0.5 * (bz[i] + bz[i + 1]) * (z[i + 1] - z[i]) for i in range(len(z) - 1))
                int2 = sum(0.5 * (bz[i] ** 2 + bz[i + 1] ** 2) * (z[i + 1] - z[i]) for i in range(len(z) - 1))
                if int1 and int2 > 0.0:
                    length, B = int1 * int1 / int2, int2 / int1
                    shift = 0.5 * (data.length_m - length)
            el = Solenoid(length=length, solenoid=SolenoidP(Bsol_T=B), **common)
            if data is not None:
                el.native["opal"] = {"kind": STATIC, "z": list(data.z_m), "values": list(data.values),
                                     "map": self._str(attrs.get("FMAPFN")), "pad": shift, "file": tag.get("file"),
                                     "scale_T": scale}
            if s is not None:
                s += shift
            if kind == "FieldMap":
                el.meta["opal_from"] = "FieldMap"
                self.rep.equivalent("FM_READ_AS_CAVITY", "a solenoid written from a field map is read as the hard-"
                                    "edge solenoid preserving the map's ∫B and ∫B² (L_eff, B_eff, centred in the "
                                    "map); the profile is kept in native['opal']", element=name, kind="Solenoid")
        elif etype in ("RFCAVITY", "TRAVELINGWAVE", "VARIABLE_RF_CAVITY"):
            el, length = self._cavity(etype, attrs, tag, common, ref, L, thin, pad)
            if s is not None:
                s += pad if not thin else 0.5 * (L if L > _THIN else 0.0)
            thin = False                                     # the centre shift is done here
        elif etype in ("KICKER", "HKICKER", "VKICKER", "CORRECTOR"):
            if attrs.get("K0") is not None or attrs.get("K0S") is not None:
                brs = ref.brho_signed or b0
                hk = -self._num(attrs.get("K0")) * L / brs
                vk = self._num(attrs.get("K0S")) * L / brs
            elif etype == "HKICKER":
                hk, vk = self._num(attrs.get("KICK")), 0.0
            elif etype == "VKICKER":
                hk, vk = 0.0, self._num(attrs.get("KICK"))
            else:
                hk, vk = self._num(attrs.get("HKICK")), self._num(attrs.get("VKICK"))
            if kind == "Multipole":
                brs = ref.brho_signed or b0
                mp = MagneticMultipoleP(BnL={0: -hk * brs} if hk else {}, BsL={0: vk * brs} if vk else {})
                el = Multipole(length=0.0 if thin else L, multipole=mp, **common)
                el.meta["opal_role"] = "dipole"
            else:
                el = Kicker(length=0.0 if thin else L, hkick=hk, vkick=vk, **common)
        elif etype in ("RCOLLIMATOR", "ECOLLIMATOR"):
            hx, hy = self._num(attrs.get("XSIZE")), self._num(attrs.get("YSIZE"))
            ap = ApertureP.rect(hx, hy) if etype == "RCOLLIMATOR" else ApertureP(
                shape="ELLIPTICAL", x_limits=(-hx, hx), y_limits=(-hy, hy))
            el = Collimator(length=0.0 if thin else L, aperture=ap, **common)
        elif etype == "MONITOR":
            params = {}
            for item in (tag.get("params") or "").split(";"):
                if ":" in item:
                    k, v = item.split(":", 1)
                    params[k] = _param_value(v)
            el = Instrument(length=L, family=str(tag.get("family") or "MONITOR"), params=params, **common)
        elif etype == "MARKER":
            el = self._marker(kind, tag, common)
        else:
            native = {k: (self._str(v) if isinstance(v, str) else v) for k, v in attrs.items()}
            self.rep.lossy("UNSUPPORTED_OPAL_ELEMENT", f"OPAL element type {etype} has no IR kind; kept as a "
                           f"{'drift' if L > _THIN else 'marker'} (attributes in native['opal'])", element=name,
                           kind="Drift" if L > _THIN else "Marker", opal_type=etype)
            el = Drift(length=L, **common) if L > _THIN else Marker(**common)
            el.native["opal"] = {"type": etype, **native}
        if el is None:
            return None, s, 0.0
        if etype not in ("RCOLLIMATOR", "ECOLLIMATOR"):
            ap = self._aperture(attrs)
            if ap is not None:
                el.aperture = ap
        self._shift(el, attrs, bend=(etype in ("SBEND", "RBEND", "RBEND3D")))
        if tag.get("from"):
            el.meta["opal_from"] = tag["from"]
        if length is None:
            length = float(el.length)
        if thin and s is not None:
            s += 0.5 * L                       # the surrogate was centred on the thin element
        return el, s, length

    def _drift_kind(self, kind, L, tag, common) -> Element:
        if kind == "Solenoid":
            return Solenoid(length=L, solenoid=SolenoidP(Bsol_T=float(tag.get("Bsol", 0.0) or 0.0)), **common)
        if kind == "Taylor":
            return Taylor(length=L, **common)
        if kind == "RFQCell":
            return RFQCell(length=L, **common)
        if kind == "RFCavity":
            return RFCavity(length=L, rf=RFP(voltage_V=float(tag.get("V", 0.0) or 0.0),
                                             phase_rad=float(tag.get("phase", 0.0) or 0.0)), **common)
        if kind == "NCells":
            return NCells(length=L, **common)
        if kind == "Bend":
            return Bend(length=L, **common)
        if kind == "Collimator":
            return Collimator(length=L, **common)
        if kind == "Foil":
            return Foil(length=L, material=str(tag.get("material") or "C"),
                        thickness_kg_per_m2=float(tag.get("thick", 0.0) or 0.0), **common)
        return Drift(length=L, **common)

    def _marker(self, kind, tag, common) -> Element:
        if kind == "Patch":
            keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
            el = Patch(**{k: float(tag[k]) for k in keys if k in tag}, **common)
            if "dt" in tag:
                el.t_offset_s = float(tag["dt"])
            if "dE" in tag:
                el.e_tot_offset_eV = float(tag["dE"])
            return el
        if kind == "ReferenceChange":
            return ReferenceChange(dE_ref_eV=float(tag["dE"]) if "dE" in tag else None,
                                   energy_eV=float(tag["E"]) if "E" in tag else None,
                                   dphase_rad=float(tag["dphase"]) if "dphase" in tag else None,
                                   dtime_s=float(tag["dt"]) if "dt" in tag else None, **common)
        if kind == "Freq":
            return Freq(frequency_Hz=float(tag.get("f", 0.0) or 0.0), **common)
        if kind == "Foil":
            el = Foil(material=str(tag.get("material") or "C"),
                      thickness_kg_per_m2=float(tag.get("thick", 0.0) or 0.0), **common)
            if "dE" in tag:
                el.dE_ref_eV = float(tag["dE"])
            return el
        if kind == "Solenoid":
            return Solenoid(length=0.0, solenoid=SolenoidP(Bsol_T=float(tag.get("Bsol", 0.0) or 0.0)), **common)
        if kind == "Collimator":
            return Collimator(length=0.0, **common)
        if kind == "Multipole":
            return Multipole(**common)
        if kind == "Instrument":
            return Instrument(family=str(tag.get("family") or "MONITOR"), **common)
        return Marker(**common)

    def _map(self, attrs: dict):
        fname = self._str(attrs.get("FMAPFN"))
        if not fname or fname.upper().startswith("1DPROFILE"):
            return None
        p = Path(fname)
        if not p.is_absolute():
            p = self.path.parent / p
        try:
            return parse_map(p.read_text())
        except (OSError, ValueError) as e:
            self.warnings.append(f"field map {fname!r} not read ({e})")
            return None

    def _cavity(self, etype, attrs, tag, common, ref, L, thin, pad):
        data = self._map(attrs)
        freq = self._num(attrs.get("FREQ")) * 1e6 if attrs.get("FREQ") is not None else \
            (data.frequency_Hz if data is not None and data.frequency_Hz else None)
        phase = float(tag["phase"]) if "phase" in tag else self._num(attrs.get("LAG"))
        n_cell = int(float(tag["n"])) if "n" in tag else (int(self._num(attrs.get("NUMCELLS"))) or None
                                                          if attrs.get("NUMCELLS") is not None else None)
        tw = bool(tag.get("tw")) or etype == "TRAVELINGWAVE"
        kind = tag.get("kind")
        volt_MVpm = self._num(attrs.get("VOLT"))
        de = self._num(attrs.get("DESIGNENERGY")) * 1e6 if attrs.get("DESIGNENERGY") is not None else None
        if "V" in tag:
            V = float(tag["V"])
        elif de is not None and de > 0:
            V = de - ref.kinetic_energy_eV
            if phase > 0.5 * math.pi or phase < -0.5 * math.pi:
                V = -V
                phase = phase - math.pi if phase > 0 else phase + math.pi
        elif data is not None and freq and volt_MVpm:
            V = self._integrated_voltage(data, volt_MVpm * 1e6, freq, ref)
        else:
            V = 0.0
            self.rep.lossy("RFCAVITY_GAIN_UNKNOWN", "no DESIGNENERGY and no readable map: the cavity's voltage is "
                           "unknown (0 written)", element=common["name"], kind="RFCavity")
        if not tag:
            self.rep.equivalent("OPAL_LAG_AS_SYNC_PHASE", "LAG is read as the synchronous phase relative to the "
                                "crest (OPAL applies it after AUTOPHASE)", element=common["name"], kind="RFCavity")
        map_len = data.length_m if data is not None else L
        length = float(tag["L"]) if "L" in tag else (map_len - 2.0 * pad if data is not None else L)
        rf = RFP(voltage_V=V, phase_rad=phase, frequency_Hz=freq, n_cell=n_cell,
                 cavity_type="TRAVELING_WAVE" if tw else "STANDING_WAVE",
                 L_active_m=float(tag["active"]) if "active" in tag else None)
        if kind == "NCells":
            el = NCells(length=length, rf=rf, params={"n_cells": n_cell or 1}, **common)
            if "mode" in tag:
                el.params["mode"] = _param_value(str(tag["mode"]))
        else:
            el = RFCavity(length=length, rf=rf, **common)
        if data is not None:
            el.native["opal"] = {"kind": DYNAMIC, "z": list(data.z_m), "values": list(data.values),
                                 "map": self._str(attrs.get("FMAPFN")), "volt": volt_MVpm, "pad": pad,
                                 "freq": data.frequency_Hz, "file": tag.get("file")}
        if kind == "FieldMap":
            el.meta["opal_from"] = "FieldMap"
            self.rep.equivalent("FM_READ_AS_CAVITY", "a cavity written from a field map is read as an RF cavity "
                                "carrying the map profile", element=common["name"], kind="RFCavity")
        return el, length

    def _integrated_voltage(self, data, scale_V_per_m: float, freq: float, ref: ReferenceParticle) -> float:
        """The crest gain of the map at ``VOLT`` for the reference entering at its energy (the largest gain
        over the driven phase, integrated through the profile)."""
        from lattix.formats.impactt.rfprofile import fourier_coefficients, gain_from_profile

        peak = data.peak or 1.0
        mid = 0.5 * (data.z_m[0] + data.z_m[-1])
        z = [v - mid for v in data.z_m]                  # the profile integrator wants z about the centre
        coefs = fourier_coefficients(z, [v / peak for v in data.values], data.length_m, 60)
        _dE, v, _psi = gain_from_profile(coefs, data.length_m, scale_V_per_m, 0.0, freq, 0.0,
                                         ref.kinetic_energy_eV, ref.species.mass_eV, float(ref.species.charge))
        return float(v)

    def _aperture(self, attrs: dict) -> ApertureP | None:
        text = self._str(attrs.get("APERTURE"))
        if not text:
            return None
        m = re.match(r"^\s*(\w+)\s*\((.*)\)\s*$", text)
        if not m:
            return None
        shape, args = m.group(1).upper(), [self._num(x) for x in m.group(2).split(",") if x.strip()]
        if not args:
            return None
        w = 0.5 * args[0]
        h = 0.5 * args[1] if len(args) > 1 and shape not in ("CIRCLE", "SQUARE") else w
        if shape in ("RECTANGLE", "SQUARE"):
            return ApertureP.rect(w, h)
        if shape in ("CIRCLE", "ELLIPSE"):
            return ApertureP(shape="ELLIPTICAL", x_limits=(-w, w), y_limits=(-h, h))
        return None

    def _shift(self, el: Element, attrs: dict, *, bend: bool) -> None:
        vals = {k: self._num(attrs.get(a)) for k, a in (("x_offset", "DX"), ("y_offset", "DY"), ("z_offset", "DZ"),
                                                         ("y_rot", "DTHETA"), ("x_rot", "DPHI"), ("tilt", "DPSI"))
                if attrs.get(a) is not None}
        if any(vals.values()):
            el.shift = BodyShiftP(**vals)

    # ------------------------------------------------------------------ assembly
    def _fold(self, elements: list[Element]) -> list[Element]:
        """A ``Multipole`` written as a KICKER (dipole terms) followed by a MULTIPOLE (the others) is one
        element again."""
        out: list[Element] = []
        for el in elements:
            role = el.meta.pop("opal_role", None)
            if role == "higher" and out and out[-1].kind == "Multipole" and out[-1].meta.get("_opal_dipole") \
                    and out[-1].name == el.name:
                prev = out[-1]
                prev.multipole.BnL.update(el.multipole.BnL)
                prev.multipole.BsL.update(el.multipole.BsL)
                prev.multipole.tilt.update(el.multipole.tilt)
                prev.meta.pop("_opal_dipole", None)
                continue
            if role == "dipole":
                el.meta["_opal_dipole"] = True
            out.append(el)
        for el in out:
            el.meta.pop("_opal_dipole", None)
        return out

    @staticmethod
    def _assemble(use: str, elements: list[Element], ref) -> Lattice:
        lat = Lattice(name=use, reference=ref)
        line = Line(name=use)
        seen: dict[str, dict] = {}
        for el in elements:
            existing = lat.elements.get(el.name)
            if existing is not None and type(existing) is type(el):
                dump = el.model_dump(mode="json")
                dump.pop("provenance", None)
                if seen.get(el.name) == dump:
                    line.items.append(LineItem(ref=el.name))
                    continue
            nm = lat.add_element(el)
            dump = el.model_dump(mode="json")
            dump.pop("provenance", None)
            seen[nm] = dump
            line.items.append(LineItem(ref=nm))
        lat.lines[use] = line
        lat.use = use
        return lat


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
