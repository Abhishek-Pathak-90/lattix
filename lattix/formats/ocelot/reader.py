"""Ocelot lattice module → IR.

The file is parsed, never executed: an AST walk over the top-level assignments collects the element
constructors (literal arguments, arithmetic on literals and on earlier numeric assignments, ``np.pi``
and friends), the sequences (``cell = (a, b) + 2 * (c,)``) and the ``MagneticLattice(cell)`` call.  A
file that builds its lattice with loops or functions cannot be read this way; ``read(...,
use_ocelot=True)`` then runs it in the Ocelot environment (GPL, out of process, through
:mod:`lattix.oracles.ocelot_worker`) and reads the resulting sequence.

Ocelot keeps no beam: the reference particle comes from the ``# lattix: reference`` tag lattix writes,
from ``read(species=, kinetic_energy_eV=)``, or — an Ocelot file without either — from ``tws0.E``
(an electron of that total energy, EQUIVALENT ``REFERENCE_FROM_TWISS``).
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

from lattix.fidelity import FidelityReport
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
    Sextupole,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.lattice import Lattice, Line, LineItem
from lattix.ir.reference import ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.walk import propagate

_ELECTRON_MASS_EV = species_by_name("electron").mass_eV
_TAG = re.compile(r"#\s*lattix:\s*(?P<body>(?!reference\b)(?!lattice\b).*)$")
_HEAD = re.compile(r"#\s*lattix:\s*lattice\s+(?P<body>.*)$")
_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')
#: positional parameter names of Ocelot 25.06's constructors
_BEND = ("l", "angle", "k1", "k2", "e1", "e2", "tilt", "gap", "h_pole1", "h_pole2", "fint", "fintx")
_POSITIONAL: dict[str, tuple[str, ...]] = {
    "Drift": ("l",), "Quadrupole": ("l", "k1", "k2", "tilt"), "Sextupole": ("l", "k2", "tilt"),
    "Octupole": ("l", "k3", "tilt"), "Multipole": ("kn",), "SBend": _BEND, "Bend": _BEND, "RBend": _BEND,
    "Solenoid": ("l", "k"), "Cavity": ("l", "v", "phi", "freq"), "TWCavity": ("l", "v", "phi", "freq"),
    "TDCavity": ("l", "freq", "phi", "v", "tilt"), "Hcor": ("l", "angle"), "Vcor": ("l", "angle"),
    "Marker": (), "Monitor": ("l",), "Aperture": ("xmax", "ymax", "dx", "dy", "type"),
    "Matrix": ("l", "delta_e"), "Undulator": ("lperiod", "nperiods", "Kx", "Ky"),
    "XYQuadrupole": ("l", "x_offs", "y_offs", "k1"), "Pulse": (), "UnknownElement": ("l",),
}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "log": math.log,
          "abs": abs, "float": float, "int": int, "radians": math.radians, "degrees": math.degrees,
          "arctan": math.atan, "atan": math.atan, "arcsin": math.asin, "arccos": math.acos, "round": round}
_CONSTS = {"pi": math.pi, "inf": math.inf, "e": math.e, "Inf": math.inf}


class _Unsupported(ValueError):
    """The file cannot be read without running it."""


class _Eval:
    def __init__(self, symbols: dict[str, float]) -> None:
        self.sym = symbols

    def value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            v = self.value(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise _Unsupported(f"unary {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            a, b = self.value(node.left), self.value(node.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b,
                   ast.Pow: lambda: a ** b, ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b}
            fn = ops.get(type(node.op))
            if fn is None:
                raise _Unsupported(f"operator {type(node.op).__name__}")
            return fn()
        if isinstance(node, ast.Name):
            if node.id in self.sym:
                return self.sym[node.id]
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise _Unsupported(f"name {node.id!r} is not a number defined earlier")
        if isinstance(node, ast.Attribute):
            if node.attr in _CONSTS:
                return _CONSTS[node.attr]
            raise _Unsupported(f"attribute {ast.unparse(node)}")
        if isinstance(node, ast.List | ast.Tuple):
            return [self.value(x) for x in node.elts]
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            fn = _FUNCS.get(name)
            if fn is None or node.keywords:
                raise _Unsupported(f"call {ast.unparse(node)}")
            return fn(*[self.value(a) for a in node.args])
        raise _Unsupported(f"expression {ast.unparse(node)}")


def _num(v, default: float = 0.0) -> float:
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v if v is not None else default)


def _parse_tag(body: str) -> dict[str, str]:
    return {k: (q if v.startswith('"') else v) for k, v, q in _KV.findall(body)}


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)`` for an Ocelot lattice module."""

    format = "ocelot"

    def read(self, path: Path, *, strict: bool = False, species: str | Species | None = None,
             kinetic_energy_eV: float | None = None, frequency_Hz: float | None = None, root: str | None = None,
             use_ocelot: bool = False) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="ocelot", source_file=str(path))
        text = path.read_text(encoding="utf-8")
        self._rep = rep
        self._named: set[int] = set()
        tags: dict[int, dict[str, str]] = {}
        head: dict[str, str] = {}
        for i, line in enumerate(text.splitlines(), 1):
            m = _HEAD.search(line)
            if m:
                head = _parse_tag(m.group("body"))
                continue
            m = _TAG.search(line)
            if m and "#" in line:
                tags[i] = _parse_tag(m.group("body"))
        try:
            defs, seq, tws_E = self._parse(text, tags)
        except _Unsupported as exc:
            if not use_ocelot:
                raise ValueError(f"{path.name}: {exc}; a lattice built with code needs read(..., use_ocelot=True) "
                                 "(runs the file in the Ocelot environment)") from None
            defs, seq, tws_E = self._dump(path)
            rep.equivalent("OCELOT_EXECUTED", "the lattice module was run in the Ocelot environment to obtain "
                           "its sequence (an AST read was not possible)")
        ref = self._reference(text, species, kinetic_energy_eV, frequency_Hz, tws_E, rep, path)
        self._norm: dict[int, tuple] = {}
        elements: dict[str, Element] = {}
        for var, (ctor, kw, tag) in defs.items():
            el = self._convert(var, ctor, kw, tag)
            if el is not None:
                elements[var] = el
        order = self._fold([elements[v] for v in seq if v in elements])
        used: set[str] = set()
        for el in order:                      # names are made unique after the folding (apertures, body
            if id(el) in self._named:         # drifts and split correctors carry their element's name)
                continue
            self._named.add(id(el))
            base = el.name
            if base in used:
                k = 2
                while f"{base}_{k}" in used:
                    k += 1
                el.name = f"{base}_{k}"
            used.add(el.name)
        name = head.get("name") or path.stem
        use = root or head.get("use") or name
        lat = Lattice(name=name, reference=ref)
        lat.meta["source_format"] = "ocelot"
        lat.meta["source_file"] = str(path)
        placed: set[int] = set()
        items = []
        for el in order:
            if id(el) not in placed:
                lat.elements[el.name] = el
                placed.add(id(el))
            items.append(LineItem(ref=el.name))
        lat.lines[use] = Line(name=use, items=items)
        lat.use = use
        self._second_pass(lat)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ the AST walk
    def _parse(self, text: str, tags: dict[int, dict[str, str]]):
        tree = ast.parse(text)
        symbols: dict[str, float] = {}
        ev = _Eval(symbols)
        defs: dict[str, tuple[str, dict, dict[str, str]]] = {}
        seqs: dict[str, list[str]] = {}
        twiss: set[str] = set()
        tws_E: float | None = None
        root_seq: list[str] | None = None
        pending: dict[str, str] = {}

        def seq_of(node) -> list[str]:
            if isinstance(node, ast.Name):
                if node.id in defs:
                    return [node.id]
                if node.id in seqs:
                    return list(seqs[node.id])
                raise _Unsupported(pending.get(node.id, f"{node.id!r} is neither an element nor a sequence"))
            if isinstance(node, ast.Tuple | ast.List):
                out: list[str] = []
                for x in node.elts:
                    out += seq_of(x)
                return out
            if isinstance(node, ast.Starred):
                return seq_of(node.value)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return seq_of(node.left) + seq_of(node.right)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                try:
                    n, s = int(ev.value(node.left)), seq_of(node.right)
                except _Unsupported:
                    n, s = int(ev.value(node.right)), seq_of(node.left)
                return s * n
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in (
                    "list", "tuple", "MagneticLattice"):
                return seq_of(node.args[0]) if node.args else []
            raise _Unsupported(f"sequence expression {ast.unparse(node)}")

        skipped: list[str] = []
        for st in tree.body:
            if isinstance(st, ast.Import | ast.ImportFrom | ast.Expr | ast.Pass):
                continue
            if not isinstance(st, ast.Assign) or len(st.targets) != 1:
                skipped.append(f"{type(st).__name__} at line {st.lineno}")
                continue
            target, value = st.targets[0], st.value
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                obj = target.value.id
                try:
                    v = ev.value(value)
                except _Unsupported:
                    continue
                if obj in twiss and target.attr == "E":
                    tws_E = float(v)
                elif obj in defs and isinstance(v, int | float):
                    defs[obj][1][target.attr] = float(v)
                continue
            if not isinstance(target, ast.Name):
                continue
            var = target.id
            if isinstance(value, ast.Call):
                fname = value.func.attr if isinstance(value.func, ast.Attribute) else getattr(value.func, "id", "")
                if fname == "MagneticLattice":
                    root_seq = seq_of(value.args[0]) if value.args else []
                    continue
                if fname == "Twiss":
                    twiss.add(var)
                    continue
                if fname in _POSITIONAL:
                    kw: dict = {}
                    names = _POSITIONAL[fname]
                    for i, a in enumerate(value.args):
                        if i < len(names):
                            kw[names[i]] = ev.value(a)
                    for k in value.keywords:
                        if k.arg is not None and k.arg != "tm":
                            kw[k.arg] = ev.value(k.value)
                    tag = tags.get(st.end_lineno or st.lineno) or tags.get(st.lineno) or {}
                    defs[var] = (fname, kw, tag)
                    continue
            try:
                v = ev.value(value)
                if isinstance(v, int | float) and not isinstance(v, bool):
                    symbols[var] = float(v)
                    continue
            except _Unsupported:
                pass
            try:
                seqs[var] = seq_of(value)
            except _Unsupported as exc:
                pending[var] = f"{var!r} at line {st.lineno}: {exc}"
        if root_seq is None:
            if "cell" in seqs:
                root_seq = seqs["cell"]
            elif seqs:
                root_seq = list(seqs.values())[-1]
            elif defs and not skipped:
                root_seq = list(defs)
            else:
                root_seq = []
        if not root_seq and skipped:
            raise _Unsupported("the lattice is built by code the AST reader does not run (" + ", ".join(skipped[:3])
                               + ")")
        if not root_seq:
            raise _Unsupported("no element definitions and no MagneticLattice(...) call")
        return defs, root_seq, tws_E

    def _dump(self, path: Path):
        from lattix.oracles.ocelot import OcelotOracle

        o = OcelotOracle()
        ok, why = o.available()
        if not ok:
            raise RuntimeError(f"use_ocelot=True but Ocelot is not available: {why}")
        data = json.loads(Path(o.dump(path)).read_text())
        defs: dict[str, tuple[str, dict, dict[str, str]]] = {}
        seq: list[str] = []
        for i, entry in enumerate(data["elements"]):
            var = f"e{i}"
            kw = dict(entry.get("params") or {})
            kw["eid"] = entry.get("eid") or var
            defs[var] = (str(entry["type"]), kw, {})
            seq.append(var)
        return defs, seq, data.get("energy_GeV")

    # ------------------------------------------------------------------ the reference
    def _reference(self, text, species, kinetic_energy_eV, frequency_Hz, tws_E, rep, path) -> ReferenceParticle:
        tag = parse_reference_tag(text)
        sp = None
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        elif tag is not None:
            sp = tag.species
        ke = kinetic_energy_eV if kinetic_energy_eV is not None else (tag.kinetic_energy_eV if tag else None)
        f = frequency_Hz if frequency_Hz is not None else (tag.rf_frequency_Hz if tag else None)
        if tag is not None and (species is None or kinetic_energy_eV is None):
            rep.equivalent("REFERENCE_FROM_TAG", "reference particle taken from the lattix tag")
        if (sp is None or ke is None) and tws_E is not None:
            sp = sp or species_by_name("electron")
            if ke is None:
                ke = float(tws_E) * 1e9 - sp.mass_eV
            rep.equivalent("REFERENCE_FROM_TWISS", f"Ocelot lattices carry no beam: an {sp.name} of "
                           f"tws0.E = {tws_E:.9g} GeV total energy")
        if sp is None or ke is None:
            raise ValueError(f"{path.name}: an Ocelot lattice carries no beam; pass species=<name> and "
                             "kinetic_energy_eV=<eV> to read(...) (or keep lattix's reference tag / tws0.E)")
        return ReferenceParticle(species=sp, kinetic_energy_eV=float(ke), rf_frequency_Hz=f)

    # ------------------------------------------------------------------ elements
    def _convert(self, var: str, ctor: str, kw: dict, tag: dict[str, str]) -> Element | None:
        name = tag.get("name") or str(kw.get("eid") or var)
        kind = tag.get("kind")
        prov = Provenance(format="ocelot", original_name=str(kw.get("eid") or var), original_type=ctor)
        common = {"name": name, "provenance": prov}
        L = _num(kw.get("l"))
        shift = None
        if kw.get("dx") or kw.get("dy"):
            if ctor != "Aperture":
                shift = BodyShiftP(x_offset=_num(kw.get("dx")), y_offset=_num(kw.get("dy")))
        el: Element | None = None
        if ctor == "Drift":
            el = self._drift_kind(kind, L, tag, common)
        elif ctor == "Quadrupole":
            el = Quadrupole(length=L, shift=shift, **common)
            el.multipole = MagneticMultipoleP(tilt={1: _num(kw.get("tilt"))} if kw.get("tilt") else {})
            self._norm[id(el)] = ("Bn", {1: _num(kw.get("k1")), 2: _num(kw.get("k2"))})
        elif ctor == "Sextupole":
            el = Sextupole(length=L, shift=shift, **common)
            el.multipole = MagneticMultipoleP(tilt={2: _num(kw.get("tilt"))} if kw.get("tilt") else {})
            self._norm[id(el)] = ("Bn", {2: _num(kw.get("k2"))})
        elif ctor == "Octupole":
            el = Octupole(length=L, shift=shift, **common)
            el.multipole = MagneticMultipoleP(tilt={3: _num(kw.get("tilt"))} if kw.get("tilt") else {})
            self._norm[id(el)] = ("Bn", {3: _num(kw.get("k3"))})
        elif ctor == "Multipole":
            kn = kw.get("kn", 0.0)
            kn = [float(x) for x in kn] if isinstance(kn, list) else [float(kn)]
            el = Multipole(length=0.0, shift=shift, **common)
            el.multipole = MagneticMultipoleP()
            self._norm[id(el)] = ("BnL", {n: v for n, v in enumerate(kn) if v})
        elif ctor in ("SBend", "Bend", "RBend"):
            angle = _num(kw.get("angle"))
            e1, e2 = kw.get("e1"), kw.get("e2")
            if ctor == "RBend":
                e1 = angle / 2.0 + (_num(e1) if e1 is not None else 0.0)
                e2 = angle / 2.0 + (_num(e2) if e2 is not None else 0.0)
            fint = _num(kw.get("fint"))
            fintx = kw.get("fintx")
            el = Bend(length=L, shift=shift, **common)
            el.bend = BendP(angle=angle, e1=_num(e1), e2=_num(e2), hgap=_num(kw.get("gap")) / 2.0, edge_int1=fint,
                            edge_int2=(float(fintx) if fintx is not None and abs(float(fintx) - fint) > 1e-15
                                       else None),
                            tilt_ref=_num(kw.get("tilt")), rect=(ctor == "RBend" or tag.get("rect") == "1"))
            if kw.get("k1") or kw.get("k2"):
                self._norm[id(el)] = ("Bn", {1: _num(kw.get("k1")), 2: _num(kw.get("k2"))})
        elif ctor == "Solenoid":
            el = Solenoid(length=L, shift=shift, **common)
            self._norm[id(el)] = ("Bsol", _num(kw.get("k")))
        elif ctor in ("Cavity", "TWCavity"):
            rf = RFP(frequency_Hz=_num(kw.get("freq")) or None, voltage_V=_num(kw.get("v")) * 1e9,
                     phase_rad=-math.radians(_num(kw.get("phi"))),
                     cavity_type="TRAVELING_WAVE" if (ctor == "TWCavity" or tag.get("tw") == "1")
                     else "STANDING_WAVE", n_cell=(int(tag["n"]) if tag.get("n") else None))
            if kind == "NCells":
                el = NCells(length=L, rf=rf, **common)
                self._rep.equivalent("NCELLS_FROM_CAVITY", "an NCells written as an Ocelot Cavity comes back "
                                     "with the cavity's voltage, phase and frequency only", element=name,
                                     kind="NCells")
            elif tag.get("L") == "0":
                el = RFCavity(length=0.0, rf=rf, shift=shift, **common)
                el.meta["ocelot_thin"] = {"length": L, "pad": _num(tag.get("pad"))}
            else:
                el = RFCavity(length=L, rf=rf, shift=shift, **common)
        elif ctor in ("Hcor", "Vcor"):
            a = _num(kw.get("angle"))
            el = Kicker(length=L, hkick=a if ctor == "Hcor" else 0.0, vkick=a if ctor == "Vcor" else 0.0,
                        shift=shift, **common)
            if tag.get("role") == "vkick":
                el.meta["ocelot_role"] = "vkick"
        elif ctor == "Marker":
            el = self._marker(kind, tag, common)
        elif ctor == "Monitor":
            params = {}
            for item in (tag.get("params") or "").split(";"):
                if ":" in item:
                    k, v = item.split(":", 1)
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v
            el = Instrument(length=L, family=tag.get("family", "MONITOR"), params=params, **common)
        elif ctor == "Aperture":
            xm, ym = abs(_num(kw.get("xmax"), math.inf)), abs(_num(kw.get("ymax"), math.inf))
            dx, dy = _num(kw.get("dx")), _num(kw.get("dy"))
            ap = ApertureP(shape="ELLIPTICAL" if str(kw.get("type", "rect")).startswith("ellip") else "RECTANGULAR",
                           x_limits=(dx - xm, dx + xm) if math.isfinite(xm) else None,
                           y_limits=(dy - ym, dy + ym) if math.isfinite(ym) else None)
            el = Collimator(length=0.0, aperture=ap, **common)
            if tag.get("role"):
                el.meta["ocelot_role"] = tag["role"]
        elif ctor == "Matrix":
            if kind == "ReferenceChange":
                dE = float(tag["dE"]) if tag.get("dE") else _num(kw.get("delta_e")) * 1e9
                el = ReferenceChange(dE_ref_eV=dE, energy_eV=(float(tag["E"]) if tag.get("E") else None), **common)
            else:
                m = [[0.0] * 6 for _ in range(6)]
                off = [0.0] * 6
                for k, v in kw.items():
                    if re.fullmatch(r"[rR][1-6][1-6]", k):
                        m[int(k[1]) - 1][int(k[2]) - 1] = float(v)
                    elif re.fullmatch(r"[bB][1-6]", k):
                        off[int(k[1]) - 1] = float(v)
                basis = tag.get("basis", "ocelot")
                el = Taylor(length=L, matrix=m, offset=off, basis=basis, **common)
                if basis == "ocelot":
                    self._norm[id(el)] = ("taylor", None)
                if kw.get("delta_e"):
                    self._rep.lossy("OCELOT_MATRIX_DELTA_E_DROPPED", "a Matrix with delta_e changes the energy in "
                                    "Ocelot; the IR Taylor keeps only the map", element=name, kind="Taylor")
        else:
            length = L if ctor != "Undulator" else _num(kw.get("lperiod")) * _num(kw.get("nperiods"))
            self._rep.dropped("UNSUPPORTED_OCELOT_ELEMENT", f"Ocelot {ctor} has no IR kind; a drift/marker of "
                              "the same length keeps the survey (parameters in native['ocelot'])",
                              element=name, kind=ctor)
            el = Drift(length=length, **common) if length else Marker(**common)
            el.native["ocelot"] = {"type": ctor, **{k: v for k, v in kw.items() if k != "eid"}}
        if el is not None and tag.get("from"):
            el.meta["ocelot_from"] = tag["from"]
        return el

    def _drift_kind(self, kind, L, tag, common) -> Element:
        if kind == "Instrument":
            return Instrument(length=L, family=tag.get("family", "MONITOR"), **common)
        if kind == "Patch":
            return Patch(length=L, **{k: float(tag.get(k, 0.0)) for k in ("x_offset", "y_offset", "z_offset",
                                                                          "x_rot", "y_rot", "tilt")}, **common)
        if kind == "Collimator" and tag.get("role") != "body":
            return Collimator(length=L, **common)             # a collimator without limits
        d = Drift(length=L, **common)
        if tag.get("role") == "body":
            d.meta["ocelot_role"] = "body"
        return d

    def _marker(self, kind, tag: dict[str, str], common: dict) -> Element:
        if kind == "Foil":
            return Foil(material=tag.get("material", "C"), thickness_kg_per_m2=float(tag.get("thick", 0.0)),
                        dE_ref_eV=(float(tag["dE"]) if tag.get("dE") else None), **common)
        if kind == "Patch":
            keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
            return Patch(**{k: float(tag.get(k, 0.0)) for k in keys}, **common)
        if kind == "Freq":
            return Freq(frequency_Hz=float(tag.get("f", 0.0)), **common)
        if kind == "Directive":
            return Directive(format=tag.get("format", "tracewin"), card=tag.get("card", ""),
                             args=tag.get("args", "").split(), role=tag.get("role", "other"), **common)
        if kind == "Instrument":
            return Instrument(length=0.0, family=tag.get("family", "MONITOR"), **common)
        if kind == "Collimator":
            return Collimator(length=0.0, **common)
        return Marker(**common)

    # ------------------------------------------------------------------ folding
    def _fold(self, order: list[Element]) -> list[Element]:
        """Apertures, body drifts, split correctors and surrogate cavities go back onto their element."""
        out: list[Element] = []
        pending: list[Element] = []
        for el in order:
            role = el.meta.get("ocelot_role")
            if role == "aperture_in" and el.kind == "Collimator":
                pending.append(el)
                continue
            if role == "aperture_out" and el.kind == "Collimator" and out and el.aperture is not None:
                self._attach(out[-1], el.aperture, "EXIT")
                continue
            if role == "body" and el.kind == "Drift" and out and out[-1].kind in ("Collimator", "Foil", "Multipole") \
                    and out[-1].length == 0.0:
                out[-1].length = el.length
                continue
            if role == "vkick" and el.kind == "Kicker" and out and out[-1].kind == "Kicker" and out[-1].name == el.name:
                out[-1].vkick = el.vkick
                continue
            for ap in pending:
                if ap.aperture is not None:
                    self._attach(el, ap.aperture, "ENTRANCE")
            pending = []
            out.append(el)
        out += pending
        # a thin gap's surrogate cavity: the length goes back to the drifts around it (a drift the
        # writer could not shorten is put beside the gap so the file's survey is kept)
        final: list[Element] = []
        for i, el in enumerate(out):
            thin = el.meta.pop("ocelot_thin", None)
            if not thin:
                final.append(el)
                continue
            pad, rest = float(thin["pad"]), round(float(thin["length"]) - float(thin["pad"]), 12)
            before = next((j for j in range(i - 1, -1, -1) if out[j].length > 0), None)
            after = next((j for j in range(i + 1, len(out)) if out[j].length > 0), None)
            if pad and before is not None and out[before].kind == "Drift":
                out[before].length = round(out[before].length + pad, 12)
                pad = 0.0
            if rest and after is not None and out[after].kind == "Drift":
                out[after].length = round(out[after].length + rest, 12)
                rest = 0.0
            if pad:
                final.append(Drift(name=f"{el.name}_pad", length=pad))
            final.append(el)
            if rest:
                final.append(Drift(name=f"{el.name}_pad", length=rest))
            if pad or rest:
                self._rep.equivalent("THIN_GAP_PAD_DRIFT", f"the surrogate cavity's {pad + rest:.6g} m had no "
                                     "neighbouring drift to return to: a drift of that length stands beside "
                                     "the gap", element=el.name, kind=el.kind)
        for el in final:
            el.meta.pop("ocelot_role", None)
        return final

    @staticmethod
    def _attach(el: Element, ap: ApertureP, where: str) -> None:
        if el.aperture is None:
            el.aperture = ap.model_copy(update={"aperture_at": where})
        elif el.aperture.aperture_at != where:
            el.aperture.aperture_at = "BOTH_ENDS"

    # ------------------------------------------------------------------ second pass
    def _second_pass(self, lat: Lattice) -> None:
        """Normalized strengths become fields with the signed rigidity at each element's entrance."""
        done: set[int] = set()
        for p in propagate(lat):
            el = p.element
            if id(el) in done or id(el) not in self._norm:
                continue
            done.add(id(el))
            ref_in = p.ref_in or lat.reference
            brho = ref_in.brho_signed
            what, a = self._norm[id(el)]
            if what == "Bn":
                for n, k in a.items():
                    if k:
                        el.multipole.Bn[int(n)] = float(k) * brho
            elif what == "BnL":
                for n, k in a.items():
                    el.multipole.BnL[int(n)] = float(k) * brho
            elif what == "Bsol":
                el.solenoid = SolenoidP(Bsol_T=2.0 * float(a) * brho)
            elif what == "taylor":
                import numpy as np

                from lattix.formats.cheetah.writer import similarity_diag
                from lattix.oracles.base import Basis
                from lattix.oracles.basis import transform_matrix

                d = np.diag(transform_matrix(Basis.OCELOT, ref_in.kinetic_energy_eV, ref_in.species.mass_eV))
                el.matrix, el.offset = similarity_diag(el.matrix, el.offset, d, np.ones(6))
                el.basis = "common"


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
