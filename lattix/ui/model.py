"""View models for the browser page: a lattice as placed rows + deduplicated definitions + its ledger, the
format catalogue with reader/writer option schemas, the fidelity-code catalogue, the sample decks."""
from __future__ import annotations

import functools
import inspect as _inspect
import json
import math
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from lattix.crossval import (
    _SIDE_FILE_FORMATS,
    DECKS,
    ENGINE_CANDIDATES,
    ENGINE_FOR_FORMAT,
    PUBLIC,
    contrib,
    readable_formats,
)
from lattix.fidelity import FidelityReport
from lattix.formats.base import FORMATS
from lattix.ir.elements import ALL_KINDS, element_to_dict
from lattix.ir.lattice import Lattice, Placed, survey
from lattix.ir.normalize import k1_from_gradient
from lattix.ir.reference import SPECIES, ReferenceParticle
from lattix.ir.walk import energy_gain_eV, propagate
from lattix.ui.inspect import derived_numbers, locate_statements, reference_state

# --------------------------------------------------------------------------------------------- utilities


def jsonable(obj: Any) -> Any:
    """Plain JSON data: numpy → python, NaN/inf → null, tuples/sets → lists, paths/enums → strings."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [jsonable(x) for x in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in vars(obj).items()}
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump(mode="json"))
    return str(obj)


SHAPES: dict[str, str] = {
    "Drift": "drift", "Quadrupole": "quad", "Sextupole": "sext", "Octupole": "oct", "Multipole": "mult",
    "Bend": "bend", "Solenoid": "sol", "RFCavity": "rf", "FieldMap": "fmap", "NCells": "ncells", "RFQCell": "rfq",
    "Kicker": "kick", "Collimator": "coll", "Marker": "marker", "Instrument": "instr", "Foil": "foil",
    "Taylor": "taylor", "Patch": "patch", "ReferenceChange": "refchange", "Freq": "freq", "Directive": "directive",
    "Superposition": "super",
}


def reference_view(ref: ReferenceParticle) -> dict:
    return {"species": {"name": ref.species.name, "mass_eV": ref.species.mass_eV, "charge": ref.species.charge},
            "kinetic_energy_eV": ref.kinetic_energy_eV, "beta": ref.beta, "gamma": ref.gamma, "pc_eV": ref.pc_eV,
            "brho_abs": ref.brho_abs, "brho_signed": ref.brho_signed, "rf_frequency_Hz": ref.rf_frequency_Hz,
            "time_s": ref.time_s}


def _sign(v: float) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def glyph_for(p: Placed, ref: ReferenceParticle) -> dict:
    """What the synoptic needs to draw this placed element (kind shape, sign, plane, thinness, strength)."""
    e = p.element
    k = e.kind
    g: dict = {"shape": SHAPES.get(k, "marker"), "sign": 0, "plane": None, "thin": e.length <= 1e-12,
               "k1": None, "angle": None, "order": None}
    flip = -1.0 if p.reversed else 1.0
    if k == "Quadrupole":
        k1 = k1_from_gradient(float(e.multipole.Bn.get(1, 0.0)), ref) if ref.brho_signed else 0.0
        g["k1"] = k1
        g["sign"] = _sign(k1)                         # +1 focuses in x for the actual species
        if e.multipole.tilt.get(1) or e.multipole.Bs.get(1):
            g["skew"] = True
        if any(v for n, v in e.multipole.Bn.items() if n != 1):
            g["order"] = max(n for n, v in e.multipole.Bn.items() if v)
    elif k in ("Sextupole", "Octupole"):
        n = 2 if k == "Sextupole" else 3
        g["sign"] = _sign(float(e.multipole.Bn.get(n, 0.0)) * (ref.species.charge or 1))
    elif k == "Multipole":
        orders = [n for n, v in {**e.multipole.BnL, **e.multipole.BsL}.items() if v]
        g["order"] = max(orders) if orders else 0
        if e.multipole.BsL:
            g["skew"] = any(e.multipole.BsL.values())
        top = max(orders) if orders else 0
        g["sign"] = _sign(float(e.multipole.BnL.get(top, 0.0)))
    elif k == "Bend":
        a = float(e.bend.angle) * flip
        g["angle"] = a
        g["sign"] = _sign(a)
        t = float(e.bend.tilt_ref)
        g["plane"] = "v" if abs(abs(t) - math.pi / 2) < 1e-6 else "h"
        g["rect"] = bool(e.bend.rect)
        if e.multipole.Bn.get(1):
            g["gradient"] = True
    elif k == "Solenoid":
        g["sign"] = _sign(float(e.solenoid.Bsol_T))
    elif k in ("RFCavity", "FieldMap", "NCells", "RFQCell", "Superposition"):
        g["sign"] = _sign(energy_gain_eV(e, ref))
        rf = getattr(e, "rf", None)
        if rf is not None:
            g["volt"] = float(rf.voltage_V or 0.0)
            g["tw"] = rf.cavity_type == "TRAVELING_WAVE"
            g["sync"] = bool(rf.phase_is_sync)
            if rf.n_cell:
                g["n_cell"] = int(rf.n_cell)
        if k == "FieldMap":
            s = (e.meta or {}).get("map_summary") or {}
            g["map_kind"] = s.get("kind")
            g["integrated"] = bool(s)
    elif k == "Kicker":
        g["hx"], g["vy"] = _sign(float(e.hkick) * flip), _sign(float(e.vkick))
        g["electric"] = bool(e.electric)
    elif k == "Instrument":
        g["family"] = e.family
    elif k == "Taylor":
        g["rf_focusing"] = bool((e.meta or {}).get("rf_focusing_of"))
    return g


def maxima(rows: list[dict]) -> dict:
    """Scale references for the glyph heights (with floors so a lone element still gets a full bar)."""
    def top(key: str, floor: float) -> float:
        vals = [abs(r["glyph"][key]) for r in rows if r["glyph"].get(key)]
        return max(vals) if vals else floor
    k1 = top("k1", 1.0)
    volt = top("volt", 1.0)
    bsol = max([abs(r["contrib"].get("BsolL", 0.0)) / max(r["L"], 1e-9) for r in rows if r["contrib"].get("BsolL")]
               or [1.0])
    k2 = max([abs(r["contrib"].get("BnL2", 0.0)) for r in rows if r["kind"] == "Sextupole"] or [1.0])
    k3 = max([abs(r["contrib"].get("BnL3", 0.0)) for r in rows if r["kind"] == "Octupole"] or [1.0])
    kick = max([abs(r["contrib"].get("hkick", 0.0)) + abs(r["contrib"].get("vkick", 0.0)) for r in rows
                if r["kind"] == "Kicker"] or [1e-3])
    return {"k1": k1, "k2": k2, "k3": k3, "voltage_V": volt, "Bsol_T": bsol, "kick": kick}


def ledger_view(report: FidelityReport | None) -> dict:
    if report is None:
        return {"source_format": None, "target_format": None, "counts": {}, "codes": {}, "ok": True,
                "n_problems": 0, "entries": [], "by_element": {}, "lattice_level": []}
    entries = []
    by: dict[str, list[int]] = {}
    lattice_level: list[int] = []
    for i, e in enumerate(report.entries):
        entries.append({"i": i, "element": e.element, "kind": e.kind, "cls": e.cls.value, "code": e.code,
                        "message": e.message, "details": jsonable(e.details), "line": e.line})
        if e.element:
            by.setdefault(e.element, []).append(i)
        else:
            lattice_level.append(i)
    return {"source_format": report.source_format, "target_format": report.target_format,
            "counts": dict(report.counts), "codes": dict(report.codes()), "ok": report.ok,
            "n_problems": len(report.problems()), "entries": entries, "by_element": by,
            "lattice_level": lattice_level}


def deck_lines_for(text: str, fmt: str, lat: Lattice, placed: list[Placed]) -> dict[str, list[dict]]:
    """``{IR name: [{"line", "text", "role"}]}`` for every definition of ``lat`` in the written ``text``."""
    lines = text.splitlines()
    out: dict[str, list[dict]] = {}
    for name, e in lat.elements.items():
        prov = e.provenance
        out[name] = locate_statements(lines, fmt, name, prov.original_name if prov else None,
                                      prov.line if prov else None)
    return out


def lattice_view(lat: Lattice, report: FidelityReport | None, *, fmt: str, path: Path | str,
                 placed: list[Placed] | None = None, walk_warnings: list[str] | None = None,
                 deck_text: str | None = None) -> dict:
    """The page's model of one lattice: header, placed rows (s, energies, glyph hints, contributions,
    derived numbers, survey), deduplicated definitions, ledger, bounding box."""
    warnings = list(walk_warnings) if walk_warnings is not None else []
    if placed is None:
        placed = propagate(lat, warnings=warnings)
    ref0 = lat.reference
    ledger = ledger_view(report)
    by_element = ledger["by_element"]
    rows: list[dict] = []
    counts: dict[str, int] = {}
    n_placed: dict[str, int] = {}
    sv = survey(placed) if placed else np.zeros((0, 4))
    prev = [0.0, 0.0, 0.0, 0.0]
    energy_profile: list[list[float]] = [[0.0, ref0.kinetic_energy_eV]]
    for p in placed:
        e = p.element
        ref_in = p.ref_in or ref0
        ref_out = p.ref_out or ref_in
        counts[e.kind] = counts.get(e.kind, 0) + 1
        n_placed[e.name] = n_placed.get(e.name, 0) + 1
        c = {q: v for q, v in contrib(p).items() if v}
        here = [float(x) for x in sv[p.index]] if p.index < len(sv) else prev
        rows.append({
            "i": p.index, "name": e.name, "def": e.name, "kind": e.kind, "s_in": p.s_in, "s_out": p.s_out,
            "L": p.length, "rev": bool(p.reversed), "path": list(p.path),
            "ke_in": ref_in.kinetic_energy_eV, "ke_out": ref_out.kinetic_energy_eV, "beta_in": ref_in.beta,
            "brho_in": ref_in.brho_signed, "glyph": glyph_for(p, ref_in), "contrib": c,
            "derived": jsonable(derived_numbers(p, lat)),
            "survey": {"in": prev, "out": here}, "ledger": list(by_element.get(e.name, [])),
        })
        prev = here
        if ref_out.kinetic_energy_eV != energy_profile[-1][1]:
            energy_profile.append([p.s_in, ref_in.kinetic_energy_eV])
            energy_profile.append([p.s_out, ref_out.kinetic_energy_eV])
    total = placed[-1].s_out if placed else 0.0
    energy_profile.append([total, (placed[-1].ref_out or ref0).kinetic_energy_eV if placed else ref0.kinetic_energy_eV])
    deck_lines = deck_lines_for(deck_text, fmt, lat, placed) if deck_text is not None else {}
    definitions: dict[str, dict] = {}
    for name, e in lat.elements.items():
        d = jsonable(element_to_dict(e))
        d["n_placed"] = n_placed.get(name, 0)
        if deck_text is not None:
            d["deck_lines"] = deck_lines.get(name, [])
        definitions[name] = d
    xs = [r["survey"]["out"][0] for r in rows] + [0.0]
    ys = [r["survey"]["out"][1] for r in rows] + [0.0]
    zs = [r["survey"]["out"][2] for r in rows] + [0.0]
    ref_out = (placed[-1].ref_out or ref0) if placed else ref0
    header = {
        "name": lat.name, "use": lat.use, "lines": list(lat.lines), "format": fmt, "file": str(path),
        "n_placed": len(rows), "n_definitions": len(definitions), "total_length": total, "counts": counts,
        "reference": reference_view(ref0), "reference_out": reference_view(ref_out),
        "reference_state": {"in": reference_state(ref0), "out": reference_state(ref_out)},
        "beam_assumed": any(e.code == "BEAM_ASSUMED" for e in (report.entries if report is not None else [])),
        "warnings": list(lat.warnings) + warnings, "energy_profile": energy_profile, "maxima": maxima(rows),
        "variables": {k: jsonable(v.value) for k, v in lat.variables.items()},
    }
    return jsonable({"header": header, "rows": rows, "definitions": definitions, "ledger": ledger,
                     "bbox": {"s": [0.0, total], "x": [min(xs), max(xs)], "y": [min(ys), max(ys)],
                              "z": [min(zs), max(zs)]}})


# --------------------------------------------------------------------------------------------- options

#: fixed choice lists the writers validate against (their ValueError sites)
CHOICES: dict[tuple[str, str, str], list[str]] = {
    ("madx", "write", "mode"): ["sequence", "line"],
    ("madx", "write", "energy_mode"): ["delta", "local", "constant"],
    ("mad8", "write", "energy_mode"): ["delta", "local", "constant"],
    ("xtrack", "write", "energy_mode"): ["delta", "local", "constant"],
    ("scibmad", "write", "energy_mode"): ["delta", "local", "constant"],
    ("madng", "write", "energy_mode"): ["delta", "local", "constant"],
    ("impactx", "write", "energy_mode"): ["local", "constant"],
    ("flame", "write", "energy_mode"): ["local", "constant"],
    ("synergia", "write", "energy_mode"): ["constant", "local", "delta"],
    ("bmad", "write", "line_mode"): ["nested", "flat"],
    ("xtrack", "write", "document"): ["auto", "line", "environment"],
    ("impactx", "write", "flavor"): ["python", "inputs"],
    ("impactz", "write", "integrator"): ["auto", "map", "lorentz"],
    ("impactz", "write", "rf_model"): ["ideal", "rfdata"],
    ("pals", "write", "format"): ["auto", "yaml", "json"],
    ("pals", "write", "flavor"): ["standard", "flat", "beamline"],
    ("scibmad", "write", "using"): ["Beamlines", "SciBmad"],
    ("tracewin", "write", "static_maps"): ["", "hard_edge"],
    ("xtrack", "read", "energy_mode"): ["delta", "local", "constant"],
    ("synergia", "read", "energy_mode"): ["constant", "local", "delta"],
}
_JSON_OPTIONS = {"grid", "envelope", "beam_extent", "twiss", "errors", "distribution", "sequences"}
_STATIC_WRITE = {"tracewin": [{"name": "frequency_Hz", "type": "number", "default": None,
                              "help": "FREQ card of the header [Hz]; default: the reference clock"},
                             {"name": "static_maps", "type": "choice", "default": "", "choices": ["", "hard_edge"],
                              "help": "write static field maps as hard-edge elements"}]}
_SKIP_PARAMS = {"self", "lattice", "path", "report", "stem", "field_file", "options", "kwargs"}
_LINE_KEYS = ("line", "sequence", "use", "root")


def _type_of(ann: Any, name: str) -> tuple[str, bool]:
    """``(field type, optional)`` from an annotation object or its string."""
    text = ann if isinstance(ann, str) else (getattr(ann, "__name__", None) or str(ann))
    text = text.replace("typing.", "")
    optional = "None" in text
    core = text.replace("| None", "").replace("None |", "").replace("Optional[", "").strip(" []")
    if name in _JSON_OPTIONS or core.startswith(("tuple", "list", "dict", "Sequence", "Iterable")):
        return "json", optional
    if core == "bool":
        return "bool", optional
    if core == "int":
        return "int", optional
    if core == "float" or "float" in core:
        return "number", optional
    if "Path" in core:
        return "path", optional
    return "string", optional


def _params(fn) -> list[_inspect.Parameter]:
    try:
        sig = _inspect.signature(fn, eval_str=True)
    except Exception:  # noqa: BLE001 - TYPE_CHECKING-only names: fall back to the string annotations
        sig = _inspect.signature(fn)
    return list(sig.parameters.values())


def _fields(fn, fmt: str, side: str) -> tuple[list[dict], bool]:
    out: list[dict] = []
    var_kw = False
    for prm in _params(fn):
        if prm.kind == prm.VAR_KEYWORD:
            var_kw = True
            continue
        if prm.name in _SKIP_PARAMS or prm.kind == prm.VAR_POSITIONAL:
            continue
        if prm.name == "strict" and side == "write":
            continue
        ann = prm.annotation if prm.annotation is not prm.empty else "str"
        if isinstance(ann, str) and "ReferenceParticle" in ann or getattr(ann, "__name__", "") == "ReferenceParticle":
            continue
        typ, optional = _type_of(ann, prm.name)
        default = None if prm.default is prm.empty else prm.default
        if isinstance(default, tuple):
            default = list(default)
        field = {"name": prm.name, "type": typ, "default": jsonable(default), "optional": optional}
        choices = CHOICES.get((fmt, side, prm.name))
        if choices:
            field["type"], field["choices"] = "choice", choices
        if prm.name == "species":
            field["type"], field["choices"], field["free"] = "choice", sorted(SPECIES), True
        if prm.name in _LINE_KEYS and side == "read":
            field["canonical"] = "line"
        out.append(field)
    return out, var_kw


@functools.cache
def option_schema(fmt: str) -> dict:
    """``{"read": [...], "write": [...]}`` field descriptions for the reader/writer keyword options."""
    spec = FORMATS[fmt]
    read: list[dict] = []
    write: list[dict] = []
    if spec.reader_attr:
        try:
            rd = spec.reader()
            read, _ = _fields(rd.read, fmt, "read")
        except Exception:  # noqa: BLE001 - a bridge reader without its tool still lists nothing
            read = []
    if spec.writer_attr:
        if fmt in _STATIC_WRITE:
            write = list(_STATIC_WRITE[fmt])
        else:
            try:
                wr = spec.writer()
                write, var_kw = _fields(wr.write, fmt, "write")
                if var_kw:
                    for alt in ("render", "to_document", "dumps"):
                        if hasattr(wr, alt):
                            extra, _ = _fields(getattr(wr, alt), fmt, "write")
                            names = {f["name"] for f in write}
                            write += [f for f in extra if f["name"] not in names]
                            break
            except Exception:  # noqa: BLE001
                write = []
    return {"read": read, "write": write}


def coerce_options(fmt: str, side: str, options: dict | None) -> dict:
    """Typed keyword options from the page's strings/JSON (unknown keys are rejected unless the writer
    takes ``**options``)."""
    schema = {f["name"]: f for f in option_schema(fmt)[side]}
    out: dict = {}
    for key, val in (options or {}).items():
        f = schema.get(key)
        if f is None:
            if side == "write" and _takes_var_kw(fmt):
                out[key] = val
                continue
            raise ValueError(f"{fmt} {side}er has no option {key!r}")
        if val is None or val == "":
            if f.get("type") == "choice" and "" in (f.get("choices") or []):
                out[key] = ""
            continue
        t = f["type"]
        if t == "bool":
            out[key] = val if isinstance(val, bool) else str(val).strip().lower() in ("1", "true", "yes", "on")
        elif t == "int":
            out[key] = int(float(val))
        elif t == "number":
            out[key] = float(val)
        elif t == "json":
            v = json.loads(val) if isinstance(val, str) else val
            out[key] = tuple(v) if isinstance(v, list) and key in ("grid", "envelope", "beam_extent") else v
        else:
            out[key] = val
    return out


def _takes_var_kw(fmt: str) -> bool:
    try:
        return any(p.kind == p.VAR_KEYWORD for p in _params(FORMATS[fmt].writer().write))
    except Exception:  # noqa: BLE001
        return False


def canonical_read_options(fmt: str, options: dict | None, warnings: list[str] | None = None) -> dict:
    """The page's reader options (``species``, ``kinetic_energy_eV``, ``frequency_Hz``, ``line`` and the
    reader's own) mapped onto this reader's keywords; what the reader cannot take is dropped with a warning."""
    schema = {f["name"]: f for f in option_schema(fmt)["read"]}
    out: dict = {}
    for key, val in (options or {}).items():
        if val is None or val == "":
            continue
        if key == "line":
            target = next((k for k in _LINE_KEYS if k in schema), None)
            if target is None:
                _warn(warnings, f"the {fmt} reader takes no line/sequence name; {val!r} ignored")
                continue
            out[target] = val
            continue
        if key not in schema:
            _warn(warnings, f"the {fmt} reader takes no option {key!r}; ignored")
            continue
        out[key] = val
    return coerce_options(fmt, "read", out)


def _warn(warnings: list[str] | None, text: str) -> None:
    if warnings is not None:
        warnings.append(text)


def _rules_of(writer) -> dict | None:
    rules = getattr(writer, "RULES", None)
    if not isinstance(rules, dict):
        return None
    out: dict = {}
    for kind, rule in rules.items():
        if is_dataclass(rule):
            d = {k: v for k, v in vars(rule).items() if isinstance(v, str | int | float | bool | type(None))}
            out[kind] = {"target": d.get("target"), "cls": str(d.get("cls", "")).upper(), "code": d.get("code"),
                         "message": d.get("message", "")}
        elif isinstance(rule, str):
            out[kind] = {"target": None, "cls": rule.upper(), "code": None, "message": ""}
        else:
            return None
    return out


@functools.cache
def format_catalog() -> dict:
    formats: dict = {}
    battery = set(readable_formats())
    for name, spec in FORMATS.items():
        rules = None
        if spec.writer_attr and not spec.options.get("bridge"):
            try:
                rules = _rules_of(spec.writer())
            except Exception:  # noqa: BLE001
                rules = None
        schema = option_schema(name)
        formats[name] = {"description": spec.description, "suffixes": list(spec.suffixes),
                         "readable": bool(spec.reader_attr), "writable": bool(spec.writer_attr),
                         "bridge": bool(spec.options.get("bridge")), "side_files": name in _SIDE_FILE_FORMATS,
                         "battery": name in battery, "read_options": schema["read"],
                         "write_options": schema["write"], "rules": rules,
                         "engine": ENGINE_FOR_FORMAT.get(name),
                         "engine_candidates": list(ENGINE_CANDIDATES.get(name, ()))}
    return {"formats": formats,
            "readable": [n for n, s in FORMATS.items() if s.reader_attr],
            "writable": [n for n, s in FORMATS.items() if s.writer_attr],
            "kinds": sorted(ALL_KINDS), "shapes": SHAPES, "species": sorted(SPECIES)}


@functools.cache
def catalog_view() -> dict:
    from lattix.fidelity_catalog import scan

    out = {}
    for code, info in scan().items():
        where = info.get("where") or []
        out[code] = {"cls": info.get("class"), "message": info.get("message", ""),
                     "where": where[0] if where else ""}
    return out


def sample_decks() -> list[dict]:
    if not PUBLIC.exists():
        return []
    return [{"label": rel, "path": rel, "format": fmt, "options": dict(opts)} for rel, fmt, opts in DECKS
            if (PUBLIC / rel).exists()]
