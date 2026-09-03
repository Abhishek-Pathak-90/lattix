"""MAD-X writer (PLAN §6 task 1.5): table-driven, never silent.

Two layouts:

``mode="sequence"`` (default)
    element definitions followed by ``name: sequence, l=…, refer=centre;`` with
    ``at=`` at each element's centre taken from :meth:`Lattice.flatten`.  Drifts
    are **not** written — MAD-X fills every gap with an implicit ``drift_N``,
    which is exactly what the reader sees again (invariant I-13).

``mode="line"``
    every element (drifts included) is defined and the IR's ``Line``s are emitted
    as ``name: line=(…)`` preserving nesting, ``3*sub`` repeats and ``-sub``
    reflections.

Reference-energy policy — MAD-X keeps ``p0`` constant through RF, so an
accelerating line cannot be represented exactly:

``energy_mode="local"`` (default)
    every normalized strength uses ``Bρ`` at *that element's* entrance from
    :func:`lattix.ir.walk.propagate`, so each section's optics is right; an
    ``EQUIVALENT:CONST_P0_LOCAL_RIGIDITY`` entry is recorded per accelerating
    element.
``energy_mode="constant"``
    one ``Bρ`` (the lattice start) for everything, recorded as
    ``EQUIVALENT:CONST_P0_START_RIGIDITY``.

Every accelerating cavity additionally gets ``EQUIVALENT:CONST_P0``.

**Measured caveat (MAD-X 5.09.03, 2026-09-03).**  MAD-X keeps ``p0`` constant but
its ``TWISS`` *does* carry a cavity's gain in the reference orbit's ``pt``
(``pt = ΔE/(p0·c)``; 50 MV on a 100 MeV proton gives ``pt = 0.1124648``) and
linearises every downstream magnet about that orbit.  So when the accelerating
cavities are present in the written deck, MAD-X already re-scales the downstream
strengths itself and ``energy_mode="constant"`` is the mode that composes with
it; ``"local"`` pre-scales a second time and suits a deck whose energy changes
are *not* modelled by MAD-X (a :class:`~lattix.ir.elements.ReferenceChange`, a
field map degraded to a drift, or a section extracted on its own with a matching
``BEAM``).  Either way the choice is in the ledger, never silent.

Conventions (measured against MAD-X 5.09.03, see ``reader.py``): ``volt`` in MV,
``freq`` in MHz, ``lag = φ/2π + 1/4`` (:func:`lattix.ir.rf.madx_lag`), RBEND
``l`` is the **chord** ``L_arc·sin(θ/2)/(θ/2)`` with pole faces referenced to it
(``e1 -= θ/2``), and all normalized strengths use the *signed* rigidity.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.madx.naming import NameMap, is_valid, name_tag
from lattix.ir.elements import ALL_KINDS, ApertureP, Directive, Drift, Element, FieldMap, Freq, Superposition
from lattix.ir.expr import ExpressionError, evaluate
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle
from lattix.ir.rf import madx_lag
from lattix.ir.walk import energy_gain_eV, propagate

#: MAD-X named particles whose mass/charge MAD-X already knows.
_NAMED_PARTICLES = frozenset({"proton", "antiproton", "electron", "positron",
                              "negmuon", "posmuon"})

#: IR ``Instrument.family`` → MAD-X base type (a superset of the "BPM → monitor"
#: rule so ``hmonitor``/``vmonitor``/``placeholder`` survive a round trip).
_FAMILY_TYPE = {"BPM": "monitor", "MONITOR": "monitor", "HMONITOR": "hmonitor",
                "VMONITOR": "vmonitor", "PLACEHOLDER": "placeholder"}

#: MAD-X base types that reject ``apertype``/``aperture`` ("illegal keyword: apertype");
#: measured by scanning every base type's command parameters in MAD-X 5.09.03.
_NO_APERTURE = frozenset({"drift", "matrix", "translation"})

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_EXPR_TOL = 1e-12
#: an element counts as accelerating above 1 µeV of reference gain — far below any
#: physical gain and far above the ``cos(±π/2) ≈ 6e-17`` noise of a bunching cavity.
_ACCEL_TOL_eV = 1e-6


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str                 # the MAD-X construct this kind becomes
    cls: str = "EXACT"          # default fidelity class
    code: str = "OK"
    message: str = ""


@dataclass
class _Item:
    """One occurrence to emit: an element, where it sits, and its local rigidity."""

    element: Element
    s_in: float
    s_out: float
    brho: float
    dE: float


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    return "0" if s in ("-0", "-0.0") else s


def _vec(values: list[float]) -> str:
    return "{" + ",".join(_num(v) for v in values) + "}"


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "madx"

    RULES: dict[str, Rule] = {
        "Drift": Rule("drift"),
        "Quadrupole": Rule("quadrupole"),
        "Sextupole": Rule("sextupole"),
        "Octupole": Rule("octupole"),
        "Multipole": Rule("multipole"),
        "Bend": Rule("sbend/rbend"),
        "Solenoid": Rule("solenoid"),
        "RFCavity": Rule("rfcavity"),
        "FieldMap": Rule("rfcavity/solenoid/quadrupole/drift", "LOSSY", "FM_TO_DRIFT",
                         "field map degraded per its integrated summary: RF → rfcavity "
                         "(FM_TO_CAVITY), static solenoid/quadrupole → hard edge with drift "
                         "padding (FM_SOL_HARDEDGE / FM_QUAD_HARDEDGE), otherwise a drift"),
        "NCells": Rule("drift", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("drift", "LOSSY", "RFQ_TO_DRIFT",
                        "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("kicker"),
        "Collimator": Rule("rcollimator/ecollimator"),
        "Marker": Rule("marker"),
        "Instrument": Rule("monitor/instrument"),
        "Foil": Rule("marker", "LOSSY", "FOIL_TO_MARKER",
                     "MAD-X has no stripping foil; written as a marker"),
        "Taylor": Rule("matrix"),
        "Patch": Rule("marker", "LOSSY", "PATCH_DROPPED",
                      "MAD-X has no patch element; written as a marker"),
        "ReferenceChange": Rule("marker", "LOSSY", "REFCHANGE_DROPPED",
                                "MAD-X cannot change the reference energy; written as a marker"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each MAD-X cavity's freq attribute"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "overlapping fields written as consecutive elements"),
    }

    #: Directive roles MAD-X can honour as a plain comment without losing anything.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              mode: str = "sequence", energy_mode: str = "local",
              use_expressions: bool = True, sequence_name: str | None = None) -> FidelityReport:
        if mode not in ("sequence", "line"):
            raise ValueError(f"mode must be 'sequence' or 'line', got {mode!r}")
        if energy_mode not in ("local", "constant"):
            raise ValueError(f"energy_mode must be 'local' or 'constant', got {energy_mode!r}")

        rep = FidelityReport(target_format="madx", target_file=str(path))
        placed = propagate(lattice)
        if mode == "sequence" and (not placed or placed[-1].s_out <= 0.0):
            # MAD-X: "fatal: missing length for sequence" — a zero-length line is accepted (measured)
            mode = "line"
            rep.equivalent("ZERO_LENGTH_LINE_MODE",
                           "MAD-X rejects a zero-length sequence; the lattice was written as a line")
        items = self._items(lattice, placed, rep)
        self._record_energy_mode(items, energy_mode, rep)

        names = NameMap()
        root = lattice.use or lattice.name
        seq_name = names.reserve(sequence_name
                                 or (root if root in lattice.lines else None)
                                 or lattice.name or root or "lattix_seq")
        line_names: dict[str, str] = {}
        if mode == "line" and root in lattice.lines:
            line_names[root] = seq_name
            for ln in lattice.lines:
                if ln != root:
                    line_names[ln] = names.reserve(ln)

        start_brho = lattice.reference.brho_signed
        defs: dict[int, tuple[str, Element, float]] = {}      # id(element) -> (name, el, brho)
        for it in items:
            key = id(it.element)
            if key in defs:
                brho0 = defs[key][2]
                local = it.brho if energy_mode == "local" else start_brho
                if abs(local - brho0) > 1e-12 * max(1.0, abs(brho0)):
                    rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                                   "one definition is used at two reference energies; the first "
                                   "occurrence's rigidity is used for its normalized strengths",
                                   element=it.element.name, kind=it.element.kind,
                                   brho_first=brho0, brho_here=local)
                continue
            defs[key] = (names.assign(it.element.name, it.element),
                         it.element, it.brho if energy_mode == "local" else start_brho)

        body: list[str] = [f"! lattix {__version__} from {lattice.meta.get('source_format', 'IR')}"]
        title = lattice.meta.get("madx_title")
        if title:
            body.append(f'title, "{title}";')
        body.append("")
        body.append(self._beam_line(lattice.reference))
        variables = self._variable_lines(lattice, use_expressions, rep)
        if variables:
            body.append("")
            body.append("! --- variables")
            body.extend(variables)

        body.append("")
        body.append("! --- elements")
        seen: set[int] = set()
        for it in items:
            key = id(it.element)
            if key in seen:
                continue
            seen.add(key)
            name, el, brho = defs[key]
            if mode == "sequence" and isinstance(el, Drift):
                self._record_omitted_drift(el, rep)
                continue
            text = self._definition(name, el, brho, lattice, use_expressions, rep)
            if text:
                body.append(text)

        body.append("")
        total = max([p.s_out for p in placed] + [it.s_out for it in items] + [0.0])
        if mode == "sequence":
            body.extend(self._sequence_block(seq_name, items, defs, total))
        else:
            body.extend(self._line_block(seq_name, line_names, lattice, defs, rep))
        body.append("")
        body.append(f"use, sequence={seq_name};")
        body.extend(self._ealign_block(defs))
        Path(path).write_text("\n".join(body).rstrip("\n") + "\n", encoding="utf-8")
        rep.raise_if(strict)
        return rep

    @staticmethod
    def _record_omitted_drift(el: Element, rep: FidelityReport) -> None:
        """A drift is a gap in a MAD-X sequence, so anything attached to it is lost."""
        ap = el.aperture
        if ap is not None and (ap.half_x is not None or ap.half_y is not None):
            rep.lossy("APERTURE_DROPPED",
                      "MAD-X 'drift' has no apertype/aperture attribute",
                      element=el.name, kind="Drift", madx_type="drift",
                      half_x=ap.half_x, half_y=ap.half_y)
        elif el.shift is not None and not el.shift.is_zero():
            rep.lossy("DRIFT_SHIFT_DROPPED",
                      "a drift is an implicit gap in a MAD-X sequence and cannot carry EALIGN",
                      element=el.name, kind="Drift")
        else:
            rep.exact(el.name, "Drift", message="implicit MAD-X drift (a gap in the sequence)")

    # -- flattening ---------------------------------------------------------
    def _items(self, lattice: Lattice, placed: list[Placed], rep: FidelityReport) -> list[_Item]:
        out: list[_Item] = []
        for p in placed:
            el = p.element
            ref: ReferenceParticle = p.ref_in or lattice.reference
            if isinstance(el, Superposition):
                rep.lossy("SUPERPOSITION_FLATTENED",
                          "overlapping fields written as consecutive MAD-X elements",
                          element=el.name, kind="Superposition", children=len(el.children))
                for offset, child_name in el.children:
                    child = lattice.elements.get(child_name)
                    if child is None:
                        rep.dropped("SUPERPOSITION_CHILD_MISSING",
                                    f"superposition child {child_name!r} is not defined",
                                    element=el.name, kind="Superposition")
                        continue
                    s0 = p.s_in + offset
                    if isinstance(child, FieldMap):     # the ladder applies inside a cluster too
                        out.extend(self._fieldmap_items(child, s0, ref, rep))
                        continue
                    out.append(_Item(child, s0, s0 + child.length, ref.brho_signed, 0.0))
                continue
            if isinstance(el, FieldMap):
                out.extend(self._fieldmap_items(el, p.s_in, ref, rep))
                continue
            out.append(_Item(el, p.s_in, p.s_out, ref.brho_signed,
                             energy_gain_eV(el, ref)))
        return self._resolve_overlaps(out, rep)

    @staticmethod
    def _fieldmap_items(el: FieldMap, s_in: float, ref: ReferenceParticle,
                        rep: FidelityReport) -> list[_Item]:
        """The field map's degradation ladder (PLAN §4.3): an RF map becomes a full-length
        ``rfcavity`` (MAD-X applies its kick at the centre between two L/2 drifts), a static
        solenoid/quadrupole map a hard edge centred in the map, anything unknown a drift."""
        r = replacement_for(el)
        rep.add(r.cls, r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, message in r.extra:
            rep.add(cls, code, message, element=el.name, kind="FieldMap")
        out, s = [], s_in
        for part in r.parts:
            out.append(_Item(part, s, s + part.length, ref.brho_signed, energy_gain_eV(part, ref)))
            s += part.length
        return out

    @staticmethod
    def _resolve_overlaps(items: list[_Item], rep: FidelityReport) -> list[_Item]:
        """MAD8 tolerates negative drifts (overlapping elements); MAD-X aborts on them
        ("negative drift between elements").  Shift an overlapping thick element to the
        previous element's end (LOSSY, geometry moved), move zero-length elements out of the
        overlap, and shorten the following drift so the total length is preserved."""
        out: list[_Item] = []
        end = 0.0
        last_thick: int | None = None
        # a MAD8 negative drift usually just lists elements out of physical order (a marker
        # placed downstream of elements that physically precede it): order by position first,
        # so only genuine thick-element collisions are treated as overlaps
        items = sorted(enumerate(items), key=lambda kv: (round(kv[1].s_in, 12), kv[0]))
        items = [it for _, it in items]
        for it in items:
            el = it.element
            if el.length < 0.0 and isinstance(el, Drift):
                # a negative drift is pure bookkeeping: the following elements' s already include it
                rep.equivalent("NEGATIVE_DRIFT_DROPPED",
                               f"negative drift {el.length:.6g} m has no MAD-X form; downstream "
                               "overlaps are resolved element by element", element=el.name, kind="Drift")
                continue
            s_in, s_out = it.s_in, it.s_out
            if s_in < end - 1e-9 and last_thick is not None and isinstance(out[last_thick].element, Drift) \
                    and out[last_thick].s_in <= s_in + 1e-9 and not isinstance(el, Drift):
                # the overlap is with a DRIFT: a drift is only a gap in a MAD-X sequence, so
                # shortening it keeps every other element exactly where the source put it
                prev = out[last_thick]
                rep.equivalent("DRIFT_SHORTENED_BY_OVERLAP",
                               f"preceding drift shortened by {end - s_in:.6g} m so {el.name!r} keeps its position",
                               element=prev.element.name, kind="Drift", shift_m=end - s_in)
                out[last_thick] = _Item(prev.element, prev.s_in, s_in, prev.brho, prev.dE)
                end = s_in
            if s_in < end - 1e-9:
                shift = end - s_in
                if isinstance(el, Drift):
                    if s_out < end - 1e-9:
                        rep.lossy("DRIFT_INSIDE_OVERLAP", f"drift {el.name!r} lies entirely inside a preceding "
                                  "element and was dropped", element=el.name, kind="Drift")
                        continue
                    rep.equivalent("DRIFT_SHORTENED_BY_OVERLAP",
                                   f"drift shortened by {shift:.6g} m to absorb an upstream overlap",
                                   element=el.name, kind="Drift", shift_m=shift)
                    s_in = end
                elif el.length == 0.0:
                    rep.lossy("MARKER_MOVED_OUT_OF_OVERLAP",
                              f"zero-length element moved {shift:.6g} m downstream out of an overlap",
                              element=el.name, kind=el.kind, shift_m=shift)
                    s_in = s_out = end
                else:
                    rep.lossy("OVERLAP_SHIFTED",
                              f"element starts {shift:.6g} m inside the previous one (MAD8 negative drift); "
                              "shifted downstream — MAD-X cannot overlap elements",
                              element=el.name, kind=el.kind, shift_m=shift)
                    s_in, s_out = end, s_out + shift
            out.append(_Item(el, s_in, s_out, it.brho, it.dE))
            if s_out > s_in:
                last_thick = len(out) - 1
            end = max(end, s_out)
        return out

    @staticmethod
    def _record_energy_mode(items: list[_Item], energy_mode: str, rep: FidelityReport) -> None:
        local = energy_mode == "local"
        code = "CONST_P0_LOCAL_RIGIDITY" if local else "CONST_P0_START_RIGIDITY"
        msg = ("normalized strengths use the local rigidity at each element's entrance"
               if local else "normalized strengths use the rigidity at the start of the lattice")
        for it in items:
            if abs(it.dE) <= _ACCEL_TOL_eV:
                continue
            rep.equivalent("CONST_P0",
                           "MAD-X keeps p0 constant across RF: the reference energy does not "
                           "follow this element's gain",
                           element=it.element.name, kind=it.element.kind, dE_eV=it.dE)
            rep.equivalent(code, msg, element=it.element.name, kind=it.element.kind,
                           dE_eV=it.dE, brho=it.brho)

    # -- header -------------------------------------------------------------
    @staticmethod
    def _beam_line(ref: ReferenceParticle) -> str:
        sp = ref.species
        particle = sp.name.lower() if sp.name.lower() in _NAMED_PARTICLES else "ion"
        return (f"beam, particle={particle}, mass={_num(sp.mass_eV / 1e9)}, "
                f"charge={_num(sp.charge)}, energy={_num(ref.total_energy_eV / 1e9)};")

    def _variable_lines(self, lattice: Lattice, use_expressions: bool,
                        rep: FidelityReport) -> list[str]:
        if not use_expressions or not lattice.variables:
            return []
        keep = {}
        for name, var in lattice.variables.items():
            if is_valid(name):
                keep[name] = var
            else:
                rep.equivalent("VARIABLE_DROPPED",
                               f"variable {name!r} is not a writable MAD-X identifier; its value "
                               "is folded into the numbers",
                               element=None, kind=None, variable=name)
        lines = []
        for name in self._topo(keep):
            var = keep[name]
            expr = var.expression
            if expr is not None and expr.text.strip():
                lines.append(f"{name} {':=' if expr.deferred else '='} {expr.text.strip()};")
            else:
                lines.append(f"{name} = {_num(var.value)};")
        return lines

    @staticmethod
    def _topo(variables: dict) -> list[str]:
        """Definition order with dependencies first (stable otherwise)."""
        deps = {name: {t for t in _ID.findall(var.expression.text if var.expression else "")
                       if t in variables and t != name}
                for name, var in variables.items()}
        out: list[str] = []
        done: set[str] = set()

        def visit(n: str, stack: frozenset[str]) -> None:
            if n in done or n in stack:
                return
            for d in sorted(deps[n]):
                visit(d, stack | {n})
            done.add(n)
            out.append(n)

        for name in variables:
            visit(name, frozenset())
        return out

    # -- element definitions ------------------------------------------------
    def _definition(self, name: str, el: Element, brho: float, lattice: Lattice,
                    use_expressions: bool, rep: FidelityReport) -> str | None:
        rule = self.RULES.get(el.kind)
        if rule is None:                                    # pragma: no cover - RULES is total
            raise KeyError(f"MAD-X writer has no rule for kind {el.kind!r}")
        if isinstance(el, Freq):
            rep.exact(el.name, "Freq", message=rule.message)
            return None
        if isinstance(el, Directive):
            if el.role in self.COMMENT_ROLES:
                rep.exact(el.name, "Directive", code="DIRECTIVE_AS_COMMENT",
                          message=f"role {el.role!r} carries no MAD-X physics; written as a comment")
            else:
                rep.dropped(rule.code, rule.message, element=el.name, kind="Directive",
                            card=el.card, role=el.role)
            return None
        base, attrs = getattr(self, f"_def_{el.kind.lower()}")(el, brho, lattice,
                                                               use_expressions, rep)
        attrs = list(attrs)
        if el.kind != "Collimator":                 # a collimator carries xsize/ysize instead
            ap_attrs = self._aperture_attrs(el.aperture)
            if ap_attrs and base in _NO_APERTURE:
                rep.lossy("APERTURE_DROPPED",
                          f"MAD-X {base!r} has no apertype/aperture attribute",
                          element=el.name, kind=el.kind, madx_type=base,
                          half_x=el.aperture.half_x, half_y=el.aperture.half_y)
            else:
                attrs += ap_attrs
        self._record(rep, el, rule)
        original = (el.provenance.original_name if el.provenance else None) or el.name
        tag = ("   " + name_tag(original, el.provenance.original_type if el.provenance else None)
               if name != original else "")
        joined = ", ".join(attrs)
        return f"{name}: {base}" + (f", {joined}" if joined else "") + ";" + tag

    @staticmethod
    def _record(rep: FidelityReport, el: Element, rule: Rule) -> None:
        if rule.cls == "EXACT":
            rep.exact(el.name, el.kind)
        else:
            rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                    target=rule.target)

    @staticmethod
    def _aperture_attrs(ap: ApertureP | None) -> list[str]:
        if ap is None or (ap.half_x is None and ap.half_y is None):
            return []
        hx = ap.half_x if ap.half_x is not None else ap.half_y
        hy = ap.half_y if ap.half_y is not None else ap.half_x
        kind = "rectangle" if ap.shape == "RECTANGULAR" else "ellipse"
        return [f"apertype={kind}", f"aperture={_vec([hx, hy])}"]

    def _attr(self, el: Element, madx_attr: str, ir_path: str, value: float,
              lattice: Lattice, use_expressions: bool, rep: FidelityReport) -> str:
        """``k1=0.6``, or ``k1 := kqf`` when the stored expression still gives *value*.

        The expression text is the **normalized MAD-X quantity** the reader captured
        (``elem.cmdpar['k1'].expr``), while the IR stores the lab field, so the value
        check is done in MAD-X units — the number this call would otherwise print.
        """
        if use_expressions:
            expr = el.expressions.get(ir_path)
            text = el.native.get("madx", {}).get(f"{madx_attr}_expr")
            if text is None and expr is not None:
                text = expr.text
            if text:
                variables = {k: v.value for k, v in lattice.variables.items()}
                try:
                    got = evaluate(str(text), variables)
                except ExpressionError:
                    got = None
                if got is not None and abs(got - value) <= _EXPR_TOL * max(1.0, abs(value)):
                    op = ":=" if (expr is None or expr.deferred) else "="
                    return f"{madx_attr} {op} {str(text).strip()}"
                why = ("cannot be evaluated by lattix (element-attribute or macro reference)"
                       if got is None else "no longer matches the IR value")
                rep.equivalent("EXPRESSION_DROPPED",
                               f"{madx_attr} expression {str(text)!r} {why}; "
                               "the number is written instead",
                               element=el.name, kind=el.kind, attr=madx_attr, value=value)
        return f"{madx_attr}={_num(value)}"

    # per-kind builders: (madx base type, [attribute strings])
    def _def_drift(self, el, brho, lat, ux, rep):
        return "drift", [self._attr(el, "l", "length", el.length, lat, ux, rep)]

    def _def_quadrupole(self, el, brho, lat, ux, rep):
        attrs = [self._attr(el, "l", "length", el.length, lat, ux, rep),
                 self._attr(el, "k1", "multipole.Bn[1]",
                            el.multipole.Bn.get(1, 0.0) / brho, lat, ux, rep)]
        if el.multipole.tilt.get(1):
            attrs.append(self._attr(el, "tilt", "multipole.tilt[1]",
                                    el.multipole.tilt[1], lat, ux, rep))
        return "quadrupole", attrs

    def _def_sextupole(self, el, brho, lat, ux, rep):
        return "sextupole", self._thick_multipole(el, brho, lat, ux, rep, 2, "k2", "k2s")

    def _def_octupole(self, el, brho, lat, ux, rep):
        return "octupole", self._thick_multipole(el, brho, lat, ux, rep, 3, "k3", "k3s")

    def _thick_multipole(self, el, brho, lat, ux, rep, order, kn, ks):
        attrs = [self._attr(el, "l", "length", el.length, lat, ux, rep),
                 self._attr(el, kn, f"multipole.Bn[{order}]",
                            el.multipole.Bn.get(order, 0.0) / brho, lat, ux, rep)]
        if el.multipole.Bs.get(order):
            attrs.append(self._attr(el, ks, f"multipole.Bs[{order}]",
                                    el.multipole.Bs[order] / brho, lat, ux, rep))
        if el.multipole.tilt.get(order):
            attrs.append(self._attr(el, "tilt", f"multipole.tilt[{order}]",
                                    el.multipole.tilt[order], lat, ux, rep))
        return attrs

    def _def_multipole(self, el, brho, lat, ux, rep):
        attrs = []
        for key, table in (("knl", el.multipole.BnL), ("ksl", el.multipole.BsL)):
            if table:
                attrs.append(f"{key}={_vec([table.get(i, 0.0) / brho for i in range(max(table) + 1)])}")
        lrad = el.native.get("madx", {}).get("lrad")
        if lrad:
            attrs.append(f"lrad={_num(lrad)}")
        if el.multipole.tilt.get(0):
            attrs.append(f"tilt={_num(el.multipole.tilt[0])}")
        return "multipole", attrs

    def _def_bend(self, el, brho, lat, ux, rep):
        b = el.bend
        base = "rbend" if b.rect else "sbend"
        half = b.angle / 2.0 if b.rect else 0.0
        length = el.length
        if b.rect and b.angle:
            length = el.length * math.sin(b.angle / 2.0) / (b.angle / 2.0)     # arc -> chord
        attrs = [f"l={_num(length)}",
                 self._attr(el, "angle", "bend.angle", b.angle, lat, ux, rep)]
        if (b.e1 - half) or (b.e2 - half):
            attrs.append(self._attr(el, "e1", "bend.e1", b.e1 - half, lat, ux, rep))
            attrs.append(self._attr(el, "e2", "bend.e2", b.e2 - half, lat, ux, rep))
        if b.edge_int1:
            attrs.append(f"fint={_num(b.edge_int1)}")
        if b.edge_int2 is not None:
            attrs.append(f"fintx={_num(b.edge_int2)}")
        if b.hgap:
            attrs.append(f"hgap={_num(b.hgap)}")
        if el.multipole.Bn.get(1):
            attrs.append(self._attr(el, "k1", "multipole.Bn[1]",
                                    el.multipole.Bn[1] / brho, lat, ux, rep))
        if el.multipole.Bn.get(2):
            attrs.append(self._attr(el, "k2", "multipole.Bn[2]",
                                    el.multipole.Bn[2] / brho, lat, ux, rep))
        if b.tilt_ref:
            attrs.append(f"tilt={_num(b.tilt_ref)}")
        return base, attrs

    def _def_solenoid(self, el, brho, lat, ux, rep):
        return "solenoid", [self._attr(el, "l", "length", el.length, lat, ux, rep),
                            self._attr(el, "ks", "solenoid.Bsol_T",
                                       el.solenoid.Bsol_T / brho, lat, ux, rep)]

    def _def_rfcavity(self, el, brho, lat, ux, rep):
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        attrs = [f"l={_num(el.length)}",
                 self._attr(el, "volt", "rf.voltage_V", volt * 1e-6, lat, ux, rep),
                 f"lag={_num(madx_lag(rf.phase_rad))}"]
        if rf.frequency_Hz:
            attrs.append(self._attr(el, "freq", "rf.frequency_Hz",
                                    rf.frequency_Hz * 1e-6, lat, ux, rep))
        harmon = el.native.get("madx", {}).get("harmon")
        if harmon:
            attrs.append(f"harmon={int(harmon)}")
        return "rfcavity", attrs

    def _def_fieldmap(self, el, brho, lat, ux, rep):   # pragma: no cover - expanded in _items
        return "drift", [f"l={_num(el.length)}"]

    def _def_ncells(self, el, brho, lat, ux, rep):
        return "drift", [f"l={_num(el.length)}"]

    _def_rfqcell = _def_ncells

    def _def_kicker(self, el, brho, lat, ux, rep):
        if el.electric:
            rep.lossy("EKICK_AS_MAGNETIC",
                      "electric steerer written as a magnetic MAD-X kicker",
                      element=el.name, kind="Kicker")
        return "kicker", [f"l={_num(el.length)}",
                          self._attr(el, "hkick", "hkick", el.hkick, lat, ux, rep),
                          self._attr(el, "vkick", "vkick", el.vkick, lat, ux, rep)]

    def _def_collimator(self, el, brho, lat, ux, rep):
        ap = el.aperture
        base = "rcollimator" if (ap is not None and ap.shape == "RECTANGULAR") else "ecollimator"
        attrs = [f"l={_num(el.length)}"]
        if ap is not None and ap.half_x is not None:
            attrs.append(f"xsize={_num(ap.half_x)}")
            attrs.append(f"ysize={_num(ap.half_y if ap.half_y is not None else ap.half_x)}")
        return base, attrs

    def _def_marker(self, el, brho, lat, ux, rep):
        return "marker", []

    def _def_instrument(self, el, brho, lat, ux, rep):
        return _FAMILY_TYPE.get(el.family.upper(), "instrument"), [f"l={_num(el.length)}"]

    def _def_foil(self, el, brho, lat, ux, rep):
        return "marker", []

    def _def_taylor(self, el, brho, lat, ux, rep):
        attrs = [f"l={_num(el.length)}"]
        for i in range(6):
            for j in range(6):
                v = el.matrix[i][j]
                if v != (1.0 if i == j else 0.0):
                    attrs.append(f"rm{i + 1}{j + 1}={_num(v)}")
        attrs += [f"kick{i + 1}={_num(v)}" for i, v in enumerate(el.offset) if v]
        return "matrix", attrs

    def _def_patch(self, el, brho, lat, ux, rep):
        return "marker", []

    def _def_referencechange(self, el, brho, lat, ux, rep):
        return "marker", []

    def _def_superposition(self, el, brho, lat, ux, rep):    # pragma: no cover - expanded in _items
        return "marker", []

    # -- sequence / line ----------------------------------------------------
    def _sequence_block(self, seq_name: str, items: list[_Item], defs: dict,
                        total: float) -> list[str]:
        if total <= 0.0:
            raise ValueError(
                "MAD-X aborts on a zero-length sequence (measured with cpymad 5.09.03); "
                "use mode='line' for a lattice of thin elements only"
            )
        out = [f"{seq_name}: sequence, l={_num(total)}, refer=centre;"]
        placed = 0
        for it in items:
            el = it.element
            if isinstance(el, Freq) or isinstance(el, Drift):
                continue
            if isinstance(el, Directive):
                out.append(f"  ! lattix directive: {el.card} {' '.join(el.args)}".rstrip())
                continue
            out.append(f"  {defs[id(el)][0]}, at={_num((it.s_in + it.s_out) / 2.0)};")
            placed += 1
        if not placed:
            out.append("  ! lattix: no non-drift element in this sequence")
        out.append("endsequence;")
        return out

    def _line_block(self, seq_name: str, line_names: dict[str, str], lattice: Lattice,
                    defs: dict, rep: FidelityReport) -> list[str]:
        by_name = {el.name: nm for nm, el, _ in defs.values()}
        root = lattice.use or lattice.name
        if root not in lattice.lines:
            rep.dropped("NO_LINE_STRUCTURE",
                        f"mode='line' needs lattice.lines[{root!r}]; wrote a flat line instead")
            flat = ", ".join(by_name[p.element.name] for p in lattice.flatten())
            return [f"{seq_name}: line=({flat});"]

        sup_lines: list[str] = []
        nil: list[str] = []

        def emits_nothing(ref: str) -> bool:
            """Freq and Directive produce no MAD-X element, so they leave the line."""
            return isinstance(lattice.elements.get(ref), (Freq, Directive))

        def nil_marker() -> str:
            if not nil:
                nil.append(f"{seq_name}_nil")
            return nil[0]

        def ref_name(ref: str) -> str:
            if ref in line_names:
                return line_names[ref]
            el = lattice.elements.get(ref)
            if isinstance(el, FieldMap):          # a padded hard edge is a sub-line of its parts
                parts = replacement_for(el).parts
                if len(parts) > 1:
                    nm = f"{ref}_fm".replace("$", "_").lower()
                    body = ", ".join(by_name[q.name] for q in parts if q.name in by_name)
                    sup_lines.append(f"{nm}: line=({body});")
                    return nm
                return by_name.get(parts[0].name, ref)
            if ref in by_name:
                return by_name[ref]
            if isinstance(el, Superposition):     # children were emitted, the parent was not
                nm = ref.replace("$", "_").lower()
                body = ", ".join(by_name[c] for _, c in el.children if c in by_name)
                sup_lines.append(f"{nm}: line=({body});")
                return nm
            return ref

        def item_text(it) -> str:
            txt = ref_name(it.ref)
            if it.reverse:
                txt = f"-{txt}"
            if it.repeat != 1:
                txt = f"{it.repeat}*{txt}"
            return txt

        emitted: list[str] = []
        done: set[str] = set()

        def emit(name: str) -> None:
            if name in done:
                return
            done.add(name)
            for it in lattice.lines[name].items:
                if it.ref in lattice.lines:
                    emit(it.ref)
                    continue
            kept = [it for it in lattice.lines[name].items if not emits_nothing(it.ref)]
            body = ", ".join(item_text(it) for it in kept) or nil_marker()
            emitted.append(f"{line_names.get(name, name)}: line=({body});")

        emit(root)
        head = [f"{nil[0]}: marker;"] if nil else []
        return head + sup_lines + emitted

    @staticmethod
    def _ealign_block(defs: dict) -> list[str]:
        shifted = [(nm, el) for nm, el, _ in defs.values()
                   if el.shift is not None and not el.shift.is_zero()]
        if not shifted:
            return []
        out = [""]
        for nm, el in shifted:
            s = el.shift
            out.append("select, flag=error, clear;")
            out.append(f"select, flag=error, range={nm};")
            out.append(f"ealign, dx={_num(s.x_offset)}, dy={_num(s.y_offset)}, "
                       f"ds={_num(s.z_offset)}, dphi={_num(s.x_rot)}, "
                       f"dtheta={_num(s.y_rot)}, dpsi={_num(s.tilt)};")
        out.append("select, flag=error, clear;")
        return out


_MISSING = set(ALL_KINDS) - set(Writer.RULES)
if _MISSING:                                        # pragma: no cover - guarded by a test too
    raise RuntimeError(f"MAD-X writer RULES do not cover {sorted(_MISSING)}")
