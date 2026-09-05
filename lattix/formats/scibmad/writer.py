"""SciBmad (Beamlines.jl) writer: the IR as a Julia lattice file.

A SciBmad lattice is Julia source::

    using Beamlines
    @elements begin
      qf = Quadrupole(L = 0.5, Kn1 = 0.36)
      d = Drift(L = 1.2)
    end
    fodo = Beamline([qf, d, qf]; pc_ref = 8.5958E+05, species_ref = Species("proton"))

Every convention below was **measured** on SciBmad 0.5.2 / BeamTracking (2026-09-05, docs/oracles.md):

* Strengths are normalized: ``Kn<n>``/``Ks<n>`` on thick magnets, ``Kn<n>L``/``Ks<n>L`` on thin ones,
  ``Ksol`` for solenoids, ``tilt<n>`` per order.  Lab fields (``Bn<n>``) are accepted too, but the
  engine keeps **one reference momentum per beamline** (a ``Patch(dE_ref=…)`` past the first element
  is refused, beamlines cannot nest), so accelerating lattices use the constant-p0 energy modes of
  :mod:`lattix.ir.energy_mode` exactly like MAD-X: ``energy_mode="delta"`` (default) normalizes with
  the momentum the engine's own particle has, rescales kicks, maps and a bend's dipole field alike,
  and moves each downstream cavity's phase by the constant-velocity clock's slip.  The mode travels
  in a ``# lattix: energy_mode=…`` tag that the reader honours.
* ``SBend``: ``g_ref`` is the coordinate curvature only — without a dipole field the element is a
  curved drift — so the design field is written as ``Kn0``; ``e1``/``e2`` are sector-referenced
  (MAD's convention, the IR's), ``edge1_int``/``edge2_int`` hold ``fint·hgap`` (kept in the file;
  BeamTracking 0.5 refuses to track them).
* ``RFCavity``: ``phi0`` is in **radians** with ``zero_phase = PhaseRef.Accelerating`` (the default)
  and the measured gain is ``ΔE = −V·cos(phi0)`` for protons and electrons alike, so the writer
  negates the voltage (:data:`GAIN_SIGN`) to keep ``phi0`` at the IR's synchronous phase (0 = crest,
  gain ``+V cos φ``).  A zero-length cavity is valid in the file; SciBmad 0.5 cannot track it (the
  oracle substitutes 1 µm).  The transverse thin-gap RF kick is carried by a ``LineElement`` lens
  (:func:`lattix.formats.base.with_rf_focusing`).
* ``Kicker``: ``Kn0L = −hkick``, ``Ks0L = +vkick`` (``px += hkick`` measured).  ``Patch`` and the
  alignment keys (``x_offset … tilt``) share the IR's names; apertures are ``x1_limit … y2_limit`` +
  ``aperture_shape`` on any element (a collimator is a drift with limits, as Bmad's own converter
  writes it).
* ``LineElement(transport_map = f)`` with ``f(v, q, p=nothing) -> (v, q)`` carries Taylor maps as
  generated Julia functions (Bmad's converter uses the same shape).
* Julia identifiers are case sensitive; renamed elements get a ``# lattix: name="…" type="…"``
  tag line above their definition, and IR-only attributes (an instrument's family, a foil, a
  reference change, a frequency marker, a cavity's ``n_cell``) travel in the same tag so the reader
  restores them — nothing physical is silent.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    ALL_KINDS,
    Directive,
    Element,
    FieldMap,
    Freq,
    Superposition,
)
from lattix.ir.energy_mode import (
    check_mode,
    mode_ratio,
    phase_slip_turns,
    probe_momentum_ratio,
    record_phase_slip,
    record_rigidity_mode,
    rigidity_for,
    scale_taylor,
    slip_is_zero,
)
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle, Species
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.walk import energy_gain_eV, propagate

#: Measured 2026-09-05 (SciBmad 0.5.2): the energy gain of a ``RFCavity`` is ``−voltage·cos(phi0)``
#: for every species.  The writer multiplies the IR voltage by this so ``phi0`` keeps the IR meaning;
#: if a later SciBmad accelerates at ``phi0 = 0`` the fingerprint test fails and this becomes +1.
GAIN_SIGN = -1.0

_ACCEL_TOL_eV = 1e-6

#: Julia keywords and the Beamlines names an element must not shadow.
RESERVED = frozenset({
    "abstract", "baremodule", "begin", "break", "catch", "const", "continue", "do", "else", "elseif",
    "end", "export", "false", "finally", "for", "function", "global", "if", "import", "in", "isa",
    "let", "local", "macro", "module", "mutable", "primitive", "quote", "return", "struct", "true",
    "try", "type", "using", "where", "while",
    "Drift", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Marker", "Kicker", "HKicker",
    "VKicker", "RFCavity", "CrabCavity", "Patch", "Solenoid", "SBend", "LineElement", "Beamline",
    "Species", "ApertureShape", "ApertureAt", "PhaseRef", "Kind", "pi", "im", "Inf", "NaN", "e",
    "Beamlines", "SciBmad", "BeamTracking", "track!", "Bunch",
})

#: IR species name -> AtomicAndPhysicalConstants name (``#1H-`` is the ¹H anion: m_p + 2 m_e).
SPECIES_NAMES = {"proton": "proton", "antiproton": "anti-proton", "h-": "#1H-", "electron": "electron",
                 "positron": "positron", "deuteron": "deuteron"}


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    if s in ("-0", "-0.0"):
        return "0"
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    return s


def sanitize(name: str) -> str:
    """Make *name* a Julia identifier (case is kept: Julia is case sensitive)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    s = s.strip("_")
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "e_" + s
    if s in RESERVED:
        s += "_x"
    return s


class NameMap:
    """Sanitize + uniquify names, remembering what was renamed (PLAN §4.4)."""

    def __init__(self) -> None:
        self._by_key: dict[int, str] = {}
        self._by_original: dict[str, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}

    def assign(self, original: str, key: object | None = None) -> str:
        cache = id(key) if key is not None else None
        if cache is not None and cache in self._by_key:
            return self._by_key[cache]
        if cache is None and original in self._by_original:
            return self._by_original[original]
        name = self._unique(sanitize(original))
        if cache is not None:
            self._by_key[cache] = name
        self._by_original[original] = name
        if name != original:
            self.renamed[name] = original
        return name

    def reserve(self, name: str) -> str:
        return self._unique(sanitize(name))

    def _unique(self, base: str) -> str:
        out, k = base, 2
        while out in self._used:
            out = f"{base}_{k}"
            k += 1
        self._used.add(out)
        return out


def name_tag(original: str, original_type: str | None = None, **extra: object) -> str:
    """The reversible provenance comment written above a definition."""
    body = f'name="{original}"'
    if original_type:
        body += f' type="{original_type}"'
    for k, v in extra.items():
        if v is None:
            continue
        body += f' {k}="{v}"'
    return f"# lattix: {body}"


def species_expr(sp: Species) -> tuple[str, bool]:
    """``Species("proton")`` for the known names; the eight-argument constructor otherwise
    (the second value says whether the species is one of the named ones)."""
    key = sp.name.lower()
    if key in SPECIES_NAMES:
        return f'Species("{SPECIES_NAMES[key]}")', True
    return (f'Species("{sp.name}", {_num(sp.charge)}, {_num(sp.mass_eV)}, 0.0, 0.0, 0.0, 0, Kind.HADRON)',
            False)


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


@dataclass
class _Item:
    """One placed occurrence: the element, its position, and the design reference there."""

    element: Element
    s_in: float
    s_out: float
    brho: float
    dE: float
    probe: float = 1.0
    ref_in: ReferenceParticle | None = None


@dataclass
class _Def:
    """One Julia definition (shared definitions keep one name)."""

    name: str
    element: Element
    brho: float                      # rigidity the normalized strengths are divided by
    ratio: float                     # Bρ_local / Bρ_used (kicks, maps, a bend's field)
    slip: float                      # phase slip [turns] at a cavity
    extra_lines: list[str] = field(default_factory=list)


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "scibmad"
    RULES: dict[str, Rule] = {
        "Drift": Rule("Drift"),
        "Quadrupole": Rule("Quadrupole"),
        "Sextupole": Rule("Sextupole"),
        "Octupole": Rule("Octupole"),
        "Multipole": Rule("Multipole"),
        "Bend": Rule("SBend"),
        "Solenoid": Rule("Solenoid"),
        "RFCavity": Rule("RFCavity"),
        "FieldMap": Rule("RFCavity/Solenoid/Quadrupole/Drift", "LOSSY", "FM_TO_DRIFT",
                         "field map degraded (see the ladder in lattix.ir.fieldmap)"),
        "NCells": Rule("Drift", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("Drift", "LOSSY", "RFQ_TO_DRIFT", "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("Kicker"),
        "Collimator": Rule("Drift + limits"),
        "Marker": Rule("Marker"),
        "Instrument": Rule("Drift/Marker + tag"),
        "Foil": Rule("Marker + tag", "LOSSY", "FOIL_TO_MARKER",
                     "SciBmad has no foil; the material travels in the tag, the physics is dropped"),
        "Taylor": Rule("LineElement(transport_map)"),
        "Patch": Rule("Patch"),
        "ReferenceChange": Rule("Marker + tag", "EQUIVALENT", "REFCHANGE_AS_TAG",
                                "SciBmad keeps one reference energy per beamline: written as a marker whose "
                                "lattix tag carries the jump (the reader restores it; downstream strengths are "
                                "normalized for it by the energy mode)"),
        "Freq": Rule("Marker + tag", "EXACT", "OK",
                     "the RF clock lives on each cavity's rf_frequency; the marker's tag keeps the FREQ card"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "overlapping fields written as consecutive elements"),
    }
    #: Directive roles a comment can carry without losing physics.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path, *, strict: bool = False, energy_mode: str = "delta",
              line_name: str | None = None, using: str = "Beamlines") -> FidelityReport:
        """Write *lattice* as a SciBmad Julia file.

        ``energy_mode`` is one of :data:`lattix.ir.energy_mode.ENERGY_MODES`; ``using`` is the
        package the file loads (``"Beamlines"`` as Bmad's own converter writes, or ``"SciBmad"``
        for an environment that only has the umbrella package).
        """
        from pathlib import Path

        rep = FidelityReport(target_format="scibmad", target_file=str(path))
        text = self._render(lattice, rep, energy_mode=energy_mode, line_name=line_name, using=using)
        Path(path).write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, **options) -> str:
        rep = FidelityReport(target_format="scibmad")
        return self._render(lattice, rep, **options)

    # ------------------------------------------------------------------
    def _render(self, lattice: Lattice, rep: FidelityReport, *, energy_mode: str = "delta",
                line_name: str | None = None, using: str = "Beamlines") -> str:
        check_mode(energy_mode)
        self._rep = rep
        self._names = NameMap()
        placed = propagate(lattice)
        items = self._items(lattice, placed, rep)
        self._record_energy_mode(items, energy_mode, rep)

        start_brho = lattice.reference.brho_signed
        defs: dict[int, _Def] = {}
        order: list[str] = []                      # beamline entries (Julia names)
        for it in items:
            el = it.element
            key = id(el)
            used = rigidity_for(energy_mode, it.brho, start_brho, it.probe)
            ratio = mode_ratio(energy_mode, it.brho, start_brho, it.probe)
            slip = 0.0
            if energy_mode == "delta" and el.kind == "RFCavity" and it.ref_in is not None:
                slip = phase_slip_turns(it.ref_in, it.s_in, el.length,
                                        el.rf.frequency_Hz or it.ref_in.rf_frequency_Hz, lattice.reference)
                if slip_is_zero(slip):
                    slip = 0.0
            if key in defs:
                d = defs[key]
                if abs(used - d.brho) > 1e-12 * max(1.0, abs(d.brho)):
                    rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                                   "one definition is used at two reference energies; the first occurrence's "
                                   "rigidity is used for its normalized strengths",
                                   element=el.name, kind=el.kind, brho_first=d.brho, brho_here=used)
                if not slip_is_zero(slip - d.slip):
                    rep.equivalent("MULTI_PHASE_SLIP_DEFINITION",
                                   "one cavity definition is used at two arrival times; the first occurrence's "
                                   "phase slip is written", element=el.name, kind=el.kind,
                                   slip_first_turns=d.slip, slip_here_turns=slip)
            else:
                if isinstance(el, (Freq, Directive)) and el.kind == "Directive":
                    pass
                defs[key] = _Def(self._names.assign(el.name, el), el, used, ratio, slip)
            order.append(defs[key].name)

        # definitions (one call per element so every ledger row is recorded exactly once)
        functions: list[str] = []
        body: list[str] = []
        emitted_directive_comments: list[str] = []
        for d in defs.values():
            lines = self._definition(d, lattice, rep, functions)
            if lines is None:                       # a directive: comment only, not in the line
                comment = self._directive_comment(d.element, rep)
                if comment:
                    emitted_directive_comments.append(comment)
                order = [n for n in order if n != d.name]
                continue
            body.extend(lines)

        ref = lattice.reference
        sp_expr, known = species_expr(ref.species)
        if not known:
            rep.lossy("SPECIES_NOT_REPRESENTABLE",
                      f"species {ref.species.name!r} is not one of the named AtomicAndPhysicalConstants "
                      "species; written with the explicit Species constructor (charge, mass) — check that "
                      "`Kind` resolves in your Julia session", element=None, kind=None)
        root = self._names.reserve(line_name or lattice.use or lattice.name or "lattice")
        out: list[str] = [
            f"# lattix {__version__} from {lattice.meta.get('source_format', 'IR')} (SciBmad / Beamlines.jl)",
            f"# lattix: energy_mode={energy_mode}",
            format_reference_tag(ref, "#"),
            f"using {using}",
            "",
        ]
        out += emitted_directive_comments
        if functions:
            out += functions + [""]
        out.append("@elements begin")
        out += ["  " + ln for ln in body]
        out.append("end")
        out.append("")
        out.append(f"{root} = Beamline([" + self._wrap(order) + "];")
        out.append(f"    pc_ref = {_num(ref.pc_eV)}, species_ref = {sp_expr})")
        out.append("")
        return "\n".join(out)

    @staticmethod
    def _wrap(names: list[str], width: int = 96) -> str:
        lines, cur = [], ""
        for n in names:
            piece = (", " if cur else "") + n
            if len(cur) + len(piece) > width and cur:
                lines.append(cur + ",")
                cur = n
            else:
                cur += piece
        lines.append(cur)
        return "\n    ".join(lines)

    # -- flattening ---------------------------------------------------------
    def _items(self, lattice: Lattice, placed: list[Placed], rep: FidelityReport) -> list[_Item]:
        out: list[_Item] = []
        probes = probe_momentum_ratio(placed, lattice.reference)
        for p, probe in zip(placed, probes, strict=True):
            el = p.element
            ref: ReferenceParticle = p.ref_in or lattice.reference
            if isinstance(el, Superposition):
                rep.lossy("SUPERPOSITION_FLATTENED", "overlapping fields written as consecutive elements",
                          element=el.name, kind="Superposition", children=len(el.children))
                for offset, child_name in el.children:
                    child = lattice.elements.get(child_name)
                    if child is None:
                        rep.dropped("SUPERPOSITION_CHILD_MISSING",
                                    f"superposition child {child_name!r} is not defined",
                                    element=el.name, kind="Superposition")
                        continue
                    s0 = p.s_in + offset
                    if isinstance(child, FieldMap):
                        out.extend(self._fieldmap_items(child, s0, ref, rep, probe))
                        continue
                    out.append(_Item(child, s0, s0 + child.length, ref.brho_signed, 0.0, probe, ref))
                continue
            if isinstance(el, FieldMap):
                out.extend(self._fieldmap_items(el, p.s_in, ref, rep, probe))
                continue
            out.append(_Item(el, p.s_in, p.s_out, ref.brho_signed, energy_gain_eV(el, ref), probe, ref))
        return out

    @staticmethod
    def _fieldmap_items(el: FieldMap, s_in: float, ref: ReferenceParticle, rep: FidelityReport,
                        probe: float) -> list[_Item]:
        r = replacement_for(el)
        rep.add(r.cls, r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, message in r.extra:
            rep.add(cls, code, message, element=el.name, kind="FieldMap")
        out, s = [], s_in
        for part in r.parts:
            dE = energy_gain_eV(part, ref)
            out.append(_Item(part, s, s + part.length, ref.brho_signed, dE, probe, ref))
            if dE:
                probe *= ref.advanced(dE_eV=dE).pc_eV / ref.pc_eV
            ref = ref.advanced(dE_eV=dE, ds_m=part.length)
            s += part.length
        return out

    @staticmethod
    def _record_energy_mode(items: list[_Item], energy_mode: str, rep: FidelityReport) -> None:
        for it in items:
            if abs(it.dE) <= _ACCEL_TOL_eV:
                continue
            rep.equivalent("CONST_P0",
                           "SciBmad keeps the reference momentum constant across RF: the reference energy "
                           "does not follow this element's gain",
                           element=it.element.name, kind=it.element.kind, dE_eV=it.dE)
            record_rigidity_mode(rep, energy_mode, element=it.element.name, kind=it.element.kind,
                                 dE_eV=it.dE, brho=it.brho)

    # -- one definition -------------------------------------------------------
    def _definition(self, d: _Def, lattice: Lattice, rep: FidelityReport,
                    functions: list[str]) -> list[str] | None:
        el = d.element
        rule = self.RULES.get(el.kind)
        if rule is None:                                       # pragma: no cover - RULES is total
            raise KeyError(f"SciBmad writer has no rule for kind {el.kind!r}")
        if isinstance(el, Directive):
            return None
        self._d = d
        self._functions = functions
        kind, attrs, extra_tag = getattr(self, f"_def_{el.kind.lower()}")(el, d, rep)
        attrs = list(attrs)
        attrs += self._aperture_attrs(el, rep)
        attrs += self._alignment_attrs(el)
        self._record(el, rule, rep)
        original = (el.provenance.original_name if el.provenance else None) or el.name
        original_type = el.provenance.original_type if el.provenance else None
        lines: list[str] = []
        if d.name != original or extra_tag:
            lines.append(name_tag(original, original_type, **extra_tag))
        joined = ", ".join(attrs)
        lines.append(f"{d.name} = {kind}({joined})")
        return lines

    def _directive_comment(self, el: Directive, rep: FidelityReport) -> str | None:
        args = " ".join(el.args)
        if el.role in self.COMMENT_ROLES:
            rep.exact(el.name, "Directive", code="DIRECTIVE_AS_COMMENT",
                      message=f"role {el.role!r} carries no SciBmad physics; written as a comment")
        else:
            rep.dropped("FOREIGN_DIRECTIVE", "format-specific directive written as a comment only",
                        element=el.name, kind="Directive", card=el.card, role=el.role)
        return f"# lattix directive: {el.card} {args}".rstrip()

    @staticmethod
    def _record(el: Element, rule: Rule, rep: FidelityReport) -> None:
        if rule.cls == "EXACT":
            if not any(e.element == el.name for e in rep.entries):
                rep.exact(el.name, el.kind, message=rule.message)
        elif rule.cls == "EQUIVALENT":
            rep.equivalent(rule.code, rule.message, element=el.name, kind=el.kind, target=rule.target)
        elif rule.cls == "LOSSY":
            rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind, target=rule.target)
        else:
            rep.dropped(rule.code, rule.message, element=el.name, kind=el.kind, target=rule.target)

    # -- shared attribute groups ---------------------------------------------
    @staticmethod
    def _aperture_attrs(el: Element, rep: FidelityReport) -> list[str]:
        ap = el.aperture
        if ap is None or (ap.x_limits is None and ap.y_limits is None):
            return []
        out: list[str] = []
        if ap.x_limits is not None:
            out += [f"x1_limit = {_num(ap.x_limits[0])}", f"x2_limit = {_num(ap.x_limits[1])}"]
        if ap.y_limits is not None:
            out += [f"y1_limit = {_num(ap.y_limits[0])}", f"y2_limit = {_num(ap.y_limits[1])}"]
        shape = "Rectangular" if ap.shape == "RECTANGULAR" else "Elliptical"
        out.append(f"aperture_shape = ApertureShape.{shape}")
        return out

    @staticmethod
    def _alignment_attrs(el: Element) -> list[str]:
        sh = el.shift
        if sh is None:
            return []
        out: list[str] = []
        for key in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt"):
            v = getattr(sh, key, 0.0)
            if v:
                out.append(f"{key} = {_num(v)}")
        return out

    def _multipole_attrs(self, el: Element, d: _Def, *, skip_orders: tuple[int, ...] = ()) -> list[str]:
        """``Kn<n>``/``Ks<n>``/``tilt<n>`` (thick) and the ``…L`` forms (thin), normalized by the
        definition's rigidity."""
        mp = getattr(el, "multipole", None)
        out: list[str] = []
        if mp is None:
            return out
        for order, v in sorted(mp.Bn.items()):
            if v and order not in skip_orders:
                out.append(f"Kn{order} = {_num(v / d.brho)}")
        for order, v in sorted(mp.Bs.items()):
            if v and order not in skip_orders:
                out.append(f"Ks{order} = {_num(v / d.brho)}")
        for order, v in sorted(mp.BnL.items()):
            if v:
                out.append(f"Kn{order}L = {_num(v / d.brho)}")
        for order, v in sorted(mp.BsL.items()):
            if v:
                out.append(f"Ks{order}L = {_num(v / d.brho)}")
        for order, v in sorted(mp.tilt.items()):
            if v:
                out.append(f"tilt{order} = {_num(v)}")
        return out

    # -- per kind ---------------------------------------------------------------
    def _def_drift(self, el, d, rep):
        return "Drift", [f"L = {_num(el.length)}"], {}

    def _def_quadrupole(self, el, d, rep):
        return "Quadrupole", [f"L = {_num(el.length)}", *self._multipole_attrs(el, d)], {}

    def _def_sextupole(self, el, d, rep):
        return "Sextupole", [f"L = {_num(el.length)}", *self._multipole_attrs(el, d)], {}

    def _def_octupole(self, el, d, rep):
        return "Octupole", [f"L = {_num(el.length)}", *self._multipole_attrs(el, d)], {}

    def _def_multipole(self, el, d, rep):
        attrs = [f"L = {_num(el.length)}"] if el.length else []
        return "Multipole", attrs + self._multipole_attrs(el, d), {}

    def _def_bend(self, el, d, rep):
        b = el.bend
        attrs = [f"L = {_num(el.length)}"]
        extra: dict[str, object] = {}
        if b.angle and el.length:
            g = b.angle / el.length
            attrs.append(f"g_ref = {_num(g)}")
            # the design dipole field: g_ref alone is a curved frame (measured); Kn0 = g·r bends the
            # engine's delta-carrying particle by the design angle (r = 1 without acceleration)
            attrs.append(f"Kn0 = {_num(g * d.ratio + el.multipole.Bn.get(0, 0.0) / d.brho)}")
        elif b.angle:
            rep.lossy("THIN_BEND_UNSUPPORTED", "a zero-length bend has no SciBmad form; written as a marker-like "
                      "SBend with no curvature", element=el.name, kind="Bend", angle=b.angle)
        if b.e1:
            attrs.append(f"e1 = {_num(b.e1)}")
        if b.e2:
            attrs.append(f"e2 = {_num(b.e2)}")
        if b.edge_int1 and b.hgap:
            attrs.append(f"edge1_int = {_num(b.edge_int1 * b.hgap)}")
            fintx = b.edge_int1 if b.edge_int2 is None else b.edge_int2
            attrs.append(f"edge2_int = {_num(fintx * b.hgap)}")
            extra["hgap"] = _num(b.hgap)
        elif b.edge_int1 or b.hgap:
            extra["fint"] = _num(b.edge_int1)
            if b.edge_int2 is not None:
                extra["fintx"] = _num(b.edge_int2)
            extra["hgap"] = _num(b.hgap)
        if b.tilt_ref:
            attrs.append(f"tilt_ref = {_num(b.tilt_ref)}")
        if b.rect:
            extra["rect"] = "true"
        if b.fringe_k2 is not None:
            extra["fringe_k2"] = _num(b.fringe_k2)
        attrs += self._multipole_attrs(el, d, skip_orders=(0,))
        return "SBend", attrs, extra

    def _def_solenoid(self, el, d, rep):
        attrs = [f"L = {_num(el.length)}", f"Ksol = {_num(el.solenoid.Bsol_T / d.brho)}"]
        return "Solenoid", attrs + self._multipole_attrs(el, d), {}

    def _def_rfcavity(self, el, d, rep):
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        attrs = [f"L = {_num(el.length)}", f"voltage = {_num(GAIN_SIGN * volt)}",
                 f"phi0 = {_num(rf.phase_rad - 2.0 * math.pi * d.slip)}"]
        if not slip_is_zero(d.slip):
            record_phase_slip(rep, element=el.name, kind=el.kind, slip_turns=d.slip, engine="SciBmad",
                              attribute="phi0")
        if el.length and volt:
            rep.equivalent("THICK_CAVITY_RF_FOCUSING",
                           "SciBmad's RFCavity applies its own transverse RF focusing along its length "
                           "(measured: R21 = 1.04e-3 for a 1 mm, 300 kV cavity at 2.1 MeV); codes with a pure "
                           "longitudinal kick at the cavity centre (MAD-X, xtrack, Elegant RFCA) have none",
                           element=el.name, kind=el.kind, length=el.length, voltage_V=volt)
        if rf.frequency_Hz:
            attrs.append(f"rf_frequency = {_num(rf.frequency_Hz)}")
        elif rf.harmon:
            attrs.append(f"harmon = {_num(rf.harmon)}")
        if rf.cavity_type == "TRAVELING_WAVE":
            attrs.append("traveling_wave = true")
        extra: dict[str, object] = {}
        if rf.n_cell is not None:
            extra["n_cell"] = int(rf.n_cell)
        if rf.L_active_m is not None:
            extra["L_active_m"] = _num(rf.L_active_m)
        if rf.dE_ref_eV is not None:
            extra["dE_ref_eV"] = _num(rf.dE_ref_eV)
        if not rf.phase_is_sync:
            extra["phase_is_sync"] = "false"
        return "RFCavity", attrs, extra

    def _def_fieldmap(self, el, d, rep):                        # pragma: no cover - expanded in _items
        return "Drift", [f"L = {_num(el.length)}"], {}

    def _def_ncells(self, el, d, rep):
        return "Drift", [f"L = {_num(el.length)}"], {}          # the physics is gone: it is a drift now

    _def_rfqcell = _def_ncells

    def _def_kicker(self, el, d, rep):
        if el.electric:
            rep.lossy("EKICK_AS_MAGNETIC", "electric steerer written as a magnetic SciBmad kicker",
                      element=el.name, kind="Kicker")
        attrs = [f"L = {_num(el.length)}"] if el.length else []
        if el.hkick:
            attrs.append(f"Kn0L = {_num(-el.hkick * d.ratio)}")
        if el.vkick:
            attrs.append(f"Ks0L = {_num(el.vkick * d.ratio)}")
        return "Kicker", attrs, ({"electric": "true"} if el.electric else {})

    def _def_collimator(self, el, d, rep):
        attrs = [f"L = {_num(el.length)}"] if el.length else []
        kind = "Drift" if el.length else "Marker"
        if el.aperture is None or (el.aperture.x_limits is None and el.aperture.y_limits is None):
            rep.equivalent("COLLIMATOR_NO_APERTURE", "collimator without limits written as a plain "
                           f"{kind.lower()}", element=el.name, kind="Collimator")
        return kind, attrs, {"kind": "Collimator"}

    def _def_marker(self, el, d, rep):
        return "Marker", [], {}

    def _def_instrument(self, el, d, rep):
        kind = "Drift" if el.length else "Marker"
        attrs = [f"L = {_num(el.length)}"] if el.length else []
        return kind, attrs, {"kind": "Instrument", "family": el.family}

    def _def_foil(self, el, d, rep):
        extra = {"kind": "Foil", "material": el.material, "thickness_kg_per_m2": _num(el.thickness_kg_per_m2)}
        if el.dE_ref_eV is not None:
            extra["dE_ref_eV"] = _num(el.dE_ref_eV)
        if el.length:
            return "Drift", [f"L = {_num(el.length)}"], extra
        return "Marker", [], extra

    def _def_taylor(self, el, d, rep):
        matrix, offset = scale_taylor(el.matrix, el.offset, d.ratio)
        fn = f"lattix_map_{d.name}"
        lines = [f"function {fn}(v, q, p=nothing)"]
        for i in range(6):
            terms = [f"{_num(matrix[i][j])}*v[{j + 1}]" for j in range(6) if matrix[i][j]]
            if offset[i]:
                terms.append(_num(offset[i]))
            lines.append(f"    v{i + 1} = " + (" + ".join(terms) if terms else "0.0"))
        lines.append("    return (v1, v2, v3, v4, v5, v6), q")
        lines.append("end")
        self._functions.append("\n".join(lines))
        attrs = [f"L = {_num(el.length)}", f"transport_map = {fn}"]
        extra: dict[str, object] = {}
        if el.basis and el.basis not in ("common", "bmad"):
            rep.equivalent("TAYLOR_BASIS_SCIBMAD",
                           f"the map's source basis {el.basis!r} is re-used verbatim in SciBmad's "
                           "(x, px, y, py, z, pz) coordinates", element=el.name, kind="Taylor")
            extra["basis"] = el.basis
        return "LineElement", attrs, extra

    def _def_patch(self, el, d, rep):
        attrs: list[str] = []
        if el.length:
            attrs.append(f"L = {_num(el.length)}")
        for ir_key, key in (("x_offset", "dx"), ("y_offset", "dy"), ("z_offset", "dz"),
                            ("x_rot", "dx_rot"), ("y_rot", "dy_rot"), ("tilt", "dz_rot")):
            v = getattr(el, ir_key, 0.0)
            if v:
                attrs.append(f"{key} = {_num(v)}")
        if el.t_offset_s:
            attrs.append(f"dt = {_num(el.t_offset_s)}")
        extra: dict[str, object] = {}
        if el.e_tot_offset_eV:
            rep.equivalent("PATCH_ENERGY_OFFSET_AS_TAG",
                           "SciBmad keeps one reference energy per beamline: the patch's energy offset "
                           "travels in the tag", element=el.name, kind="Patch", e_tot_offset_eV=el.e_tot_offset_eV)
            extra["e_tot_offset_eV"] = _num(el.e_tot_offset_eV)
        return "Patch", attrs, extra

    def _def_referencechange(self, el, d, rep):
        extra: dict[str, object] = {"kind": "ReferenceChange"}
        for f in ("dE_ref_eV", "energy_eV", "dtime_s", "dphase_rad"):
            v = getattr(el, f)
            if v is not None:
                extra[f] = f"{v:.17g}"
        return "Marker", [], extra

    def _def_freq(self, el, d, rep):
        return "Marker", [], {"kind": "Freq", "frequency_Hz": _num(el.frequency_Hz)}

    def _def_directive(self, el, d, rep):                       # pragma: no cover - handled earlier
        return "Marker", [], {}

    def _def_superposition(self, el, d, rep):                   # pragma: no cover - expanded in _items
        return "Marker", [], {}


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert set(Writer.RULES) == set(ALL_KINDS), set(ALL_KINDS) ^ set(Writer.RULES)
