"""FLAME GLPS writer (PLAN §6 task 3.4).

Emits a deck FLAME's own ``GLPSParser``/``Machine`` accepts:

* a global block built from ``lattice.reference`` — FLAME is **per nucleon**, so
  ``IonEs = mass/A``, ``IonEk = E_kin/A`` [eV/u] and ``IonChargeStates = [Q/A]``
  (``src/flame/moment.h:47-57``).  ``A`` comes from ``meta["flame"]["mass_number"]``
  when the lattice came from FLAME and is 1 otherwise;
* a ``source`` element first — ``Machine`` needs one to seed the state, together
  with the ``vector_variable``/``matrix_variable`` arrays it names;
* one definition per element and ``name: LINE = (…);`` for every IR line, keeping the
  hierarchy (``n*line`` repeats and ``-line`` reverses, ``src/glps_ops.cpp:236-238``);
  a lattice whose lines cannot be expressed that way falls back to one flat line;
* a final ``USE:``.

Rules that are *not* symmetric with the reader:

======================  ====================================================
IR                      FLAME
======================  ====================================================
``Quadrupole`` skew     a body ``roll`` of ``−atan2(Bs, Bn)/(n+1)`` — FLAME's
                        ``roll`` *is* MAD-X's ``tilt`` (measured) — EQUIVALENT
                        ``SKEW_AS_ROLL``
``Octupole``            no such element → ``drift`` + LOSSY
``Multipole`` (thin)    dipole content → ``orbtrim``; anything higher →
                        ``marker`` + LOSSY
``RFCavity``            ``rfcavity`` only when a ``cavtype`` is known (FLAME
                        takes its field from a tabulated model, never a bare
                        voltage) — otherwise ``drift`` + LOSSY
                        ``RFCAVITY_NEEDS_CAVTYPE``
``FieldMap``            ``drift`` + LOSSY ``FM_TO_DRIFT``
``Bend``                ``sbend``; ``K = Bn[1]/Bρ`` is normalized like MAD-X's
                        ``k1``, ``ver = 1`` for ``|tilt_ref| ≈ π/2``
======================  ====================================================

Numbers are ``%.15g`` but re-spelled for FLAME's lexer, which has **no leading
``.``** and no signed exponent-less form (``src/glps.l``): ``.5`` is written
``0.5``.  Names are FLAME identifiers (``[A-Za-z][A-Za-z0-9_:]*[A-Za-z0-9_]``),
**case sensitive** and unique, with a ``# lattix: name="…" type="…"`` comment
above any definition that had to be renamed (invariant I-15).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import note_quad_higher_orders
from lattix.formats.flame.reader import SAMPLE_FREQ_DEFAULT, AMU_eV
from lattix.ir.elements import ALL_KINDS, Element, RFCavity
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.normalize import k1_from_gradient, kick_from_bl
from lattix.ir.reference import ReferenceParticle
from lattix.ir.walk import propagate

#: FLAME identifier (``src/glps.l``): a letter, then letters/digits/``_``/``:``,
#: ending on a letter, digit or ``_``.
NAME_RE = re.compile(r"^[A-Za-z](?:[A-Za-z0-9_:]*[A-Za-z0-9_])?$")

#: names that would change the meaning of a statement.  ``USE`` is the magic
#: element FLAME looks for (``src/config.cpp:283``), ``END`` the only command,
#: ``print`` the only global function, ``LINE`` the only line keyword.
RESERVED: frozenset[str] = frozenset({"USE", "END", "LINE", "line", "Line", "print"})

#: FLAME cavity models with data shipped in ``flame/data`` (``sphinx_doc/element.rst``).
KNOWN_CAVTYPES: frozenset[str] = frozenset({"Generic", "0.041QWR", "0.085QWR",
                                            "0.29HWR", "0.53HWR"})

#: IR ``Instrument.family`` values FLAME has a real element for.
BPM_FAMILIES = frozenset({"BPM", "MONITOR", "HMONITOR", "VMONITOR"})

#: default beam-envelope seed (mm², rad², … the 7th row/column is the augmentation).
DEFAULT_ENVELOPE = (1.0, 1e-6, 1.0, 1e-6, 1.0, 1e-6, 0.0)

#: FLAME element types with no ``L`` parameter (``src/moment.cpp``): an IR element
#: of non-zero length mapped onto one of these gets a trailing drift so ``Σ length``
#: survives (invariant I-1).
THIN_TYPES: frozenset[str] = frozenset({"marker", "bpm", "source", "orbtrim",
                                        "stripper", "tmatrix"})

#: a tilt this close to ±π/2 is a vertical bend (FLAME ``ver = 1``).
_VERTICAL_TOL = 1e-9


def fmt(x: float) -> str:
    r"""A GLPS number literal: ``%.15g`` when that reads back exactly, else ``repr``.

    ``%.15g`` is the house format (PLAN §6 task 1.3) but it is *not* round-trip
    exact — ``IonChargeStates = 33/238`` needs 17 significant digits, and writing 15
    moved the whole ``ALL_lattice.lat`` line by 7.4e-11 in the per-element maps
    (measured; 1.1e-13 with the exact literal).  So the short form is used whenever it
    round-trips and ``repr`` — the shortest exact decimal — otherwise.

    The result is also re-spelled for FLAME's lexer, which has no leading ``.``
    (``src/glps.l``: ``[0-9]+(\.[0-9]*)?…``) and no inf/nan.
    """
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"FLAME decks cannot hold {v!r}")
    s = f"{v:.15g}"
    if float(s) != v:
        s = repr(v)
    if s in ("-0", "-0.0"):
        return "0"
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    return s


def sanitize(name: str) -> str:
    """Make *name* a FLAME identifier (case is preserved — FLAME is case sensitive)."""
    s = re.sub(r"[^A-Za-z0-9_:]", "_", (name or "").strip())
    s = s.lstrip("_:")
    if not s or not s[0].isalpha():
        s = "e_" + s
    s = s.rstrip("_:") or "e"
    if not s[-1].isalnum() and s[-1] != "_":
        s += "e"
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


def name_tag(original: str, original_type: str | None = None) -> str:
    body = f'name="{original}"'
    if original_type:
        body += f' type="{original_type}"'
    return f"# lattix: {body}"


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


@dataclass
class _Def:
    """One FLAME element definition."""

    name: str
    type: str
    attrs: list[tuple[str, str]]
    tag: str | None = None

    def render(self) -> list[str]:
        head = f"{self.name}: {self.type}"
        body = "".join(f", {k} = {v}" for k, v in self.attrs)
        return ([self.tag] if self.tag else []) + [f"{head}{body};"]


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "flame"

    RULES: dict[str, Rule] = {
        "Drift": Rule("drift"),
        "Quadrupole": Rule("quadrupole"),
        "Sextupole": Rule("sextupole"),
        "Octupole": Rule("drift", "LOSSY", "OCTUPOLE_TO_DRIFT",
                         "FLAME has no octupole; written as a drift of the same length"),
        "Multipole": Rule("orbtrim/marker", "LOSSY", "MULTIPOLE_TO_ORBTRIM",
                          "FLAME has no thin multipole; the dipole term becomes an orbtrim "
                          "and higher orders are dropped"),
        "Bend": Rule("sbend"),
        "Solenoid": Rule("solenoid"),
        "RFCavity": Rule("rfcavity"),
        "FieldMap": Rule("drift", "LOSSY", "FM_TO_DRIFT",
                         "FLAME has no field-map element; written as a drift of the same length"),
        "NCells": Rule("drift", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("drift", "LOSSY", "RFQ_TO_DRIFT",
                        "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("orbtrim"),
        "Collimator": Rule("marker", "LOSSY", "COLLIMATOR_TO_MARKER",
                           "FLAME does not collimate; 'aper' is written but its own code "
                           "never reads it"),
        "Marker": Rule("marker"),
        "Instrument": Rule("bpm/marker"),
        "Foil": Rule("stripper", "EQUIVALENT", "FOIL_AS_STRIPPER",
                     "FLAME's stripper redistributes charge states with the Baron formula; "
                     "the IR Foil carries no such model"),
        "Taylor": Rule("tmatrix"),
        "Patch": Rule("marker", "LOSSY", "PATCH_DROPPED",
                      "FLAME has no patch element; written as a marker"),
        "ReferenceChange": Rule("marker", "LOSSY", "REFCHANGE_DROPPED",
                                "FLAME cannot change the reference energy outside a cavity or "
                                "stripper; written as a marker"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each rfcavity's 'f'; FLAME's global SampleFreq "
                     "comes from the lattice reference"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "overlapping fields written as consecutive elements"),
    }

    #: Directive roles a comment can carry without losing physics.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              energy_mode: str = "constant", line_name: str | None = None,
              envelope: tuple[float, ...] | None = None,
              eng_data_dir: str | None = None, header: bool = True) -> FidelityReport:
        """Write *lattice* as a GLPS deck.

        ``energy_mode`` picks the rigidity that normalizes a bend's ``K``
        (``"constant"`` = the lattice start, ``"local"`` = each element's entrance);
        ``eng_data_dir`` sets ``Eng_Data_Dir`` for the cavity tables.
        """
        if energy_mode not in ("local", "constant"):
            raise ValueError(f"energy_mode must be 'local' or 'constant', got {energy_mode!r}")
        rep = FidelityReport(target_format="flame", target_file=str(path))
        self._rep = rep
        self._names = NameMap()
        self._energy_mode = energy_mode

        placed = propagate(lattice)
        text = self._render(lattice, placed, rep, line_name=line_name, envelope=envelope,
                            eng_data_dir=eng_data_dir, header=header)
        Path(path).write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, **options) -> str:
        """The deck as a string (used by the goldens and the oracle)."""
        rep = FidelityReport(target_format="flame")
        self._rep = rep
        self._names = NameMap()
        self._energy_mode = options.pop("energy_mode", "constant")
        placed = propagate(lattice)
        return self._render(lattice, placed, rep, **options)

    # ------------------------------------------------------------------
    def _render(self, lat: Lattice, placed: list[Placed], rep: FidelityReport, *,
                line_name: str | None = None, envelope: tuple[float, ...] | None = None,
                eng_data_dir: str | None = None, header: bool = True) -> str:
        meta = lat.meta.get("flame") or {}
        out: list[str] = []
        if header:
            out += [f"# written by lattix {__version__} (FLAME GLPS)", ""]
        out += self._globals(lat, placed, meta, rep, eng_data_dir)

        # -- source element (FLAME needs one to seed the state) ------------
        first = placed[0].element if placed else None
        src_native = (first.native.get("flame") or {}) if first is not None else {}
        reuse_source = src_native.get("type") == "source"
        vec_var, mat_var = "BaryCenter", "S"
        if reuse_source:
            attrs = src_native.get("attrs") or {}
            vec_var = str(attrs.get("vector_variable", vec_var))
            mat_var = str(attrs.get("matrix_variable", mat_var))
        env = tuple(envelope) if envelope is not None else DEFAULT_ENVELOPE
        n_states = len(meta.get("globals", {}).get("IonChargeStates") or [None])
        out += self._beam_arrays(vec_var, mat_var, env, n_states,
                                 meta.get("beam_vectors") or {})
        out.append("")

        # -- element definitions ------------------------------------------
        defs: list[_Def] = []
        emitted: dict[int, list[str]] = {}          # id(element) -> emitted names
        source_name = None
        for i, p in enumerate(placed):
            el = p.element
            if i == 0 and reuse_source:
                source_name = self._names.assign(el.name, el)
                defs.append(_Def(source_name, "source",
                                 [("vector_variable", f'"{vec_var}"'),
                                  ("matrix_variable", f'"{mat_var}"')]))
                emitted[id(el)] = [source_name]
                rep.exact(el.name, "Marker", "OK")
                continue
            if id(el) in emitted:                    # a definition reused by the line
                continue
            names = self._emit(el, p, lat, defs, rep)
            emitted[id(el)] = names
        if source_name is None:
            source_name = self._names.reserve("S_lattix")
            defs.insert(0, _Def(source_name, "source",
                                [("vector_variable", f'"{vec_var}"'),
                                 ("matrix_variable", f'"{mat_var}"')]))
            rep.equivalent("FLAME_SOURCE_ADDED",
                           "FLAME needs a 'source' element to seed the beam state; one was "
                           "prepended with a unit envelope")
        for d in defs:
            out += d.render()
        out.append("")

        # -- lines ----------------------------------------------------------
        by_name = {name: emitted.get(id(el), []) for name, el in lat.elements.items()}
        nested = self._nested_lines(lat, by_name, source_name, reuse_source, line_name, rep)
        if nested is not None:
            out += nested
        else:
            root = self._names.reserve(line_name or lat.use or lat.name or "cell")
            members = [source_name]
            for p in placed:
                if p.index == 0 and reuse_source:
                    continue
                members += emitted.get(id(p.element), [])
            out += _render_line(root, members)
            out.append("")
            out.append(f"USE: {root};")
        return "\n".join(out) + "\n"

    def _nested_lines(self, lat: Lattice, by_name: dict[str, list[str]], source_name: str,
                      reuse_source: bool, line_name: str | None,
                      rep: FidelityReport) -> list[str] | None:
        """Re-emit the IR's line hierarchy (``glps_ops.cpp`` supports ``n*line``/``-line``).

        Returns ``None`` when the hierarchy cannot be expressed — a line referring to an
        element the used line never places (so it has no rigidity and was not written), or
        an element that expanded to something other than one definition inside a sub-line.
        The caller then writes one flat line.
        """
        root = lat.use
        if not root or root not in lat.lines or line_name:
            return None
        order = _line_order(lat, root)
        if order is None:
            return None
        line_names = {name: self._names.reserve(name) for name in order}
        out: list[str] = []
        for name in order:
            members: list[str] = []
            if name == root and not reuse_source:
                members.append(source_name)
            for it in lat.lines[name].items:
                if it.ref in lat.lines:
                    token = line_names[it.ref]
                    if it.repeat != 1:
                        token = f"{it.repeat}*{token}"
                    if it.reverse:
                        token = f"-{token}"
                    members.append(token)
                    continue
                got = by_name.get(it.ref)
                if not got and it.ref in lat.elements and lat.elements[it.ref].kind in (
                        "Freq", "Directive"):
                    continue                     # legitimately writes nothing
                if got is None or (it.reverse and len(got) != 1):
                    return None
                if it.reverse:
                    rep.lossy("FLAME_ELEMENT_REVERSED",
                              f"element {it.ref!r} is placed reversed; FLAME has no element "
                              "reversal, the forward definition is used",
                              element=it.ref)
                members += got * max(1, it.repeat)
            if name == root and reuse_source and members and members[0] != source_name:
                members.insert(0, source_name)
            out += _render_line(line_names[name], members)
            out.append("")
        out.append(f"USE: {line_names[root]};")
        return out

    # -- header ------------------------------------------------------------
    def _globals(self, lat: Lattice, placed: list[Placed], meta: dict, rep: FidelityReport,
                 eng_data_dir: str | None) -> list[str]:
        ref = lat.reference
        sp = ref.species
        mass_number = int(meta.get("mass_number") or 0)
        if mass_number <= 0:
            mass_number = 1
        q_over_a = sp.charge / mass_number
        states = meta.get("globals", {}).get("IonChargeStates")
        if not isinstance(states, list) or not states:
            states = [q_over_a]
        ncharge = meta.get("globals", {}).get("NCharge")
        if not isinstance(ncharge, list) or len(ncharge) != len(states):
            ncharge = [1.0] * len(states)

        g = meta.get("globals", {})
        out = [f'sim_type = "{g.get("sim_type", "MomentMatrix")}";']
        for key in ("MpoleLevel", "EmitGrowth", "HdipoleFitMode"):
            if key in g:
                out.append(f'{key} = "{g[key]}";')
        out += [
            f"IonEs = {fmt(sp.mass_eV / mass_number)};",
            f"IonEk = {fmt(ref.kinetic_energy_eV / mass_number)};",
            f"IonZ = {fmt(q_over_a)};",
            "IonChargeStates = [" + ", ".join(fmt(float(s)) for s in states) + "];",
            "NCharge = [" + ", ".join(fmt(float(c)) for c in ncharge) + "];",
        ]
        freq = ref.rf_frequency_Hz or SAMPLE_FREQ_DEFAULT
        out.append(f"SampleFreq = {fmt(freq)};")
        if mass_number == 1 and abs(sp.mass_eV - AMU_eV) > 1e3 and sp.charge not in (0,):
            rep.equivalent("FLAME_PER_NUCLEON",
                           f"species {sp.name!r} has no mass number in this lattice; written as "
                           f"A = 1 (IonEs = {sp.mass_eV:.9g} eV/u, IonZ = {q_over_a:g})",
                           mass_eV=sp.mass_eV, charge=sp.charge)
        data_dir = eng_data_dir if eng_data_dir is not None else g.get("Eng_Data_Dir")
        needs = any(isinstance(p.element, RFCavity) for p in placed)
        if needs and data_dir:
            out.append(f'Eng_Data_Dir = dir("{data_dir}");')
        elif needs and not data_dir:
            rep.lossy("FLAME_NO_ENG_DATA_DIR",
                      "the deck has rfcavity elements but no Eng_Data_Dir; FLAME will look for "
                      "the cavity tables in its built-in data directory")
        # anything else the source deck put in the global block (Stripper_*, cstate, …):
        # the stripper element needs Stripper_IonChargeStates/Stripper_NCharge to run
        written = {"sim_type", "MpoleLevel", "EmitGrowth", "HdipoleFitMode", "IonEs", "IonEk",
                   "IonZ", "IonChargeStates", "NCharge", "SampleFreq", "Eng_Data_Dir",
                   "IonW", "AMU"}
        for key, value in g.items():
            if key in written:
                continue
            out.append(f"{key} = {_render_value(value)};")
        out.append("")
        return out

    @staticmethod
    def _beam_arrays(vec_var: str, mat_var: str, env: tuple[float, ...], n_states: int,
                     stored: dict | None = None) -> list[str]:
        """``{vec}{k}`` (7) and ``{mat}{k}`` (49) arrays the ``source`` element names.

        FLAME seeds each charge state's *real* particle from these — in particular
        ``real[k].IonEk += moment0[k][PS_PS]`` (``src/moment.cpp:172``), so dropping the
        deck's own centroid changes every downstream cavity map by that energy offset.
        A stored array is therefore re-emitted verbatim; the diagonal ``env`` is only
        a fallback for lattices that did not come from FLAME.
        """
        stored = stored or {}
        out: list[str] = []
        for k in range(max(1, n_states)):
            vec = stored.get(f"{vec_var}{k}")
            if isinstance(vec, list) and len(vec) == 7:
                out.append(f"{vec_var}{k} = [" + ", ".join(fmt(float(v)) for v in vec) + "];")
            else:
                out.append(f"{vec_var}{k} = [" + ", ".join(["0"] * 6) + ", 1];")
        diag = list(env) + [0.0] * (7 - len(env))
        default = [", ".join(fmt(diag[i] if i == j else 0.0) for j in range(7)) for i in range(7)]
        for k in range(max(1, n_states)):
            mat = stored.get(f"{mat_var}{k}")
            if isinstance(mat, list) and len(mat) == 49:
                rows = [", ".join(fmt(float(v)) for v in mat[7 * i:7 * i + 7]) for i in range(7)]
            else:
                rows = default
            out.append(f"{mat_var}{k} = [")
            out += [f"  {r}," for r in rows[:-1]]
            out.append(f"  {rows[-1]}")
            out.append("];")
        return out

    # -- per-element dispatch ----------------------------------------------
    def _emit(self, el: Element, p: Placed, lat: Lattice, defs: list[_Def],
              rep: FidelityReport) -> list[str]:
        rule = self.RULES[el.kind]
        name = self._names.assign(el.name, el)
        native = el.native.get("flame") or {}

        handler = getattr(self, f"_emit_{el.kind.lower()}")
        type_, attrs, owned = handler(el, p, lat, rep, rule)
        if type_ is None:                                   # nothing to emit (Freq, Directive)
            self._record(rep, el, rule)
            return []

        # the tag is only worth a line when something is not recoverable from the deck
        # itself: a renamed element, or a source type FLAME cannot spell (invariant I-15)
        original_type = el.provenance.original_type if el.provenance else None
        tag = None
        if name != el.name or (original_type and original_type != type_):
            tag = name_tag(el.name, original_type)

        attrs += self._common_attrs(el, owned)
        attrs += self._leftover(native, owned)
        defs.append(_Def(name, type_, attrs, tag))
        names = [name]
        if type_ in THIN_TYPES and el.length:
            pad = self._names.reserve(f"{name}_L")
            defs.append(_Def(pad, "drift", [("L", fmt(el.length))]))
            names.append(pad)
            rep.lossy("THIN_TYPE_LENGTH_PADDED",
                      f"FLAME's {type_!r} has no length; {el.length:g} m written as the "
                      f"following drift {pad!r}",
                      element=el.name, kind=el.kind, length=el.length)
        self._record(rep, el, rule)
        return names

    @staticmethod
    def _record(rep: FidelityReport, el: Element, rule: Rule) -> None:
        if rule.cls == "EXACT":
            rep.exact(el.name, el.kind, rule.code)
        else:
            rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind)

    def _common_attrs(self, el: Element, owned: set[str]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        shift = el.shift
        if shift is not None and not shift.is_zero():
            for key, value in (("dx", shift.x_offset), ("dy", shift.y_offset),
                               ("pitch", shift.x_rot), ("yaw", shift.y_rot),
                               ("roll", shift.tilt + _extra_roll(el))):
                if value and key not in owned:
                    out.append((key, fmt(value)))
                    owned.add(key)
        else:
            roll = _extra_roll(el)
            if roll and "roll" not in owned:
                out.append(("roll", fmt(roll)))
                owned.add("roll")
        ap = el.aperture
        if ap is not None and "aper" not in owned:
            radius = _aperture_radius(ap)
            if radius:
                out.append(("aper", fmt(radius)))
                owned.add("aper")
        return out

    @staticmethod
    def _leftover(native: dict, owned: set[str]) -> list[tuple[str, str]]:
        """Re-emit FLAME attributes the IR has no home for (edipole/equad bodies,
        stripper parameters, cavity data files …)."""
        out: list[tuple[str, str]] = []
        for key, value in (native.get("attrs") or {}).items():
            if key in owned:
                continue
            out.append((key, _render_value(value)))
        return out

    # -- handlers -----------------------------------------------------------
    def _emit_drift(self, el, p, lat, rep, rule):
        return "drift", [("L", fmt(el.length))], {"L"}

    def _emit_marker(self, el, p, lat, rep, rule):
        native = (el.native.get("flame") or {}).get("type")
        if native in ("edipole", "equad"):        # re-emit an electrostatic body verbatim
            return native, [("L", fmt(el.length))], {"L"}
        if native == "source":                    # a source that is not the first element
            return "marker", [], set()
        return "marker", [], set()

    def _emit_instrument(self, el, p, lat, rep, rule):
        if (el.family or "").upper() in BPM_FAMILIES:
            return "bpm", [], set()
        return "marker", [], set()

    def _emit_quadrupole(self, el, p, lat, rep, rule):
        note_quad_higher_orders(el, rep, "FLAME")
        attrs = [("L", fmt(el.length)), ("B2", fmt(el.multipole.Bn.get(1, 0.0)))]
        owned = {"L", "B2"}
        if el.multipole.Bs.get(1):
            rep.equivalent("SKEW_AS_ROLL",
                           "FLAME has no skew quadrupole; the skew gradient is written as a "
                           "'roll' misalignment of the normal one",
                           element=el.name, kind=el.kind)
        return "quadrupole", attrs, owned

    def _emit_sextupole(self, el, p, lat, rep, rule):
        return "sextupole", [("L", fmt(el.length)),
                             ("B3", fmt(el.multipole.Bn.get(2, 0.0)))], {"L", "B3"}

    def _emit_octupole(self, el, p, lat, rep, rule):
        return "drift", [("L", fmt(el.length))], {"L"}

    def _emit_solenoid(self, el, p, lat, rep, rule):
        return "solenoid", [("L", fmt(el.length)),
                            ("B", fmt(el.solenoid.Bsol_T))], {"L", "B"}

    def _emit_bend(self, el, p, lat, rep, rule):
        b = el.bend
        attrs = [("L", fmt(el.length)), ("phi", fmt(math.degrees(b.angle))),
                 ("phi1", fmt(math.degrees(b.e1))), ("phi2", fmt(math.degrees(b.e2)))]
        owned = {"L", "phi", "phi1", "phi2", "K", "ver", "bg"}
        grad = el.multipole.Bn.get(1, 0.0)
        if grad:
            attrs.append(("K", fmt(k1_from_gradient(grad, self._ref(p, lat)))))
        if abs(abs(b.tilt_ref) - math.pi / 2) < _VERTICAL_TOL:
            attrs.append(("ver", "1"))
        elif b.tilt_ref:
            rep.lossy("BEND_TILT_DROPPED",
                      f"FLAME sbend only bends horizontally (ver = 0) or vertically (ver = 1); "
                      f"tilt_ref = {b.tilt_ref:g} rad cannot be written",
                      element=el.name, kind=el.kind, tilt_ref=b.tilt_ref)
        bg = el.meta.get("flame_bg")
        if bg is not None:
            attrs.append(("bg", fmt(bg)))
        if b.edge_int1 or b.hgap:
            rep.lossy("BEND_FRINGE_DROPPED",
                      "FLAME's sbend edge model takes only pole-face angles; fint/hgap are lost",
                      element=el.name, kind=el.kind, fint=b.edge_int1, hgap=b.hgap)
        return "sbend", attrs, owned

    def _emit_kicker(self, el, p, lat, rep, rule):
        return "orbtrim", [("theta_x", fmt(el.hkick)),
                           ("theta_y", fmt(el.vkick))], {"theta_x", "theta_y", "realpara",
                                                         "tm_xkick", "tm_ykick"}

    def _emit_multipole(self, el, p, lat, rep, rule):
        ref = self._ref(p, lat)
        hk = -kick_from_bl(el.multipole.BnL.get(0, 0.0), ref)
        vk = kick_from_bl(el.multipole.BsL.get(0, 0.0), ref)
        higher = any(v for n, v in el.multipole.BnL.items() if n) or \
            any(v for n, v in el.multipole.BsL.items() if n)
        if higher:
            rep.lossy("MULTIPOLE_ORDERS_DROPPED",
                      "FLAME has no thin multipole; orders above the dipole are dropped",
                      element=el.name, kind=el.kind)
        if hk or vk:
            return "orbtrim", [("theta_x", fmt(hk)), ("theta_y", fmt(vk))], {"theta_x", "theta_y"}
        return "marker", [], set()

    def _emit_rfcavity(self, el, p, lat, rep, rule):
        cavtype = el.meta.get("flame_cavtype") or \
            (el.native.get("flame") or {}).get("attrs", {}).get("cavtype")
        if not cavtype:
            rep.lossy("RFCAVITY_NEEDS_CAVTYPE",
                      f"FLAME takes a cavity's field from a tabulated model, never a bare "
                      f"voltage; {el.name!r} has no 'cavtype' so it is written as a "
                      f"{el.length:g} m drift",
                      element=el.name, kind=el.kind, voltage_V=el.rf.voltage_V,
                      frequency_Hz=el.rf.frequency_Hz)
            return "drift", [("L", fmt(el.length))], {"L"}
        if str(cavtype) not in KNOWN_CAVTYPES:
            rep.equivalent("FLAME_UNKNOWN_CAVTYPE",
                           f"cavtype {cavtype!r} is not one of {sorted(KNOWN_CAVTYPES)}; FLAME "
                           "will need a matching data file",
                           element=el.name, kind=el.kind)
        attrs = [("L", fmt(el.length)), ("cavtype", f'"{cavtype}"')]
        owned = {"L", "cavtype", "f", "phi", "scl_fac", "syncflag"}
        if el.rf.frequency_Hz:
            attrs.append(("f", fmt(el.rf.frequency_Hz)))
        attrs.append(("phi", fmt(math.degrees(el.rf.phase_rad))))
        scl = el.meta.get("flame_scl_fac")
        attrs.append(("scl_fac", fmt(1.0 if scl is None else float(scl))))
        # syncflag: 0 = driven phase, 1 = synchronous (complex fit, FLAME's default),
        # 2 = synchronous (sinusoidal fit).  1 and 2 differ in the fitted phase offset,
        # so a deck that said 2 must keep saying 2.
        syncflag = el.meta.get("flame_syncflag")
        if syncflag is None:
            syncflag = 1 if el.rf.phase_is_sync else 0
        if int(syncflag) != 1:
            attrs.append(("syncflag", str(int(syncflag))))
        return "rfcavity", attrs, owned

    def _emit_foil(self, el, p, lat, rep, rule):
        return "stripper", [], set()

    def _emit_taylor(self, el, p, lat, rep, rule):
        row6 = el.meta.get("flame_matrix_row6") or [0.0] * 6 + [1.0]
        matrix = [list(map(float, r[:6])) for r in el.matrix]
        offset = [float(v) for v in el.offset]
        if el.basis != "flame":
            # FLAME's state is (x mm, x' rad, y mm, y' rad, ...): a matrix in metres gets its
            # transverse rows/columns rescaled (R21 -> R21/1000, R12 -> 1000 R12, ...); this is
            # the exact inverse of what the reader does, so a FLAME map round-trips verbatim
            scale = [1e3, 1.0, 1e3, 1.0, 1.0, 1.0]
            matrix = [[matrix[i][j] * scale[i] / scale[j] for j in range(6)] for i in range(6)]
            offset = [offset[i] * scale[i] for i in range(6)]
            coupled = any(matrix[i][j] for i in range(4) for j in (4, 5)) or \
                any(matrix[i][j] for i in (4, 5) for j in range(4))
            if coupled and "flame_matrix_row6" not in el.meta:
                rep.lossy("TAYLOR_BASIS_FLAME",
                          "the map couples transverse and longitudinal coordinates; FLAME's phase/energy "
                          "units (rad, MeV/u) were not converted for those terms",
                          element=el.name, kind=el.kind)
        flat: list[float] = []
        for i in range(6):
            flat += matrix[i] + [offset[i]]
        flat += [float(v) for v in list(row6)[:7]]
        body = ", ".join(fmt(v) for v in flat)
        return "tmatrix", [("matrix", f"[{body}]")], {"matrix"}

    def _emit_collimator(self, el, p, lat, rep, rule):
        return "marker", [], set()

    def _emit_freq(self, el, p, lat, rep, rule):
        return None, [], set()

    def _emit_directive(self, el, p, lat, rep, rule):
        return None, [], set()

    def _emit_superposition(self, el, p, lat, rep, rule):
        return "drift", [("L", fmt(el.length))], {"L"}

    def _emit_patch(self, el, p, lat, rep, rule):
        return "marker", [], set()

    def _emit_referencechange(self, el, p, lat, rep, rule):
        return "marker", [], set()

    def _emit_fieldmap(self, el, p, lat, rep, rule):
        return "drift", [("L", fmt(el.length))], {"L"}

    def _emit_ncells(self, el, p, lat, rep, rule):
        return "drift", [("L", fmt(el.length))], {"L"}

    def _emit_rfqcell(self, el, p, lat, rep, rule):
        return "drift", [("L", fmt(el.length))], {"L"}

    # -- helpers ------------------------------------------------------------
    def _ref(self, p: Placed, lat: Lattice) -> ReferenceParticle:
        if self._energy_mode == "local" and p.ref_in is not None:
            return p.ref_in
        return lat.reference


def _extra_roll(el: Element) -> float:
    """FLAME has no skew magnet: a skew component is written as a body ``roll``.

    ``(Bn, Bs)`` of order *n* is ``hypot(Bn, Bs)`` rotated by
    ``−atan2(Bs, Bn)/(n+1)``, the same rule MAD-X's ``(k1, k1s)`` follows.  The sign
    was **measured** on 2026-09-03: a FLAME ``quadrupole, B2 = k·Bρ, roll = +π/8``
    reproduces MAD-X's ``quadrupole, k1 = k·cos(π/4), k1s = −k·sin(π/4)`` (tilt =
    +π/8) to nine digits in R21/R43/R23/R41 — so FLAME's ``roll`` *is* MAD-X's
    ``tilt`` (``tests/oracles/test_flame_adapter.py::test_roll_is_the_madx_tilt``).
    """
    mp = getattr(el, "multipole", None)
    if mp is None:
        return 0.0
    for order in sorted(set(mp.Bn) | set(mp.Bs) | set(mp.tilt)):
        skew = mp.Bs.get(order, 0.0)
        if skew:
            return -math.atan2(skew, mp.Bn.get(order, 0.0)) / (order + 1)
        tilt = mp.tilt.get(order, 0.0)
        if tilt:
            return tilt
    return 0.0


def _aperture_radius(ap) -> float:
    limits = [v for v in (ap.half_x, ap.half_y) if v]
    return min(limits) if limits else 0.0


def _render_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_render_value(v) for v in value) + "]"
    return fmt(float(value))


def _line_order(lat: Lattice, root: str, depth: int = 0) -> list[str] | None:
    """Sub-lines before the lines that use them (FLAME is a single-pass parser)."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(name: str, level: int) -> bool:
        if level > 64 or name in seen:
            return name in seen
        seen.add(name)
        for it in lat.lines[name].items:
            if it.ref in lat.lines and not visit(it.ref, level + 1):
                return False
            if it.ref not in lat.lines and it.ref not in lat.elements:
                return False
        out.append(name)
        return True

    return out if visit(root, depth) else None


def _render_line(name: str, members: list[str], per_line: int = 6) -> list[str]:
    if not members:
        return [f"{name}: LINE = ();"]
    out = [f"{name}: LINE = ("]
    for i in range(0, len(members), per_line):
        chunk = members[i:i + per_line]
        comma = "," if i + per_line < len(members) else ""
        out.append("  " + ", ".join(chunk) + comma)
    out.append(");")
    return out


_MISSING = set(ALL_KINDS) - set(Writer.RULES)
if _MISSING:  # pragma: no cover
    raise RuntimeError(f"FLAME writer RULES do not cover {sorted(_MISSING)}")


def write(lattice: Lattice, path: str | Path, **options) -> FidelityReport:
    return Writer().write(lattice, Path(path), **options)


def dumps(lattice: Lattice, **options) -> str:
    return Writer().dumps(lattice, **options)


__all__ = ["BPM_FAMILIES", "DEFAULT_ENVELOPE", "KNOWN_CAVTYPES", "NAME_RE", "NameMap",
           "RESERVED", "Rule", "Writer", "dumps", "fmt", "name_tag", "sanitize", "write"]
