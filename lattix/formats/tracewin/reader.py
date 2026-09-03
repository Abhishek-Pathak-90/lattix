"""TraceWin ``.dat`` reader → lattix IR (PLAN §6 task 1.2).

Provenance: ported from HELIX ``linac_gen/io/tracewin_parser.py`` — the label grammar
(``NAME : CARD`` / ``NAME: CARD`` / ``NAME:CARD`` / standalone ``NAME :``), latin-1 decoding,
``;`` comments including the ``;@LG``, ``; HELIX_FOIL`` and ``; HELIX_SC_GRID`` comment cards,
the FREQ / FIELD_MAP_PATH / SET_SYNC_PHASE state machine, SUPERPOSE_MAP clusters, the
DIAG_POSITION 1e50 "unconstrained" sentinels, the diagnostic-hardware no-op cards
(``BPM :`` …), the ERROR_* family and the strict/permissive ``_downgrade`` design (here: entries
in a :class:`~lattix.fidelity.FidelityReport`, ``raise_if(strict)`` at the end of the parse).

Deliberate differences from HELIX:

* every card becomes an IR element in deck order: commands are inline zero-length
  ``Directive``s (``role`` classifies them) that the TraceWin writer re-emits verbatim, and
  unknown cards are kept the same way with a DROPPED ``UNKNOWN_CARD`` entry;
* ``EDGE`` + ``BEND`` + ``EDGE`` are clustered into ONE :class:`~lattix.ir.elements.Bend`
  (HELIX keeps three elements); the field index becomes a lab gradient through the local
  rigidity (``G = −N·Bρ_signed/ρ²``);
* a FIELD_MAP whose component files are missing is KEPT with a LOSSY ``FM_FILES_MISSING``
  entry (HELIX drops the element in permissive mode);
* ``SET_SYNC_PHASE`` binds to the next RF card *including* a thin GAP (TraceWin semantics;
  HELIX binds field maps / NCELLS only and warns on a GAP).  The flag is pending across
  non-RF cards and consumed once;
* raw (non-sync) RF phases are converted to the IR's species-independent synchronous phase
  with :func:`lattix.ir.rf.phase_from_tracewin_deg` (π shift for negative species); the
  species is a reader option and ``SPECIES_ASSUMED`` is recorded when it was defaulted.

Units: deck mm / deg / MHz / MeV / V → IR m / rad / Hz / eV / V.
"""

from __future__ import annotations

import math
import os
import re
import shlex
from pathlib import Path

from lattix.fidelity import FidelityClass, FidelityReport
from lattix.formats.tracewin.fieldmap_files import has_electric_channel, resolve_field_files
from lattix.formats.tracewin.syntax import (
    COMMAND_ROLES,
    DEFAULT_FREQ_MHZ,
    DIAGNOSTIC_CARDS,
    HARDWARE_MARKER_CARDS,
    SCHEMA,
    UNSUPPORTED_ELEMENT_CARDS,
    parse_positionals,
)
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
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    NCells,
    Provenance,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Solenoid,
    SolenoidP,
    Superposition,
)
from lattix.ir.expr import Expression, ExpressionError, evaluate
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.normalize import gradient_from_k1, k1_from_field_index
from lattix.ir.reference import ReferenceParticle, Species
from lattix.ir.reference import species as _species
from lattix.ir.rf import phase_from_tracewin_deg
from lattix.ir.units import C_LIGHT, DEG, MEV, MHZ, MM
from lattix.ir.walk import propagate

# Comment cards (HELIX extensions living in ``;`` comments so TraceWin ignores them).
_LG_DIRECTIVE = re.compile(r"^\s*;\s*@LG\s+(\S+?)\s*=\s*(.+?)\s*$")
_HELIX_FOIL = re.compile(
    r"^\s*;\s*HELIX_FOIL\s+(\S+)\s+(\S+)\s+([\d.eE+-]+)(?:\s+(auto|landau|gaussian))?\s*(?:;.*)?$"
)
_HELIX_SC_GRID = re.compile(r"^\s*;\s*HELIX_SC_GRID\s+([\d.eE+-]+)\s*(?:;.*)?$")
# lattix's own reversible name tag (PLAN §4.4, I-15) and the TITLE-as-comment the writer emits.
_LATTIX_TAG = re.compile(r'^\s*;\s*lattix:\s*name="([^"]*)"(?:\s+type="([^"]*)")?\s*$')
_TITLE_COMMENT = re.compile(r"^\s*;\s*TITLE\s+(.*?)\s*$")

_UG_PER_CM2 = 1e-5  # kg/m² per µg/cm²
_SENTINEL = 1e50  # DIAG_POSITION "unconstrained" target

#: Zero-length command cards that neither close a SUPERPOSE cluster nor break EDGE/BEND adjacency.
_ZERO_LENGTH_COMMANDS = frozenset({"FREQ", "TITLE", "FIELD_MAP_PATH", "PARTRAN_STEP", "VARIABLE"})
#: Card-name prefixes of the TraceWin command families → Directive role (unlisted cards).
_PREFIX_ROLES: tuple[tuple[str, str], ...] = (
    ("SET_", "matching"),
    ("MIN_", "matching"),
    ("MAX_", "matching"),
    ("ADJUST", "matching"),
    ("MATCH_", "matching"),
    ("DIAG_", "matching"),
    ("START_", "matching"),
    ("PLOT_", "tracking"),
    ("CHART_", "tracking"),
)
#: Cards that leave an open SUPERPOSE cluster open (HELIX ``transparent`` set).
_CLUSTER_TRANSPARENT = frozenset(
    {
        "SUPERPOSE_MAP",
        "TITLE",
        "PARTRAN_STEP",
        "FIELD_MAP_PATH",
        "SET_SYNC_PHASE",
        "SET_BEAM_ENERGY",
        "SET_BEAM_E0_P0",
    }
)
_CLUSTER_TRANSPARENT_ROLES = frozenset({"matching", "sync_phase", "title", "tracking"})


def _split_tokens(code: str) -> list[str]:
    """Tokenise one card; quoted tokens (paths with spaces) stay whole, quotes are stripped."""
    try:
        toks = shlex.split(code, posix=False)
    except ValueError:  # unbalanced quote / apostrophe in a card
        toks = code.split()
    return [t.strip('"') for t in toks]


def _split_label(tokens: list[str]) -> tuple[str | None, str, str, list[str], bool]:
    """HELIX label grammar → ``(label, KEYWORD, raw card token, params, label_only)``.

    ``NAME : CARD …`` / ``NAME: CARD …`` / ``NAME:CARD …`` drop the label; a standalone
    ``NAME :`` / ``NAME:`` keeps NAME as the (marker) card with ``label_only=True``.
    """
    label = None
    label_only = False
    if len(tokens) >= 3 and tokens[-1] == ":":
        # ``Dump Entrance :`` — multi-word label-only line (PIP-II decks)
        raw = " ".join(tokens[:-1])
        return None, raw.upper(), raw, [], True
    if (
        len(tokens) >= 2
        and tokens[1] != ":"
        and tokens[-1].endswith(":")
        and len(tokens[-1]) > 1
        and ":" not in tokens[0]
    ):
        # ``LB650 CM:`` — multi-word label-only line, colon glued to the last word
        raw = " ".join(tokens)[:-1]
        return None, raw.upper(), raw, [], True
    if len(tokens) >= 2 and tokens[1] == ":":
        if len(tokens) == 2:
            tokens = [tokens[0]]
            label_only = True
        else:
            label = tokens[0]
            tokens = tokens[2:]
    elif (
        len(tokens) >= 2
        and tokens[1].startswith(":")
        and len(tokens[1]) > 1
        and tokens[0][0].isalpha()
        and ":" not in tokens[0]
    ):
        # ``NAME :CARD …`` — colon glued to the card
        label = tokens[0]
        tokens = [tokens[1][1:]] + tokens[2:]
    elif tokens[0].endswith(":") and len(tokens[0]) > 1 and tokens[0][0].isalpha():
        if len(tokens) >= 2:
            label = tokens[0][:-1]
            tokens = tokens[1:]
        else:
            tokens = [tokens[0][:-1]]
            label_only = True
    elif (
        ":" in tokens[0]
        and not tokens[0].startswith(":")
        and not tokens[0].endswith(":")
        and tokens[0][0].isalpha()
    ):
        lab, _, card = tokens[0].partition(":")
        if card:
            label = lab
            tokens = [card] + tokens[1:]
    raw = tokens[0].rstrip(":")
    return label, raw.upper(), raw, tokens[1:], label_only


def _guess_role(keyword: str) -> str | None:
    """Role of an unlisted card that belongs to a known TraceWin command family by prefix."""
    for prefix, role in _PREFIX_ROLES:
        if keyword.startswith(prefix):
            return role
    return None


def _is_command(keyword: str) -> bool:
    """Zero-length command card (no transport effect): transparent for EDGE/BEND adjacency."""
    return (
        keyword.startswith("ERROR_")
        or keyword in COMMAND_ROLES
        or keyword in _ZERO_LENGTH_COMMANDS
        or _guess_role(keyword) is not None
    )


def _circle(r_mm: float) -> ApertureP | None:
    return None if r_mm is None or r_mm <= 0 else ApertureP.circle(r_mm * MM)


def _coerce_float(v, default: float) -> float:
    return default if v is None else float(v)


class _Parser:
    """One deck → (Lattice, FidelityReport).  Mutable parse state lives on the instance."""

    def __init__(
        self,
        path: Path,
        *,
        species: str | Species | None,
        kinetic_energy_eV: float,
        frequency_Hz: float | None,
        base_dir: str | Path | None,
        name: str | None,
    ):
        self.path = Path(path)
        self.base_dir = str(Path(base_dir) if base_dir is not None else self.path.parent)
        self.species_given = species is not None
        self.sp = _species(species if species is not None else "proton")
        self.charge = self.sp.charge
        self.ke = float(kinetic_energy_eV)
        self.default_freq_hz = float(frequency_Hz) if frequency_Hz else DEFAULT_FREQ_MHZ * MHZ
        self.name = name or self.path.stem or "lattice"
        self.lat = Lattice(
            name=self.name, reference=ReferenceParticle(species=self.sp, kinetic_energy_eV=self.ke)
        )
        self.line = Line(name=self.name)
        self.report = FidelityReport(source_format="tracewin", source_file=str(self.path))
        self.meta: dict = {"title": "", "source": str(self.path), "lg_options": {}}
        # state
        self.freq_hz: float | None = None
        self.first_freq_hz: float | None = None
        self.field_map_path: str | None = None
        self.pending_sync = False
        self.counters: dict[str, int] = {}
        self.pending_name: str | None = None
        self.pending_edge: tuple[dict, list[str], int, str | None] | None = None
        self.last_bend: Bend | None = None
        self.open_cluster: list[tuple[float, FieldMap]] = []
        self.pending_superpose: list[tuple[float, list[str], int]] = []
        self.n_raw_phases = 0
        self.line_no = 0
        self.postpass: list[tuple[str, Element, dict]] = []  # (what, element, data)
        self.variables: dict[str, float] = {}  # ``VARIABLE name value`` definitions (lower-case keys)
        self.expr_pending: dict[str, Expression] = {}  # operand expressions of the card being built

    # -- bookkeeping ----------------------------------------------------------------------
    def _auto_name(self, card: str) -> str:
        key = re.sub(r"[^A-Za-z0-9]+", "_", card).strip("_").upper() or "CARD"
        self.counters[key] = self.counters.get(key, 0) + 1
        return f"{key}_{self.counters[key]:04d}"

    def _name(self, card: str, label: str | None, *, element: bool = True) -> str:
        if label:
            return label
        if element and self.pending_name:
            nm, self.pending_name = self.pending_name, None
            return nm
        return self._auto_name(card)

    def _prov(self, keyword: str, label: str | None) -> Provenance:
        return Provenance(
            format="tracewin",
            file=str(self.path),
            line=self.line_no,
            original_name=label,
            original_type=keyword,
        )

    def _add(self, e: Element) -> str:
        """Register ``e`` in the lattice and append it to the flat line (deck order)."""
        if self.expr_pending:
            e.expressions = dict(self.expr_pending)
            self.expr_pending = {}
        nm = self.lat.add_element(e)
        self.line.items.append(LineItem(ref=nm))
        self.report.exact(nm, e.kind)
        return nm

    def _entry(
        self,
        cls: FidelityClass,
        code: str,
        msg: str,
        *,
        element: str | None = None,
        kind: str | None = None,
        **details,
    ) -> None:
        self.report.add(cls, code, msg, element=element, kind=kind, line=self.line_no, **details)

    def _resolve_expressions(self, card: str, fields: list, params: list[str]) -> list[str]:
        """Numeric operands that are expressions over ``VARIABLE`` definitions (BTL ``mad2tw``
        decks: ``QUAD 50 qfs06_mad*mad2tw 25.4``) are evaluated; the source text is kept in
        ``Element.expressions`` and an EQUIVALENT ``EXPRESSION_EVALUATED`` entry is recorded."""
        out = list(params)
        names = []
        for i, f in enumerate(fields[: len(params)]):
            if f.cast is str:
                continue
            tok = params[i]
            try:
                float(tok)
                continue
            except ValueError:
                pass
            try:
                val = evaluate(tok, self.variables)
            except ExpressionError:
                continue  # parse_positionals reports the cast failure
            out[i] = repr(val)
            self.expr_pending[f.name] = Expression(text=tok)
            names.append(f.name)
        if names:
            self._entry(
                FidelityClass.EQUIVALENT,
                "EXPRESSION_EVALUATED",
                f"{card}: operand(s) {names} evaluated from VARIABLE definitions (TraceWin itself has no "
                "variables; the source text is kept in Element.expressions)",
                kind=card,
            )
        return out

    def _pos(self, card: str, params: list[str]) -> dict:
        fields = SCHEMA[card]
        if self.variables:
            params = self._resolve_expressions(card, fields, params)
        kw = parse_positionals(fields, params)
        if len(params) > len(fields):
            self._entry(
                FidelityClass.EQUIVALENT,
                "EXTRA_TOKENS_IGNORED",
                f"{card}: {len(params) - len(fields)} trailing token(s) beyond the schema ignored: "
                f"{params[len(fields) :]}",
                kind=card,
                extra=params[len(fields) :],
            )
        return kw

    def _directive(self, keyword: str, params: list[str], role: str, *, name: str | None = None) -> Directive:
        d = Directive(
            name=name or self._auto_name(keyword),
            format="tracewin",
            card=keyword,
            args=list(params),
            role=role,
            provenance=self._prov(keyword, None),
        )
        self._add(d)
        return d

    def _rf_frequency(self) -> float:
        if self.freq_hz is None:
            self.freq_hz = self.default_freq_hz
            msg = f"RF card before any FREQ: assuming {self.default_freq_hz / MHZ:g} MHz"
            self._entry(FidelityClass.EQUIVALENT, "FREQ_ASSUMED", msg)
            self.lat.warnings.append(f"line {self.line_no}: {msg}")
        return self.freq_hz

    def _take_sync(self, consume: bool = True) -> bool:
        flag = self.pending_sync
        if consume:
            self.pending_sync = False
        return flag

    # -- comment cards ----------------------------------------------------------------------
    def _comment_card(self, raw: str) -> bool:
        m = _LG_DIRECTIVE.match(raw)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
            self.meta["lg_options"][key] = _coerce_option(value)
            d = Directive(
                name=self._auto_name("LG"),
                format="tracewin",
                card="@LG",
                args=[f"{key}={value}"],
                role="tracking",
                provenance=self._prov("@LG", None),
            )
            self._add(d)
            return True
        m = _HELIX_FOIL.match(raw)
        if m:
            thick = float(m.group(3))
            f = Foil(
                name=m.group(1),
                material=m.group(2),
                thickness_kg_per_m2=thick * _UG_PER_CM2,
                provenance=self._prov("HELIX_FOIL", m.group(1)),
                native={"tracewin": {"straggling": m.group(4) or "auto"}},
            )
            self._add(f)
            return True
        m = _HELIX_SC_GRID.match(raw)
        if m:
            d = Directive(
                name=self._auto_name("HELIX_SC_GRID"),
                format="tracewin",
                card="HELIX_SC_GRID",
                args=[m.group(1)],
                role="tracking",
                provenance=self._prov("HELIX_SC_GRID", None),
            )
            self._add(d)
            return True
        m = _LATTIX_TAG.match(raw)
        if m:
            self.pending_name = m.group(1)
            return True
        m = _TITLE_COMMENT.match(raw)
        if m:
            self._title(m.group(1).split())
            return True
        return False

    def _title(self, words: list[str]) -> None:
        self.meta["title"] = " ".join(words)
        self._directive("TITLE", words, "title")

    # -- main loop --------------------------------------------------------------------------
    def parse(self) -> tuple[Lattice, FidelityReport]:
        text = self.path.read_text(encoding="latin-1")
        for self.line_no, raw in enumerate(text.splitlines(), 1):
            if raw.lstrip().startswith(";") and self._comment_card(raw):
                continue
            code = raw.split(";", 1)[0].strip()
            if not code:
                continue
            tokens = _split_tokens(code)
            if not tokens:
                continue
            label, keyword, raw_card, params, label_only = _split_label(tokens)
            if not keyword:
                continue
            if keyword == "END":
                break
            self._dispatch(keyword, raw_card, params, label, label_only)
        self._finish()
        return self.lat, self.report

    def _dispatch(
        self, keyword: str, raw_card: str, params: list[str], label: str | None, label_only: bool
    ) -> None:
        self.expr_pending = {}
        # EDGE+BEND+EDGE clustering: an entry EDGE binds only to the next BEND, with nothing but
        # zero-length command cards (ERROR_BEND_*, SET_* …) in between.
        if keyword == "EDGE":
            self._on_edge(params, label)
            return
        if keyword != "BEND" and not _is_command(keyword):
            self._flush_pending_edge()
            self.last_bend = None
        # SUPERPOSE cluster close condition (the next ordinary element ends an open cluster).
        if self.open_cluster or self.pending_superpose:
            role = COMMAND_ROLES.get(keyword)
            transparent = (
                keyword in _CLUSTER_TRANSPARENT
                or role in _CLUSTER_TRANSPARENT_ROLES
                or keyword.startswith("ERROR_")
                or (keyword == "FIELD_MAP" and bool(self.pending_superpose))
            )
            if keyword == "FREQ":
                self._entry(
                    FidelityClass.LOSSY,
                    "CLUSTER_SPLIT_BY_FREQ",
                    "FREQ closes the open SUPERPOSE cluster (mixed-frequency clusters are not "
                    "modelled) — the maps are laid out sequentially, lengthening the line",
                )
                transparent = False
            if not transparent:
                self._flush_cluster()
        try:
            self._handle(keyword, raw_card, params, label, label_only)
        except (ValueError, IndexError, KeyError) as exc:
            d = Directive(
                name=self._auto_name(keyword),
                format="tracewin",
                card=raw_card if label_only else keyword,
                args=list(params),
                role="unknown",
                provenance=self._prov(keyword, label),
            )
            self._add(d)
            self._entry(
                FidelityClass.DROPPED,
                "MALFORMED_CARD",
                f"{keyword}: {exc} — card kept verbatim only",
                element=d.name,
                kind="Directive",
            )

    # -- card handlers ----------------------------------------------------------------------
    def _handle(
        self, keyword: str, raw_card: str, params: list[str], label: str | None, label_only: bool
    ) -> None:
        if (
            label_only
            and keyword not in SCHEMA
            and keyword not in COMMAND_ROLES
            and not keyword.startswith("ERROR_")
        ):
            self._instrument(keyword, raw_card, params, label, label_only=True)
            return
        if keyword == "TITLE":
            self._title(params)
        elif keyword == "FREQ":
            kw = self._pos("FREQ", params)
            self.freq_hz = kw["frequency"] * MHZ
            if self.first_freq_hz is None:
                self.first_freq_hz = self.freq_hz
            self._add(
                Freq(
                    name=self._name("FREQ", label, element=False),
                    frequency_Hz=self.freq_hz,
                    provenance=self._prov(keyword, label),
                )
            )
        elif keyword == "FIELD_MAP_PATH":
            kw = self._pos("FIELD_MAP_PATH", params)
            p = kw["path"]
            self.field_map_path = os.path.normpath(p if os.path.isabs(p) else os.path.join(self.base_dir, p))
            self.meta.setdefault("tracewin", {})["field_map_path"] = self.field_map_path
        elif keyword == "DRIFT":
            self._drift(params, label)
        elif keyword == "QUAD":
            self._quad(params, label)
        elif keyword == "SOLENOID":
            kw = self._pos("SOLENOID", params)
            self._add(
                Solenoid(
                    name=self._name("SOLENOID", label),
                    length=kw["length"] * MM,
                    aperture=_circle(kw["aperture"]),
                    solenoid=SolenoidP(Bsol_T=kw["field"]),
                    provenance=self._prov(keyword, label),
                )
            )
        elif keyword == "GAP":
            self._gap(params, label)
        elif keyword == "FIELD_MAP":
            self._field_map(params, label)
        elif keyword == "BEND":
            self._bend(params, label)
        elif keyword in ("THIN_STEERING", "STEERER"):
            self._steerer(params, label)
        elif keyword == "APERTURE":
            self._aperture(params, label)
        elif keyword == "MARKER":
            self._add(
                Marker(
                    name=self._name("MARKER", label),
                    provenance=self._prov(keyword, label),
                    native={"tracewin": {"card": "MARKER", "args": list(params)}},
                )
            )
        elif keyword in DIAGNOSTIC_CARDS or keyword in HARDWARE_MARKER_CARDS:
            self._instrument(keyword, raw_card, params, label, label_only=label_only)
        elif keyword == "NCELLS":
            self._ncells(params, label)
        elif keyword == "RFQ_CELL":
            self._rfq_cell(params, label)
        elif keyword == "SET_BEAM_ENERGY":
            kw = self._pos("SET_BEAM_ENERGY", params)
            self._add(
                ReferenceChange(
                    name=self._name(keyword, label, element=False),
                    energy_eV=kw["energy_MeV"] * MEV,
                    provenance=self._prov(keyword, label),
                    native={"tracewin": {"card": keyword, "args": list(params)}},
                )
            )
        elif keyword == "SET_BEAM_E0_P0":
            kw = self._pos("SET_BEAM_E0_P0", params)
            self._add(
                ReferenceChange(
                    name=self._name(keyword, label, element=False),
                    dE_ref_eV=kw["dE_MeV"] * MEV if kw["ke"] else None,
                    provenance=self._prov(keyword, label),
                    native={"tracewin": {"card": keyword, "args": list(params)}},
                )
            )
        elif keyword == "SET_SYNC_PHASE":
            self._directive(keyword, params, "sync_phase")
            self.pending_sync = True
        elif keyword == "SUPERPOSE_MAP":
            self._superpose_map(params)
        elif keyword == "SHIFT_IN_FIELD_MAP":
            d = self._directive(keyword, params, "superpose")
            self._entry(
                FidelityClass.EQUIVALENT,
                "SHIFT_IN_FIELD_MAP_INLINE",
                "SHIFT_IN_FIELD_MAP kept inline: the following diagnostic sits at the card position "
                "(TraceWin reads it dz inside the next field map)",
                element=d.name,
                kind="Directive",
            )
        elif keyword == "SUPERPOSE_MAP_OUT":
            d = self._directive(keyword, params, "superpose")
            self._entry(
                FidelityClass.LOSSY,
                "SUPERPOSE_MAP_OUT_UNSUPPORTED",
                "SUPERPOSE_MAP_OUT (curved reference through a cluster) is not modelled; card kept verbatim",
                element=d.name,
                kind="Directive",
            )
        elif keyword in ("READ_DST", "BEAM_ROT"):
            d = self._directive(keyword, params, "beam")
            self._entry(
                FidelityClass.LOSSY,
                "BEAM_DIRECTIVE",
                f"{keyword} mutates the beam mid-line (not modelled; the reference keeps its upstream "
                "state); card kept verbatim",
                element=d.name,
                kind="Directive",
            )
        elif keyword == "REPEAT_ELE":
            d = self._directive(keyword, params, "other")
            self._entry(
                FidelityClass.LOSSY,
                "REPEAT_ELE_NOT_EXPANDED",
                "REPEAT_ELE is kept verbatim but NOT expanded: the IR sequence contains the repeated "
                "block once",
                element=d.name,
                kind="Directive",
            )
        elif keyword == "PARTRAN_STEP":
            self.meta["partran_step"] = list(params)
            self._directive(keyword, params, "tracking")
        elif keyword.startswith("ERROR_"):
            self._directive(keyword, params, "error")
        elif keyword in UNSUPPORTED_ELEMENT_CARDS:
            d = self._directive(keyword, params, "other")
            self._entry(
                FidelityClass.LOSSY,
                "UNSUPPORTED_ELEMENT",
                f"{keyword} ({UNSUPPORTED_ELEMENT_CARDS[keyword]}) is not modelled by the IR; "
                "kept verbatim as a zero-length directive",
                element=d.name,
                kind="Directive",
            )
        elif keyword == "VARIABLE":
            self._variable(params)
        elif keyword in COMMAND_ROLES:
            self._directive(keyword, params, COMMAND_ROLES[keyword])
        elif (role := _guess_role(keyword)) is not None:
            d = self._directive(keyword, params, role)
            self._entry(
                FidelityClass.EQUIVALENT,
                "UNLISTED_COMMAND",
                f"{keyword} is not in the card table; kept verbatim as a {role} directive (TraceWin command "
                "family by prefix, no transport effect assumed)",
                element=d.name,
                kind="Directive",
            )
        else:
            d = Directive(
                name=self._auto_name(keyword),
                format="tracewin",
                card=raw_card if label_only else keyword,
                args=list(params),
                role="unknown",
                provenance=self._prov(keyword, label),
            )
            self._add(d)
            self._entry(
                FidelityClass.DROPPED,
                "UNKNOWN_CARD",
                f"unsupported card {keyword!r} kept verbatim only",
                element=d.name,
                kind="Directive",
            )

    def _variable(self, params: list[str]) -> None:
        """``VARIABLE name value`` (BTL ``mad2tw`` export dialect): a named constant usable in
        later numeric operands.  Kept verbatim as a directive; the value lands in
        ``Lattice.variables``."""
        if len(params) < 2:
            raise ValueError("VARIABLE needs a name and a value")
        name, text = params[0], params[1]
        value = evaluate(text, self.variables)
        self.variables[name.lower()] = value
        self.lat.variables[name] = Variable(value=value, expression=Expression(text=text))
        self._directive("VARIABLE", params, "other")

    def _drift(self, params: list[str], label: str | None) -> None:
        kw = self._pos("DRIFT", params)
        ry = kw["aperture_y"]
        if ry is not None and ry > 0:
            ap: ApertureP | None = ApertureP.rect(kw["aperture"] * MM, ry * MM)
        else:
            ap = _circle(kw["aperture"])
        shift = None
        if kw["x_shift"] or kw["y_shift"]:
            shift = BodyShiftP(x_offset=kw["x_shift"] * MM, y_offset=kw["y_shift"] * MM)
        self._add(
            Drift(
                name=self._name("DRIFT", label),
                length=kw["length"] * MM,
                aperture=ap,
                shift=shift,
                provenance=self._prov("DRIFT", label),
            )
        )

    def _quad(self, params: list[str], label: str | None) -> None:
        kw = self._pos("QUAD", params)
        Bn = {1: kw["gradient"]}
        for n, key in ((2, "g3"), (3, "g4"), (4, "g5"), (5, "g6")):
            if kw[key]:
                Bn[n] = kw[key]
        tilt = {1: kw["skew_angle"] * DEG} if kw["skew_angle"] else {}
        native = {"tracewin": {"gfr": kw["gfr"]}} if kw["gfr"] else {}
        self._add(
            Quadrupole(
                name=self._name("QUAD", label),
                length=kw["length"] * MM,
                aperture=_circle(kw["aperture"]),
                multipole=MagneticMultipoleP(Bn=Bn, tilt=tilt),
                native=native,
                provenance=self._prov("QUAD", label),
            )
        )

    def _gap(self, params: list[str], label: str | None) -> None:
        kw = self._pos("GAP", params)
        freq = self._rf_frequency()
        sync = self._take_sync()
        if not sync:
            self.n_raw_phases += 1
        name = self._name("GAP", label)
        native = {"tracewin": {"p_flag": kw["p_flag"]}} if kw["p_flag"] else {}
        self._add(
            RFCavity(
                name=name,
                length=0.0,
                aperture=_circle(kw["aperture"]),
                rf=RFP(
                    frequency_Hz=freq,
                    voltage_V=kw["e0tl"],
                    phase_rad=phase_from_tracewin_deg(kw["phase"], self.charge, sync),
                    phase_is_sync=sync,
                ),
                native=native,
                provenance=self._prov("GAP", label),
            )
        )
        if kw["p_flag"]:
            self._entry(
                FidelityClass.LOSSY,
                "GAP_ABSOLUTE_PHASE",
                f"GAP p_flag={kw['p_flag']} (absolute-phase mode): the deck phase is not the synchronous "
                "phase the IR energy rule assumes; flag kept in native",
                element=name,
                kind="RFCavity",
            )

    def _field_map(self, params: list[str], label: str | None) -> None:
        kw = self._pos("FIELD_MAP", params)
        freq = self._rf_frequency()
        raw = kw["filename"]
        dir_part, base = os.path.split(raw)
        if os.path.isabs(raw):
            map_dir = dir_part
        elif self.field_map_path is not None:
            map_dir = os.path.join(self.field_map_path, dir_part)
        else:
            map_dir = os.path.join(self.base_dir, dir_part)
        map_dir = os.path.normpath(map_dir)
        existing, missing, err = resolve_field_files(kw["geom"], map_dir, base)
        in_cluster = bool(self.pending_superpose)
        consume = (not in_cluster) or has_electric_channel(kw["geom"])
        sync = self._take_sync(consume=consume) if consume else False
        if not sync:
            self.n_raw_phases += 1
        name = self._name("FIELD_MAP", label)
        fm = FieldMap(
            name=name,
            length=kw["length"] * MM,
            aperture=_circle(kw["aperture"]),
            geom=kw["geom"],
            files=[base],
            ke=kw["ke"],
            kb=kw["kb"],
            ki=kw["ki"],
            ka=kw["ka"],
            p_flag=kw["p_flag"],
            rf=RFP(
                frequency_Hz=freq,
                phase_rad=phase_from_tracewin_deg(kw["phase"], self.charge, sync),
                phase_is_sync=sync,
            ),
            meta={"field_map_dir": map_dir, "field_files": existing, "field_files_missing": missing},
            provenance=self._prov("FIELD_MAP", label),
        )
        if err:
            self._entry(
                FidelityClass.LOSSY,
                "FM_GEOM_UNSUPPORTED",
                f"FIELD_MAP geom {kw['geom']}: {err}",
                element=name,
                kind="FieldMap",
                geom=kw["geom"],
            )
        elif missing:
            self._entry(
                FidelityClass.LOSSY,
                "FM_FILES_MISSING",
                f"FIELD_MAP component file(s) not found: {', '.join(missing)} — element kept",
                element=name,
                kind="FieldMap",
                missing=missing,
            )
        if kw["p_flag"] == 1:
            self._entry(
                FidelityClass.EQUIVALENT,
                "RF_ABSOLUTE_PHASE",
                "FIELD_MAP p_flag=1: absolute RF phase (kept on the element)",
                element=name,
                kind="FieldMap",
            )
        if in_cluster:
            z0, sp_args, _ln = self.pending_superpose.pop()
            fm.native["tracewin"] = {"superpose": sp_args}
            self.open_cluster.append((z0, fm))
        else:
            self._add(fm)

    def _bend(self, params: list[str], label: str | None) -> None:
        kw = self._pos("BEND", params)
        angle = kw["angle"] * DEG
        length = abs(kw["rho"]) * abs(angle) * MM
        tilt_ref = math.pi / 2 if kw["hv"] else 0.0
        name = self._name("BEND", label)
        b = Bend(
            name=name,
            length=length,
            aperture=_circle(kw["aperture"]),
            bend=BendP(angle=angle, tilt_ref=tilt_ref),
            native={"tracewin": {"bend": dict(kw), "has_edges": False}},
            provenance=self._prov("BEND", label),
        )
        if self.pending_edge is not None:
            ekw, eparams, eline, elabel = self.pending_edge
            self.pending_edge = None
            self._attach_edge(b, ekw, entry=True)
        self._add(b)
        if kw["field_index"]:
            self.postpass.append(("bend", b, {"N": kw["field_index"], "rho_m": abs(kw["rho"]) * MM}))
        self.last_bend = b

    def _on_edge(self, params: list[str], label: str | None) -> None:
        kw = self._pos("EDGE", params)
        if self.last_bend is not None and not self.last_bend.native["tracewin"].get("edge_out"):
            self._attach_edge(self.last_bend, kw, entry=False)
            self.last_bend = None
            return
        self.last_bend = None
        if self.pending_edge is not None:
            self._flush_pending_edge()
        self.pending_edge = (kw, list(params), self.line_no, label)

    def _attach_edge(self, b: Bend, kw: dict, *, entry: bool) -> None:
        nat = b.native["tracewin"]
        nat["has_edges"] = True
        nat["edge_in" if entry else "edge_out"] = dict(kw)
        bp = b.bend
        beta = kw["pole_rotation"] * DEG
        if entry:
            bp.e1 = beta
            bp.edge_int1 = kw["k1"]
        else:
            bp.e2 = beta
            bp.edge_int2 = kw["k1"]
        if kw["gap"]:
            bp.hgap = kw["gap"] / 2 * MM
        bp.fringe_k2 = kw["k2"]

    def _flush_pending_edge(self) -> None:
        if self.pending_edge is None:
            return
        kw, params, line_no, label = self.pending_edge
        self.pending_edge = None
        d = Directive(
            name=self._auto_name("EDGE"),
            format="tracewin",
            card="EDGE",
            args=params,
            role="edge",
            provenance=Provenance(
                format="tracewin",
                file=str(self.path),
                line=line_no,
                original_name=label,
                original_type="EDGE",
            ),
        )
        self._add(d)
        self.report.add(
            FidelityClass.LOSSY,
            "ORPHAN_EDGE",
            "EDGE card with no adjacent BEND kept as a directive (its pole-face focusing is not modelled)",
            element=d.name,
            kind="Directive",
            line=line_no,
        )

    def _steerer(self, params: list[str], label: str | None) -> None:
        kw = self._pos("THIN_STEERING", params)
        k = Kicker(
            name=self._name("THIN_STEERING", label),
            aperture=_circle(kw["aperture"]),
            electric=bool(kw["elec"]),
            provenance=self._prov("THIN_STEERING", label),
        )
        self._add(k)
        self.postpass.append(("kicker", k, {"bl_x": kw["bl_x"], "bl_y": kw["bl_y"]}))

    def _aperture(self, params: list[str], label: str | None) -> None:
        kw = self._pos("APERTURE", params)
        dx, dy, t = kw["dx"], kw["dy"], kw["ap_type"]
        name = self._name("APERTURE", label)
        native = {}
        if t == 1:
            ap = ApertureP(
                shape="ELLIPTICAL",
                x_limits=(-dx * MM, dx * MM),
                y_limits=(-(dy if dy > 0 else dx) * MM, (dy if dy > 0 else dx) * MM),
            )
        else:
            ap = ApertureP.rect(dx * MM, (dy if dy > 0 else dx) * MM)
            if t != 0:
                native = {"tracewin": {"ap_type": t}}
        self._add(Collimator(name=name, aperture=ap, native=native, provenance=self._prov("APERTURE", label)))
        if t not in (0, 1):
            self._entry(
                FidelityClass.LOSSY,
                "APERTURE_SHAPE",
                f"APERTURE type {t} (pepperpot/fraction/finger/ring) kept as a rectangle; type in native",
                element=name,
                kind="Collimator",
                ap_type=t,
            )

    def _instrument(
        self, keyword: str, raw_card: str, params: list[str], label: str | None, *, label_only: bool
    ) -> None:
        pdict: dict = {"args": list(params)}
        if keyword == "DIAG_POSITION":
            kw = parse_positionals(SCHEMA["DIAG_POSITION"], params)
            pdict = {
                "diag": kw["diag"],
                "x_target_m": None
                if kw["x_target"] is None or abs(kw["x_target"]) >= _SENTINEL
                else kw["x_target"] * MM,
                "y_target_m": None
                if kw["y_target"] is None or abs(kw["y_target"]) >= _SENTINEL
                else kw["y_target"] * MM,
                "accuracy_m": kw["accuracy"] * MM,
            }
        elif keyword in DIAGNOSTIC_CARDS and params:
            try:
                pdict["diag"] = int(float(params[0]))
            except ValueError:
                pass
        self._add(
            Instrument(
                name=self._name(keyword, label),
                family=keyword,
                params=pdict,
                provenance=self._prov(keyword, label),
                native={
                    "tracewin": {
                        "card": keyword,
                        "raw": raw_card,
                        "args": list(params),
                        "label_only": label_only,
                    }
                },
            )
        )

    def _ncells(self, params: list[str], label: str | None) -> None:
        fields = SCHEMA["NCELLS"]
        kw = parse_positionals(fields, params[: len(fields)])
        tail = [float(t) for t in params[len(fields) :]]
        kw["ttf_tail"] = tail
        freq = self._rf_frequency()
        sync = self._take_sync()
        if not sync:
            self.n_raw_phases += 1
        name = self._name("NCELLS", label)
        beta_g = kw["beta_g"]
        length = _ncells_length(kw, beta_g, freq) if beta_g > 0 else 0.0
        nc = NCells(
            name=name,
            length=length,
            aperture=_circle(kw["aperture"]),
            params=kw,
            rf=RFP(
                frequency_Hz=freq,
                voltage_V=kw["eot"] * length,
                ttf=1.0,
                phase_rad=phase_from_tracewin_deg(kw["theta_s"], self.charge, sync),
                phase_is_sync=sync,
            ),
            provenance=self._prov("NCELLS", label),
        )
        self._add(nc)
        if beta_g <= 0:
            self.postpass.append(("ncells", nc, {"kw": kw, "freq": freq}))
        if kw["p_flag"] == 1:
            self._entry(
                FidelityClass.EQUIVALENT,
                "RF_ABSOLUTE_PHASE",
                "NCELLS P=1: absolute RF phase (kept in params)",
                element=name,
                kind="NCells",
            )

    def _rfq_cell(self, params: list[str], label: str | None) -> None:
        kw = self._pos("RFQ_CELL", params)
        freq = self._rf_frequency()
        name = self._name("RFQ_CELL", label)
        self._add(
            RFQCell(
                name=name,
                length=kw["length"] * MM,
                aperture=_circle(kw["r0"]),
                params=kw,
                rf=RFP(
                    frequency_Hz=freq,
                    voltage_V=kw["voltage"],
                    phase_rad=kw["phi_s"] * DEG,
                    phase_is_sync=True,
                ),
                provenance=self._prov("RFQ_CELL", label),
            )
        )

    def _superpose_map(self, params: list[str]) -> None:
        kw = self._pos("SUPERPOSE_MAP", params)
        if self.pending_superpose:
            _z0, sp_args, ln = self.pending_superpose.pop()
            d = self._directive("SUPERPOSE_MAP", sp_args, "superpose")
            self.report.add(
                FidelityClass.LOSSY,
                "SUPERPOSE_DANGLING",
                "SUPERPOSE_MAP not followed by a FIELD_MAP — placement kept verbatim, ignored",
                element=d.name,
                kind="Directive",
                line=ln,
            )
        if any(kw[k] for k in ("x0", "y0", "theta_z", "theta_x", "theta_y")):
            self._entry(
                FidelityClass.EQUIVALENT,
                "SUPERPOSE_MAP_OFFSETS_IGNORED",
                "SUPERPOSE_MAP transverse/rotation operands are kept verbatim but not modelled "
                "(TraceWin honours them only with SUPERPOSE_MAP_OUT)",
            )
        self.pending_superpose.append((kw["z0"], list(params), self.line_no))

    def _flush_cluster(self) -> None:
        while self.pending_superpose:
            _z0, sp_args, ln = self.pending_superpose.pop(0)
            d = self._directive("SUPERPOSE_MAP", sp_args, "superpose")
            self.report.add(
                FidelityClass.LOSSY,
                "SUPERPOSE_DANGLING",
                "SUPERPOSE_MAP not followed by a FIELD_MAP — placement kept verbatim, ignored",
                element=d.name,
                kind="Directive",
                line=ln,
            )
        if not self.open_cluster:
            return
        children = list(self.open_cluster)
        self.open_cluster.clear()
        if len(children) == 1 and abs(children[0][0]) < 1e-12:
            # "SUPERPOSE_MAP 0 as furniture": the plain element with writer provenance (HELIX idiom)
            self._add(children[0][1])
            return
        span_mm = max(z0 + fm.length / MM for z0, fm in children)
        names = []
        for z0, fm in children:
            nm = self.lat.add_element(fm)
            self.report.exact(nm, fm.kind)
            names.append((z0 * MM, nm))
        cont = Superposition(
            name=self._auto_name("SUPERPOSE"),
            length=span_mm * MM,
            children=names,
            provenance=Provenance(
                format="tracewin", file=str(self.path), line=self.line_no, original_type="SUPERPOSE_MAP"
            ),
        )
        self._add(cont)

    # -- finish -----------------------------------------------------------------------------
    def _finish(self) -> None:
        self._flush_pending_edge()
        self._flush_cluster()
        lat = self.lat
        lat.lines[self.name] = self.line
        lat.use = self.name
        rf = self.first_freq_hz if self.first_freq_hz is not None else self.default_freq_hz
        lat.reference = ReferenceParticle(species=self.sp, kinetic_energy_eV=self.ke, rf_frequency_Hz=rf)
        lat.meta.update(self.meta)
        if self.n_raw_phases and not self.species_given:
            self.report.add(
                FidelityClass.EQUIVALENT,
                "SPECIES_ASSUMED",
                f"{self.n_raw_phases} raw RF phase(s) interpreted for proton (no species given; "
                "negative species need a π shift)",
            )
        if self.postpass:
            self._postpass()

    def _postpass(self) -> None:
        """Conversions that need the local reference particle (rigidity, β) at each element."""
        todo = {id(e): (what, data) for what, e, data in self.postpass}
        for p in propagate(self.lat, warnings=[]):
            item = todo.get(id(p.element))
            if item is None:
                continue
            what, data = item
            e = p.element
            ref = p.ref_in
            if what == "bend":
                k1 = k1_from_field_index(data["N"], data["rho_m"])
                e.multipole.Bn[1] = gradient_from_k1(k1, ref)
            elif what == "kicker":
                brho = ref.brho_signed
                if e.electric:
                    # electric rigidity Eρ = βc·Bρ [V]; same-plane kick (HELIX steerer.py:41-45)
                    erho = ref.beta * C_LIGHT * brho
                    e.hkick = data["bl_x"] / erho
                    e.vkick = data["bl_y"] / erho
                else:
                    # magnetic: crossed Lorentz kick Δx' = ∫By·dl/Bρ, Δy' = ∫Bx·dl/Bρ (HELIX steerer.py:46-48)
                    e.hkick = data["bl_y"] / brho
                    e.vkick = data["bl_x"] / brho
            elif what == "ncells":
                kw = data["kw"]
                beta = abs(kw["beta_g"]) if kw["beta_g"] < 0 else ref.beta
                e.length = _ncells_length(kw, beta, data["freq"])
                e.rf.voltage_V = kw["eot"] * e.length
                self.report.add(
                    FidelityClass.EQUIVALENT,
                    "NCELLS_LENGTH_ESTIMATED",
                    f"NCELLS βg={kw['beta_g']:g}: cell length taken from "
                    f"{'|βg|' if kw['beta_g'] < 0 else 'the entrance β'} ({beta:.4f}); TraceWin resolves "
                    "it from the running velocity",
                    element=e.name,
                    kind="NCells",
                )


def _ncells_length(kw: dict, beta: float, freq_hz: float) -> float:
    """Total NCELLS length [m] per HELIX ``ncells.py`` (``_cell_dims``): interior cells βλ (2π mode),
    βλ/2 (π mode); π&2π mode: interior βλ with 0.75·βλ end cells."""
    lam = C_LIGHT / freq_hz
    mode, n = kw["mode"], kw["n_cells"]
    total = 0.0
    for k in range(n):
        kind = "middle" if n == 1 else ("input" if k == 0 else ("output" if k == n - 1 else "middle"))
        if mode == 2 and kind in ("input", "output"):
            total += 0.75 * beta * lam
        elif mode == 1:
            total += 0.5 * beta * lam
        else:
            total += beta * lam
    return total


def _coerce_option(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class Reader:
    format = "tracewin"

    def read(
        self,
        path: Path,
        *,
        strict: bool = False,
        species: str | Species | None = None,
        kinetic_energy_eV: float = 2.1e6,
        frequency_Hz: float | None = None,
        base_dir: str | Path | None = None,
        name: str | None = None,
        **_ignored,
    ) -> tuple[Lattice, FidelityReport]:
        """Parse a TraceWin deck.

        ``species`` defaults to proton; when the deck carries raw (non-sync) RF phases and no
        species was given, an EQUIVALENT ``SPECIES_ASSUMED`` entry is recorded.  ``frequency_Hz``
        replaces TraceWin's 352.21 MHz default for RF cards that precede any FREQ card.
        ``base_dir`` resolves ``FIELD_MAP`` files (default: the deck's directory).  In strict mode
        the first LOSSY/DROPPED entry raises :class:`~lattix.fidelity.TranslationError`.
        """
        p = _Parser(
            Path(path),
            species=species,
            kinetic_energy_eV=_coerce_float(kinetic_energy_eV, 2.1e6),
            frequency_Hz=None if frequency_Hz in (None, "") else float(frequency_Hz),
            base_dir=base_dir,
            name=name,
        )
        lat, rep = p.parse()
        rep.raise_if(strict)
        return lat, rep


def read(path: str | Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
