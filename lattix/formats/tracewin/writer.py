"""TraceWin ``.dat`` writer (PLAN §6 task 1.3).

Provenance: ported from HELIX ``linac_gen/io/tracewin_writer.py`` — ``_emit_card`` trailing-
default elision (defaults from :data:`~lattix.formats.tracewin.syntax.SCHEMA`), idempotent
``FREQ`` auto-emission before RF elements whose frequency differs from the current state,
``FIELD_MAP`` re-emission behind a stateful ``FIELD_MAP_PATH`` made relocatable with
``best_relpath`` (HELIX ``portable_paths``, relative to the OUTPUT directory), latin-1 output.

New here (PLAN 1.3): ``EDGE`` cards are generated from ``Bend.e1/e2/hgap/edge_int`` (HELIX's
BEND round trip lost the pole faces); a thick ``RFCavity`` becomes DRIFT + GAP + DRIFT
(EQUIVALENT ``THICK_CAVITY_AS_GAP``); ``TITLE`` is written as a ``; TITLE`` comment because a
real TraceWin rejects the card (measured, docs/oracles.md); ``RULES`` covers every IR kind and
every placed element gets exactly one fidelity entry.

Conventions: lengths mm, angles deg, frequencies MHz, ``%.15g`` (HELIX writes ``%.10g``,
which costs 4e-8 on a converted k1 — measured in gate A1); the RF phase is
:func:`lattix.ir.rf.tracewin_phase_deg` (raw charge-signed phase, or the synchronous phase when
a ``SET_SYNC_PHASE`` precedes the card — emitted automatically for ``rf.phase_is_sync`` elements);
``THIN_STEERING`` operands are ``∫B·dl = kick·Bρ_signed`` crossed back (∫By ↔ hkick); the BEND
field index is ``N = −(G/Bρ_signed)·ρ²`` with the local rigidity at the element entrance.
Labels: ``NAME: CARD …`` when the element has a deck label or a non-auto name expressible as a
label, else a ``; lattix: name="…" type="…"`` comment tag that the reader parses back.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path

from lattix.fidelity import FidelityClass, FidelityReport
from lattix.formats.base import rule_table
from lattix.formats.tracewin.fieldmap_files import best_relpath, has_electric_channel
from lattix.formats.tracewin.syntax import DEFAULT_FREQ_MHZ, HARDWARE_MARKER_CARDS, SCHEMA
from lattix.ir.elements import Element, FieldMap
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.normalize import field_index_from_k1, k1_from_gradient
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.rf import tracewin_phase_deg
from lattix.ir.units import C_LIGHT, DEG, MEV, MHZ, MM
from lattix.ir.walk import propagate

_RF_LIKE = frozenset({"RFCavity", "FieldMap", "NCells", "RFQCell", "Superposition"})
_LABEL_OK = re.compile(r"^[A-Za-z][^\s:;]*$")
_AUTO_NAME = re.compile(r"^[A-Z][A-Z0-9_]*_\d{4}(?:_\d+)?$")
_UG_PER_CM2 = 1e-5
_K1_DEFAULT = 0.45
_K2_DEFAULT = 2.80


def _fmt(v: float) -> str:
    s = f"{float(v):.15g}"
    return "0" if s == "-0" else s


def _quote(tok: str) -> str:
    return f'"{tok}"' if any(c.isspace() for c in tok) else tok


def _aperture_mm(e: Element) -> tuple[float, float | None, bool]:
    """``(R_mm, Ry_mm or None, is_rect)`` from an element's ApertureP (0 = no aperture).

    A circle (or an ellipse whose half-sizes agree) reports ``Ry=None``; a rectangle always
    reports both half-sizes; a genuine ellipse reports both (callers decide how to degrade)."""
    ap = e.aperture
    if ap is None or ap.x_limits is None:
        return 0.0, None, False
    hx = ap.half_x or 0.0
    hy = ap.half_y
    if ap.shape == "RECTANGULAR":
        return hx / MM, (hy if hy is not None else hx) / MM, True
    if hy is None or abs(hy - hx) <= 1e-15:
        return hx / MM, None, False
    return hx / MM, hy / MM, False


class _Emitter:
    def __init__(self, lattice: Lattice, out_dir: Path, report: FidelityReport, options: dict):
        self.lat = lattice
        self.out_dir = str(out_dir)
        self.rep = report
        self.lines: list[str] = []
        self.freq_hz: float | None = None
        self.sync_armed = False
        self.pending_sync_line = False  # a SET_SYNC_PHASE directive deferred to its RF card
        self.emitted_fmp: str | None = None
        self.warned_dirs: set[str] = set()
        self.charge = lattice.reference.species.charge
        self.header_freq = options.get("frequency_Hz")
        self.options = dict(options)
        self.placed = propagate(lattice, warnings=[])

    # -- helpers ---------------------------------------------------------------------------------

    # TraceWin counts these cards inside a ``LATTICE n`` cell (manual: DIAG_*, APERTURE and
    # THIN_STEERING are not counted; markers/commands neither)
    _LATTICE_COUNTED = frozenset({"DRIFT", "QUAD", "SOLENOID", "GAP", "FIELD_MAP", "BEND", "EDGE", "NCELLS",
                                  "RFQ_CELL", "DTL_CEL", "CAVSIN", "MULTIPOLE", "STEERER", "THIN_LENS",
                                  "SEXTUPOLE", "OCTUPOLE", "CHOPPER_MAGNET"})

    @staticmethod
    def _card_keyword(line: str) -> str:
        body = line.split(";", 1)[0].strip()
        if not body:
            return ""
        if ":" in body.split()[0] or (len(body.split()) > 1 and body.split()[1].startswith(":")):
            body = body.split(":", 1)[1].strip()
        return body.split()[0].upper() if body else ""

    def _recount_lattice_cards(self) -> None:
        """Rewrite ``LATTICE n …`` so n matches the element cards actually emitted up to the
        matching ``LATTICE_END`` — the count depends on how this writer splits elements."""
        changed = 0
        for i, line in enumerate(self.lines):
            if self._card_keyword(line) != "LATTICE":
                continue
            n = 0
            for later in self.lines[i + 1:]:
                kw = self._card_keyword(later)
                if kw == "LATTICE_END":
                    break
                if kw in self._LATTICE_COUNTED:
                    n += 1
            body = line.split(";", 1)[0]
            prefix = body[: body.upper().index("LATTICE")]
            args = body.upper().split("LATTICE", 1)[1].split()
            if not args:
                continue
            if args[0] != str(n):
                self.lines[i] = (prefix + "LATTICE " + " ".join([str(n)] + args[1:])).rstrip()
                changed += 1
        if changed:
            self.rep.equivalent("LATTICE_COUNT_RECOMPUTED",
                                f"{changed} LATTICE card(s) renumbered to the emitted element count")

    def render(self) -> str:
        self._recount_lattice_cards()
        # the deck carries no beam: the reference particle travels in a comment the reader parses
        self.lines.append(format_reference_tag(self.lat.reference, ";"))
        if self.header_freq:
            self._freq(float(self.header_freq), force=True)
        for p in self.placed:
            if self.pending_sync_line and p.element.kind not in _RF_LIKE:
                self.flush_sync()  # keeps the deck order for non-RF cards
            RULES[p.element.kind](self, p)
        if self.pending_sync_line:
            self.flush_sync()
        self.lines.append("END")
        return "\n".join(self.lines) + "\n"

    def flush_sync(self) -> None:
        self.lines.append("SET_SYNC_PHASE")
        self.pending_sync_line = False

    def emit_card(
        self, keyword: str, required: list[str], optionals: list[tuple[str, bool]], prefix: str = ""
    ) -> None:
        """``keyword`` + required tokens + trailing optionals up to the last non-default one."""
        last = -1
        for i, (_, is_default) in enumerate(optionals):
            if not is_default:
                last = i
        tokens = list(required) + [tok for tok, _ in optionals[: last + 1]]
        self.lines.append(prefix + keyword + (" " + " ".join(tokens) if tokens else ""))

    def label(self, e: Element) -> str:
        """``"NAME: "`` prefix for a labelled element, else "" (with a ``; lattix:`` tag when the
        name cannot be a TraceWin label)."""
        prov = e.provenance
        if prov is not None and prov.format == "tracewin":
            name = prov.original_name
            if not name:
                return ""
        else:
            name = e.name
            if _AUTO_NAME.match(name):
                return ""
        if _LABEL_OK.match(name):
            return f"{name}: "
        self.lines.append(f'; lattix: name="{name}" type="{e.kind}"')
        return ""

    def _freq(self, f_hz: float, *, force: bool = False) -> None:
        if force or f_hz != self.freq_hz:
            self.lines.append(f"FREQ {_fmt(f_hz / MHZ)}")
            self.freq_hz = f_hz

    def ensure_freq(self, e: Element) -> None:
        f = (
            getattr(e.rf, "frequency_Hz", None)
            or self.freq_hz
            or self.lat.reference.rf_frequency_Hz
            or DEFAULT_FREQ_MHZ * MHZ
        )
        self._freq(f)

    def sync_before(self, e: Element, *, consume: bool = True) -> bool:
        """Arm SET_SYNC_PHASE for an RF card whose IR phase is synchronous; returns the ``sync``
        flag to hand to :func:`tracewin_phase_deg`."""
        want = e.rf.phase_is_sync
        if self.pending_sync_line:
            self.flush_sync()  # the deck's own directive, now placed after any FREQ card
        elif want and not self.sync_armed:
            self.lines.append("SET_SYNC_PHASE")
            self.sync_armed = True
        use = self.sync_armed
        if use and not want:
            self.rep.equivalent(
                "SYNC_PHASE_PROMOTED",
                "a preceding SET_SYNC_PHASE makes this card's phase "
                "synchronous; the IR synchronous phase is written",
                element=e.name,
                kind=e.kind,
            )
        if consume:
            self.sync_armed = False
        return use

    def brho(self, p: Placed) -> float:
        return p.ref_in.brho_signed

    def comment_drop(
        self, p: Placed, code: str, msg: str, cls: FidelityClass = FidelityClass.DROPPED
    ) -> None:
        e = p.element
        self.lines.append(f"; lattix: {e.kind} '{e.name}' not representable in TraceWin ({msg})")
        self.rep.add(cls, code, msg, element=e.name, kind=e.kind)

    def fm_file_token(self, e: FieldMap) -> str | None:
        """The file token of a FIELD_MAP card, emitting the stateful ``FIELD_MAP_PATH`` first when
        the map directory changes; ``None`` (with a DROPPED entry) when the map has no provenance."""
        if e.geom is None or not e.files:
            self.lines.append(f"; lattix: FieldMap '{e.name}' has no geom/file provenance, cannot re-export")
            self.rep.dropped(
                "FIELDMAP_NO_SOURCE",
                "FieldMap without geom/files cannot be written as FIELD_MAP",
                element=e.name,
                kind=e.kind,
            )
            return None
        map_dir = e.meta.get("field_map_dir")
        map_name = e.files[0]
        if map_dir:
            map_dir = os.path.abspath(str(map_dir))
            rel_dir, ok = best_relpath(map_dir, self.out_dir)
            if not ok:
                if map_dir not in self.warned_dirs:
                    self.warned_dirs.add(map_dir)
                    warnings.warn(
                        f"FIELD_MAP files under {map_dir!r} share no usable ancestor with the output "
                        f"directory {self.out_dir!r} — writing an absolute FIELD_MAP_PATH; the deck is "
                        "not relocatable",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.lat.warnings.append(f"FIELD_MAP_PATH {map_dir} written absolute (not relocatable)")
                if map_dir != self.emitted_fmp:
                    self.lines.append(f"FIELD_MAP_PATH {_quote(map_dir)}")
                    self.emitted_fmp = map_dir
            elif map_dir == self.out_dir and self.emitted_fmp is None:
                pass
            elif map_dir != self.emitted_fmp:
                self.lines.append(f"FIELD_MAP_PATH {_quote(rel_dir)}")
                self.emitted_fmp = map_dir
        return _quote(map_name)

    def field_map_line(self, e: FieldMap, sync: bool, file_token: str, prefix: str = "") -> None:
        """``FIELD_MAP geom L φ R kb ke ki ka file [p_flag]``."""
        R, _ry, _rect = _aperture_mm(e)
        required = [
            str(int(e.geom)),
            _fmt(e.length / MM),
            _fmt(tracewin_phase_deg(e.rf.phase_rad, self.charge, sync)),
            _fmt(R),
            _fmt(e.kb),
            _fmt(e.ke),
            _fmt(e.ki),
            _fmt(e.ka),
            file_token,
        ]
        self.emit_card("FIELD_MAP", required, [(str(int(e.p_flag)), int(e.p_flag) == 0)], prefix)

    def misalign_check(self, e: Element, *, allow_xy: bool = False) -> None:
        s = e.shift
        if s is None or s.is_zero():
            return
        dropped = [s.z_offset, s.x_rot, s.y_rot, s.tilt] + ([] if allow_xy else [s.x_offset, s.y_offset])
        if any(dropped):
            self.rep.lossy(
                "MISALIGN_DROPPED",
                "body shift/rotation has no TraceWin card slot on this element",
                element=e.name,
                kind=e.kind,
            )


# -- per-kind rules ----------------------------------------------------------------------------------
def _drift(em: _Emitter, p: Placed) -> None:
    e = p.element
    R, Ry, rect = _aperture_mm(e)
    if not rect and Ry is not None and abs(Ry - R) > 1e-12:
        em.rep.lossy(
            "APERTURE_SHAPE",
            "elliptical aperture written as a circle of the horizontal half-size",
            element=e.name,
            kind=e.kind,
        )
        Ry = None
    dx = dy = 0.0
    if e.shift is not None:
        dx, dy = e.shift.x_offset / MM, e.shift.y_offset / MM
        em.misalign_check(e, allow_xy=True)
    ry_tok = _fmt(Ry) if Ry is not None else "0"
    em.emit_card(
        "DRIFT",
        [_fmt(e.length / MM), _fmt(R)],
        [(ry_tok, Ry is None), (_fmt(dx), dx == 0.0), (_fmt(dy), dy == 0.0)],
        em.label(e),
    )
    if not (
        e.shift is not None and any((e.shift.z_offset, e.shift.x_rot, e.shift.y_rot, e.shift.tilt))
    ) and not (not rect and Ry is not None):
        em.rep.exact(e.name, e.kind)


def _quad_card(em: _Emitter, p: Placed, G: float, g3: float, g4: float, g5: float, g6: float) -> None:
    e = p.element
    mp = e.multipole
    R, _ry, _rect = _aperture_mm(e)
    skew = mp.tilt.get(1, 0.0) / DEG
    gfr = float(e.native.get("tracewin", {}).get("gfr", 0.0) or 0.0)
    em.emit_card(
        "QUAD",
        [_fmt(e.length / MM), _fmt(G), _fmt(R)],
        [
            (_fmt(skew), skew == 0.0),
            (_fmt(g3), g3 == 0.0),
            (_fmt(g4), g4 == 0.0),
            (_fmt(g5), g5 == 0.0),
            (_fmt(g6), g6 == 0.0),
            (_fmt(gfr), gfr == 0.0),
        ],
        em.label(e),
    )


def _multipole_extras_lost(mp) -> bool:
    return (
        any(mp.Bs.values())
        or any(mp.BnL.values())
        or any(mp.BsL.values())
        or any(v for n, v in mp.tilt.items() if n != 1)
        or any(v for n, v in mp.Bn.items() if n == 0 or n > 5)
    )


def _quadrupole(em: _Emitter, p: Placed) -> None:
    e = p.element
    mp = e.multipole
    _quad_card(
        em, p, mp.Bn.get(1, 0.0), mp.Bn.get(2, 0.0), mp.Bn.get(3, 0.0), mp.Bn.get(4, 0.0), mp.Bn.get(5, 0.0)
    )
    em.misalign_check(e)
    if _multipole_extras_lost(mp):
        em.rep.lossy(
            "SKEW_MULTIPOLE_DROPPED",
            "skew / integrated / high-order multipole content has no QUAD slot",
            element=e.name,
            kind=e.kind,
        )
    else:
        em.rep.exact(e.name, e.kind)


def _sextupole(em: _Emitter, p: Placed) -> None:
    e = p.element
    mp = e.multipole
    _quad_card(
        em, p, mp.Bn.get(1, 0.0), mp.Bn.get(2, 0.0), mp.Bn.get(3, 0.0), mp.Bn.get(4, 0.0), mp.Bn.get(5, 0.0)
    )
    em.misalign_check(e)
    if _multipole_extras_lost(mp):
        em.rep.lossy(
            "SKEW_MULTIPOLE_DROPPED",
            "skew / integrated multipole content has no QUAD slot",
            element=e.name,
            kind=e.kind,
        )
    else:
        em.rep.equivalent(
            "THICK_MULTIPOLE_AS_QUAD_CARD",
            f"thick {e.kind} written as a QUAD card with G=0 and the higher-order gradient in G3..G6",
            element=e.name,
            kind=e.kind,
        )


def _multipole(em: _Emitter, p: Placed) -> None:
    e = p.element
    if e.multipole.is_zero():
        em.lines.append(f"; lattix: Multipole '{e.name}' has zero strength (omitted)")
        em.rep.equivalent(
            "ZERO_MULTIPOLE_OMITTED", "zero-strength thin multipole omitted", element=e.name, kind=e.kind
        )
        return
    em.comment_drop(p, "THIN_MULTIPOLE_UNSUPPORTED", "TraceWin has no thin multipole card")


def _bend(em: _Emitter, p: Placed) -> None:
    e = p.element
    b = e.bend
    R, _ry, _rect = _aperture_mm(e)
    if b.angle == 0.0:
        em.emit_card("DRIFT", [_fmt(e.length / MM), _fmt(R)], [], em.label(e))
        em.rep.equivalent(
            "ZERO_ANGLE_BEND_AS_DRIFT", "zero-angle bend written as a DRIFT", element=e.name, kind=e.kind
        )
        return
    rho_m = e.length / abs(b.angle)
    theta = b.angle / DEG
    hv = 0
    tilt_ok = True
    if abs(abs(b.tilt_ref) - math.pi / 2) < 1e-9:
        hv = 1
        if b.tilt_ref < 0:
            # TraceWin has one vertical plane (HV=1): a tilt of −π/2 is the +π/2 plane with
            # the angle sign flipped; the EDGE angles follow sign(θ) through the rule below
            theta = -theta
    elif abs(b.tilt_ref) > 1e-12:
        tilt_ok = False
    G = e.multipole.Bn.get(1, 0.0)
    N = field_index_from_k1(k1_from_gradient(G, p.ref_in), rho_m) if G else 0.0
    nat = e.native.get("tracewin", {})
    has_edges = bool(nat.get("has_edges")) or any((b.e1, b.e2, b.hgap))
    gap_mm = 2.0 * b.hgap / MM
    # the IR values are written as they are (a zero gap makes K1/K2 inert in TraceWin, but a
    # default written here would come back as a different IR on re-read)
    k1_in = b.edge_int1
    k1_out = b.edge_int2 if b.edge_int2 is not None else b.edge_int1
    k2 = b.fringe_k2 if b.fringe_k2 is not None else _K2_DEFAULT
    rho_mm = rho_m / MM

    def edge(beta_rad: float, k1: float) -> None:
        em.emit_card(
            "EDGE",
            [_fmt(beta_rad / DEG), _fmt(rho_mm)],
            [
                (_fmt(gap_mm), gap_mm == 0.0),
                (_fmt(k1), k1 == _K1_DEFAULT),
                (_fmt(k2), k2 == _K2_DEFAULT),
                (_fmt(R), R == 0.0),
                (str(hv), hv == 0),
            ],
        )

    if has_edges:
        edge(math.copysign(1.0, b.angle) * b.e1, k1_in)   # β = sign(θ)·e, see reader
    em.emit_card(
        "BEND",
        [_fmt(theta), _fmt(rho_mm)],
        [(_fmt(N), N == 0.0), (_fmt(R), R == 0.0), (str(hv), hv == 0)],
        em.label(e),
    )
    if has_edges:
        edge(math.copysign(1.0, b.angle) * b.e2, k1_out)
    em.misalign_check(e)
    mp = e.multipole
    if not tilt_ok:
        em.rep.lossy(
            "BEND_TILT_UNSUPPORTED",
            f"tilt_ref={b.tilt_ref:.4g} rad is neither 0 nor ±π/2; written as a horizontal bend",
            element=e.name,
            kind=e.kind,
        )
    elif any(v for n, v in mp.Bn.items() if n != 1) or any(mp.Bs.values()) or any(mp.BnL.values()):
        em.rep.lossy(
            "BEND_MULTIPOLE_DROPPED",
            "only the gradient (field index) survives on a BEND card",
            element=e.name,
            kind=e.kind,
        )
    else:
        em.rep.exact(e.name, e.kind)


def _solenoid(em: _Emitter, p: Placed) -> None:
    e = p.element
    R, _ry, _rect = _aperture_mm(e)
    em.lines.append(f"{em.label(e)}SOLENOID {_fmt(e.length / MM)} {_fmt(e.solenoid.Bsol_T)} {_fmt(R)}")
    em.misalign_check(e)
    em.rep.exact(e.name, e.kind)


def _rf_voltage(e: Element) -> float:
    rf = e.rf
    if rf.gradient_V_per_m is not None and not rf.voltage_V:
        return rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else e.length)
    return rf.voltage_V


def _rfcavity(em: _Emitter, p: Placed) -> None:
    e = p.element
    R, _ry, _rect = _aperture_mm(e)
    em.ensure_freq(e)
    p_flag = int(e.native.get("tracewin", {}).get("p_flag", 0) or 0)
    thick = e.length > 0.0
    if thick:
        em.lines.append(f"DRIFT {_fmt(e.length / 2 / MM)} {_fmt(R)}")
    sync = em.sync_before(e)
    phase = tracewin_phase_deg(e.rf.phase_rad, em.charge, sync)
    em.emit_card(
        "GAP", [_fmt(_rf_voltage(e)), _fmt(phase), _fmt(R)], [(str(p_flag), p_flag == 0)], em.label(e)
    )
    if thick:
        em.lines.append(f"DRIFT {_fmt(e.length / 2 / MM)} {_fmt(R)}")
        em.rep.equivalent(
            "THICK_CAVITY_AS_GAP",
            f"thick cavity (L={e.length:.4g} m) written as DRIFT L/2 + GAP + DRIFT L/2",
            element=e.name,
            kind=e.kind,
        )
    else:
        em.rep.exact(e.name, e.kind)
    em.misalign_check(e)


def _fieldmap(em: _Emitter, p: Placed) -> None:
    e = p.element
    if em.options.get("static_maps") == "hard_edge" and \
            ((e.meta or {}).get("map_summary") or {}).get("kind") in ("solenoid", "quad"):
        # write option ``static_maps="hard_edge"``: a static magnetic map becomes the hard-edge
        # magnet of lattix.ir.fieldmap.replacement_for (∫B and ∫B² preserved, drift padding), for
        # engines without static field maps (LightWin's Envelope3D refuses geometry 10)
        from lattix.ir.fieldmap import replacement_for

        r = replacement_for(e)
        for part in r.parts:
            RULES[part.kind](em, p.model_copy(update={"element": part}))
        em.rep.equivalent(r.code, r.message, element=e.name, kind=e.kind, **r.details)
        return
    em.ensure_freq(e)
    tok = em.fm_file_token(e)  # FIELD_MAP_PATH (if any) precedes SET_SYNC_PHASE / SUPERPOSE_MAP
    if tok is None:
        return
    sync = em.sync_before(e)
    sp = e.native.get("tracewin", {}).get("superpose")
    if sp is not None:
        em.lines.append("SUPERPOSE_MAP " + " ".join(sp))
    em.field_map_line(e, sync, tok, em.label(e))
    em.misalign_check(e)
    em.rep.exact(e.name, e.kind)


def _ncells(em: _Emitter, p: Placed) -> None:
    e = p.element
    kw = e.params
    fields = SCHEMA["NCELLS"]
    if any(f.name not in kw for f in fields if f.required):
        em.comment_drop(p, "NCELLS_PARAMS_MISSING", "NCells.params lacks the TraceWin operands")
        return
    em.ensure_freq(e)
    em.sync_before(e)
    required = [_fmt(kw[f.name]) if f.cast is float else str(int(kw[f.name])) for f in fields if f.required]
    optionals = []
    for f in fields:
        if f.required:
            continue
        v = kw.get(f.name, f.default)
        tok = _fmt(v) if f.cast is float else str(int(v))
        optionals.append((tok, v == f.elide_value))
    for v in kw.get("ttf_tail", []) or []:
        optionals.append((_fmt(v), False))
    em.emit_card("NCELLS", required, optionals, em.label(e))
    em.rep.exact(e.name, e.kind)


def _rfqcell(em: _Emitter, p: Placed) -> None:
    e = p.element
    kw = e.params
    fields = SCHEMA["RFQ_CELL"]
    if any(f.name not in kw for f in fields if f.required):
        em.comment_drop(p, "RFQ_PARAMS_MISSING", "RFQCell.params lacks the TraceWin operands")
        return
    em.ensure_freq(e)
    required = [_fmt(kw[f.name]) if f.cast is float else str(int(kw[f.name])) for f in fields if f.required]
    optionals = [
        (_fmt(kw.get(f.name, f.default)), kw.get(f.name, f.default) == f.elide_value)
        for f in fields
        if not f.required
    ]
    em.emit_card("RFQ_CELL", required, optionals, em.label(e))
    em.rep.exact(e.name, e.kind)


def _kicker(em: _Emitter, p: Placed) -> None:
    """Corrector.  TraceWin's THIN_STEERING is thin and is NOT counted in LATTICE cells
    (manual); a thick corrector with no kick — every corrector in the PIP-II design
    exports — is exactly its body drift, and a thick corrector with a kick becomes
    DRIFT L/2 + THIN_STEERING + DRIFT L/2 (kick at the centre, length preserved)."""
    e = p.element
    ref = p.ref_in
    brho = ref.brho_signed
    if e.electric:
        erho = ref.beta * C_LIGHT * brho
        bx, by = e.hkick * erho, e.vkick * erho
    else:
        bx, by = e.vkick * brho, e.hkick * brho
    R, _ry, _rect = _aperture_mm(e)
    elec = 1 if e.electric else 0
    has_kick = bool(e.hkick or e.vkick)
    L_mm = e.length / MM
    if e.length and not has_kick:
        em.emit_card("DRIFT", [_fmt(L_mm), _fmt(R)], [], em.label(e))
        em.rep.exact(e.name, e.kind, message="zero-kick corrector written as its body drift")
        return
    if e.length:
        em.emit_card("DRIFT", [_fmt(L_mm / 2), _fmt(R)], [], "")
    em.emit_card(
        "THIN_STEERING", [_fmt(bx), _fmt(by)], [(_fmt(R), R == 0.0), (str(elec), elec == 0)], em.label(e)
    )
    if e.length:
        em.emit_card("DRIFT", [_fmt(L_mm / 2), _fmt(R)], [], "")
        em.rep.equivalent(
            "THICK_KICKER_SPLIT",
            f"kicker body {e.length:.4g} m written as DRIFT + THIN_STEERING + DRIFT (kick at the centre)",
            element=e.name,
            kind=e.kind,
        )
    else:
        em.rep.exact(e.name, e.kind)


def _collimator(em: _Emitter, p: Placed) -> None:
    e = p.element
    R, Ry, rect = _aperture_mm(e)
    if e.aperture is None:
        em.lines.append(f"; lattix: Collimator '{e.name}' has no aperture (omitted)")
        if e.length:
            em.lines.append(f"DRIFT {_fmt(e.length * 1e3)} 0")
        em.rep.equivalent(
            "COLLIMATOR_NO_APERTURE",
            "collimator without aperture limits omitted" + (" (body kept as a DRIFT)" if e.length else ""),
            element=e.name,
            kind=e.kind,
        )
        return
    t = int(e.native.get("tracewin", {}).get("ap_type", 0 if rect else 1))
    dy = Ry if Ry is not None else R
    em.lines.append(f"{em.label(e)}APERTURE {_fmt(R)} {_fmt(dy)} {t}")
    if e.length:
        # TraceWin's APERTURE is thin: the aperture acts at the entrance and the body is a DRIFT
        # of the same length and radius, so the lattice keeps its length (invariant I-1)
        em.lines.append(f"DRIFT {_fmt(e.length * 1e3)} {_fmt(R)}")
        em.rep.equivalent(
            "THICK_COLLIMATOR_AS_THIN",
            f"collimator written as APERTURE at the entrance + DRIFT {e.length:.4g} m",
            element=e.name,
            kind=e.kind,
        )
    else:
        em.rep.exact(e.name, e.kind)


def _marker(em: _Emitter, p: Placed) -> None:
    e = p.element
    nat = e.native.get("tracewin", {})
    card = nat.get("card", "MARKER")
    args = nat.get("args", [])
    em.lines.append(f"{em.label(e)}{card}" + (" " + " ".join(args) if args else ""))
    em.rep.exact(e.name, e.kind)


def _instrument(em: _Emitter, p: Placed) -> None:
    e = p.element
    nat = e.native.get("tracewin")
    if nat and "card" in nat:
        if nat.get("label_only"):
            em.lines.append(f"{nat.get('raw', nat['card'])} :")
        else:
            args = nat.get("args", [])
            em.lines.append(f"{em.label(e)}{nat['card']}" + (" " + " ".join(args) if args else ""))
        em.rep.exact(e.name, e.kind)
        return
    fam = (e.family or "MONITOR").upper()
    if fam in ("MONITOR", "HMONITOR", "VMONITOR", "BPM"):
        em.lines.append("BPM :")
    elif fam in HARDWARE_MARKER_CARDS:
        em.lines.append(f"{fam} :")
    else:
        em.lines.append(f"{em.label(e)}MARKER")
    if e.length:
        R, _ry, _rect = _aperture_mm(e)
        em.lines.append(f"DRIFT {_fmt(e.length / MM)} {_fmt(R)}")      # the body keeps its length
    em.rep.equivalent(
        "INSTRUMENT_AS_MARKER",
        f"Instrument family {fam!r} written as a TraceWin marker"
        + (f" followed by a DRIFT of {e.length:.4g} m" if e.length else ""),
        element=e.name,
        kind=e.kind,
    )


def _foil(em: _Emitter, p: Placed) -> None:
    e = p.element
    strag = e.native.get("tracewin", {}).get("straggling", "auto")
    tok = f" {strag}" if strag and strag != "auto" else ""
    em.lines.append(f"; HELIX_FOIL {e.name} {e.material} {_fmt(e.thickness_kg_per_m2 / _UG_PER_CM2)}{tok}")
    em.rep.lossy(
        "FOIL_AS_COMMENT",
        "foil written as a HELIX comment card (TraceWin ignores it)",
        element=e.name,
        kind=e.kind,
    )


def _taylor(em: _Emitter, p: Placed) -> None:
    em.comment_drop(p, "TAYLOR_UNSUPPORTED", "TraceWin has no explicit map card")


def _patch(em: _Emitter, p: Placed) -> None:
    em.comment_drop(p, "PATCH_UNSUPPORTED", "TraceWin has no coordinate patch")


def _reference_change(em: _Emitter, p: Placed) -> None:
    e = p.element
    nat = e.native.get("tracewin")
    if nat and nat.get("card"):
        em.lines.append(nat["card"] + (" " + " ".join(nat.get("args", [])) if nat.get("args") else ""))
    elif e.energy_eV is not None:
        em.lines.append(f"SET_BEAM_ENERGY 1 {_fmt(e.energy_eV / MEV)}")
    elif e.dE_ref_eV is not None:
        em.lines.append(f"SET_BEAM_E0_P0 1 {_fmt(e.dE_ref_eV / MEV)} 0 1 0")
    else:
        em.lines.append(f"; lattix: ReferenceChange '{e.name}' carries no change (omitted)")
    em.rep.exact(e.name, e.kind)


def _freq(em: _Emitter, p: Placed) -> None:
    em._freq(p.element.frequency_Hz, force=True)
    em.rep.exact(p.element.name, p.element.kind)


def _directive(em: _Emitter, p: Placed) -> None:
    e = p.element
    if e.format != "tracewin":
        em.lines.append(f"; lattix: directive {e.format} {e.card} {' '.join(e.args)}".rstrip())
        em.rep.dropped(
            "FOREIGN_DIRECTIVE",
            f"{e.format} directive {e.card!r} has no TraceWin form",
            element=e.name,
            kind=e.kind,
        )
        return
    card = e.card
    args = " ".join(e.args)
    if card == "TITLE":
        em.lines.append(f"; TITLE {args}".rstrip())
    elif card == "@LG":
        em.lines.append(f";@LG {args}")
    elif card == "HELIX_SC_GRID":
        em.lines.append(f"; HELIX_SC_GRID {args}")
    elif card == "SET_SYNC_PHASE":
        # deferred to the RF card it binds to, so an auto-emitted FREQ never separates them
        em.sync_armed = True
        em.pending_sync_line = True
    else:
        em.lines.append(card + (" " + args if args else ""))
    em.rep.exact(e.name, e.kind)


def _superposition(em: _Emitter, p: Placed) -> None:
    e = p.element
    for z0_m, nm in e.children:
        child = em.lat.elements.get(nm)
        if not isinstance(child, FieldMap):
            em.lines.append(f"; lattix: Superposition child '{nm}' is not a FieldMap (omitted)")
            em.rep.lossy(
                "SUPERPOSE_CHILD_UNSUPPORTED",
                f"child {nm!r} of {e.name!r} is not a FieldMap",
                element=e.name,
                kind=e.kind,
            )
            continue
        em.ensure_freq(child)
        tok = em.fm_file_token(child)
        if tok is None:
            continue
        consume = has_electric_channel(child.geom) if child.geom is not None else True
        sync = em.sync_before(child, consume=consume) if consume else False
        sp = child.native.get("tracewin", {}).get("superpose")
        if sp and abs(float(sp[0]) * MM - z0_m) < 1e-12:
            em.lines.append("SUPERPOSE_MAP " + " ".join(sp))
        else:
            em.lines.append(f"SUPERPOSE_MAP {_fmt(z0_m / MM)}")
        em.field_map_line(child, sync, tok, em.label(child))
        em.rep.exact(child.name, child.kind)
    em.rep.exact(e.name, e.kind)


RULES = rule_table(
    Drift=_drift,
    Quadrupole=_quadrupole,
    Sextupole=_sextupole,
    Octupole=_sextupole,
    Multipole=_multipole,
    Bend=_bend,
    Solenoid=_solenoid,
    RFCavity=_rfcavity,
    FieldMap=_fieldmap,
    NCells=_ncells,
    RFQCell=_rfqcell,
    Kicker=_kicker,
    Collimator=_collimator,
    Marker=_marker,
    Instrument=_instrument,
    Foil=_foil,
    Taylor=_taylor,
    Patch=_patch,
    ReferenceChange=_reference_change,
    Freq=_freq,
    Directive=_directive,
    Superposition=_superposition,
)


class Writer:
    format = "tracewin"
    RULES = RULES

    def write(self, lattice: Lattice, path: Path, *, strict: bool = False, **options) -> FidelityReport:
        """Write ``lattice`` as a TraceWin deck.  Options: ``frequency_Hz`` (header FREQ card).
        In strict mode the first LOSSY/DROPPED entry raises before anything is written."""
        path = Path(path)
        text, rep = render(lattice, path.parent, **options)
        rep.target_file = str(path)
        rep.raise_if(strict)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="latin-1", errors="replace")
        return rep


def render(lattice: Lattice, out_dir: str | Path | None = None, **options) -> tuple[str, FidelityReport]:
    """The deck text (and its fidelity report) without touching the disk; ``out_dir`` anchors the
    relative ``FIELD_MAP_PATH`` cards (default: the current directory)."""
    rep = FidelityReport(target_format="tracewin")
    out = Path(out_dir) if out_dir is not None else Path.cwd()
    em = _Emitter(lattice, out.resolve(), rep, options)
    return em.render(), rep


def write(lattice: Lattice, path: str | Path, *, strict: bool = False, **options) -> FidelityReport:
    return Writer().write(lattice, Path(path), strict=strict, **options)
