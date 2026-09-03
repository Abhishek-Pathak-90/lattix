"""MAD-X reader — cpymad-backed and authoritative (PLAN §6 task 1.4).

The deck is handed to a real MAD-X process (``Madx.call``), so ``CALL`` chains,
macros, ``IF``, variables and deferred ``:=`` expressions all behave exactly as
MAD-X defines them.  The IR is then built from ``sequence.expanded_elements``,
i.e. from the *expanded* sequence including MAD-X's own implicit drifts
(``drift_0``, ``drift_1``, …), so the element boundaries are MAD-X's own.

Conventions and facts measured on MAD-X 5.09.03 / cpymad 1.19.0 (2026-09-03):

* **rigidity** — MAD-X keeps ``p0`` constant, so every normalized strength in a
  deck refers to the one BEAM rigidity; ``Bρ_signed = sign(q)·pc/(|q|c)`` turns
  it into the IR's lab field (HELIX ``madx_parser._signed_brho``), which is what
  flips an H⁻ deck's gradients against a proton deck's.
* **rbend length** — ``option, rbarc`` defaults to true, so a deck's ``l`` on an
  RBEND is the *chord*; the node occupies the **arc**
  ``L_arc = L_chord·(θ/2)/sin(θ/2)`` (verified: ``br: rbend, l=0.8, angle=0.08``
  → twiss ``l = 0.800213373``).  The IR always stores the arc length, and
  ``BendP.rect`` records that the source was rectangular.  RBEND pole faces are
  additionally referenced to the chord, so the reader converts them to the IR's
  sector reference with ``e1 += angle/2``, ``e2 += angle/2``.
* **skew quadrupoles** — measured: ``quadrupole, k1, k1s`` is *exactly* the
  normal quadrupole ``k1_eff = hypot(k1, k1s)`` rotated by
  ``-atan2(k1s, k1)/2`` (``k1s`` alone → tilt ``-π/4``), so the reader folds
  ``k1s`` into ``MagneticMultipoleP.tilt[1]`` and keeps a single ``Bn[1]``.
* **RF** — ``lag`` is in turns with gain ``V·sin(2π·lag)``; measured on a 1 MV
  gap at ``lag = 1/6``: 866 025.4037844387 eV for both a proton and a charge −1
  ion, i.e. the species-independent ``V·cos φ`` the IR defines
  (:func:`lattix.ir.rf.phase_from_madx_lag`).
* **``fintx``** defaults to −1, meaning "same as ``fint``" → IR ``edge_int2 =
  None``.
* **sequence bookkeeping** — MAD-X wraps every expanded sequence in
  ``<seq>$start`` / ``<seq>$end`` markers.  They carry no physics, cannot be
  re-declared by a deck (``$`` is not writable) and are regenerated on write, so
  the reader drops them and records the raw row count in
  ``meta["madx_expanded_rows"]``.

The lark fallback for environments without cpymad is deferred (PLAN 1.4 lists it
as the non-authoritative path); until it lands, an import failure raises.
"""
from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any

from lattix.fidelity import FidelityReport
from lattix.formats.madx.naming import parse_tags, parse_title
from lattix.ir.elements import (
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    Octupole,
    Provenance,
    Quadrupole,
    RFCavity,
    Sextupole,
    Solenoid,
    Taylor,
)
from lattix.ir.expr import Expression
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.rf import phase_from_madx_lag

#: names present in a pristine ``Madx()`` — never user variables (measured).
BUILTIN_GLOBALS: frozenset[str] = frozenset({
    "amu0", "clight", "degrad", "e", "emass", "erad", "hbar", "mumass", "nmass",
    "pi", "pmass", "prad", "qelect", "raddeg", "twiss_tol", "twopi", "umass", "version",
})

#: MAD-X creates numbered temporaries (``__0__``, ``__1__`` …) when it stores a
#: deferred expression that reaches into an element attribute (``sc->l``) or a
#: macro argument.  They are bookkeeping, not user variables, and cannot be
#: re-declared by a deck.
_MADX_TEMP_VAR = re.compile(r"^_+\d+_+$")

#: MAD-X base types that become an :class:`Instrument` with this ``family``.
_INSTRUMENT_FAMILY = {
    "monitor": "BPM", "hmonitor": "HMONITOR", "vmonitor": "VMONITOR",
    "instrument": "INSTRUMENT", "placeholder": "PLACEHOLDER",
}


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _arr(v: Any) -> list[float]:
    if v is None:
        return []
    try:
        return [float(x) for x in v]
    except TypeError:
        return []


def _sandbox(deck: Path) -> tuple[Path, Path]:
    """A working directory for MAD-X that mirrors *deck*'s folder through symlinks.

    MAD-X resolves a relative ``CALL`` against the current directory (PLAN 1.4:
    "CALL resolved relative to the deck") *and* decks routinely write files into it
    (twiss tables, ``sectormap``).  Running from a symlink mirror keeps every
    relative ``CALL`` working while any output lands in the temp directory instead
    of the user's data folder.  Falls back to the deck's own folder if the mirror
    cannot be built.
    """
    work = Path(tempfile.mkdtemp(prefix="lattix_madx_read_"))
    try:
        children = list(deck.parent.iterdir())
        if len(children) > 5000:                 # pragma: no cover - pathological folder
            raise OSError("too many entries to mirror")
        for child in children:
            (work / child.name).symlink_to(child)
    except OSError:                              # pragma: no cover - platform dependent
        return deck.parent, deck
    return work, work / deck.name


class _Row:
    """A snapshot of one expanded-sequence node (cpymad proxies die with the process)."""

    __slots__ = ("name", "base", "length", "position", "align", "field_err", "params",
                 "exprs", "inform")

    def __init__(self, el) -> None:
        self.name = str(el.name)
        self.base = str(el.base_name).lower()
        self.length = _f(el.length)
        self.position = _f(el.position)
        self.align = el.align_errors
        self.field_err = el.field_errors
        self.params: dict[str, Any] = {}
        self.exprs: dict[str, Any] = {}
        self.inform: dict[str, int] = {}
        for key, par in el.cmdpar.items():
            try:
                self.params[key] = list(par.value) if isinstance(par.value, list) else par.value
            except Exception:  # noqa: BLE001 - defensive: exotic parameter types
                continue
            self.inform[key] = int(par.inform or 0)
            if par.expr:
                self.exprs[key] = par.expr

    def get(self, key: str, default: float = 0.0) -> float:
        return _f(self.params.get(key), default)

    def arr(self, key: str) -> list[float]:
        return _arr(self.params.get(key))

    def txt(self, key: str) -> str:
        v = self.params.get(key)
        return "" if v is None else str(v)


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "madx"

    def read(self, path: Path, *, sequence: str | None = None, strict: bool = False,
             species: str | Species | None = None, kinetic_energy_eV: float | None = None,
             keep_expressions: bool = True) -> tuple[Lattice, FidelityReport]:
        try:
            from cpymad.madx import Madx
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("MAD-X reader needs cpymad (pip install cpymad)") from exc

        path = Path(path).resolve()
        text = path.read_text(encoding="utf-8", errors="replace")
        tags = parse_tags(text)
        title = parse_title(text)

        rep = FidelityReport(source_format="madx", source_file=str(path))
        warnings: list[str] = []

        workdir, entry = _sandbox(path)
        madx = Madx(stdout=False)
        try:
            madx.chdir(str(workdir))
            madx.call(str(entry))
            seq_name = self._pick_sequence(madx, sequence, warnings)
            seq = madx.sequence[seq_name]
            if not seq.is_expanded:
                madx.use(sequence=seq_name)
                seq = madx.sequence[seq_name]
            beam = {k: seq.beam[k] for k in ("particle", "mass", "charge", "energy")}
            rows = [_Row(e) for e in seq.expanded_elements]
            variables = self._variables(madx, keep_expressions)
        finally:
            madx.quit()

        ref = self._reference(beam, species, kinetic_energy_eV, warnings)
        brho = ref.brho_signed

        n_rows = len(rows)
        rows = [r for r in rows if not self._is_sequence_marker(r, seq_name)]

        elements: list[Element] = []
        for row in rows:
            el = self._convert(row, brho, rep, keep_expressions)
            tag = tags.get(row.name.lower(), {})
            el.provenance = Provenance(format="madx", file=str(path),
                                       original_name=tag.get("name", row.name),
                                       original_type=tag.get("type", row.base))
            self._apply_aperture(el, row, rep)
            self._apply_errors(el, row, rep)
            elements.append(el)

        elements = self._fold_dipedges(elements, rep)

        lat = Lattice(name=seq_name, reference=ref, variables=variables, warnings=warnings)
        line = Line(name=seq_name)
        for el in elements:
            line.items.append(LineItem(ref=self._register(lat, el)))
        lat.lines[seq_name] = line
        lat.use = seq_name
        lat.meta["source_format"] = "madx"
        lat.meta["madx_sequence"] = seq_name
        lat.meta["madx_expanded_rows"] = n_rows
        if title is not None:
            lat.meta["madx_title"] = title
        rep.raise_if(strict)
        return lat, rep

    # -- sequence / beam ----------------------------------------------------
    @staticmethod
    def _pick_sequence(madx, requested: str | None, warnings: list[str]) -> str:
        names = list(madx.sequence.keys())
        if not names:
            raise ValueError("deck defines no SEQUENCE")
        if requested is not None:
            if requested.lower() not in [n.lower() for n in names]:
                raise KeyError(f"deck has no sequence {requested!r}; known: {names}")
            return requested.lower()
        used = [n for n in names if madx.sequence[n].is_expanded]
        pool = used or names
        if len(pool) > 1:
            warnings.append(f"deck defines {len(pool)} candidate sequences {pool}; "
                            f"using the last one ({pool[-1]!r}) — pass sequence= to choose")
        return pool[-1]

    @staticmethod
    def _is_sequence_marker(row: _Row, seq_name: str) -> bool:
        return row.base == "marker" and row.name.lower() in (f"{seq_name}$start", f"{seq_name}$end")

    @staticmethod
    def _reference(beam: dict, species: str | Species | None, kinetic_energy_eV: float | None,
                   warnings: list[str]) -> ReferenceParticle:
        mass_eV = _f(beam.get("mass"), 0.0) * 1e9
        charge = int(round(_f(beam.get("charge"), 1.0)))
        total_eV = _f(beam.get("energy"), 0.0) * 1e9
        name = str(beam.get("particle", "proton")).strip().lower()

        sp: Species | None = None
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        else:
            try:
                cand = species_by_name(name)
                if abs(cand.mass_eV - mass_eV) <= 1e-6 * max(mass_eV, 1.0) and cand.charge == charge:
                    sp = cand
            except KeyError:
                sp = None
            if sp is None:
                for cand in SPECIES.values():
                    if cand.charge == charge and abs(cand.mass_eV - mass_eV) <= 1e-6 * max(mass_eV, 1.0):
                        sp = cand
                        break
            if sp is None:
                sp = Species(name=name or "ion", mass_eV=mass_eV, charge=charge)
                warnings.append(f"BEAM particle={name!r} mass={mass_eV / 1e9} GeV charge={charge} "
                                "matches no known species — kept as a custom Species")
        ke = kinetic_energy_eV if kinetic_energy_eV is not None else total_eV - sp.mass_eV
        if ke <= 0:
            warnings.append(f"BEAM gives a non-positive kinetic energy ({ke} eV); check ENERGY")
        return ReferenceParticle(species=sp, kinetic_energy_eV=ke)

    @staticmethod
    def _variables(madx, keep_expressions: bool) -> dict[str, Variable]:
        out: dict[str, Variable] = {}
        for name in madx.globals:
            key = str(name).lower()
            if key in BUILTIN_GLOBALS or _MADX_TEMP_VAR.match(key):
                continue
            par = madx.globals.cmdpar[key]
            expr = None
            if keep_expressions and par.expr:
                expr = Expression(text=str(par.expr), deferred=True, dialect="infix")
            out[key] = Variable(value=_f(par.value), expression=expr)
        return out

    @staticmethod
    def _register(lat: Lattice, el: Element) -> str:
        """Reuse a definition when an identical element name reappears (``b1`` twice)."""
        existing = lat.elements.get(el.name)
        if existing is not None and type(existing) is type(el):
            a = existing.model_dump(mode="json")
            b = el.model_dump(mode="json")
            a.pop("provenance", None)
            b.pop("provenance", None)
            if a == b:
                return el.name
        return lat.add_element(el)

    # -- per-element conversion --------------------------------------------
    def _convert(self, row: _Row, brho: float, rep: FidelityReport, keep: bool) -> Element:
        handler = getattr(self, f"_conv_{row.base}", None)
        if handler is None:
            rep.dropped("UNSUPPORTED_MADX_TYPE",
                        f"MAD-X base type {row.base!r} has no IR mapping; kept as a marker",
                        element=row.name, kind="Marker", madx_type=row.base)
            el = Marker(name=row.name, length=row.length)
            el.native["madx"] = {"base": row.base, "params": self._informed(row)}
            return el
        n_before = len(rep.entries)
        el = handler(row, brho, rep, keep)
        if len(rep.entries) == n_before:          # one ledger entry per source element
            rep.exact(row.name, el.kind)
        return el

    @staticmethod
    def _informed(row: _Row) -> dict:
        """Explicitly-set parameters of a row (``inform`` flag), for ``native``."""
        return {k: v for k, v in row.params.items()
                if row.inform.get(k) and k not in ("at", "from")}

    def _expr(self, el: Element, row: _Row, attr: str, path: str, keep: bool) -> None:
        """Record a deferred MAD-X expression.

        The text is the **normalized MAD-X quantity** (``k1 := kqf``), not the IR
        lab field, so the writer re-emits it verbatim only after checking that the
        expression's value matches the number it would otherwise print.
        """
        txt = row.exprs.get(attr)
        if not txt or not keep:
            return
        if isinstance(txt, (list, tuple)):
            txt = ",".join(str(t) for t in txt if t)
            if not txt:
                return
        el.expressions[path] = Expression(text=str(txt), deferred=True, dialect="infix")
        el.native.setdefault("madx", {})[f"{attr}_expr"] = str(txt)

    # drift ------------------------------------------------------------------
    def _conv_drift(self, row, brho, rep, keep):
        el = Drift(name=row.name, length=row.length)
        self._expr(el, row, "l", "length", keep)
        return el

    # quadrupole -------------------------------------------------------------
    def _conv_quadrupole(self, row, brho, rep, keep):
        el = Quadrupole(name=row.name, length=row.length)
        k1, k1s, tilt = row.get("k1"), row.get("k1s"), row.get("tilt")
        if k1s:
            # measured: (k1, k1s) == normal quad hypot(k1, k1s) rotated by -atan2(k1s, k1)/2
            el.multipole.Bn[1] = math.hypot(k1, k1s) * brho
            el.multipole.tilt[1] = tilt - math.atan2(k1s, k1) / 2.0
            el.native.setdefault("madx", {}).update({"k1": k1, "k1s": k1s, "tilt": tilt})
            rep.equivalent("SKEW_QUAD_AS_TILT",
                           "k1/k1s folded into one rotated normal quadrupole (exact)",
                           element=row.name, kind="Quadrupole", k1=k1, k1s=k1s)
        else:
            el.multipole.Bn[1] = k1 * brho
            if tilt:
                el.multipole.tilt[1] = tilt
            self._expr(el, row, "k1", "multipole.Bn[1]", keep)
        self._expr(el, row, "l", "length", keep)
        self._expr(el, row, "tilt", "multipole.tilt[1]", keep)
        return el

    # sextupole / octupole ---------------------------------------------------
    def _conv_sextupole(self, row, brho, rep, keep):
        return self._conv_thick_multipole(row, brho, keep, Sextupole, 2, "k2", "k2s")

    def _conv_octupole(self, row, brho, rep, keep):
        return self._conv_thick_multipole(row, brho, keep, Octupole, 3, "k3", "k3s")

    def _conv_thick_multipole(self, row, brho, keep, cls, order, kn, ks):
        el = cls(name=row.name, length=row.length)
        if row.get(kn):
            el.multipole.Bn[order] = row.get(kn) * brho
        if row.get(ks):
            el.multipole.Bs[order] = row.get(ks) * brho
        if row.get("tilt"):
            el.multipole.tilt[order] = row.get("tilt")
        self._expr(el, row, "l", "length", keep)
        self._expr(el, row, kn, f"multipole.Bn[{order}]", keep)
        self._expr(el, row, ks, f"multipole.Bs[{order}]", keep)
        self._expr(el, row, "tilt", f"multipole.tilt[{order}]", keep)
        return el

    # thin multipole ---------------------------------------------------------
    def _conv_multipole(self, row, brho, rep, keep):
        el = Multipole(name=row.name, length=0.0)
        for order, v in enumerate(row.arr("knl")):
            if v:
                el.multipole.BnL[order] = v * brho
        for order, v in enumerate(row.arr("ksl")):
            if v:
                el.multipole.BsL[order] = v * brho
        if row.get("tilt"):
            el.multipole.tilt[0] = row.get("tilt")
        lrad = row.get("lrad")
        if lrad:
            el.native.setdefault("madx", {})["lrad"] = lrad
        for key in ("knl", "ksl"):
            if key in row.exprs:       # array expressions are kept for information only
                el.native.setdefault("madx", {})[f"{key}_expr"] = row.exprs[key]
        return el

    # bends ------------------------------------------------------------------
    def _conv_sbend(self, row, brho, rep, keep):
        return self._conv_bend(row, brho, rep, keep, rect=False)

    def _conv_rbend(self, row, brho, rep, keep):
        return self._conv_bend(row, brho, rep, keep, rect=True)

    def _conv_bend(self, row, brho, rep, keep, *, rect: bool):
        angle = row.get("angle")
        # row.length is the NODE length: MAD-X reports the arc for an rbend under
        # the default `option, rbarc=true` while `l` stays the chord.
        el = Bend(name=row.name, length=row.length)
        fintx = row.get("fintx", -1.0)
        half = angle / 2.0 if rect else 0.0
        el.bend = BendP(angle=angle, e1=row.get("e1") + half, e2=row.get("e2") + half,
                        edge_int1=row.get("fint"),
                        edge_int2=None if fintx < 0 else fintx,
                        hgap=row.get("hgap"), tilt_ref=row.get("tilt"), rect=rect)
        if row.get("k1"):
            el.multipole.Bn[1] = row.get("k1") * brho
        if row.get("k2"):
            el.multipole.Bn[2] = row.get("k2") * brho
        self._expr(el, row, "angle", "bend.angle", keep)
        self._expr(el, row, "e1", "bend.e1", keep)
        self._expr(el, row, "e2", "bend.e2", keep)
        self._expr(el, row, "k1", "multipole.Bn[1]", keep)
        self._expr(el, row, "k2", "multipole.Bn[2]", keep)
        self._expr(el, row, "l", "length", keep)
        k0 = row.get("k0")
        g = el.bend.g_ref(el.length)
        if k0 and abs(k0 - g) > 1e-12 * max(1.0, abs(g)):
            el.native.setdefault("madx", {})["k0"] = k0
            rep.lossy("BEND_K0_NE_ANGLE",
                      f"k0={k0!r} differs from angle/l={g!r}; the IR keeps the geometric angle",
                      element=row.name, kind="Bend", k0=k0, g_ref=g)
        extra = {k: row.get(k) for k in ("h1", "h2", "k1s", "k2s", "k3", "k3s", "ktap")
                 if row.get(k)}
        if extra:
            el.native.setdefault("madx", {}).update(extra)
            rep.lossy("BEND_ATTR_DROPPED", f"bend attributes not modelled by the IR: {sorted(extra)}",
                      element=row.name, kind="Bend", **extra)
        return el

    def _conv_dipedge(self, row, brho, rep, keep):
        """Kept as a placeholder; :meth:`_fold_dipedges` folds it into a neighbour."""
        el = Directive(name=row.name, length=0.0, format="madx", card="dipedge", role="other",
                       args=[f"h={row.get('h')!r}", f"e1={row.get('e1')!r}",
                             f"fint={row.get('fint')!r}", f"hgap={row.get('hgap')!r}"])
        el.native["madx"] = {"base": "dipedge", "h": row.get("h"), "e1": row.get("e1"),
                             "fint": row.get("fint"), "hgap": row.get("hgap"),
                             "tilt": row.get("tilt")}
        return el

    # solenoid ---------------------------------------------------------------
    def _conv_solenoid(self, row, brho, rep, keep):
        el = Solenoid(name=row.name, length=row.length)
        el.solenoid.Bsol_T = row.get("ks") * brho
        self._expr(el, row, "ks", "solenoid.Bsol_T", keep)
        self._expr(el, row, "l", "length", keep)
        if row.get("ksi"):
            el.native.setdefault("madx", {})["ksi"] = row.get("ksi")
            rep.lossy("SOLENOID_KSI_DROPPED", "integrated solenoid strength ksi is not modelled",
                      element=row.name, kind="Solenoid", ksi=row.get("ksi"))
        return el

    # rf ---------------------------------------------------------------------
    def _conv_rfcavity(self, row, brho, rep, keep):
        el = RFCavity(name=row.name, length=row.length)
        el.rf.voltage_V = row.get("volt") * 1e6
        el.rf.phase_rad = phase_from_madx_lag(row.get("lag"))
        freq = row.get("freq")
        el.rf.frequency_Hz = freq * 1e6 if freq else None
        if row.get("harmon"):
            el.native.setdefault("madx", {})["harmon"] = int(row.get("harmon"))
        if row.length:
            el.rf.L_active_m = row.length
        self._expr(el, row, "volt", "rf.voltage_V", keep)
        self._expr(el, row, "lag", "rf.phase_rad", keep)
        self._expr(el, row, "freq", "rf.frequency_Hz", keep)
        self._expr(el, row, "l", "length", keep)
        return el

    # kickers ----------------------------------------------------------------
    def _conv_hkicker(self, row, brho, rep, keep):
        el = Kicker(name=row.name, length=row.length, hkick=row.get("kick"))
        self._expr(el, row, "kick", "hkick", keep)
        return el

    def _conv_vkicker(self, row, brho, rep, keep):
        el = Kicker(name=row.name, length=row.length, vkick=row.get("kick"))
        self._expr(el, row, "kick", "vkick", keep)
        return el

    def _conv_kicker(self, row, brho, rep, keep):
        el = Kicker(name=row.name, length=row.length, hkick=row.get("hkick"), vkick=row.get("vkick"))
        self._expr(el, row, "hkick", "hkick", keep)
        self._expr(el, row, "vkick", "vkick", keep)
        return el

    _conv_tkicker = _conv_kicker

    # markers, monitors, collimators ----------------------------------------
    def _conv_marker(self, row, brho, rep, keep):
        return Marker(name=row.name, length=row.length)

    def _conv_monitor(self, row, brho, rep, keep):
        el = Instrument(name=row.name, length=row.length, family=_INSTRUMENT_FAMILY[row.base])
        self._expr(el, row, "l", "length", keep)
        return el

    _conv_hmonitor = _conv_monitor
    _conv_vmonitor = _conv_monitor
    _conv_instrument = _conv_monitor
    _conv_placeholder = _conv_monitor

    def _conv_rcollimator(self, row, brho, rep, keep):
        el = Collimator(name=row.name, length=row.length)
        xs, ys = row.get("xsize"), row.get("ysize")
        if xs or ys:
            el.aperture = ApertureP.rect(xs, ys)
        return el

    def _conv_ecollimator(self, row, brho, rep, keep):
        el = Collimator(name=row.name, length=row.length)
        xs, ys = row.get("xsize"), row.get("ysize")
        if xs or ys:
            el.aperture = ApertureP(shape="ELLIPTICAL", x_limits=(-xs, xs), y_limits=(-ys, ys))
        return el

    def _conv_collimator(self, row, brho, rep, keep):
        return Collimator(name=row.name, length=row.length)

    # matrix -----------------------------------------------------------------
    def _conv_matrix(self, row, brho, rep, keep):
        el = Taylor(name=row.name, length=row.length)
        el.matrix = [[row.get(f"rm{i}{j}", 1.0 if i == j else 0.0) for j in range(1, 7)]
                     for i in range(1, 7)]
        el.offset = [row.get(f"kick{i}") for i in range(1, 7)]
        return el

    # -- aperture, errors ----------------------------------------------------
    def _apply_aperture(self, el: Element, row: _Row, rep: FidelityReport) -> None:
        ap = row.arr("aperture")
        if not any(ap):
            return
        kind = row.txt("apertype").strip().lower() or "circle"
        if kind == "circle":
            el.aperture = ApertureP.circle(ap[0])
        elif kind == "ellipse":
            el.aperture = ApertureP(shape="ELLIPTICAL", x_limits=(-ap[0], ap[0]),
                                    y_limits=(-ap[1], ap[1]) if len(ap) > 1 else (-ap[0], ap[0]))
        elif kind == "rectangle":
            el.aperture = ApertureP.rect(ap[0], ap[1] if len(ap) > 1 else ap[0])
        else:
            el.native.setdefault("madx", {}).update({"apertype": kind, "aperture": ap})
            el.aperture = ApertureP.rect(ap[0], ap[1] if len(ap) > 1 else ap[0])
            rep.lossy("APERTYPE_UNSUPPORTED",
                      f"apertype {kind!r} approximated by its bounding rectangle",
                      element=row.name, kind=el.kind, apertype=kind, aperture=ap)

    def _apply_errors(self, el: Element, row: _Row, rep: FidelityReport) -> None:
        al = row.align
        if al is not None and any((al.dx, al.dy, al.ds, al.dphi, al.dtheta, al.dpsi)):
            el.shift = BodyShiftP(x_offset=al.dx, y_offset=al.dy, z_offset=al.ds,
                                  x_rot=al.dphi, y_rot=al.dtheta, tilt=al.dpsi)
        fe = row.field_err
        if fe is not None and (any(fe.dkn) or any(fe.dks)):
            el.native.setdefault("madx", {})["field_errors"] = {
                "dkn": list(fe.dkn), "dks": list(fe.dks)}
            rep.lossy("EFCOMP_DROPPED",
                      "EFCOMP field errors are recorded in native['madx'] but not modelled",
                      element=row.name, kind=el.kind)

    # -- dipedge folding -----------------------------------------------------
    @staticmethod
    def _edge_is_free(b: Element | None, edge: str, h: float, hgap: float) -> bool:
        """True when *b* is a bend of curvature *h* whose *edge* is still unset."""
        if not (isinstance(b, Bend) and b.length > 0
                and abs(b.bend.g_ref(b.length) - h) <= 1e-9 * max(1.0, abs(h))):
            return False
        if b.bend.hgap and hgap and abs(b.bend.hgap - hgap) > 1e-12:
            return False               # two different gaps would double-count the fringe
        if edge == "e1":
            return not b.bend.e1 and not b.bend.edge_int1
        return not b.bend.e2 and not b.bend.edge_int2

    @staticmethod
    def _fold_dipedges(elements: list[Element], rep: FidelityReport) -> list[Element]:
        """Fold a ``dipedge`` into the bend it touches; keep it as a Directive otherwise."""
        out: list[Element] = []
        for i, el in enumerate(elements):
            if not (isinstance(el, Directive) and el.card == "dipedge"):
                out.append(el)
                continue
            nat = el.native.get("madx", {})
            h, e1, fint, hgap = (nat.get("h", 0.0), nat.get("e1", 0.0),
                                 nat.get("fint", 0.0), nat.get("hgap", 0.0))
            nxt = elements[i + 1] if i + 1 < len(elements) else None
            prv = out[-1] if out else None

            free = Reader._edge_is_free
            if free(nxt, "e1", h, hgap):
                nxt.bend.e1 = e1
                nxt.bend.edge_int1 = fint
                nxt.bend.hgap = hgap or nxt.bend.hgap
                target, where = nxt, "entrance"
            elif free(prv, "e2", h, hgap):
                prv.bend.e2 = e1
                prv.bend.edge_int2 = fint
                prv.bend.hgap = hgap or prv.bend.hgap
                target, where = prv, "exit"
            else:
                rep.lossy("DIPEDGE_UNFOLDED",
                          "DIPEDGE has no adjacent bend of matching curvature; kept as a Directive",
                          element=el.name, kind="Directive", h=h)
                out.append(el)
                continue
            rep.equivalent("DIPEDGE_FOLDED", f"DIPEDGE folded into the {where} of {target.name!r}",
                           element=el.name, kind="Bend", bend=target.name)
        return out
