"""IR → DYNAC deck (see the package docstring for the measured conventions).

The deck is a title, a ``GEBEAM``/``INPUT`` beam block for the reference particle (a small placeholder
ellipsoid — DYNAC needs a bunch, lattix has none), ``EMIPRT 2`` and one card group per element, each
preceded by a ``; lattix: name=… kind=…`` comment the reader restores names and kinds from.  Thick RF
elements become ``FIELD`` blocks in one side file (``<deck>.fields.txt``) driven by ``CAVNUM``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage
from lattix.formats.dynac.cards import fnum
from lattix.ir.elements import Element
from lattix.ir.fieldmap import load_field_map, replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.walk import propagate

C_LIGHT = 299_792_458.0
_THIN = 1e-12
_AMU_EV = 931.49410242e6
_PLAIN = re.compile(r"[\w.+\-]+")
#: DYNAC's default REJECT window (user guide 6.5.3 remark 2): restored after a collimator
REJECT_DEFAULT = "1 1000. 4000. 100. 100. 400."
#: stripper materials: (atomic number, atomic mass [u])
_MATERIALS = {"C": (6, 12.011), "CARBON": (6, 12.011), "DIAMOND": (6, 12.011), "AL": (13, 26.982),
              "BE": (4, 9.012), "TI": (22, 47.867), "CU": (29, 63.546), "AU": (79, 196.97), "NI": (28, 58.693)}


@dataclass(frozen=True)
class Rule:
    message: str
    cls: str
    code: str


@dataclass
class _Group:
    """One element's cards: ``; lattix:`` tag lines then ``[(type code, [parameter lines])]``."""
    tag: dict
    cards: list[tuple[str, list[str]]] = field(default_factory=list)

    def render(self) -> list[str]:
        out = ["; lattix: " + _tag(self.tag)]
        for code, lines in self.cards:
            out.append(code)
            out += lines
        return out


def _tag(kv: dict) -> str:
    parts = []
    for k, v in kv.items():
        if v is None:
            continue
        s = ("1" if v else "0") if isinstance(v, bool) else (fnum(v, 15) if isinstance(v, int | float) else str(v))
        if not _PLAIN.fullmatch(s):
            s = '"' + s.replace('"', "'").replace("\n", " ") + '"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport` (a DYNAC deck, plus ``<stem>.fields.txt``
    when it has thick RF elements)."""

    format = "dynac"
    RULES: dict[str, Rule] = {
        "Drift": Rule("DRIFT L[cm]", "EXACT", "OK"),
        "Quadrupole": Rule("QUADRUPO L[cm] B_tip[kG] = Bn1·R/10 R[cm]", "EXACT", "OK"),
        "Sextupole": Rule("SEXTUPO 1 B_tip[kG] = Bn2·R²/2·10 L[cm] R[cm] (a drift at first order)", "EXACT", "OK"),
        "Octupole": Rule("DRIFT (DYNAC has no octupole)", "LOSSY", "OCTUPOLE_TO_DRIFT"),
        "Multipole": Rule("STEER for the dipole terms (∫B·dl in T·m)", "EQUIVALENT", "MULTIPOLE_AS_STEER"),
        "Bend": Rule("BMAGNET (angle, ρ, field for negative species, n = −k1ρ², pole faces, EK1 = fint, "
                     "APB = hgap) inside ZROT ±tilt", "EXACT", "OK"),
        "Solenoid": Rule("SOLENO 1 L[cm] B[kG]", "EXACT", "OK"),
        "RFCavity": Rule("BUNCHER V[MV] φ[deg] h R[cm] (a thick one at its centre between half drifts)", "EXACT",
                         "OK"),
        "FieldMap": Rule("FIELD (the 1-D E_z map) + CAVNUM at the synchronous phase, or the hard-edge ladder",
                         "EQUIVALENT", "FM_AS_CAVNUM"),
        "NCells": Rule("FIELD (a cell-train profile of the train's voltage) + CAVNUM", "EQUIVALENT",
                       "NCELLS_AS_CAVNUM"),
        "RFQCell": Rule("DRIFT (RFQPTQ needs PARMTEQ cell data)", "LOSSY", "RFQCELL_TO_DRIFT"),
        "Kicker": Rule("STEER FLD[T·m] = kick·Bρ_signed, NVF 0/1", "EXACT", "OK"),
        "Collimator": Rule("REJECT window over the element, DYNAC's defaults restored after it", "EQUIVALENT",
                           "COLLIMATOR_AS_REJECT"),
        "Marker": Rule("EMIT (beam characteristics printed there)", "EXACT", "OK"),
        "Instrument": Rule("EMIT (the family in the tag)", "EQUIVALENT", "INSTRUMENT_AS_EMIT"),
        "Foil": Rule("STRIPPER (material Z, A, g/cm²; DYNAC's own loss and scattering model)", "EQUIVALENT",
                     "FOIL_AS_STRIPPER"),
        "Taylor": Rule("DRIFT of the length (DYNAC has no matrix element)", "LOSSY", "TAYLOR_DROPPED"),
        "Patch": Rule("ALINER for the offsets, ZROT for the tilt; pitch, yaw and z dropped", "EQUIVALENT",
                      "PATCH_AS_ALINER"),
        "ReferenceChange": Rule("NREF 0 dW[MeV] 0 1", "EQUIVALENT", "REFCHANGE_AS_NREF"),
        "Freq": Rule("NEWF f[Hz]", "EXACT", "OK"),
        "Directive": Rule("a ';' comment (DYNAC has no such card)", "DROPPED", "FOREIGN_DIRECTIVE"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED"),
    }

    def write(self, lattice: Lattice, path, *, strict: bool = False, **options) -> FidelityReport:
        path = Path(path)
        rep = FidelityReport(target_format="dynac", target_file=str(path))
        stem = path.name.split(".")[0] or path.stem
        text, files = self.render(lattice, report=rep, field_file=f"{stem}.fields.txt", **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="latin-1", errors="replace")
        for name, content in files.items():
            (path.parent / name).write_text(content)
        rep.raise_if(strict)
        return rep

    def render(self, lattice: Lattice, *, report: FidelityReport | None = None, n_particles: int = 1000,
               beam_extent=(0.1, 1.0, 0.1, 1.0, 1e-3, 1e-11), quad_radius_m: float = 0.05,
               field_file: str = "lattice.fields.txt", install_apertures: bool = True,
               n_intervals: int = 10) -> tuple[str, dict[str, str]]:
        self.rep = report if report is not None else FidelityReport(target_format="dynac")
        self.lat = lattice
        self.quad_radius_m = quad_radius_m
        self.install_apertures = install_apertures
        self.n_intervals = n_intervals
        self.field_file = field_file
        self.field_blocks: list[str] = []
        ref = lattice.reference
        self.master_f = float(ref.rf_frequency_Hz or 0.0)
        placed = propagate(lattice)
        if not self.master_f:
            first = next((float(p.element.rf.frequency_Hz) for p in placed
                          if getattr(p.element, "rf", None) is not None and p.element.rf.frequency_Hz), 0.0)
            self.master_f = first or 1e6
            self.rep.equivalent("DYNAC_FREQUENCY_ASSUMED", "the reference carries no RF frequency: the GEBEAM "
                                f"frequency is {self.master_f:.6g} Hz (the first cavity's, else 1 MHz); it only "
                                "scales DYNAC's phase coordinate")
        groups: list[_Group] = []
        for p in placed:
            groups += self._element(p)
        atm = max(1, int(round(ref.species.mass_eV / _AMU_EV)))
        mass_MeV = ref.species.mass_eV * 1e-6 / atm            # DYNAC's rest mass is UEM × ATM
        x, xp, y, yp, dw, dt = beam_extent
        lines = [f"{lattice.name} (lattix {__version__})"[:80],
                 format_reference_tag(ref, ";"),
                 "; lattix: lattice " + _tag({"name": lattice.name, "use": lattice.use or lattice.name}),
                 "; the beam block is a placeholder ellipsoid for DYNAC's bunch (lattix carries no emittance)",
                 "GEBEAM", "2 0", f"{fnum(self.master_f)} {int(n_particles)}", "0. 0. 0. 0. 0. 0.",
                 f"{fnum(x)} {fnum(xp)} {fnum(y)} {fnum(yp)} {fnum(dw)} {fnum(dt)}",
                 "INPUT", f"{fnum(mass_MeV)} {atm} {fnum(float(ref.species.charge))}",
                 f"{fnum(ref.kinetic_energy_eV * 1e-6)} 0.", "EMIPRT", "2"]
        for g in groups:
            lines += g.render()
        lines.append("STOP")
        files = {field_file: "".join(self.field_blocks)} if self.field_blocks else {}
        return "\n".join(lines) + "\n", files

    # ------------------------------------------------------------------ helpers
    def _record(self, el: Element, rule: Rule, **details) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind, code=rule.code, message=rule.message)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind, **details)

    def _group(self, el: Element, cards: list[tuple[str, list[str]]], *, kind: str | None = None,
               tag: dict | None = None) -> _Group:
        full = {"name": el.name, "kind": kind or el.kind}
        if el.meta.get("dynac_from"):
            full["from"] = str(el.meta["dynac_from"])        # a re-read replacement keeps its origin
        if tag:
            full.update(tag)
        return _Group(full, list(cards))

    @staticmethod
    def _brho(p: Placed) -> float:
        return p.ref_in.brho_signed if p.ref_in is not None else 0.0

    def _radius_cm(self, el: Element) -> float:
        ap = el.aperture
        if ap is not None and ap.x_limits is not None:
            return 100.0 * max(abs(ap.x_limits[0]), abs(ap.x_limits[1]))
        return 100.0 * self.quad_radius_m

    def _with_shift(self, el: Element, cards: list[tuple[str, list[str]]], *, tilt_ok: bool = False):
        """Misalignment as a permanent beam shift before and its inverse after the element."""
        sh = el.shift
        if sh is None:
            return cards
        out = list(cards)
        if sh.x_offset or sh.y_offset:
            dx, dy = 100.0 * sh.x_offset, 100.0 * sh.y_offset
            out = ([("ALINER", [f"{fnum(-dx)} {fnum(-dy)} 0. 0."])] + out
                   + [("ALINER", [f"{fnum(dx)} {fnum(dy)} 0. 0."])])
            self.rep.equivalent("MISALIGN_AS_ALINER", "the transverse offsets are a pair of opposite ALINER beam "
                                "shifts around the element", element=el.name, kind=el.kind)
        lost = [k for k in ("z_offset", "x_rot", "y_rot") if getattr(sh, k)]
        if sh.tilt and not tilt_ok:
            lost.append("tilt")
        if lost:
            self.rep.lossy("MISALIGN_DROPPED", "DYNAC's ALINER carries transverse offsets (and TWQA a quadrupole "
                           "roll); dropped: " + ", ".join(lost), element=el.name, kind=el.kind)
        return out

    def _apertures(self, el: Element) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
        ap = el.aperture
        # QUADRUPO, SEXTUPO and BUNCHER carry their own radius: no window around them
        own_radius = el.kind in ("Quadrupole", "Sextupole", "RFCavity")
        if ap is None or el.kind == "Collimator" or own_radius or not self.install_apertures:
            return [], []
        line = self._reject_line(ap, el)
        if line is None:
            return [], []
        self.rep.equivalent("APERTURE_AS_REJECT", "the element's aperture is a REJECT window over it (DYNAC's "
                            "defaults restored after)", element=el.name, kind=el.kind)
        return [("REJECT", [line])], [("REJECT", [REJECT_DEFAULT])]

    def _reject_line(self, ap, el: Element) -> str | None:
        if ap.x_limits is None and ap.y_limits is None:
            return None
        xl = ap.x_limits or (-1.0, 1.0)
        yl = ap.y_limits or (-1.0, 1.0)
        hx, hy = 100.0 * (xl[1] - xl[0]) / 2, 100.0 * (yl[1] - yl[0]) / 2
        if abs(xl[0] + xl[1]) > _THIN or abs(yl[0] + yl[1]) > _THIN:
            self.rep.lossy("APERTURE_OFFSET_DROPPED", "DYNAC's REJECT window is centred; the limits were "
                           "symmetrized", element=el.name, kind=el.kind)
        if ap.shape == "ELLIPTICAL":
            if abs(hx - hy) > 1e-12:
                self.rep.lossy("APERTURE_SHAPE", "an elliptical aperture is a circle of the larger half axis "
                               "in DYNAC", element=el.name, kind=el.kind)
            return f"1 1000. 4000. 100. 100. {fnum(max(hx, hy))}"
        return f"1 1000. 4000. {fnum(hx)} {fnum(hy)} 400."

    # ------------------------------------------------------------------ elements
    def _element(self, p: Placed) -> list[_Group]:
        el = p.element
        rule = self.RULES.get(el.kind)
        if rule is None:                                          # pragma: no cover - RULES is total
            raise KeyError(f"the DYNAC writer has no rule for kind {el.kind!r}")
        if el.kind == "Superposition":
            return self._w_superposition(el, p, rule)
        if el.kind == "FieldMap":
            return self._w_fieldmap(el, p, rule)
        pre, post = self._apertures(el)
        fn = getattr(self, f"_w_{el.kind.lower()}")
        groups = fn(el, p, rule)
        if groups and (pre or post):
            groups[0].cards = pre + groups[0].cards
            groups[-1].cards = groups[-1].cards + post
        return groups

    def _w_drift(self, el, p, rule):
        self._record(el, rule)
        return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])])]

    def _w_quadrupole(self, el, p, rule):
        mp = el.multipole
        bn = float(mp.Bn.get(1, 0.0))
        bs = float(mp.Bs.get(1, 0.0))
        tilt = float(mp.tilt.get(1, 0.0)) + (float(el.shift.tilt) if el.shift is not None else 0.0)
        if bs:
            tilt += math.atan2(bs, bn) / 2.0
            bn = math.hypot(bn, bs)
        r_cm = self._radius_cm(el)
        b_tip = bn * (r_cm / 100.0) * 10.0                       # T/m · m → T → kG
        cards = [("QUADRUPO", [f"{fnum(100.0 * el.length)} {fnum(b_tip)} {fnum(r_cm)}"])]
        if tilt:
            cards = [("TWQA", [f"0 {fnum(math.degrees(tilt))}"])] + cards + [("TWQA", ["0 0."])]
            self.rep.equivalent("QUAD_TILT_AS_TWQA", "the quadrupole roll is a TWQA card before it (reset after)",
                                element=el.name, kind=el.kind)
        others = sorted(n for n, v in {**mp.Bn, **mp.Bs}.items() if v and n != 1)
        if others:
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "DYNAC's QUADRUPO holds the gradient only; dropped orders "
                           f"{others}", element=el.name, kind=el.kind)
        self._record(el, rule)
        return [self._group(el, self._with_shift(el, cards, tilt_ok=True))]

    def _w_sextupole(self, el, p, rule):
        mp = el.multipole
        bn2 = float(mp.Bn.get(2, 0.0))
        r_cm = self._radius_cm(el)
        b_tip = 0.5 * bn2 * (r_cm / 100.0) ** 2 * 10.0             # T/m² · m² / 2 → T → kG
        cards = [("SEXTUPO", [f"1 {fnum(b_tip)} {fnum(100.0 * el.length)} {fnum(r_cm)}"])]
        others = sorted(n for n, v in {**mp.Bn, **mp.Bs}.items() if v and n != 2)
        if others:
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "DYNAC's SEXTUPO holds the sextupole field only; dropped "
                           f"orders {others}", element=el.name, kind=el.kind)
        self._record(el, rule)
        return [self._group(el, self._with_shift(el, cards))]

    def _w_octupole(self, el, p, rule):
        self._record(el, rule)
        return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])])]

    def _w_multipole(self, el, p, rule):
        mp = el.multipole
        cards: list[tuple[str, list[str]]] = []
        bnl0, bsl0 = float(mp.BnL.get(0, 0.0)), float(mp.BsL.get(0, 0.0))
        if bnl0:
            cards.append(("STEER", [f"{fnum(-bnl0)} 0"]))         # ∫By·dl deflects a positive charge to −x
        if bsl0:
            cards.append(("STEER", [f"{fnum(bsl0)} 1"]))
        higher = sorted({n for n, v in {**mp.BnL, **mp.BsL}.items() if n > 0 and v})
        if higher:
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "DYNAC has no thin multipole: only the dipole terms are "
                           f"written (as STEER); dropped orders {higher}", element=el.name, kind=el.kind)
        if el.length > _THIN:
            cards.append(("DRIFT", [fnum(100.0 * el.length)]))
        if not cards:
            cards.append(("DRIFT", ["0."]))
        self._record(el, rule)
        return [self._group(el, cards)]

    def _w_bend(self, el, p, rule):
        b = el.bend
        ref = p.ref_in or self.lat.reference
        angle = float(b.angle)
        if abs(angle) < 1e-15:
            self.rep.equivalent("ZERO_ANGLE_BEND_AS_DRIFT", "a bend without angle is a drift", element=el.name,
                                kind=el.kind)
            return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])])]
        rho = el.length / abs(angle)                                # m, > 0
        brho = self._brho(p)
        k1 = float(el.multipole.Bn.get(1, 0.0)) / brho if brho else 0.0
        xn = -k1 * rho * rho
        q = ref.species.charge
        baim = 0.0 if q > 0 else -(ref.brho_abs / rho) * 10.0     # kG: a negative species needs the field
        fint1 = float(b.edge_int1 or 0.0)
        fint2 = float(b.edge_int2) if b.edge_int2 is not None else fint1
        k2 = float(b.fringe_k2) if b.fringe_k2 is not None else 0.0
        hgap = 100.0 * float(b.hgap or 0.0)
        # MEASURED (docs/oracles.md, Phase 5.8): DYNAC's BMAGNET bends to the right (towards −x) for a
        # positive ANGL; a negative ANGL (or a negative RMO) mirrors the map or displaces the reference — a
        # bend to the left is the positive magnet turned by 180° with ZROT, as DYNAC's own manual shows
        # MEASURED (cpymad 5.09.03): MAD-X's (angle < 0, e1, e2) is (angle > 0, tilt = π, −e1, −e2) to 1e-17 — the
        # IR's pole-face angles follow MAD-X, so the turned magnet gets them negated
        e1, e2 = (-b.e1, -b.e2) if angle < 0 else (b.e1, b.e2)
        cards = [("BMAGNET", ["1", f"{fnum(math.degrees(abs(angle)))} {fnum(100.0 * rho)} {fnum(baim)} "
                                   f"{fnum(xn)} 0.",
                              f"{fnum(math.degrees(e1))} 0. {fnum(fint1)} {fnum(k2)} {fnum(hgap)}",
                              f"{fnum(math.degrees(e2))} 0. {fnum(fint2)} {fnum(k2)} {fnum(hgap)}"])]
        tilt = float(b.tilt_ref) + (float(el.shift.tilt) if el.shift is not None else 0.0)
        zrot = math.degrees(tilt) + (180.0 if angle < 0 else 0.0)
        zrot = (zrot + 180.0) % 360.0 - 180.0
        if abs(zrot) > 1e-12:
            cards = [("ZROT", [fnum(zrot)])] + cards + [("ZROT", [fnum(-zrot)])]
        if any(v for n, v in {**el.multipole.Bn, **el.multipole.Bs}.items() if n != 1):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "DYNAC's BMAGNET carries the field index n only",
                           element=el.name, kind=el.kind)
        if hgap and not fint1 and not k2:
            self.rep.equivalent("BEND_FRINGE_DROPPED", "a gap without fringe integral: DYNAC's EK1 = 0",
                                element=el.name, kind=el.kind)
        self._record(el, rule)
        tag = {}
        if b.rect:
            tag["rect"] = True
        if angle < 0:
            tag["neg"] = True
        return [self._group(el, self._with_shift(el, cards, tilt_ok=True), tag=tag or None)]

    def _w_solenoid(self, el, p, rule):
        self._record(el, rule)
        cards = [("SOLENO", [f"1 {fnum(100.0 * el.length)} {fnum(10.0 * float(el.solenoid.Bsol_T))}"])]
        return [self._group(el, self._with_shift(el, cards))]

    # -- RF ------------------------------------------------------------------------------------
    def _buncher(self, el, p, rule, *, V: float, phase: float, freq: float, tag: dict | None = None):
        ref = p.ref_in or self.lat.reference
        deg = math.degrees(phase)
        if ref.species.charge < 0:
            deg += 180.0                                          # the gain is q·V·cos φ
        deg = (deg + 180.0) % 360.0 - 180.0
        harm = freq / self.master_f if self.master_f else 1.0
        cards = [("BUNCHER", [f"{fnum(V * 1e-6)} {fnum(deg)} {fnum(harm)} {fnum(self._radius_cm(el))}"])]
        if abs(harm - round(harm)) > 1e-9:
            self.rep.equivalent("BUNCHER_HARMONIC", f"the cavity frequency is {harm:.6g} × the master frequency "
                                "(BUNCHER PHARM)", element=el.name, kind=el.kind)
        self._record(el, rule, voltage_V=V)
        return [self._group(el, self._with_shift(el, cards), kind="RFCavity", tag=tag)]

    def _cavnum(self, el, p, rule, *, V: float, phase: float, freq: float, z: list[float], ez: list[float],
                kind: str, tag: dict | None = None, calibrate: bool = True, **details):
        """A ``FIELD`` block (the profile scaled to the voltage) driven by ``CAVNUM`` at the synchronous
        phase relative to DYNAC's crest."""
        ref = p.ref_in or self.lat.reference
        if calibrate:
            from lattix.formats.impactt.rfprofile import calibrate as _calibrate
            from lattix.formats.impactt.rfprofile import fourier_coefficients

            period = float(z[-1] - z[0])
            mid = 0.5 * (z[0] + z[-1])
            zz = [v - mid for v in z]                  # the profile integrator wants z about the centre
            coefs = fourier_coefficients(zz, ez, period, 60)
            scale, _theta = _calibrate(coefs, period, V, phase, freq, 0.0, ref.kinetic_energy_eV,
                                       ref.species.mass_eV, ref.species.charge)
            amp = [scale * e for e in ez]
        else:
            amp = list(ez)
        # DYNAC counts cells from the zero crossings of the field and dies on long runs of zeros: the
        # block covers the nonzero span only, drifts make up the element's length around it
        nz = [i for i, e in enumerate(amp) if e != 0.0]
        lo, hi = (max(nz[0] - 1, 0), min(nz[-1] + 1, len(amp) - 1)) if nz else (0, len(amp) - 1)
        pad_before, pad_after = z[lo] - z[0], z[-1] - z[hi]
        block = [fnum(freq)]
        for zi, ei in zip(z[lo:hi + 1], amp[lo:hi + 1], strict=True):
            block.append(f"{fnum(zi - z[lo])} {fnum(ei)}")
        block.append("0. 0.")
        self.field_blocks.append("\n".join(block) + "\n")
        ielec = 0 if ref.species.mass_eV < 1e6 else 1
        cards = []
        if pad_before > 1e-12:
            cards.append(("DRIFT", [fnum(100.0 * pad_before)]))
        cards += [("FIELD", [self.field_file, "1."]),
                  ("CAVNUM", [str(len(self.field_blocks)),
                              f"0. {fnum(math.degrees(phase))} 0 {self.n_intervals} {ielec}"])]
        if pad_after > 1e-12:
            cards.append(("DRIFT", [fnum(100.0 * pad_after)]))
        self._record(el, rule, voltage_V=V, **details)
        full_tag = {"V": V, "L": float(z[-1] - z[0])}
        if tag:
            full_tag.update(tag)
        return [self._group(el, self._with_shift(el, cards), kind=kind, tag=full_tag)]

    def _native_block(self, el, p, rule, kind: str):
        """A FIELD + CAVNUM group read from a DYNAC deck goes back as it was (its own profile and phase)."""
        nat = el.native["dynac"]
        self.rep.exact(el.name, el.kind, code="DYNAC_NATIVE_PASSTHROUGH",
                       message=f"{nat['card']} with its FIELD block re-emitted from native['dynac']")
        block = [fnum(float(nat["f"]))]
        for zi, ei in zip(nat["z"], nat["ez"], strict=True):
            block.append(f"{fnum(float(zi))} {fnum(float(ei))}")
        block.append("0. 0.")
        self.field_blocks.append("\n".join(block) + "\n")
        cards: list[tuple[str, list[str]]] = []
        if float(nat.get("pad_before", 0.0)) > 1e-12:
            cards.append(("DRIFT", [fnum(100.0 * float(nat["pad_before"]))]))
        cards += [("FIELD", [self.field_file, "1."]),
                  (str(nat["card"]), [str(len(self.field_blocks)), *[str(x) for x in nat["cavnum"][1:]]])]
        if float(nat.get("pad_after", 0.0)) > 1e-12:
            cards.append(("DRIFT", [fnum(100.0 * float(nat["pad_after"]))]))
        return [self._group(el, self._with_shift(el, cards), kind=kind, tag=dict(nat.get("tag") or {}))]

    def _w_rfcavity(self, el, p, rule):
        rf = el.rf
        ref = p.ref_in or self.lat.reference
        if el.native.get("dynac", {}).get("z"):
            return self._native_block(el, p, rule, "RFCavity")
        V = rf.voltage_V
        if not V and rf.gradient_V_per_m is not None:
            V = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        if not V and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-9:
            V = rf.dE_ref_eV / math.cos(rf.phase_rad)
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        if not freq or not V:
            self.rep.lossy("DYNAC_CAVITY_TO_DRIFT", "an RF cavity without a frequency or a voltage cannot be "
                           "written; replaced by a drift", element=el.name, kind="RFCavity")
            return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])], kind="RFCavity")]
        tag = {}
        if rf.cavity_type == "TRAVELING_WAVE":
            tag["tw"] = True
        if rf.n_cell:
            tag["n"] = int(rf.n_cell)
        if not rf.phase_is_sync:
            self.rep.equivalent("RF_RAW_PHASE_AS_SYNC", "DYNAC's phases are relative to the crest: the deck's "
                                "raw RF phase is written as the synchronous phase", element=el.name, kind=el.kind)
        if el.length <= _THIN:
            return self._buncher(el, p, rule, V=float(V), phase=rf.phase_rad, freq=float(freq), tag=tag or None)
        # MEASURED (docs/oracles.md, Phase 5.8): CAVNUM's integration of a generated profile and lattix's
        # calibration part company on strongly accelerating low-β cavities (1.6× at 2.1 MeV / 1 MV), so a
        # thick cavity is the exact-gain thin BUNCHER at its centre between two half-length drifts
        self.rep.equivalent("THICK_CAVITY_AS_BUNCHER", "a thick cavity is a thin BUNCHER of its voltage at the "
                            "centre between two half-length drifts (DYNAC's buncher kick, no field profile)",
                            element=el.name, kind=el.kind, length_m=el.length)
        tag = dict(tag or {})
        tag["L"] = el.length
        groups = self._buncher(el, p, rule, V=float(V), phase=rf.phase_rad, freq=float(freq), tag=tag)
        half = fnum(50.0 * el.length)
        groups[0].cards = [("DRIFT", [half])] + groups[0].cards + [("DRIFT", [half])]
        return groups

    def _w_ncells(self, el, p, rule):
        rf = el.rf
        ref = p.ref_in or self.lat.reference
        if el.native.get("dynac", {}).get("z"):
            return self._native_block(el, p, rule, "NCells")
        V = rf.voltage_V or (rf.gradient_V_per_m or 0.0) * (rf.L_active_m or el.length)
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        n_cell = int(el.params.get("n_cells") or rf.n_cell or 1)
        if not (V and freq and el.length > _THIN):
            self.rep.lossy("NCELLS_TO_DRIFT", "an NCells without a voltage, a frequency or a length is a drift",
                           element=el.name, kind="NCells")
            return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])], kind="NCells")]
        from lattix.formats.impactt.rfprofile import cell_train

        mode = el.params.get("mode")
        z, ez = cell_train(el.length, n_cell, pi_mode=(mode != 0), n=max(401, 64 * n_cell + 1))
        unmapped = {k: v for k, v in el.params.items()
                    if k in ("beta_g", "k_eot_i", "k_eot_o", "dz_i", "dz_o") and v}
        if unmapped:
            self.rep.lossy("DYNAC_NCELLS_PARAMS", "the cell-train profile has no geometric β_g, entrance/exit "
                           "half-cell corrections or TTF tail; those NCELLS columns are dropped",
                           element=el.name, kind="NCells", **unmapped)
        return self._cavnum(el, p, rule, V=float(V), phase=rf.phase_rad, freq=float(freq), z=z, ez=ez,
                            kind="NCells", tag={"n": n_cell, "mode": int(mode or 0)}, n_cells=n_cell)

    def _w_fieldmap(self, el, p, rule):
        s = (el.meta or {}).get("map_summary") or {}
        data = None
        try:
            data = load_field_map(el)
        except (OSError, ValueError, NotImplementedError):
            data = None
        ez_channel = None
        if data is not None:
            for ch in data.channels.values():
                if ch.is_electric and not ch.is_static and ch.z is not None and ch.Fz is not None \
                        and getattr(ch.Fz, "ndim", 1) == 1:
                    ez_channel = ch
                    break
        if ez_channel is not None and s.get("kind") == "rf" and el.rf.frequency_Hz:
            z = [float(v) for v in ez_channel.z]
            ez = [float(v) * float(el.ke) for v in ez_channel.Fz]
            phase = float(s.get("phase_sync_rad") or 0.0)
            V = float(s.get("v_c_V") or 0.0)
            self._record(el, rule, voltage_V=V, file=el.files[0] if el.files else None)
            return self._cavnum(el, p, self.RULES["RFCavity"], V=V, phase=phase, freq=float(el.rf.frequency_Hz),
                                z=z, ez=ez, kind="FieldMap", calibrate=False,
                                tag={"file": el.files[0] if el.files else None})
        r = replacement_for(el)
        out: list[_Group] = []
        st = p.s_in
        for part in r.parts:
            sub = p.model_copy(update={"element": part, "s_in": st, "s_out": st + part.length})
            fn = getattr(self, f"_w_{part.kind.lower()}")
            for g in fn(part, sub, self.RULES[part.kind]):
                g.tag["from"] = "FieldMap"
                out.append(g)
            st += part.length
        getattr(self.rep, r.cls.lower())(r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, msg in r.extra:
            getattr(self.rep, cls.lower())(code, msg, element=el.name, kind="FieldMap")
        return out

    def _w_rfqcell(self, el, p, rule):
        self._record(el, rule)
        return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])], kind="RFQCell")]

    def _w_kicker(self, el, p, rule):
        brho = self._brho(p)
        if el.electric:
            self.rep.lossy("EKICK_AS_MAGNETIC", "an electric kicker is written as a magnetic STEER of the same "
                           "deflection", element=el.name, kind=el.kind)
        cards: list[tuple[str, list[str]]] = []
        if el.hkick or not el.vkick:
            cards.append(("STEER", [f"{fnum(float(el.hkick) * brho)} 0"]))
        if el.vkick:
            cards.append(("STEER", [f"{fnum(float(el.vkick) * brho)} 1"]))
        if el.length > _THIN:
            cards.append(("DRIFT", [fnum(100.0 * el.length)]))
            self.rep.equivalent("KICKER_THIN_AT_ENTRANCE", "DYNAC's STEER is thin: the kick at the entrance, a "
                                "drift for the length", element=el.name, kind=el.kind)
        self._record(el, rule)
        return [self._group(el, self._with_shift(el, cards))]

    def _w_collimator(self, el, p, rule):
        line = self._reject_line(el.aperture, el) if el.aperture is not None else None
        if line is None:
            self.rep.lossy("COLLIMATOR_TO_MARKER", "a collimator without limits is a drift", element=el.name,
                           kind=el.kind)
            return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])])]
        self._record(el, rule)
        cards = [("REJECT", [line]), ("DRIFT", [fnum(100.0 * el.length)]), ("REJECT", [REJECT_DEFAULT])]
        return [self._group(el, cards)]

    def _w_marker(self, el, p, rule):
        self._record(el, rule)
        return [self._group(el, [("EMIT", [])])]

    def _w_instrument(self, el, p, rule):
        self._record(el, rule, family=el.family)
        cards = [("EMIT", [])]
        if el.length > _THIN:
            cards.append(("DRIFT", [fnum(100.0 * el.length)]))
        return [self._group(el, cards, tag={"family": el.family})]

    def _w_foil(self, el, p, rule):
        ref = p.ref_in or self.lat.reference
        z_a = _MATERIALS.get((el.material or "C").upper())
        if z_a is None:
            self.rep.lossy("FOIL_MATERIAL_DROPPED", f"stripper material {el.material!r} unknown to lattix's "
                           "table; carbon written", element=el.name, kind=el.kind)
            z_a = _MATERIALS["C"]
        anp = max(1, int(round(ref.species.mass_eV / _AMU_EV)))
        cards = [("STRIPPER", [f"{z_a[0]} {fnum(z_a[1])} {fnum(0.1 * float(el.thickness_kg_per_m2))} {anp}"])]
        if el.length > _THIN:
            cards.append(("DRIFT", [fnum(100.0 * el.length)]))
        self._record(el, rule, material=el.material)
        return [self._group(el, cards, tag={"material": el.material, "thick": el.thickness_kg_per_m2,
                                           "dE": el.dE_ref_eV})]

    def _w_taylor(self, el, p, rule):
        self._record(el, rule)
        return [self._group(el, [("DRIFT", [fnum(100.0 * el.length)])])]

    def _w_patch(self, el, p, rule):
        cards: list[tuple[str, list[str]]] = []
        if el.x_offset or el.y_offset:
            cards.append(("ALINER", [f"{fnum(-100.0 * el.x_offset)} {fnum(-100.0 * el.y_offset)} 0. 0."]))
        if el.tilt:
            cards.append(("ZROT", [fnum(math.degrees(el.tilt))]))
        lost = [k for k in ("z_offset", "x_rot", "y_rot") if getattr(el, k)]
        if lost:
            self.rep.lossy("PATCH_ROTATION_DROPPED", "DYNAC has no pitch/yaw/longitudinal patch; dropped: "
                           + ", ".join(lost), element=el.name, kind=el.kind)
        if el.length > _THIN:
            cards.append(("DRIFT", [fnum(100.0 * el.length)]))
        if not cards:
            cards.append(("DRIFT", ["0."]))
        self._record(el, rule)
        tag = {k: getattr(el, k) for k in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
               if getattr(el, k)}
        return [self._group(el, cards, tag=tag)]

    def _w_referencechange(self, el, p, rule):
        ref = p.ref_in or self.lat.reference
        dE = el.dE_ref_eV
        if dE is None and el.energy_eV is not None:
            dE = float(el.energy_eV) - ref.kinetic_energy_eV
        self._record(el, rule, dE_ref_eV=dE)
        tag = {}
        if el.dE_ref_eV is not None:
            tag["dE"] = el.dE_ref_eV
        if el.energy_eV is not None:
            tag["E"] = el.energy_eV
        return [self._group(el, [("NREF", [f"0. {fnum(float(dE or 0.0) * 1e-6)} 0 1"])], tag=tag or None)]

    def _w_freq(self, el, p, rule):
        self._record(el, rule)
        self.master_f = float(el.frequency_Hz)
        return [self._group(el, [("NEWF", [fnum(float(el.frequency_Hz))])])]

    def _w_directive(self, el, p, rule):
        if el.format == "dynac" and el.card and el.card != "EMIPRT":
            self.rep.exact(el.name, el.kind, code="DYNAC_DIRECTIVE_KEPT", message=f"{el.card} re-emitted as read")
            return [self._group(el, [(el.card, list(el.args))])]
        self._record(el, rule)
        return [_Group({"name": el.name, "kind": "Directive", "format": el.format, "card": el.card,
                        "args": " ".join(el.args), "role": el.role}, [])]

    def _w_superposition(self, el, p, rule):
        self._record(el, rule)
        out: list[_Group] = []
        st = p.s_in
        for _off, name in el.children:
            child = self.lat.elements.get(name)
            if child is None:
                continue
            sub = p.model_copy(update={"element": child, "s_in": st, "s_out": st + child.length})
            out += self._element(sub)
            st += child.length
        return out


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert check_rules_coverage(Writer()) == set()
