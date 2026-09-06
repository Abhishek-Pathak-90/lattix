"""DYNAC deck → IR (see the package docstring for the measured conventions).

The beam block (``GEBEAM``/``INPUT`` or ``RDBEAM``) gives the reference particle (or the ``; lattix:
reference`` tag, or ``read(species=, kinetic_energy_eV=)``); every optics card becomes an element, the
``; lattix: name=… kind=…`` tags restoring the kinds lattix wrote, and the modifier cards (``ALINER``
pairs, ``TWQA``, ``ZROT``, ``REJECT`` windows, ``NEWF``) are folded onto their elements.  Cards that have
no IR meaning (space charge, prints, plots …) are kept as ``Directive(format="dynac")`` so that a
DYNAC → DYNAC round trip re-emits them.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.formats.dynac.cards import Card, parse_deck
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
from lattix.ir.lattice import Lattice
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.walk import propagate

_TAG = re.compile(r";\s*lattix:\s*(?P<body>(?!reference\b)(?!lattice\b).*)$")
_HEAD = re.compile(r";\s*lattix:\s*lattice\s+(?P<body>.*)$")
_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')
_AMU_EV = 931.49410242e6
#: (Z, A) → material name for STRIPPER
_MATERIAL_OF = {6: "C", 13: "Al", 4: "Be", 22: "Ti", 29: "Cu", 79: "Au", 28: "Ni"}
_DEFAULT_REJECT = (1000.0, 4000.0, 100.0, 100.0, 400.0)


def _parse_tag(body: str) -> dict[str, str]:
    return {k: (q if v.startswith('"') else v) for k, v, q in _KV.findall(body)}


def _f(tokens: list[str], i: int, default: float = 0.0) -> float:
    try:
        return float(tokens[i].replace("D", "e").replace("d", "e"))
    except (IndexError, ValueError):
        return default


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)`` for a DYNAC deck."""

    format = "dynac"

    def read(self, path: Path, *, strict: bool = False, species: str | Species | None = None,
             kinetic_energy_eV: float | None = None,
             frequency_Hz: float | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="dynac", source_file=str(path))
        text = path.read_text(encoding="latin-1", errors="replace")
        title, cards = parse_deck(text)
        self._rep = rep
        self._path = path
        self._norm: dict[int, tuple] = {}
        head: dict[str, str] = {}
        for c in cards:
            for ln in c.comments:
                m = _HEAD.search(ln)
                if m:
                    head = _parse_tag(m.group("body"))
        ref, self.master_f, beam_cards = self._reference(text, cards, species, kinetic_energy_eV, frequency_Hz)
        self._q = ref.species.charge
        self._elements: list[Element] = []
        self._pending_shift: tuple[float, float] | None = None
        self._pending_tilt: float | None = None       # TWQA roll for the quadrupoles that follow
        self._pending_zrot: float | None = None       # a ZROT waiting for the next element
        self._pending_reject: ApertureP | None = None
        self._field_file: str | None = None
        self._field_att = 1.0
        self._field_index: dict[str, int] = {}
        self._harm: tuple | None = None
        self._n = 0
        tag: dict[str, str] = {}
        for c in cards:
            if c.name in beam_cards or c.name == "STOP":
                continue
            for ln in c.comments:                       # a tag holds for the cards of its element's group
                m = _TAG.search(ln)
                if m:
                    tag = _parse_tag(m.group("body"))
            self._card(c, tag)
        self._flush_pending()
        name = head.get("name") or (title.split(" (lattix")[0].strip() if title else path.stem) or path.stem
        lat = Lattice.from_sequence(head.get("use") or name, self._elements, ref)
        lat.name = name
        lat.meta["source_format"] = "dynac"
        lat.meta["source_file"] = str(path)
        lat.meta["dynac_title"] = title
        self._second_pass(lat)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ the reference
    def _reference(self, text, cards, species, kinetic_energy_eV, frequency_Hz):
        tag = parse_reference_tag(text)
        beam_cards = {"GEBEAM", "INPUT", "RDBEAM", "ETAC"}
        mass = charge = ke = f = None
        atm = 1
        for c in cards:
            if c.name == "GEBEAM" and len(c.lines) >= 2:
                f = _f(c.lines[1].split(), 0)
            elif c.name == "INPUT" and len(c.lines) >= 2:
                t0, t1 = c.lines[0].split(), c.lines[1].split()
                atm, charge, ke = int(_f(t0, 1, 1.0)) or 1, _f(t0, 2, 1.0), _f(t1, 0) * 1e6
                mass = _f(t0, 0) * 1e6 * atm                    # DYNAC's rest mass is UEM × ATM
            elif c.name == "RDBEAM" and len(c.lines) >= 5:
                f = _f(c.lines[2].split(), 0) * 1e6
                atm = int(_f(c.lines[3].split(), 1, 1.0)) or 1
                mass = _f(c.lines[3].split(), 0) * 1e6 * atm
                ke, charge = _f(c.lines[4].split(), 0) * 1e6, _f(c.lines[4].split(), 1, 1.0)
        sp = None
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        elif tag is not None:
            sp = tag.species
        elif mass:
            q = int(round(charge or 1.0))
            for cand in SPECIES.values():
                if abs(cand.mass_eV - mass) / mass < 1e-4 and cand.charge == q:
                    sp = cand
                    break
            if sp is None:
                sp = Species(name=f"ion_A{atm}_Q{q}", mass_eV=float(mass), charge=q)
                self._rep.equivalent("SPECIES_ASSUMED", f"INPUT rest mass {mass * 1e-6:.6f} MeV, charge {q}: not one "
                                     "of lattix's named species; kept as an ion of that mass")
        ke_eV = kinetic_energy_eV if kinetic_energy_eV is not None else (
            tag.kinetic_energy_eV if tag is not None else ke)
        # a lattix tag carries the reference's frequency (or none: the GEBEAM frequency was only a placeholder)
        freq = frequency_Hz if frequency_Hz is not None else (tag.rf_frequency_Hz if tag is not None else f)
        if tag is not None and (species is None or kinetic_energy_eV is None):
            self._rep.equivalent("REFERENCE_FROM_TAG", "reference particle taken from the lattix tag")
        if sp is None or ke_eV is None:
            raise ValueError(f"{self._path.name}: no GEBEAM/INPUT or RDBEAM beam block; pass species=<name> and "
                             "kinetic_energy_eV=<eV> to read(...)")
        ref = ReferenceParticle(species=sp, kinetic_energy_eV=float(ke_eV), rf_frequency_Hz=freq or None)
        return ref, float(freq or f or 0.0), beam_cards

    # ------------------------------------------------------------------ cards
    def _add(self, el: Element, tag: dict | None = None) -> Element:
        if tag and tag.get("name"):
            el.name = str(tag["name"])
        if self._pending_reject is not None and el.kind != "Collimator" and el.length >= 0:
            el.aperture = self._pending_reject.model_copy(update={"aperture_at": "ENTRANCE"})
        if self._pending_shift is not None and el.kind not in ("Patch",):
            dx, dy = self._pending_shift
            el.shift = BodyShiftP(x_offset=dx, y_offset=dy)
            self._pending_shift = None
            self._shift_open = el
        if tag and tag.get("from"):
            el.meta["dynac_from"] = tag["from"]
        self._elements.append(el)
        return el

    def _prov(self, c: Card) -> Provenance:
        return Provenance(format="dynac", file=str(self._path), line=c.lineno, original_type=c.name)

    def _name(self, c: Card, tag: dict) -> str:
        self._n += 1
        return str(tag.get("name") or f"{c.name.lower()}_{self._n}")

    def _flush_pending(self) -> None:
        if self._pending_zrot:
            self._elements.append(Patch(name=f"zrot_{self._n + 1}", tilt=math.radians(self._pending_zrot)))
            self._n += 1
            self._pending_zrot = None

    def _card(self, c: Card, tag: dict) -> None:
        name = c.name
        if name == "EMIPRT":
            return                                   # print control (the writer emits its own)
        kind = tag.get("kind")
        fn = getattr(self, f"_c_{name.lower()}", None)
        if fn is None:
            self._directive(c, tag)
            return
        fn(c, tag, kind)

    # -- optics -------------------------------------------------------------------------------
    def _c_drift(self, c: Card, tag: dict, kind: str | None) -> None:
        L = _f(c.lines[0].split(), 0) * 1e-2
        prov = self._prov(c)
        prev = self._elements[-1] if self._elements else None
        same = prev is not None and tag.get("name") and prev.name == tag.get("name")
        if tag.get("L") and kind in ("RFCavity", "NCells", "FieldMap"):
            # padding around a FIELD block or a centred buncher: the element carries the full length
            if same and "dynac" in prev.native:
                prev.native["dynac"]["pad_after"] += L
            else:
                self._cavity_pad = getattr(self, "_cavity_pad", 0.0) + L
            return
        if kind in ("Multipole", "Kicker", "Foil", "Instrument", "Patch") and same and prev.kind == kind \
                and prev.length == 0.0:
            prev.length = L                                # the drift body of a thin card's element
        elif kind == "Instrument":
            self._add(Instrument(name=self._name(c, tag), length=L, family=tag.get("family", "MONITOR"),
                                 provenance=prov), tag)
        elif kind == "Multipole":
            el = self._add(Multipole(name=self._name(c, tag), length=L, provenance=prov), tag)
            el.multipole = MagneticMultipoleP()            # its orders were dropped on write (no dipole term)
        elif kind == "Collimator" and self._pending_reject is not None:
            el = Collimator(name=self._name(c, tag), length=L, aperture=self._pending_reject, provenance=prov)
            self._pending_reject = None
            self._add(el, tag)
        elif kind == "Patch":
            keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
            self._add(Patch(name=self._name(c, tag), length=L, provenance=prov,
                            **{k: float(tag.get(k, 0.0)) for k in keys}), tag)
        elif kind == "Octupole":
            self._add(Octupole(name=self._name(c, tag), length=L, provenance=prov), tag)
        elif kind == "Taylor":
            self._add(Taylor(name=self._name(c, tag), length=L, provenance=prov), tag)     # identity: lost on write
        elif kind == "RFQCell":
            self._add(RFQCell(name=self._name(c, tag), length=L, provenance=prov), tag)
        elif kind == "RFCavity":
            self._add(RFCavity(name=self._name(c, tag), length=L, provenance=prov), tag)   # no voltage / frequency
        elif kind == "NCells":
            self._add(NCells(name=self._name(c, tag), length=L, provenance=prov), tag)
        else:
            self._add(Drift(name=self._name(c, tag), length=L, provenance=prov), tag)

    _c_fdrift = _c_drift

    def _c_quadrupo(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        L, b_kG, r_cm = _f(t, 0) * 1e-2, _f(t, 1), _f(t, 2)
        g = 10.0 * b_kG / r_cm if r_cm else 0.0                    # kG/cm → T/m
        el = Quadrupole(name=self._name(c, tag), length=L, provenance=self._prov(c),
                        aperture=ApertureP.circle(r_cm * 1e-2) if r_cm else None)
        el.multipole = MagneticMultipoleP(Bn={1: g})
        if self._pending_tilt:
            el.multipole.tilt[1] = math.radians(self._pending_tilt)
        self._add(el, tag)

    def _c_sextupo(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        imks, arg, L, r_cm = int(_f(t, 0)), _f(t, 1), _f(t, 2) * 1e-2, _f(t, 3)
        el = Sextupole(name=self._name(c, tag), length=L, provenance=self._prov(c),
                       aperture=ApertureP.circle(r_cm * 1e-2) if r_cm else None)
        el.multipole = MagneticMultipoleP()
        if imks != 0:
            el.multipole.Bn[2] = 2.0 * (arg * 0.1) / (r_cm * 1e-2) ** 2 if r_cm else 0.0   # kG at R → T/m²
        else:
            self._norm[id(el)] = ("ks2", arg)                     # (B/R²)/Bρ in cm⁻³
        self._add(el, tag)

    def _c_soleno(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        imks, L, arg = int(_f(t, 0)), _f(t, 1) * 1e-2, _f(t, 2)
        el = Solenoid(name=self._name(c, tag), length=L, provenance=self._prov(c))
        if imks != 0:
            el.solenoid = SolenoidP(Bsol_T=arg * 0.1)
        else:
            self._norm[id(el)] = ("ksol", arg * 100.0)              # cm⁻¹ → 1/m, k = B/(2Bρ)
        self._add(el, tag)

    def _c_bmagnet(self, c: Card, tag: dict, kind: str | None) -> None:
        t1, t2, t3 = (c.lines[i].split() for i in (1, 2, 3))
        angle = math.radians(_f(t1, 0))
        rho = abs(_f(t1, 1)) * 1e-2
        xn = _f(t1, 3)
        zrot = self._pending_zrot or 0.0
        e1, e2 = math.radians(_f(t2, 0)), math.radians(_f(t3, 0))
        if tag.get("neg") == "1":                      # lattix writes a left bend as ZROT 180 + a right bend
            angle, zrot, e1, e2 = -abs(angle), (zrot - 180.0 + 180.0) % 360.0 - 180.0, -e1, -e2
        el = Bend(name=self._name(c, tag), length=rho * abs(angle), provenance=self._prov(c))
        fint1, fint2 = _f(t2, 2), _f(t3, 2)
        hgap = _f(t2, 4) * 1e-2
        k2 = _f(t2, 3)
        el.bend = BendP(angle=angle, e1=e1, e2=e2, edge_int1=fint1,
                        edge_int2=(fint2 if abs(fint2 - fint1) > 1e-15 else None), hgap=hgap,
                        fringe_k2=(k2 if k2 else None), rect=(tag.get("rect") == "1"),
                        tilt_ref=math.radians(zrot) if abs(zrot) > 1e-12 else 0.0)
        zrot = self._pending_zrot
        self._pending_zrot = None
        if xn:
            self._norm[id(el)] = ("n", xn, rho)
        if _f(t2, 1) or _f(t3, 1):
            self._rep.lossy("BEND_POLE_CURVATURE_DROPPED", "BMAGNET pole-face curvature (RAB1/RAB2) has no IR field",
                            element=el.name, kind="Bend")
        self._add(el, tag)
        self._zrot_to_close = -zrot if zrot else None

    def _c_zrot(self, c: Card, tag: dict, kind: str | None) -> None:
        a = _f(c.lines[0].split(), 0)
        close = getattr(self, "_zrot_to_close", None)
        if close is not None and abs(a - close) < 1e-9:
            self._zrot_to_close = None                            # the ZROT that undoes a rotated magnet
            return
        self._zrot_to_close = None
        if kind == "Patch":
            prev = self._elements[-1] if self._elements else None
            if prev is not None and prev.kind == "Patch" and prev.name == tag.get("name"):
                prev.tilt = math.radians(a)                       # the ZROT of a patch's tilt
            else:
                keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot")
                self._add(Patch(name=self._name(c, tag), tilt=math.radians(a),
                                **{k: float(tag.get(k, 0.0)) for k in keys}, provenance=self._prov(c)), tag)
            return
        if self._pending_zrot is not None:
            self._flush_pending()
        self._pending_zrot = a

    def _c_steer(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        fld, nvf = _f(t, 0), int(_f(t, 1))
        prev = self._elements[-1] if self._elements else None
        same = prev is not None and prev.name == tag.get("name") and tag.get("name")
        if kind == "Multipole":
            if same and prev.kind == "Multipole":
                el = prev
            else:
                el = self._add(Multipole(name=self._name(c, tag), provenance=self._prov(c)), tag)
                el.multipole = MagneticMultipoleP()
            if nvf in (0, 2):
                el.multipole.BnL[0] = -fld
            else:
                el.multipole.BsL[0] = fld
            return
        if same and prev.kind == "Kicker" and prev.length == 0.0:
            el = prev
        else:
            el = self._add(Kicker(name=self._name(c, tag), provenance=self._prov(c), electric=nvf >= 2), tag)
        self._norm.setdefault(id(el), ("kick", []))[1].append((nvf, fld))

    def _c_buncher(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        V, deg, harm, r_cm = _f(t, 0) * 1e6, _f(t, 1), _f(t, 2, 1.0) or 1.0, _f(t, 3)
        if self._q < 0:
            deg -= 180.0
        deg = (deg + 180.0) % 360.0 - 180.0
        rf = RFP(frequency_Hz=harm * self.master_f if self.master_f else None, voltage_V=V,
                 phase_rad=math.radians(deg), cavity_type="TRAVELING_WAVE" if tag.get("tw") == "1" else "STANDING_WAVE",
                 n_cell=int(tag["n"]) if tag.get("n") else None)
        full = float(tag["L"]) if tag.get("L") else 0.0           # a thick cavity written at its centre
        self._cavity_pad = 0.0
        self._add(RFCavity(name=self._name(c, tag), length=full, rf=rf, provenance=self._prov(c),
                           aperture=ApertureP.circle(r_cm * 1e-2) if r_cm else None), tag)

    def _c_field(self, c: Card, tag: dict, kind: str | None) -> None:
        self._field_file = c.lines[0].strip()
        self._field_att = _f(c.lines[1].split(), 0, 1.0)
        self._harm = None

    def _c_harm(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        zlg, fh, atte, ncel = _f(t, 0) * 1e-2, _f(t, 1), _f(t, 2, 1.0), int(_f(t, 3, 1.0))
        coefs = []
        for ln in c.lines[2:]:
            coefs += [float(x) for x in ln.split()]
        self._harm = (zlg, fh, atte, ncel, coefs)
        self._field_file = None

    def _field_block(self, el_name: str) -> tuple[list[float], list[float], float] | None:
        """``(z [m], E_z [V/m], f [Hz])`` of the next block of the current FIELD file, or of the HARM series."""
        if self._harm is not None:
            zlg, fh, atte, _ncel, coefs = self._harm
            n = 401
            z = [zlg * i / (n - 1) for i in range(n)]
            ez = [sum(a * math.cos(math.pi * j * zi / zlg) for j, a in enumerate(coefs)) * atte * 1e8 for zi in z]
            return z, ez, fh
        if not self._field_file:
            return None
        path = self._path.parent / self._field_file
        if not path.is_file():
            self._rep.lossy("FIELD_FILE_MISSING", f"FIELD file {self._field_file!r} not found beside the deck",
                            element=el_name, kind="RFCavity")
            return None
        blocks = _read_field_file(path)
        i = self._field_index.get(self._field_file, 0)
        if i >= len(blocks):
            i = len(blocks) - 1                                   # DYNAC keeps the last field
        self._field_index[self._field_file] = i + 1
        z, ez, f = blocks[i]
        return z, [self._field_att * v for v in ez], f

    def _cavity_from_field(self, c: Card, tag: dict, kind: str | None, *, dphase_deg: float,
                           ffield_pct: float) -> None:
        name = self._name(c, tag)
        blk = self._field_block(name)
        if blk is None:
            self._rep.lossy("CAVITY_FIELD_UNKNOWN", f"{c.name} without a readable FIELD/HARM field: a zero-length "
                            "marker", element=name, kind="RFCavity")
            self._add(Marker(name=name, provenance=self._prov(c)), tag)
            return
        z, ez, f = blk
        ez = [v * (1.0 + ffield_pct / 100.0) for v in ez]
        full = float(tag["L"]) if tag.get("L") else float(z[-1] - z[0])
        pad_before = getattr(self, "_cavity_pad", 0.0)
        self._cavity_pad = 0.0
        el = RFCavity(name=name, length=full, provenance=self._prov(c),
                      rf=RFP(frequency_Hz=f or (self.master_f or None), phase_rad=math.radians(dphase_deg),
                             voltage_V=0.0, n_cell=int(tag["n"]) if tag.get("n") else None,
                             cavity_type="TRAVELING_WAVE" if tag.get("tw") == "1" else "STANDING_WAVE"))
        if tag.get("V"):
            el.rf.voltage_V = float(tag["V"])          # lattix's own deck: the voltage it calibrated the field to
        else:
            el.meta["dynac_field"] = {"z": z, "ez": ez}
            self._norm[id(el)] = ("cavity", None)
        el.meta["dynac_active_m"] = float(z[-1] - z[0])
        # the block itself travels with the element so a DYNAC → DYNAC pass re-emits it as it was
        el.native["dynac"] = {"tag": dict(tag), "card": c.name, "cavnum": list(c.lines), "z": list(z), "ez": list(ez),
                              "f": f, "pad_before": pad_before, "pad_after": 0.0}
        if kind == "NCells":
            nc = NCells(name=name, length=el.length, provenance=el.provenance, rf=el.rf,
                        params={"n_cells": int(tag.get("n", 1)), "mode": int(tag.get("mode", 1))})
            nc.native["dynac"] = el.native["dynac"]
            if "dynac_field" in el.meta:
                nc.meta["dynac_field"] = el.meta["dynac_field"]
                self._norm.pop(id(el))
                self._norm[id(nc)] = ("cavity", None)
            el = nc
            self._rep.equivalent("NCELLS_FROM_CAVNUM", "an NCells written as a DYNAC cell-train field comes back "
                                 "with the train's voltage, phase and frequency", element=name, kind="NCells")
        elif kind == "FieldMap":
            self._rep.equivalent("FM_READ_AS_CAVITY", "a field map written as a DYNAC FIELD block comes back as a "
                                 "thick cavity of the map's integrated voltage", element=name, kind="RFCavity")
        self._add(el, tag)

    def _c_cavnum(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[1].split()
        self._cavity_from_field(c, tag, kind, dphase_deg=_f(t, 1), ffield_pct=_f(t, 2))

    def _c_cavmc(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[1].split()
        self._cavity_from_field(c, tag, kind, dphase_deg=_f(t, 1), ffield_pct=_f(t, 2))

    def _c_cavsc(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        cl, T, e0, phase_deg, f_MHz, att = _f(t, 3) * 1e-2, _f(t, 4), _f(t, 10), _f(t, 11), _f(t, 14), _f(t, 15, 1.0)
        V = e0 * 1e6 * cl * T * att
        name = self._name(c, tag)
        # MEASURED (docs/oracles.md, Phase 5.8): with a βλ cell a CAVSC gap gains E0·T·L·cos φ within 0.3 %
        # for a proton and an H⁻ alike (its phase is not charge-signed, unlike BUNCHER's)
        self._rep.equivalent("CAVSC_AS_GAP", "a DYNAC DTL cell (CAVSC) is a thin gap of E0·T·L at the cell "
                             "middle between two half-cell drifts; the TTF derivatives (0.3 % here) are dropped",
                             element=name, kind="RFCavity")
        self._add(Drift(name=f"{name}_in", length=cl / 2, provenance=self._prov(c)))
        rf = RFP(frequency_Hz=f_MHz * 1e6 if f_MHz else (self.master_f or None), voltage_V=V,
                 phase_rad=math.radians(phase_deg), ttf=T)
        self._add(RFCavity(name=name, length=0.0, rf=rf, provenance=self._prov(c)), tag)
        self._add(Drift(name=f"{name}_out", length=cl / 2, provenance=self._prov(c)))

    def _c_newf(self, c: Card, tag: dict, kind: str | None) -> None:
        f = _f(c.lines[0].split(), 0)
        self.master_f = f
        self._add(Freq(name=self._name(c, tag), frequency_Hz=f, provenance=self._prov(c)), tag)

    def _c_nref(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        dephas, dew, _iref, irewf = _f(t, 0), _f(t, 1), int(_f(t, 2)), int(_f(t, 3, 1.0))
        name = self._name(c, tag)
        if irewf == 1:
            el = ReferenceChange(name=name, dE_ref_eV=dew * 1e6, provenance=self._prov(c))
        elif irewf == 2:
            el = ReferenceChange(name=name, energy_eV=dew * 1e6, provenance=self._prov(c))
        else:
            el = ReferenceChange(name=name, provenance=self._prov(c))
            self._norm[id(el)] = ("nref_pct", dew)
        if tag.get("dE"):
            el.dE_ref_eV = float(tag["dE"])
        if tag.get("E"):
            el.energy_eV = float(tag["E"])
        if dephas:
            el.dphase_rad = math.radians(dephas)
        self._add(el, tag)

    def _c_aliner(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        xl, yl, xpl, ypl = _f(t, 0) * 1e-2, _f(t, 1) * 1e-2, _f(t, 2) * 1e-3, _f(t, 3) * 1e-3
        open_el = getattr(self, "_shift_open", None)
        if open_el is not None and open_el.shift is not None and self._elements and self._elements[-1] is open_el \
                and abs(xl - open_el.shift.x_offset) < 1e-12 and abs(yl - open_el.shift.y_offset) < 1e-12:
            self._shift_open = None                              # the closing ALINER of a misaligned element
            return
        if xpl or ypl:
            self._rep.lossy("ALINER_ANGLES_AS_PATCH", "ALINER angle offsets have no exact IR counterpart; kept as "
                            "a patch's pitch/yaw", element=self._name(c, tag), kind="Patch")
        if kind == "Patch":
            keys = ("z_offset", "x_rot", "y_rot")
            self._add(Patch(name=self._name(c, tag), x_offset=-xl, y_offset=-yl,
                            **{k: float(tag.get(k, 0.0)) for k in keys}, provenance=self._prov(c)), tag)
            return
        if self._pending_shift is None and not (xpl or ypl):
            self._pending_shift = (-xl, -yl)                     # an element misaligned by +d is a beam shift −d
            self._pending_shift_card = c
            return
        self._add(Patch(name=self._name(c, tag), x_offset=-xl, y_offset=-yl, y_rot=-xpl, x_rot=ypl,
                        provenance=self._prov(c)), tag)

    def _c_changref(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        a = _f(t, 2)
        self._rep.equivalent("CHANGREF_AS_PATCH", "CHANGREF yaws the reference direction: a patch with y_rot",
                             element=self._name(c, tag), kind="Patch")
        self._add(Patch(name=f"changref_{self._n}", y_rot=math.radians(a), provenance=self._prov(c)), tag)

    def _c_twqa(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        q = _f(t, 1)
        self._pending_tilt = q if q else None

    def _c_reject(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        vals = tuple(_f(t, i, d) for i, d in zip(range(1, 6), _DEFAULT_REJECT, strict=True))
        if all(abs(v - d) < 1e-12 for v, d in zip(vals, _DEFAULT_REJECT, strict=True)):
            self._pending_reject = None
            return
        _wdisp, _wphas, wx, wy, rlim = vals
        if rlim < 400.0 and rlim <= min(wx, wy):
            ap = ApertureP.circle(rlim * 1e-2)
        else:
            ap = ApertureP.rect(wx * 1e-2, wy * 1e-2)
        self._pending_reject = ap

    def _c_emit(self, c: Card, tag: dict, kind: str | None) -> None:
        name = self._name(c, tag)
        if kind == "Instrument":
            self._add(Instrument(name=name, length=0.0, family=tag.get("family", "MONITOR"), provenance=self._prov(c)),
                      tag)
        elif kind == "Freq":
            self._add(Freq(name=name, frequency_Hz=float(tag.get("f", 0.0)), provenance=self._prov(c)), tag)
        else:
            self._add(Marker(name=name, provenance=self._prov(c)), tag)

    _c_emitl = _c_emit

    def _c_stripper(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        qs, ths = int(_f(t, 0)), _f(t, 2)
        mat = tag.get("material") or _MATERIAL_OF.get(qs, f"Z{qs}")
        el = Foil(name=self._name(c, tag), material=mat, thickness_kg_per_m2=float(tag.get("thick", ths * 10.0)),
                  dE_ref_eV=(float(tag["dE"]) if tag.get("dE") else None), provenance=self._prov(c))
        self._add(el, tag)

    def _c_rfqptq(self, c: Card, tag: dict, kind: str | None) -> None:
        fname = c.lines[0].strip()
        ncell = int(_f(c.lines[1].split(), 0))
        length = 0.0
        path = self._path.parent / fname
        if path.is_file():
            for k, ln in enumerate(path.read_text(encoding="latin-1", errors="replace").splitlines()):
                t = ln.split()
                if len(t) >= 4 and k < ncell:
                    try:
                        if int(float(t[0])) >= 1:
                            length += float(t[3]) * 1e-2
                    except ValueError:
                        continue
        name = self._name(c, tag)
        self._rep.lossy("RFQ_NOT_TRANSLATED", f"RFQPTQ ({ncell} cells from {fname!r}) has no IR representation; a "
                        "drift of the cells' total length keeps the survey", element=name, kind="RFQCell")
        self._add(Drift(name=name, length=length, provenance=self._prov(c)), tag)

    def _c_egun(self, c: Card, tag: dict, kind: str | None) -> None:
        name = self._name(c, tag)
        self._rep.lossy("EGUN_NOT_TRANSLATED", "a DC electron gun has no IR representation; a marker",
                        element=name, kind="RFCavity")
        self._add(Marker(name=name, provenance=self._prov(c)), tag)

    def _c_fsole(self, c: Card, tag: dict, kind: str | None) -> None:
        name = self._name(c, tag)
        self._rep.lossy("FSOLE_NOT_TRANSLATED", "a tabulated solenoid field (FSOLE) is not read; a marker",
                        element=name, kind="Solenoid")
        self._add(Marker(name=name, provenance=self._prov(c)), tag)

    def _c_quadsxt(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        iksq, args, argq, L, r_cm = int(_f(t, 0)), _f(t, 1), _f(t, 2), _f(t, 3) * 1e-2, _f(t, 4)
        el = Quadrupole(name=self._name(c, tag), length=L, provenance=self._prov(c),
                        aperture=ApertureP.circle(r_cm * 1e-2) if r_cm else None)
        el.multipole = MagneticMultipoleP()
        if iksq != 0:
            el.multipole.Bn[1] = 10.0 * argq / r_cm if r_cm else 0.0
            el.multipole.Bn[2] = 2.0 * (args * 0.1) / (r_cm * 1e-2) ** 2 if r_cm else 0.0
        else:
            self._norm[id(el)] = ("kq2_ks2", argq, args)
        self._add(el, tag)

    def _c_soquad(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        iksq, args, argq, L = int(_f(t, 0)), _f(t, 1), _f(t, 2), _f(t, 3) * 1e-2
        name = self._name(c, tag)
        el = Solenoid(name=name, length=L, provenance=self._prov(c))
        if iksq != 0:
            el.solenoid = SolenoidP(Bsol_T=args * 0.1)
        else:
            self._norm[id(el)] = ("ksol", args * 100.0)
        if argq:
            self._rep.lossy("SOQUAD_QUAD_DROPPED", "the quadrupole part of a SOQUAD is dropped (no combined element "
                            "in the IR)", element=name, kind="Solenoid")
        self._add(el, tag)

    def _c_quaelec(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        L = _f(t, 0) * 1e-2
        name = self._name(c, tag)
        self._rep.lossy("EQUAD_TO_DRIFT", "an electrostatic quadrupole has no magnetic equivalent in the IR; a drift",
                        element=name, kind="Quadrupole")
        self._add(Drift(name=name, length=L, provenance=self._prov(c)), tag)

    def _c_quafk(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[0].split()
        ityqu, k, L, r_cm = int(_f(t, 0)), _f(t, 1), _f(t, 2) * 1e-2, _f(t, 3)
        name = self._name(c, tag)
        if ityqu == 0:
            self._rep.lossy("EQUAD_TO_DRIFT", "an electrostatic quadrupole has no magnetic equivalent in the IR; a "
                            "drift", element=name, kind="Quadrupole")
            self._add(Drift(name=name, length=L, provenance=self._prov(c)), tag)
            return
        el = Quadrupole(name=name, length=L, provenance=self._prov(c),
                        aperture=ApertureP.circle(r_cm * 1e-2) if r_cm else None)
        el.multipole = MagneticMultipoleP()
        self._norm[id(el)] = ("kq2", k)
        self._add(el, tag)

    def _c_edflec(self, c: Card, tag: dict, kind: str | None) -> None:
        t = c.lines[1].split()
        rmo, angl = _f(t, 0) * 1e-2, math.radians(_f(t, 1))
        name = self._name(c, tag)
        self._rep.lossy("EDFLEC_TO_DRIFT", "an electrostatic dipole has no IR kind; a drift of its arc length",
                        element=name, kind="Bend")
        self._add(Drift(name=name, length=abs(rmo * angl), provenance=self._prov(c)), tag)

    def _c_rfqcl(self, c: Card, tag: dict, kind: str | None) -> None:
        name = self._name(c, tag)
        self._rep.lossy("RFQ_NOT_TRANSLATED", "RFQCL (deprecated) is not read; a marker", element=name, kind="RFQCell")
        self._add(Marker(name=name, provenance=self._prov(c)), tag)

    def _c_tilt(self, c: Card, tag: dict, kind: str | None) -> None:
        self._directive(c, tag)

    def _directive(self, c: Card, tag: dict) -> None:
        name = self._name(c, tag) if tag.get("name") else f"{c.name.lower()}_{self._n + 1}"
        self._n += 1
        role = "other"
        if c.name in ("SCDYNAC", "SCDYNEL", "SCPOS"):
            role = "space_charge"
        elif c.name in ("EMITGR", "ENVEL", "PROFGR", "WRBEAM", "EMIPRT", "ACCEPT", "T3D"):
            role = "output"
        elif c.name in ("TOF", "REFCOG", "MMODE", "RANDALI", "CHASE", "COMPRES", "DCBEAM", "ZONES", "SECORD",
                        "FIRORD", "RASYN", "RWFIELD", "TILT", "TILZ", "COMMENT"):
            role = "mode"
        fmt = tag.get("format", "dynac")
        card = tag.get("card", c.name) if fmt != "dynac" else c.name
        args = tag.get("args", "").split() if fmt != "dynac" else list(c.lines)
        self._elements.append(Directive(name=name, format=fmt, card=card, args=args,
                                        role=tag.get("role", role) if fmt != "dynac" else role,
                                        provenance=self._prov(c)))

    # ------------------------------------------------------------------ second pass
    def _second_pass(self, lat: Lattice) -> None:
        from lattix.formats.impactt.rfprofile import fourier_coefficients, gain_from_profile

        done: set[int] = set()
        cavities = [pl.element for pl in propagate(lat) if self._norm.get(id(pl.element), ("",))[0] == "cavity"]
        for cav in cavities:                     # the voltage depends on the entry energy: resolve in order
            pl = next(q for q in propagate(lat) if q.element is cav)
            ref = pl.ref_in or lat.reference
            fld = cav.meta.pop("dynac_field")
            z, ez = fld["z"], fld["ez"]
            period = float(z[-1] - z[0])
            f = float(cav.rf.frequency_Hz or self.master_f or 0.0)
            if period > 0 and f:
                coefs = fourier_coefficients([v - z[0] for v in z], ez, period, 60)
                _dE, V, _phi = gain_from_profile(coefs, period, 1.0, 0.0, f, 0.0, ref.kinetic_energy_eV,
                                                 ref.species.mass_eV, ref.species.charge)
                cav.rf.voltage_V = float(V)
                cav.rf.frequency_Hz = f
            done.add(id(cav))
        for p in propagate(lat):
            el = p.element
            if id(el) in done or id(el) not in self._norm:
                continue
            done.add(id(el))
            ref = p.ref_in or lat.reference
            brho = ref.brho_signed
            what, *a = self._norm[id(el)]
            if what == "ksol":
                el.solenoid = SolenoidP(Bsol_T=2.0 * float(a[0]) * brho)
            elif what == "ks2":
                el.multipole.Bn[2] = 2.0 * float(a[0]) * 1e6 * brho          # cm⁻³ → m⁻³, times Bρ [T·m]
            elif what == "kq2":
                el.multipole.Bn[1] = float(a[0]) * 1e4 * brho                # cm⁻² → m⁻²
            elif what == "kq2_ks2":
                el.multipole.Bn[1] = float(a[0]) * 1e4 * brho
                el.multipole.Bn[2] = 2.0 * float(a[1]) * 1e6 * brho
            elif what == "n":
                xn, rho = float(a[0]), float(a[1])
                el.multipole.Bn[1] = -xn / (rho * rho) * brho
            elif what == "kick":
                for nvf, fld in a[0]:
                    if nvf >= 2:
                        erho = 1e3 * ref.species.mass_eV * 1e-6 * (ref.gamma ** 2 - 1.0) / ref.species.charge
                        kick = fld / erho if erho else 0.0
                    else:
                        kick = fld / brho if brho else 0.0
                    if nvf in (0, 2):
                        el.hkick = kick
                    else:
                        el.vkick = kick
            elif what == "nref_pct":
                el.dE_ref_eV = float(a[0]) * 1e-2 * ref.kinetic_energy_eV


def _read_field_file(path: Path) -> list[tuple[list[float], list[float], float]]:
    """The ``(z [m], E_z [V/m], f [Hz])`` blocks of a DYNAC FIELD file (frequency line, pairs, ``0 0``)."""
    blocks: list[tuple[list[float], list[float], float]] = []
    z: list[float] = []
    e: list[float] = []
    f = 0.0
    expect_freq = True
    for ln in path.read_text(encoding="latin-1", errors="replace").splitlines():
        t = ln.split()
        if not t:
            continue
        if expect_freq:
            try:
                f = float(t[0])
            except ValueError:
                continue
            expect_freq = False
            continue
        if len(t) < 2:
            continue
        zi, ei = float(t[0]), float(t[1])
        if zi == 0.0 and ei == 0.0 and z:
            blocks.append((z, e, f))
            z, e, expect_freq = [], [], True
            continue
        z.append(zi)
        e.append(ei)
    if z:
        blocks.append((z, e, f))
    return blocks


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
