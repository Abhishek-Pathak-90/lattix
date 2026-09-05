"""Cheetah LatticeJSON → IR (see the package docstring for the measured conventions)."""
from __future__ import annotations

import json
import math
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

_DROPPED = ("TransverseDeflectingCavity", "Undulator", "SpaceChargeKick")


def _f(params: dict, key: str, default: float = 0.0) -> float:
    v = params.get(key, default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v if v is not None else default)


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)`` for a Cheetah LatticeJSON file."""

    format = "cheetah"

    def read(self, path: Path, *, strict: bool = False, species: str | Species | None = None,
             kinetic_energy_eV: float | None = None, frequency_Hz: float | None = None,
             root: str | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="cheetah", source_file=str(path))
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or "elements" not in doc:
            raise ValueError(f"{path.name}: not a Cheetah LatticeJSON document (no 'elements')")
        version = str(doc.get("version", ""))
        if not version.startswith("cheetah"):
            rep.equivalent("LATTICEJSON_VERSION", f"LatticeJSON version {version!r} is not a cheetah-* one; "
                           "read with Cheetah 0.8 semantics")
        ref = self._reference(doc, species, kinetic_energy_eV, frequency_Hz, rep, path)
        self._norm: dict[int, tuple[str, float, float]] = {}      # id(element) → (what, value, value2)
        self._rep = rep
        self._q = ref.species.charge or 1
        elements: dict[str, Element] = {}
        by_json: dict[str, Element] = {}
        used: set[str] = set()
        for jname, entry in (doc.get("elements") or {}).items():
            ctype, params = entry[0], dict(entry[1] or {})
            el = self._convert(jname, ctype, params)
            if el is None:
                continue
            name = el.name
            if name in used:
                k = 2
                while f"{name}_{k}" in used:
                    k += 1
                name = f"{name}_{k}"
                el.name = name
            used.add(name)
            elements[name] = el
            by_json[jname] = el
        lattices = doc.get("lattices") or {}
        root_name = root or doc.get("root") or (next(iter(lattices)) if lattices else None)
        lat = Lattice(name=str(doc.get("title") or root_name or path.stem), reference=ref)
        lat.meta["source_format"] = "cheetah"
        lat.meta["source_file"] = str(path)
        if doc.get("info"):
            lat.meta["cheetah_info"] = str(doc["info"])
        for el in elements.values():
            lat.elements[el.name] = el
        self._folded: set[str] = set()
        for lname, members in lattices.items():
            items = []
            folded = self._fold(members, by_json, lattices)
            for m in folded:
                if m in lattices:
                    items.append(LineItem(ref=m))
                elif m in by_json:
                    items.append(LineItem(ref=by_json[m].name))
                else:
                    rep.dropped("LATTICEJSON_UNKNOWN_MEMBER", f"lattice {lname!r} refers to {m!r}, which is "
                                "neither an element nor a lattice", element=m, kind=None)
            lat.lines[lname] = Line(name=lname, items=items)
        if root_name is None:
            raise ValueError(f"{path.name}: the document defines no lattice")
        lat.use = root_name
        for name in self._folded:                         # apertures and body drifts folded onto their element
            lat.elements.pop(name, None)
        self._second_pass(lat)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ reference
    def _reference(self, doc, species, kinetic_energy_eV, frequency_Hz, rep, path) -> ReferenceParticle:
        tag = parse_reference_tag(str(doc.get("info") or ""))
        sp = None
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        elif tag is not None:
            sp = tag.species
        ke = kinetic_energy_eV if kinetic_energy_eV is not None else (tag.kinetic_energy_eV if tag else None)
        f = frequency_Hz if frequency_Hz is not None else (tag.rf_frequency_Hz if tag else None)
        if sp is None or ke is None:
            raise ValueError(f"{path.name}: a Cheetah LatticeJSON carries no beam; pass species=<name> and "
                             "kinetic_energy_eV=<eV> to read(...) (or keep lattix's reference tag in 'info')")
        if tag is not None and (species is None or kinetic_energy_eV is None):
            rep.equivalent("REFERENCE_FROM_TAG", "reference particle taken from the lattix tag in 'info'")
        return ReferenceParticle(species=sp, kinetic_energy_eV=float(ke), rf_frequency_Hz=f)

    # ------------------------------------------------------------------ folding
    def _fold(self, members: list[str], by_json: dict[str, Element], lattices: dict) -> list[str]:
        """Apertures and body drifts the writer put beside an element go back onto it."""
        out: list[str] = []
        pending_entry: list[tuple[str, Element]] = []
        for m in members:
            el = by_json.get(m)
            if el is None:
                out.append(m)
                continue
            role = (el.meta.get("cheetah_lattix") or {}).get("role")
            if role == "aperture_entry":
                pending_entry.append((m, el))
                continue
            if role == "aperture_exit" and out:
                prev = by_json.get(out[-1])
                if prev is not None and el.kind == "Collimator" and el.aperture is not None:
                    self._attach_aperture(prev, el.aperture, "EXIT")
                    self._folded.add(el.name)
                    continue
            if role == "extra" and out and el.kind == "Drift":
                prev = by_json.get(out[-1])
                if prev is not None and prev.kind == "Collimator":
                    prev.length = el.length
                    self._folded.add(el.name)
                    continue
            for _, ap_el in pending_entry:
                if ap_el.aperture is not None:
                    self._attach_aperture(el, ap_el.aperture, "ENTRANCE")
                    self._folded.add(ap_el.name)
            pending_entry = []
            out.append(m)
        for m, _ in pending_entry:
            out.append(m)
        return out

    @staticmethod
    def _attach_aperture(el: Element, ap: ApertureP, where: str) -> None:
        if el.aperture is None:
            el.aperture = ap.model_copy(update={"aperture_at": where})
        elif el.aperture.aperture_at != where:
            el.aperture.aperture_at = "BOTH_ENDS"

    # ------------------------------------------------------------------ elements
    def _convert(self, jname: str, ctype: str, params: dict) -> Element | None:
        meta = dict((params.get("metadata") or {}).get("lattix") or {})
        role = meta.get("role")
        # the apertures and body drifts written beside an element carry its name in the metadata,
        # but only the main element may claim it (they are folded back onto it afterwards)
        name = str(meta.get("name") or jname) if role in (None, "main") else jname
        kind = meta.get("kind")
        prov = Provenance(format="cheetah", original_name=jname,
                          original_type=str(meta.get("original_type") or ctype))
        common = {"name": name, "provenance": prov}
        length = _f(params, "length")
        mis = params.get("misalignment")
        shift = None
        if isinstance(mis, list) and len(mis) >= 2 and (mis[0] or mis[1]):
            shift = BodyShiftP(x_offset=float(mis[0]), y_offset=float(mis[1]))
        el: Element | None = None
        if ctype == "Drift":
            if kind == "Instrument":
                el = Instrument(length=length, family=str(meta.get("family", "MONITOR")),
                                params=dict(meta.get("params") or {}), **common)
            elif kind == "Patch":
                el = Patch(length=length, **{k: float(meta.get(k, 0.0)) for k in ("x_offset", "y_offset", "z_offset",
                                                                                   "x_rot", "y_rot", "tilt")}, **common)
            elif kind == "Collimator" and role in (None, "main"):
                el = Collimator(length=length, **common)          # a collimator without limits
            else:
                el = Drift(length=length, **common)
        elif ctype == "Quadrupole":
            el = Quadrupole(length=length, shift=shift, **common)
            el.multipole = MagneticMultipoleP(tilt={1: _f(params, "tilt")} if params.get("tilt") else {})
            self._norm[id(el)] = ("Bn", 1, _f(params, "k1"))
        elif ctype == "Sextupole":
            el = Sextupole(length=length, shift=shift, **common)
            el.multipole = MagneticMultipoleP(tilt={2: _f(params, "tilt")} if params.get("tilt") else {})
            self._norm[id(el)] = ("Bn", 2, _f(params, "k2"))
        elif ctype in ("Dipole", "RBend"):
            angle = _f(params, "angle")
            if ctype == "RBend":
                e1, e2, rect = _f(params, "rbend_e1") + angle / 2.0, _f(params, "rbend_e2") + angle / 2.0, True
            else:
                e1, e2, rect = _f(params, "dipole_e1"), _f(params, "dipole_e2"), bool(meta.get("rect", False))
            fint = _f(params, "fringe_integral")
            fintx = _f(params, "fringe_integral_exit", fint)
            el = Bend(length=length, shift=shift, **common)
            el.bend = BendP(angle=angle, e1=e1, e2=e2, hgap=_f(params, "gap") / 2.0, edge_int1=fint,
                            edge_int2=(fintx if abs(fintx - fint) > 1e-15 else None), tilt_ref=_f(params, "tilt"),
                            rect=rect)
            if _f(params, "k1"):
                el.multipole = MagneticMultipoleP()
                self._norm[id(el)] = ("Bn", 1, _f(params, "k1"))
        elif ctype == "Solenoid":
            el = Solenoid(length=length, shift=shift, **common)
            self._norm[id(el)] = ("Bsol", 0, _f(params, "k"))
        elif ctype == "Cavity":
            voltage = -_f(params, "voltage") * self._q
            phase = float(meta["phase_rad"]) if "phase_rad" in meta else -math.radians(_f(params, "phase"))
            rf = RFP(frequency_Hz=_f(params, "frequency") or None, voltage_V=voltage, phase_rad=phase,
                     cavity_type="TRAVELING_WAVE" if params.get("cavity_type") == "traveling_wave" else "STANDING_WAVE",
                     dE_ref_eV=(float(meta["dE_ref_eV"]) if "dE_ref_eV" in meta else None),
                     n_cell=(int(meta["n_cell"]) if "n_cell" in meta else None),
                     L_active_m=(float(meta["L_active_m"]) if "L_active_m" in meta else None))
            el = RFCavity(length=length, rf=rf, **common)
        elif ctype in ("HorizontalCorrector", "VerticalCorrector", "CombinedCorrector"):
            hk = _f(params, "angle") if ctype == "HorizontalCorrector" else _f(params, "horizontal_angle")
            vk = _f(params, "angle") if ctype == "VerticalCorrector" else _f(params, "vertical_angle")
            if kind == "Multipole":
                el = Multipole(length=length, **common)
                el.multipole = MagneticMultipoleP()
                self._norm[id(el)] = ("kickL", hk, vk)
            else:
                el = Kicker(length=length, hkick=hk, vkick=vk, **common)
        elif ctype == "Aperture":
            x, y = abs(_f(params, "x_max")), abs(_f(params, "y_max"))
            ap = ApertureP(shape="ELLIPTICAL" if params.get("shape") == "elliptical" else "RECTANGULAR",
                           x_limits=(-x, x), y_limits=(-y, y))
            el = Collimator(length=0.0, aperture=ap, **common)
        elif ctype in ("BPM", "Screen"):
            el = Instrument(length=0.0, family=str(meta.get("family") or ctype.upper()),
                            params=dict(meta.get("params") or {}), **common)
        elif ctype == "Marker":
            el = self._marker(kind, meta, common)
        elif ctype == "CustomTransferMap":
            tm = params.get("predefined_transfer_map") or []
            m = [[float(tm[i][j]) for j in range(6)] for i in range(6)] if len(tm) >= 6 else None
            if m is None:
                self._rep.lossy("TAYLOR_UNSUPPORTED", "CustomTransferMap without a 7x7 map read as a drift",
                                element=name, kind="Taylor")
                el = Drift(length=length, **common)
            else:
                off = [float(tm[i][6]) if len(tm[i]) > 6 else 0.0 for i in range(6)]
                basis = meta.get("basis") or "cheetah"
                el = Taylor(length=length, matrix=m, offset=off, basis=basis, **common)
                if basis == "cheetah":
                    self._norm[id(el)] = ("taylor", 0, 0.0)      # → common basis at the entry energy (2nd pass)
                else:
                    self._rep.equivalent("TAYLOR_BASIS_CHEETAH", f"the map's source basis {basis!r} was re-used "
                                         "verbatim in Cheetah and is kept as such", element=name, kind="Taylor")
        elif ctype in _DROPPED:
            self._rep.dropped("UNSUPPORTED_CHEETAH_ELEMENT", f"Cheetah {ctype} has no IR kind; a drift/marker of "
                              "the same length keeps the survey (parameters in native['cheetah'])",
                              element=name, kind=ctype)
            el = Drift(length=length, **common) if length else Marker(**common)
            el.native["cheetah"] = {"type": ctype, **{k: v for k, v in params.items() if k != "metadata"}}
        else:
            self._rep.dropped("UNSUPPORTED_CHEETAH_ELEMENT", f"unknown Cheetah element type {ctype!r}; a "
                              "drift/marker of the same length keeps the survey", element=name, kind=ctype)
            el = Drift(length=length, **common) if length else Marker(**common)
            el.native["cheetah"] = {"type": ctype, **{k: v for k, v in params.items() if k != "metadata"}}
        if el is not None and meta:
            el.meta["cheetah_lattix"] = meta
            if meta.get("degraded_from"):
                el.meta["cheetah_degraded_from"] = str(meta["degraded_from"])
        return el

    def _marker(self, kind, meta: dict, common: dict) -> Element:
        if kind == "Instrument":
            return Instrument(length=0.0, family=str(meta.get("family", "MONITOR")),
                              params=dict(meta.get("params") or {}), **common)
        if kind == "Foil":
            return Foil(material=str(meta.get("material", "C")),
                        thickness_kg_per_m2=float(meta.get("thickness_kg_per_m2", 0.0)),
                        dE_ref_eV=meta.get("dE_ref_eV"), **common)
        if kind == "Patch":
            keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
            return Patch(**{k: float(meta.get(k, 0.0)) for k in keys}, **common)
        if kind == "ReferenceChange":
            return ReferenceChange(dE_ref_eV=meta.get("dE_ref_eV"), energy_eV=meta.get("energy_eV"), **common)
        if kind == "Freq":
            return Freq(frequency_Hz=float(meta.get("frequency_Hz", 0.0)), **common)
        if kind == "Directive":
            return Directive(format=str(meta.get("format", "tracewin")), card=str(meta.get("card", "")),
                             args=[str(a) for a in meta.get("args") or []], role=str(meta.get("ir_role", "other")),
                             **common)
        if kind == "Collimator":
            return Collimator(length=0.0, **common)
        return Marker(**common)

    # ------------------------------------------------------------------ second pass
    def _second_pass(self, lat: Lattice) -> None:
        """Normalized strengths become fields with the signed rigidity at each element's entrance."""
        done: set[int] = set()
        for p in propagate(lat):
            el = p.element
            if id(el) in done or id(el) not in self._norm:
                continue
            done.add(id(el))
            brho = (p.ref_in or lat.reference).brho_signed
            what, a, b = self._norm[id(el)]
            if what == "Bn":
                el.multipole.Bn[int(a)] = float(b) * brho
            elif what == "Bsol":
                el.solenoid = SolenoidP(Bsol_T=2.0 * float(b) * brho)
            elif what == "kickL":
                el.multipole.BnL[0] = -float(a) * brho
                el.multipole.BsL[0] = float(b) * brho
            elif what == "taylor":
                # Cheetah's (x, px, y, py, τ, ΔE/p0c) map into the IR's common basis at the entry energy:
                # R_common = T R_cheetah T⁻¹ with x_common = T x_cheetah (the writer's inverse)
                import numpy as np

                from lattix.formats.cheetah.writer import similarity_diag
                from lattix.oracles.base import Basis
                from lattix.oracles.basis import transform_matrix

                ref_in = p.ref_in or lat.reference
                d = np.diag(transform_matrix(Basis.CHEETAH, ref_in.kinetic_energy_eV, ref_in.species.mass_eV))
                el.matrix, el.offset = similarity_diag(el.matrix, el.offset, d, np.ones(6))
                el.basis = "common"


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
