"""Synergia lattice JSON → IR.

The reference particle comes from the archive itself (charge, mass, total energy in GeV; lattix's own
``lattix`` block adds the species name and the RF clock); every element's MAD-X-named attributes become
lab fields with the rigidity the deck's ``energy_mode`` implies (:func:`lattix.ir.energy_mode.restore_energy_mode`,
``"delta"`` when the deck is not lattix's).  The ``lattix`` string attribute restores the IR kinds and the
data Synergia has no attribute for.
"""
from __future__ import annotations

import json
import math
import re
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
from lattix.ir.energy_mode import ENERGY_MODES, restore_energy_mode
from lattix.ir.lattice import Lattice
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name

_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')


def _parse_tag(text: str) -> dict[str, str]:
    return {k: (q if v.startswith('"') else v) for k, v, q in _KV.findall(text or "")}


def _param_value(v: str):
    """An instrument parameter from the tag: an int stays an int (``diag:1``), a float a float, else text."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _val(v) -> float:
    """A lazy attribute value: ``{"value0": "<number>"}``, a bare number or a numeric string."""
    if isinstance(v, dict):
        v = v.get("value0", 0.0)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"attribute expression {v!r} cannot be evaluated by lattix's Synergia reader") from None
    return float(v)


def _aperture_of(strings: dict, d: dict) -> ApertureP | None:
    """Synergia's aperture operation attributes (aperture_operation.h) as an :class:`ApertureP`."""
    kind = str(strings.get("aperture_type") or "").lower()
    if kind == "circular" and d.get("circular_aperture_radius") is not None:
        r = float(d["circular_aperture_radius"])
        return ApertureP(shape="ELLIPTICAL", x_limits=(-r, r), y_limits=(-r, r))
    if kind == "elliptical" and d.get("elliptical_aperture_horizontal_radius") is not None:
        hx = float(d["elliptical_aperture_horizontal_radius"])
        hy = float(d.get("elliptical_aperture_vertical_radius", 0.0))
        return ApertureP(shape="ELLIPTICAL", x_limits=(-hx, hx), y_limits=(-hy, hy))
    if kind == "rectangular" and d.get("rectangular_aperture_width") is not None:
        hx = float(d["rectangular_aperture_width"]) / 2.0
        hy = float(d.get("rectangular_aperture_height", 0.0)) / 2.0
        return ApertureP(shape="RECTANGULAR", x_limits=(-hx, hx), y_limits=(-hy, hy))
    return None


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)`` for a Synergia lattice JSON."""

    format = "synergia"

    def read(self, path: Path, *, strict: bool = False, species: str | Species | None = None,
             kinetic_energy_eV: float | None = None, frequency_Hz: float | None = None,
             energy_mode: str | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="synergia", source_file=str(path))
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or "value0" not in doc or "elements" not in (doc.get("value0") or {}):
            raise ValueError(f"{path.name}: not a Synergia lattice JSON (no value0.elements)")
        body = doc["value0"]
        meta = doc.get("lattix") or {}
        self._rep = rep
        ref = self._reference(body, meta, species, kinetic_energy_eV, frequency_Hz, rep, path)
        self._q = ref.species.charge
        # a lattix deck says how its normalized strengths were formed; a native Synergia deck is read like a
        # native MAD-X one: every strength with the start rigidity, the phases straight from lag
        mode = energy_mode or meta.get("energy_mode")
        if mode is not None and mode not in ENERGY_MODES:
            raise ValueError(f"unknown energy_mode {mode!r}")
        elements: list[Element] = []
        self._brho = ref.brho_signed
        for j in body.get("elements") or []:
            el = self._convert(j)
            if el is not None:
                elements.append(el)
        elements = self._fold(elements)
        name = str(body.get("name") or path.stem)
        lat = self._assemble(str(meta.get("use") or name), elements, ref)
        lat.name = name
        lat.meta["source_format"] = "synergia"
        lat.meta["source_file"] = str(path)
        self._recombine_taylors(lat)                  # matrix ∘ drift first: the inverse of the writer's order
        if mode is not None:
            restore_energy_mode(lat, rep, mode)       # Synergia's cavities carry no phase slip (measured)
        for e in elements:
            e.meta.pop("synergia_lag", None)
        self._second_pass(lat)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ reference
    def _reference(self, body, meta, species, kinetic_energy_eV, frequency_Hz, rep, path) -> ReferenceParticle:
        rp = body.get("reference_particle_value") or {}
        fm = rp.get("four_momentum") or {}
        mass = float(fm.get("mass", 0.0)) * 1e9
        etot = float(fm.get("energy", 0.0)) * 1e9
        q = int(rp.get("charge", 1))
        sp = None
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        elif meta.get("species"):
            sp = Species(name=str(meta["species"]), mass_eV=float(meta.get("mass_eV", mass)),
                         charge=int(meta.get("charge", q)))
        elif mass:
            for cand in SPECIES.values():
                if abs(cand.mass_eV - mass) / mass < 1e-4 and cand.charge == q:
                    sp = cand
                    break
            if sp is None:
                sp = Species(name=f"ion_m{mass * 1e-9:.6g}GeV_q{q}", mass_eV=mass, charge=q)
                rep.equivalent("SPECIES_ASSUMED", f"reference mass {mass * 1e-9:.6f} GeV, charge {q}: not one of "
                               "lattix's named species; kept as an ion of that mass")
        ke = kinetic_energy_eV if kinetic_energy_eV is not None else (
            float(meta["kinetic_energy_eV"]) if meta.get("kinetic_energy_eV") is not None else
            (etot - sp.mass_eV if (etot and sp is not None) else None))
        if sp is None or ke is None:
            raise ValueError(f"{path.name}: the archive has no reference particle; pass species=<name> and "
                             "kinetic_energy_eV=<eV> to read(...)")
        f = frequency_Hz if frequency_Hz is not None else meta.get("rf_frequency_Hz")
        return ReferenceParticle(species=sp, kinetic_energy_eV=float(ke), rf_frequency_Hz=float(f) if f else None)

    # ------------------------------------------------------------------ elements
    def _convert(self, j: dict) -> Element | None:
        stype = str(j.get("stype") or "").lower()
        name = str(j.get("name") or "")
        strings = {a["key"]: a["value"] for a in j.get("string_attributes") or []}
        d = {a["key"]: _val(a["value"]) for a in j.get("lazy_double_attributes") or []}
        vec = {a["key"]: [_val(x) for x in (a["value"] or [])] for a in j.get("lazy_vector_attributes") or []}
        tag = _parse_tag(strings.get("lattix", ""))
        kind = tag.get("kind")
        prov = Provenance(format="synergia", original_name=name, original_type=stype)
        ir_name = tag.get("name") or name
        common = {"name": ir_name, "provenance": prov}
        L = d.get("l", 0.0)
        brho = self._brho
        el: Element | None = None
        if stype == "drift":
            el = self._drift_kind(kind, L, tag, common)
        elif stype == "quadrupole":
            el = Quadrupole(length=L, **common)
            el.multipole = MagneticMultipoleP(Bn={1: d.get("k1", 0.0) * brho},
                                              Bs={1: d["k1s"] * brho} if d.get("k1s") else {},
                                              tilt={1: d["tilt"]} if d.get("tilt") else {})
            if d.get("hoffset") or d.get("voffset"):
                el.shift = BodyShiftP(x_offset=d.get("hoffset", 0.0), y_offset=d.get("voffset", 0.0))
        elif stype == "sextupole":
            el = Sextupole(length=L, **common)
            el.multipole = MagneticMultipoleP(Bn={2: d.get("k2", 0.0) * brho},
                                              Bs={2: d["k2s"] * brho} if d.get("k2s") else {},
                                              tilt={2: d["tilt"]} if d.get("tilt") else {})
        elif stype == "octupole":
            el = Octupole(length=L, **common)
            el.multipole = MagneticMultipoleP(Bn={3: d.get("k3", 0.0) * brho},
                                              Bs={3: d["k3s"] * brho} if d.get("k3s") else {},
                                              tilt={3: d["tilt"]} if d.get("tilt") else {})
        elif stype == "multipole":
            el = Multipole(**common)
            knl, ksl = vec.get("knl", []), vec.get("ksl", [])
            el.multipole = MagneticMultipoleP(BnL={n: v * brho for n, v in enumerate(knl) if v},
                                              BsL={n: v * brho for n, v in enumerate(ksl) if v})
            if d.get("tilt"):
                el.shift = BodyShiftP(tilt=d["tilt"])
        elif stype in ("sbend", "rbend"):
            angle = d.get("angle", 0.0)
            e1, e2 = d.get("e1", 0.0), d.get("e2", 0.0)
            rect = stype == "rbend" or tag.get("rect") == "1"
            if stype == "rbend":
                e1, e2 = e1 + angle / 2.0, e2 + angle / 2.0
                self._rep.equivalent("RBEND_AS_SECTOR", "an rbend's faces are stored sector-referenced (angle/2 "
                                     "added); its l is taken as the arc", element=ir_name, kind="Bend")
            fint = d.get("fint", 0.0)
            fintx = d.get("fintx")
            el = Bend(length=L, **common)
            el.bend = BendP(angle=angle, e1=e1, e2=e2, edge_int1=fint,
                            edge_int2=(fintx if fintx is not None and abs(fintx - fint) > 1e-15 else None),
                            hgap=d.get("hgap", 0.0), tilt_ref=d.get("tilt", 0.0), rect=rect)
            bn = {}
            if d.get("k1"):
                bn[1] = d["k1"] * brho
            if d.get("k2"):
                bn[2] = d["k2"] * brho
            el.multipole = MagneticMultipoleP(Bn=bn)
            if d.get("k0") is not None:
                el.native["synergia"] = {"k0": d["k0"]}
        elif stype == "solenoid":
            el = Solenoid(length=L, solenoid=SolenoidP(Bsol_T=d.get("ks", 0.0) * brho), **common)
        elif stype == "rfcavity":
            lag = d.get("lag", 0.0)
            phase = float(tag["phase"]) if tag.get("phase") else (lag - 0.25) * 2.0 * math.pi
            rf = RFP(frequency_Hz=d.get("freq", 0.0) * 1e6 or None, voltage_V=d.get("volt", 0.0) * 1e6,
                     phase_rad=phase, harmon=d.get("harmon") or None,
                     cavity_type="TRAVELING_WAVE" if tag.get("tw") == "1" else "STANDING_WAVE",
                     n_cell=int(tag["n"]) if tag.get("n") else None, phase_is_sync=tag.get("raw") != "1")
            el = RFCavity(length=L, rf=rf, **common)
            if tag.get("phase") is None:
                el.meta["synergia_lag"] = lag
        elif stype in ("kicker", "hkicker", "vkicker"):
            el = Kicker(length=L, hkick=d.get("hkick", d.get("kick", 0.0) if stype == "hkicker" else 0.0),
                        vkick=d.get("vkick", d.get("kick", 0.0) if stype == "vkicker" else 0.0), **common)
            if d.get("tilt"):
                el.shift = BodyShiftP(tilt=d["tilt"])
        elif stype == "rcollimator":
            ap = _aperture_of(strings, d)
            if ap is None and "xsize" in d and "ysize" in d:
                ap = ApertureP(shape="RECTANGULAR", x_limits=(-d["xsize"], d["xsize"]),
                               y_limits=(-d["ysize"], d["ysize"]))
            el = Collimator(length=L, aperture=ap, **common)
        elif stype in ("monitor", "hmonitor", "vmonitor", "instrument"):
            params = {}
            for item in (tag.get("params") or "").split(";"):
                if ":" in item:
                    k, v = item.split(":", 1)
                    params[k] = _param_value(v)
            fam = tag.get("family") or {"monitor": "BPM", "hmonitor": "HMONITOR", "vmonitor": "VMONITOR"}.get(
                stype, "INSTRUMENT")
            el = Instrument(length=L, family=fam, params=params, **common)
        elif stype == "marker":
            el = self._marker(kind, tag, common)
        elif stype == "matrix":
            m = [[0.0] * 6 for _ in range(6)]
            off = [0.0] * 6
            for k, v in d.items():
                if re.fullmatch(r"rm[1-6][1-6]", k):
                    m[int(k[2]) - 1][int(k[3]) - 1] = v
                elif re.fullmatch(r"kick[1-6]", k):
                    off[int(k[4]) - 1] = v
            basis = tag.get("basis", "synergia")
            el = Taylor(length=float(tag["L"]) if tag.get("L") else L, matrix=m, offset=off, basis=basis, **common)
            if basis == "synergia":
                el.meta["synergia_taylor"] = True
            if tag.get("L"):
                el.meta["synergia_taylor_split"] = True       # the drift body was divided out of the map
        else:
            self._rep.dropped("UNSUPPORTED_SYNERGIA_ELEMENT", f"Synergia {stype} has no IR kind; a drift/marker "
                              "of the same length keeps the survey (attributes in native['synergia'])",
                              element=ir_name, kind=stype)
            el = Drift(length=L, **common) if L else Marker(**common)
            el.native["synergia"] = {"stype": stype, **d, **{k: v for k, v in vec.items()}}
        if el is not None:
            if el.kind != "Collimator":
                ap = _aperture_of(strings, d)
                if ap is not None:
                    el.aperture = ap
            if tag.get("from"):
                el.meta["synergia_from"] = tag["from"]
            if tag.get("role"):
                el.meta["synergia_role"] = tag["role"]
        return el

    def _drift_kind(self, kind, L, tag, common) -> Element:
        if kind == "NCells":
            return NCells(length=L, **common)
        if kind == "RFQCell":
            return RFQCell(length=L, **common)
        if kind == "Patch":
            keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
            return Patch(length=L, **{k: float(tag.get(k, 0.0)) for k in keys}, **common)
        d = Drift(length=L, **common)
        return d

    def _marker(self, kind, tag, common) -> Element:
        if kind == "Foil":
            return Foil(material=tag.get("material", "C"), thickness_kg_per_m2=float(tag.get("thick", 0.0)),
                        dE_ref_eV=(float(tag["dE"]) if tag.get("dE") else None), **common)
        if kind == "Patch":
            keys = ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
            return Patch(**{k: float(tag.get(k, 0.0)) for k in keys}, **common)
        if kind == "ReferenceChange":
            return ReferenceChange(dE_ref_eV=(float(tag["dE"]) if tag.get("dE") else None),
                                   energy_eV=(float(tag["E"]) if tag.get("E") else None), **common)
        if kind == "Freq":
            return Freq(frequency_Hz=float(tag.get("f", 0.0)), **common)
        if kind == "Directive":
            return Directive(format=tag.get("format", "tracewin"), card=tag.get("card", ""),
                             args=tag.get("args", "").split(), role=tag.get("role", "other"), **common)
        return Marker(**common)

    # ------------------------------------------------------------------ folding / second pass
    @staticmethod
    def _assemble(use: str, elements: list[Element], ref) -> Lattice:
        """A flat archive repeats an element per placement: an identical definition under the same name is
        placed again (``b1`` twice, as the MAD-X reader does); a different one gets a ``_2`` suffix."""
        from lattix.ir.lattice import Line, LineItem

        lat = Lattice(name=use, reference=ref)
        line = Line(name=use)
        seen: dict[str, dict] = {}
        for el in elements:
            existing = lat.elements.get(el.name)
            if existing is not None and type(existing) is type(el):
                dump = el.model_dump(mode="json")
                dump.pop("provenance", None)
                if seen[el.name] == dump:
                    line.items.append(LineItem(ref=el.name))
                    continue
            nm = lat.add_element(el)
            dump = el.model_dump(mode="json")
            dump.pop("provenance", None)
            seen[nm] = dump
            line.items.append(LineItem(ref=nm))
        lat.lines[use] = line
        lat.use = use
        return lat

    def _fold(self, elements: list[Element]) -> list[Element]:
        out: list[Element] = []
        for el in elements:
            role = el.meta.pop("synergia_role", None)
            if role == "body" and el.kind == "Drift" and out and out[-1].kind in ("Multipole", "Foil", "Taylor") \
                    and out[-1].name == el.name:
                if out[-1].kind != "Taylor":
                    out[-1].length = el.length
                continue
            out.append(el)
        return out

    def _recombine_taylors(self, lat: Lattice) -> None:
        """A thick Taylor was written as a thin matrix with the drift over its length divided out (Synergia's
        matrix is thin): multiply the drift back, in Synergia's basis at the entry energy."""
        import numpy as np

        from lattix.ir.walk import propagate
        from lattix.oracles.base import Basis
        from lattix.oracles.basis import drift_common, transform_matrix

        for p in propagate(lat):
            el = p.element
            if el.kind == "Taylor" and el.meta.pop("synergia_taylor_split", None):
                ref_in = p.ref_in or lat.reference
                T = transform_matrix(Basis.SYNERGIA, ref_in.kinetic_energy_eV, ref_in.species.mass_eV)
                D = np.linalg.inv(T) @ drift_common(el.length, ref_in.kinetic_energy_eV, ref_in.species.mass_eV) @ T
                el.matrix = (np.asarray(el.matrix) @ D).tolist()

    def _second_pass(self, lat: Lattice) -> None:
        """Taylor maps written in Synergia's basis go back to the IR's common basis at the entry energy."""
        import numpy as np

        from lattix.formats.cheetah.writer import similarity_diag
        from lattix.ir.walk import propagate
        from lattix.oracles.base import Basis
        from lattix.oracles.basis import transform_matrix

        for p in propagate(lat):
            el = p.element
            if el.kind == "Taylor" and el.meta.pop("synergia_taylor", None):
                ref_in = p.ref_in or lat.reference
                d = np.diag(transform_matrix(Basis.SYNERGIA, ref_in.kinetic_energy_eV, ref_in.species.mass_eV))
                el.matrix, el.offset = similarity_diag(el.matrix, el.offset, d, np.ones(6))
                el.basis = "common"


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
