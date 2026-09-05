"""PyORBIT3 linac XML → IR (see the package docstring for the measured conventions)."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    Drift,
    Element,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Provenance,
    Quadrupole,
    RFCavity,
    Solenoid,
    SolenoidP,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.walk import propagate

_TOL = 1e-9
_TOL2 = 2000          # the same, in half-picometres


def _half_pm(text: str | None) -> int:
    """A position as written, exactly, in half-picometres (the writer's grid)."""
    return int((Decimal(text or "0") * 2 * 10**12).to_integral_value())


def _pm(text: str | None) -> int:
    """A length as written, exactly, in picometres."""
    return int((Decimal(text or "0") * 10**12).to_integral_value())


def _f(params: dict, key: str, default: float = 0.0) -> float:
    v = params.get(key)
    return float(v) if v not in (None, "") else default


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)`` for a PyORBIT3 linac XML."""

    format = "pyorbit"

    def read(self, path: Path, *, strict: bool = False, species: str | Species | None = None,
             kinetic_energy_eV: float | None = None, frequency_Hz: float | None = None,
             sequences: list[str] | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="pyorbit", source_file=str(path))
        text = path.read_text(encoding="utf-8", errors="replace")
        root = ET.fromstring(text)
        seqs = [c for c in root if c.get("length") is not None and (sequences is None or c.tag in sequences)]
        if not seqs:
            raise ValueError(f"{path.name}: no sequence element (a child with a 'length' attribute) found")
        bpm = next((float(s.get("bpmFrequency")) for s in seqs if s.get("bpmFrequency")), None)
        ref = self._reference(text, species, kinetic_energy_eV, frequency_Hz, bpm, rep, path)
        self._rep = rep
        self._q = ref.species.charge or 1
        self._norm: dict[int, tuple[str, float, float]] = {}
        elements: list[Element] = []
        used: set[str] = set()
        # positions are handled exactly, as the decimals written (half-picometres; lengths in
        # picometres); a gap below 1 nm is a touching pair — the writer pushes a node 1 pm past
        # its neighbour when PyORBIT's own arithmetic would see an overlap, and ignoring that
        # push here is what makes write → read → write a fixed point
        s2 = 0
        n_drift = 0
        for seq in seqs:
            cavities = {c.get("name"): float(c.get("frequency")) for cavs in seq.findall("Cavities")
                        for c in cavs.findall("Cavity")}
            seq_len2 = _half_pm(seq.get("length"))
            seq_start2 = s2
            items = [(_half_pm(acc.get("pos")), _pm(acc.get("length")), acc) for acc in seq.findall("accElement")]
            thick = [(pos2 - lp, pos2 + lp, acc) for pos2, lp, acc in items if lp > 0]     # entrance, exit
            thin = [(pos2, acc) for pos2, lp, acc in items if lp <= 0]
            inside: dict[int, list] = {id(acc): [] for _, _, acc in thick}
            standalone = []
            for pos2, acc in thin:
                host = next((h for a, b, h in thick if a < pos2 < b), None)
                if host is None:
                    standalone.append((pos2, 0, acc, None))
                else:
                    inside[id(host)].append((pos2, acc))
            units = standalone + [(a, 1, acc, sorted(inside[id(acc)], key=lambda t: t[0])) for a, b, acc in thick]
            units.sort(key=lambda u: (u[0], u[1]))          # a thin node at a magnet's entrance goes first
            for entrance_rel2, _, acc, inner in units:
                lp = _pm(acc.get("length"))
                el = self._convert(acc, cavities, used, seq.tag)
                if el is None:
                    continue
                entrance2 = seq_start2 + entrance_rel2
                gap2 = entrance2 - s2
                if gap2 > _TOL2:
                    n_drift += 1
                    elements.append(Drift(name=self._unique(f"{seq.tag}_drift_{n_drift}", used),
                                          length=gap2 * 0.5e-12))
                    s2 = entrance2
                elif gap2 < -_TOL2:
                    rep.lossy("PYORBIT_OVERLAP", f"{el.name!r} starts {-gap2 * 0.5e-12:.3g} m before the previous "
                              "element ends; its position was pushed", element=el.name, kind=el.kind)
                if inner:
                    # thin nodes inside this magnet: PyORBIT applies them between the magnet's parts
                    cursor2 = entrance2
                    n_part = 0
                    for tpos2, tacc in inner:
                        tel = self._convert(tacc, cavities, used, seq.tag)
                        seg2 = seq_start2 + tpos2 - cursor2
                        if seg2 > _TOL2:                        # two thin nodes at one spot: no empty part
                            n_part += 1
                            seg = el.model_copy(update={"name": f"{el.name}_{n_part}", "length": seg2 * 0.5e-12})
                            self._register_split(seg, el)
                            elements.append(seg)
                            cursor2 = seq_start2 + tpos2
                        if tel is not None:
                            rep.equivalent("PYORBIT_THIN_NODE_SPLITS_MAGNET", f"thin node {tel.name!r} sits inside "
                                           f"{el.name!r}: the magnet is split around it",
                                           element=tel.name, kind=tel.kind)
                            elements.append(tel)
                    seg2 = entrance2 + 2 * lp - cursor2
                    if seg2 > _TOL2:
                        n_part += 1
                        seg = el.model_copy(update={"name": f"{el.name}_{n_part}", "length": seg2 * 0.5e-12})
                        self._register_split(seg, el)
                        elements.append(seg)
                else:
                    elements.append(el)
                s2 += 2 * lp
            end2 = seq_start2 + seq_len2
            if end2 - s2 > _TOL2:
                n_drift += 1
                elements.append(Drift(name=self._unique(f"{seq.tag}_drift_{n_drift}", used),
                                      length=(end2 - s2) * 0.5e-12))
                s2 = end2
        elements = self._fold_end_markers(self._merge_correctors(elements))
        generic = root.tag in ("lattix", "sns", "ess")
        lat_name = (seqs[0].tag if len(seqs) == 1 else path.stem) if generic else root.tag
        lat = Lattice.from_sequence(lat_name, elements, ref)
        lat.meta["source_format"] = "pyorbit"
        lat.meta["source_file"] = str(path)
        lat.meta["pyorbit_sequences"] = [sq.tag for sq in seqs]
        self._second_pass(lat)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _unique(name: str, used: set[str]) -> str:
        base, n = name, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        return name

    def _register_split(self, seg: Element, whole: Element) -> None:
        if id(whole) in self._norm:
            self._norm[id(seg)] = self._norm[id(whole)]

    def _reference(self, text, species, kinetic_energy_eV, frequency_Hz, bpm, rep, path) -> ReferenceParticle:
        tag = parse_reference_tag(text)
        sp = None
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        elif tag is not None:
            sp = tag.species
        ke = kinetic_energy_eV if kinetic_energy_eV is not None else (tag.kinetic_energy_eV if tag else None)
        f = frequency_Hz if frequency_Hz is not None else (tag.rf_frequency_Hz if tag else bpm)
        if sp is None or ke is None:
            raise ValueError(f"{path.name}: a PyORBIT linac XML carries no beam; pass species=<name> and "
                             "kinetic_energy_eV=<eV> to read(...) (or keep lattix's reference tag in a comment)")
        if tag is not None and (species is None or kinetic_energy_eV is None):
            rep.equivalent("REFERENCE_FROM_TAG", "reference particle taken from the lattix tag in the XML comment")
        return ReferenceParticle(species=sp, kinetic_energy_eV=float(ke), rf_frequency_Hz=f)

    def _convert(self, acc, cavities: dict, used: set[str], seq_tag: str) -> Element | None:
        etype = acc.get("type", "").upper()
        name = self._unique(acc.get("name") or etype.lower(), used)
        L = float(acc.get("length", "0") or 0.0)
        pnode = acc.find("parameters")
        params = dict(pnode.attrib) if pnode is not None else {}
        prov = Provenance(format="pyorbit", original_name=acc.get("name"), original_type=etype)
        common = {"name": name, "provenance": prov}
        ap = None
        if params.get("aperture") and params.get("aprt_type") == "1":
            ap = ApertureP.circle(_f(params, "aperture") / 2.0)
        if etype == "QUAD":
            el = Quadrupole(length=L, aperture=ap, **common)
            el.multipole = MagneticMultipoleP(Bn={1: _f(params, "field")})
            return el
        if etype == "BEND":
            el = Bend(length=L, **common)
            el.bend = BendP(angle=_f(params, "theta"), e1=_f(params, "ea1"), e2=_f(params, "ea2"))
            if params.get("aperture_x") and params.get("aperture_y"):
                hx, hy = _f(params, "aperture_x") / 2.0, _f(params, "aperture_y") / 2.0
                el.aperture = ApertureP(shape="RECTANGULAR" if params.get("aprt_type") == "3" else "ELLIPTICAL",
                                        x_limits=(-hx, hx), y_limits=(-hy, hy))
            kls = [float(x) for x in str(params.get("kls", "")).split() if x not in ("", "0", "0.0")]
            if kls:
                self._rep.lossy("MULTIPOLE_ORDERS_DROPPED", "BEND kls/poles multipole content (undocumented "
                                "normalization) dropped", element=name, kind="Bend")
            return el
        if etype == "SOLENOID":
            el = Solenoid(length=L, aperture=ap, **common)
            self._norm[id(el)] = ("Bsol", _f(params, "B"), 0.0)
            return el
        if etype == "RFGAP":
            freq = cavities.get(params.get("cavity"))
            if freq is None:
                self._rep.lossy("RF_FREQUENCY_UNKNOWN", f"RFGAP {name!r} names cavity {params.get('cavity')!r}, which "
                                "the sequence does not define", element=name, kind="RFCavity")
            phase = math.radians(_f(params, "phase")) + (math.pi if self._q < 0 else 0.0)
            phase = (phase + math.pi) % (2.0 * math.pi) - math.pi
            rf = RFP(frequency_Hz=freq, voltage_V=_f(params, "E0TL") * 1e9, phase_rad=phase)
            el = RFCavity(length=0.0, aperture=ap, rf=rf, **common)
            ttfs = acc.find("TTFs")
            el.native["pyorbit"] = {"E0L_GeV": _f(params, "E0L"), "mode": params.get("mode", "0"),
                                    "EzFile": params.get("EzFile", ""), "cavity": params.get("cavity", "")}
            if ttfs is not None:
                el.native["pyorbit"]["ttfs_xml"] = ET.tostring(ttfs, encoding="unicode").strip()
                self._rep.equivalent("PYORBIT_TTF_POLYNOMIALS", "the gap's transit-time polynomials are kept as native "
                                     "passthrough; the IR's voltage is E0TL (T at the design velocity)",
                                     element=name, kind="RFCavity")
            return el
        if etype in ("DCH", "DCV"):
            el = Kicker(length=0.0, **common)
            self._norm[id(el)] = ("kick_h" if etype == "DCH" else "kick_v", _f(params, "B"), _f(params, "effLength"))
            return el
        if etype == "MARKER":
            return Marker(**common)
        self._rep.dropped("UNSUPPORTED_PYORBIT_ELEMENT", f"PyORBIT element type {etype!r} has no IR kind; a marker "
                          "or drift keeps the position", element=name, kind=etype)
        el = Drift(length=L, **common) if L > _TOL else Marker(**common)
        el.native["pyorbit"] = {"type": etype, **params}
        return el

    def _merge_correctors(self, elements: list[Element]) -> list[Element]:
        """A DCH followed by the DCV the writer put beside it (``<name>_V``) is one Kicker again."""
        out: list[Element] = []
        for el in elements:
            if el.kind == "Kicker" and out and out[-1].kind == "Kicker" and el.name == out[-1].name + "_V" and \
                    self._norm.get(id(el), ("",))[0] == "kick_v" and self._norm.get(id(out[-1]), ("",))[0] == "kick_h":
                prev = out[-1]
                self._norm[id(prev)] = ("kick_hv", self._norm[id(prev)][1], self._norm[id(prev)][2],
                                        self._norm[id(el)][1], self._norm[id(el)][2])
                continue
            out.append(el)
        return out

    def _fold_end_markers(self, elements: list[Element]) -> list[Element]:
        """``<x>_in`` MARKER, drift, thin ``x`` (gap or corrector), drift, ``<x>_out`` MARKER — the
        writer's picture of a thick cavity or kicker — is one thick element again."""
        out: list[Element] = []
        i = 0
        while i < len(elements):
            el = elements[i]
            folded = None
            if el.kind == "Marker" and el.name.endswith("_in"):
                base = el.name[:-3]
                j = i + 1
                drifts: list[Element] = []
                core = None
                while j < len(elements):
                    e = elements[j]
                    if e.kind == "Drift":
                        drifts.append(e)
                    elif core is None and e.kind in ("RFCavity", "Kicker") and e.name == base and e.length <= _TOL:
                        core = e
                    elif e.kind == "Marker" and e.name == base + "_out":
                        break
                    else:
                        core = None
                        break
                    j += 1
                if core is not None and j < len(elements) and drifts:
                    total = sum(d.length for d in drifts)
                    folded = core.model_copy(update={"length": total})
                    self._register_split(folded, core)
                    self._rep.equivalent("PYORBIT_END_MARKERS_FOLDED", f"the thin {core.kind} {base!r} between "
                                         f"markers {el.name!r} and {base + '_out'!r} is one {total:.6g} m element "
                                         "again (PyORBIT applies it at the centre)", element=base, kind=core.kind,
                                         length=total)
                    out.append(folded)
                    i = j + 1
                    continue
            out.append(el)
            i += 1
        return out

    def _second_pass(self, lat: Lattice) -> None:
        done: set[int] = set()
        for p in propagate(lat):
            el = p.element
            if id(el) in done or id(el) not in self._norm:
                continue
            done.add(id(el))
            brho = (p.ref_in or lat.reference).brho_signed
            spec = self._norm[id(el)]
            if spec[0] == "Bsol":
                el.solenoid = SolenoidP(Bsol_T=float(spec[1]) * abs(brho))
            elif spec[0] == "kick_h":
                el.hkick = -float(spec[1]) * float(spec[2]) / brho
            elif spec[0] == "kick_v":
                el.vkick = float(spec[1]) * float(spec[2]) / brho
            elif spec[0] == "kick_hv":
                el.hkick = -float(spec[1]) * float(spec[2]) / brho
                el.vkick = float(spec[3]) * float(spec[4]) / brho


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
