"""ImpactX reader: the AMReX ParmParse ``inputs`` file → IR.

Grammar (``amrex::ParmParse``): ``key = value…`` one per line, ``#`` starts a comment,
a trailing ``\\`` continues the line, values are whitespace separated.  Keys used here:

* ``beam.kin_energy`` [MeV], ``beam.particle`` (``proton``/``electron``/``positron``/
  ``Hminus``), ``beam.charge`` [C] and the distribution keys, which are kept verbatim in
  ``lattice.meta["impactx_beam"]`` so a re-write reproduces the deck (invariant I-13);
* ``lattice.elements``, ``lattice.nslice``, ``lattice.periods``, ``lattice.reverse``;
* ``<name>.type`` plus that type's parameters (``src/initialization/InitElement.cpp``).

The sequence is **flattened**: ``line`` sub-lattices, ``periods`` and ``reverse`` are
expanded in place (``EQUIVALENT:LINE_FLATTENED``), which is also what the writer emits,
so ``write ∘ read`` is a fixed point.  ``dipedge`` elements that sit directly against an
``sbend``/``cfbend`` are folded into the IR :class:`~lattix.ir.elements.Bend`'s ``e1``/
``e2``/``fint``/``hgap`` (the same clustering the TraceWin reader does for EDGE cards).

Normalized ImpactX strengths (``k``, ``ks``, ``K_normal``…) are turned into lab fields
with the **signed** rigidity of the reference particle at that element's first
occurrence, propagated through ``shortrf`` gains exactly as ImpactX does.

An ImpactX **Python script** is not parsed (executing a deck is not reading it); such a
file is reported as ``DROPPED:IMPACTX_PYTHON_NOT_READ``.  ImpactX's own MAD-X subset
(``KnownElementsList.load_file``) is read on the lattix side by
:mod:`lattix.formats.madx`, which is a superset of it.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from lattix.fidelity import FidelityReport, TranslationError
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Drift,
    Element,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    Patch,
    Quadrupole,
    RFCavity,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.lattice import Lattice, Line, LineItem
from lattix.ir.reference import ReferenceParticle, Species, species
from lattix.ir.units import C_LIGHT

#: ``beam.particle`` → lattix species name
SPECIES_FROM_IMPACTX = {"proton": "proton", "electron": "electron", "positron": "positron",
                        "hminus": "h-"}

#: element types that carry a length
_THICK = frozenset({"drift", "drift_exact", "drift_chromatic", "quad", "quad_exact",
                    "quad_chromatic", "sbend", "sbend_exact", "cfbend", "cfbend_exact",
                    "solenoid", "solenoid_softedge", "quadrupole_softedge", "rfcavity",
                    "constf", "multipole_exact", "plasma_lens_chromatic",
                    "uniform_acc_chromatic", "linear_map", "spin_map"})

#: types whose ``rotation`` is stored in an element field (bend plane, skew angle)
#: instead of in :class:`~lattix.ir.elements.BodyShiftP`, so a round trip cannot
#: double-count it
_ROT_IN_ELEMENT = frozenset({"quad", "quad_exact", "quad_chromatic", "multipole",
                             "sbend", "cfbend", "sbend_exact", "dipedge"})

_RE_ASSIGN = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*(.*)$")
_RE_MATRIX = re.compile(r"^R(\d)(\d)$")


class ImpactxParseError(ValueError):
    pass


def parse_inputs(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """AMReX inputs text → ``{key: [tokens]}`` (insertion ordered) + the key order."""
    out: dict[str, list[str]] = {}
    order: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            if not buf:
                continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        line = buf + line
        buf = ""
        m = _RE_ASSIGN.match(line)
        if not m:
            if line.strip():
                raise ImpactxParseError(f"cannot parse inputs line: {raw!r}")
            continue
        key, rest = m.group(1), m.group(2).strip()
        if key not in out:
            order.append(key)
        out[key] = rest.split()
    if buf.strip():
        m = _RE_ASSIGN.match(buf)
        if m:
            key = m.group(1)
            if key not in out:
                order.append(key)
            out[key] = m.group(2).split()
    return out, order


def _float(tokens: list[str], values: dict[str, list[str]], key: str) -> float | None:
    """One numeric value; resolves ``a.b = c.d`` and ``-c.d`` references (AMReX parser)."""
    if not tokens:
        return None
    tok = tokens[0]
    try:
        return float(tok)
    except ValueError:
        pass
    neg = tok.startswith("-")
    ref = tok[1:] if neg else tok
    if ref in values and ref != key:
        v = _float(values[ref], values, ref)
        if v is not None:
            return -v if neg else v
    return None


def _floats(tokens: list[str]) -> list[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            return out
    return out


def _native_value(tokens: list[str], values: dict[str, list[str]], key: str):
    """Passthrough value for ``native["impactx"]``: number, list of numbers, or text."""
    if len(tokens) > 1:
        nums = _floats(tokens)
        return nums if len(nums) == len(tokens) else " ".join(tokens)
    v = _float(tokens, values, key)
    return " ".join(tokens) if v is None else v


def _bool(tokens: list[str]) -> bool:
    return bool(tokens) and tokens[0].lower() in ("1", "true", "t", "yes")


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "impactx"

    def read(self, path: Path, *, reference: ReferenceParticle | None = None,
             strict: bool = False, name: str | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="impactx", source_file=str(path))
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name.endswith(".py") or re.search(r"^\s*(from|import)\s+impactx", text, re.M):
            entry = rep.dropped(
                "IMPACTX_PYTHON_NOT_READ",
                "an ImpactX Python script is a program, not a deck: lattix writes it "
                "(flavor='python') but does not parse it; write flavor='inputs' (or the MAD-X "
                "subset) to read a lattice back", file=str(path))
            raise TranslationError(entry)
        values, order = parse_inputs(text)
        return self._build(values, order, path, rep, reference=reference, strict=strict,
                           name=name or path.name.split(".")[0])

    # ------------------------------------------------------------------
    def _build(self, values: dict[str, list[str]], order: list[str], path: Path,
               rep: FidelityReport, *, reference: ReferenceParticle | None, strict: bool,
               name: str) -> tuple[Lattice, FidelityReport]:
        ref0, beam_meta = self._reference(values, rep, reference)
        default_nslice = int(_float(values.get("lattice.nslice", []), values, "lattice.nslice") or 1)

        seq, flattened = self._expand(values, rep)
        types = {n: (values.get(f"{n}.type") or [""])[0] for n in dict.fromkeys(seq)}
        missing = [n for n, t in types.items() if not t]
        for n in missing:
            rep.dropped("UNDEFINED_ELEMENT", f"lattice.elements names {n!r} but {n}.type is unset",
                        element=n)
        seq = [n for n in seq if types.get(n)]

        clustered, edges = self._cluster_dipedges(seq, types, values, rep)
        brho_by_name = self._rigidities(clustered, types, values, ref0, rep)

        elements: dict[str, Element] = {}
        items: list[LineItem] = []
        for entry in clustered:
            n = entry[0]
            if n not in elements:
                elements[n] = self._element(n, types[n], values, brho_by_name[n], ref0,
                                            default_nslice, entry[1], entry[2], rep)
            items.append(LineItem(ref=n))
        for e in edges:
            rep.equivalent("DIPEDGE_FOLDED",
                           "a dipedge next to its bend became the bend's e1/e2 + fint/hgap",
                           element=e)

        lat = Lattice(name=name, reference=ref0, elements=elements,
                      lines={name: Line(name=name, items=items)}, use=name)
        lat.meta["source_format"] = "impactx"
        lat.meta["source_file"] = str(path)
        lat.meta["impactx_nslice"] = default_nslice
        if beam_meta:
            lat.meta["impactx_beam"] = beam_meta
        extra = {k: " ".join(values[k]) for k in order
                 if k.split(".")[0] in ("algo", "diag", "amr", "geometry", "amrex")}
        if extra:
            lat.meta["impactx_extra"] = extra
            rep.equivalent("SIM_SETTINGS_KEPT_IN_META",
                           "algo/diag/amr settings are simulation controls, not lattice content; "
                           "kept in meta['impactx_extra']", keys=sorted(extra))
        if flattened:
            rep.equivalent("LINE_FLATTENED",
                           "line sub-lattices / periods / reverse were expanded in place",
                           **flattened)
        rep.raise_if(strict)
        return lat, rep

    # -- beam -----------------------------------------------------------
    @staticmethod
    def _reference(values: dict[str, list[str]], rep: FidelityReport,
                   override: ReferenceParticle | None) -> tuple[ReferenceParticle, dict]:
        beam_meta: dict = {}
        for k, v in values.items():
            if not k.startswith("beam."):
                continue
            short = k[5:]
            if short in ("kin_energy", "particle"):
                continue
            f = _float(v, values, k)
            beam_meta[short] = " ".join(v) if f is None else f
        if override is not None:
            return override, beam_meta
        pname = (values.get("beam.particle") or ["proton"])[0]
        key = SPECIES_FROM_IMPACTX.get(pname.lower())
        if key is None:
            rep.lossy("UNKNOWN_SPECIES", f"beam.particle = {pname!r} is not a lattix species; "
                                         "the reference was taken as a proton", particle=pname)
            sp: Species = species("proton")
        else:
            sp = species(key)
        ke_MeV = _float(values.get("beam.kin_energy", []), values, "beam.kin_energy")
        if ke_MeV is None:
            ke_MeV = 1.0
            rep.lossy("NO_BEAM_ENERGY",
                      "the deck sets no beam.kin_energy; 1 MeV was assumed so normalized "
                      "strengths can be converted — pass reference= to fix it")
        return ReferenceParticle(species=sp, kinetic_energy_eV=float(ke_MeV) * 1e6), beam_meta

    # -- sequence -------------------------------------------------------
    @staticmethod
    def _expand(values: dict[str, list[str]], rep: FidelityReport) -> tuple[list[str], dict]:
        root = values.get("lattice.elements", [])
        info: dict = {}

        def expand(names: list[str], depth: int) -> list[str]:
            if depth > 32:
                raise ImpactxParseError("line nesting deeper than 32 (cycle?)")
            out: list[str] = []
            for n in names:
                if (values.get(f"{n}.type") or [""])[0] == "line":
                    info["sub_lines"] = info.get("sub_lines", 0) + 1
                    sub = list(values.get(f"{n}.elements", []))
                    if _bool(values.get(f"{n}.reverse", [])):
                        sub.reverse()
                    rep_n = int(_float(values.get(f"{n}.repeat", []), values, f"{n}.repeat") or 1)
                    out.extend(expand(sub, depth + 1) * rep_n)
                else:
                    out.append(n)
            return out

        seq = list(root)
        if _bool(values.get("lattice.reverse", [])):
            seq.reverse()
            info["reverse"] = True
        seq = expand(seq, 0)
        periods = int(_float(values.get("lattice.periods", []), values, "lattice.periods") or 1)
        if periods > 1:
            seq = seq * periods
            info["periods"] = periods
        return seq, info

    # -- dipedge clustering ---------------------------------------------
    @staticmethod
    def _cluster_dipedges(seq: list[str], types: dict[str, str], values: dict[str, list[str]],
                          rep: FidelityReport) -> tuple[list[tuple[str, dict, dict]], list[str]]:
        """``[(bend_or_element, entry_edge, exit_edge)]`` with folded dipedges removed."""
        out: list[tuple[str, dict, dict]] = []
        folded: list[str] = []
        i = 0
        while i < len(seq):
            n = seq[i]
            t = types[n]
            if t == "dipedge":
                nxt = seq[i + 1] if i + 1 < len(seq) else None
                loc = (values.get(f"{n}.location") or ["entry"])[0]
                if nxt is not None and types.get(nxt) in ("sbend", "cfbend") and loc == "entry":
                    entry = {"name": n}
                    exit_ = {}
                    j = i + 2
                    if j < len(seq) and types.get(seq[j]) == "dipedge" \
                            and (values.get(f"{seq[j]}.location") or ["entry"])[0] == "exit":
                        exit_ = {"name": seq[j]}
                        folded.append(seq[j])
                        j += 1
                    folded.append(n)
                    out.append((nxt, entry, exit_))
                    i = j
                    continue
                rep.lossy("DIPEDGE_ORPHAN",
                          "a dipedge with no adjacent sbend/cfbend has no IR element; it became "
                          "a marker and its parameters are kept in native['impactx']", element=n)
                out.append((n, {}, {}))
                i += 1
                continue
            if t in ("sbend", "cfbend"):
                exit_ = {}
                j = i + 1
                if j < len(seq) and types.get(seq[j]) == "dipedge" \
                        and (values.get(f"{seq[j]}.location") or ["entry"])[0] == "exit":
                    exit_ = {"name": seq[j]}
                    folded.append(seq[j])
                    j += 1
                out.append((n, {}, exit_))
                i = j
                continue
            out.append((n, {}, {}))
            i += 1
        return out, folded

    # -- rigidity walk ---------------------------------------------------
    @staticmethod
    def _rigidities(clustered: list[tuple[str, dict, dict]], types: dict[str, str],
                    values: dict[str, list[str]], ref0: ReferenceParticle,
                    rep: FidelityReport) -> dict[str, float]:
        """Signed rigidity at each definition's FIRST occurrence (ImpactX follows p0)."""
        out: dict[str, float] = {}
        ref = ref0
        mass = ref0.species.mass_eV
        for n, _e, _x in clustered:
            brho = ref.brho_signed
            if n in out and abs(out[n] - brho) > 1e-12 * max(1.0, abs(brho)):
                rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                               "one definition is used at two reference energies; the first "
                               "occurrence's rigidity converts its normalized strengths",
                               element=n, brho_first=out[n], brho_here=brho)
            out.setdefault(n, brho)
            if types[n] == "shortrf":
                v = _float(values.get(f"{n}.V", []), values, f"{n}.V") or 0.0
                ph = _float(values.get(f"{n}.phase", []), values, f"{n}.phase")
                ph = -90.0 if ph is None else ph
                ref = ref.advanced(dE_eV=v * mass * math.cos(math.radians(ph)))
            elif types[n] == "rfcavity":
                rep.lossy("RFCAVITY_GAIN_UNKNOWN",
                          "a Fourier rfcavity's reference gain needs the field integral; the "
                          "reference energy was left unchanged for the rigidity walk", element=n)
        return out

    # -- one element ------------------------------------------------------
    def _element(self, n: str, t: str, values: dict[str, list[str]], brho: float,
                 ref0: ReferenceParticle, default_nslice: int, entry: dict, exit_: dict,
                 rep: FidelityReport) -> Element:
        def num(key: str, default: float | None = None) -> float | None:
            v = _float(values.get(f"{n}.{key}", []), values, f"{n}.{key}")
            return default if v is None else v

        def txt(key: str, default: str) -> str:
            v = values.get(f"{n}.{key}")
            return default if not v else v[0]

        ds = float(num("ds", 0.0) or 0.0)
        native = {k[len(n) + 1:]: _native_value(v, values, k)
                  for k, v in values.items() if k.startswith(f"{n}.")}
        nslice = int(num("nslice", default_nslice) or default_nslice)
        common: dict = {"name": n, "native": {"impactx": native},
                        "tracking": {"nslice": nslice} if t in _THICK else {}}
        dx, dy = float(num("dx", 0.0) or 0.0), float(num("dy", 0.0) or 0.0)
        rot = math.radians(float(num("rotation", 0.0) or 0.0))
        shift_tilt = 0.0 if t in _ROT_IN_ELEMENT else rot
        if dx or dy or shift_tilt:
            common["shift"] = BodyShiftP(x_offset=dx, y_offset=dy, tilt=shift_tilt)
        apx, apy = num("aperture_x"), num("aperture_y")
        if t != "aperture" and apx is not None and apy is not None and (apx or apy):
            common["aperture"] = ApertureP(shape="ELLIPTICAL", x_limits=(-apx, apx),
                                           y_limits=(-apy, apy))

        mass = ref0.species.mass_eV
        if t in ("drift", "drift_exact", "drift_chromatic"):
            if t != "drift":
                rep.equivalent("EXACT_DRIFT_MODEL",
                               f"{t} was read as a plain Drift; the nonlinear/chromatic model is "
                               "kept in native['impactx']", element=n, kind="Drift")
            return Drift(length=ds, **common)

        if t in ("quad", "quad_exact", "quad_chromatic"):
            k = float(num("k", 0.0) or 0.0)
            unit = int(num("unit", num("units", 0.0)) or 0)
            g = k if unit == 1 else k * brho
            if t != "quad":
                rep.equivalent("EXACT_QUAD_MODEL", f"{t} was read as a plain Quadrupole",
                               element=n, kind="Quadrupole")
            mp = MagneticMultipoleP(Bn={1: g})
            if rot:
                mp.tilt[1] = rot
            return Quadrupole(length=ds, multipole=mp, **common)

        if t in ("sbend", "cfbend", "sbend_exact"):
            return self._bend(n, t, values, brho, ds, rot, entry, exit_, common, rep, num, txt)

        if t == "dipedge":
            return Marker(length=0.0, **common)

        if t in ("solenoid", "solenoid_softedge"):
            if t == "solenoid":
                ks = float(num("ks", 0.0) or 0.0)
                b = ks * brho
            else:
                scale = float(num("bscale", 0.0) or 0.0)
                b = scale if int(num("unit", 0.0) or 0) == 1 else scale * brho
                rep.equivalent("SOFT_SOLENOID_HARD_EDGE",
                               "solenoid_softedge was read as a hard-edge Solenoid with its peak "
                               "field; the Fourier profile is in native['impactx']",
                               element=n, kind="Solenoid")
            return Solenoid(length=ds, solenoid=SolenoidP(Bsol_T=b), **common)

        if t == "shortrf":
            v = float(num("V", 0.0) or 0.0)
            ph = num("phase")
            rf = RFP(frequency_Hz=float(num("freq", 0.0) or 0.0), voltage_V=v * mass,
                     phase_rad=math.radians(-90.0 if ph is None else ph))
            return RFCavity(length=0.0, rf=rf, **common)

        if t == "rfcavity":
            escale = float(num("escale", 0.0) or 0.0)
            rf = RFP(frequency_Hz=float(num("freq", 0.0) or 0.0),
                     gradient_V_per_m=escale * mass, L_active_m=ds,
                     phase_rad=math.radians(float(num("phase", 0.0) or 0.0)),
                     phase_is_sync=False)
            rep.equivalent("RFCAVITY_FOURIER_PROFILE",
                           "the on-axis Fourier field of an ImpactX rfcavity is kept in "
                           "native['impactx']; gradient_V_per_m is escale·mc^2 (the peak field), "
                           "so the reference gain is not V·cos(phase)",
                           element=n, kind="RFCavity", escale=escale)
            return RFCavity(length=ds, rf=rf, **common)

        if t == "buncher":
            v = float(num("V", 0.0) or 0.0) * C_LIGHT * abs(brho)
            k = float(num("k", 0.0) or 0.0)
            rf = RFP(frequency_Hz=k * C_LIGHT / (2 * math.pi) if k else None,
                     voltage_V=v, phase_rad=-math.pi / 2)
            rep.equivalent("BUNCHER_AS_CAVITY",
                           "an ImpactX buncher is a thin cavity at the zero crossing "
                           "(phase = -90 deg)", element=n, kind="RFCavity")
            return RFCavity(length=0.0, rf=rf, **common)

        if t == "kicker":
            xk = float(num("xkick", 0.0) or 0.0)
            yk = float(num("ykick", 0.0) or 0.0)
            if txt("units", txt("unit", "dimensionless")) == "T-m":
                xk, yk = xk / brho, yk / brho
            return Kicker(length=0.0, hkick=xk, vkick=yk, **common)

        if t == "multipole":
            m = int(num("multipole", 1) or 1)
            mp = MagneticMultipoleP(BnL={m - 1: float(num("k_normal", 0.0) or 0.0) * brho},
                                    BsL={m - 1: float(num("k_skew", 0.0) or 0.0) * brho})
            if rot:
                mp.tilt[m - 1] = rot
            return Multipole(length=0.0, multipole=mp, **common)

        if t == "thin_dipole":
            theta = math.radians(float(num("theta", 0.0) or 0.0))
            rep.lossy("THIN_DIPOLE_AS_MULTIPOLE",
                      "an ImpactX thin_dipole also applies a curvature-dependent weak-focusing "
                      "term that a thin IR Multipole does not carry", element=n, kind="Multipole",
                      rc=num("rc"))
            return Multipole(length=0.0, multipole=MagneticMultipoleP(BnL={0: theta * brho}),
                             **common)

        if t == "aperture":
            apx = float(num("aperture_x", num("xmax", 0.0)) or 0.0)
            apy = float(num("aperture_y", num("ymax", 0.0)) or 0.0)
            shape = "RECTANGULAR" if txt("shape", "rectangular") == "rectangular" else "ELLIPTICAL"
            common.pop("shift", None)
            return Collimator(length=0.0, aperture=ApertureP(
                shape=shape, x_limits=(dx - apx, dx + apx), y_limits=(dy - apy, dy + apy)),
                **common)

        if t == "beam_monitor":
            return Instrument(length=0.0, family="MONITOR", **common)

        if t == "linear_map":
            rows = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
            for key, v in values.items():
                if not key.startswith(f"{n}.R"):
                    continue
                m = _RE_MATRIX.match(key[len(n) + 1:])
                if m:
                    val = _float(v, values, key) or 0.0
                    i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
                    rows[i][j] = -val if (i >= 4) != (j >= 4) else val
            rep.equivalent("TAYLOR_TIME_SIGN",
                           "the map was conjugated with diag(1,1,1,1,-1,-1): ImpactX's t is "
                           "late-positive and its pt is the negative energy deviation",
                           element=n, kind="Taylor")
            return Taylor(length=ds, matrix=rows, **common)

        if t == "plane_xyrotation":
            return Patch(length=0.0, tilt=math.radians(float(num("angle", 0.0) or 0.0)), **common)

        rep.lossy("UNSUPPORTED_TYPE",
                  f"ImpactX element type {t!r} has no IR kind; written as a "
                  f"{'drift' if ds else 'marker'} and kept in native['impactx']",
                  element=n, kind="Drift" if ds else "Marker", impactx_type=t)
        return Drift(length=ds, **common) if ds else Marker(length=0.0, **common)

    # -- bend -------------------------------------------------------------
    @staticmethod
    def _bend(n: str, t: str, values: dict[str, list[str]], brho: float, ds: float, rot: float,
              entry: dict, exit_: dict, common: dict, rep: FidelityReport, num, txt) -> Bend:
        if t == "sbend_exact":
            angle = math.radians(float(num("phi", 0.0) or 0.0))
            b_field = float(num("B", 0.0) or 0.0)
            if b_field:
                rep.equivalent("SBEND_EXACT_B",
                               "sbend_exact with B != 0 defines rc = rigidity/B; the IR keeps the "
                               "angle from phi", element=n, kind="Bend", B=b_field)
            rep.equivalent("EXACT_SBEND_MODEL", "sbend_exact was read as a sector Bend",
                           element=n, kind="Bend")
        else:
            rc = float(num("rc", 0.0) or 0.0)
            angle = ds / rc if rc else 0.0
        bend = BendP(angle=angle, tilt_ref=rot)
        mp = MagneticMultipoleP()
        if t == "cfbend":
            mp.Bn[1] = float(num("k", 0.0) or 0.0) * brho

        def edge(spec: dict, which: str) -> None:
            e = spec.get("name")
            if not e:
                return
            psi = _float(values.get(f"{e}.psi", []), values, f"{e}.psi") or 0.0
            g = _float(values.get(f"{e}.g", []), values, f"{e}.g") or 0.0
            k2 = _float(values.get(f"{e}.K2", []), values, f"{e}.K2")
            k2 = 1.0 if k2 is None else k2
            if which == "e1":
                bend.e1, bend.edge_int1 = psi, k2
            else:
                bend.e2, bend.edge_int2 = psi, k2
            bend.hgap = max(bend.hgap, g / 2.0)
            for extra in ("K0", "K1", "K3", "K4", "K5", "K6", "R"):
                v = _float(values.get(f"{e}.{extra}", []), values, f"{e}.{extra}")
                if v is not None:
                    rep.lossy("DIPEDGE_FIELD_INTEGRALS",
                              f"{e}.{extra} (Hwang-Lee fringe integral) has no IR field",
                              element=n, kind="Bend", integral=extra, value=v)
            if (values.get(f"{e}.model") or ["linear"])[0] != "linear":
                rep.equivalent("DIPEDGE_NONLINEAR_MODEL",
                               "the nonlinear dipedge model was read as a linear edge",
                               element=n, kind="Bend")

        edge(entry, "e1")
        edge(exit_, "e2")
        if bend.edge_int2 is None and exit_:
            bend.edge_int2 = bend.edge_int1
        return Bend(length=ds, bend=bend, multipole=mp, **common)


def read(path: str | Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
