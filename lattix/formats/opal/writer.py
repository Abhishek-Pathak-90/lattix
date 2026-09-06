"""IR → OPAL-T input deck (OPAL-X 078ff1e, ``src/Elements/*.cpp``, ``src/Classic/AbsBeamline/*.cpp``).

Conventions taken from the source (no OPAL engine runs here, see ``docs/formats/opal.md``):

* every element is placed by ``ELEMEDGE`` (its entrance path length [m]); drifts carry no field and are
  written only so that the ``LINE`` and a reader keep the structure;
* normalized strengths ``K1``, ``K2``, ``K3``, ``KN``/``KS`` and ``KS`` are multiplied by the **BEAM's**
  ``P0/c`` (``OpalData::getP0()``, the initial momentum, unsigned): lattix divides the lab fields by the
  start rigidity ``|Bρ|`` so that the written field is the IR's at every energy and for either charge
  sign — a negative species therefore gets the opposite ``K1`` sign from MAD-X's;
* ``KICKER HKICK/VKICK`` are deflections [rad] of the actual particle at ``DESIGNENERGY`` [MeV kinetic]
  (``Corrector::goOnline``: ``kickField = q·p/(c·L)·(vkick, −hkick)``), ``SBEND ANGLE`` the geometric bend
  angle (``compute3DLattice`` rotates the axis; a positive angle bends towards −x like MAD-X's) with a
  required ``DESIGNENERGY``;
* ``RFCAVITY VOLT`` is the peak on-axis field [MV/m] scaling a ``1DDynamic`` map normalised to 1,
  ``LAG`` [rad] is added to the crest phase ``OPTION, AUTOPHASE`` finds (``CavityAutophaser``:
  ``newPhase = LAG + optimizedPhase``; ``E ∝ cos(ωt + φ)``, so a late particle gains more at a negative
  ``LAG`` — the IR's synchronous phase) and ``DESIGNENERGY`` [MeV] makes OPAL scale the amplitude so that
  the crest gain reaches it (``E_kin,in + V_eff``); a cavity's and a solenoid's length is the extent of
  its map (``initialise`` → ``setElementLength(zEnd − zBegin)``), so thin gaps and hard-edge solenoids
  get generated maps of a surrogate / padded length.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.impactt.rfprofile import transit_factor
from lattix.formats.opal.maps import (
    DYNAMIC,
    STATIC,
    cell_profile,
    dynamic_map_text,
    plateau,
    raised_bump,
    static_map_text,
)
from lattix.ir.elements import Element
from lattix.ir.fieldmap import load_field_map, replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference_tag import format_reference_tag
from lattix.ir.rf import thin_gap_surrogate_length
from lattix.ir.walk import propagate

C_LIGHT = 299_792_458.0
_THIN = 1e-12

#: OPAL statement keywords, element types and predefined constants (identifiers are case-insensitive)
RESERVED = {
    "DRIFT", "QUADRUPOLE", "SEXTUPOLE", "OCTUPOLE", "MULTIPOLE", "SBEND", "RBEND", "RBEND3D", "SOLENOID",
    "RFCAVITY", "TRAVELINGWAVE", "VARIABLE_RF_CAVITY", "KICKER", "HKICKER", "VKICKER", "CORRECTOR",
    "RCOLLIMATOR", "ECOLLIMATOR", "CCOLLIMATOR", "FLEXIBLECOLLIMATOR", "MONITOR", "MARKER", "DEGRADER",
    "SEPTUM", "STRIPPER", "PROBE", "SOURCE", "CYCLOTRON", "RINGDEFINITION", "UNDULATOR", "MULTIPOLET",
    "LINE", "SEQUENCE", "BEAM", "TRACK", "RUN", "ENDTRACK", "OPTION", "TITLE", "REAL", "BOOL", "STRING",
    "VECTOR", "VALUE", "FIELDSOLVER", "DISTRIBUTION", "SELECT", "START", "STOP", "QUIT", "CALL", "SYSTEM",
    "WAKE", "GEOMETRY", "PARTICLEMATTERINTERACTION", "OUTPUTPLANE", "TRUE", "FALSE", "PI", "TWOPI",
    "RADDEG", "DEGRAD", "E", "EMASS", "PMASS", "HMMASS", "UMASS", "CMASS", "MMASS", "DMASS", "XEMASS",
    "CLIGHT", "P0", "SEED", "MATRIX", "SLICE", "MICADO", "CORRECT", "EIGEN", "ENVELOPE", "TWISS",
}
_PARTICLE = {"proton": "PROTON", "electron": "ELECTRON", "positron": "POSITRON", "antiproton": "ANTIPROTON",
             "deuteron": "DEUTERON", "h-": "HMINUS"}


@dataclass(frozen=True)
class Rule:
    message: str
    cls: str
    code: str


def fmt(v: float) -> str:
    s = f"{float(v):.15g}"
    return s


def _tag(kv: dict) -> str:
    parts = []
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, bool):
            s = "1" if v else "0"
        elif isinstance(v, int | float):
            s = fmt(v) if isinstance(v, float) else str(v)
        else:
            s = str(v)
        if " " in s or "=" in s or not s or '"' in s:
            s = '"' + s.replace('"', "'") + '"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


def _ident(name: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_") or "e"
    if base[0].isdigit() or base.upper() in RESERVED:
        base = "e_" + base
    cand, n = base, 1
    while cand.upper() in used:
        n += 1
        cand = f"{base}_{n}"
    used.add(cand.upper())
    return cand


@dataclass
class _Stmt:
    name: str                          # the OPAL identifier ("" for a pure comment)
    etype: str                         # DRIFT, QUADRUPOLE, … ("" for a comment line)
    attrs: list[tuple[str, str]] = field(default_factory=list)
    tag: dict = field(default_factory=dict)
    in_line: bool = True

    def render(self) -> str:
        if not self.etype:
            return "// lattix: " + _tag(self.tag)
        body = ", ".join(f"{k}={v}" for k, v in self.attrs)
        line = f"{self.name}: {self.etype}" + (", " + body if body else "") + ";"
        return line + ("  // lattix: " + _tag(self.tag) if self.tag else "")


class Writer:
    format = "opal"

    RULES: dict[str, Rule] = {
        "Drift": Rule("DRIFT L ELEMEDGE (OPAL-T places elements by ELEMEDGE; kept for the LINE)", "EXACT", "OK"),
        "Quadrupole": Rule("QUADRUPOLE L K1 = Bn1/(P0/c) K1S PSI ELEMEDGE", "EXACT", "OK"),
        "Sextupole": Rule("SEXTUPOLE L K2 = Bn2/(P0/c) K2S PSI", "EXACT", "OK"),
        "Octupole": Rule("OCTUPOLE L K3 = Bn3/(P0/c) K3S PSI", "EXACT", "OK"),
        "Multipole": Rule("KICKER for the dipole terms and a short MULTIPOLE KN/KS for the others (OPAL-T "
                          "integrates fields over a length)", "EQUIVALENT", "MULTIPOLE_AS_SHORT"),
        "Bend": Rule("SBEND L ANGLE E1 E2 HGAP FINT K1 K2 DESIGNENERGY[MeV] GAP PSI with OPAL's default "
                     "1DPROFILE1 fringe profile", "EQUIVALENT", "OPAL_BEND_DEFAULT_PROFILE"),
        "Solenoid": Rule("SOLENOID KS = Bsol/(P0/c) with a generated 1DMagnetoStatic flat-top map whose ramps keep "
                         "∫B·dz = Bsol·L (the element spans the map)", "EQUIVALENT", "OPAL_SOLENOID_MAP"),
        "RFCavity": Rule("RFCAVITY VOLT[MV/m] FREQ[MHz] LAG[rad] DESIGNENERGY[MeV] with a generated 1DDynamic "
                         "profile (a thin gap over a short surrogate length)", "EQUIVALENT", "OPAL_CAVITY_MAP"),
        "FieldMap": Rule("the map's on-axis E_z or B_z as a 1DDynamic / 1DMagnetoStatic file (else the hard-edge "
                         "ladder)", "EQUIVALENT", "FM_AS_OPAL_MAP"),
        "NCells": Rule("RFCAVITY with a cell-train 1DDynamic profile of the train's voltage", "EQUIVALENT",
                       "NCELLS_AS_CAVITY"),
        "RFQCell": Rule("DRIFT (OPAL-T has no RFQ cell)", "LOSSY", "RFQCELL_TO_DRIFT"),
        "Kicker": Rule("KICKER HKICK VKICK[rad] DESIGNENERGY[MeV] (a thin one over a short length)", "EXACT",
                       "OK"),
        "Collimator": Rule("RCOLLIMATOR / ECOLLIMATOR XSIZE YSIZE (half apertures)", "EXACT", "OK"),
        "Marker": Rule("MARKER ELEMEDGE", "EXACT", "OK"),
        "Instrument": Rule("MONITOR (the family in the tag)", "EQUIVALENT", "INSTRUMENT_AS_MONITOR"),
        "Foil": Rule("MARKER (no particle-matter interaction written)", "LOSSY", "FOIL_TO_MARKER"),
        "Taylor": Rule("DRIFT of the length (OPAL-T tracks fields, not maps)", "LOSSY", "TAYLOR_DROPPED"),
        "Patch": Rule("MARKER carrying the offsets in its tag (OPAL-T has no coordinate patch)", "LOSSY",
                      "PATCH_DROPPED"),
        "ReferenceChange": Rule("MARKER with the change in its tag (OPAL-T tracks the energy itself)",
                                "EQUIVALENT", "REFCHANGE_AS_TAG"),
        "Freq": Rule("MARKER with the frequency in its tag (every cavity carries its own FREQ)", "EXACT", "OK"),
        "Directive": Rule("a comment (OPAL has no such statement)", "DROPPED", "FOREIGN_DIRECTIVE"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED"),
    }

    # ------------------------------------------------------------------ entry points
    def write(self, lattice: Lattice, path, *, strict: bool = False, **options) -> FidelityReport:
        path = Path(path)
        rep = FidelityReport(target_format="opal", target_file=str(path))
        stem = path.name.split(".")[0] or path.stem
        text, files = self.render(lattice, report=rep, stem=stem, **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        for name, content in files.items():
            (path.parent / name).write_text(content)
        rep.raise_if(strict)
        return rep

    def render(self, lattice: Lattice, *, report: FidelityReport | None = None, stem: str = "lattice",
               thin_length_m: float = 1e-3, solenoid_ramp_m: float = 0.02, map_points: int = 201,
               n_particles: int = 1000, dt_s: float = 1e-12, install_apertures: bool = True,
               autophase: int = 4) -> tuple[str, dict[str, str]]:
        self.rep = report if report is not None else FidelityReport(target_format="opal")
        self.lat = lattice
        self.stem = stem
        self.thin = float(thin_length_m)
        self.sol_ramp = float(solenoid_ramp_m)
        self.map_points = int(map_points)
        self.install_apertures = install_apertures
        self.used: set[str] = set()
        self.files: dict[str, str] = {}
        ref = lattice.reference
        self.brho0 = ref.brho_abs                    # OPAL scales every normalized strength by the BEAM's P0/c
        placed = propagate(lattice)
        stmts: list[_Stmt] = []
        for p in placed:
            stmts += self._element(p)
        line = _ident(lattice.use or lattice.name or "lattix", self.used)
        beam, fs, dist = (_ident(n, self.used) for n in ("lattix_beam", "lattix_fs", "lattix_dist"))
        total = placed[-1].s_out if placed else 0.0
        beta_min = min((p.ref_in.beta for p in placed if p.ref_in is not None), default=ref.beta) or ref.beta
        maxsteps = int(1.5 * total / (beta_min * C_LIGHT * dt_s)) + 1000
        title = (lattice.name or "lattix").replace('"', "'")
        head = [f"// {title} - written by lattix {__version__} from {lattice.meta.get('source_format', 'IR')}",
                "// OPAL-T deck: elements placed by ELEMEDGE, strengths normalised by the BEAM's P0/c, cavity LAG "
                "relative to the autophased crest",
                format_reference_tag(ref, "//"),
                "// lattix: lattice " + _tag({"name": lattice.name, "use": lattice.use or lattice.name}),
                f'Title, string="{title}";',
                f"OPTION, AUTOPHASE={int(autophase)}, ECHO=FALSE, INFO=FALSE;", ""]
        body = [s.render() for s in stmts]
        members = [s.name for s in stmts if s.in_line and s.etype]
        line_stmt = self._wrap(f"{line}: LINE = (", members, ");")
        sp = ref.species
        particle = _PARTICLE.get(sp.name.lower())
        beam_attrs = ([f"PARTICLE={particle}"] if particle else []) + [
            f"MASS={fmt(sp.mass_eV * 1e-9)}", f"CHARGE={fmt(float(sp.charge))}", f"PC={fmt(ref.pc_eV * 1e-9)}"]
        if ref.rf_frequency_Hz:
            beam_attrs.append(f"BFREQ={fmt(ref.rf_frequency_Hz * 1e-6)}")
        beam_attrs.append(f"NPART={int(n_particles)}")
        tail = ["", line_stmt, "",
                f"{beam}: BEAM, " + ", ".join(beam_attrs) + ";",
                f"{fs}: FIELDSOLVER, FSTYPE=NONE, MX=8, MY=8, MT=8, PARFFTX=FALSE, PARFFTY=FALSE, PARFFTT=TRUE, "
                "BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN, BBOXINCR=1;",
                "// the distribution is a placeholder (lattix carries no beam): edit before a real run",
                f"{dist}: DISTRIBUTION, TYPE=GAUSS, SIGMAX=1.0e-3, SIGMAPX=1.0e-4, SIGMAY=1.0e-3, SIGMAPY=1.0e-4, "
                "SIGMAZ=1.0e-3, SIGMAPZ=1.0e-4;",
                f"TRACK, LINE={line}, BEAM={beam}, MAXSTEPS={maxsteps}, DT={fmt(dt_s)}, ZSTOP={fmt(total)};",
                f'RUN, METHOD="PARALLEL-T", BEAM={beam}, FIELDSOLVER={fs}, DISTRIBUTION={dist};',
                "ENDTRACK;", "QUIT;"]
        return "\n".join(head + body + tail) + "\n", dict(self.files)

    @staticmethod
    def _wrap(head: str, items: list[str], tail: str, width: int = 100) -> str:
        if not items:
            return head + tail
        lines, cur = [], head
        for i, it in enumerate(items):
            piece = it + ("," if i < len(items) - 1 else "")
            if len(cur) + len(piece) + 1 > width and cur.strip() != head.strip():
                lines.append(cur.rstrip())
                cur = "    " + piece
            else:
                cur = cur + (" " if cur and not cur.endswith("(") else "") + piece
        lines.append(cur + tail)
        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers
    def _record(self, el: Element, rule: Rule, **details) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind, code=rule.code, message=rule.message)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind, **details)

    def _stmt(self, el: Element, etype: str, attrs: dict, *, s: float, kind: str | None = None,
              tag: dict | None = None, suffix: str = "", in_line: bool = True) -> _Stmt:
        ident = _ident(el.name + suffix, self.used)
        full: dict = {"kind": kind or el.kind}
        if ident != el.name + suffix or suffix:
            full["name"] = el.name
        if el.meta.get("opal_from") and "from" not in full and full["kind"] != el.meta["opal_from"]:
            full["from"] = str(el.meta["opal_from"])        # a re-read replacement keeps its origin
        if tag:
            full.update(tag)
        rendered = []
        for k, v in attrs.items():
            if v is None:
                continue
            rendered.append((k, v if isinstance(v, str) else fmt(v)))
        rendered.append(("ELEMEDGE", fmt(s)))
        return _Stmt(ident, etype, rendered, full, in_line)

    @staticmethod
    def _ke_MeV(p: Placed, lat: Lattice) -> float:
        return (p.ref_in or lat.reference).kinetic_energy_eV * 1e-6

    def _shift(self, el: Element, st: _Stmt, *, bend: bool = False) -> None:
        sh = el.shift
        if sh is None or not st.etype:
            return
        for key, val in (("DX", sh.x_offset), ("DY", sh.y_offset), ("DZ", sh.z_offset),
                         ("DTHETA", sh.y_rot), ("DPHI", sh.x_rot), ("DPSI", 0.0 if bend else sh.tilt)):
            if val:
                st.attrs.insert(len(st.attrs) - 1, (key, fmt(val)))

    def _aperture(self, el: Element, st: _Stmt) -> None:
        ap = el.aperture
        if ap is None or not self.install_apertures or el.kind == "Collimator" or not st.etype:
            return
        hx, hy = ap.half_x, ap.half_y
        if hx is None and hy is None:
            return
        hx = hx if hx is not None else hy
        hy = hy if hy is not None else hx
        if ap.shape == "RECTANGULAR":
            text = f'"RECTANGLE({fmt(2.0 * hx)},{fmt(2.0 * hy)})"'
        elif abs(hx - hy) <= 1e-15:
            text = f'"CIRCLE({fmt(2.0 * hx)})"'
        else:
            text = f'"ELLIPSE({fmt(2.0 * hx)},{fmt(2.0 * hy)})"'
        st.attrs.insert(len(st.attrs) - 1, ("APERTURE", text))
        if (ap.x_limits and abs(ap.x_limits[0] + ap.x_limits[1]) > 1e-15) or \
                (ap.y_limits and abs(ap.y_limits[0] + ap.y_limits[1]) > 1e-15):
            self.rep.lossy("APERTURE_OFFSET_DROPPED", "OPAL's APERTURE is centred on the element; the aperture's "
                           "offset is dropped", element=el.name, kind=el.kind)
        self.rep.equivalent("APERTURE_AS_ATTRIBUTE", "the element's aperture is OPAL's APERTURE attribute "
                            "(full width/height, applied where the element is)", element=el.name, kind=el.kind)

    def _element(self, p: Placed) -> list[_Stmt]:
        el = p.element
        rule = self.RULES.get(el.kind)
        if rule is None:                                          # pragma: no cover - RULES is total
            raise KeyError(f"the OPAL writer has no rule for kind {el.kind!r}")
        if el.kind == "Superposition":
            return self._w_superposition(el, p, rule)
        if el.kind == "FieldMap":
            return self._w_fieldmap(el, p, rule)
        fn = getattr(self, f"_w_{el.kind.lower()}")
        out = fn(el, p, rule)
        if out and out[0].etype:
            if el.kind not in ("Collimator", "Marker", "Patch", "ReferenceChange", "Freq", "Directive"):
                self._aperture(el, out[0])
            if el.kind not in ("Patch", "ReferenceChange", "Freq", "Directive"):
                self._shift(el, out[0], bend=(el.kind == "Bend"))
        return out

    def _map_file(self, ident: str, suffix: str, text: str) -> str:
        name = f"{self.stem}_{ident}{suffix}"
        self.files[name] = text
        return name

    # ------------------------------------------------------------------ magnets
    def _w_drift(self, el, p, rule):
        self._record(el, rule)
        return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in)]

    def _multipole_attrs(self, el, order: int, key: str, skey: str) -> dict:
        mp = el.multipole
        d: dict = {"L": el.length, key: float(mp.Bn.get(order, 0.0)) / self.brho0}
        if mp.Bs.get(order):
            d[skey] = float(mp.Bs[order]) / self.brho0
        if mp.tilt.get(order):
            d["PSI"] = float(mp.tilt[order])
        others = sorted(n for n, v in {**mp.Bn, **mp.Bs}.items() if v and n != order)
        if others:
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", f"OPAL's element carries order {order} only; dropped "
                           f"orders {others}", element=el.name, kind=el.kind)
        return d

    def _w_quadrupole(self, el, p, rule):
        d = self._multipole_attrs(el, 1, "K1", "K1S")
        self._record(el, rule)
        return [self._stmt(el, "QUADRUPOLE", d, s=p.s_in)]

    def _w_sextupole(self, el, p, rule):
        d = self._multipole_attrs(el, 2, "K2", "K2S")
        self._record(el, rule)
        return [self._stmt(el, "SEXTUPOLE", d, s=p.s_in)]

    def _w_octupole(self, el, p, rule):
        d = self._multipole_attrs(el, 3, "K3", "K3S")
        self._record(el, rule)
        return [self._stmt(el, "OCTUPOLE", d, s=p.s_in)]

    def _w_multipole(self, el, p, rule):
        mp = el.multipole
        ref = p.ref_in or self.lat.reference
        thin = el.length <= _THIN
        ell = self.thin if thin else float(el.length)
        s = p.s_in - 0.5 * ell if thin else p.s_in
        kn = {n: float(v) for n, v in mp.BnL.items() if v}
        ks = {n: float(v) for n, v in mp.BsL.items() if v}
        hk = -kn.pop(0, 0.0) / ref.brho_signed if ref.brho_signed else 0.0
        vk = ks.pop(0, 0.0) / ref.brho_signed if ref.brho_signed else 0.0
        tag = {"L": 0.0} if thin else {}
        out: list[_Stmt] = []
        if hk or vk:
            out.append(self._stmt(el, "KICKER", {"L": ell, "HKICK": hk, "VKICK": vk,
                                                 "DESIGNENERGY": self._ke_MeV(p, self.lat)},
                                  s=s, tag={**tag, "role": "dipole"}, suffix="_k" if (kn or ks) else ""))
        if kn or ks:
            top = max([*kn, *ks])
            knv = [kn.get(n, 0.0) / (ell * self.brho0) for n in range(top + 1)]
            ksv = [ks.get(n, 0.0) / (ell * self.brho0) for n in range(top + 1)]
            d: dict = {"L": ell, "KN": "{" + ",".join(fmt(v) for v in knv) + "}"}
            if any(ksv):
                d["KS"] = "{" + ",".join(fmt(v) for v in ksv) + "}"
            if mp.tilt:
                d["PSI"] = float(next(iter(mp.tilt.values())))
            out.append(self._stmt(el, "MULTIPOLE", d, s=s, tag={**tag, "role": "higher"}))
        if not out:
            out.append(self._stmt(el, "MARKER", {}, s=p.s_in, tag={**tag, "role": "empty"}))
        self._record(el, rule, length_m=ell, thin=thin)
        return out

    def _w_bend(self, el, p, rule):
        b = el.bend
        mp = el.multipole
        if abs(b.angle) < 1e-15 and not mp.Bn.get(1):
            self.rep.lossy("ZERO_ANGLE_BEND_AS_DRIFT", "a bend without an angle is written as a drift",
                           element=el.name, kind=el.kind)
            return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in)]
        d: dict = {"L": el.length, "ANGLE": float(b.angle)}
        if b.e1:
            d["E1"] = float(b.e1)
        if b.e2:
            d["E2"] = float(b.e2)
        if b.hgap:
            d["HGAP"] = float(b.hgap)
            d["GAP"] = 2.0 * float(b.hgap)
        if b.edge_int1:
            d["FINT"] = float(b.edge_int1)
        if mp.Bn.get(1):
            d["K1"] = float(mp.Bn[1]) / self.brho0
        if mp.Bn.get(2):
            d["K2"] = float(mp.Bn[2]) / self.brho0
        d["DESIGNENERGY"] = self._ke_MeV(p, self.lat)
        d["FMAPFN"] = '"1DPROFILE1-DEFAULT"'
        tilt = float(b.tilt_ref) + (float(el.shift.tilt) if el.shift is not None else 0.0)
        if tilt:
            d["PSI"] = tilt
        tag = {"rect": True} if b.rect else {}
        if b.edge_int2 is not None and abs(float(b.edge_int2) - float(b.edge_int1 or 0.0)) > 1e-15:
            self.rep.lossy("BEND_FINTX_DROPPED", "OPAL's SBEND has one FINT for both faces", element=el.name,
                           kind=el.kind, fint=b.edge_int1, fintx=b.edge_int2)
        if b.fringe_k2 is not None and abs(float(b.fringe_k2) - 2.80) > 1e-12:
            self.rep.lossy("FRINGE_K2_DROPPED", "OPAL's default bend profile has no K2 fringe parameter",
                           element=el.name, kind=el.kind, k2=b.fringe_k2)
        if any(v for n, v in {**mp.Bn, **mp.Bs}.items() if n not in (1, 2)):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "OPAL's SBEND carries K1 and K2 only", element=el.name,
                           kind=el.kind)
        self._record(el, rule, design_energy_MeV=d["DESIGNENERGY"])
        return [self._stmt(el, "SBEND", d, s=p.s_in, tag=tag)]

    def _native_profile(self, el, kind: str):
        nat = el.native.get("opal") or {}
        if nat.get("kind") != kind or not nat.get("z") or len(nat["z"]) < 2:
            return None
        z0 = float(nat["z"][0])
        return [float(v) - z0 for v in nat["z"]], [float(v) for v in nat["values"]], nat

    def _w_solenoid(self, el, p, rule):
        L, B = float(el.length), float(el.solenoid.Bsol_T)
        nat = self._native_profile(el, STATIC)
        if nat is not None and B and L > _THIN:
            z, bz, info = nat
            pad = float(info.get("pad") or 0.0)
            origin = el.meta.get("opal_from")
            scale = float(info.get("scale_T") or B)        # the profile's own field scale (a hard edge read back)
            tag = ({"file": info.get("file"), "L": L, "Bpeak": scale} if origin == "FieldMap"
                   else {"L": L, "pad": pad})
            st = self._stmt(el, "SOLENOID", {"L": z[-1], "KS": scale / self.brho0}, s=p.s_in - pad,
                            kind=origin or "Solenoid", tag=tag)
            fname = self._map_file(st.name, ".1dms", static_map_text(z, bz, r_max_m=self._radius(el)))
            st.attrs.insert(len(st.attrs) - 1, ("FMAPFN", f'"{fname}"'))
            self._record(el, self.RULES[origin or "Solenoid"], map_file=fname, reused_map=True)
            return [st]
        if L <= _THIN:
            self.rep.lossy("SOLENOID_TO_MARKER", "a zero-length solenoid has no field to integrate; a marker",
                           element=el.name, kind=el.kind)
            return [self._stmt(el, "MARKER", {}, s=p.s_in, tag={"Bsol": B})]
        if B == 0.0:
            self._record(el, rule, ramp_m=0.0)
            return [self._stmt(el, "DRIFT", {"L": L}, s=p.s_in, tag={"Bsol": 0.0})]
        w = min(self.sol_ramp, L)
        z, bz = plateau(L, w, self.map_points)
        st = self._stmt(el, "SOLENOID", {"L": L + w, "KS": B / self.brho0}, s=p.s_in - 0.5 * w,
                        tag={"L": L, "pad": 0.5 * w})
        fname = self._map_file(st.name, ".1dms", static_map_text(z, bz, r_max_m=self._radius(el)))
        st.attrs.insert(len(st.attrs) - 1, ("FMAPFN", f'"{fname}"'))
        self._record(el, rule, ramp_m=w, map_file=fname)
        return [st]

    @staticmethod
    def _radius(el: Element) -> float:
        ap = el.aperture
        if ap is not None and ap.half_x is not None:
            return float(ap.half_x)
        return 0.05

    # ------------------------------------------------------------------ RF
    def _cavity(self, el, p, rule, *, V: float, phase: float, freq: float, z: list[float], ez: list[float],
                length: float, s: float, kind: str, tag: dict, peak_scale: float | None = None, **details):
        """An ``RFCAVITY`` with the profile ``ez(z)`` (``z`` from 0 to ``length``) as its ``1DDynamic`` map.
        ``VOLT`` is the peak field that gives the effective voltage ``V`` at the entrance velocity (first
        order transit-time factor); ``DESIGNENERGY`` lets OPAL refine the amplitude to the crest gain."""
        ref = p.ref_in or self.lat.reference
        peak = max((abs(v) for v in ez), default=0.0)
        if peak <= 0.0:
            raise ValueError(f"{el.name}: the RF profile is identically zero")
        if peak_scale is None:
            k = 2.0 * math.pi * freq / (ref.beta * C_LIGHT)
            f_abs = abs(transit_factor(z, [v / peak for v in ez], k))
            peak_scale = abs(V) / f_abs if f_abs > 1e-300 else 0.0
        st = self._stmt(el, "RFCAVITY", {"L": length, "VOLT": peak_scale * 1e-6, "FREQ": freq * 1e-6,
                                         "LAG": phase + (math.pi if V < 0 else 0.0),
                                         "DESIGNENERGY": ref.kinetic_energy_eV * 1e-6 + abs(V) * 1e-6,
                                         "TYPE": '"STANDING"'}, s=s, kind=kind, tag=tag)
        fname = self._map_file(st.name, ".1dd", dynamic_map_text(z, [v / peak for v in ez], freq,
                                                                  r_max_m=self._radius(el)))
        st.attrs.insert(len(st.attrs) - 1, ("FMAPFN", f'"{fname}"'))
        self._record(el, rule, voltage_V=V, map_file=fname, **details)
        return [st]

    def _rf_numbers(self, el, ref) -> tuple[float, float]:
        rf = el.rf
        V = float(rf.voltage_V or 0.0)
        if not V and rf.gradient_V_per_m is not None:
            V = float(rf.gradient_V_per_m) * float(rf.L_active_m if rf.L_active_m is not None else el.length)
        if not V and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-12:
            V = float(rf.dE_ref_eV) / math.cos(rf.phase_rad)
        freq = float(rf.frequency_Hz or ref.rf_frequency_Hz or 0.0)
        return V, freq

    def _w_rfcavity(self, el, p, rule):
        ref = p.ref_in or self.lat.reference
        rf = el.rf
        V, freq = self._rf_numbers(el, ref)
        if not V or not freq:
            self.rep.lossy("OPAL_CAVITY_TO_DRIFT", "an RF cavity without a voltage or a frequency has no map to "
                           "write; a drift of its length", element=el.name, kind=el.kind, voltage_V=V,
                           frequency_Hz=freq)
            return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in, tag={"V": V, "phase": rf.phase_rad})]
        if not rf.phase_is_sync:
            self.rep.equivalent("RF_RAW_PHASE_AS_SYNC", "the source gave a driven RF phase; written as OPAL's LAG "
                                "relative to the autophased crest", element=el.name, kind=el.kind)
        nat = self._native_profile(el, DYNAMIC)
        if nat is not None:
            z, ez, info = nat
            origin = el.meta.get("opal_from")
            length = z[-1]
            if origin == "FieldMap":
                tag = {"file": info.get("file"), "L": float(el.length), "V": V, "phase": rf.phase_rad}
                if abs(length - float(el.length)) > 1e-9:
                    tag["pad"] = 0.5 * (length - float(el.length))
            else:
                tag = {"V": V, "phase": rf.phase_rad, "L": float(el.length)}
                if rf.n_cell:
                    tag["n"] = int(rf.n_cell)
                if rf.cavity_type == "TRAVELING_WAVE":
                    tag["tw"] = True
                if rf.L_active_m is not None:
                    tag["active"] = float(rf.L_active_m)
            volt = info.get("volt")
            return self._cavity(el, p, self.RULES[origin or "RFCavity"], V=V, phase=rf.phase_rad, freq=freq, z=z,
                                ez=ez, length=length, s=p.s_in - 0.5 * (length - float(el.length)),
                                kind=origin or "RFCavity", tag=tag,
                                peak_scale=float(volt) * 1e6 if volt else None, reused_map=True)
        thin = el.length <= _THIN
        beta_lambda = ref.beta * C_LIGHT / freq
        length = thin_gap_surrogate_length(V, rf.phase_rad, freq, ref) if thin else float(el.length)
        active = min(max(float(rf.L_active_m or 0.0), 0.0) or length, length)
        n_cell = int(rf.n_cell or 0)
        tw = rf.cavity_type == "TRAVELING_WAVE"
        if n_cell > 1 and not thin:
            z, ez = cell_profile(length, n_cell, pi_mode=not tw, active_m=active)
        else:
            z, ez = raised_bump(length, min(active, 0.5 * beta_lambda), self.map_points)
        tag: dict = {"V": V, "phase": rf.phase_rad}
        if thin:
            tag["L"] = 0.0
        else:
            tag["L"] = float(el.length)
        if n_cell:
            tag["n"] = n_cell
        if tw:
            tag["tw"] = True
        if rf.L_active_m is not None:
            tag["active"] = float(rf.L_active_m)
        s = p.s_in - 0.5 * length if thin else p.s_in
        return self._cavity(el, p, rule, V=V, phase=rf.phase_rad, freq=freq, z=z, ez=ez, length=length, s=s,
                            kind="RFCavity", tag=tag, thin=thin, surrogate_m=length if thin else None)

    def _w_ncells(self, el, p, rule):
        ref = p.ref_in or self.lat.reference
        rf = el.rf
        V = float(rf.voltage_V or (rf.gradient_V_per_m or 0.0) * el.length)
        freq = float(rf.frequency_Hz or ref.rf_frequency_Hz or 0.0)
        n_cell = int(el.params.get("n_cells") or rf.n_cell or 1)
        if not (V and freq and el.length > 0):
            self.rep.lossy("NCELLS_TO_DRIFT", "an NCells without a voltage, a frequency or a length is written "
                           "as a drift", element=el.name, kind="NCells")
            return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in)]
        mode = el.params.get("mode")
        z, ez = cell_profile(float(el.length), n_cell, pi_mode=(mode != 0))
        tag = {"V": V, "phase": rf.phase_rad, "n": n_cell, "L": float(el.length)}
        if mode is not None:
            tag["mode"] = mode
        unmapped = {k: v for k, v in el.params.items()
                    if k in ("beta_g", "k_eot_i", "k_eot_o", "dz_i", "dz_o") and v}
        if unmapped:
            self.rep.lossy("DYNAC_NCELLS_PARAMS", "the cell-train profile has no geometric β_g or half-cell "
                           "corrections; those NCELLS columns are dropped", element=el.name, kind="NCells",
                           **unmapped)
        return self._cavity(el, p, rule, V=V, phase=rf.phase_rad, freq=freq, z=z, ez=ez, length=float(el.length),
                            s=p.s_in, kind="NCells", tag=tag, n_cells=n_cell)

    def _w_fieldmap(self, el, p, rule):
        s = (el.meta or {}).get("map_summary") or {}
        data = None
        try:
            data = load_field_map(el)
        except (OSError, ValueError, NotImplementedError):
            data = None
        ez = bz = None
        if data is not None:
            for ch in data.channels.values():
                if ch.z is None or ch.Fz is None or getattr(ch.Fz, "ndim", 1) != 1:
                    continue
                if ch.is_electric and not ch.is_static and ez is None:
                    ez = ([float(v) for v in ch.z], [float(v) * float(el.ke) for v in ch.Fz])
                elif not ch.is_electric and ch.is_static and bz is None and ch.digit != 9:
                    bz = ([float(v) for v in ch.z], [float(v) * float(el.kb) for v in ch.Fz])
        file = el.files[0] if el.files else None
        if ez is not None and s.get("kind") == "rf" and el.rf.frequency_Hz and s.get("v_c_V"):
            z, e = ez
            z0 = z[0]
            zz = [v - z0 for v in z]
            L = zz[-1]
            tag = {"file": file, "L": float(el.length), "V": float(s["v_c_V"]),
                   "phase": float(s.get("phase_sync_rad") or 0.0)}
            if abs(L - float(el.length)) > 1e-9:
                tag["pad"] = 0.5 * (L - float(el.length))
            peak = max(abs(v) for v in e)
            out = self._cavity(el, p, self.RULES["FieldMap"], V=float(s["v_c_V"]),
                               phase=float(s.get("phase_sync_rad") or 0.0), freq=float(el.rf.frequency_Hz),
                               z=zz, ez=e, length=L, s=p.s_in - 0.5 * (L - float(el.length)), kind="FieldMap",
                               tag=tag, peak_scale=peak, file=file, static="none")
            if bz is not None:
                self.rep.lossy("FM_STATIC_B_DROPPED", "the map's static magnetic channel has no place on an "
                               "RFCAVITY; dropped", element=el.name, kind="FieldMap")
            return out
        if bz is not None and s.get("kind") == "solenoid" and ez is None:
            z, b = bz
            z0 = z[0]
            zz = [v - z0 for v in z]
            L = zz[-1]
            peak = max(abs(v) for v in b)
            if peak > 0:
                st = self._stmt(el, "SOLENOID", {"L": L, "KS": peak / self.brho0},
                                s=p.s_in - 0.5 * (L - float(el.length)), kind="FieldMap",
                                tag={"file": file, "L": float(el.length), "Bpeak": peak})
                fname = self._map_file(st.name, ".1dms", static_map_text(zz, [v / peak for v in b],
                                                                         r_max_m=self._radius(el)))
                st.attrs.insert(len(st.attrs) - 1, ("FMAPFN", f'"{fname}"'))
                self._record(el, self.RULES["FieldMap"], map_file=fname, file=file, peak_T=peak)
                return [st]
        r = replacement_for(el)
        out: list[_Stmt] = []
        st_ = p.s_in
        for part in r.parts:
            sub = p.model_copy(update={"element": part, "s_in": st_, "s_out": st_ + part.length})
            fn = getattr(self, f"_w_{part.kind.lower()}")
            for g in fn(part, sub, self.RULES[part.kind]):
                g.tag["from"] = "FieldMap"
                out.append(g)
            st_ += part.length
        getattr(self.rep, r.cls.lower())(r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, msg in r.extra:
            getattr(self.rep, cls.lower())(code, msg, element=el.name, kind="FieldMap")
        return out

    def _w_rfqcell(self, el, p, rule):
        self._record(el, rule)
        return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in)]

    # ------------------------------------------------------------------ the rest
    def _w_kicker(self, el, p, rule):
        if el.electric:
            self.rep.lossy("EKICK_AS_MAGNETIC", "an electric kicker is written as a magnetic KICKER of the same "
                           "deflection", element=el.name, kind=el.kind)
        thin = el.length <= _THIN
        ell = self.thin if thin else float(el.length)
        s = p.s_in - 0.5 * ell if thin else p.s_in
        d = {"L": ell, "HKICK": float(el.hkick), "VKICK": float(el.vkick),
             "DESIGNENERGY": self._ke_MeV(p, self.lat)}
        self._record(el, rule)
        return [self._stmt(el, "KICKER", d, s=s, tag={"L": 0.0} if thin else None)]

    def _w_collimator(self, el, p, rule):
        ap = el.aperture
        if ap is None or (ap.half_x is None and ap.half_y is None):
            self.rep.lossy("COLLIMATOR_TO_MARKER", "a collimator without limits is a marker (a drift over its "
                           "length)", element=el.name, kind=el.kind)
            if el.length > _THIN:
                return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in)]
            return [self._stmt(el, "MARKER", {}, s=p.s_in)]
        thin = el.length <= _THIN
        ell = self.thin if thin else float(el.length)
        s = p.s_in - 0.5 * ell if thin else p.s_in
        hx = ap.half_x if ap.half_x is not None else ap.half_y
        hy = ap.half_y if ap.half_y is not None else ap.half_x
        etype = "RCOLLIMATOR" if ap.shape == "RECTANGULAR" else "ECOLLIMATOR"
        if (ap.x_limits and abs(ap.x_limits[0] + ap.x_limits[1]) > 1e-15) or \
                (ap.y_limits and abs(ap.y_limits[0] + ap.y_limits[1]) > 1e-15):
            self.rep.lossy("APERTURE_OFFSET_DROPPED", "OPAL's collimator is centred; the aperture offset is dropped",
                           element=el.name, kind=el.kind)
        self._record(el, rule)
        return [self._stmt(el, etype, {"L": ell, "XSIZE": hx, "YSIZE": hy}, s=s,
                           tag={"L": 0.0} if thin else None)]

    def _w_marker(self, el, p, rule):
        self._record(el, rule)
        return [self._stmt(el, "MARKER", {}, s=p.s_in)]

    def _w_instrument(self, el, p, rule):
        self._record(el, rule, family=el.family)
        tag: dict = {"family": el.family}
        if el.params:
            tag["params"] = ";".join(f"{k}:{v}" for k, v in el.params.items())
        st = self._stmt(el, "MONITOR", {"L": el.length}, s=p.s_in, tag=tag)
        st.attrs.insert(len(st.attrs) - 1, ("OUTFN", f'"{st.name}.h5"'))
        return [st]

    def _w_foil(self, el, p, rule):
        self._record(el, rule)
        tag = {"material": el.material, "thick": el.thickness_kg_per_m2}
        if el.dE_ref_eV:
            tag["dE"] = el.dE_ref_eV
        if el.length > _THIN:
            return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in, tag=tag)]
        return [self._stmt(el, "MARKER", {}, s=p.s_in, tag=tag)]

    def _w_taylor(self, el, p, rule):
        self._record(el, rule)
        return [self._stmt(el, "DRIFT", {"L": el.length}, s=p.s_in)]

    def _w_patch(self, el, p, rule):
        self._record(el, rule)
        tag = {k: getattr(el, k) for k in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
               if getattr(el, k)}
        if el.t_offset_s:
            tag["dt"] = el.t_offset_s
        if el.e_tot_offset_eV:
            tag["dE"] = el.e_tot_offset_eV
        return [self._stmt(el, "MARKER", {}, s=p.s_in, tag=tag)]

    def _w_referencechange(self, el, p, rule):
        self._record(el, rule)
        tag = {}
        if el.dE_ref_eV is not None:
            tag["dE"] = el.dE_ref_eV
        if el.energy_eV is not None:
            tag["E"] = el.energy_eV
        if el.dphase_rad:
            tag["dphase"] = el.dphase_rad
        if el.dtime_s:
            tag["dt"] = el.dtime_s
        return [self._stmt(el, "MARKER", {}, s=p.s_in, tag=tag)]

    def _w_freq(self, el, p, rule):
        self._record(el, rule)
        return [self._stmt(el, "MARKER", {}, s=p.s_in, tag={"f": el.frequency_Hz})]

    def _w_directive(self, el, p, rule):
        self._record(el, rule, card=el.card)
        return [_Stmt("", "", [], {"directive": 1, "name": el.name, "format": el.format, "card": el.card,
                                   "args": " ".join(el.args), "role": el.role, "s": p.s_in}, in_line=False)]

    def _w_superposition(self, el, p, rule):
        self._record(el, rule)
        out: list[_Stmt] = []
        s = p.s_in
        for _off, name in el.children:
            child = self.lat.elements.get(name)
            if child is None:
                continue
            sub = p.model_copy(update={"element": child, "s_in": s, "s_out": s + child.length})
            out += self._element(sub)
            s += child.length
        return out


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)
