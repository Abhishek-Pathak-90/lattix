"""ImpactX writer (PLAN §6 task 3.2): table-driven, never silent.

Two flavours out of one element model (:func:`to_elements`):

``flavor="python"`` (default)
    a runnable script — ``ImpactX()``, ``sim.init_grids()``, the reference particle
    from :attr:`Lattice.reference`, an ``elements.KnownElementsList`` appended in flat
    order and handed to ``sim.lattice.extend(...)``.  No tracking call is emitted
    (the deck carries no beam distribution).
``flavor="inputs"``
    the AMReX ParmParse ``inputs`` file (``lattice.elements``, ``<name>.type = …``,
    ``beam.kin_energy = …``), which :class:`lattix.formats.impactx.Reader` reads back
    byte-for-byte (invariant I-13).

Conventions — **measured against impactx 26.01 and 26.08 (osx-arm64, 2026-09-03;
identical numbers on both)**, cross-checked against ImpactX's own
``src/python/impactx/madx_to_impactx.py``:

* ``Quad.k``  = G / Bρ_signed [1/m²], ``Sol.ks`` = B_z / Bρ_signed [1/m] — MAD-X
  convention; ImpactX's own rigidity ``RefPart::rigidity_Tm() = m βγ c / q`` is
  **signed by the charge**, so an H⁻ deck flips exactly as lattix's ``brho_signed``
  does (``ReferenceParticle.H:270``);
* ``Sbend.rc`` = L_arc / angle (signed, MAD-X ``angle`` sign) and a bend plane
  rotation (``BendP.tilt_ref``) becomes ``rotation`` **in degrees**;
* ``DipEdge(psi, rc, g = 2·hgap, K2 = fint, location = "entry"|"exit")`` — ImpactX's
  ``K2`` is MAD-X's ``FINT`` but its **default is 1, not 0**, so the writer always
  states ``K2`` explicitly;
* ``Multipole(multipole = n+1, K_normal, K_skew)`` is the MAD-X ``knl``/``ksl``
  convention with **no extra factorial**: the push is
  ``Δpx + iΔpy = −(K_n + i K_s)·(x+iy)^m / m!`` with ``m = multipole − 1``
  (``src/elements/Multipole.H:170-176``), so ``K_normal = B_nL / Bρ_signed``;
* ``ShortRF(V, freq, phase)``: ``V`` is **dimensionless** — the maximum energy gain
  divided by the rest energy, i.e. ``voltage_V / mass_eV`` — and ``phase`` is the
  synchronous phase in **degrees with 0 = crest**, identical to the IR.  MEASURED:
  a proton at 2.1 MeV through ``ShortRF(V = 1 MV/mc², freq = 162.5 MHz, phase = −30)``
  gains 866 025.403 784 4 eV = +V·cos 30° (expected 866 025.403 784 4);
* ``Kicker(xkick, ykick, unit="dimensionless")`` — the kick in radians, MAD-X
  ``hkick``/``vkick`` semantics;
* ``Aperture(aperture_x, aperture_y, shape, action)`` takes **half**-apertures; an
  off-centre IR aperture is centred with ``dx``/``dy``;
* alignment: ``dx``/``dy`` [m] and ``rotation`` [**degrees**] on every element.

**ImpactX follows the reference energy** (``ShortRF`` pushes the reference particle
first, then the beam), so an accelerating line is EXACT here: with the default
``energy_mode="local"`` every normalized strength uses Bρ at that element's entrance
from :func:`lattix.ir.walk.propagate`, which is exactly the rigidity ImpactX itself
will have there.  ``energy_mode="constant"`` uses the lattice-start rigidity
everywhere and is recorded as ``EQUIVALENT:CONST_START_RIGIDITY``.
"""
from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    ALL_KINDS,
    Bend,
    Collimator,
    Directive,
    Drift,
    Element,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    NCells,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    Superposition,
    Taylor,
)
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.walk import energy_gain_eV, propagate

#: ImpactX Python class name -> ``<name>.type`` string in an AMReX inputs file.
#: ``Marker`` has **no** inputs equivalent (``src/initialization/InitElement.cpp``
#: aborts on an unknown type), so it is written as a zero-length ``drift`` there.
INPUTS_TYPE: dict[str, str | None] = {
    "Drift": "drift", "ExactDrift": "drift_exact", "ChrDrift": "drift_chromatic",
    "Quad": "quad", "ExactQuad": "quad_exact", "ChrQuad": "quad_chromatic",
    "Sbend": "sbend", "ExactSbend": "sbend_exact", "CFbend": "cfbend",
    "ExactCFbend": "cfbend_exact", "DipEdge": "dipedge", "QuadEdge": "quadedge",
    "Sol": "solenoid", "SoftSolenoid": "solenoid_softedge",
    "SoftQuadrupole": "quadrupole_softedge", "ShortRF": "shortrf",
    "RFCavity": "rfcavity", "Buncher": "buncher", "Kicker": "kicker",
    "Multipole": "multipole", "ExactMultipole": "multipole_exact",
    "Aperture": "aperture", "PolygonAperture": "polygon_aperture",
    "BeamMonitor": "beam_monitor", "LinearMap": "linear_map", "ConstF": "constf",
    "NonlinearLens": "nonlinear_lens", "PlaneXYRot": "plane_xyrotation",
    "PRot": "prot", "ThinDipole": "thin_dipole", "TaperedPL": "tapered_plasma_lens",
    "ChrAcc": "uniform_acc_chromatic", "ChrPlasmaLens": "plasma_lens_chromatic",
    "Source": "source", "Marker": None,
}

#: keyword renames between the Python API and the inputs file
INPUTS_RENAME: dict[str, dict[str, str]] = {"Kicker": {"unit": "units"}}

#: ImpactX species names accepted by ``beam.particle`` (inputs) — everything else
#: needs the Python API (``set_charge_qe``/``set_mass_MeV``).
IMPACTX_SPECIES = {"proton": "proton", "electron": "electron", "positron": "positron",
                   "h-": "Hminus"}

#: names an inputs file cannot use for an element (they are ParmParse prefixes)
RESERVED = frozenset({"lattice", "beam", "algo", "diag", "amr", "amrex", "geometry",
                      "impactx", "particles", "sim", "elements", "ref", "monitor_ns"})

MAX_NAME_LEN = 64
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: an element counts as accelerating above 1 µeV of reference gain
_ACCEL_TOL_eV = 1e-6


# ---------------------------------------------------------------------------
def is_valid(name: str) -> bool:
    """A legal ImpactX element name for both flavours: a Python identifier that is
    neither a Python keyword nor an AMReX ParmParse prefix used by ImpactX."""
    return (bool(_NAME_RE.match(name)) and len(name) <= MAX_NAME_LEN
            and not keyword.iskeyword(name) and name.lower() not in RESERVED)


def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "e_" + s
    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN]
    if keyword.iskeyword(s) or s.lower() in RESERVED:
        s = (s[: MAX_NAME_LEN - 2] + "_x")
    return s


class NameMap:
    """Sanitize + uniquify; remembers renames so provenance can be re-emitted."""

    def __init__(self) -> None:
        self._used: set[str] = set()
        self._by_key: dict[int, str] = {}
        self._derived: dict[tuple[str, str], str] = {}
        self.renamed: dict[str, str] = {}

    def derive(self, base: str, suffix: str) -> str:
        """A stable extra name for one source element (its edges, split drifts, …) so a
        definition used twice in the line keeps one set of names."""
        key = (base, suffix)
        if key not in self._derived:
            self._derived[key] = self.assign(f"{base}_{suffix}")
        return self._derived[key]

    def assign(self, original: str, key: object | None = None) -> str:
        if key is not None and id(key) in self._by_key:
            return self._by_key[id(key)]
        base = sanitize(original)
        name, k = base, 2
        while name in self._used:
            suffix = f"_{k}"
            name = base[: MAX_NAME_LEN - len(suffix)] + suffix
            k += 1
        self._used.add(name)
        if key is not None:
            self._by_key[id(key)] = name
        if name != original:
            self.renamed[name] = original
        return name


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    return "0" if s in ("-0", "-0.0") else s


def _deg(rad: float) -> float:
    return math.degrees(rad)


def _is_matrix(v) -> bool:
    return isinstance(v, (list, tuple)) and bool(v) and isinstance(v[0], (list, tuple))


def flavor_for_path(path: str | Path) -> str:
    """``inputs`` for a ``.in`` file, ``python`` otherwise (used when ``write()`` is
    called through the format registry, which passes no ``flavor``)."""
    return "inputs" if str(path).lower().endswith(".in") else "python"


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


@dataclass
class Emit:
    """One ImpactX element to build: the Python class name and its keyword arguments.

    ``params`` is ordered exactly as it should be printed; ``length`` is the ``ds``
    the element contributes to ``s`` (0 for thin elements)."""

    name: str
    cls: str
    params: dict = field(default_factory=dict)
    length: float = 0.0
    source: str | None = None       # IR element name this came from

    def inputs_lines(self) -> list[str]:
        """``<name>.key = value`` lines for an AMReX inputs file."""
        t = INPUTS_TYPE.get(self.cls)
        if t is None:
            raise ValueError(f"{self.cls} has no inputs-file type")
        rename = INPUTS_RENAME.get(self.cls, {})
        out = [f"{self.name}.type = {t}"]
        for k, v in self.params.items():
            if k == "name":
                continue
            key = f"{self.name}.{rename.get(k, k)}"
            if k == "R" and _is_matrix(v):
                # ImpactX inputs spell the 6x6 map R11 … R66, 1-indexed, identity default
                for i, row in enumerate(v, start=1):
                    for j, x in enumerate(row, start=1):
                        if x != (1.0 if i == j else 0.0):
                            out.append(f"{self.name}.R{i}{j} = {_num(x)}")
                continue
            if isinstance(v, bool):
                out.append(f"{key} = {'true' if v else 'false'}")
            elif isinstance(v, str):
                out.append(f"{key} = {v}")
            elif isinstance(v, int):
                out.append(f"{key} = {v}")
            elif isinstance(v, (list, tuple)):
                out.append(f"{key} = " + " ".join(_num(x) for x in v))
            else:
                out.append(f"{key} = {_num(v)}")
        return out

    def python_call(self) -> str:
        args = [f'name="{self.name}"']
        for k, v in self.params.items():
            if k == "name":
                continue
            if isinstance(v, bool):
                args.append(f"{k}={v}")
            elif isinstance(v, str):
                args.append(f'{k}="{v}"')
            elif isinstance(v, int):
                args.append(f"{k}={v}")
            elif k == "R" and _is_matrix(v):
                args.append("R=_map6([" + ", ".join("[" + ", ".join(_num(x) for x in row) + "]"
                                                    for row in v) + "])")
            elif isinstance(v, (list, tuple)):
                args.append(f"{k}=[" + ", ".join(_num(x) for x in v) + "]")
            else:
                args.append(f"{k}={_num(v)}")
        return f"elements.{self.cls}(" + ", ".join(args) + ")"


# ---------------------------------------------------------------------------
class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "impactx"

    RULES: dict[str, Rule] = {
        "Drift": Rule("Drift"),
        "Quadrupole": Rule("Quad"),
        "Sextupole": Rule("Multipole (thin, drift-kick-drift)", "LOSSY", "THICK_TO_THIN_MULTIPOLE",
                          "a thick sextupole is written as drift + thin Multipole + drift"),
        "Octupole": Rule("Multipole (thin, drift-kick-drift)", "LOSSY", "THICK_TO_THIN_MULTIPOLE",
                         "a thick octupole is written as drift + thin Multipole + drift"),
        "Multipole": Rule("Multipole"),
        "Bend": Rule("DipEdge + Sbend/CFbend + DipEdge"),
        "Solenoid": Rule("Sol"),
        "RFCavity": Rule("ShortRF"),
        "FieldMap": Rule("ShortRF or Drift", "LOSSY", "FM_TO_DRIFT",
                         "field map replaced by a thin cavity (known gain) or a drift"),
        "NCells": Rule("Drift", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("Drift", "LOSSY", "RFQ_TO_DRIFT",
                        "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("Kicker"),
        "Collimator": Rule("Aperture"),
        "Marker": Rule("Marker"),
        "Instrument": Rule("Marker or BeamMonitor", "EQUIVALENT", "INSTRUMENT_AS_MARKER",
                           "diagnostics are optically transparent; pass instrument='beam_monitor' "
                           "to write an ImpactX BeamMonitor instead"),
        "Foil": Rule("Marker", "LOSSY", "FOIL_TO_MARKER",
                     "ImpactX has no stripping foil; written as a marker"),
        "Taylor": Rule("LinearMap"),
        "Patch": Rule("PlaneXYRot / Drift / Marker", "LOSSY", "PATCH_DROPPED",
                      "ImpactX has no general patch; only a pure roll or a pure z shift survives"),
        "ReferenceChange": Rule("Marker", "LOSSY", "REFCHANGE_DROPPED",
                                "ImpactX has no explicit reference-energy jump; written as a marker"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each ImpactX cavity's freq attribute"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "overlapping fields written as consecutive elements"),
    }

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              flavor: str | None = None, nslice: int = 1, energy_mode: str = "local",
              instrument: str = "marker", npart: int = 10000,
              bunch_charge_C: float | None = None) -> FidelityReport:
        # one format, two deck flavours: the file name picks one unless the caller says
        # otherwise (``…​.impactx.in`` -> AMReX inputs, ``…​.impactx.py`` -> Python script)
        flavor = flavor or flavor_for_path(path)
        if flavor not in ("python", "inputs"):
            raise ValueError(f"flavor must be 'python' or 'inputs', got {flavor!r}")
        if energy_mode not in ("local", "constant"):
            raise ValueError(f"energy_mode must be 'local' or 'constant', got {energy_mode!r}")
        if int(nslice) < 1:
            raise ValueError(f"nslice must be >= 1, got {nslice!r}")

        text, rep = self._render(lattice, flavor=flavor, nslice=nslice, energy_mode=energy_mode,
                                 instrument=instrument, npart=npart,
                                 bunch_charge_C=bunch_charge_C)
        rep.target_file = str(path)
        Path(path).write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def render(self, lattice: Lattice, **options) -> str:
        """The deck text, without writing it (used by the golden tests)."""
        return self._render(lattice, **options)[0]

    def _render(self, lattice: Lattice, *, flavor: str = "python", nslice: int = 1,
                energy_mode: str = "local", instrument: str = "marker", npart: int = 10000,
                bunch_charge_C: float | None = None) -> tuple[str, FidelityReport]:
        emits, rep = to_elements(lattice, flavor=flavor, nslice=nslice, energy_mode=energy_mode,
                                 instrument=instrument)
        rep.target_format = "impactx"
        text = (self._python_text(lattice, emits, rep, nslice=nslice)
                if flavor == "python" else
                self._inputs_text(lattice, emits, rep, nslice=nslice, npart=npart,
                                  bunch_charge_C=bunch_charge_C))
        return text, rep

    # -- python flavour -------------------------------------------------
    @staticmethod
    def _python_text(lattice: Lattice, emits: list[Emit], rep: FidelityReport, *, nslice: int) -> str:
        ref = lattice.reference
        src = lattice.meta.get("source_format", "IR")
        out = ["#!/usr/bin/env python3",
               f"# lattix {__version__} from {src}",
               '"""ImpactX lattice generated by lattix.',
               "",
               "Strengths are normalized with the SIGNED rigidity at each element's entrance",
               "(energy_mode='local'), which is the rigidity ImpactX itself carries there.",
               "Add a beam distribution and call sim.track_particles() to run.",
               '"""',
               "from impactx import ImpactX, elements",
               ]
        if any(e.cls == "LinearMap" for e in emits):
            out += ["import amrex.space3d as amr",
                    "",
                    "",
                    "def _map6(rows):",
                    '    """6x6 transport matrix for elements.LinearMap (1-indexed)."""',
                    "    m = amr.SmallMatrix_6x6_F_SI1_double()",
                    '    base = getattr(m, "starting_index", 1)',
                    "    for i, row in enumerate(rows, start=base):",
                    "        for j, v in enumerate(row, start=base):",
                    "            m[i, j] = v",
                    "    return m",
                    "",
                    ]
        out += ["",
                "sim = ImpactX()",
                "sim.particle_shape = 2",
                "sim.space_charge = False",
                "sim.slice_step_diagnostics = False",
                "sim.init_grids()",
                "",
                f"# --- reference particle: {ref.species.name}, "
                f"{_num(ref.kinetic_energy_eV * 1e-6)} MeV kinetic",
                "ref = sim.particle_container().ref_particle()",
                f"ref.set_charge_qe({_num(ref.species.charge)})",
                f"ref.set_mass_MeV({_num(ref.species.mass_eV * 1e-6)})",
                f"ref.set_kin_energy_MeV({_num(ref.kinetic_energy_eV * 1e-6)})",
                "",
                f"# --- lattice ({len(emits)} elements, default nslice = {nslice})",
                "lattice = elements.KnownElementsList()",
                ]
        for e in emits:
            out.append(f"lattice.append({e.python_call()})")
        out += ["", "sim.lattice.extend(lattice)", ""]
        for line in Writer._trailer(lattice, rep):
            out.append(line)
        out += ["", "# sim.track_particles()   # add a beam distribution first",
                "# sim.finalize()"]
        return "\n".join(out).rstrip("\n") + "\n"

    # -- inputs flavour -------------------------------------------------
    @staticmethod
    def _inputs_text(lattice: Lattice, emits: list[Emit], rep: FidelityReport, *, nslice: int,
                     npart: int, bunch_charge_C: float | None) -> str:
        ref = lattice.reference
        src = lattice.meta.get("source_format", "IR")
        beam = dict(lattice.meta.get("impactx_beam") or {})
        rule = "#" * 79
        sp = IMPACTX_SPECIES.get(ref.species.name.lower())
        if sp is None:
            sp = "proton"
            rep.lossy("SPECIES_NOT_REPRESENTABLE",
                      f"an ImpactX inputs file only names {sorted(IMPACTX_SPECIES)}; "
                      f"{ref.species.name!r} was written as 'proton' — use flavor='python' "
                      "(set_charge_qe/set_mass_MeV) for other species",
                      species=ref.species.name)
        charge = bunch_charge_C if bunch_charge_C is not None else float(beam.pop("charge", 1.0e-9))
        out = [rule, f"# lattix {__version__} from {src}", "# Particle Beam(s)", rule,
               f"beam.npart = {int(beam.pop('npart', npart))}",
               f"beam.units = {beam.pop('units', 'static')}",
               f"beam.kin_energy = {_num(ref.kinetic_energy_eV * 1e-6)}",
               f"beam.charge = {_num(charge)}",
               f"beam.particle = {sp}",
               f"beam.distribution = {beam.pop('distribution', 'waterbag')}"]
        defaults = {"lambdaX": 1e-4, "lambdaY": 1e-4, "lambdaT": 1e-4,
                    "lambdaPx": 1e-4, "lambdaPy": 1e-4, "lambdaPt": 1e-4}
        for k, v in defaults.items():
            out.append(f"beam.{k} = {_num(beam.pop(k, v))}")
        for k in sorted(beam):
            v = beam[k]
            out.append(f"beam.{k} = {v if isinstance(v, str) else _num(v)}")

        out += ["", rule, "# Beamline: lattice elements and segments", rule]
        defs = _unique_definitions(emits, rep)
        out.append("lattice.elements = " + " ".join(e.name for e in emits))
        out.append(f"lattice.nslice = {int(nslice)}")
        for e in defs:
            out.append("")
            out.extend(e.inputs_lines())
        out += ["", rule, "# Algorithms", rule, "algo.space_charge = false", ""]
        trailer = Writer._trailer(lattice, rep)
        if trailer:
            out += [rule, "# lattix notes", rule] + [f"# {line[2:]}" if line.startswith("# ")
                                                     else line for line in trailer]
        return "\n".join(out).rstrip("\n") + "\n"

    @staticmethod
    def _trailer(lattice: Lattice, rep: FidelityReport) -> list[str]:
        """Comment lines recording what could not be written as an element."""
        out: list[str] = []
        for e in rep.entries:
            if e.code == "FOREIGN_DIRECTIVE":
                out.append(f"# dropped directive {e.element}: {e.details.get('card', '')} "
                           f"{' '.join(e.details.get('args', []))}".rstrip())
        if lattice.warnings:
            out.append("# source warnings: " + "; ".join(lattice.warnings[:5]))
        return out


def _unique_definitions(emits: list[Emit], rep: FidelityReport) -> list[Emit]:
    """One ``<name>.type = …`` block per name.  A repeated name whose parameters differ
    (the same definition used at two reference energies) is renamed in place."""
    defs: dict[str, Emit] = {}
    used = {e.name for e in emits}
    order: list[Emit] = []
    for e in emits:
        first = defs.get(e.name)
        if first is None:
            defs[e.name] = e
            order.append(e)
            continue
        if first.cls == e.cls and first.params == e.params:
            continue
        base, k = e.name, 2
        while f"{base}_{k}" in used:
            k += 1
        rep.equivalent("DEFINITION_SPLIT",
                       "one source element is used at two reference energies, so its ImpactX "
                       "parameters differ; the second occurrence was renamed",
                       element=e.source, first=e.name, renamed=f"{base}_{k}")
        e.name = f"{base}_{k}"
        used.add(e.name)
        defs[e.name] = e
        order.append(e)
    return order


# ---------------------------------------------------------------------------
def to_elements(lattice: Lattice, *, flavor: str = "python", nslice: int = 1,
                energy_mode: str = "local", instrument: str = "marker",
                report: FidelityReport | None = None) -> tuple[list[Emit], FidelityReport]:
    """Flatten *lattice* into the ImpactX element list both flavours (and the oracle)
    build from.  One ledger entry per source element."""
    rep = report if report is not None else FidelityReport(target_format="impactx")
    placed = propagate(lattice)
    names = NameMap()
    start_brho = lattice.reference.brho_signed
    mass_eV = lattice.reference.species.mass_eV
    out: list[Emit] = []
    accel = 0
    for p in placed:
        ref = p.ref_in or lattice.reference
        brho = ref.brho_signed if energy_mode == "local" else start_brho
        if abs(energy_gain_eV(p.element, ref)) > _ACCEL_TOL_eV:
            accel += 1
        _emit_element(p, p.element, brho, mass_eV, lattice, names, rep, out,
                      flavor=flavor, nslice=nslice, instrument=instrument)
    if accel and energy_mode == "constant":
        rep.equivalent("CONST_START_RIGIDITY",
                       "normalized strengths use the lattice-start rigidity although ImpactX "
                       "follows the reference energy through RF; energy_mode='local' is exact",
                       accelerating_elements=accel)
    elif accel:
        rep.exact(None, None, code="FOLLOWS_P0",
                  message=f"ImpactX follows the reference energy through {accel} accelerating "
                          "element(s); local rigidities are exact")
    return out, rep


# ---------------------------------------------------------------------------
def _alignment(el: Element, extra_tilt_rad: float, rep: FidelityReport) -> dict:
    """``dx``/``dy``/``rotation`` from BodyShiftP plus an element-intrinsic roll."""
    p: dict = {}
    sh = el.shift
    tilt = extra_tilt_rad
    if sh is not None and not sh.is_zero():
        if sh.x_offset:
            p["dx"] = sh.x_offset
        if sh.y_offset:
            p["dy"] = sh.y_offset
        tilt += sh.tilt
        if sh.z_offset or sh.x_rot or sh.y_rot:
            rep.lossy("MISALIGN_DROPPED",
                      "ImpactX alignment errors are dx, dy and a transverse rotation only; "
                      "z_offset/x_rot/y_rot were dropped",
                      element=el.name, kind=el.kind, z_offset=sh.z_offset,
                      x_rot=sh.x_rot, y_rot=sh.y_rot)
    if tilt:
        p["rotation"] = _deg(tilt)
    return p


def _aperture(el: Element, thick: bool, rep: FidelityReport) -> dict:
    """``aperture_x``/``aperture_y`` for a thick element (ImpactX: elliptical only)."""
    ap = el.aperture
    if ap is None or (ap.half_x is None and ap.half_y is None):
        return {}
    if not thick:
        rep.lossy("APERTURE_DROPPED",
                  "a thin ImpactX element carries no aperture; add a separate Aperture element",
                  element=el.name, kind=el.kind)
        return {}
    hx, hy = ap.half_x, ap.half_y
    if hx is None or hy is None:
        rep.lossy("APERTURE_PARTIAL",
                  "ImpactX needs both aperture_x and aperture_y; a one-sided limit was dropped",
                  element=el.name, kind=el.kind, half_x=hx, half_y=hy)
        return {}
    if ap.shape == "RECTANGULAR":
        rep.equivalent("APERTURE_SHAPE_ELLIPTICAL",
                       "a thick ImpactX element's aperture is always elliptical; the rectangular "
                       "half-sizes were kept as the ellipse semi-axes",
                       element=el.name, kind=el.kind)
    return {"aperture_x": hx, "aperture_y": hy}


def _nslice_of(el: Element, default: int) -> int:
    """A per-element ``tracking['nslice']`` (what the ImpactX reader stores) wins over the
    writer's default, so an inputs deck survives write ∘ read ∘ write unchanged."""
    try:
        return max(1, int(el.tracking.get("nslice", default)))
    except (TypeError, ValueError):
        return int(default)


def _thick(params: dict, ds: float, nslice: int) -> dict:
    out = {"ds": ds}
    out.update(params)
    out["nslice"] = int(nslice)
    return out


def _multipole_emits(el: Element, base: str, brho: float, orders: dict[int, float],
                     skew: dict[int, float], tilt: dict[int, float], names: NameMap,
                     rep: FidelityReport) -> list[Emit]:
    """One thin ``Multipole`` per multipole order present (MAD-X knl/ksl convention)."""
    out: list[Emit] = []
    for n in sorted(set(orders) | set(skew)):
        kn = orders.get(n, 0.0) / brho
        ks = skew.get(n, 0.0) / brho
        if not kn and not ks:
            continue
        nm = base if len(set(orders) | set(skew)) == 1 else names.derive(base, f"m{n + 1}")
        params: dict = {"multipole": n + 1, "K_normal": kn, "K_skew": ks}
        params.update(_alignment(el, tilt.get(n, 0.0), rep))
        out.append(Emit(nm, "Multipole", params, 0.0, el.name))
    return out


def _rf_voltage(el: Element, rep: FidelityReport) -> float | None:
    """Effective accelerating voltage [V] of an RF element, or None when unknown."""
    rf = getattr(el, "rf", None)
    if rf is None:
        return None
    if rf.voltage_V:
        return float(rf.voltage_V)
    if rf.gradient_V_per_m:
        active = rf.L_active_m if rf.L_active_m is not None else el.length
        if active:
            return float(rf.gradient_V_per_m * active)
    if rf.dE_ref_eV:
        c = math.cos(rf.phase_rad)
        if abs(c) > 1e-9:
            return float(rf.dE_ref_eV / c)
        rep.lossy("RF_VOLTAGE_UNKNOWN",
                  "only dE_ref is known and the cavity sits at a zero crossing, so no "
                  "equivalent ShortRF voltage exists", element=el.name, kind=el.kind)
    return None


def _shortrf(el: Element, name: str, voltage_V: float, mass_eV: float, rep: FidelityReport) -> Emit:
    rf = el.rf
    freq = float(rf.frequency_Hz or 0.0)
    if not freq:
        rep.lossy("RF_FREQUENCY_UNKNOWN",
                  "ImpactX ShortRF needs an RF frequency; 0 Hz was written",
                  element=el.name, kind=el.kind)
    params: dict = {"V": voltage_V / mass_eV, "freq": freq, "phase": _deg(rf.phase_rad)}
    params.update(_alignment(el, 0.0, rep))
    return Emit(name, "ShortRF", params, 0.0, el.name)


def _split_thin(name: str, el: Element, thin: list[Emit], nslice: int,
                names: NameMap) -> list[Emit]:
    """drift(L/2) + thin kick(s) + drift(L/2) for a thick element modelled thin."""
    if el.length <= 0.0:
        return thin
    half = el.length / 2.0
    a = Emit(names.derive(name, "in"), "Drift", {"ds": half, "nslice": _nslice_of(el, nslice)}, half, el.name)
    b = Emit(names.derive(name, "out"), "Drift", {"ds": half, "nslice": _nslice_of(el, nslice)}, half, el.name)
    return [a, *thin, b]


def _emit_element(p: Placed, el: Element, brho: float, mass_eV: float, lattice: Lattice,
                  names: NameMap, rep: FidelityReport, out: list[Emit], *, flavor: str,
                  nslice: int, instrument: str) -> None:
    """Append the ImpactX elements for one IR element and record exactly one ledger entry."""
    rule = Writer.RULES[el.kind]
    name = names.assign(el.name, el)
    ap_thick = _aperture(el, el.length > 0.0, rep)

    # -- straight optics ------------------------------------------------
    if isinstance(el, Drift):
        params = _thick({**_alignment(el, 0.0, rep), **ap_thick}, el.length, _nslice_of(el, nslice))
        out.append(Emit(name, "Drift", params, el.length, el.name))
        rep.exact(el.name, el.kind)
        return

    if isinstance(el, Quadrupole):
        g = el.multipole.Bn.get(1, 0.0)
        if el.length > 0.0:
            params = _thick({"k": g / brho, **_alignment(el, el.skew_rad, rep), **ap_thick},
                            el.length, nslice)
            out.append(Emit(name, "Quad", params, el.length, el.name))
            rep.exact(el.name, el.kind, message=f"k = {g / brho:.10g} 1/m^2 at Brho = {brho:.10g} T·m")
            return
        gl = el.multipole.BnL.get(1, 0.0)
        sl = el.multipole.BsL.get(1, 0.0)
        if gl or sl:
            out.extend(_multipole_emits(el, name, brho, {1: gl}, {1: sl}, el.multipole.tilt,
                                        names, rep))
            rep.exact(el.name, el.kind, message="thin quadrupole written as a thin Multipole")
        else:
            out.append(_marker(name, el, flavor))
            rep.exact(el.name, el.kind, message="zero-length quadrupole with no strength")
        return

    if isinstance(el, (Sextupole, Octupole)):
        n = 2 if isinstance(el, Sextupole) else 3
        kn = {n: el.multipole.Bn.get(n, 0.0) * el.length + el.multipole.BnL.get(n, 0.0)}
        ks = {n: el.multipole.Bs.get(n, 0.0) * el.length + el.multipole.BsL.get(n, 0.0)}
        thin = _multipole_emits(el, name, brho, kn, ks, el.multipole.tilt, names, rep)
        if not thin:
            out.append(Emit(name, "Drift", _thick({**ap_thick}, el.length, _nslice_of(el, nslice)),
                            el.length, el.name)
                       if el.length > 0 else _marker(name, el, flavor))
            rep.exact(el.name, el.kind, message="no multipole strength; written as a drift/marker")
            return
        if el.length > 0.0:
            out.extend(_split_thin(name, el, thin, nslice, names))
            rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind, length=el.length,
                      integrated_K=[e.params["K_normal"] for e in thin])
        else:
            out.extend(thin)
            rep.exact(el.name, el.kind)
        return

    if isinstance(el, Multipole):
        kn = dict(el.multipole.BnL)
        ks = dict(el.multipole.BsL)
        for n, v in el.multipole.Bn.items():
            kn[n] = kn.get(n, 0.0) + v * el.length
        for n, v in el.multipole.Bs.items():
            ks[n] = ks.get(n, 0.0) + v * el.length
        thin = _multipole_emits(el, name, brho, kn, ks, el.multipole.tilt, names, rep)
        if not thin:
            out.append(_marker(name, el, flavor))
            rep.exact(el.name, el.kind, message="thin multipole with no strength")
            return
        if el.length > 0.0:
            out.extend(_split_thin(name, el, thin, nslice, names))
            rep.lossy("THICK_TO_THIN_MULTIPOLE",
                      "a multipole with a finite length is written as drift + kick + drift",
                      element=el.name, kind=el.kind, length=el.length)
        else:
            out.extend(thin)
            rep.exact(el.name, el.kind)
        return

    if isinstance(el, Solenoid):
        b = el.solenoid.Bsol_T
        if el.length > 0.0:
            params = _thick({"ks": b / brho, **_alignment(el, 0.0, rep), **ap_thick},
                            el.length, nslice)
            out.append(Emit(name, "Sol", params, el.length, el.name))
            rep.exact(el.name, el.kind, message=f"ks = {b / brho:.10g} 1/m at Brho = {brho:.10g} T·m")
        else:
            out.append(_marker(name, el, flavor))
            rep.lossy("THIN_SOLENOID_DROPPED",
                      "ImpactX has no zero-length solenoid; written as a marker",
                      element=el.name, kind=el.kind, Bsol_T=b)
        return

    # -- bends ----------------------------------------------------------
    if isinstance(el, Bend):
        _emit_bend(el, name, brho, names, rep, out, nslice=_nslice_of(el, nslice),
                   ap=ap_thick, flavor=flavor)
        return

    # -- RF -------------------------------------------------------------
    if isinstance(el, RFCavity):
        v = _rf_voltage(el, rep)
        if v is None:
            out.append(Emit(name, "Drift", _thick({}, el.length, _nslice_of(el, nslice)),
                            el.length, el.name)
                       if el.length > 0 else _marker(name, el, flavor))
            rep.lossy("RF_VOLTAGE_UNKNOWN", "cavity with no voltage/gradient; written as a drift",
                      element=el.name, kind=el.kind)
            return
        thin = [_shortrf(el, name if el.length == 0.0 else names.derive(name, "rf"), v,
                         mass_eV, rep)]
        if el.length > 0.0:
            out.extend(_split_thin(name, el, thin, nslice, names))
            rep.equivalent("THICK_CAVITY_AS_SHORTRF",
                           "a cavity with a finite length is written as drift + ShortRF + drift "
                           "(the reference gain and the length are exact; the transit-time "
                           "transverse RF focusing is not)",
                           element=el.name, kind=el.kind, length=el.length, voltage_V=v)
        else:
            out.extend(thin)
            rep.exact(el.name, el.kind,
                      message=f"ShortRF V = {v / mass_eV:.10g} (= {v:.10g} V / mc^2), "
                              f"phase = {_deg(el.rf.phase_rad):.10g} deg (0 = crest)")
        return

    if isinstance(el, FieldMap):
        v = _rf_voltage(el, rep)
        if v is not None and el.rf.frequency_Hz:
            thin = [_shortrf(el, names.derive(name, "rf"), v, mass_eV, rep)]
            out.extend(_split_thin(name, el, thin, nslice, names) if el.length > 0 else thin)
            rep.lossy("FM_TO_CAVITY",
                      "field map replaced by drift + thin ShortRF + drift with the map's "
                      "effective voltage", element=el.name, kind=el.kind, voltage_V=v,
                      length=el.length, geom=el.geom)
        else:
            out.append(Emit(name, "Drift", _thick({**ap_thick}, el.length, _nslice_of(el, nslice)), el.length,
                            el.name))
            rep.lossy("FM_TO_DRIFT", "field map replaced by a drift of the same length",
                      element=el.name, kind=el.kind, length=el.length, geom=el.geom)
        return

    if isinstance(el, (NCells, RFQCell)):
        out.append(Emit(name, "Drift", _thick({**ap_thick}, el.length, _nslice_of(el, nslice)), el.length, el.name))
        rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind, length=el.length)
        return

    # -- thin hardware --------------------------------------------------
    if isinstance(el, Kicker):
        params: dict = {"xkick": el.hkick, "ykick": el.vkick, "unit": "dimensionless"}
        params.update(_alignment(el, 0.0, rep))
        thin = [Emit(name if el.length == 0 else names.derive(name, "k"), "Kicker", params,
                     0.0, el.name)]
        if el.length > 0.0:
            out.extend(_split_thin(name, el, thin, nslice, names))
            rep.equivalent("THICK_KICKER_SPLIT",
                           "a corrector with a finite length is written as drift + kick + drift",
                           element=el.name, kind=el.kind, length=el.length)
        else:
            out.extend(thin)
            if el.electric:
                rep.lossy("ELECTRIC_KICKER",
                          "ImpactX Kicker is magnetic; the deflection angle was kept",
                          element=el.name, kind=el.kind)
            else:
                rep.exact(el.name, el.kind)
        return

    if isinstance(el, Collimator):
        ap = el.aperture
        hx = None if ap is None else ap.half_x
        hy = None if ap is None else ap.half_y
        if hx is None or hy is None:
            out.append(_marker(name, el, flavor))
            rep.lossy("APERTURE_UNSET",
                      "collimator without both half-apertures; written as a marker so nothing "
                      "is absorbed", element=el.name, kind=el.kind)
            return
        cx = 0.5 * (ap.x_limits[0] + ap.x_limits[1])
        cy = 0.5 * (ap.y_limits[0] + ap.y_limits[1])
        params = {"aperture_x": hx, "aperture_y": hy,
                  "shape": "rectangular" if ap.shape == "RECTANGULAR" else "elliptical",
                  "action": "absorb"}
        align = _alignment(el, 0.0, rep)
        if cx:
            align["dx"] = align.get("dx", 0.0) + cx
        if cy:
            align["dy"] = align.get("dy", 0.0) + cy
        params.update(align)
        thin = [Emit(name if el.length == 0 else names.derive(name, "ap"), "Aperture", params,
                     0.0, el.name)]
        if el.length > 0.0:
            out.extend(_split_thin(name, el, thin, nslice, names))
            rep.equivalent("THICK_COLLIMATOR_SPLIT",
                           "ImpactX Aperture is thin; written as drift + aperture + drift",
                           element=el.name, kind=el.kind, length=el.length)
        else:
            out.extend(thin)
            rep.exact(el.name, el.kind)
        return

    if isinstance(el, Marker):
        out.append(_marker(name, el, flavor))
        rep.exact(el.name, el.kind, code="OK" if flavor == "python" else "MARKER_AS_ZERO_DRIFT")
        return

    if isinstance(el, Instrument):
        if instrument == "beam_monitor":
            out.append(Emit(name, "BeamMonitor", {"backend": "h5"}, 0.0, el.name))
            rep.exact(el.name, el.kind, message="written as an ImpactX BeamMonitor")
        else:
            out.append(_marker(name, el, flavor))
            rep.equivalent("INSTRUMENT_AS_MARKER", Writer.RULES["Instrument"].message,
                           element=el.name, kind=el.kind, family=el.family)
        return

    if isinstance(el, Foil):
        out.append(_marker(name, el, flavor))
        rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind,
                  material=el.material, thickness_kg_per_m2=el.thickness_kg_per_m2)
        return

    if isinstance(el, Taylor):
        rows = [[float(v) for v in row] for row in el.matrix]
        # IR/MAD-X (x, px, y, py, T, pt) -> ImpactX (x, px, y, py, t, pt): t and pt are
        # both sign-flipped, so the map is conjugated with S = diag(1,1,1,1,-1,-1).
        for i in range(6):
            for j in range(6):
                if (i >= 4) != (j >= 4) and rows[i][j]:
                    rows[i][j] = -rows[i][j]
        params = {"ds": el.length, "R": rows}
        params.update(_alignment(el, 0.0, rep))
        out.append(Emit(name, "LinearMap", params, el.length, el.name))
        if any(el.offset):
            rep.lossy("TAYLOR_OFFSET_DROPPED",
                      "ImpactX LinearMap has no constant offset vector", element=el.name,
                      kind=el.kind, offset=list(el.offset))
        else:
            rep.equivalent("TAYLOR_TIME_SIGN",
                           "the map was conjugated with diag(1,1,1,1,-1,-1): ImpactX's t is "
                           "late-positive and its pt is the negative energy deviation",
                           element=el.name, kind=el.kind)
        return

    if isinstance(el, Patch):
        pure_roll = el.tilt and not any((el.x_offset, el.y_offset, el.z_offset, el.x_rot,
                                         el.y_rot))
        pure_shift = el.z_offset and not any((el.x_offset, el.y_offset, el.x_rot, el.y_rot,
                                              el.tilt))
        if pure_roll:
            out.append(Emit(name, "PlaneXYRot", {"angle": _deg(el.tilt)}, 0.0, el.name))
            rep.exact(el.name, el.kind, message="pure roll written as PlaneXYRot")
        elif pure_shift:
            out.append(Emit(name, "Drift", _thick({}, el.z_offset, _nslice_of(el, nslice)), el.z_offset, el.name))
            rep.equivalent("PATCH_AS_DRIFT", "a pure longitudinal patch became a drift",
                           element=el.name, kind=el.kind, z_offset=el.z_offset)
        else:
            out.append(_marker(name, el, flavor))
            rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind)
        return

    if isinstance(el, ReferenceChange):
        out.append(_marker(name, el, flavor))
        rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind,
                  dE_ref_eV=el.dE_ref_eV, energy_eV=el.energy_eV)
        return

    if isinstance(el, Freq):
        rep.exact(el.name, el.kind, message=Writer.RULES["Freq"].message)
        return

    if isinstance(el, Directive):
        rep.dropped(rule.code, rule.message, element=el.name, kind=el.kind,
                    card=el.card, args=list(el.args), role=el.role)
        return

    if isinstance(el, Superposition):
        rep.lossy(rule.code, rule.message, element=el.name, kind=el.kind,
                  children=len(el.children))
        for _offset, child_name in el.children:
            child = lattice.elements.get(child_name)
            if child is None:
                rep.dropped("SUPERPOSITION_CHILD_MISSING",
                            f"superposition child {child_name!r} is not defined",
                            element=el.name, kind=el.kind)
                continue
            _emit_element(p, child, brho, mass_eV, lattice, names, rep, out, flavor=flavor,
                          nslice=nslice, instrument=instrument)
        return

    raise NotImplementedError(f"ImpactX writer has no rule for kind {el.kind!r}")  # pragma: no cover


def _marker(name: str, el: Element, flavor: str) -> Emit:
    """``elements.Marker`` in a Python script; a zero-length ``drift`` in an inputs file
    (ImpactX's inputs parser has no ``marker`` type and aborts on unknown types)."""
    if flavor == "python":
        return Emit(name, "Marker", {}, 0.0, el.name)
    return Emit(name, "Drift", {"ds": 0.0, "nslice": 1}, 0.0, el.name)


def _emit_bend(el: Bend, name: str, brho: float, names: NameMap, rep: FidelityReport,
               out: list[Emit], *, nslice: int, ap: dict, flavor: str) -> None:
    b = el.bend
    align = _alignment(el, 0.0, rep)
    rot = {"rotation": _deg(b.tilt_ref)} if b.tilt_ref else {}
    if "rotation" in align:
        rot = {"rotation": align.pop("rotation") + _deg(b.tilt_ref)}
    k1 = el.multipole.Bn.get(1, 0.0) / brho
    if el.length <= 0.0 or b.angle == 0.0:
        if b.angle:
            out.append(Emit(name, "Multipole",
                            {"multipole": 1, "K_normal": b.angle, "K_skew": 0.0, **align, **rot},
                            0.0, el.name))
            rep.lossy("THIN_BEND_KICK",
                      "a zero-length bend became a thin dipole kick (no edge focusing, no "
                      "path lengthening)", element=el.name, kind=el.kind, angle=b.angle)
        elif el.length > 0.0:
            body = "Quad" if k1 else "Drift"
            params = _thick({**({"k": k1} if k1 else {}), **align, **rot, **ap}, el.length,
                            nslice)
            out.append(Emit(name, body, params, el.length, el.name))
            rep.equivalent("ZERO_ANGLE_BEND",
                           "a bend with zero angle became a straight element",
                           element=el.name, kind=el.kind, k1=k1)
        else:
            out.append(_marker(name, el, flavor))
            rep.lossy("EMPTY_BEND", "zero-length bend with no angle; written as a marker",
                      element=el.name, kind=el.kind)
        return

    rc = el.length / b.angle
    g = 2.0 * b.hgap
    f1 = b.edge_int1
    f2 = b.edge_int1 if b.edge_int2 is None else b.edge_int2
    if b.e1 or (g and f1):
        out.append(Emit(names.derive(name, "e1"), "DipEdge",
                        {"psi": b.e1, "rc": rc, "g": g, "K2": f1, "location": "entry", **rot},
                        0.0, el.name))
    body = "CFbend" if k1 else "Sbend"
    params = _thick({"rc": rc, **({"k": k1} if k1 else {}), **align, **rot, **ap},
                    el.length, nslice)
    out.append(Emit(name, body, params, el.length, el.name))
    if b.e2 or (g and f2):
        out.append(Emit(names.derive(name, "e2"), "DipEdge",
                        {"psi": b.e2, "rc": rc, "g": g, "K2": f2, "location": "exit", **rot},
                        0.0, el.name))
    details = {"rc": rc, "body": body}
    if b.fringe_k2 is not None:
        details["fringe_k2_dropped"] = b.fringe_k2
    rep.exact(el.name, el.kind,
              message=f"{body}(ds={el.length:.10g}, rc={rc:.10g})"
                      + (" + DipEdge" if (b.e1 or b.e2 or (g and (f1 or f2))) else ""))
    rep.entries[-1].details.update(details)


# sanity: the RULES table must mention every IR kind (asserted again in the tests)
assert set(Writer.RULES) == set(ALL_KINDS), sorted(set(ALL_KINDS) - set(Writer.RULES))
