"""Cross-format validation battery: every ordered pair of formats on every deck, both ways.

Three checks per (deck, source format, target format):

``ir``
    read(src) → write(dst) → read(dst): the re-read lattice must carry the same *integrated
    physics* as the source at every non-drift element exit — length, bend angles per plane,
    ∫G·dl (normal and skew), ∫B₂·dl, ∫B₃·dl, thin BnL/BsL per order, ∫Bsol·dl, kicks, RF gain,
    RF voltage, and the reference kinetic energy — to 1e-9 relative, *after* the source has been
    neutralised where the writer's ledger declared a LOSSY/DROPPED entry.  A difference the
    ledger does not explain is a bug.
``fixed``
    write(dst) → read(dst) → write(dst) again must reproduce the text byte for byte
    (invariant I-13 for every format).
``engine``
    when both formats have an engine adapter on this machine, the source deck runs in the source
    engine and the translation in the target engine; cumulative maps are compared in the common
    basis at shared boundaries (:mod:`lattix.oracles.compare`) with the tier implied by the
    ledger: all EXACT → exact tier, any EQUIVALENT → equivalent tier, LOSSY/DROPPED → report only.

The battery is a library (:func:`run_matrix`), a CLI (``lattix crossval``) and a pytest wrapper
(``tests/crossval``).
"""
from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lattix.fidelity import FidelityReport
from lattix.formats.base import FORMATS, read, write
from lattix.ir.elements import Element
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.walk import energy_gain_eV, propagate

# ---------------------------------------------------------------------------
# public decks

PUBLIC = Path(__file__).resolve().parents[1] / "tests" / "data" / "public"

#: (relative path, format, read options) — every public deck the battery can start from
DECKS: list[tuple[str, str, dict]] = [
    ("helix/fodo.madx", "madx", {}),
    ("helix/transport.madx", "madx", {}),
    ("helix/fodo.bmad", "bmad", {}),
    ("helix/fodo_cell.dat", "tracewin", {}),
    ("helix/bend_line.dat", "tracewin", {}),
    ("helix/mebt_line.dat", "tracewin", {}),
    ("helix/dtl_section.dat", "tracewin", {}),
    ("helix/solenoid_channel.dat", "tracewin", {}),
    ("helix/csr_chicane.dat", "tracewin", {}),
    ("helix/halo_fodo.dat", "tracewin", {}),
    ("helix/matching_demo.dat", "tracewin", {}),
    # LightWin's ADS linac (MIT): 142 one-dimensional RF maps with relative phases, 627 elements
    ("lightwin/example.dat", "tracewin", {"species": "proton", "kinetic_energy_eV": 20e6, "frequency_Hz": 352.2e6}),
    ("flame/LS1.lat", "flame", {}),
    ("flame/ALL_lattice.lat", "flame", {}),
    ("flame/TMtest.lat", "flame", {}),
    ("xtrack/psb.seq", "madx", {"species": "proton", "kinetic_energy_eV": 160e6}),
    ("xtrack/elena.seq", "madx", {"species": "proton", "kinetic_energy_eV": 5.3e6}),
    ("pals/fodo.pals.yaml", "pals", {}),
    ("pals/drift_quad_bend.pals.yaml", "pals", {}),
    ("pals/rf_voltage.pals.yaml", "pals", {}),
    ("pals/bend_angle_radius.pals.yaml", "pals", {}),
    ("impactx/solenoid.madx", "madx", {}),
    ("pyorbit3/fodo.lat", "mad8", {"species": "proton", "brho": 5.65737309979}),      # 1 GeV proton
    ("pyorbit3/sis18.lat", "mad8", {"species": "proton", "brho": 5.65737309979}),
    # lattix's own bend decks (BSD-3): pole faces, a negative-angle bend, vertical bends — the RF clock keeps the
    # RF-based writers (IMPACT-Z, IMPACT-T, DYNAC) at the exact tier on an RF-free MAD-X source
    ("lattix/rect_bends.madx", "madx", {"frequency_Hz": 352.21e6}),
    ("lattix/vertical_bends.madx", "madx", {"frequency_Hz": 352.21e6}),
]

#: decks that seed the *derived* sources: each is written to every other format and those files
#: become sources in turn, so formats without a public deck of their own (Elegant, ImpactX,
#: IMPACT-Z, xtrack, lattix JSON) are exercised as sources — and against their engines
DERIVED_BASES: list[tuple[str, str, dict]] = [
    ("helix/fodo.madx", "madx", {}),
    ("helix/transport.madx", "madx", {}),
    ("helix/mebt_line.dat", "tracewin", {}),
    ("helix/solenoid_channel.dat", "tracewin", {}),
    ("helix/csr_chicane.dat", "tracewin", {}),
]


#: read-option key under which a derived deck carries the tier of the write that made it
DERIVATION_TIER = "_derivation_tier"


#: formats whose decks come with data files next to them (IMPACT-T and IMPACT-Z's rfdataN, 1TN.T7)
_SIDE_FILE_FORMATS = frozenset({"impactt", "impactz", "dynac", "opal"})


def derived_decks(workdir: Path, bases: list[tuple[str, str, dict]] | None = None) -> list[tuple[Path, str, dict]]:
    """Write every base deck to every other format; the results are sources for the matrix.
    Each derived deck remembers the tier of the write that produced it (a thin gap that became
    a FLAME drift, an IMPACT-Z short cavity, …): a case starting from it can be no better."""
    out: list[tuple[Path, str, dict]] = []
    d = workdir / "derived"
    d.mkdir(parents=True, exist_ok=True)
    for rel, fmt, opts in bases if bases is not None else DERIVED_BASES:
        lat, _ = read(PUBLIC / rel, fmt, **opts)
        for target in readable_formats():
            if target == fmt:
                continue
            path = d / f"{Path(rel).stem}.{target}{_suffix(target)}"
            if target in _SIDE_FILE_FORMATS:
                # rfdataN / 1TN.T7 files are numbered from 1 per deck: one directory per derived deck
                path = d / f"{Path(rel).stem}.{target}" / f"{Path(rel).stem}.{target}"
                path.parent.mkdir(parents=True, exist_ok=True)
            try:
                rep = write(lat, path, target, strict=False)
            except Exception:  # noqa: BLE001 - a writer that cannot hold the deck is a normal finding elsewhere
                continue
            out.append((path, target, {DERIVATION_TIER: tier_of(rep)}))
    return out


_TIER_RANK = {"exact": 0, "equivalent": 1, "lossy": 2}


def _cap_tier(tier: str, cap: str | None) -> str:
    if not cap:
        return tier
    return tier if _TIER_RANK.get(tier, 2) >= _TIER_RANK.get(cap, 2) else cap


#: engine adapter that reads each format (None: no engine)
ENGINE_FOR_FORMAT: dict[str, str | None] = {
    "madx": "madx", "xtrack": "xtrack", "bmad": "bmad", "elegant": "elegant", "impactx": "impactx",
    "impactz": "impactz", "flame": "flame", "tracewin": "helix", "mad8": None, "pals": None, "lattix": None,
    "scibmad": "scibmad", "cheetah": "cheetah", "pyorbit": "pyorbit", "impactt": "impactt", "ocelot": "ocelot",
    "dynac": "dynac", "synergia": "synergia", "opal": None,
}
#: fallback engines per format, tried in order when the primary one is unavailable (CI has no HELIX)
ENGINE_CANDIDATES: dict[str, tuple[str, ...]] = {"tracewin": ("helix", "lightwin")}
FOLLOWS_P0 = {"helix": True, "bmad": True, "elegant": True, "tracewin": True, "impactx": True, "lightwin": True,
              "cheetah": True, "pyorbit": True, "impactt": True, "ocelot": True, "dynac": True, "synergia": True,
              "impactz": True, "flame": True, "madx": False, "xtrack": False, "scibmad": False}
#: a constant-p0 engine applies every normalised strength at the start rigidity and expands its maps about
#: that momentum: MEASURED 2026-09-06 — helix/dtl_section.dat (p ×1.7) still compares on the transverse block
#: and the dispersion, lightwin/example.dat (20 → 502 MeV, p ×5.6) makes MAD-X's own twiss fail ("open line -
#: error with deltap"). Above this momentum ratio such an engine is not run: report only, with the reason.
P0_RATIO_LIMIT = 2.0


def momentum_ratio(lat: Lattice) -> float:
    """``max(p/p_start)`` a constant-p0 engine's reference would reach along the line (RF gains only)."""
    from lattix.ir.energy_mode import probe_momentum_ratio

    placed = propagate(lat)
    if not placed:
        return 1.0
    ratios = probe_momentum_ratio(placed, lat.reference)
    last = placed[-1]
    if last.ref_out is not None and last.ref_in is not None:
        ratios = [*ratios, ratios[-1] * (last.ref_out.pc_eV / last.ref_in.pc_eV if last.ref_in.pc_eV else 1.0)]
    return max(ratios) if ratios else 1.0


def constant_p0_note(lat: Lattice, engines) -> str | None:
    """The report-only reason when a constant-p0 engine in ``engines`` cannot follow this line, else None."""
    fixed = [e for e in engines if not FOLLOWS_P0.get(e, True)]
    if not fixed:
        return None
    ratio = momentum_ratio(lat)
    if ratio <= P0_RATIO_LIMIT:
        return None
    return (f"{'/'.join(fixed)} keep{'s' if len(fixed) == 1 else ''} p0 constant: the momentum grows ×{ratio:.2g} "
            f"along this line (limit ×{P0_RATIO_LIMIT:g}); not run, report only")

RTOL = 1e-9
#: lattice-level codes after which normalized strengths no longer mean the same thing
_SPECIES_LOSS = ("SPECIES_NOT_REPRESENTABLE", "UNKNOWN_SPECIES", "PALS_SPECIES_UNKNOWN", "SPECIES_ASSUMED")
EXACT_MAP_TOL = 1e-7          # max |ΔR̂cum| relative, exact tier (roundoff over long lines)
#: engines whose output files limit the maps fitted from them (the exact tier is held at that floor)
ENGINE_PRECISION = {"dynac": 5e-5, "tracewin": 1e-6}
#: ledger codes after which the written optics differ from the source by construction (the verdict names them)
_OPTICS_CODES = {
    "IMPACTZ_NO_REF_TILT": "vertical (tilted) bends written in the horizontal plane",
    "PYORBIT_BEND_TILT_DROPPED": "vertical (tilted) bends written in the horizontal plane",
    "IMPACTT_BEND_TILT_DROPPED": "vertical (tilted) bends written in the horizontal plane",
    "BEND_TILT_DROPPED": "the bend tilt is dropped",
    "BEND_TILT_UNSUPPORTED": "the bend tilt is dropped",
    "BEND_FRINGE_DROPPED": "the fringe-field integral (fint·hgap) is dropped",
    "IMPACTZ_NO_FRINGE_K2": "the second fringe coefficient is dropped",
    "IMPACTZ_SINGLE_FINT": "the exit fringe integral is set equal to the entrance one",
    "MULTIPOLE_ORDERS_DROPPED": "higher multipole orders (a bend's gradient, skew terms) are dropped",
}
EQUIV_MAP_TOL = 2e-2          # equivalent tier
EXACT_ENERGY_TOL = 1e-9
EQUIV_ENERGY_TOL = 5e-3

# ---------------------------------------------------------------------------
# integrated-physics profiles

QUANTITIES = ("length", "angle_h", "angle_v", "BsolL", "hkick", "vkick", "gain", "volt", "energy",
              *[f"BnL{k}" for k in range(6)], *[f"BsL{k}" for k in range(6)])
# aliases used in AFFECTS: ∫G·dl is order 1
GL_N, GL_S, B2L, B3L = "BnL1", "BsL1", "BnL2", "BnL3"

#: ledger codes whose effect is limited to some quantities (default for LOSSY/DROPPED: the whole element)
AFFECTS: dict[str, set[str]] = {
    "APERTURE_DROPPED": set(), "APERTURE_APPROXIMATED": set(), "APERTURE_SHAPE": set(),
    "APERTURE_TYPE_UNMODELLED": set(), "APERTURE_PARTIAL": set(), "APERTURE_OFFSET_DROPPED": set(),
    "IMPACTZ_ELLIPSE_AS_SLIT": set(), "COLLIMATOR_TO_MARKER": set(), "PALS_APERTURE_SHAPE_UNSUPPORTED": set(),
    "FINTX_DROPPED": set(), "BEND_FINTX_DROPPED": set(), "IMPACTZ_SINGLE_FINT": set(),
    "BEND_FRINGE_DROPPED": set(), "FRINGE_K2_DROPPED": set(), "PALS_FRINGE_K2_DROPPED": set(),
    "BEND_POLE_CURVATURE_DROPPED": set(), "IMPACTZ_POLE_FACE_CURVATURE": set(),
    "NCELL_DROPPED": set(), "MISALIGN_DROPPED": set(), "IMPACTZ_MISALIGNMENT_DROPPED": set(),
    "KICKER_TILT_DROPPED": set(), "KICKER_TILT_KEPT_NATIVE": set(), "PALS_KICKER_TILT_DROPPED": set(),
    "SKEW_MULTIPOLE_DROPPED": {*[f"BsL{k}" for k in range(6)]},
    "IMPACTZ_SKEW_MULTIPOLE": {*[f"BsL{k}" for k in range(6)]},
    "IMPACTZ_SKEW_QUAD": {"BsL1"},
    "SKEW_COMPONENT_DROPPED": {*[f"BsL{k}" for k in range(6)]},
    "MULTIPOLE_ORDERS_DROPPED": {*[f"BnL{k}" for k in range(1, 6)], *[f"BsL{k}" for k in range(1, 6)]},
    "FOIL_MATERIAL_DROPPED": set(), "FOIL_THICKNESS_UNKNOWN": set(), "FOIL_AS_COMMENT": set(),
    "FOIL_TO_MARKER": set(), "FOIL_DROPPED": set(),
    "INSTRUMENT_PARAMS_DROPPED": set(), "INSTRUMENT_TO_MARKER": set(), "INSTRUMENT_AS_MARKER": set(),
    "DIRECTIVE_DROPPED": set(), "FOREIGN_DIRECTIVE": set(), "DIRECTIVE_AS_COMMENT": set(),
    "PATCH_DROPPED": set(), "PATCH_UNSUPPORTED": set(), "PATCH_NOT_SUPPORTED": set(),
    "TAYLOR_OFFSET_DROPPED": set(), "TAYLOR_DROPPED": set(), "TAYLOR_UNSUPPORTED": set(),
    "REFCHANGE_DROPPED": {"energy", "gain"},
    "IMPACTZ_RF_GAIN_UNKNOWN": {"gain", "volt", "energy"},
    "PYORBIT_MULTIPOLE_AS_CORRECTOR": {"BnL0", "BsL0", "hkick", "vkick"},
    "PYORBIT_QUAD_TILT_DROPPED": {"BnL1", "BsL1"},
    "PYORBIT_GAP_NEEDS_FREQUENCY": {"gain", "volt", "energy"},
    "IMPACTT_MULTIPOLE_AS_KICK": {"BnL0", "BsL0", "hkick", "vkick"},
    "IMPACTT_CAVITY_TO_DRIFT": {"gain", "volt", "energy"},
    "PYORBIT_NO_MULTIPOLE": {*[f"BnL{k}" for k in range(6)], *[f"BsL{k}" for k in range(6)]},
    "FLAME_NO_ENG_DATA_DIR": set(), "FLAME_PER_NUCLEON": set(), "FLAME_SOURCE_ADDED": set(),
    "DEFINITION_NOT_IN_LINE": set(), "ELEGANT_PHASE_FOR_SPECIES": set(),
    "CONST_P0": set(), "CONST_P0_LOCAL_RIGIDITY": set(), "CONST_P0_START_RIGIDITY": set(),
    "THIN_CAVITY_NO_RF_FOCUSING": set(), "THIN_CAVITY_TRAVELING_WAVE": set(), "RFCA_DEFAULT_FREQ": set(),
    "RF_FREQUENCY_UNKNOWN": set(), "IMPACTZ_NO_FREQUENCY": set(), "RFCAVITY_GAIN_UNKNOWN": set(),
    "ZERO_ANGLE_BEND_AS_DRIFT": set(), "IMPACTZ_RF_FORM_FACTOR": set(), "THIN_GAP_AS_SHORT_CAVITY": set(),
    "PALS_TTF_DROPPED": set(), "RF_FREQUENCY_MISSING": set(),
    "OCELOT_SKEW_MULTIPOLE_DROPPED": {*[f"BsL{k}" for k in range(6)]}, "KICKER_SPLIT_HV": set(),
    "INSTRUMENT_AS_MONITOR": set(), "THICK_MULTIPOLE_SPLIT": set(), "REFCHANGE_AS_MATRIX": set(),
    "NCELLS_AS_CAVITY": set(), "TAYLOR_BASIS_OCELOT": set(), "THIN_GAP_PAD_DRIFT": set(),
    "OCELOT_ELECTRON_ONLY": set(), "APERTURE_CONTINUOUS_AT_ENDS": set(),
    "MULTIPOLE_AS_STEER": set(), "COLLIMATOR_AS_REJECT": set(), "INSTRUMENT_AS_EMIT": set(),
    "FOIL_AS_STRIPPER": set(), "PATCH_AS_ALINER": set(), "REFCHANGE_AS_NREF": set(), "MISALIGN_AS_ALINER": set(),
    "QUAD_TILT_AS_TWQA": set(), "APERTURE_AS_REJECT": set(), "KICKER_THIN_AT_ENTRANCE": set(),
    "THICK_CAVITY_AS_BUNCHER": set(), "NCELLS_AS_CAVNUM": set(), "FM_AS_CAVNUM": set(), "BUNCHER_HARMONIC": set(),
    "DYNAC_FREQUENCY_ASSUMED": set(), "DYNAC_NCELLS_PARAMS": set(), "PATCH_ROTATION_DROPPED": set(),
    "RF_RAW_PHASE_AS_SYNC": set(), "DYNAC_DIRECTIVE_KEPT": set(),
    "CAVSC_AS_GAP": {"gain", "volt", "energy"}, "NCELLS_FROM_CAVNUM": set(), "FM_READ_AS_CAVITY": set(),
    "CHANGREF_AS_PATCH": set(),
    "APERTURE_AS_ATTRIBUTE": set(), "CONST_P0_BEND_K0": set(), "TAYLOR_BASIS_SYNERGIA": set(),
    "RBEND_AS_SECTOR": set(), "TAYLOR_THIN_PLUS_DRIFT": set(), "CONST_P0_BEND_UNDERBENT": set(),
    "MULTIPOLE_AS_SHORT": set(), "OPAL_BEND_DEFAULT_PROFILE": set(), "OPAL_SOLENOID_MAP": set(),
    "OPAL_CAVITY_MAP": set(), "FM_AS_OPAL_MAP": set(), "OPAL_CAVITY_TO_DRIFT": {"gain", "volt", "energy"},
    "SOLENOID_TO_MARKER": set(), "OPAL_GAP_DRIFT_INSERTED": set(), "OPAL_LAG_AS_SYNC_PHASE": set(),
    "UNSUPPORTED_OPAL_ELEMENT": set(), "FM_STATIC_B_DROPPED": set(),
    # a bend written in the wrong plane: its angle moves between the horizontal and the vertical sums
    "IMPACTZ_NO_REF_TILT": {"angle_h", "angle_v"}, "PYORBIT_BEND_TILT_DROPPED": {"angle_h", "angle_v"},
    "IMPACTT_BEND_TILT_DROPPED": {"angle_h", "angle_v"}, "BEND_TILT_DROPPED": {"angle_h", "angle_v"},
    "BEND_TILT_UNSUPPORTED": {"angle_h", "angle_v"},
}

#: codes whose model moves an element boundary by up to this many metres (short cavities)
FUZZY_S: dict[str, float] = {"THIN_GAP_AS_SHORT_CAVITY": 2e-2, "THICK_CAVITY_AS_SHORTRF": 2e-2}

_THIN_BENDLESS = ("Marker", "Instrument", "Directive", "Freq", "Patch", "ReferenceChange")


def rotated(bn: dict, bs: dict, tilt: dict, base_tilt: float = 0.0) -> tuple[dict, dict]:
    """Normal/skew components in the lab frame: a 2(k+1)-pole tilted by t rotates by (k+1)·t."""
    n_out: dict[int, float] = {}
    s_out: dict[int, float] = {}
    for k in set(bn) | set(bs):
        t = tilt.get(k, 0.0) + base_tilt
        a, b = bn.get(k, 0.0), bs.get(k, 0.0)
        c, sn = math.cos((k + 1) * t), math.sin((k + 1) * t)
        n_out[k] = a * c - b * sn
        s_out[k] = a * sn + b * c
    return n_out, s_out


def _static_contrib(e: Element, length: float, c: dict[str, float]) -> None:
    """Add one element's static fields over ``length`` to ``c`` (multipoles, bends, solenoids, kickers and the
    integrals a static map's hard-edge replacement preserves)."""
    k = e.kind
    mp = getattr(e, "multipole", None)
    if mp is not None:
        base = (e.bend.tilt_ref if k == "Bend" else 0.0) + (e.shift.tilt if e.shift is not None else 0.0)
        bn, bs = rotated(mp.Bn, mp.Bs, mp.tilt, base)        # thick: field × length
        for n, v in bn.items():
            if n < 6:
                c[f"BnL{n}"] += v * length
        for n, v in bs.items():
            if n < 6:
                c[f"BsL{n}"] += v * length
        bnl, bsl = rotated(mp.BnL, mp.BsL, mp.tilt, base)    # thin: integrated already
        for n, v in bnl.items():
            if n < 6:
                c[f"BnL{n}"] += v
        for n, v in bsl.items():
            if n < 6:
                c[f"BsL{n}"] += v
    if k == "Bend":
        b = e.bend
        c["angle_h"] += b.angle * math.cos(b.tilt_ref)
        c["angle_v"] += b.angle * math.sin(b.tilt_ref)
    elif k == "Solenoid":
        c["BsolL"] += e.solenoid.Bsol_T * length
    elif k == "Kicker":
        c["hkick"] += e.hkick
        c["vkick"] += e.vkick
    elif k == "FieldMap":
        # a static magnetic map integrated by the reader: the quantities its hard-edge replacement
        # preserves (lattix.ir.fieldmap.replacement_for: ∫B for a solenoid map, ∫G for a quadrupole map)
        s = ((e.meta or {}).get("map_summary") or {})
        if s.get("kind") == "solenoid":
            c["BsolL"] += float(s.get("int_Bz_Tm") or 0.0)
        elif s.get("kind") == "quad":
            c["BnL1"] += float(s.get("int_Gz_Tm_per_m") or 0.0)


def contrib(p: Placed, elements: Mapping[str, Element] | None = None) -> dict[str, float]:
    """This placed element's additive contribution to every cumulative quantity (energy is a state).

    ``elements`` (the lattice's definitions) lets a ``Superposition`` count the static fields of its
    unplaced children (a PALS ``UnionEle`` holding a hard-edge solenoid, a TraceWin map cluster).
    """
    e = p.element
    c = dict.fromkeys(QUANTITIES, 0.0)
    c["length"] = p.length
    k = e.kind
    _static_contrib(e, p.length, c)
    if k == "Superposition" and elements:
        for _offset, name in e.children:
            child = elements.get(name)
            if child is not None:
                _static_contrib(child, float(child.length), c)
    if k in ("RFCavity", "FieldMap", "NCells", "RFQCell", "Superposition"):
        c["gain"] = energy_gain_eV(e, p.ref_in) if p.ref_in is not None else 0.0
        volt = float(getattr(getattr(e, "rf", None), "voltage_V", 0.0) or 0.0)
        if k == "FieldMap":
            # the voltage a derived cavity carries is the map's V_c (dE = V_c·cos φs by
            # construction, lattix.ir.fieldmap._cavity_numbers), not the fixed-β V_eff of RFP
            s = ((e.meta or {}).get("map_summary") or {})
            volt = float(s.get("v_c_V") or volt or 0.0)
        c["volt"] = volt
    return c


_contrib = contrib          # the names the battery used before they became public
_rotated = rotated


@dataclass
class Profile:
    names: list[str]
    kinds: list[str]
    s_out: np.ndarray
    cum: dict[str, np.ndarray]          # cumulative at each exit
    energy: np.ndarray                  # reference kinetic energy at each exit
    drift_like: list[bool] = field(default_factory=list)   # contributes nothing but length (may coalesce)
    e_dropped: np.ndarray | None = None  # neutralised reference gain accumulated up to each exit [eV]


def profile(lat: Lattice, neutral: dict[str, set[str]] | None = None) -> Profile:
    """Cumulative integrated quantities at every element exit.

    ``neutral`` maps element names to the quantities whose contribution must be dropped
    (``{"*"}`` = everything, i.e. the element behaves as a drift of the same length).
    """
    neutral = neutral or {}
    placed = propagate(lat)
    n = len(placed)
    cum = {q: np.zeros(n) for q in QUANTITIES if q != "energy"}
    energy = np.zeros(n)
    run = dict.fromkeys(cum, 0.0)
    e_drop = 0.0
    e_dropped = np.zeros(n)
    drift_like: list[bool] = []
    for i, p in enumerate(placed):
        c = contrib(p, lat.elements)
        drop = neutral.get(p.name, set())
        drift_like.append(all(v == 0.0 for q, v in c.items() if q != "length" and not ("*" in drop or q in drop)))
        for q in cum:
            if q == "length" or not ("*" in drop or q in drop):
                run[q] += c[q]
            cum[q][i] = run[q]             # a neutralised element still carries the running value
        if ("*" in drop or "gain" in drop) and p.ref_out is not None and p.ref_in is not None:
            e_drop += p.ref_out.kinetic_energy_eV - p.ref_in.kinetic_energy_eV
        energy[i] = (p.ref_out.kinetic_energy_eV if p.ref_out else float("nan")) - e_drop
        e_dropped[i] = e_drop
    return Profile([p.name for p in placed], [p.element.kind for p in placed],
                   np.array([p.s_out for p in placed]), cum, energy, drift_like, e_dropped)


def neutral_set(report: FidelityReport) -> dict[str, set[str]]:
    """Element → quantities the writer's ledger says were lost (LOSSY/DROPPED entries)."""
    out: dict[str, set[str]] = {}
    for e in report.entries:
        if e.cls.value not in ("LOSSY", "DROPPED") or not e.element:
            continue
        affected = AFFECTS.get(e.code, {"*"})
        out.setdefault(e.element, set()).update(affected)
    return out


@dataclass
class IRDiff:
    ok: bool
    worst: float = 0.0
    problems: list[str] = field(default_factory=list)
    n_boundaries: int = 0


def _last_per_s(s_out: np.ndarray, tol: float = 1e-9) -> list[int]:
    """Indices of the last element ending at each distinct position (state after everything there)."""
    out: list[int] = []
    for i in range(len(s_out)):
        if i + 1 < len(s_out) and abs(s_out[i + 1] - s_out[i]) <= tol:
            continue
        out.append(i)
    return out


#: absolute floors for the relative comparison: a cavity at the zero crossing has a gain that is pure
#: roundoff of V·cos(φ) (nano-eV on a megavolt), not a translation difference
_FLOOR = {"gain": 1e3, "volt": 1e3}


def compare_profiles(src: Profile, dst: Profile, *, rtol: float = RTOL, skip_energy: bool = False,
                     skip: set[str] | None = None, fuzzy: dict[str, float] | None = None,
                     max_problems: int = 12) -> IRDiff:
    """Compare cumulative quantities at every distinct position of ``src`` that ``dst`` also reaches.

    Both sides are sampled *after* everything located at that position, so thin elements that
    share a position with the end of another element compare correctly.  A position of ``src``
    that ``dst`` lacks is a problem unless only drifts/markers end there.
    """
    problems: list[str] = []
    skip = skip or set()
    fuzzy = fuzzy or {}
    worst = 0.0
    n = 0
    dst_last = _last_per_s(dst.s_out)
    dst_s = dst.s_out[dst_last]
    j = 0
    for i in _last_per_s(src.s_out):
        s = src.s_out[i]
        tol_s = max([1e-9, *[fuzzy.get(src.names[m], 0.0) for m in range(len(src.s_out))
                             if abs(src.s_out[m] - s) <= 1e-9]])
        while j < len(dst_s) and dst_s[j] < s - tol_s:
            j += 1
        # several dst boundaries may sit inside a fuzzy window (a short cavity centred on a thin
        # gap): take the first one at or after s, i.e. the state once the element has acted
        if tol_s > 1e-9:
            jj = j
            while jj < len(dst_s) and dst_s[jj] < s - 1e-9:
                jj += 1
            if jj < len(dst_s) and abs(dst_s[jj] - s) <= tol_s:
                j = jj
        if j >= len(dst_s) or abs(dst_s[j] - s) > tol_s:
            here = [m for m in range(len(src.s_out)) if abs(src.s_out[m] - s) <= tol_s]
            kinds_here = {src.kinds[m] for m in here}
            if not all(src.drift_like[m] if src.drift_like else src.kinds[m] == "Drift" for m in here) and \
                    not kinds_here <= {"Drift", *_THIN_BENDLESS}:
                problems.append(f"no boundary at s={s:.9g} for {sorted(kinds_here)} ({src.names[i]!r})")
            continue
        jj = dst_last[j]
        n += 1
        moved = abs(dst_s[j] - s) > 1e-9          # a fuzzy match: the boundary itself moved
        for q, arr in src.cum.items():
            if q in skip or (moved and q == "length"):
                continue
            a, b = arr[i], dst.cum[q][jj]
            scale = max(_FLOOR.get(q, 1.0), abs(a), abs(b))
            d = abs(a - b) / scale
            worst = max(worst, d)
            if d > rtol and len(problems) < max_problems:
                problems.append(f"{q} at s={s:.6g} ({src.names[i]}): src {a:.12g} vs dst {b:.12g}")
        if not skip_energy:
            a, b = src.energy[i], dst.energy[jj]
            if np.isfinite(a) and np.isfinite(b):
                d = abs(a - b) / max(1.0, abs(a))
                worst = max(worst, d)
                if d > rtol and len(problems) < max_problems:
                    problems.append(f"energy at s={s:.6g} ({src.names[i]}): src {a:.12g} vs dst {b:.12g} eV")
    if n == 0:
        problems.append("no shared boundaries")
    return IRDiff(ok=not problems, worst=worst, problems=problems, n_boundaries=n)


# ---------------------------------------------------------------------------
# one (deck, src, dst) case

@dataclass
class CaseResult:
    deck: str
    src: str
    dst: str
    ledger: dict[str, int] = field(default_factory=dict)
    codes: dict[str, int] = field(default_factory=dict)
    ir_ok: bool | None = None
    ir_worst: float | None = None
    ir_problems: list[str] = field(default_factory=list)
    fixed_ok: bool | None = None
    fixed_note: str = ""
    tier: str = ""
    engine_ok: bool | None = None
    engine_metric: float | None = None
    engine_energy: float | None = None
    engine_note: str = ""
    error: str = ""
    seconds: float = 0.0


def _read_options(fmt: str, species: str | None) -> dict:
    """Pass the source species to readers whose format carries none."""
    spec = FORMATS[fmt]
    try:
        params = inspect.signature(spec.reader().read).parameters
    except (TypeError, ValueError):
        return {}
    if species is not None and "species" in params:
        return {"species": species}
    return {}


def tier_of(report: FidelityReport) -> str:
    classes = {e.cls.value for e in report.entries}
    if classes & {"LOSSY", "DROPPED"}:
        return "lossy"
    if "EQUIVALENT" in classes:
        return "equivalent"
    return "exact"


@dataclass
class RoundTripSettings:
    """What the ledgers say the IR round trip may not be held to (the battery's step (a) rules)."""

    neutral: dict[str, set[str]] = field(default_factory=dict)   # element -> quantities the ledgers lost
    skip: set[str] = field(default_factory=set)                   # strengths suspended by a species loss
    skip_energy: bool = False                                     # CONST_P0 / REFCHANGE_DROPPED / species loss
    fuzzy: dict[str, float] = field(default_factory=dict)         # element -> boundary tolerance [m]
    species_loss: bool = False


def roundtrip_settings(rep_write: FidelityReport, rep_read: FidelityReport | None = None) -> RoundTripSettings:
    """The neutral set, suspended quantities, energy skip and fuzzy boundaries for a written deck read
    back (writer ledger ``rep_write``, read-back ledger ``rep_read``)."""
    codes = rep_write.codes()
    neutral = neutral_set(rep_write)
    read_codes: dict[str, int] = {}
    if rep_read is not None:
        read_codes = rep_read.codes()
        # LOSSY reader entries on the way back also neutralise (what the target format cannot say)
        for e in rep_read.entries:
            if e.cls.value in ("LOSSY", "DROPPED") and e.element:
                neutral.setdefault(e.element, set()).update(AFFECTS.get(e.code, {"*"}))
    skip_energy = any(c in codes for c in ("CONST_P0", "REFCHANGE_DROPPED"))
    skip: set[str] = set()
    species_loss = any(c in codes for c in _SPECIES_LOSS) or any(c in read_codes for c in _SPECIES_LOSS[:-1])
    if species_loss:
        # the target cannot name this species: every normalized strength changes meaning
        skip = {q for q in QUANTITIES if q not in ("length", "angle_h", "angle_v", "hkick", "vkick")}
        skip_energy = True
    fuzzy = {e.element: FUZZY_S[e.code] for e in rep_write.entries if e.code in FUZZY_S and e.element}
    return RoundTripSettings(neutral, skip, skip_energy, fuzzy, species_loss)


@dataclass
class RoundTrip:
    diff: IRDiff
    tier: str
    settings: RoundTripSettings


def ir_roundtrip(lat: Lattice, rep_write: FidelityReport, lat2: Lattice, rep_read: FidelityReport, *,
                 derivation_tier: str | None = None) -> RoundTrip:
    """The battery's IR round-trip verdict for ``lat`` written (``rep_write``) and read back as ``lat2``
    (``rep_read``): the same neutralisation, suspension and fuzzy boundaries as :func:`run_case`."""
    st = roundtrip_settings(rep_write, rep_read)
    tier = _cap_tier(tier_of(rep_write), derivation_tier)
    if st.species_loss:
        tier = "lossy"
    diff = compare_profiles(profile(lat, st.neutral), profile(lat2, st.neutral), skip_energy=st.skip_energy,
                            skip=st.skip, fuzzy=st.fuzzy)
    return RoundTrip(diff, tier, st)


def run_case(deck: Path, src: str, dst: str, workdir: Path, *, engines: bool = False,
             engine_cache: dict | None = None, read_options: dict | None = None) -> CaseResult:
    t0 = time.perf_counter()
    res = CaseResult(str(deck), src, dst)
    workdir.mkdir(parents=True, exist_ok=True)
    read_options = dict(read_options or {})
    derivation_tier = read_options.pop(DERIVATION_TIER, None)
    try:
        lat, _ = read(deck, src, **read_options)
        species = lat.reference.species
        out = workdir / f"{deck.stem}.{src}.to.{dst}{_suffix(dst)}"
        rep = write(lat, out, dst, strict=False)
        res.ledger, res.codes, res.tier = rep.counts, rep.codes(), _cap_tier(tier_of(rep), derivation_tier)
        # (a) IR round trip
        lat2, rep2 = read(out, dst, **_read_options(dst, species))
        rt = ir_roundtrip(lat, rep, lat2, rep2, derivation_tier=derivation_tier)
        res.tier = rt.tier
        res.ir_ok, res.ir_worst, res.ir_problems = rt.diff.ok, rt.diff.worst, rt.diff.problems
        # (c) fixed point
        out2 = workdir / f"{deck.stem}.{src}.to.{dst}.again{_suffix(dst)}"
        write(lat2, out2, dst, strict=False)
        a, b = _normalised(out.read_bytes()), _normalised(out2.read_bytes())
        res.fixed_ok = a == b
        if not res.fixed_ok:
            la, lb = a.decode("latin-1").splitlines(), b.decode("latin-1").splitlines()
            first = next((k for k, (x, y) in enumerate(zip(la, lb, strict=False)) if x != y), min(len(la), len(lb)))
            x = la[first][:80] if first < len(la) else "<eof>"
            y = lb[first][:80] if first < len(lb) else "<eof>"
            res.fixed_note = f"first difference at line {first + 1}: {x!r} vs {y!r}"
        # (b) engines
        if engines:
            _engine_check(res, deck, src, out, dst, lat, workdir, engine_cache if engine_cache is not None else {})
    except Exception as exc:  # noqa: BLE001 - the battery reports, the test asserts
        res.error = f"{type(exc).__name__}: {exc}"[:300]
    res.seconds = time.perf_counter() - t0
    return res


def _canon_float(m) -> str:
    try:
        return f"{float(m.group(0)):.12g}"
    except ValueError:  # pragma: no cover
        return m.group(0)


def _normalised(raw: bytes) -> bytes:
    """Text as physics: full-line comments (headers, notes, dropped-element remarks) go, floats are
    rounded to 12 significant digits, JSON is compared structurally with sorted keys."""
    import json
    import re

    text = raw.decode("latin-1")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(text)
            obj = _round_json(obj)
            for k in ("source_format", "source_file", "lattice", "written_by", "line", "original_type"):
                _scrub(obj, k)          # provenance: where an element was read from, not physics
            return json.dumps(obj, sort_keys=True).encode("latin-1")
        except ValueError:
            pass
    lines = []
    for line in text.splitlines():
        t = line.strip()
        if t.startswith(("!", ";", "#")) or not t:
            continue
        line = re.sub(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])", _canon_float, line)
        line = re.sub(r"written by lattix [0-9.]+ from [^\"']*", "written by lattix", line)
        lines.append(line.replace(" ", "").replace("\t", "").lower())   # spacing and case carry no physics
    return "\n".join(lines).encode("latin-1")


def _round_json(obj, key: str | None = None):
    if isinstance(obj, float):
        return float(f"{obj:.12g}")
    if isinstance(obj, str) and key == "value0":          # Synergia's lazy attributes are numbers in strings
        try:
            return f"{float(obj):.12g}"
        except ValueError:
            return obj
    if isinstance(obj, list):
        return [_round_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_json(v, k) for k, v in obj.items()}
    return obj


def _scrub(obj, key: str) -> None:
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (str, int, float)) and not isinstance(obj[key], bool):
            del obj[key]            # absent on one side, present on the other: still no physics
        for v in obj.values():
            _scrub(v, key)
    elif isinstance(obj, list):
        for v in obj:
            _scrub(v, key)


def _suffix(fmt: str) -> str:
    s = FORMATS[fmt].suffixes[0]
    return s if s.startswith(".") else ""


def beam_from_lattice(lat: Lattice):
    """A :class:`BeamSpec` for the engines (named species only)."""
    from lattix.oracles.base import SPECIES, BeamSpec

    sp = lat.reference.species
    name = next((k for k, (m, q) in SPECIES.items() if abs(m - sp.mass_eV) / m < 1e-6 and q == sp.charge), None)
    if name is None:
        raise RuntimeError(f"species {sp.name!r} not known to the engines")
    return BeamSpec(name, lat.reference.kinetic_energy_eV, lat.reference.rf_frequency_Hz)


_beam = beam_from_lattice


def _has_relative_phase_maps(lat: Lattice) -> bool:
    return any(e.kind == "FieldMap" and getattr(e.rf, "frequency_Hz", None) and not e.rf.phase_is_sync
               and ((e.meta or {}).get("map_summary") or {}).get("kind") in ("rf", "cavity")
               for e in lat.elements.values())


def _has_fringe_bends(lat: Lattice) -> bool:
    for e in lat.elements.values():
        if e.kind == "Bend":
            b = e.bend
            if b.hgap and (b.edge_int1 or (b.edge_int2 or 0.0)):
                return True
    return False


def pick_engine(fmt: str) -> str | None:
    """The format's engine, or the first available fallback from :data:`ENGINE_CANDIDATES`."""
    from lattix.oracles import get_oracle

    primary = ENGINE_FOR_FORMAT.get(fmt)
    for name in ENGINE_CANDIDATES.get(fmt, ()):
        if get_oracle(name).available()[0]:
            return name
    return primary


_pick_engine = pick_engine


@dataclass
class EngineVerdict:
    """What the battery concludes from one engine pair on one deck (:func:`engine_verdict`)."""

    ok: bool
    tier: str
    metric: float                       # max |ΔR̂cum| over the blocks the pair is held to
    metric_rel: float
    energy_rel: float
    blocks_used: set[str]
    map_tol: float | None
    energy_tol: float | None
    energy_checked: bool
    floor: float
    note: str                           # the battery's engine_note text (caveats + the comparison row)
    notes: list[str] = field(default_factory=list)   # the caveats one by one


def engine_verdict(pc, ea: str, eb: str, lat: Lattice, tier: str, codes: dict[str, int], ra, rb, *,
                   note: str = "") -> EngineVerdict:
    """The battery's verdict on ``compare_pair(ra, rb)``: which map blocks the pair can be held to, the
    measured engine limits that cap the tier (docs/oracles.md), the tolerances of the tier and the
    engine-precision floor.  ``note`` is any caveat recorded before the engines ran."""
    if pc.n_shared == 0:
        return EngineVerdict(False, tier, pc.max_rcum_rel, pc.max_rcum_rel, pc.energy_rel, set(), None, None,
                             False, 0.0, "no shared boundaries", ["no shared boundaries"])
    notes: list[str] = []

    def caveat(text: str, *, append: bool = False) -> None:
        nonlocal note
        note = ((note or "") + text) if append else text
        notes.append(text.strip("; ").strip())

    # which blocks the pair can be held to (measured engine limits, docs/oracles.md):
    # * HELIX's bend map has no path-length coupling (R51/R52 = 0) and its own R56;
    # * the thin-cavity longitudinal row differs between constant-p0 and p0-following engines and
    #   Elegant's RFCA matrix (its ultra-relativistic phase slip), so with RF only the transverse
    #   block and the dispersion column are compared across such pairs
    blocks = {"T4x4", "disp", "path", "R56", "R5x_z", "E_row", "z_col"}
    has_bends = any(e.kind == "Bend" for e in lat.elements.values())
    has_rf = any(e.kind in ("RFCavity", "FieldMap", "NCells", "RFQCell") for e in lat.elements.values())
    if "helix" in (ea, eb) and has_bends:
        blocks -= {"path", "R56"}
        caveat("HELIX bend map has no path-length row (R51/R52) and no dispersive R56 (known HELIX limit): "
               "path/R56 not compared; ", append=True)
        fixed = any(bool((r.meta or {}).get("dipole_negative_bend_fixed")) for r in (ra, rb)
                    if getattr(r, "engine", None) == "helix")
        if any(e.kind == "Bend" and e.bend.angle < 0 for e in lat.elements.values()) and tier != "lossy" \
                and not fixed:
            # measured 2026-09-04 (docs/oracles.md): HELIX's BEND body evaluated the sector map at the signed
            # angle with rho > 0 (R12 < 0, R21 > 0 on a negative bend). Fixed in HELIX d3f281a (2026-09-06):
            # psb.seq then agrees on T4x4/disp to 2.3e-13 — the oracle reports whether its tree has the fix
            tier = "lossy"
            caveat("HELIX negative-angle bend body (HELIX before d3f281a, report only); ")
    if "impactt" in (ea, eb):
        # MEASURED (docs/oracles.md, Phase 5.5): IMPACT-T's dipole (getfldt_Dipole) bends the whole
        # bunch by the reference angle — no pole-face focusing, R21 = R26 = 0 — so bend decks are report
        # only; its solenoid is a generated (r, z) table (9e-7 at a 1 ps step) and its thin gaps are
        # short profile cavities with the profile's own RF focusing (mebt_line 8.1e-3) — Equivalent tier
        if has_bends and tier != "lossy":
            tier = "lossy"
            caveat("IMPACT-T dipole model (no pole-face focusing, report only); ")
        elif tier == "exact" and any(e.kind in ("Solenoid", "RFCavity", "FieldMap", "NCells")
                                     for e in lat.elements.values()):
            tier = "equivalent"
            caveat("IMPACT-T solenoid table / RF profile: engine models differ; ")
    if "ocelot" in (ea, eb) and lat.reference.species.name.lower() not in ("electron", "positron") \
            and tier != "lossy":
        # MEASURED (docs/oracles.md, Phase 5.7): every Ocelot map divides by the electron mass — the
        # transverse blocks of a proton deck are right (normalized strengths), the longitudinal ones and
        # the cavity model are an electron's: report only for any other species
        tier = "lossy"
        caveat(f"Ocelot is electron-only ({lat.reference.species.name} deck, report only); ")
    elif "ocelot" in (ea, eb) and tier == "exact" and any(
            e.kind in ("RFCavity", "FieldMap", "NCells") for e in lat.elements.values()):
        # Ocelot's Cavity (Rosenzweig–Serafini edges + body) against the thin-gap or field-map models
        tier = "equivalent"
        caveat("Ocelot cavity model (RF focusing of its own): engine models differ; ")
    if "synergia" in (ea, eb) and tier != "lossy" and any(e.kind == "Solenoid" for e in lat.elements.values()):
        # MEASURED (docs/oracles.md, Phase 5.9): Synergia's ff_solenoid passes (ksl, ks) to a body that takes
        # (ks, ksl) — the rotation angle is ks and the displacement is divided by ks·L: solenoid decks are
        # report only until the upstream fix
        tier = "lossy"
        caveat("Synergia solenoid body (ks/ksl swapped upstream, report only); ")
    if "dynac" in (ea, eb) and tier == "exact" and has_rf:
        # MEASURED (docs/oracles.md, Phase 5.8): DYNAC's BUNCHER applies the RF defocusing with the mid-gap
        # velocity (TraceWin/HELIX: the entrance one) and CAVNUM integrates a generated profile against its
        # own crest (7e-4): engine models differ on every RF element
        tier = "equivalent"
        caveat("DYNAC buncher / CAVNUM models: engine models differ; ")
    if "dynac" in (ea, eb) and tier == "exact" and has_bends:
        # MEASURED 2026-09-06 (docs/oracles.md, bend faces): the maps fitted from DYNAC's 6-digit dumps drift
        # by ~1e-6 per BMAGNET (9.5e-5 over the 4-bend csr_chicane vs Elegant, 1.7e-3 over the 36 bends of the
        # PIP-II BTL vs HELIX): engine precision, not a translation difference
        tier = "equivalent"
        caveat("DYNAC bend maps (6-digit dumps, ~1e-6 per bend): engine precision; ")
    if "synergia" in (ea, eb) and has_bends:
        faces = any(e.kind == "Bend" and (e.bend.e1 or e.bend.e2) for e in lat.elements.values())
        if faces and lat.reference.species.charge < 0:
            # MEASURED 2026-09-06 (docs/oracles.md, bend faces): Synergia's sbend pole-face focusing follows the
            # sign of the charge — a 0.1 rad bend with 0.05 rad faces agrees with MAD-X/Bmad to 1.3e-8 for a
            # proton and differs by 2.7e-2 for H⁻ (the sector bend agrees for both): report only, upstream bug
            tier = "lossy"
            caveat("Synergia pole-face focusing flips with a negative species (known Synergia limit, report only); ")
        elif tier == "exact":
            # MEASURED 2026-09-06 (docs/oracles.md, bend faces): Synergia's sbend differs from MAD-X by 5.7e-6 on
            # the csr_chicane (sector bends) and by 4.5e-4 with a fringe field (fint·hgap, its own fringe model)
            tier = "equivalent"
            caveat("Synergia sbend body / fringe model (5.7e-6 sector, 4.5e-4 with fint·hgap): engine models differ; ")
    # LOSSY codes that change the optics itself: the pair is report only (lossy tier) — say why
    for code, why in _OPTICS_CODES.items():
        n = codes.get(code, 0)
        if n:
            caveat(f"{code} ×{n}: {why} — the maps may differ from the first such element; ")
    fm_derived = any(c.startswith("FM_") for c in codes)
    if has_rf and (FOLLOWS_P0.get(ea, True) != FOLLOWS_P0.get(eb, True) or "elegant" in (ea, eb) or fm_derived):
        # a field map integrated by one engine and a cavity element in the other agree on the
        # transverse block and the dispersion, never on the longitudinal model
        blocks = {"T4x4", "disp"}
    metric = max(pc.blocks[b] for b in blocks if b in pc.blocks)
    scale = max(1.0, pc.max_rcum_abs / max(pc.max_rcum_rel, 1e-300)) if pc.max_rcum_rel else 1.0
    metric_rel = metric / scale
    for r_ in (ra, rb):
        missing = (len(r_.meta.get("substituted", [])) + len(r_.meta.get("dropped", []))
                   + int(r_.meta.get("solenoids_as_drifts", 0) or 0))
        if r_.engine == "lightwin" and missing and tier != "lossy":
            # LightWin propagates elements without an Envelope3D model (EDGE, THIN_STEERING …) as
            # drifts of their length and skips keywords it does not implement (GAP, NCELLS …): the
            # comparison cannot be held to any tier — report only
            tier = "lossy"
            caveat(f"LightWin has no model for {missing} element(s) (report only); ")
    if "scibmad" in (ea, eb) and tier == "exact" and any(
            e.kind == "RFCavity" and e.length > 0 and (e.rf.voltage_V or e.rf.gradient_V_per_m)
            for e in lat.elements.values()):
        # SciBmad's thick RFCavity applies its own transverse RF focusing (measured, docs/oracles.md);
        # MAD-X, xtrack and Elegant kick at the centre only
        tier = "equivalent"
        caveat("SciBmad thick-cavity RF focusing: engine models differ; ")
    if "scibmad" in (ea, eb) and _has_fringe_bends(lat) and tier != "lossy":
        # BeamTracking 0.5 stores edge1_int/edge2_int but cannot track them (the worker zeroes them):
        # the fringe correction is missing entirely, not modelled differently — report only
        tier = "lossy"
        caveat("SciBmad 0.5 does not track fringe integrals (report only); ")
    if tier == "exact" and _has_fringe_bends(lat) and ea != eb:
        # the codes agree on the parameters but not on the fringe-field model: MAD-X and Bmad
        # differ at O(ψ²) in the fint·hgap correction (8.6e-4 on ELENA's 60° bend, 0 without it)
        tier = "equivalent"
        caveat("fringe-integral bends: engine models differ; ")
    floor = max(ENGINE_PRECISION.get(ea, 0.0), ENGINE_PRECISION.get(eb, 0.0))
    energy_checked = bool(FOLLOWS_P0.get(ea, True) and FOLLOWS_P0.get(eb, True))
    map_tol: float | None
    energy_tol: float | None
    if tier == "exact":
        map_tol, energy_tol = max(EXACT_MAP_TOL, floor), max(EXACT_ENERGY_TOL, floor)
        map_ok = metric_rel <= map_tol
        e_ok = (not energy_checked) or pc.energy_rel <= energy_tol
    elif tier == "equivalent":
        map_tol, energy_tol = EQUIV_MAP_TOL, EQUIV_ENERGY_TOL
        map_ok = metric_rel <= map_tol
        e_ok = (not energy_checked) or pc.energy_rel <= energy_tol
        if fm_derived and not map_ok:
            # a field map integrated by one engine against a derived cavity element in the other:
            # the transverse RF focusing is modelled differently by every code (and not at all by
            # some), so over a linac of cavities the transverse map is reported, not asserted; the
            # reference energy still is (measured on LightWin's ADS deck, docs/oracles.md)
            map_ok = True
            caveat("field-map cavities: transverse RF focusing differs by engine (map report only); ",
                   append=True)
    else:
        map_tol = energy_tol = None
        map_ok = e_ok = True          # lossy: report only
    row = pc.row().splitlines()[0].strip()
    return EngineVerdict(bool(map_ok and e_ok), tier, metric, metric_rel, pc.energy_rel, blocks, map_tol,
                         energy_tol, energy_checked, floor, (note or "") + row, notes)


def _engine_check(res: CaseResult, deck: Path, src: str, out: Path, dst: str, lat: Lattice,
                  workdir: Path, cache: dict) -> None:
    from lattix.oracles import get_oracle
    from lattix.oracles.compare import compare_pair

    ea, eb = pick_engine(src), pick_engine(dst)
    if not ea or not eb:
        res.engine_note = "no engine pair"
        return
    if _has_relative_phase_maps(lat):
        # HELIX adds the running bunch phase to relative field-map phases (measured 2026-09-05,
        # docs/oracles.md): LightWin is the TraceWin-semantics engine for such decks
        for which, name in (("a", ea), ("b", eb)):
            if name == "helix":
                if get_oracle("lightwin").available()[0]:
                    ea, eb = (("lightwin", eb) if which == "a" else (ea, "lightwin"))
                elif res.tier != "lossy":
                    res.tier = "lossy"
                    res.engine_note = "HELIX relative field-map phases (known HELIX limit, report only); "
    for name in (ea, eb):
        ok, why = get_oracle(name).available()
        if not ok:
            res.engine_note = f"{name} unavailable: {why[:60]}"
            return
    try:
        beam = beam_from_lattice(lat)
    except RuntimeError as exc:
        res.engine_note = f"skipped: {exc}"          # custom ion species: the adapters take named species only
        return
    p0_note = constant_p0_note(lat, (ea, eb))
    if p0_note:
        res.engine_note = p0_note
        return
    key_a = (str(deck), src, ea)
    if key_a not in cache:
        cache[key_a] = get_oracle(ea).run(deck, fmt=src, beam=beam, workdir=workdir / f"eng_{ea}_{deck.stem}")
    ra = cache[key_a]
    rb = get_oracle(eb).run(out, fmt=dst, beam=beam, workdir=workdir / f"eng_{eb}_{out.stem}")
    pc = compare_pair(ra, rb)
    res.engine_metric = pc.max_rcum_rel
    res.engine_energy = pc.energy_rel
    if pc.n_shared == 0:
        res.engine_ok, res.engine_note = False, "no shared boundaries"
        return
    v = engine_verdict(pc, ea, eb, lat, res.tier, res.codes, ra, rb, note=res.engine_note)
    res.tier, res.engine_metric, res.engine_ok, res.engine_note = v.tier, v.metric, v.ok, v.note


# ---------------------------------------------------------------------------
# the matrix

def readable_formats() -> list[str]:
    """Formats with both a reader and a writer (a writer-only format cannot be read back); the
    formats reached through Bmad's converters are not battery formats (no engine, an external
    Bmad source tree for their readers)."""
    return [f for f, spec in FORMATS.items()
            if spec.reader_attr and spec.writer_attr and not spec.options.get("bridge")]


def pairs(formats: list[str] | None = None) -> list[tuple[str, str]]:
    fmts = formats or readable_formats()
    return [(a, b) for a in fmts for b in fmts if a != b]


def run_matrix(decks: list[tuple[str, str, dict]] | None = None, formats: list[str] | None = None,
               *, workdir: Path, engines: bool = False, only_src: str | None = None,
               only_dst: str | None = None, derived: bool = False) -> list[CaseResult]:
    decks = decks if decks is not None else DECKS
    fmts = formats or readable_formats()
    cache: dict = {}
    results = []
    sources: list[tuple[Path, str, dict]] = [(PUBLIC / rel, fmt, opts) for rel, fmt, opts in decks]
    if derived:
        sources += derived_decks(workdir)
    for path, fmt, opts in sources:
        if only_src and fmt != only_src:
            continue
        for dst in fmts:
            if dst == fmt or (only_dst and dst != only_dst):
                continue
            wd = workdir / f"{path.stem}_{fmt}"
            results.append(run_case(path, fmt, dst, wd, engines=engines, engine_cache=cache, read_options=opts))
    return results


def to_markdown(results: list[CaseResult]) -> str:
    lines = ["| deck | src → dst | ledger | tier | IR | fixed | engine | note |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        led = " ".join(f"{k[0]}{v}" for k, v in sorted(r.ledger.items()))
        ir = "—" if r.ir_ok is None else ("ok" if r.ir_ok else f"FAIL {r.ir_worst:.1e}")
        fx = "—" if r.fixed_ok is None else ("ok" if r.fixed_ok else "FAIL")
        if r.engine_ok is None:
            eng = "—"
        else:
            eng = ("ok" if r.engine_ok else "FAIL") + (f" {r.engine_metric:.1e}" if r.engine_metric is not None else "")
        note = r.error or (r.ir_problems[0] if r.ir_problems else "") or r.fixed_note or r.engine_note
        lines.append(f"| {Path(r.deck).name} | {r.src} → {r.dst} | {led} | {r.tier} | {ir} | {fx} | {eng} | "
                     f"{note[:110].replace('|', '/')} |")
    return "\n".join(lines)


def to_json(results: list[CaseResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=1)


def summary(results: list[CaseResult]) -> str:
    n = len(results)
    ir_bad = sum(1 for r in results if r.ir_ok is False)
    fx_bad = sum(1 for r in results if r.fixed_ok is False)
    en_bad = sum(1 for r in results if r.engine_ok is False)
    en_run = sum(1 for r in results if r.engine_ok is not None)
    err = sum(1 for r in results if r.error)
    return (f"{n} cases: IR failures {ir_bad}, fixed-point failures {fx_bad}, "
            f"engine failures {en_bad}/{en_run} run, errors {err}")
