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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lattix.fidelity import FidelityReport
from lattix.formats.base import FORMATS, read, write
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
    "scibmad": "scibmad",
}
#: fallback engines per format, tried in order when the primary one is unavailable (CI has no HELIX)
ENGINE_CANDIDATES: dict[str, tuple[str, ...]] = {"tracewin": ("helix", "lightwin")}
FOLLOWS_P0 = {"helix": True, "bmad": True, "elegant": True, "tracewin": True, "impactx": True, "lightwin": True,
              "impactz": True, "flame": True, "madx": False, "xtrack": False, "scibmad": False}

RTOL = 1e-9
#: lattice-level codes after which normalized strengths no longer mean the same thing
_SPECIES_LOSS = ("SPECIES_NOT_REPRESENTABLE", "UNKNOWN_SPECIES", "PALS_SPECIES_UNKNOWN", "SPECIES_ASSUMED")
EXACT_MAP_TOL = 1e-7          # max |ΔR̂cum| relative, exact tier (roundoff over long lines)
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
    "FLAME_NO_ENG_DATA_DIR": set(), "FLAME_PER_NUCLEON": set(), "FLAME_SOURCE_ADDED": set(),
    "DEFINITION_NOT_IN_LINE": set(), "ELEGANT_PHASE_FOR_SPECIES": set(),
    "CONST_P0": set(), "CONST_P0_LOCAL_RIGIDITY": set(), "CONST_P0_START_RIGIDITY": set(),
    "THIN_CAVITY_NO_RF_FOCUSING": set(), "THIN_CAVITY_TRAVELING_WAVE": set(), "RFCA_DEFAULT_FREQ": set(),
    "RF_FREQUENCY_UNKNOWN": set(), "IMPACTZ_NO_FREQUENCY": set(), "RFCAVITY_GAIN_UNKNOWN": set(),
    "ZERO_ANGLE_BEND_AS_DRIFT": set(), "IMPACTZ_RF_FORM_FACTOR": set(), "THIN_GAP_AS_SHORT_CAVITY": set(),
    "PALS_TTF_DROPPED": set(), "RF_FREQUENCY_MISSING": set(),
}

#: codes whose model moves an element boundary by up to this many metres (short cavities)
FUZZY_S: dict[str, float] = {"THIN_GAP_AS_SHORT_CAVITY": 2e-2, "THICK_CAVITY_AS_SHORTRF": 2e-2}

_THIN_BENDLESS = ("Marker", "Instrument", "Directive", "Freq", "Patch", "ReferenceChange")


def _rotated(bn: dict, bs: dict, tilt: dict, base_tilt: float = 0.0) -> tuple[dict, dict]:
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


def _contrib(p: Placed) -> dict[str, float]:
    """This placed element's additive contribution to every cumulative quantity (energy is a state)."""
    e = p.element
    c = dict.fromkeys(QUANTITIES, 0.0)
    c["length"] = p.length
    k = e.kind
    mp = getattr(e, "multipole", None)
    if mp is not None:
        base = (e.bend.tilt_ref if k == "Bend" else 0.0) + (e.shift.tilt if e.shift is not None else 0.0)
        bn, bs = _rotated(mp.Bn, mp.Bs, mp.tilt, base)       # thick: field × length
        for n, v in bn.items():
            if n < 6:
                c[f"BnL{n}"] += v * p.length
        for n, v in bs.items():
            if n < 6:
                c[f"BsL{n}"] += v * p.length
        bnl, bsl = _rotated(mp.BnL, mp.BsL, mp.tilt, base)   # thin: integrated already
        for n, v in bnl.items():
            if n < 6:
                c[f"BnL{n}"] += v
        for n, v in bsl.items():
            if n < 6:
                c[f"BsL{n}"] += v
    if k == "Bend":
        b = e.bend
        c["angle_h"] = b.angle * math.cos(b.tilt_ref)
        c["angle_v"] = b.angle * math.sin(b.tilt_ref)
    elif k == "Solenoid":
        c["BsolL"] = e.solenoid.Bsol_T * p.length
    elif k == "Kicker":
        c["hkick"], c["vkick"] = e.hkick, e.vkick
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


@dataclass
class Profile:
    names: list[str]
    kinds: list[str]
    s_out: np.ndarray
    cum: dict[str, np.ndarray]          # cumulative at each exit
    energy: np.ndarray                  # reference kinetic energy at each exit
    drift_like: list[bool] = field(default_factory=list)   # contributes nothing but length (may coalesce)


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
    drift_like: list[bool] = []
    for i, p in enumerate(placed):
        c = _contrib(p)
        drop = neutral.get(p.name, set())
        drift_like.append(all(v == 0.0 for q, v in c.items() if q != "length" and not ("*" in drop or q in drop)))
        for q in cum:
            if q == "length" or not ("*" in drop or q in drop):
                run[q] += c[q]
            cum[q][i] = run[q]             # a neutralised element still carries the running value
        if ("*" in drop or "gain" in drop) and p.ref_out is not None and p.ref_in is not None:
            e_drop += p.ref_out.kinetic_energy_eV - p.ref_in.kinetic_energy_eV
        energy[i] = (p.ref_out.kinetic_energy_eV if p.ref_out else float("nan")) - e_drop
    return Profile([p.name for p in placed], [p.element.kind for p in placed],
                   np.array([p.s_out for p in placed]), cum, energy, drift_like)


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
        neutral = neutral_set(rep)
        # LOSSY reader entries on the way back also neutralise (what the target format cannot say)
        for e in rep2.entries:
            if e.cls.value in ("LOSSY", "DROPPED") and e.element:
                neutral.setdefault(e.element, set()).update(AFFECTS.get(e.code, {"*"}))
        skip_energy = any(c in res.codes for c in ("CONST_P0", "REFCHANGE_DROPPED"))
        skip: set[str] = set()
        if any(c in res.codes for c in _SPECIES_LOSS) or any(c in rep2.codes() for c in _SPECIES_LOSS[:-1]):
            # the target cannot name this species: every normalized strength changes meaning
            skip = {q for q in QUANTITIES if q not in ("length", "angle_h", "angle_v", "hkick", "vkick")}
            skip_energy = True
            res.tier = "lossy"
        fuzzy = {e.element: FUZZY_S[e.code] for e in rep.entries if e.code in FUZZY_S and e.element}
        diff = compare_profiles(profile(lat, neutral), profile(lat2, neutral), skip_energy=skip_energy, skip=skip,
                                fuzzy=fuzzy)
        res.ir_ok, res.ir_worst, res.ir_problems = diff.ok, diff.worst, diff.problems
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


def _round_json(obj):
    if isinstance(obj, float):
        return float(f"{obj:.12g}")
    if isinstance(obj, list):
        return [_round_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_json(v) for k, v in obj.items()}
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


def _beam(lat: Lattice):
    from lattix.oracles.base import SPECIES, BeamSpec

    sp = lat.reference.species
    name = next((k for k, (m, q) in SPECIES.items() if abs(m - sp.mass_eV) / m < 1e-6 and q == sp.charge), None)
    if name is None:
        raise RuntimeError(f"species {sp.name!r} not known to the engines")
    return BeamSpec(name, lat.reference.kinetic_energy_eV, lat.reference.rf_frequency_Hz)


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


def _pick_engine(fmt: str) -> str | None:
    """The format's engine, or the first available fallback from :data:`ENGINE_CANDIDATES`."""
    from lattix.oracles import get_oracle

    primary = ENGINE_FOR_FORMAT.get(fmt)
    for name in ENGINE_CANDIDATES.get(fmt, ()):
        if get_oracle(name).available()[0]:
            return name
    return primary


def _engine_check(res: CaseResult, deck: Path, src: str, out: Path, dst: str, lat: Lattice,
                  workdir: Path, cache: dict) -> None:
    from lattix.oracles import get_oracle
    from lattix.oracles.compare import compare_pair

    ea, eb = _pick_engine(src), _pick_engine(dst)
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
        beam = _beam(lat)
    except RuntimeError as exc:
        res.engine_note = f"skipped: {exc}"          # custom ion species: the adapters take named species only
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
        if any(e.kind == "Bend" and e.bend.angle < 0 for e in lat.elements.values()) and res.tier != "lossy":
            # measured 2026-09-04 (docs/oracles.md): HELIX's BEND body evaluates the sector map at the
            # signed angle with rho > 0, so a negative-angle bend gets R12 < 0 and R21 > 0 (TraceWin
            # itself gives the positive-bend block with R16/R26 flipped) — report only for such decks
            res.tier = "lossy"
            res.engine_note = "HELIX negative-angle bend body (known HELIX limit, report only); "
    fm_derived = any(c.startswith("FM_") for c in res.codes)
    if has_rf and (FOLLOWS_P0[ea] != FOLLOWS_P0[eb] or "elegant" in (ea, eb) or fm_derived):
        # a field map integrated by one engine and a cavity element in the other agree on the
        # transverse block and the dispersion, never on the longitudinal model
        blocks = {"T4x4", "disp"}
    metric = max(pc.blocks[b] for b in blocks if b in pc.blocks)
    res.engine_metric = metric
    scale = max(1.0, pc.max_rcum_abs / max(pc.max_rcum_rel, 1e-300)) if pc.max_rcum_rel else 1.0
    metric_rel = metric / scale
    for r_ in (ra, rb):
        missing = (len(r_.meta.get("substituted", [])) + len(r_.meta.get("dropped", []))
                   + int(r_.meta.get("solenoids_as_drifts", 0) or 0))
        if r_.engine == "lightwin" and missing and res.tier != "lossy":
            # LightWin propagates elements without an Envelope3D model (EDGE, THIN_STEERING …) as
            # drifts of their length and skips keywords it does not implement (GAP, NCELLS …): the
            # comparison cannot be held to any tier — report only
            res.tier = "lossy"
            res.engine_note = f"LightWin has no model for {missing} element(s) (report only); "
    if "scibmad" in (ea, eb) and res.tier == "exact" and any(
            e.kind == "RFCavity" and e.length > 0 and (e.rf.voltage_V or e.rf.gradient_V_per_m)
            for e in lat.elements.values()):
        # SciBmad's thick RFCavity applies its own transverse RF focusing (measured, docs/oracles.md);
        # MAD-X, xtrack and Elegant kick at the centre only
        res.tier = "equivalent"
        res.engine_note = "SciBmad thick-cavity RF focusing: engine models differ; "
    if "scibmad" in (ea, eb) and _has_fringe_bends(lat) and res.tier != "lossy":
        # BeamTracking 0.5 stores edge1_int/edge2_int but cannot track them (the worker zeroes them):
        # the fringe correction is missing entirely, not modelled differently — report only
        res.tier = "lossy"
        res.engine_note = "SciBmad 0.5 does not track fringe integrals (report only); "
    if res.tier == "exact" and _has_fringe_bends(lat) and ea != eb:
        # the codes agree on the parameters but not on the fringe-field model: MAD-X and Bmad
        # differ at O(ψ²) in the fint·hgap correction (8.6e-4 on ELENA's 60° bend, 0 without it)
        res.tier = "equivalent"
        res.engine_note = "fringe-integral bends: engine models differ; "
    if res.tier == "exact":
        map_ok = metric_rel <= EXACT_MAP_TOL
        e_ok = (not (FOLLOWS_P0[ea] and FOLLOWS_P0[eb])) or pc.energy_rel <= EXACT_ENERGY_TOL
    elif res.tier == "equivalent":
        map_ok = metric_rel <= EQUIV_MAP_TOL
        e_ok = (not (FOLLOWS_P0[ea] and FOLLOWS_P0[eb])) or pc.energy_rel <= EQUIV_ENERGY_TOL
        if fm_derived and not map_ok:
            # a field map integrated by one engine against a derived cavity element in the other:
            # the transverse RF focusing is modelled differently by every code (and not at all by
            # some), so over a linac of cavities the transverse map is reported, not asserted; the
            # reference energy still is (measured on LightWin's ADS deck, docs/oracles.md)
            map_ok = True
            res.engine_note = ((res.engine_note or "")
                               + "field-map cavities: transverse RF focusing differs by engine (map report only); ")
    else:
        map_ok = e_ok = True          # lossy: report only
    res.engine_ok = bool(map_ok and e_ok)
    res.engine_note = (res.engine_note or "") + pc.row().splitlines()[0].strip()


# ---------------------------------------------------------------------------
# the matrix

def readable_formats() -> list[str]:
    """Formats with both a reader and a writer (a writer-only format cannot be read back)."""
    return [f for f, spec in FORMATS.items() if spec.reader_attr and spec.writer_attr]


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
