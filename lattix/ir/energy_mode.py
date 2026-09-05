"""Normalized strengths for constant-``p0`` targets (MAD-X, MAD8, xtrack).

Those engines keep one reference momentum through a lattice, so a strength normalized as
``k = G / Bρ`` needs one rigidity to be chosen per element.  Three policies exist, all recorded
in the fidelity ledger (``EQUIVALENT:CONST_P0_*_RIGIDITY``) and tagged in the deck
(``! lattix: energy_mode=…``) so the reader can undo them:

``"delta"`` (default)
    The rigidity the engine's **own reference particle** has at that element.  The engine
    applies the RF energy gains the deck contains, but as ``δ = Δp/p0`` on the particle rather
    than as a new ``p0``; it cannot apply an explicit reference change (TraceWin
    ``SET_BEAM_ENERGY``, a ``ReferenceChange``) at all.  So ``Bρ_used = Bρ_start · p_local/p_probe``
    with ``p_probe`` the momentum after the RF gains only: the start rigidity across pure RF
    acceleration (Bmad's ``bmad_to_mad`` renormalises ``k1·p0c/p0c_start`` the same way), the
    local rigidity across reference changes.  Measured: MAD-X ``twiss`` and xtrack tracking both
    carry the RF gain in the orbit's ``δ`` (:mod:`lattix.oracles`), so a magnet written with the
    local rigidity would be *double* corrected — a 1 MV on-crest gap at 2 MeV followed by a
    ``k1 = 4.08`` quad reads back as ``k_eff = 3.33`` in the engine.
``"local"``
    ``Bρ_used = Bρ_local`` at each element's entrance: the section-wise deck a MAD-X user writes
    by hand (correct only when the engine's reference particle does not gain energy).
``"constant"``
    ``Bρ_used = Bρ_start`` everywhere.

A ``Taylor`` map lives in the IR's local-``p0`` coordinates ``(x, px/p_local, …)``; in the
engine it acts on ``(x, px/p_start·…)``, so its momentum rows/columns are rescaled by the same
ratio ``r = Bρ_local / Bρ_used`` (:func:`scale_taylor`).

**Phase slip.**  The same engines keep the reference *velocity* fixed too: their RF phase at a
cavity is ``φ_seen = φ_written + 2π f Δt`` with ``Δt`` the particle's arrival time relative to
the clock ``s/(β0 c)`` (measured 2026-09-04: xtrack ``ΔE = V sin(lag − 2π ζ/(β0 λ))`` and MAD-X
``ΔE = V sin(2π lag − 2π f t/c)``, both with the kick at the centre of a thick cavity).  Their
reference particle, having gained energy at the design rate, arrives early by exactly the
design's ``t(s) − s/(β0 c)``, so in ``"delta"`` mode every cavity's phase is written as
``φ − 2π f Δt_design`` (:func:`phase_slip_turns`, ``EQUIVALENT:CONST_P0_PHASE_SLIP``) and the
reader adds it back (:func:`undo_phase_slip`).  Without it the second gap of a DTL already sits
tens of degrees off its synchronous phase.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lattix.ir.lattice import Lattice, Placed
    from lattix.ir.reference import ReferenceParticle

ENERGY_MODES = ("delta", "local", "constant")
PHASE_SLIP_CODE = "CONST_P0_PHASE_SLIP"
_SLIP_TOL_TURNS = 1e-9           # f·Δt below this is roundoff (3e-12 turns on a 30 m FLAME line with no acceleration)

#: ledger code per mode
RIGIDITY_CODE = {
    "delta": "CONST_P0_DELTA_RIGIDITY",
    "local": "CONST_P0_LOCAL_RIGIDITY",
    "constant": "CONST_P0_START_RIGIDITY",
}
RIGIDITY_MESSAGE = {
    "delta": ("normalized strengths use the rigidity the engine's own reference particle has "
              "here: the start rigidity scaled by the RF gains it accumulates as delta, and the "
              "local rigidity across explicit reference changes"),
    "local": "normalized strengths use the local rigidity at each element's entrance",
    "constant": "normalized strengths use the rigidity at the start of the lattice",
}

_MOMENTUM_ROWS = (1, 3, 5)
#: reader notes that a bend's k0 differs from angle/l (undone when the energy mode explains it)
_K0_NOTES = ("BEND_K0_NE_ANGLE", "BEND_K0_NE_H")


def record_rigidity_mode(rep, energy_mode: str, *, element: str, kind: str, **details) -> None:
    """The per-accelerating-element ledger row that names the mode (one literal site per code)."""
    if energy_mode == "delta":
        rep.equivalent("CONST_P0_DELTA_RIGIDITY", RIGIDITY_MESSAGE["delta"], element=element, kind=kind, **details)
    elif energy_mode == "local":
        rep.equivalent("CONST_P0_LOCAL_RIGIDITY", RIGIDITY_MESSAGE["local"], element=element, kind=kind, **details)
    else:
        rep.equivalent("CONST_P0_START_RIGIDITY", RIGIDITY_MESSAGE["constant"], element=element, kind=kind,
                       **details)


def record_phase_slip(rep, *, element: str, kind: str, slip_turns: float, engine: str, attribute: str) -> None:
    rep.equivalent("CONST_P0_PHASE_SLIP",
                   f"{attribute} moved by the phase slip of {engine}'s constant-velocity clock so its "
                   "reference particle meets the design synchronous phase here",
                   element=element, kind=kind, slip_turns=slip_turns)


def check_mode(energy_mode: str) -> str:
    if energy_mode not in ENERGY_MODES:
        raise ValueError(f"energy_mode must be one of {ENERGY_MODES}, got {energy_mode!r}")
    return energy_mode


def probe_momentum_ratio(placed: list[Placed], start: ReferenceParticle) -> list[float]:
    """``p_probe / p_start`` at every placed element's entrance: the momentum a constant-``p0``
    engine's reference particle has there, i.e. the start momentum advanced by the RF gains
    only (a ``ReferenceChange`` is invisible to such an engine)."""
    from lattix.ir.elements import ReferenceChange
    from lattix.ir.walk import energy_gain_eV

    p0 = start.pc_eV
    probe = start
    out: list[float] = []
    for p in placed:
        out.append(probe.pc_eV / p0)
        e = p.element
        if isinstance(e, ReferenceChange):
            continue
        ref = p.ref_in or start
        dE = energy_gain_eV(e, ref)
        if dE:
            probe = probe.advanced(dE_eV=dE)
    return out


def phase_slip_turns(ref_in: ReferenceParticle, s_in: float, length: float, frequency_Hz: float | None,
                     start: ReferenceParticle) -> float:
    """``f·Δt`` at the cavity's kick (its centre when thick): the design arrival time of the
    reference particle minus the constant-``β0`` clock's ``s/(β0 c)``.  Zero at the lattice
    start and whenever nothing upstream changed the velocity."""
    from lattix.ir.reference import C_LIGHT

    if not frequency_Hz:
        return 0.0
    half = 0.5 * length
    t_kick = ref_in.time_s - start.time_s + (half / (ref_in.beta * C_LIGHT) if half else 0.0)
    return frequency_Hz * (t_kick - (s_in + half) / (start.beta * C_LIGHT))


def slip_is_zero(slip_turns: float) -> bool:
    return abs(slip_turns) <= _SLIP_TOL_TURNS


def undo_phase_slip(lat: Lattice, rep, *, resolve_p0c_steps: bool = False) -> int:
    """Reader side of the ``"delta"`` phase convention: walk the line with the design reference
    (the same walk :func:`lattix.ir.walk.propagate` does) and add ``2π f Δt`` back to every RF
    cavity's phase, using the phases already corrected upstream.  With ``resolve_p0c_steps`` a
    ``ReferenceChange`` that only carries an xtrack ``Delta_p0c`` gets its energy jump from the
    reference at that point first.  Returns the number of cavities touched."""
    import math

    from lattix.ir.elements import Freq, ReferenceChange, RFCavity
    from lattix.ir.units import wrap_rad
    from lattix.ir.walk import energy_gain_eV

    start = lat.reference
    ref = start
    done: set[int] = set()
    n = 0
    for p in lat.flatten():
        e = p.element
        if isinstance(e, Freq):
            ref = ref.advanced(rf_frequency_Hz=e.frequency_Hz)
            continue
        if resolve_p0c_steps and isinstance(e, ReferenceChange) and e.dE_ref_eV is None and e.energy_eV is None:
            dp = (e.native.get("xtrack") or {}).get("Delta_p0c")
            if dp:
                pc = ref.pc_eV + float(dp)
                mass = ref.species.mass_eV
                e.dE_ref_eV = math.sqrt(pc * pc + mass * mass) - mass - ref.kinetic_energy_eV
        if isinstance(e, RFCavity) and id(e) not in done:
            done.add(id(e))
            slip = phase_slip_turns(ref, p.s_in, e.length, e.rf.frequency_Hz or ref.rf_frequency_Hz, start)
            if not slip_is_zero(slip):
                e.rf.phase_rad = wrap_rad(e.rf.phase_rad + 2.0 * math.pi * slip)
                n += 1
        dE = energy_gain_eV(e, ref)
        rf = getattr(e, "rf", None)
        f_new = rf.frequency_Hz if (rf is not None and rf.frequency_Hz) else None
        ref = ref.advanced(dE_eV=dE, ds_m=e.length, rf_frequency_Hz=f_new)
        if isinstance(e, ReferenceChange) and e.dtime_s:
            ref = ref.model_copy(update={"time_s": ref.time_s + e.dtime_s})
    if n:
        rep.equivalent("PHASE_SLIP_RESTORED",
                       f"{n} cavity phase(s) moved back from the deck's constant-velocity clock to the "
                       "design synchronous phase", element=None, kind=None)
    return n


def mode_ratio(energy_mode: str, brho_local: float, brho_start: float, probe_ratio: float) -> float:
    """``r = Bρ_local / Bρ_used``: the factor between the IR's local normalization and the
    engine's; ``Bρ_used = brho_local / r``."""
    if energy_mode == "local":
        return 1.0
    if energy_mode == "constant":
        return brho_local / brho_start if brho_start else 1.0
    return probe_ratio                              # delta


def rigidity_for(energy_mode: str, brho_local: float, brho_start: float, probe_ratio: float) -> float:
    r = mode_ratio(energy_mode, brho_local, brho_start, probe_ratio)
    return brho_local / r if r else brho_local


def scale_taylor(matrix, offset, r: float, *, inverse: bool = False) -> tuple[list[list[float]], list[float]]:
    """Express a map given in ``(x, px/p_a, y, py/p_a, z, δ_a)`` in coordinates normalized to
    ``p_b = p_a / r`` (``inverse=True`` goes back): momentum rows are multiplied by ``r``,
    momentum columns divided by ``r``."""
    if r == 1.0:
        return [list(map(float, row)) for row in matrix], [float(v) for v in offset]
    f = 1.0 / r if inverse else r
    s = [f if i in _MOMENTUM_ROWS else 1.0 for i in range(6)]
    m = [[float(matrix[i][j]) * s[i] / s[j] for j in range(6)] for i in range(6)]
    o = [float(offset[i]) * s[i] for i in range(6)]
    return m, o


def restore_energy_mode(lat: Lattice, rep, energy_mode: str, *, brho_read: float | None = None) -> int:
    """Undo a writer's normalization on a lattice whose fields were rebuilt with one rigidity
    (``brho_read``, default the start rigidity): rescale every magnet's fields by
    ``Bρ_local / (r · Bρ_read)`` and every ``Taylor`` back into local coordinates.  Returns the
    number of elements touched; records ``EQUIVALENT:ENERGY_MODE_RESTORED``."""
    from lattix.ir.walk import propagate

    start = lat.reference
    brho_start = start.brho_signed
    brho_read = brho_start if brho_read is None else brho_read
    if not brho_start:
        return 0
    placed = propagate(lat)
    ratios = probe_momentum_ratio(placed, start)
    done: dict[str, float] = {}
    n = 0
    for p, pr in zip(placed, ratios, strict=True):
        e = p.element
        if p.ref_in is None:
            continue
        brho_local = p.ref_in.brho_signed
        r = mode_ratio(energy_mode, brho_local, brho_start, pr)
        factor = brho_local / (r * brho_read)
        if e.name in done:
            if abs(done[e.name] - factor) > 1e-12:
                rep.lossy("LOCAL_RIGIDITY_SHARED_DEFINITION",
                          "element definition reused at different reference energies; the first "
                          "occurrence's rigidity was used to restore its lab fields",
                          element=e.name, kind=e.kind)
            continue
        done[e.name] = factor
        if e.kind == "Taylor":
            if abs(r - 1.0) > 1e-15:
                e.matrix, e.offset = scale_taylor(e.matrix, e.offset, r, inverse=True)
                n += 1
            continue
        if e.kind == "Kicker":
            if abs(r - 1.0) > 1e-15:
                e.hkick, e.vkick = e.hkick / r, e.vkick / r
                n += 1
            continue
        if e.kind == "Bend" and abs(r - 1.0) > 1e-15:
            # the writer set k0 = h·r so the engine's delta-carrying orbit follows the design
            # angle: that is the convention, not a field/geometry mismatch
            g = e.bend.g_ref(e.length) if e.length else 0.0
            for fmt in ("madx", "xtrack"):
                nat = e.native.get(fmt) or {}
                k0 = nat.get("k0")
                if k0 is not None and abs(k0 - g * r) <= 1e-9 * max(1.0, abs(g * r)):
                    del nat["k0"]
                    rep.entries = [x for x in rep.entries
                                   if not (x.element == e.name and x.code in _K0_NOTES)]
                    n += 1
        if abs(factor - 1.0) <= 1e-15:
            continue
        mp = getattr(e, "multipole", None)
        if mp is not None:
            for d in (mp.Bn, mp.Bs, mp.BnL, mp.BsL):
                for k in list(d):
                    d[k] *= factor
        if e.kind == "Solenoid":
            e.solenoid.Bsol_T *= factor
        n += 1
    if n:
        rep.equivalent("ENERGY_MODE_RESTORED",
                       f"{n} element(s) re-normalized from the deck's energy_mode={energy_mode!r} "
                       "convention back to the IR's local rigidity", element=None, kind=None)
    return n
