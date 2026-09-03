"""Propagate s, time and the reference energy through the expanded lattice.

Energy rule (PLAN §4.1): ``RFCavity``: dE = voltage_V·cos(phase_rad) (synchronous phase
for the reference particle, species-independent); ``FieldMap``/``NCells``/``RFQCell``/
``Superposition``: ``rf.dE_ref_eV`` if known (else 0 with a warning); ``ReferenceChange``:
explicit; ``Freq``: switches the RF clock.  Every RF element whose ``rf.frequency_Hz`` is
set also moves the clock to its own frequency (HELIX ``RFGap.advance_ref`` semantics).
Everything else leaves the energy unchanged.
"""
from __future__ import annotations

import math

from lattix.ir.elements import Freq, ReferenceChange, RFCavity
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle


def energy_gain_eV(e, ref: ReferenceParticle, warnings: list[str] | None = None) -> float:
    if isinstance(e, RFCavity):
        rf = e.rf
        if rf.dE_ref_eV is not None:
            return rf.dE_ref_eV
        V = rf.voltage_V if rf.gradient_V_per_m is None or rf.voltage_V else \
            rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else e.length)
        return V * math.cos(rf.phase_rad)
    if isinstance(e, ReferenceChange):
        if e.energy_eV is not None:
            return e.energy_eV - ref.kinetic_energy_eV
        return e.dE_ref_eV or 0.0
    rf = getattr(e, "rf", None)
    if rf is not None:                      # FieldMap, NCells, RFQCell
        if rf.dE_ref_eV is not None:
            return rf.dE_ref_eV
        if warnings is not None and (rf.voltage_V or rf.gradient_V_per_m or e.kind == "FieldMap"):
            warnings.append(f"{e.kind} {e.name!r}: reference energy gain unknown (dE_ref_eV not set); "
                            "treated as 0")
    return 0.0


def propagate(lat: Lattice, use: str | None = None, *, reference: ReferenceParticle | None = None,
              warnings: list[str] | None = None) -> list[Placed]:
    """flatten() + reference particle at the entry and exit of every element."""
    ref = reference or lat.reference
    out = []
    for p in lat.flatten(use):
        e = p.element
        ref_in = ref
        if isinstance(e, Freq):
            ref = ref.advanced(rf_frequency_Hz=e.frequency_Hz)
        else:
            dE = energy_gain_eV(e, ref, warnings)
            rf = getattr(e, "rf", None)
            f_new = rf.frequency_Hz if (rf is not None and rf.frequency_Hz) else None
            ref = ref.advanced(dE_eV=dE, ds_m=e.length, rf_frequency_Hz=f_new)
            if isinstance(e, ReferenceChange) and e.dtime_s:
                ref = ref.model_copy(update={"time_s": ref.time_s + e.dtime_s})
        out.append(p.model_copy(update={"ref_in": ref_in, "ref_out": ref}))
    return out
