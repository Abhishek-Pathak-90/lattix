"""Synergia worker: per-element first-order maps of a Synergia lattice JSON, from Synergia's own tracking.

Started by :mod:`lattix.oracles.synergia` as ``<python> synergia_worker.py lattice.json --out r.json …`` inside
the Synergia environment (the clone's pixi install).  Nothing here imports lattix.

Synergia keeps the lattice's *design* reference particle fixed and lets the *bunch* reference particle
follow the RF gains (``ff_rfcavity``: ``new_pref_b``), so every element is applied the way a continuous run
would apply it: one single-element lattice per element with the design reference, a fresh 13-particle probe
bunch whose reference carries the energy reached so far (an on-axis particle and ± offsets in each of
``(x, xp, y, yp, cdt, dpop)``), one ``Propagator`` pass, and the map fitted from the offsets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROBE_AMP = (1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-4)


def _probe_rows(amp) -> list[list[float]]:
    rows = [[0.0] * 6]
    for k in range(6):
        for sgn in (+1.0, -1.0):
            r = [0.0] * 6
            r[k] = sgn * amp[k]
            rows.append(r)
    return rows


def _logger(synergia):
    utils = synergia.utils
    for path in ("parallel_utils.Logger", "Logger"):
        obj = utils
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        try:
            lv = getattr(getattr(utils, "parallel_utils", utils), "LoggerV", None)
            return obj(0, lv.ERROR) if lv is not None else obj(0)
        except TypeError:
            return obj(0)
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("deck")
    p.add_argument("--out", required=True)
    p.add_argument("--ke-ev", type=float, default=None, help="probe kinetic energy (default: the lattice's)")
    p.add_argument("--mass-ev", type=float, default=None)
    p.add_argument("--charge", type=float, default=None)
    p.add_argument("--species", default="proton")
    a = p.parse_args(argv)
    import numpy as np
    import synergia
    from synergia.foundation import Four_momentum, Reference_particle
    from synergia.lattice import Lattice
    from synergia.simulation import Bunch_simulator, Independent_stepper_elements, Propagator

    lattice = Lattice.load_from_json(Path(a.deck).read_text())
    design = lattice.get_reference_particle()
    mass_GeV = design.get_mass() if hasattr(design, "get_mass") else design.get_four_momentum().get_mass()
    charge = design.get_charge()
    if a.mass_ev is not None and abs(a.mass_ev * 1e-9 - mass_GeV) > 1e-6 * mass_GeV:
        print(f"warning: the deck's mass {mass_GeV} GeV differs from the beam's {a.mass_ev * 1e-9}", file=sys.stderr)
    w_GeV = (a.ke_ev * 1e-9) if a.ke_ev is not None else (design.get_total_energy() - mass_GeV)
    logger = _logger(synergia)
    names, kinds, length, s_out, R, w_in, w_out, warns = [], [], [], [], [], [], [], []
    s = 0.0
    rows = _probe_rows(PROBE_AMP)
    n = len(rows)
    cdt0 = 0.0          # the bunch's time offset from the design reference accumulates (it sets the RF phases)
    for e in lattice.get_elements():
        kind = e.get_type_name() if hasattr(e, "get_type_name") else str(e.get_type())
        L = float(e.get_length())
        sub = Lattice("probe", design)
        sub.append(e)
        fm = Four_momentum(mass_GeV, w_GeV + mass_GeV)
        ref_b = Reference_particle(charge, fm)
        try:
            sim = Bunch_simulator.create_single_bunch_simulator(ref_b, n, 1.0e9)
        except TypeError:
            sim = Bunch_simulator.create_single_bunch_simulator(ref_b, n, 1.0e9, synergia.utils.Commxx())
        bunch = sim.get_bunch(0, 0)
        parts = bunch.get_particles_numpy()
        for i, r in enumerate(rows):
            for j in range(6):
                parts[i, j] = r[j]
            parts[i, 4] += cdt0
            parts[i, 6] = i
        bunch.checkin_particles()
        prop = Propagator(sub, Independent_stepper_elements(1))
        if logger is not None:
            prop.propagate(sim, logger, 1)
        else:
            prop.propagate(sim, 1)
        bunch.checkout_particles()
        out = np.array(bunch.get_particles_numpy()[:n, :6], dtype=float)
        if not np.all(np.isfinite(out)):
            warns.append(f"{e.get_name()}: non-finite probe coordinates")
        da = np.array(rows, dtype=float)[1:] - np.array(rows[0])
        db = out[1:] - out[0]
        Rk, *_ = np.linalg.lstsq(da, db, rcond=None)
        cdt0 = float(out[0, 4])
        ref_after = bunch.get_reference_particle()
        w_after = ref_after.get_total_energy() - mass_GeV
        names.append(e.get_name())
        kinds.append(kind)
        length.append(L)
        s += L
        s_out.append(s)
        R.extend(float(x) for x in Rk.T.reshape(-1))
        w_in.append(w_GeV * 1e9)
        w_GeV = w_after
        w_out.append(w_GeV * 1e9)
    Path(a.out).write_text(json.dumps({
        "synergia_version": getattr(synergia, "__version__", "?"), "python": sys.version.split()[0],
        "names": names, "kinds": kinds, "length": length, "s_out": s_out, "R": R, "w_in_ev": w_in, "w_out_ev": w_out,
        "warnings": warns, "mass_ev": mass_GeV * 1e9, "charge": charge,
        "p0_model": "design reference fixed; the bunch reference follows the RF (ff_rfcavity new_pref_b)",
        "gamma_check": (w_in[0] * 1e-9 + mass_GeV) / mass_GeV if w_in else math.nan}))
    return 0


if __name__ == "__main__":
    # Synergia's Bunch_simulator destructor can crash the interpreter at finalization (pybind11 dealloc
    # inside Py_FinalizeEx): the result is on disk, so leave without running the destructors
    import os
    import traceback

    try:
        code = main()
    except BaseException:                                       # noqa: BLE001 - reported verbatim to the caller
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
