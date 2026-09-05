"""Ocelot worker: run a lattice module (or Ocelot's own Elegant converter) and dump per-element first-order
maps as JSON.

Started by :mod:`lattix.oracles.ocelot` as ``<python> -I ocelot_worker.py lattice.py --out r.json …`` inside
the Ocelot environment (GPL-3; nothing here imports lattix).  The maps are what
``MagneticLattice.transfer_maps`` composes: every element's transformations ``elem.R(E)`` at the total
energy ``E`` [GeV] the reference has when it reaches the element, ``E`` advanced by each transformation's
``get_delta_e()`` (a ``Cavity`` gains ``v·cos(phi)``).  Ocelot's basis is MAD-X's
``(x, px, y, py, τ = c·Δt late-positive, ΔE/(p0 c))`` and its maps assume an electron.

``--dump`` writes the sequence itself (constructor type and numeric parameters per element) for the
lattix reader's fallback on modules that build their lattice with code.
"""
from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
import warnings
from pathlib import Path


def _load(deck: str, elegant: bool):
    from ocelot.cpbd.magnetic_lattice import MagneticLattice

    if elegant:
        from ocelot.adaptors.elegant_lattice_converter import ElegantLatticeConverter

        cell = ElegantLatticeConverter().elegant2ocelot(deck)
        return MagneticLattice(cell), None
    ns = runpy.run_path(deck)
    tws_E = None
    for v in ns.values():
        if type(v).__name__ == "Twiss" and getattr(v, "E", None):
            tws_E = float(v.E)
    lat = ns.get("lattice")
    if not isinstance(lat, MagneticLattice):
        lat = next((v for v in ns.values() if isinstance(v, MagneticLattice)), None)
    if lat is None:
        cell = ns.get("cell")
        if cell is None:
            raise SystemExit("the module defines neither a MagneticLattice nor a `cell`")
        lat = MagneticLattice(cell)
    return lat, tws_E


def _params(elem) -> dict:
    atom = getattr(elem, "element", elem)
    out = {}
    for k, v in vars(atom).items():
        if k.startswith("_") or k in ("id", "params", "has_edge"):
            continue
        if isinstance(v, bool | int | float | str):
            out[k] = v
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, list | tuple):
            out[k] = list(v)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("deck")
    p.add_argument("--out", required=True)
    p.add_argument("--ke-ev", type=float, default=0.0)
    p.add_argument("--mass-ev", type=float, default=510998.95)
    p.add_argument("--charge", type=float, default=-1.0)
    p.add_argument("--species", default="electron")
    p.add_argument("--elegant", action="store_true", help="the deck is an Elegant .lte: Ocelot's own converter")
    p.add_argument("--dump", action="store_true", help="write the sequence (types and parameters), no maps")
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    import numpy as np
    import ocelot

    lat, tws_E = _load(a.deck, a.elegant)
    if a.dump:
        Path(a.out).write_text(json.dumps({
            "ocelot_version": getattr(ocelot, "__version__", "?"), "energy_GeV": tws_E,
            "elements": [{"eid": str(e.id), "type": type(e).__name__, "params": _params(e)} for e in lat.sequence]}))
        return 0
    from ocelot.common.globals import m_e_GeV

    # Ocelot's γ is E/m_e with its own electron mass (CODATA 1998, 9e-8 below lattix's): the total
    # energy handed to Ocelot is the one that gives lattix's γ, so the maps see the same β, γ; the
    # reported kinetic energies are lattix's, advanced by exactly each transformation's ΔE
    mass = float(a.mass_ev)
    w = float(a.ke_ev)

    def total(w_ev: float) -> float:
        return (1.0 + w_ev / mass) * m_e_GeV

    E = total(w)
    names, kinds, length, s_out, R, w_in, w_out, warns = [], [], [], [], [], [], [], []
    if a.species.lower() not in ("electron", "positron"):
        warns.append(f"Ocelot's maps assume an electron; the {a.species} beam is tracked with m_e in the "
                     "longitudinal and cavity terms (report only)")
    s = 0.0
    for e in lat.sequence:
        kind = type(e).__name__
        L = float(getattr(e, "l", 0.0) or 0.0)
        M = np.eye(6)
        dE = 0.0
        try:
            for Rb, tm in zip(e.R(E), e.tms, strict=True):
                M = np.asarray(Rb, dtype=float) @ M
                dE += float(tm.get_delta_e() or 0.0)
        except Exception as exc:  # noqa: BLE001 - an element Ocelot cannot map is reported, not fatal
            M = np.eye(6)
            warns.append(f"{e.id}: {kind} has no first-order map in Ocelot ({exc}); identity used")
        if not np.all(np.isfinite(M)):
            warns.append(f"{e.id}: {kind} map is not finite; identity used")
            M = np.eye(6)
        names.append(str(e.id))
        kinds.append(kind)
        length.append(L)
        s += L
        s_out.append(s)
        R.extend(float(x) for x in M.reshape(-1))
        w_in.append(w)
        w += dE * 1e9
        E = total(w)
        w_out.append(w)
    Path(a.out).write_text(json.dumps({
        "ocelot_version": getattr(ocelot, "__version__", "?"), "python": sys.version.split()[0],
        "names": names, "kinds": kinds, "length": length, "s_out": s_out, "R": R, "w_in_ev": w_in,
        "w_out_ev": w_out, "warnings": warns, "species": a.species, "energy_GeV": tws_E,
        "p0_model": "follows the cavities (E advanced by every transformation's get_delta_e)",
        "m_e_GeV_ocelot": float(m_e_GeV), "gamma_start": 1.0 + float(a.ke_ev) / mass if mass else math.nan}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
