"""Cheetah worker: load a LatticeJSON file and dump per-element first-order maps as JSON.

Started by :mod:`lattix.oracles.cheetah` as ``<python> -I cheetah_worker.py deck.json --out r.json …``
inside the Cheetah environment (torch).  Nothing here imports lattix.

Every element's 7×7 augmented map is evaluated with ``first_order_transfer_map(E, species)`` at the
reference *total* energy the beam has when it reaches the element, which the worker advances by each
cavity's gain ``−voltage·q·cos(phase)`` (Cheetah's ``track`` does the same to ``Beam.energy``).  The
basis is MAD-X's ``(x, px, y, py, τ, ΔE/(p0 c))``.  A zero-length cavity evaluates to ``inf`` in
Cheetah (its matrix divides by the length): the worker tracks a 1 µm copy and says so.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

THIN_CAVITY_M = 1e-6


def _flatten(element) -> list:
    out = []
    for e in getattr(element, "elements", None) or []:
        if type(e).__name__ == "Segment":
            out += _flatten(e)
        else:
            out.append(e)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("deck")
    p.add_argument("--out", required=True)
    p.add_argument("--ke-ev", type=float, required=True)
    p.add_argument("--mass-ev", type=float, required=True)
    p.add_argument("--charge", type=float, default=1.0)
    p.add_argument("--species", default="proton")
    p.add_argument("--bmad", action="store_true",
                   help="the deck is a Bmad lattice: load it with Cheetah's own converters.bmad (lockstep check)")
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    import cheetah
    import numpy as np
    import torch
    from cheetah.particles import Species

    torch.set_default_dtype(torch.float64)
    if a.bmad:
        from cheetah.converters.bmad import convert_lattice

        # Cheetah's Bmad reader is a namelist parser that rejects statements it does not model
        # (``title, "…"``); drop them from a working copy
        src = Path(a.deck)
        kept = [ln for ln in src.read_text(encoding="latin-1").splitlines()
                if not ln.strip().lower().startswith("title")]
        local = Path.cwd() / src.name
        local.write_text("\n".join(kept) + "\n")
        seg = convert_lattice(local)
    else:
        seg = cheetah.latticejson.load_cheetah_model(a.deck)
    elements = _flatten(seg)
    if not elements and type(seg).__name__ != "Segment":
        elements = [seg]
    q = int(round(a.charge))
    known = {"electron", "positron", "proton", "antiproton", "deuteron"}
    if a.species.lower() in known:
        sp = Species(a.species.lower())                 # Cheetah refuses charge/mass for its named species
    else:
        sp = Species(a.species, num_elementary_charges=torch.tensor(q), mass_eV=torch.tensor(float(a.mass_ev)))
    # total energy from Cheetah's own mass for its named species (CODATA 2022 vs lattix's 2018 value:
    # 1.3 eV on a proton, i.e. 6e-7 in β²γ² at 2.1 MeV if the total energy were passed instead)
    mass = float(sp.mass_eV.detach().reshape(-1)[0]) if hasattr(sp.mass_eV, "detach") else float(sp.mass_eV)
    E = mass + float(a.ke_ev)
    names, kinds, length, s_out, R, w_in, w_out, warns = [], [], [], [], [], [], [], []
    s = 0.0
    for e in elements:
        kind = type(e).__name__
        el = e
        L = float(e.length.detach().reshape(-1)[0]) if hasattr(e, "length") else 0.0
        if kind == "Cavity" and L <= 0.0 and float(e.voltage.detach().reshape(-1)[0]) != 0.0:
            el = e.clone()
            el.length = torch.tensor(THIN_CAVITY_M)
            warns.append(f"{e.name}: zero-length Cavity tracked as {THIN_CAVITY_M} m "
                         "(Cheetah's matrix divides by the length)")
        try:
            tm = el.first_order_transfer_map(torch.tensor(E), sp)
        except NotImplementedError:
            tm = torch.eye(7)
            warns.append(f"{e.name}: {kind} has no first-order map in Cheetah; identity used")
        M = np.asarray(tm.detach().numpy(), dtype=float)
        if M.ndim > 2:
            M = M.reshape(-1, 7, 7)[0]
        gain = 0.0
        if kind == "Cavity":
            volt = float(e.voltage.detach().reshape(-1)[0])
            phase = math.radians(float(e.phase.detach().reshape(-1)[0]))
            gain = -volt * q * math.cos(phase)
        names.append(str(e.name))
        kinds.append(kind)
        length.append(L)
        s += L
        s_out.append(s)
        R.extend(float(x) for x in M[:6, :6].reshape(-1))
        w_in.append(E - mass)
        E += gain
        w_out.append(E - mass)
    Path(a.out).write_text(json.dumps({
        "cheetah_version": getattr(cheetah, "__version__", "?"), "torch_version": torch.__version__,
        "python": sys.version.split()[0], "names": names, "kinds": kinds, "length": length, "s_out": s_out,
        "R": R, "w_in_ev": w_in, "w_out_ev": w_out, "warnings": warns, "species": a.species,
        "p0_model": "follows the cavities (energy advanced by -voltage*q*cos(phase))"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
