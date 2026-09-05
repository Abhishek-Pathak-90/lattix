"""PyORBIT3 worker: build a linac lattice from its XML, generate transport matrices, dump JSON.

Started by :mod:`lattix.oracles.pyorbit` as ``<python> -I pyorbit_worker.py deck.xml --out r.json …`` in
the PyORBIT3 environment.  Nothing here imports lattix.

PyORBIT's ``LinacTrMatricesController`` puts a ``LinacTrMatrixGenNode`` at the *entrance* of every
node and each of them holds the 7×7 map from the first node to itself (measured 2026-09-05), so the
map of node ``k`` is ``R_cum[k+1] · R_cum[k]⁻¹`` — one more node at the exit of the last element
closes the chain.  The maps are fitted from a probe bunch of ±1e-5 offsets on every coordinate
(the "initial coordinates" attributes of the particles), tracked twice (eps and eps/2) and
Richardson-extrapolated to cancel the O(eps²) residue of the tracker's third-order terms; coordinates are PyORBIT's
``(x [m], x', y [m], y', z [m] ahead-positive, dE [GeV])``.  The reference energy follows the gaps
(``kinEnergy`` of the synchronous particle at each node).  ``libfftw3`` is preloaded on macOS, where
the meson build leaves ``orbit.core`` without it (flat-namespace symbol lookup).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _preload_fftw() -> str | None:
    prefix = Path(sys.prefix)
    for name in ("libfftw3.3.dylib", "libfftw3.dylib", "libfftw3.so.3", "libfftw3.so"):
        cand = prefix / "lib" / name
        if cand.exists():
            try:
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                return str(cand)
            except OSError:
                continue
    return None


def _sequence_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [child.tag for child in root if child.get("length") is not None]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("deck")
    p.add_argument("--out", required=True)
    p.add_argument("--ke-ev", type=float, required=True)
    p.add_argument("--mass-ev", type=float, required=True)
    p.add_argument("--charge", type=float, default=1.0)
    p.add_argument("--gap-model", default="BaseRfGap", choices=("BaseRfGap", "MatrixRfGap", "RfGapTTF"))
    p.add_argument("--max-drift", type=float, default=1.0e6, help="factory maxDriftLength (no splitting by default)")
    p.add_argument("--probe-scale", type=float, default=1.0, help="multiplies the probe offsets (convergence checks)")
    a = p.parse_args(argv)
    fftw = _preload_fftw()
    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.linac import BaseRfGap, MatrixRfGap, RfGapTTF
    from orbit.lattice import AccNode
    from orbit.py_linac.lattice import LinacTrMatricesController, LinacTrMatrixGenNode
    from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory

    deck = Path(a.deck)
    names = _sequence_names(deck)
    factory = SNS_LinacLatticeFactory()
    factory.setMaxDriftLength(a.max_drift)
    lat = factory.getLinacAccLattice(names, str(deck))
    model = {"BaseRfGap": BaseRfGap, "MatrixRfGap": MatrixRfGap, "RfGapTTF": RfGapTTF}[a.gap_model]
    for gap in lat.getRF_Gaps():
        gap.setCppGapModel(model())
    ctrl = LinacTrMatricesController()
    nodes = list(lat.getNodes())
    tr = list(ctrl.addTrMatrixGenNodes(lat, nodes))
    last = LinacTrMatrixGenNode(ctrl, "lattix:exit:trMatrx")
    nodes[-1].addChildNode(last, AccNode.EXIT)
    tr.append(last)
    for t in tr:
        t.setTwissWeightUse(False, False, False)
    mass_gev, ke_gev = a.mass_ev * 1e-9, a.ke_ev * 1e-9
    b = Bunch()
    b.mass(mass_gev)
    b.charge(a.charge)
    b.getSyncParticle().kinEnergy(ke_gev)
    # probe offsets: 10 µm / 10 µrad transversely, 10 µm in z, 100 eV in dE (a first-order fit
    # of a nonlinear tracker: 1e-5 GeV on a 2.1 MeV beam already costs 1e-5 on R56)
    eps = [e * a.probe_scale for e in (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-7)]

    def probe_bunch(scale: float) -> Bunch:
        pb = Bunch()
        pb.mass(mass_gev)
        pb.charge(a.charge)
        pb.getSyncParticle().kinEnergy(ke_gev)
        for i in range(6):
            for s in (+1.0, -1.0):
                c = [0.0] * 6
                c[i] = s * eps[i] * scale
                pb.addParticle(*c)
        pb.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return pb

    def cumulated() -> list:
        return [np.array([[t.getTransportMatrix().get(i, j) for j in range(7)] for i in range(7)]) for t in tr]

    lat.trackDesignBunch(b)
    # The least-squares fit of symmetric probes is clean to second order, but the third-order
    # terms of the exact tracker leave an O(eps²) residue (2e-7 relative over a 5 m chicane at
    # 1e-5, 2e-9 at 1e-6); a second pass at eps/2 removes it by Richardson extrapolation.
    lat.trackBunch(probe_bunch(1.0))
    cums_full = cumulated()
    lat.trackBunch(probe_bunch(0.5))
    cums_half = cumulated()
    cums = [(4.0 * h - f) / 3.0 for f, h in zip(cums_full, cums_half, strict=True)]
    out_names, kinds, length, s_out, R, w_in, w_out = [], [], [], [], [], [], []
    s = 0.0
    for k, n in enumerate(nodes):
        r = cums[k + 1] @ np.linalg.inv(cums[k])
        out_names.append(str(n.getName()))
        kinds.append(type(n).__name__)
        L = float(n.getLength())
        length.append(L)
        s += L
        s_out.append(s)
        R.extend(float(x) for x in r[:6, :6].reshape(-1))
        g_in, g_out = float(tr[k].getGamma()), float(tr[k + 1].getGamma())
        w_in.append((g_in - 1.0) * a.mass_ev)
        w_out.append((g_out - 1.0) * a.mass_ev)
    Path(a.out).write_text(json.dumps({
        "python": sys.version.split()[0], "fftw_preloaded": fftw, "sequences": names, "gap_model": a.gap_model,
        "names": out_names, "kinds": kinds, "length": length, "s_out": s_out, "R": R, "w_in_ev": w_in,
        "w_out_ev": w_out, "lattice_length": float(lat.getLength()),
        "p0_model": "follows the RF gaps (synchronous particle)", "warnings": []}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
