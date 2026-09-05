"""Basis fingerprints: measure each engine's longitudinal conventions.

Two hand-written decks per engine — a 1 m drift and a thin cavity at
φs = −30° — are run through the adapter and mapped onto the common basis.
The drift must reproduce the analytic map (R56 = +L/γ² with z ahead-positive);
the cavity must show R65 < 0 (a late particle gains more energy: bunching)
and, on p0-following engines, a reference energy gain of q·V·cos(30°).

The measured native-basis numbers are stored in
``tests/oracles/goldens/fingerprints.json`` so an engine upgrade that flips a
convention fails loudly (PLAN §5.1).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lattix.oracles.base import SPECIES, BeamSpec, get_oracle
from lattix.oracles.basis import drift_common

KE_EV = 2.1e6          # proton kinetic energy
FREQ_HZ = 162.5e6
V_VOLT = 1.0e6         # cavity effective voltage
PHI_S_DEG = -30.0      # synchronous phase, cos convention, 0 = crest
FOLLOWS_P0 = {"scibmad": False, "helix": True, "bmad": True, "elegant": True, "tracewin": True, "impactx": True,
              "impactz": True, "madx": False, "xtrack": False}

_MASS = SPECIES["proton"][0]
_ETOT_GEV = (_MASS + KE_EV) * 1e-9
_LAG = 0.25 + PHI_S_DEG / 360.0           # MAD-X: gain = V·sin(2π·lag)

DECKS: dict[str, dict[str, tuple[str, str]]] = {
    # engine: {"drift": (format, text), "cavity": (format, text)}
    "madx": {
        "drift": ("madx", f"beam, particle=proton, energy={_ETOT_GEV:.12g};\n"
                          "d: drift, l=1.0;\nfp: sequence, l=1.0, refer=centre;\n"
                          "d, at=0.5;\nendsequence;\nuse, sequence=fp;\n"),
        "cavity": ("madx", f"beam, particle=proton, energy={_ETOT_GEV:.12g};\n"
                           "d1: drift, l=0.5;\nd2: drift, l=0.5;\n"
                           f"c: rfcavity, l=0, volt={V_VOLT * 1e-6:.6g}, lag={_LAG:.12g}, "
                           f"freq={FREQ_HZ * 1e-6:.6g};\n"
                           "fp: sequence, l=1.0, refer=centre;\nd1, at=0.25;\nc, at=0.5;\n"
                           "d2, at=0.75;\nendsequence;\nuse, sequence=fp;\n"),
    },
    "helix": {
        "drift": ("tracewin", f"FREQ {FREQ_HZ * 1e-6:.6g}\nDRIFT 1000 30\nEND\n"),
        "cavity": ("tracewin", f"FREQ {FREQ_HZ * 1e-6:.6g}\nDRIFT 500 30\n"
                               f"GAP {V_VOLT:.6g} {PHI_S_DEG:.6g} 30\nDRIFT 500 30\nEND\n"),
    },
    "bmad": {
        "drift": ("bmad", "parameter[particle] = proton\n"
                          f"parameter[e_tot] = {_ETOT_GEV * 1e9:.12g}\n"
                          "parameter[geometry] = open\n"
                          "d: drift, l = 1.0\nfp: line = (d)\nuse, fp\n"),
        "cavity": ("bmad", "parameter[particle] = proton\n"
                           f"parameter[e_tot] = {_ETOT_GEV * 1e9:.12g}\n"
                           "parameter[geometry] = open\n"
                           "d1: drift, l = 0.5\nd2: drift, l = 0.5\n"
                           f"c: lcavity, l = 1e-3, voltage = {V_VOLT:.6g}, "
                           f"phi0 = {PHI_S_DEG / 360.0:.12g}, rf_frequency = {FREQ_HZ:.6g}\n"
                           "fp: line = (d1, c, d2)\nuse, fp\n"),
    },
    "elegant": {
        "drift": ("elegant", "d: drift, l=1.0\nfp: line=(d)\n"),
        # elegant's RFCA phase convention follows the charge sign relative to the electron:
        # negative species (e-, H-) crest at +90°, positive species (protons) at -90°
        # (MEASURED 2026-09-03: phase=60 decelerates protons, phase=240 gives +V cos 30°).
        "cavity": ("elegant", "d1: drift, l=0.5\nd2: drift, l=0.5\n"
                              f"c: rfca, l=0, volt={V_VOLT:.6g}, phase={(PHI_S_DEG - 90.0) % 360:.6g}, "
                              f"freq={FREQ_HZ:.6g}, change_p0=1\n"
                              "fp: line=(d1,c,d2)\n"),
    },
}
DECKS["xtrack"] = DECKS["madx"]
DECKS["impactx"] = DECKS["madx"]     # ImpactX reads MAD-X through lattix; ShortRF follows p0
DECKS["tracewin"] = DECKS["helix"]
# SciBmad (Beamlines.jl): phi0 in radians, measured gain −V·cos(phi0) (the writer's GAIN_SIGN), one
# reference momentum per beamline; a zero-length cavity is substituted by 1 µm in the worker
DECKS["scibmad"] = {
    "drift": ("scibmad", "using Beamlines\n@elements begin\n  d = Drift(L = 1.0)\nend\n"
                         f"fp = Beamline([d]; E_ref = {_ETOT_GEV * 1e9:.12g}, "
                         "species_ref = Species(\"proton\"))\n"),
    "cavity": ("scibmad", "using Beamlines\n@elements begin\n  d1 = Drift(L = 0.5)\n  d2 = Drift(L = 0.5)\n"
                          f"  c = RFCavity(L = 0, voltage = {-V_VOLT:.6g}, phi0 = {math.radians(PHI_S_DEG):.12g}, "
                          f"rf_frequency = {FREQ_HZ:.6g})\nend\n"
                          f"fp = Beamline([d1, c, d2]; E_ref = {_ETOT_GEV * 1e9:.12g}, "
                          "species_ref = Species(\"proton\"))\n"),
}

_SUFFIX = {"madx": ".madx", "tracewin": ".dat", "bmad": ".bmad", "elegant": ".lte", "scibmad": ".jl"}


def write_decks(engine: str, workdir: Path) -> dict[str, tuple[Path, str]]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, (fmt, text) in DECKS[engine].items():
        p = workdir / f"fp_{kind}{_SUFFIX[fmt]}"
        p.write_text(text)
        out[kind] = (p, fmt)
    return out


def fingerprint(engine: str, workdir: Path) -> dict:
    """Run both decks through *engine*; return measured + expected numbers."""
    beam = BeamSpec(species="proton", kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ)
    decks = write_decks(engine, Path(workdir))
    o = get_oracle(engine)

    dr = o.run(decks["drift"][0], fmt=decks["drift"][1], beam=beam, workdir=Path(workdir) / "drift")
    i = int(np.argmax(dr.length))
    R_nat = dr.R_elem[i]
    R_com = dr.to_common().R_elem[i]
    expect = drift_common(float(dr.length[i]), KE_EV, _MASS)

    cv = o.run(decks["cavity"][0], fmt=decks["cavity"][1], beam=beam, workdir=Path(workdir) / "cavity")
    j = int(np.argmax(np.abs(cv.R_elem[:, 5, 4])))
    C_nat = cv.R_elem[j]
    C_com = cv.to_common().R_elem[j]
    gain = float(cv.ref_kinetic_eV_out[j] - cv.ref_kinetic_eV_in[j])

    return {
        "engine": engine,
        "basis": dr.basis.value,
        "drift": {"length_m": float(dr.length[i]), "R56_native": float(R_nat[4, 5]),
                  "R65_native": float(R_nat[5, 4]), "R56_common": float(R_com[4, 5]),
                  "R56_expected": float(expect[4, 5]),
                  "max_abs_err_common": float(np.max(np.abs(R_com - expect)))},
        "cavity": {"row": cv.names[j], "R65_native": float(C_nat[5, 4]),
                   "R66_native": float(C_nat[5, 5]), "R65_common": float(C_com[5, 4]),
                   "gain_eV": gain,
                   "gain_expected_eV": V_VOLT * math.cos(math.radians(PHI_S_DEG))
                   if FOLLOWS_P0.get(engine, False) else 0.0,
                   "follows_p0": FOLLOWS_P0.get(engine, False)},
    }
