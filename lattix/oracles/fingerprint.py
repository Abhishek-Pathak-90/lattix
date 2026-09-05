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
              "impactz": True, "madx": False, "xtrack": False, "lightwin": True, "cheetah": True,
              "pyorbit": True, "impactt": True, "ocelot": True}
#: engines whose maps assume one species (Ocelot divides by m_e): their fingerprint beam
BEAM_SPECIES = {"ocelot": "electron"}

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

# LightWin's Envelope3D has no thin-gap model: its cavity is a 1-D field map (constant Ez over 2 cm,
# amplitude chosen for ~1 MV effective at β = 0.0668) driven at φs = −30° through SET_SYNC_PHASE.
_FP_MAP_L_M, _FP_MAP_NZ = 0.02, 200
_FP_MAP_E0 = V_VOLT / (_FP_MAP_L_M * 0.957) * 1e-6          # MV/m; T(β) ≈ 0.957 for 2 cm at 2.1 MeV
_FP_MAP_TEXT = (f"{_FP_MAP_NZ} {_FP_MAP_L_M:.6g}\n1.0\n"
                + "".join(f"{_FP_MAP_E0:.9g}\n" for _ in range(_FP_MAP_NZ + 1)))
DECKS["lightwin"] = {
    "drift": ("tracewin", f"FREQ {FREQ_HZ * 1e-6:.6g}\nDRIFT 1000 30\nEND\n"),
    "cavity": ("tracewin", f"FREQ {FREQ_HZ * 1e-6:.6g}\nDRIFT 500 30\nSET_SYNC_PHASE\n"
                           f"FIELD_MAP 100 {_FP_MAP_L_M * 1e3:.6g} {PHI_S_DEG:.6g} 30 0 1 0 0 fp_map 0\n"
                           "DRIFT 500 30\nEND\n", {"fp_map.edz": _FP_MAP_TEXT}),
}

def _cheetah_doc(elements: dict, order: list[str]) -> str:
    import json as _json

    info = (f'# lattix: reference species="proton" mass_eV={_MASS:.12g} charge=1 kinetic_energy_eV={KE_EV:.12g} '
            f'rf_frequency_Hz={FREQ_HZ:.12g}')
    return _json.dumps({"version": "cheetah-0.8", "title": "fp", "info": info, "root": "fp",
                        "elements": elements, "lattices": {"fp": order}}, indent=1)


# Cheetah: gain = −voltage·q·cos(phase) → voltage = −V for a proton, and its phase runs the other way
# (r65 ∝ sin(phase) with τ late-positive: a late particle gains more for phase > 0) → phase = −φs;
# a zero-length cavity is inf in Cheetah's own matrix (the oracle tracks 1 µm), so the fingerprint
# cavity is written thin like a GAP
DECKS["cheetah"] = {
    "drift": ("cheetah", _cheetah_doc({"d": ["Drift", {"length": 1.0}]}, ["d"])),
    "cavity": ("cheetah", _cheetah_doc({"d1": ["Drift", {"length": 0.5}],
                                        "c": ["Cavity", {"length": 0.0, "voltage": -V_VOLT, "phase": -PHI_S_DEG,
                                                         "frequency": FREQ_HZ, "cavity_type": "standing_wave"}],
                                        "d2": ["Drift", {"length": 0.5}]}, ["d1", "c", "d2"])),
}
_SUFFIX["cheetah"] = ".cheetah.json"


def _pyorbit_xml(body: str, length: float) -> str:
    tag = (f'# lattix: reference species="proton" mass_eV={_MASS:.12g} charge=1 kinetic_energy_eV={KE_EV:.12g} '
           f'rf_frequency_Hz={FREQ_HZ:.12g}')
    return ('<?xml version="1.0" ?>\n<!--\n' + tag + '\n-->\n<lattix>\n'
            f' <FP bpmFrequency="{FREQ_HZ:.6g}" length="{length:.6g}" name="FP">\n'
            '  <accElement length="0.0" name="START" pos="0.0" type="MARKER"><parameters/></accElement>\n'
            + body
            + f'  <accElement length="0.0" name="END" pos="{length:.6g}" type="MARKER"><parameters/></accElement>\n'
            ' </FP>\n</lattix>\n')


# PyORBIT: a thin RFGAP with E0TL in GeV and the phase in degrees; ΔE = q·E0TL·cos(phase)
DECKS["pyorbit"] = {
    "drift": ("pyorbit", _pyorbit_xml("", 1.0)),
    "cavity": ("pyorbit", _pyorbit_xml(
        f'  <Cavities><Cavity ampl="1.0" frequency="{FREQ_HZ:.6g}" name="C1" pos="0.5"/></Cavities>\n'
        f'  <accElement length="0.0" name="C" pos="0.5" type="RFGAP"><parameters E0L="{V_VOLT * 1e-9:.9g}" '
        f'E0TL="{V_VOLT * 1e-9:.9g}" EzFile="" aperture="0.03" aprt_type="1" cavity="C1" mode="0" '
        f'phase="{PHI_S_DEG:.6g}"/><TTFs beta_max="1.0" beta_min="0.0"><polyT order="0" pcoefs="1.0"/>'
        '<polyS order="0" pcoefs="0.0"/><polyTP order="0" pcoefs="0.0"/><polySP order="0" pcoefs="0.0"/>'
        '</TTFs></accElement>\n', 1.0)),
}
_SUFFIX["pyorbit"] = ".pyorbit.xml"
#: engines whose fingerprint cavity is an integrated field map, not a thin gap of exactly V_VOLT


def _impactt_deck(cavity: bool) -> tuple[str, str, dict[str, str]]:
    """The drift / thin-cavity decks through lattix's own IMPACT-T writer (a raised-cosine surrogate
    cavity stands in for the thin gap; its rfdata file travels with the deck)."""
    from lattix.formats.impactt import Writer
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    if cavity:
        els = [Drift(name="d1", length=0.5),
               RFCavity(name="c", length=0.0, rf=RFP(frequency_Hz=FREQ_HZ, voltage_V=V_VOLT,
                                                    phase_rad=math.radians(PHI_S_DEG))),
               Drift(name="d2", length=0.5)]
    else:
        els = [Drift(name="d", length=1.0)]
    text, files = Writer().render(Lattice.from_sequence("fp", els, ref))
    return "impactt", text, files


DECKS["impactt"] = {"drift": _impactt_deck(False), "cavity": _impactt_deck(True)}
_SUFFIX["impactt"] = ".impactt.in"
GAIN_RTOL = {"lightwin": 0.05, "impactt": 1e-5}     # IMPACT-T: MEASURED 4e-7 (surrogate gap, 1 ps step)


def _ocelot_deck(cavity: bool) -> tuple[str, str]:
    """The drift / thin-cavity decks through lattix's own Ocelot writer for an electron (Ocelot's maps
    divide by m_e): the thin gap becomes the short surrogate cavity Ocelot's matrix needs."""
    from lattix.formats.ocelot import Writer
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("electron"), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    if cavity:
        els = [Drift(name="d1", length=0.5),
               RFCavity(name="c", length=0.0, rf=RFP(frequency_Hz=FREQ_HZ, voltage_V=V_VOLT,
                                                    phase_rad=math.radians(PHI_S_DEG))),
               Drift(name="d2", length=0.5)]
    else:
        els = [Drift(name="d", length=1.0)]
    return "ocelot", Writer().render(Lattice.from_sequence("fp", els, ref))


DECKS["ocelot"] = {"drift": _ocelot_deck(False), "cavity": _ocelot_deck(True)}
_SUFFIX["ocelot"] = ".ocelot.py"


def write_decks(engine: str, workdir: Path) -> dict[str, tuple[Path, str]]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, entry in DECKS[engine].items():
        fmt, text = entry[0], entry[1]
        p = workdir / f"fp_{kind}{_SUFFIX[fmt]}"
        p.write_text(text)
        for name, content in (entry[2] if len(entry) > 2 else {}).items():
            (workdir / name).write_text(content)          # e.g. a field-map file next to the deck
        out[kind] = (p, fmt)
    return out


def fingerprint(engine: str, workdir: Path) -> dict:
    """Run both decks through *engine*; return measured + expected numbers."""
    sp = BEAM_SPECIES.get(engine, "proton")
    beam = BeamSpec(species=sp, kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ)
    decks = write_decks(engine, Path(workdir))
    o = get_oracle(engine)

    dr = o.run(decks["drift"][0], fmt=decks["drift"][1], beam=beam, workdir=Path(workdir) / "drift")
    i = int(np.argmax(dr.length))
    R_nat = dr.R_elem[i]
    R_com = dr.to_common().R_elem[i]
    expect = drift_common(float(dr.length[i]), KE_EV, SPECIES[sp][0])

    cv = o.run(decks["cavity"][0], fmt=decks["cavity"][1], beam=beam, workdir=Path(workdir) / "cavity")
    j = int(np.argmax(np.abs(cv.R_elem[:, 5, 4])))
    C_nat = cv.R_elem[j]
    C_com = cv.to_common().R_elem[j]
    gain = float(cv.ref_kinetic_eV_out[j] - cv.ref_kinetic_eV_in[j])

    return {
        "engine": engine,
        "basis": dr.basis.value,
        "species": sp,
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
