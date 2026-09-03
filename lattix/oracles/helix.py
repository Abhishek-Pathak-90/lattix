"""HELIX oracle: ``linac_gen`` matrix tracking, in-process.

HELIX (GPL-3, private until release) is imported from ``HELIX_ROOT`` (default:
the local dev checkout) — it is never a dependency of lattix.  The adapter
mirrors ``linac_gen.tracking.matrix_tracking.compute_transfer_matrix``: the
reference particle is advanced element by element (field maps via
``advance_ref``, thin kicks via ``advance_ref``, everything else by length),
so RF elements see the correct energy.  ``Freq`` / ``SET_BEAM_ENERGY`` command
cards are applied to the reference and produce no matrix row.

Basis: HELIX phase space (x mm, x' mrad, y mm, y' mrad, Δφ deg of the
machine clock, ΔW MeV); ``rf_frequency_Hz`` carries the clock per element so
``to_common()`` can convert the phase coordinate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from lattix.oracles.base import (
    SPECIES,
    Basis,
    BeamSpec,
    OracleResult,
    Probe,
    guess_format,
    register,
)

_DEFAULT_ROOT = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3")
_DEFAULT_FREQ_MHZ = 352.21   # tracewin_parser._DEFAULT_FREQ_MHZ


def helix_root() -> Path | None:
    v = os.environ.get("HELIX_ROOT")
    p = Path(v).expanduser() if v else _DEFAULT_ROOT
    return p if (p / "linac_gen").is_dir() else None


def _import_helix() -> Path:
    root = helix_root()
    if root is None:
        raise ModuleNotFoundError("HELIX_ROOT is not set and the default checkout is absent")
    for p in (str(root), str(root / "gui")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import linac_gen  # noqa: F401

    return root


@register
class HelixOracle:
    name = "helix"
    formats = ("tracewin", "madx", "mad8", "elegant")

    def available(self) -> tuple[bool, str]:
        try:
            root = _import_helix()
        except Exception as e:  # noqa: BLE001
            return False, f"HELIX not importable: {e}"
        return True, f"linac_gen at {root}"

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None) -> OracleResult:
        root = _import_helix()
        from linac_gen.core.particle import DEUTERON, H_MINUS, PROTON
        from linac_gen.core.reference import ReferenceParticle
        from linac_gen.elements.base import FieldMapElement, ThinKickElement
        from linac_gen.tracking.matrix_tracking import get_element_matrix

        deck = Path(deck).resolve()
        fmt = fmt or guess_format(deck)
        lat, meta = self._parse(deck, fmt, beam)
        warnings = list(meta.get("warnings", [])) if isinstance(meta, dict) else []
        if beam is None:
            beam = self._beam_from_meta(meta) or BeamSpec()
        species = {"proton": PROTON, "h-": H_MINUS, "deuteron": DEUTERON}.get(beam.species.lower())
        if species is None:
            raise ValueError(f"HELIX has no species {beam.species!r} (proton, h-, deuteron)")
        f_mhz = (beam.frequency_Hz or self._first_freq_hz(lat) or _DEFAULT_FREQ_MHZ * 1e6) / 1e6
        ref = ReferenceParticle(species, w_kin=beam.kinetic_energy_eV * 1e-6, frequency=f_mhz)

        names, lengths, s_out, mats, w_in, w_out, f_in = [], [], [], [], [], [], []
        s = 0.0
        for e in lat.elements:
            cls = type(e).__name__
            if cls == "Freq":                       # machine clock switches at the card
                ref.frequency = float(e.frequency_mhz)
                continue
            if cls == "SetBeamEnergy":
                ref.w_kin = float(e.energy_MeV)
                continue
            w0, f0 = ref.w_kin, ref.frequency
            if isinstance(e, FieldMapElement):
                e.reset_run_state()
            M = np.asarray(get_element_matrix(e, ref), dtype=float)
            if isinstance(e, FieldMapElement):
                e.advance_ref(ref)
            else:
                ref.s += e.length
                if e.length > 0:
                    ref.phi_s += 360.0 * e.length / (ref.beta * ref.wavelength)
                if isinstance(e, ThinKickElement):
                    e.advance_ref(ref)
            L = float(e.length) * 1e-3
            s += L
            names.append(str(e.name))
            lengths.append(L)
            s_out.append(s)
            mats.append(M)
            w_in.append(w0 * 1e6)
            w_out.append(ref.w_kin * 1e6)
            f_in.append(f0 * 1e6)

        return OracleResult(
            engine=self.name, basis=Basis.HELIX, names=names, length=np.array(lengths),
            s_out=np.array(s_out), R_elem=np.array(mats),
            ref_kinetic_eV_in=np.array(w_in), ref_kinetic_eV_out=np.array(w_out),
            mass_eV=float(species.mass) * 1e6, charge=int(species.charge),
            rf_frequency_Hz=np.array(f_in), warnings=warnings,
            meta={"root": str(root), "format": fmt, "n_elements": len(lat.elements),
                  "probe": "not implemented in Phase 0"},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(deck: Path, fmt: str, beam: BeamSpec | None):
        if fmt == "madx":
            from linac_gen.io.madx_parser import parse_madx

            lat, meta = parse_madx(str(deck))[:2]
        elif fmt == "mad8":
            from linac_gen.io.mad8_parser import parse_mad8

            lat, meta = parse_mad8(str(deck))[:2]
        elif fmt == "elegant":
            from linac_gen.io.elegant_parser import parse_elegant

            kw = {}
            if beam is not None:
                kw = {"species": beam.species.upper() if beam.species.lower() == "h-" else beam.species,
                      "w_kin": beam.kinetic_energy_eV * 1e-6,
                      "frequency": (beam.frequency_Hz or _DEFAULT_FREQ_MHZ * 1e6) / 1e6}
            lat, meta = parse_elegant(str(deck), **kw)[:2]
        else:
            from linac_gen.io.tracewin_parser import parse_tracewin

            lat, meta = parse_tracewin(str(deck))
        return lat, meta

    @staticmethod
    def _beam_from_meta(meta) -> BeamSpec | None:
        ref = meta.get("reference") if isinstance(meta, dict) else None
        if ref is None:
            return None
        mass_eV = float(ref.species.mass) * 1e6
        charge = int(ref.species.charge)
        species = next((n for n, (m, q) in SPECIES.items()
                        if abs(m - mass_eV) / m < 1e-4 and q == charge), "proton")
        return BeamSpec(species=species, kinetic_energy_eV=float(ref.w_kin) * 1e6,
                        frequency_Hz=float(ref.frequency) * 1e6)

    @staticmethod
    def _first_freq_hz(lat) -> float | None:
        for e in lat.elements:
            if type(e).__name__ == "Freq":
                return float(e.frequency_mhz) * 1e6
            f = getattr(e, "frequency", None)
            if f:
                return float(f) * 1e6
        return None
