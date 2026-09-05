"""``Reader``/``Writer`` for xtrack JSON decks (``Line.to_json`` / ``Line.from_json``).

Both the *line* form (``{"elements": {...}, "element_names": [...], "particle_ref": …}``)
and the *environment* form (``{"elements": {...}, "lines": {name: …}, …}`` — what
``Environment.to_json`` writes) are accepted; for an environment the ``line=`` option
picks the sequence (the only one, or the last one, otherwise).

Sniffing.  The suffix is a plain ``.json``, which lattix also uses for its own
``*.lattix.json`` (and PALS for ``*.pals.json``), so the registry entry must come **after**
those two in ``FORMATS`` and content sniffing settles ambiguous cases:
:func:`sniff_json` accepts a document that has an ``elements`` mapping together with
``element_names``/``lines``, or an ``xtrack_version``/``xsuite_data_type`` key, and rejects
lattix's own IR JSON (``kind``-tagged elements plus ``reference``).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.formats.xtrack.convert import RULES, Rule, from_line, to_line
from lattix.ir.elements import ALL_KINDS
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, Species, species

#: keys that identify an xtrack document without loading xtrack itself.
_LINE_KEYS = ("element_names", "lines")
_MARKER_KEYS = ("xtrack_version", "xsuite_data_type")


def sniff_json(path: str | Path) -> bool:
    """True when *path* looks like an xtrack ``Line``/``Environment`` JSON document."""
    try:
        with open(path, encoding="utf-8") as fh:
            head = fh.read(4096)
    except OSError:                                        # pragma: no cover - caller's problem
        return False
    if '"elements"' not in head:
        return False
    if any(f'"{k}"' in head for k in _MARKER_KEYS):
        return True
    if '"reference"' in head and '"kind"' in head:          # lattix's own IR JSON
        return False
    return any(f'"{k}"' in head for k in _LINE_KEYS)


def load_document(path: str | Path, *, line: str | None = None):
    """``(xt.Line, note)`` from an xtrack JSON file (line or environment form)."""
    import xtrack as xt

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{Path(path).name}: not an xtrack JSON document (top level is "
                         f"{type(doc).__name__}, expected an object)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if "lines" in doc:
            env = xt.Environment.from_dict(doc)
            available = list(env.lines.keys())
            if not available:
                raise ValueError(f"{Path(path).name}: the environment defines no line")
            if line is None:
                if len(available) > 1:
                    meta_use = ((doc.get("metadata") or {}).get("lattix") or {}).get("use")
                    chosen = meta_use if meta_use in available else available[-1]
                    extra = {n: list((doc["lines"][n].get("composer") or {}).get("components") or [])
                             for n in available if n != chosen}
                    env.lines[chosen]._lattix_extra_lines = {k: v for k, v in extra.items() if v}
                    env.lines[chosen]._lattix_chosen = chosen
                    env.lines[chosen]._lattix_root_components = list(
                        (doc["lines"][chosen].get("composer") or {}).get("components") or [])
                    return env.lines[chosen], (f"the environment defines {len(available)} lines "
                                               f"{available}; {chosen!r} was used (pass line=)")
                chosen = available[0]
            elif line in available:
                chosen = line
            else:
                raise KeyError(f"{Path(path).name}: no line {line!r} (has {available})")
            extra = {n: list((doc["lines"][n].get("composer") or {}).get("components") or [])
                     for n in available if n != chosen}
            env.lines[chosen]._lattix_extra_lines = {k: v for k, v in extra.items() if v}
            env.lines[chosen]._lattix_chosen = chosen
            env.lines[chosen]._lattix_root_components = list(
                (doc["lines"][chosen].get("composer") or {}).get("components") or [])
            return env.lines[chosen], None
        if "element_names" not in doc:
            raise ValueError(f"{Path(path).name}: neither 'element_names' (a Line) nor 'lines' "
                             f"(an Environment) is present")
        return xt.Line.from_dict(doc), None


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)`` for an xtrack JSON deck."""

    format = "xtrack"

    def read(self, path: Path, *, line: str | None = None, strict: bool = False,
             species: str | Species | None = None, kinetic_energy_eV: float | None = None,
             energy_mode: str | None = None, name: str | None = None,
             ) -> tuple[Lattice, FidelityReport]:
        rep = FidelityReport(source_format="xtrack", source_file=str(path))
        xline, note = load_document(path, line=line)
        if note:
            rep.equivalent("MULTI_LINE_ENVIRONMENT", note)
        ref = _reference_override(xline, species, kinetic_energy_eV)
        lat = from_line(xline, ref, report=rep, name=name,           # metadata keeps the lattice name
                        energy_mode=energy_mode,
                        extra_lines=getattr(xline, "_lattix_extra_lines", None),
                        root_components=getattr(xline, "_lattix_root_components", None))
        chosen = getattr(xline, "_lattix_chosen", None)
        if chosen and chosen != lat.use and lat.use in lat.lines:
            lat.lines[chosen] = lat.lines.pop(lat.use).model_copy(update={"name": chosen})
            lat.use = chosen
        for el in lat.elements.values():
            if el.provenance is not None:
                el.provenance.file = str(path)
        rep.raise_if(strict)
        return lat, rep


class Writer:
    """``Writer().write(lattice, path)`` → an xtrack ``Line`` JSON deck."""

    format = "xtrack"
    RULES: dict[str, Rule] = RULES

    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              energy_mode: str = "delta", install_apertures: bool = True,
              indent: int = 1, name: str | None = None, document: str = "auto",
              bend_model: str | None = None, edge_model: str | None = None, rbend: bool = False) -> FidelityReport:
        """``document``: ``"line"`` (a flat ``Line`` JSON), ``"environment"`` (an ``Environment``
        JSON whose lines mirror the IR's nested lines) or ``"auto"`` (environment when the IR has
        more than one line)."""
        if document not in ("auto", "line", "environment"):
            raise ValueError(f"document must be 'auto', 'line' or 'environment', got {document!r}")
        rep = FidelityReport(target_format="xtrack", target_file=str(path))
        nested = len(lattice.lines) > 1
        if document == "environment" or (document == "auto" and nested):
            from lattix.formats.xtrack.convert import to_environment

            env = to_environment(lattice, energy_mode=energy_mode, report=rep, install_apertures=install_apertures)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                env.to_json(str(path), indent=indent)
            rep.raise_if(strict)
            return rep
        line = to_line(lattice, energy_mode=energy_mode, report=rep,
                       install_apertures=install_apertures, name=name,
                       bend_model=bend_model, edge_model=edge_model, rbend=rbend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            line.to_json(str(path), indent=indent)
        rep.raise_if(strict)
        return rep


def _reference_override(xline, sp: str | Species | None,
                        kinetic_energy_eV: float | None) -> ReferenceParticle | None:
    """Explicit ``species=``/``kinetic_energy_eV=`` beat the line's ``particle_ref``."""
    if sp is None and kinetic_energy_eV is None:
        return None
    import numpy as np

    pref = getattr(xline, "particle_ref", None)
    ke = kinetic_energy_eV
    if ke is None:
        if pref is None:
            raise ValueError("kinetic_energy_eV is required when the line has no particle_ref")
        ke = float(np.atleast_1d(pref.kinetic_energy0)[0])
    if sp is None:
        if pref is None:
            raise ValueError("species is required when the line has no particle_ref")
        from lattix.formats.xtrack.convert import _species_name

        mass_eV = float(np.atleast_1d(pref.mass0)[0])
        charge = int(round(float(np.atleast_1d(pref.q0)[0])))
        nm = _species_name(mass_eV, charge)
        s = species(nm) if nm else Species(name=f"q{charge}m{mass_eV:.6g}", mass_eV=mass_eV,
                                           charge=charge)
    else:
        s = species(sp)
    return ReferenceParticle(species=s, kinetic_energy_eV=ke)


_MISSING = set(ALL_KINDS) - set(Writer.RULES)
if _MISSING:                                        # pragma: no cover - guarded by a test too
    raise RuntimeError(f"xtrack writer RULES do not cover {sorted(_MISSING)}")
