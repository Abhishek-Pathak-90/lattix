"""Lossless IR serialisation (``*.lattix.json``)."""
from __future__ import annotations

import json
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.ir.elements import ALL_KINDS
from lattix.ir.lattice import Lattice


class Reader:
    format = "lattix"

    def read(self, path: Path, **options) -> tuple[Lattice, FidelityReport]:
        lat = Lattice.from_dict(json.loads(Path(path).read_text()))
        return lat, FidelityReport()


class Writer:
    format = "lattix"
    RULES = {k: "exact" for k in ALL_KINDS}

    def write(self, lattice: Lattice, path: Path, *, strict: bool = False, **options) -> FidelityReport:
        Path(path).write_text(json.dumps(lattice.to_dict(), indent=1, sort_keys=True) + "\n")
        rep = FidelityReport()
        for p in lattice.flatten():
            rep.exact(p.name, p.element.kind)
        return rep
