"""Synergia (3) lattice JSON — the cereal archive ``Lattice.as_json()`` writes and ``Lattice.load_from_json``
reads: a name, a reference particle (charge, mass and total energy in GeV) and a flat list of elements
with MAD-X's type names and attributes (``k1``, ``angle``, ``volt`` [MV], ``lag`` [turns], ``freq`` [MHz],
``knl``/``ksl`` vectors …).  Synergia is Fermilab's open-source code (its own DOE licence, redistributable):
lattix writes and reads the JSON; the engine runs through :mod:`lattix.oracles.synergia` where a built
install is at hand (``LATTIX_SYNERGIA_PYTHON``; the clone's pixi build locally).

Measured conventions live in ``docs/formats/synergia.md`` (Phase 5.9).
"""
from lattix.formats.synergia.reader import Reader, read
from lattix.formats.synergia.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
