"""ImpactX (AMReX ``inputs`` file and Python script) reader/writer.

Two deck flavours share one element model:

``flavor="python"``
    a runnable ``impactx`` Python script (``ImpactX()``, ``init_grids()``,
    ``elements.KnownElementsList``, ``sim.lattice.extend(...)``).  Written only —
    :class:`Reader` does not execute or parse Python (``IMPACTX_PYTHON_NOT_READ``).

``flavor="inputs"``
    the AMReX ParmParse ``inputs`` file (``lattice.elements``, ``<name>.type``,
    ``beam.kin_energy``).  Read *and* written, and a write∘read fixed point.

ImpactX also reads a MAD-X subset itself (``KnownElementsList.load_file``); on the
lattix side a ``.madx`` deck is read by :mod:`lattix.formats.madx`, which is a
superset of what ImpactX's own parser accepts.

Registered in ``lattix/formats/base.py`` as::

    "impactx": FormatSpec("impactx", (".impactx.in", ".impactx.py"), "lattix.formats.impactx",
                          description="ImpactX inputs / python"),

so ``lattix convert deck.madx out.impactx.in`` writes the inputs flavour and
``out.impactx.py`` the script: :func:`flavor_for_path` picks the flavour from the file
name whenever the caller (the registry) passes no explicit ``flavor=``.
"""
from __future__ import annotations

from lattix.formats.impactx.reader import Reader
from lattix.formats.impactx.writer import Writer, flavor_for_path, to_elements

__all__ = ["Reader", "Writer", "flavor_for_path", "to_elements"]
