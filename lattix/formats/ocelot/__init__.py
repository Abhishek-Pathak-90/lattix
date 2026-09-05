"""Ocelot lattice files (a Python module of element constructors, a ``cell`` and
``MagneticLattice(cell)``).  Ocelot is GPL-3: lattix writes and reads the file as text (an AST
parse of literal constructors) and never imports Ocelot; the engine runs in its own environment
through :mod:`lattix.oracles.ocelot`."""
from lattix.formats.ocelot.reader import Reader, read
from lattix.formats.ocelot.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
