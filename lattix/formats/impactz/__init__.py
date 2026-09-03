"""IMPACT-Z ``ImpactZ.in`` reader and writer (PLAN §6 task 3.3).

IMPACT-Z (LBNL, Ji Qiang) reads a single positional file that must be named
``ImpactZ.in``: eleven header lines of numerical/beam parameters followed by one
line per beam-line element, ``length nseg mapstp type value1 … value24 /``,
terminated by a ``-99`` card.  Comment lines start with ``!``.  Everything after
a ``/`` on a data line is ignored (Fortran list-directed input), which is where
lattix keeps element names.

See :mod:`lattix.formats.impactz.reader` for the header and type-code tables
with their evidence in the IMPACT-Z sources.
"""
from __future__ import annotations

from lattix.formats.impactz.reader import Reader
from lattix.formats.impactz.writer import Writer

__all__ = ["Reader", "Writer"]
