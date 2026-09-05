"""IMPACT-T ``ImpactT.in`` reader and writer (PLAN_PHASE5 §5.5).

IMPACT-T (LBNL, Ji Qiang) is the time-domain sibling of IMPACT-Z: nine header records
(processor grid; time step and step count; particle count and flags; space-charge mesh;
distribution; three lines of distribution parameters; ``current energy mass charge frequency
phase``) followed by one card per element, ``length nseg mapstp type zedge value2 … /``,
positioned by the absolute starting edge ``zedge`` and terminated by a ``-99`` card.  Comment
lines start with ``!``.  See :mod:`lattix.formats.impactt.reader` for the type-code table with
its evidence in the sources and the measured conventions.
"""
from __future__ import annotations

from lattix.formats.impactt.reader import Reader, read
from lattix.formats.impactt.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
