"""FLAME GLPS support (PLAN §6 task 3.4).

    from lattix.formats.flame import Reader, Writer

FLAME (the FRIB Linear Accelerator Modelling Engine, `frib-high-level-controls/FLAME
<https://github.com/frib-high-level-controls/FLAME>`_) reads *GLPS* decks — a
``name: type, key = value, …;`` dialect with vectors, strings, ``name: LINE = (…);``
and ``USE: name;``.  The reader is a hand port of FLAME's own grammar
(``src/glps.y``/``src/glps.l``); FLAME's ``GLPSParser`` is used in the tests as an
independent cross-check, never as the implementation.

FLAME is a *per-nucleon* code: ``IonEs``/``IonEk`` are eV/u and ``IonChargeStates``
are charge-to-mass ratios, so the reader recovers ``(Q, A)`` and the writer divides
by ``A`` again (see :mod:`lattix.formats.flame.reader`).
"""
from lattix.formats.flame.reader import GLPSError, Reader, read
from lattix.formats.flame.writer import KNOWN_CAVTYPES, NameMap, Rule, Writer, dumps, sanitize, write

__all__ = ["GLPSError", "KNOWN_CAVTYPES", "NameMap", "Reader", "Rule", "Writer", "dumps",
           "read", "sanitize", "write"]
