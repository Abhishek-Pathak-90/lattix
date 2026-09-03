"""MAD8 flat-file support (PLAN §6 task 2.3).

    from lattix.formats.mad8 import Reader, Writer

The reader is a port of HELIX's ``mad8_parser`` onto the lattix IR (lazy ``:=``
resolution with ``NAME[ATTR]`` references, LINE expansion, root-line detection,
FODO periodicity brackets); the writer is the MAD-X writer's dialect —
``&`` continuations, ``LINE=(…)`` instead of ``sequence``, upper-case 16-character
names, ``!`` comments.
"""
from lattix.formats.mad8.reader import Reader, read
from lattix.formats.mad8.writer import MAX_NAME_LEN, NameMap, Rule, Writer, name_tag, sanitize, write

__all__ = ["MAX_NAME_LEN", "NameMap", "Reader", "Rule", "Writer", "name_tag", "read",
           "sanitize", "write"]
