"""Bmad lattice reader and writer (PLAN §6 task 2.2).

    from lattix.formats.bmad import Reader, Writer

Bmad is the *exact* target for accelerating linacs: its ``lcavity`` moves the
reference momentum, so unlike MAD-X no energy-mode compromise is needed.  Both
directions were pinned against Tao (``lattix.oracles.pytao``) and against Bmad's
own converters ``madx_to_bmad.py`` and ``bmad_to_mad_sad_elegant``.
"""
from lattix.formats.bmad.naming import MAX_NAME_LEN, NameMap, name_tag, parse_tags, sanitize
from lattix.formats.bmad.reader import Reader, read
from lattix.formats.bmad.writer import Rule, Writer, render, write

__all__ = ["MAX_NAME_LEN", "NameMap", "Reader", "Rule", "Writer", "name_tag", "parse_tags",
           "read", "render", "sanitize", "write"]
