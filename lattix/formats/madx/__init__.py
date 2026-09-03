"""MAD-X format support (PLAN §6 tasks 1.4 / 1.5).

    from lattix.formats.madx import Reader, Writer

The reader drives a real MAD-X through cpymad, so expressions, ``CALL`` chains
and sequence expansion are MAD-X's own; the writer is table driven (``RULES``
covers every IR kind) and records a fidelity entry for every element.
"""
from lattix.formats.madx.naming import MAX_NAME_LEN, NameMap, name_tag, parse_tags, sanitize
from lattix.formats.madx.reader import Reader
from lattix.formats.madx.writer import Rule, Writer

__all__ = ["MAX_NAME_LEN", "NameMap", "Reader", "Rule", "Writer", "name_tag", "parse_tags",
           "sanitize"]
