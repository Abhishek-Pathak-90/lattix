"""Elegant ``.lte`` support (PLAN §6 task 2.1).

    from lattix.formats.elegant import Reader, Writer

Both sides were pinned against ``elegant 2026.3.0`` itself (see the module
docstrings for what was measured): ``RBEN``'s ``L`` is the chord, ``FINT``
defaults to 0.5, ``RFCA``'s ``FREQ`` defaults to 500 MHz and ``CHANGE_P0`` to 0,
and the ``RFCA`` phase crest follows the charge sign (+90 deg for negative
species, -90 deg for positive ones).
"""
from lattix.formats.elegant.naming import MAX_NAME_LEN, NameMap, name_tag, parse_tags, sanitize
from lattix.formats.elegant.reader import (
    DEFAULT_FINT,
    DEFAULT_RF_FREQUENCY_HZ,
    DEFAULT_SPECIES,
    DEFAULT_KINETIC_ENERGY_eV,
    Reader,
    logical_statements,
)
from lattix.formats.elegant.writer import Rule, Writer

__all__ = ["DEFAULT_FINT", "DEFAULT_KINETIC_ENERGY_eV", "DEFAULT_RF_FREQUENCY_HZ",
           "DEFAULT_SPECIES", "MAX_NAME_LEN", "NameMap", "Reader", "Rule", "Writer",
           "logical_statements", "name_tag", "parse_tags", "sanitize"]
