"""PALS (Particle Accelerator Lattice Standard) support (PLAN §6 task 2.4).

    from lattix.formats.pals import Reader, Writer

The IR was designed to mirror PALS names (PLAN §4.1); :mod:`lattix.formats.pals.reader`
documents where the standard has since moved (``BendP.angle_ref``, ``RFP.num_cells``,
``BodyShiftP.z_rot``, ``edge1_int`` = ``fint·hgap``, …).  Both modules cite
<https://github.com/pals-project/pals> at commit ``a2b1083`` (2026-09-01).
"""
from lattix.formats.pals.reader import PALS_CONSTANTS, Reader, ir_species, pals_species_name
from lattix.formats.pals.writer import NameMap, Rule, Writer, sanitize

__all__ = ["PALS_CONSTANTS", "NameMap", "Reader", "Rule", "Writer", "ir_species",
           "pals_species_name", "sanitize"]
