"""PyORBIT3 linac XML (``orbit.py_linac.linac_parsers.SNS_LinacLatticeFactory``): reader and writer.

Measured on PyORBIT3 (meson build of 2026-05-14, 2026-09-05, docs/oracles.md):

* ``<seq name length bpmFrequency>`` holds ``<accElement type name length pos>`` with ``pos`` the
  element *centre*; drifts are implicit (the factory fills the gaps between elements).
* ``QUAD field`` is the lab gradient [T/m] (``kq = field/Bρ`` with the charge in ``Bρ``: a positive
  field focuses ``x`` for a proton and defocuses it for H⁻); ``BEND theta`` [rad] with
  sector-referenced ``ea1/ea2`` and MAD-X's sign (``R16 > 0`` for ``theta > 0``); ``SOLENOID B`` is
  the *normalized* strength ``B₀/Bρ`` [1/m] (TEAPOT ``soln``: "magnetic field (1/m)"), the charge
  applied by the tracker; ``DCH/DCV B·effLength`` kick ``x' -= B·L/Bρ_signed`` and
  ``y' += B·L/Bρ_signed``.
* ``RFGAP``: ``E0TL`` in GeV (transit-time factor folded, like a TraceWin GAP), ``phase`` in degrees,
  ``ΔE = q·E0TL·cos(phase)`` — a negative particle needs ``phase + 180°`` for the IR's
  species-independent ``V·cos φ``; the gap's frequency comes from its ``<Cavity>``; the ``TTFs``
  polynomials are kept as native passthrough.
* Coordinates ``(x [m], x', y [m], y', z [m] ahead-positive, dE [GeV])`` (``Basis.PYORBIT``).
"""
from lattix.formats.pyorbit.reader import Reader, read
from lattix.formats.pyorbit.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
