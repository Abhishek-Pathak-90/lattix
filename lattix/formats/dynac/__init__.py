"""DYNAC decks (V6R16: cm, kG, MV, MeV, deg; a title line, unnamed type codes with their parameter lines).

DYNAC is freeware under its own EULA (redistribution of the program is free, commercialising it is not):
lattix writes and reads the deck text; the engine runs locally through :mod:`lattix.oracles.dynac` when a
built ``dynac`` binary is at hand (``LATTIX_DYNAC_EXE``).  Measured on DYNAC V6R16 built with gfortran 16
(2026-09-05, ``docs/oracles.md`` Phase 5.8):

* particle coordinates ``(x cm, x′ rad, y cm, y′ rad, φ rad late-positive w.r.t. the master frequency,
  ΔW MeV)``; a 1 m drift at 2.1 MeV gives ``R56 = −6.05 rad/MeV`` (a faster particle arrives earlier);
* ``QUADRUPO L B_tip R``: a positive pole-tip field focuses positively charged particles in x (a lab
  field: H⁻ is defocused by the same card); ``SOLENO 1 L B``: ``k = B/(2Bρ)`` with the rotation following
  the charge; ``STEER FLD NVF``: the kick is ``∫B·dl / Bρ_signed``;
* ``BMAGNET``: TRANSPORT conventions (a positive angle bends to the right, i.e. towards −x, pole-face
  angles as MAD's e1/e2, ``EK1`` the fringe integral, ``APB`` the half gap); with ``BAIM = 0`` the field is
  derived from the reference — for a negative species the field must be given explicitly and negative
  (``BAIM = −|Bρ|/ρ``) to keep the geometry; a 10° wedge with faces, fint and gap agrees with MAD-X to 1e-5;
* ``BUNCHER V φ h R``: the particles gain ``q·V·cos φ`` (the phase is charge-signed like TraceWin's raw
  phase: a negative species needs ``φ + 180°``); a late particle gains more for ``φ < 0``; the transverse
  RF kick uses the mid-gap velocity (TraceWin/HELIX use the entrance one);
* ``FIELD`` (a file of ``z [m], E_z [V/m]`` blocks, frequency first) + ``CAVNUM``: a numerically integrated
  cavity whose phase is relative to DYNAC's own crest (7e-4 above an RK4 integration of the same field);
  ``CAVNUM`` after ``HARM`` (no ``FIELD``) has no frequency and diverges;
* ``ALINER`` shifts the beam coordinates permanently (a misaligned element is a pair of opposite
  ``ALINER`` cards); ``ZROT +90`` rotates the frame like a patch tilt of +90°; ``NREF`` changes the
  reference energy only; ``CHANGREF`` yaws the reference direction.
"""
from lattix.formats.dynac.reader import Reader, read
from lattix.formats.dynac.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
