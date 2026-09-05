"""Cheetah (LatticeJSON) lattices: the JSON files ``cheetah.latticejson.save_cheetah_model`` writes and
``load_cheetah_model`` reads (Cheetah 0.8, GPL-3 — lattix touches only the JSON, never imports Cheetah;
the engine runs in its own environment through :mod:`lattix.oracles.cheetah`).

Measured on Cheetah 0.8.4 (2026-09-05, docs/oracles.md):

* 7×7 augmented first-order maps in ``(x, px/p0, y, py/p0, tau [m], ΔE/(p0 c))`` — MAD-X's basis
  (``tau`` late-positive: a 1 m drift at 2.1 MeV gives ``R56 = −L/(β²γ²)``); the reference energy
  follows the cavities (``Beam.energy`` after ``track``).
* ``Cavity``: gain ``ΔE = −voltage · q · cos(phase)`` with ``phase`` in degrees, so the IR's
  species-independent ``V·cos φ`` is ``voltage = −V/q`` (protons get a negative voltage); the
  slope ``r65 ∝ sin(phase)`` with ``τ`` late-positive means a late particle gains more for
  ``phase > 0``, the opposite of the IR's ``φ < 0`` bunching convention: ``phase = −φ`` (degrees).
  The R-matrix is Rosenzweig–Serafini's and divides by the length: a zero-length cavity is
  ``inf`` (the oracle substitutes 1 µm).
* ``Quadrupole.k1`` focuses ``x`` when positive for every species (no charge in the map), so
  ``k1 = Bn1 / Bρ_signed``; ``Solenoid.k = B0/(2 Bρ)`` (half of MAD's ``ks``) and its rotation
  ignores the charge too: ``k = Bsol/(2 Bρ_signed)``.
* ``Dipole``: arc ``length``, ``angle``, sector-referenced ``dipole_e1/e2`` (``RBend`` adds
  ``angle/2`` itself), ``gap = 2·hgap``, ``fringe_integral`` (enters the linear edge map),
  ``tilt``; ``k1`` normalized like the quadrupole's.
* Correctors kick ``px += angle`` (rad, species-agnostic, like the IR); ``Aperture``, ``BPM``,
  ``Screen`` and ``Marker`` are zero-length identities.
"""
from lattix.formats.cheetah.reader import Reader, read
from lattix.formats.cheetah.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
