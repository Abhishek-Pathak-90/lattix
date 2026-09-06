"""OPAL-T input decks (OPAL-X 078ff1e; ``name: TYPE, attr=value, …;`` statements, ``ELEMEDGE`` placement).

OPAL is GPL-3: lattix writes and reads the deck text and its 1-D field-map files only; no OPAL engine runs
here (a build needs MPI, H5hut, Boost, GSL), so the conventions are taken from the source and marked as
such in ``docs/formats/opal.md``:

* elements sit at ``ELEMEDGE`` (entrance path length, m); drifts carry no field;
* ``K1``, ``K2``, ``K3``, ``KN``/``KS``, ``KS`` are multiplied by the **BEAM's** ``P0/c`` (unsigned, the
  initial momentum: ``OpalData::getP0()``) — lattix writes lab fields divided by the start ``|Bρ|``;
* ``KICKER HKICK/VKICK`` deflect the actual particle by that angle at ``DESIGNENERGY`` [MeV kinetic];
  ``SBEND ANGLE`` is geometric (positive towards −x, ``compute3DLattice``) and needs ``DESIGNENERGY``;
* ``RFCAVITY``: ``VOLT`` scales a ``1DDynamic`` map normalised to 1 [MV/m], ``LAG`` [rad] is added to the
  crest phase found by ``OPTION, AUTOPHASE`` (``E ∝ cos(ωt + φ)``: a late particle gains more at a
  negative ``LAG``), ``DESIGNENERGY`` [MeV] makes OPAL scale the amplitude to that crest energy; the
  map's extent is the element's length (thin gaps get a surrogate length, solenoids a padded map).
"""
from lattix.formats.opal.reader import Reader, read
from lattix.formats.opal.writer import Writer, write

__all__ = ["Reader", "Writer", "read", "write"]
