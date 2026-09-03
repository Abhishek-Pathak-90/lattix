"""xtrack support (PLAN §6 task 3.5): IR ⇄ ``xt.Line`` and the JSON deck format.

    from lattix.formats.xtrack import Reader, Writer, to_line, from_line

``to_line``/``from_line`` work on live objects (the xtrack oracle, a notebook, an
Xsuite pipeline); ``Reader``/``Writer`` read and write the ``Line.to_json`` deck and
also accept an ``Environment`` JSON (``line=`` picks the sequence).

Conventions measured on this machine against xtrack 0.103.5 **and** 0.112.0 — the two
releases expose identical ``_xofields`` for every class used here:

* ``Cavity``: ``lag`` [deg] and ``phase`` [rad] add; the IR synchronous phase φ maps to
  ``lag = φ·180/π + 90`` ≡ ``phase = φ + π/2``.  ``V = 1 MV`` at φ = −30° gives
  ΔE = +866 025.4038 eV = ``V·cos 30°`` in both releases.  The writer emits ``phase``
  (``lag`` is deprecated in 0.112); the reader accepts both.
* ``Kicker`` → ``Multipole(knl=[-hkick], ksl=[+vkick])`` (measured: px = +hkick).
* ``rot_s_rad`` is exactly MAD-X ``tilt`` (2.2e-16 against cpymad's sector map);
  ``shift_x``/``shift_y``/``shift_s``/``rot_s_rad_no_frame`` are ``dx``/``dy``/``ds``/``dpsi``.
* xtrack keeps ``p0c`` constant through RF → ``EQUIVALENT:CONST_P0`` and the same
  ``energy_mode="local"|"constant"`` switch the MAD-X writer has.

Registry entry (``lattix/formats/base.py``; must be listed **after** ``pals`` and
``lattix`` so their longer ``.pals.json``/``.lattix.json`` suffixes win)::

    "xtrack": FormatSpec("xtrack", (".json",), "lattix.formats.xtrack",
                         description="xtrack Line/Environment JSON"),

:func:`sniff_json` settles a ``.json`` whose suffix alone is ambiguous.
"""
from lattix.formats.xtrack.convert import (
    COMMENT_ROLES,
    KNOWN_CLASSES,
    METADATA_KEY,
    RULES,
    NameMap,
    Rule,
    from_line,
    phase_from_xtrack,
    sanitize,
    to_line,
    xtrack_lag_deg,
    xtrack_phase_rad,
)
from lattix.formats.xtrack.json_io import Reader, Writer, load_document, sniff_json

__all__ = ["COMMENT_ROLES", "KNOWN_CLASSES", "METADATA_KEY", "NameMap", "RULES", "Reader",
           "Rule", "Writer", "from_line", "load_document", "phase_from_xtrack", "sanitize",
           "sniff_json", "to_line", "xtrack_lag_deg", "xtrack_phase_rad"]
