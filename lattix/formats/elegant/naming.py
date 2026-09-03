"""Elegant identifier rules and reversible renaming (PLAN §4.4 ``NameRules``).

Measured against ``elegant 2026.3.0`` (conda-forge osx-arm64) on 2026-09-03 by
defining ``<name>: QUAD, L=0.1, K1=1`` and running the deck:

* names are **case insensitive** — elegant upper-cases every name it reads, so
  the writer emits upper case and a deck's own spelling is kept only in
  :class:`~lattix.ir.elements.Provenance`;
* the accepted character set is very wide: ``. : $ - + @ % & / * [`` and digits
  all parse.  Only ``!`` and ``#`` are rejected (both start a comment).  The
  PLAN's draft pattern ``[A-Za-z0-9_.:$#-]`` is therefore wrong on ``#``;
* ``-`` *is* accepted by elegant (it quotes such a name on output) but a leading
  ``-`` means "reverse" inside ``LINE=(…)``, so it is **not** in the writable
  set here;
* there is no length limit worth caring about: a 200-character name parses.
  The writer caps at 128 so a name always fits a wrapped line.

The writable set is therefore ``^[A-Za-z][A-Za-z0-9_.:$]*$``, and an element
must never be named after an elegant type keyword: ``A: QUAD, …`` would then be
read as a *template* reference to the element ``QUAD`` instead of the type.

A renamed element carries an adjacent comment tag ``! lattix: name="…"`` that
:func:`parse_tags` reads back, so the rename is reversible (invariant I-15).
"""
from __future__ import annotations

import re

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:$]*$")
#: elegant itself accepts these too; the reader must not reject a deck using them.
READABLE_RE = re.compile(r"^[^\s!#,;=()*&%]+$")
MAX_NAME_LEN = 128

#: Every elegant type keyword this package knows, plus the statement keywords.
#: An element may not carry one of these names (it would become a template base).
RESERVED: frozenset[str] = frozenset({
    "LINE", "USE", "RETURN", "STO", "INCLUDE",
    # drifts
    "DRIF", "DRIFT", "EDRIFT", "CSRDRIFT", "CSRDRIF", "LSCDRIFT", "LSCDRIF",
    # magnets
    "QUAD", "QUADRUPOLE", "KQUAD", "KQUSE", "SEXT", "SEXTUPOLE", "KSEXT",
    "OCTU", "OCTUPOLE", "KOCT", "MULT", "MULTIPOLE", "FMULT", "SOLE", "SOLENOID",
    "SBEN", "SBEND", "CSBEND", "CSRCSBEND", "CSRCSBEN", "NIBEND", "NISEPT",
    "RBEN", "RBEND", "CCBEND", "BRAT", "TUBEND",
    # rf
    "RFCA", "RFCW", "RFDF", "TWLA", "TWMTA", "TMCF", "MODRF", "RAMPRF", "RFTMEZ0",
    # correctors
    "HKICK", "HKIC", "VKICK", "VKIC", "KICK", "KICKER", "EHKICK", "EVKICK",
    "EKICKER", "BUMPER", "MBUMPER",
    # apertures, diagnostics, markers
    "ECOL", "RCOL", "MAXAMP", "SCRAPER", "MARK", "MARKER", "MONI", "MONITOR",
    "HMON", "VMON", "WATCH", "HISTOGRAM", "MHISTOGRAM", "SLICE", "FLOOR",
    # maps and beam data
    "EMATRIX", "MATR", "ILMATRIX", "CHARGE", "WAKE", "TRWAKE", "ZLONGIT",
    "ZTRANSVERSE", "RFMODE", "TRFMODE", "FRFMODE", "FTRFMODE", "SCATTER",
    "DSCATTER", "SREFFECTS", "IBSCATTER", "MALIGN", "ROTATE", "TWISSELEMENT",
    "ENERGY", "CENTER", "RECIRC", "SCRIPT", "PFILTER", "REMCOR", "LTHINLENS",
    "BEAMBEAM", "WIGGLER", "CWIGGLER", "UKICKMAP", "LSRMDLTR", "SAMPLE",
})

_TAG_LINE = re.compile(r"^\s*(?P<name>[^\s:,!]+)\s*:.*?!\s*lattix:\s*(?P<body>.*?)\s*$")
_TAG_KV = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def is_valid(name: str) -> bool:
    """True when *name* can be written into a ``.lte`` unchanged (upper case)."""
    return bool(NAME_RE.match(name)) and len(name) <= MAX_NAME_LEN and name.upper() not in RESERVED


def sanitize(name: str) -> str:
    """Upper-case *name* and replace everything elegant cannot spell with ``_``."""
    s = re.sub(r"[^A-Z0-9_.:$]", "_", (name or "").strip().upper())
    s = s.lstrip("_.:$")
    if not s or not s[0].isalpha():
        s = "E_" + s
    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN]
    if s in RESERVED:
        s = s[: MAX_NAME_LEN - 2] + "_X"
    return s


class NameMap:
    """Sanitize + uniquify element names, remembering what was renamed."""

    def __init__(self) -> None:
        self._assigned: dict[int, str] = {}       # id(source object) -> elegant name
        self._by_original: dict[str, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}         # elegant name -> original name

    def assign(self, original: str, key: object | None = None) -> str:
        """Return the elegant name for *original*; stable per ``key`` (or per name)."""
        cache_key = id(key) if key is not None else None
        if cache_key is not None and cache_key in self._assigned:
            return self._assigned[cache_key]
        if cache_key is None and original in self._by_original:
            return self._by_original[original]
        name = self._unique(sanitize(original))
        if cache_key is not None:
            self._assigned[cache_key] = name
        self._by_original[original] = name
        if name != original:
            self.renamed[name] = original
        return name

    def reserve(self, name: str) -> str:
        """Claim a name (for lines) without recording a rename."""
        return self._unique(sanitize(name))

    def _unique(self, base: str) -> str:
        out = base
        k = 2
        while out in self._used:
            suffix = f"_{k}"
            out = base[: MAX_NAME_LEN - len(suffix)] + suffix
            k += 1
        self._used.add(out)
        return out


def name_tag(original: str, original_type: str | None = None) -> str:
    """The reversible provenance comment written next to a renamed definition."""
    body = f'name="{original}"'
    if original_type:
        body += f' type="{original_type}"'
    return f"! lattix: {body}"


def parse_tags(text: str) -> dict[str, dict[str, str]]:
    """``{elegant_name_upper: {"name": original, "type": original_type}}`` from a deck."""
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if "lattix:" not in line:
            continue
        m = _TAG_LINE.match(line)
        if not m:
            continue
        kv = dict(_TAG_KV.findall(m.group("body")))
        if kv:
            out[m.group("name").strip('"').upper()] = kv
    return out
