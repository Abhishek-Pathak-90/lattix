"""Bmad identifier rules and reversible renaming (PLAN §4.4 ``NameRules``).

Measured on Bmad 20260828.0 / Tao (2026-09-03) and confirmed by the manual
(``bmad/doc/lattice-file.tex``: *"Names of constants, elements, lines, etc. are
limited to 40 characters. The first character must be a letter (A–Z). The other
characters may be a letter, a digit (0–9) or an underscore (_)"*):

* the longest accepted element name is **40 characters** — a 41-character name
  makes ``bmad_parser`` report ``SUB-ELEMENT NAME IS BLANK FOR LINE/LIST``
  (measured: 40 parses, 41/42/60/80/100 fail);
* names are **case insensitive**; Bmad upper-cases them internally, so the
  writer emits lower case and comparisons are done lower-cased;
* ``.`` and ``\\`` are legal characters but Bmad *uses* them itself (tagged
  elements, superposition slaves), so they are not in the writable set;
* Bmad separates commands with newlines or ``;``; a line continues when it ends
  with ``&`` or with one of ``, ( { [ =`` (see :mod:`lattix.formats.bmad.reader`).

A renamed element carries an adjacent comment tag ``! lattix: name="…"`` that
:func:`parse_tags` reads back, so the rename is reversible (invariant I-15).
"""
from __future__ import annotations

import re

#: measured with Tao: 40 characters parse, 41 do not.
MAX_NAME_LEN = 40
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Bmad element keys and statement keywords that must not be reused as a name.
RESERVED: frozenset[str] = frozenset({
    # element keys (bmad/modules/bmad_struct.f90 key names, lower case)
    "drift", "pipe", "marker", "monitor", "instrument", "detector", "quadrupole",
    "sextupole", "octupole", "multipole", "ab_multipole", "thick_multipole",
    "sbend", "rbend", "solenoid", "sol_quad", "lcavity", "rfcavity", "crab_cavity",
    "hkicker", "vkicker", "kicker", "ac_kicker", "gkicker", "elseparator",
    "rcollimator", "ecollimator", "collimator", "mask", "taylor", "patch", "match",
    "em_field", "wiggler", "undulator", "beambeam", "e_gun", "converter", "foil",
    "fiducial", "floor_shift", "girder", "group", "overlay", "ramper", "null_ele",
    "sad_mult", "fork", "photon_fork", "capillary", "crystal", "mirror",
    "multilayer_mirror", "diffraction_plate", "sample", "photon_init", "lens",
    "beginning_ele", "hybrid", "rf_bend", "feedback", "pickup", "kick",
    # statements / reserved words
    "parameter", "beginning", "particle_start", "beam_start", "line", "list",
    "use", "call", "title", "expand_lattice", "no_digested", "end_file", "return",
    "superimpose", "beam", "set", "print", "merge_elements", "combine_consecutive_elements",
    "redef", "if", "then", "else", "end", "debug_marker", "write_digested",
    # constants
    "pi", "twopi", "fourpi", "e", "e_log", "sqrt_2", "degrad", "raddeg", "clight",
    "true", "false", "t", "f", "anomalous_moment_of", "mass_of", "charge_of",
})

_TAG_LINE = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9_.]*)\s*:.*?!\s*lattix:\s*(?P<body>.*?)\s*$")
_TAG_KV = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def is_valid(name: str) -> bool:
    return bool(NAME_RE.match(name)) and len(name) <= MAX_NAME_LEN and name not in RESERVED


def sanitize(name: str) -> str:
    """Lower-case *name* and replace everything Bmad cannot spell with ``_``."""
    s = re.sub(r"[^a-z0-9_]", "_", (name or "").strip().lower())
    s = s.lstrip("_")
    if not s or not s[0].isalpha():
        s = "e_" + s
    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN]
    if s in RESERVED:
        s = s[: MAX_NAME_LEN - 2] + "_x"
    return s


class NameMap:
    """Sanitize + uniquify element names, remembering what was renamed."""

    def __init__(self) -> None:
        self._assigned: dict[int, str] = {}       # id(source object) -> bmad name
        self._by_original: dict[str, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}         # bmad name -> original name

    def assign(self, original: str, key: object | None = None) -> str:
        """Return the Bmad name for *original*; stable per ``key`` (or per name)."""
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
    """``{bmad_name: {"name": original, "type": original_type}}`` from a deck's text."""
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        m = _TAG_LINE.match(line)
        if not m:
            continue
        kv = dict(_TAG_KV.findall(m.group("body")))
        if kv:
            out[m.group("name").lower()] = kv
    return out
