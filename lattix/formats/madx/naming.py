"""MAD-X identifier rules and reversible renaming (PLAN §4.4 ``NameRules``).

Measured on MAD-X 5.09.03 / cpymad 1.19.0 (2026-09-03):

* names are **case insensitive** — MAD-X lower-cases everything, so the writer
  emits lower case and the reader gets lower case back;
* the accepted character set is a letter followed by letters, digits, ``_`` and
  ``.``; MAD-X also tolerates ``$`` (it names its own sequence bookkeeping
  markers ``<seq>$start`` / ``<seq>$end`` that way) but a deck cannot re-declare
  those, so ``$`` is *not* in the writable set;
* the longest name this build accepts is **41 characters** (42 aborts with
  ``fatal: String is too long``); the writer caps at 40 for headroom.

A renamed element carries an adjacent comment tag ``! lattix: name="…"`` that
:func:`parse_tags` reads back, so the rename is reversible (invariant I-15).
"""
from __future__ import annotations

import re

NAME_RE = re.compile(r"^[a-z][a-z0-9_.]*$")
MAX_NAME_LEN = 40

#: MAD-X keywords that must never be used as an element or line name.
RESERVED: frozenset[str] = frozenset({
    # base element types
    "drift", "quadrupole", "sextupole", "octupole", "multipole", "sbend", "rbend",
    "solenoid", "rfcavity", "crabcavity", "twcavity", "rfmultipole", "hkicker",
    "vkicker", "kicker", "tkicker", "marker", "monitor", "hmonitor", "vmonitor",
    "instrument", "placeholder", "rcollimator", "ecollimator", "collimator",
    "matrix", "dipedge", "elseparator", "beambeam", "yrotation", "xrotation",
    "srotation", "translation", "changeref", "nllens", "wire", "sequence", "line",
    # commands and reserved names met in real decks
    "beam", "use", "select", "twiss", "survey", "track", "ealign", "efcomp",
    "call", "title", "option", "set", "value", "return", "stop", "exit", "if",
    "while", "macro", "endsequence", "at", "from", "true", "false",
    # protected globals
    "pi", "twopi", "degrad", "raddeg", "e", "emass", "pmass", "nmass", "mumass",
    "umass", "amu0", "clight", "qelect", "hbar", "erad", "prad", "version",
    "twiss_tol",
})

_TAG_LINE = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9_.$]*)\s*:.*?!\s*lattix:\s*(?P<body>.*?)\s*$")
_TAG_KV = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_TITLE = re.compile(r"^\s*title\s*[, ]\s*(?:\"([^\"]*)\"|'([^']*)')", re.IGNORECASE | re.MULTILINE)


def is_valid(name: str) -> bool:
    return bool(NAME_RE.match(name)) and len(name) <= MAX_NAME_LEN and name not in RESERVED


def sanitize(name: str) -> str:
    """Lower-case *name* and replace everything MAD-X cannot spell with ``_``."""
    s = re.sub(r"[^a-z0-9_.]", "_", (name or "").strip().lower())
    s = s.lstrip("_.")
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
        self._assigned: dict[int, str] = {}      # id(source object) -> madx name
        self._by_original: dict[str, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}        # madx name -> original name

    def assign(self, original: str, key: object | None = None) -> str:
        """Return the MAD-X name for *original*; stable per ``key`` (or per name)."""
        cache_key = id(key) if key is not None else None
        if cache_key is not None and cache_key in self._assigned:
            return self._assigned[cache_key]
        if cache_key is None and original in self._by_original:
            return self._by_original[original]
        base = sanitize(original)
        name = base
        k = 2
        while name in self._used:
            suffix = f"_{k}"
            name = base[: MAX_NAME_LEN - len(suffix)] + suffix
            k += 1
        self._used.add(name)
        if cache_key is not None:
            self._assigned[cache_key] = name
        self._by_original[original] = name
        if name != original:
            self.renamed[name] = original
        return name

    def reserve(self, name: str) -> str:
        """Claim a name (for sequences/lines) without recording a rename."""
        base = sanitize(name)
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
    """``{madx_name: {"name": original, "type": original_type}}`` from a deck's text."""
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        m = _TAG_LINE.match(line)
        if not m:
            continue
        kv = dict(_TAG_KV.findall(m.group("body")))
        if kv:
            out[m.group("name").lower()] = kv
    return out


def parse_title(text: str) -> str | None:
    """The deck's ``TITLE, "…";`` string.

    cpymad exposes no accessor for MAD-X's title, so it is taken from the text of
    the file that was called (``CALL``ed sub-files are not scanned).
    """
    m = _TITLE.search(text)
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2)
