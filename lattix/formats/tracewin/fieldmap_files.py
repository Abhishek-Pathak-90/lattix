"""TraceWin field-map file bookkeeping: ``geom`` decoding, component file names, portable paths.

Ports of HELIX ``linac_gen/io/tracewin_geom.py`` (``decode_geom`` / ``component_files`` /
``enabled_channels``) and ``linac_gen/io/portable_paths.py`` (``best_relpath``).  Field DATA are
not read here (Phase 3, ``ir/fieldmap.py``); this module only resolves which files a FIELD_MAP
card refers to, so the reader can record them and the writer can re-emit a relocatable
``FIELD_MAP_PATH``.

TraceWin manual, section FIELD_MAP::

    geom = aper·10⁴ + rf_B·10³ + rf_E·10² + stat_B·10 + stat_E

each digit 0..9 describes the geometry of one field channel (0 absent, 1 1-D, 4 2-D cyl E-type,
5 2-D cyl B-type, 6 2-D Cartesian, 7 3-D Cartesian, 8 3-D cyl (N/A), 9 1-D G(z) quad gradient).
A negative ``geom`` asks for TraceWin's second-order off-axis expansion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class GeomCode:
    stat_E: int
    stat_B: int
    rf_E: int
    rf_B: int
    aper: int
    second_order: bool


def decode_geom(geom: int) -> GeomCode:
    """Decode the FIELD_MAP ``geom`` integer into a :class:`GeomCode`."""
    second_order = geom < 0
    g = abs(int(geom))
    return GeomCode(
        stat_E=g % 10,
        stat_B=(g // 10) % 10,
        rf_E=(g // 100) % 10,
        rf_B=(g // 1000) % 10,
        aper=(g // 10000) % 10,
        second_order=second_order,
    )


class Channel(Enum):
    """One of the four field channels a FIELD_MAP can contain."""

    STAT_E = ("e", "s")
    STAT_B = ("b", "s")
    RF_E = ("e", "d")
    RF_B = ("b", "d")

    @property
    def field_letter(self) -> str:
        return self.value[0]

    @property
    def type_letter(self) -> str:
        return self.value[1]

    @property
    def is_rf(self) -> bool:
        return self.type_letter == "d"

    @property
    def is_electric(self) -> bool:
        return self.field_letter == "e"

    @property
    def is_magnetic(self) -> bool:
        return self.field_letter == "b"


def component_files(channel: Channel, digit: int) -> list[str]:
    """File-extension list (with the leading dot) for one (channel, geometry digit) pair.

    Raises ``ValueError`` for a digit that is meaningless in this channel and
    ``NotImplementedError`` for digit 8 (3-D cylindrical, "not implemented yet" in the manual).
    """
    if digit == 0:
        return []
    fl, tl = channel.field_letter, channel.type_letter
    if digit == 1:
        return [f".{fl}{tl}z"]
    if digit == 4:
        if not channel.is_electric:
            raise ValueError(f"digit 4 (2-D cyl E-type) not valid in {channel.name} (only stat_E / rf_E)")
        base = [f".{fl}{tl}r", f".{fl}{tl}z"]
        if channel.is_rf:
            base.append(".bdq")  # TM mode Bθ
        return base
    if digit == 5:
        if not channel.is_magnetic:
            raise ValueError(f"digit 5 (2-D cyl B-type) not valid in {channel.name} (only stat_B / rf_B)")
        base = [f".{fl}{tl}r", f".{fl}{tl}z"]
        if channel.is_rf:
            base.append(".edq")  # TE mode Eθ
        return base
    if digit == 6:
        return [f".{fl}{tl}x", f".{fl}{tl}y"]
    if digit == 7:
        return [f".{fl}{tl}x", f".{fl}{tl}y", f".{fl}{tl}z"]
    if digit == 8:
        raise NotImplementedError("digit 8 (3-D cyl) is marked 'not implemented yet' in the TraceWin manual")
    if digit == 9:
        if channel is not Channel.STAT_B:
            raise ValueError(f"digit 9 (1-D quad G(z)) only valid in stat_B channel, got {channel.name}")
        return [f".{fl}{tl}z"]
    raise ValueError(f"unknown geometry digit: {digit}")


def enabled_channels(code: GeomCode) -> list[tuple[Channel, int]]:
    """(channel, digit) pairs with a non-zero digit, in the fixed order STAT_E, STAT_B, RF_E, RF_B."""
    out: list[tuple[Channel, int]] = []
    for ch, d in (
        (Channel.STAT_E, code.stat_E),
        (Channel.STAT_B, code.stat_B),
        (Channel.RF_E, code.rf_E),
        (Channel.RF_B, code.rf_B),
    ):
        if d != 0:
            out.append((ch, d))
    return out


def has_electric_channel(geom: int) -> bool:
    """True when the map carries an electric (static or RF) channel — HELIX ``_has_electric_channel``;
    only such maps consume a pending SET_SYNC_PHASE inside a SUPERPOSE cluster."""
    code = decode_geom(geom)
    return code.stat_E != 0 or code.rf_E != 0


def expected_files(geom: int, prefix: str) -> list[str]:
    """Every component file a FIELD_MAP card with this ``geom`` refers to (``prefix`` = dir/base).

    The aperture digit adds ``prefix.ouv``.  Propagates the ``component_files`` exceptions.
    """
    code = decode_geom(geom)
    files: list[str] = []
    for ch, d in enabled_channels(code):
        files.extend(prefix + ext for ext in component_files(ch, d))
    if code.aper != 0:
        files.append(prefix + ".ouv")
    return files


def resolve_field_files(geom: int, map_dir: str, base: str) -> tuple[list[str], list[str], str | None]:
    """``(existing, missing, error)`` for the component files of ``map_dir/base``.

    ``error`` carries the message of an invalid geom (then both lists are empty).
    """
    prefix = os.path.join(map_dir, base)
    try:
        files = expected_files(geom, prefix)
    except (ValueError, NotImplementedError) as exc:
        return [], [], str(exc)
    existing = [f for f in files if Path(f).is_file()]
    missing = [f for f in files if not Path(f).is_file()]
    return existing, missing, None


def best_relpath(target: str, anchor_dir: str) -> tuple[str, bool]:
    """Relativize ``target`` against ``anchor_dir`` when meaningful (HELIX ``portable_paths``).

    Returns ``(path, ok)``: ``ok=True`` with a POSIX-separator relative path (``../../Fields``
    allowed — the trees move together); ``ok=False`` with the absolute target when no meaningful
    relative form exists (Windows cross-drive, or the only common ancestor is the filesystem
    root — a ``../../..``-to-root relpath is not portable).
    """
    t = os.path.abspath(str(target))
    a = os.path.abspath(str(anchor_dir))
    try:
        rel = os.path.relpath(t, a)
    except ValueError:  # Windows: different drives
        return t, False
    try:
        common = os.path.commonpath([t, a])
    except ValueError:  # pragma: no cover — mixed absolute forms
        return t, False
    if os.path.splitdrive(common)[1] in (os.sep, "/"):
        return t, False
    return rel.replace(os.sep, "/"), True
