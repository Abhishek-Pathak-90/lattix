"""Format registry: one suffix → format map, Reader/Writer protocols, read/write/translate."""
from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from lattix.fidelity import FidelityReport, TranslationError
from lattix.ir.elements import ALL_KINDS
from lattix.ir.lattice import Lattice


class Reader(Protocol):
    format: str

    def read(self, path: Path, **options) -> tuple[Lattice, FidelityReport]: ...


class Writer(Protocol):
    format: str
    RULES: dict[str, object]       # kind -> rule; must cover ALL_KINDS (tested)

    def write(self, lattice: Lattice, path: Path, *, strict: bool = False, **options) -> FidelityReport: ...


@dataclass
class FormatSpec:
    name: str
    suffixes: tuple[str, ...]
    module: str                                   # "lattix.formats.tracewin"
    reader_attr: str | None = "Reader"
    writer_attr: str | None = "Writer"
    description: str = ""
    options: dict = field(default_factory=dict)

    def reader(self) -> Reader | None:
        if not self.reader_attr:
            return None
        mod = importlib.import_module(self.module)
        return getattr(mod, self.reader_attr)()

    def writer(self) -> Writer | None:
        if not self.writer_attr:
            return None
        mod = importlib.import_module(self.module)
        return getattr(mod, self.writer_attr)()


FORMATS: dict[str, FormatSpec] = {
    "tracewin": FormatSpec("tracewin", (".dat",), "lattix.formats.tracewin", description="TraceWin .dat deck"),
    "madx": FormatSpec("madx", (".madx", ".seq", ".mad", ".str"), "lattix.formats.madx", description="MAD-X"),
    "mad8": FormatSpec("mad8", (".lat", ".flat"), "lattix.formats.mad8", description="MAD8 flat file"),
    "elegant": FormatSpec("elegant", (".lte",), "lattix.formats.elegant", description="Elegant .lte"),
    "bmad": FormatSpec("bmad", (".bmad",), "lattix.formats.bmad", description="Bmad"),
    "pals": FormatSpec("pals", (".pals.yaml", ".pals.yml", ".pals.json"), "lattix.formats.pals",
                       description="PALS lattice standard"),
    "lattix": FormatSpec("lattix", (".lattix.json",), "lattix.formats.lattix_json",
                         description="lattix IR as JSON (lossless)"),
    "flame": FormatSpec("flame", (".flame.lat", ".lat"), "lattix.formats.flame", description="FLAME GLPS deck"),
    "impactx": FormatSpec("impactx", (".impactx.in", ".impactx.py"), "lattix.formats.impactx",
                          description="ImpactX inputs / python"),
    "impactz": FormatSpec("impactz", ("impactz.in",), "lattix.formats.impactz", description="IMPACT-Z ImpactZ.in"),
    # keep last: a bare .json is xtrack's unless the content says otherwise (sniffed below)
    "xtrack": FormatSpec("xtrack", (".json",), "lattix.formats.xtrack", description="xtrack Line/Environment JSON"),
}


def _sniff_lat(p: Path) -> str | None:
    """`.lat` is both MAD8 flat and FLAME GLPS.  `!` comments and `:=` are illegal in GLPS;
    `sim_type =` and `USE:` (colon) only exist in FLAME."""
    import re

    try:
        head = p.read_text(encoding="latin-1", errors="replace")[:8192]
    except OSError:
        return None
    if "!" in head or ":=" in head:
        return "mad8"
    body = re.sub(r"#.*", "", head)
    if re.search(r"\bsim_type\s*=", body) or re.search(r"^\s*USE\s*:", body, re.M):
        return "flame"
    return None


def guess_format(path: str | Path) -> str:
    """Longest matching suffix wins (`.pals.json` before `.json`, `.flame.lat` before `.lat`);
    a bare `.lat`/`.flat` is sniffed for FLAME vs MAD8, a bare `.json` for xtrack vs lattix."""
    p = Path(path)
    name = p.name.lower()
    best: tuple[int, str] | None = None
    for spec in FORMATS.values():
        for suf in spec.suffixes:
            if name.endswith(suf) and (best is None or len(suf) > best[0]):
                best = (len(suf), spec.name)
    if best is None:
        raise ValueError(f"cannot guess the lattice format of {p.name!r}; pass fmt=")
    fmt = best[1]
    if fmt in ("mad8", "flame") and name.endswith((".lat", ".flat")) and not name.endswith(".flame.lat"):
        sniffed = _sniff_lat(p) if p.is_file() else None
        return sniffed or "mad8"
    if fmt == "xtrack" and p.is_file():
        try:
            import json

            head = json.loads(p.read_text()[:200000] if p.stat().st_size < 200000 else "{}") or {}
        except Exception:  # noqa: BLE001
            head = {}
        if isinstance(head, dict) and "reference" in head and "elements" in head and "lines" in head:
            return "lattix"
    return fmt


def read(path: str | Path, fmt: str | None = None, **options) -> tuple[Lattice, FidelityReport]:
    fmt = fmt or guess_format(path)
    rd = FORMATS[fmt].reader()
    if rd is None:
        raise ValueError(f"format {fmt!r} has no reader")
    lat, rep = rd.read(Path(path), **options)
    rep.source_format = fmt
    rep.source_file = str(path)
    return lat, rep


def write(lattice: Lattice, path: str | Path, fmt: str | None = None, *, strict: bool = False,
          **options) -> FidelityReport:
    fmt = fmt or guess_format(path)
    wr = FORMATS[fmt].writer()
    if wr is None:
        raise ValueError(f"format {fmt!r} has no writer")
    existed = Path(path).exists()
    try:
        rep = wr.write(lattice, Path(path), strict=strict, **options)
    except TranslationError:
        if strict and not existed and Path(path).exists():
            Path(path).unlink()          # a strict failure leaves no half-written deck behind
        raise
    rep.target_format = fmt
    rep.target_file = str(path)
    rep.raise_if(strict)
    return rep


def translate(src: str | Path, dst: str | Path, *, src_fmt: str | None = None, dst_fmt: str | None = None,
              strict: bool = False, read_options: dict | None = None,
              write_options: dict | None = None) -> FidelityReport:
    lat, rep_in = read(src, src_fmt, **(read_options or {}))
    rep_out = write(lat, dst, dst_fmt, strict=strict, **(write_options or {}))
    rep = FidelityReport(source_format=rep_in.source_format, target_format=rep_out.target_format,
                         source_file=str(src), target_file=str(dst), allowlist=rep_out.allowlist)
    rep.entries = rep_in.entries + rep_out.entries
    rep.raise_if(strict)
    return rep


def check_rules_coverage(writer: Writer) -> set[str]:
    """Kinds a writer's RULES table does not mention (must be empty; tested per writer)."""
    return set(ALL_KINDS) - set(writer.RULES)


def rule_table(**rules: Callable) -> dict[str, Callable]:
    return dict(rules)
