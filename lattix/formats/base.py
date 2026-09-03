"""Format registry: one suffix → format map, Reader/Writer protocols, read/write/translate."""
from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from lattix.fidelity import FidelityReport
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
}


def guess_format(path: str | Path) -> str:
    p = Path(path)
    name = p.name.lower()
    for spec in FORMATS.values():
        for suf in spec.suffixes:
            if name.endswith(suf):
                return spec.name
    raise ValueError(f"cannot guess the lattice format of {p.name!r}; pass fmt=")


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
    rep = wr.write(lattice, Path(path), strict=strict, **options)
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
