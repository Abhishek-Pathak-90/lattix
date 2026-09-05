"""IR → MAD-NG Lua sequence through xtrack (``xtrack.mad_writer.to_madng_sequence``)."""
from __future__ import annotations

from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage as _check
from lattix.ir.lattice import Lattice


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport` (MAD-NG ``.madng`` Lua text)."""

    format = "madng"

    @property
    def RULES(self):  # noqa: N802 - the coverage hook reads this name
        from lattix.formats.xtrack.convert import RULES

        return RULES

    def write(self, lattice: Lattice, path, *, strict: bool = False, energy_mode: str = "delta",
              name: str | None = None, install_apertures: bool = True) -> FidelityReport:
        rep = FidelityReport(target_format="madng", target_file=str(path))
        text = self.dumps(lattice, report=rep, energy_mode=energy_mode, name=name, install_apertures=install_apertures)
        Path(path).write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    def dumps(self, lattice: Lattice, *, report: FidelityReport | None = None, energy_mode: str = "delta",
              name: str | None = None, install_apertures: bool = True) -> str:
        from lattix.formats.xtrack.convert import allow_jit, to_line

        allow_jit()
        from xtrack import mad_writer

        rep = report if report is not None else FidelityReport(target_format="madng")
        line = to_line(lattice, energy_mode=energy_mode, report=rep, install_apertures=install_apertures)
        seq = name or lattice.use or lattice.name or "seq"
        seq = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in seq) or "seq"
        text = mad_writer.to_madng_sequence(line, name=seq)
        for e in list(rep.entries):
            if e.element:
                rep.equivalent("VIA_XTRACK", "MAD-NG text rendered by xtrack's mad_writer from the xtrack "
                               "conversion of this element", element=e.element, kind=e.kind)
        header = (f"-- lattix {__version__} from {lattice.meta.get('source_format', 'IR')} (MAD-NG via xtrack "
                  f"{_xtrack_version()})\n-- lattix: energy_mode={energy_mode}\n")
        return header + text


def _xtrack_version() -> str:
    try:
        import xtrack

        return str(xtrack.__version__)
    except Exception:                                    # noqa: BLE001
        return "?"


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert _check(Writer()) == set()
