"""Formats reached through Bmad's own converters (Phase 5.6).

Astra, GPT, CSRtrack, Merlin++, SLICKTRACK and SAD decks are *written* by handing
lattix's Bmad output to the converters that ship with Bmad: ``bmad_to_astra``, ``bmad_to_gpt``,
``bmad_to_csrtrack``, ``bmad_to_merlin`` and ``bmad_to_slicktrack`` (binaries of the ``bmad``
conda environment, with the namelist inputs they want generated here) and Tao's ``write sad``
(the Tao of Bmad 20260828 accepts no OPAL or XSIF ``write``: ``UNKNOWN "WHAT"`` /
``BAD OUT_TYPE``).  SAD, SXF and Accelerator-Toolkit lattices are *read* with Bmad's
``sad_to_bmad.py``, ``sxf_to_bmad.py`` and ``accelerator_toolkit_to_bmad.py`` — scripts of a Bmad
source tree's ``util_programs`` (``LATTIX_BMAD_UTIL_DIR``, ``$ACC_ROOT_DIR/util_programs``) —
(the ``ptc_flat_file_to_bmad`` binary of the distribution reads a PTC flat file but writes no
lattice, so PTC is not bridged).

Every element goes through lattix's Bmad writer or reader, whose ledger applies, and carries
``EQUIVALENT VIA_BMAD``; what the converter printed is recorded too (``BMAD_CONVERTER_NOTE``,
``BMAD_CONVERTER_LOSS`` for anything it could not translate).  There is no engine on the far
side: the tests are structural and check that the intermediate Bmad file re-reads to the IR.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.formats.bmad import Reader as BmadReader
from lattix.formats.bmad import Writer as BmadWriter
from lattix.ir.lattice import Lattice
from lattix.oracles.envs import resolve_python

TIMEOUT_S = 600.0
_UTIL_VARS = ("LATTIX_BMAD_UTIL_DIR", "ACC_ROOT_DIR", "BMAD_DIST", "DIST_BASE_DIR")
_LOSS_RE = re.compile(r"ERROR|WARNING|NOT TRANSLATED|CANNOT|IGNOR|NOT IMPLEMENTED|UNABLE|NO TRANSLATION|\?\?\?", re.I)
_SKIP_RE = re.compile(r"^\s*(\[INFO\]|Parsing lattice|Written:|Opening:|$)|^\s*This might take", re.I)
_PYWARN_RE = re.compile(r"\.py:\d+: \w*Warning")      # a Python warning of the converter script (and its echoed line)
MAX_NOTES = 40


@dataclass(frozen=True)
class Target:
    name: str
    tool: str                 # a binary of the bmad environment, or "tao"
    suffix: str
    kind: str                 # "namelist" (an input file lattix writes), "arg" (the lattice as argument), "tao"
    output: str               # the file the tool leaves in the work directory
    tao_cmd: str = ""


TARGETS: dict[str, Target] = {
    "astra": Target("astra", "bmad_to_astra", ".astra", "namelist", "lat.astra"),
    "gpt": Target("gpt", "bmad_to_gpt", ".gpt", "namelist", "lat.gpt"),
    "csrtrack": Target("csrtrack", "bmad_to_csrtrack", ".csrtrk.in", "namelist", "csrtrk.in"),
    "merlin": Target("merlin", "bmad_to_merlin", ".tfs", "arg", "lat.bmad.tfs"),
    "slicktrack": Target("slicktrack", "bmad_to_slicktrack", ".slick", "arg", "lat.slick"),
    "sad": Target("sad", "tao", ".sad", "tao", "lat.sad", tao_cmd="write sad lat.sad"),
}


@dataclass(frozen=True)
class Source:
    name: str
    tool: str
    where: str                # "util" (a script of util_programs/<dir>/) or "bin"
    subdir: str = ""


SOURCES: dict[str, Source] = {
    "sad": Source("sad", "sad_to_bmad.py", "util", "sad_to_bmad"),
    "sxf": Source("sxf", "sxf_to_bmad.py", "util", "sxf_to_bmad"),
    "at": Source("at", "accelerator_toolkit_to_bmad.py", "util", "accelerator_toolkit_to_bmad"),
}


# ---------------------------------------------------------------------------- where Bmad lives
def bmad_python() -> tuple[str | None, str]:
    """The interpreter of the ``bmad`` environment (pytao importable), or why none was found."""
    py, tried = resolve_python("pytao", "bmad", "LATTIX_BMAD_PYTHON", try_current=True)
    return py, "; ".join(tried)


def bmad_tool(tool: str) -> tuple[Path | None, str]:
    """A converter binary: ``LATTIX_BMAD_BIN/<tool>``, next to the bmad environment's python, on PATH."""
    explicit = os.environ.get("LATTIX_BMAD_BIN")
    if explicit and (Path(explicit) / tool).is_file():
        return Path(explicit) / tool, ""
    py, why = bmad_python()
    if py:
        cand = Path(py).parent / tool
        if cand.is_file():
            return cand, ""
    found = shutil.which(tool)
    if found:
        return Path(found), ""
    return None, f"{tool} not found (LATTIX_BMAD_BIN, the bmad environment's bin, PATH); {why}"


def util_dir() -> Path | None:
    """Bmad's ``util_programs`` source directory (the Python converters are not installed by conda)."""
    for var in _UTIL_VARS:
        v = os.environ.get(var)
        if not v:
            continue
        for cand in (Path(v), Path(v) / "util_programs"):
            if (cand / "sad_to_bmad" / "sad_to_bmad.py").is_file():
                return cand
    return None


def available(name: str) -> tuple[bool, str]:
    """Whether the bridge for *name* (a target or a source) can run here."""
    if name in TARGETS:
        t = TARGETS[name]
        if t.kind == "tao":
            py, why = bmad_python()
            return (py is not None), ("" if py else why)
        tool, why = bmad_tool(t.tool)
        return (tool is not None), why
    if name in SOURCES:
        s = SOURCES[name]
        if s.where == "bin":
            tool, why = bmad_tool(s.tool)
            return (tool is not None), why
        py, why = bmad_python()
        if not py:
            return False, why
        if util_dir() is None:
            return False, (f"Bmad's util_programs not found: set LATTIX_BMAD_UTIL_DIR (or ACC_ROOT_DIR) to a Bmad "
                           f"source tree; {s.tool} is a script there, not part of the conda package")
        return True, ""
    raise KeyError(name)


def _run(cmd: list[str], cwd: Path, extra_env: dict | None = None) -> str:
    env = dict(os.environ)
    env.update(extra_env or {})
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S,
                           stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cmd[0]} timed out after {TIMEOUT_S:.0f} s in {cwd}") from exc
    log = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
    (cwd / "convert.log").write_text(log)
    if r.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-25:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed (exit {r.returncode}) in {cwd}:\n{tail}")
    return log


_OUT_IO_RE = re.compile(r"^\[(ERROR|WARNING|MESSAGE|INFO|SUCCESS)\s*\|[^\]]*\]\s*(\S+)\s*$")


def _notes(rep: FidelityReport, log: str, tool: str) -> None:
    """What the converter said, in the ledger: losses it announced, and its other remarks.  Bmad's
    ``out_io`` prints a ``[LEVEL | time] routine:`` header followed by indented message lines: the
    block is one entry, an ERROR or WARNING a loss."""
    blocks: list[tuple[str, str]] = []            # (level, text)
    skip_next = False
    for raw in log.splitlines():
        line = raw.rstrip()
        if skip_next:
            skip_next = False
            continue
        if _PYWARN_RE.search(line):
            skip_next = True
            continue
        if not line.strip() or line.startswith("---"):
            continue
        m = _OUT_IO_RE.match(line.strip())
        if m:
            blocks.append((m.group(1), m.group(2)))
            continue
        if line[:1].isspace() and blocks and blocks[-1][0] != "":
            level, text = blocks[-1]                     # a continuation of the last out_io block
            blocks[-1] = (level, text + " " + line.strip())
            continue
        if _SKIP_RE.match(line.strip()):
            continue
        blocks.append(("", line.strip()))
    n = 0
    seen: set[str] = set()
    for level, text in blocks:
        if level in ("INFO", "SUCCESS") or text in seen or _SKIP_RE.match(text):
            continue
        seen.add(text)
        loss = level in ("ERROR", "WARNING") or (not level and _LOSS_RE.search(text)
                                                  and not text.lower().startswith("note"))
        if loss:
            rep.lossy("BMAD_CONVERTER_LOSS", f"{tool}: {text[:400]}")
        else:
            rep.equivalent("BMAD_CONVERTER_NOTE", f"{tool}: {text[:400]}")
        n += 1
        if n >= MAX_NOTES:
            rep.equivalent("BMAD_CONVERTER_NOTE", f"{tool}: … (further output in convert.log)")
            break


def _via(rep: FidelityReport, lattice: Lattice, what: str) -> None:
    for p in lattice.flatten():
        rep.equivalent("VIA_BMAD", f"{p.element.kind} {p.element.name!r} {what}",
                       element=p.element.name, kind=p.element.kind)


# ---------------------------------------------------------------------------- writers
class BridgeWriter:
    """IR → Bmad (lattix's writer) → the target format (Bmad's converter)."""

    target: str = ""

    def write(self, lattice: Lattice, path: str | Path, *, strict: bool = False, bmad_copy: str | Path | None = None,
              keep_workdir: str | Path | None = None, fieldmap_dimension: int = 3, template: str | Path | None = None,
              no_split: bool = False, **bmad_options) -> FidelityReport:
        t = TARGETS[self.target]
        path = Path(path)
        wd = Path(keep_workdir) if keep_workdir else Path(tempfile.mkdtemp(prefix=f"lattix_{t.name}_"))
        wd.mkdir(parents=True, exist_ok=True)
        bmad_path = wd / "lat.bmad"
        rep = BmadWriter().write(lattice, bmad_path, strict=False, **bmad_options)
        rep.target_format = t.name
        rep.target_file = str(path)
        if bmad_copy:
            Path(bmad_copy).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bmad_path, bmad_copy)
        log = self._convert(t, wd, fieldmap_dimension=int(fieldmap_dimension), template=template, no_split=no_split)
        out = wd / t.output
        if not out.is_file():
            raise RuntimeError(f"{t.tool} left no {t.output} in {wd}:\n{log[-2000:]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, path)
        _via(rep, lattice, f"written for {t.name} by Bmad's {t.tool} from lattix's Bmad file")
        _notes(rep, log, t.tool)
        rep.raise_if(strict)
        return rep

    @staticmethod
    def _convert(t: Target, wd: Path, *, fieldmap_dimension: int, template, no_split: bool) -> str:
        if t.kind == "tao":
            py, why = bmad_python()
            if not py:
                raise RuntimeError(f"Tao (pytao) is not available for the {t.name} writer: {why}")
            snippet = ("import sys\nfrom pytao import Tao\n"
                       "t = Tao('-noplot -lat lat.bmad')\n"
                       f"print('\\n'.join(t.cmd({t.tao_cmd!r})))\n")
            return _run([py, "-c", snippet], wd)
        tool, why = bmad_tool(t.tool)
        if tool is None:
            raise RuntimeError(f"the {t.name} writer needs Bmad's {t.tool}: {why}")
        if t.kind == "arg":
            cmd = [str(tool), "lat.bmad"]
            if t.name == "slicktrack" and no_split:
                cmd.append("-no_split")
            return _run(cmd, wd)
        # namelist inputs
        if t.name == "astra":
            (wd / "bmad_to_astra.in").write_text(
                "&bmad_to_astra_params\n  lat_filename = 'lat.bmad'\n  astra_filename = 'lat.astra'\n"
                f"  astra_lattice_param%fieldmap_dimension = {fieldmap_dimension}\n"
                "  write_time_particles = .false.\n  write_astra_particles = .false.\n/\n")
            return _run([str(tool), "bmad_to_astra.in"], wd)
        if t.name == "gpt":
            (wd / "bmad_to_gpt.in").write_text(
                "&bmad_to_gpt_params\n  bmad_lat_filename = 'lat.bmad'\n  gpt_lat_param%gpt_filename = 'lat.gpt'\n"
                f"  gpt_lat_param%fieldmap_dimension = {fieldmap_dimension}\n"
                "  write_gpt_particles = .false.\n  write_bmad_time_particles = .false.\n/\n")
            return _run([str(tool), "bmad_to_gpt.in"], wd)
        if t.name == "csrtrack":
            body = Path(template).read_text() if template else (
                "io_path{input=in,output=out,logfile=log.txt}\n\nINSERT_LATTICE_AND_BUNCH_PARAMS_HERE\n\n"
                "tracker{end_time_marker=$end_marker,end_time_shift_c0=1.00}\n\nexit\n")
            (wd / "bmad_to_csrtrack.in").write_text(
                "&general_params\n  bmad_lattice = 'lat.bmad'\n  particle_out_file = 'particles.csrtrk'\n/\n\n"
                "&beam_params\n  beam_init%n_particle = 1\n/\n\n" + body)
            return _run([str(tool)], wd)
        raise RuntimeError(f"no converter recipe for {t.name}")   # pragma: no cover


class AstraWriter(BridgeWriter):
    target = "astra"


class GptWriter(BridgeWriter):
    target = "gpt"


class CsrtrackWriter(BridgeWriter):
    target = "csrtrack"


class MerlinWriter(BridgeWriter):
    target = "merlin"


class SlicktrackWriter(BridgeWriter):
    target = "slicktrack"


class SadWriter(BridgeWriter):
    target = "sad"


# ---------------------------------------------------------------------------- readers
class BridgeReader:
    """The source format → Bmad (Bmad's converter) → IR (lattix's Bmad reader)."""

    source: str = ""

    def read(self, path: str | Path, *, strict: bool = False, keep_workdir: str | Path | None = None,
             bmad_copy: str | Path | None = None, **bmad_options) -> tuple[Lattice, FidelityReport]:
        s = SOURCES[self.source]
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        wd = Path(keep_workdir) if keep_workdir else Path(tempfile.mkdtemp(prefix=f"lattix_{s.name}_"))
        wd.mkdir(parents=True, exist_ok=True)
        log, bmad_path = self._convert(s, path, wd)
        if bmad_copy:
            Path(bmad_copy).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bmad_path, bmad_copy)
        lat, rep = BmadReader().read(bmad_path, **bmad_options)
        rep.source_format = s.name
        rep.source_file = str(path)
        lat.meta["source_format"] = s.name
        lat.meta["bmad_bridge"] = {"source": s.name, "tool": s.tool, "bmad_file": str(bmad_path)}
        _via(rep, lat, f"read from {s.name} through Bmad's {s.tool} and lattix's Bmad reader")
        _notes(rep, log, s.tool)
        rep.raise_if(strict)
        return lat, rep

    @staticmethod
    def _convert(s: Source, path: Path, wd: Path) -> tuple[str, Path]:
        if s.where == "bin":
            tool, why = bmad_tool(s.tool)
            if tool is None:
                raise RuntimeError(f"the {s.name} reader needs Bmad's {s.tool}: {why}")
            log = _run([str(tool), str(path)], wd)
            produced = sorted(wd.glob("*.bmad"))
            if not produced:
                raise RuntimeError(f"{s.tool} left no .bmad file in {wd}:\n{log[-2000:]}")
            return log, produced[0]
        py, why = bmad_python()
        if not py:
            raise RuntimeError(f"the {s.name} reader needs the bmad environment's python: {why}")
        util = util_dir()
        if util is None:
            raise RuntimeError(f"the {s.name} reader needs Bmad's util_programs ({s.tool}): set LATTIX_BMAD_UTIL_DIR "
                               "(or ACC_ROOT_DIR) to a Bmad source tree")
        script = util / s.subdir / s.tool
        if s.name == "sad":
            params = (util / s.subdir / "sad_to_bmad.params").read_text().splitlines()
            out = []
            for ln in params:
                key = ln.split("=", 1)[0].strip() if "=" in ln else ""
                if key == "sad_lattice_file":
                    ln = f'sad_lattice_file = "{path}"'
                elif key == "bmad_lattice_file":
                    ln = 'bmad_lattice_file = "lat.bmad"'
                out.append(ln)
            (wd / "sad_to_bmad.params").write_text("\n".join(out) + "\n")
            log = _run([py, str(script), "sad_to_bmad.params", str(path)], wd)
            return log, wd / "lat.bmad"
        if s.name == "sxf":
            local = wd / "lat.sxf"
            shutil.copy2(path, local)
            log = _run([py, str(script), "lat.sxf"], wd)             # writes lat.bmad (the .sxf suffix stripped)
            produced = sorted(wd.glob("*.bmad"))
            if not produced:
                raise RuntimeError(f"{s.tool} left no .bmad file in {wd}:\n{log[-2000:]}")
            return log, produced[0]
        if s.name == "at":
            log = _run([py, str(script), str(path), "lat.bmad"], wd)
            return log, wd / "lat.bmad"
        raise RuntimeError(f"no converter recipe for {s.name}")      # pragma: no cover


class SadReader(BridgeReader):
    source = "sad"


class SxfReader(BridgeReader):
    source = "sxf"


class AtReader(BridgeReader):
    source = "at"


def write(lattice: Lattice, path, fmt: str, **options) -> FidelityReport:
    cls = {"astra": AstraWriter, "gpt": GptWriter, "csrtrack": CsrtrackWriter, "merlin": MerlinWriter,
           "slicktrack": SlicktrackWriter, "sad": SadWriter}[fmt]
    return cls().write(lattice, path, **options)


def read(path, fmt: str, **options) -> tuple[Lattice, FidelityReport]:
    cls = {"sad": SadReader, "sxf": SxfReader, "at": AtReader}[fmt]
    return cls().read(path, **options)
