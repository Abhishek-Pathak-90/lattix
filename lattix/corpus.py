"""Corpus manifest tool (PLAN.md section 5.4).

The corpus is the set of lattice decks, field maps and engine outputs the test
suite and the nightly job read.  Most of it is private (PIP-II decks, CEA/ANL
field maps, TraceWin outputs) and never enters the repository.  What the
repository carries is a *sources file* (``corpus/sources.yaml``) that points at
those files by absolute path, and this module, which turns the pointer file into
a *manifest* (``$LATTIX_CORPUS_DIR/manifest.yaml``) with one entry per file::

    id, path, source, format, sha256, size, origin, license, redistributable,
    [notes], [detected_format], expected_warning_classes, anchors

``verify`` recomputes every sha256, so the corpus job fails on silent edits.

Command line::

    python -m lattix.corpus build-manifest [--root R] [--sources S] [--out M] [--fresh]
    python -m lattix.corpus verify        [--root R]
    python -m lattix.corpus list          [--root R] [--format F] [--redistributable] [--source S]

Sources file schema (YAML)::

    version: 1
    sources:
      - name: pipii-tracewin             # id prefix, [a-z0-9-]
        root: /abs/dir                   # every pattern below is relative to it
        paths: ["**/*.dat", "sub/{a.lat,b.lte}", "Fields"]   # globs (with {a,b} brace
                                         # expansion), single files, or directories
        origin: "PIP-II design decks (Fermilab)"
        license: private
        redistributable: false
        formats: [tracewin]              # optional: keep only files sniffed as one of these
        format: fieldmap                 # optional: override the sniff for every file
        notes: free text                 # optional, copied to every entry

Manifest ids are ``<source name>/<slug of the path relative to root>``; the
manifest is sorted by id so rebuilds diff cleanly.  Hand-curated fields
(``expected_warning_classes``, ``anchors``) survive a rebuild unless ``--fresh``.

Only the standard library and PyYAML are used.  Nothing here prints except the
``cmd_*`` CLI functions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ENV_VAR",
    "FORMATS",
    "MANIFEST_NAME",
    "MAX_FILE_SIZE",
    "BuildResult",
    "Problem",
    "build_manifest",
    "corpus_root",
    "expand_braces",
    "expand_source_paths",
    "is_binary",
    "iter_decks",
    "load_manifest",
    "load_sources",
    "main",
    "resolve_path",
    "sha256_of",
    "slugify",
    "sniff_format",
    "sniff_text",
    "verify_manifest",
    "write_manifest",
]

ENV_VAR = "LATTIX_CORPUS_DIR"
MANIFEST_NAME = "manifest.yaml"
MANIFEST_VERSION = 1
MAX_FILE_SIZE = 50 * 1024 * 1024
"""Files larger than this are skipped by ``build-manifest``."""
SNIFF_BYTES = 256 * 1024
"""How much of a file the sniffer reads."""
SNIFF_LINES = 800
"""How many non-blank, non-comment lines of that window the sniffer scores (re-saved
TraceWin decks can start with thousands of stacked ';' header lines)."""

FORMATS: tuple[str, ...] = (
    "tracewin",
    "madx",
    "mad8",
    "elegant",
    "bmad",
    "pals",
    "flame",
    "impactx",
    "impactz",
    "pyorbit",
    "tfs",
    "fieldmap",
    "tracewin_output",
    "table",
    "unknown",
)
"""Format ids the sniffer can return (lattice formats first, then data files)."""

LATTICE_FORMATS: frozenset[str] = frozenset(
    {"tracewin", "madx", "mad8", "elegant", "bmad", "pals", "flame", "impactx", "impactz", "pyorbit"}
)

# --------------------------------------------------------------------------- sniffing

_TW_CARDS: tuple[str, ...] = (
    # cards that start a TraceWin statement; longest first so ``\b`` cannot split them
    "ERROR_QUAD_NCPL_STAT",
    "ERROR_QUAD_NCPL_DYN",
    "ERROR_BEND_NCPL_STAT",
    "ERROR_BEND_NCPL_DYN",
    "ERROR_CAV_NCPL_STAT",
    "ERROR_CAV_NCPL_DYN",
    "ERROR_GAUSSIAN_CUT_OFF",
    "ERROR_BEAM_STAT",
    "ERROR_BEAM_DYN",
    "SET_GAUSSIAN_CUT_OFF",
    "SET_BEAM_PHASE_ERROR",
    "SET_BEAM_PHASE_ADV",
    "MIN_EMIT_4D_GROWTH",
    "ADJUST_BEAM_CENTROID",
    "ADJUST_BEAM_CURRENT",
    "ADJUST_BEAM_EMIT",
    "ADJUST_BEAM_TWISS",
    "ADJUST_STEERER_BX",
    "ADJUST_STEERER_BY",
    "ADJUST_STEERER",
    "SPACE_CHARGE_COMP",
    "SET_BEAM_ENERGY",
    "SET_BEAM_E0_P0",
    "MATCH_FAM_FIELD",
    "MATCH_FAM_PHASE",
    "MATCH_FAM_GRAD",
    "MIN_EMIT_GROWTH",
    "MIN_TRANSMISSION",
    "SET_SYNC_PHASE",
    "SET_SEPARATION",
    "START_ACHROMAT",
    "SET_ACHROMAT",
    "FIELD_MAP_PATH",
    "THIN_STEERING",
    "SUPERPOSE_MAP",
    "DIAG_POSITION",
    "DIAG_ACHROMAT",
    "PARTRAN_STEP",
    "SET_KE_OUT_MIN",
    "SET_POSITION",
    "SET_SIZE_MAX",
    "SET_SIZE_MIN",
    "BUNCHED_BEAM",
    "DIAG_ENERGY",
    "CHANGE_FREQ",
    "LATTICE_END",
    "DIAG_PHASE",
    "DIAG_TWISS",
    "DIAG_WAIST",
    "REPEAT_ELE",
    "FUNNEL_GAP",
    "DIAG_SIZE",
    "DIAG_EMIT",
    "DIAG_DSIZE",
    "DIAG_DPHASE",
    "DIAG_DENERGY",
    "THIN_LENS",
    "FIELD_MAP",
    "SET_TWISS",
    "SET_SIZE",
    "SOLENOID",
    "ELECTRODE",
    "RFQ_CELL",
    "BEAM_ROT",
    "PLOT_DST",
    "APERTURE",
    "LATTICE",
    "STEERER",
    "SET_ADV",
    "CHOPPER",
    "DTL_CEL",
    "NCELLS",
    "CAVSIN",
    "ADJUST",
    "DRIFT",
    "SHIFT",
    "FREQ",
    "QUAD",
    "BEND",
    "EDGE",
    "GAP",
    "END",
)
# ``NAME: CARD args`` labels are allowed; the card must not be followed by ':' '=' ',' '(' ';'
# (that is a MAD/Elegant/Bmad definition such as ``OLQ: DRIFT, L=0.15``).
_TW_RE = re.compile(
    r"^\s*(?:[A-Za-z0-9_.+\-#]+\s*:\s*)?(?:" + "|".join(_TW_CARDS) + r")\b(?!\s*[:=,(;])",
    re.IGNORECASE,
)
_MAD_DEF_RE = re.compile(r"^\s*[A-Za-z_][\w.$]*\s*:\s*[A-Za-z_]")
_MAD_ASSIGN_RE = re.compile(r"^\s*[A-Za-z_][\w.$\[\]]*\s*:?=\s*\S")
_LINE_DEF_RE = re.compile(r"\bline\s*=\s*\(", re.IGNORECASE)
_CONTINUATION_RE = re.compile(r"&\s*$")
_BARE_USE_RE = re.compile(r"^\s*use\s*,\s*(?:period\s*=\s*)?[\w.$]+\s*$", re.IGNORECASE)
_RETURN_RE = re.compile(r"^\s*(?:return|stop)\s*$", re.IGNORECASE)

_MADX_KEYS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bsequence\s*,",
        r"\bendsequence\b",
        r"\bat\s*=",
        r"\brefer\s*=",
        r"\bmakethin\b",
        r"\bptc_\w+",
        r"\buse\s*,\s*(?:sequence|period)\s*=",
        r"\bseqedit\b",
        r"\bsixtrack\b",
        r"\bcentre\b",
    )
)
_FLAME_KEYS = tuple(
    re.compile(p)
    for p in (
        r"\bsim_type\s*=",
        r"\bMpoleLevel\b",
        r"\bHdipoleFitMode\b",
        r"\bIonEs\b",
        r"\bIonEk\b",
        r"\bIonZ\b",
        r"\bIonW\b",
        r"\bEng_Data_Dir\b",
        r"\bIonChargeStates\b",
        r"\bNCharge\b",
        r"\bBaryCenter\d",
        r"\bMomentMatrix\b",
    )
)
_BMAD_KEYS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bparameter\[",
        r"\bbeginning\[",
        r"\blcavity\b",
        r"\bsuperimpose\b",
        r"\bp0c\b",
        r"\be_tot\b",
        r"\bbeam_start\[",
        r"\bparticle_start\[",
        r"\blat_geometry\b",
        r"\bgeometry\s*=",
        r"\boverlay\b",
        r"\bgirder\b",
        r"\bref_origin\b",
        r"\bele_origin\b",
        r"\bno_digested\b",
        r"\bexpand_lattice\b",
        r"\bmultipass\b",
        r"\bsol_quad\b",
        r"\bab_multipole\b",
        r"\bfloor_shift\b",
        r"\bcall\s*,\s*file\s*=",
    )
)
_ELEGANT_TYPE_RE = re.compile(
    r":\s*(?:DRIF|SBEN|RBEN|CSBEND|CSRCSBEND|CSRDRIFT|EDRIFT|LSCDRIFT|KQUAD|KSEXT|KOCT|QUAD|"
    r"SEXT|OCTU|RFCA|RFCW|RFDF|MARK|WATCH|CHARGE|MALIGN|MAXAMP|SOLE|MONI|HMON|VMON|HKICK|"
    r"VKICK|EHKICK|EVKICK|EMATRIX|ILMATRIX|RCOL|ECOL|SCRAPER|ENERGY|CENTER|TWISS|MULT|"
    r"CWIGGLER|TWLA|LRWAKE|WAKE|ZLONGIT|ZTRANSVERSE|RFMODE|TRFMODE|SREFFECTS|IBSCATTER|"
    r"SCATTER|BUMPER|MBUMPER|FTABLE|BGGEXP|BMAPXY|TUBEND|LGBEND|CCBEND|BRAT|MAGNIFY|ROTATE|"
    r"RECIRC|REFLECT|STRAY|SAMPLE|CLEAN|PFILTER|HISTOGRAM|MHISTOGRAM|SLICE|SCRIPT|EMITTANCE|"
    r"FMULT|QUFRINGE|UKICKMAP|KICKMAP|MODRF|RAMPRF|NIBEND|NISEPT|PEPPOT|LTHINLENS|LMIRROR|"
    r"CORGPIPE|TFBPICKUP|TFBDRIVER|TRCOUNT|IONEFFECTS|SPEEDBUMP|TAYLORSERIES|RFTM110|RFTMEZ0|"
    r"MAPSOLENOID|KSBEND|KPOLY|SCMULT|TRWAKE|MATR|HCOR|VCOR|CPICKUP|SHRFDF|MRFDF)\b",
    re.IGNORECASE,
)
_ELEGANT_RPN_RE = re.compile(r"^\s*%\s")

_TFS_RE = re.compile(r"^@\s+\w+\s+%\S*\s")
_TW_OUTPUT_RES = tuple(
    re.compile(p)
    for p in (
        r"^ELE#\s*\d+\s*:",
        r"^Cav#",
        r"^rect diaph\b",
        r"^circ diaph\b",
        r"^elps diaph\b",
        r"^DIAG\s*#",
        r"^Ele_name\s+ele#",
        r"^position\s+gam-1",
        r"^Position\s+(?:Linac\b|\S*Ɛ|\s*$)",
        r"^##\s*TraceWin\b",
    )
)
_IMPACTX_RE = re.compile(r"^\s*(?:beam|lattice|algo|amr|diag|geometry|impactx)\.\w+\s*=", re.IGNORECASE)
_IMPACTZ_ROW_RE = re.compile(r"^\s*[-+\d.eEdD\s]+/\s*$")
_IMPACTZ_HINT_RE = re.compile(r"IMPACT-?Z", re.IGNORECASE)
_PALS_RE = re.compile(
    r"\bkind:\s*(?:BeamLine|Drift|Quadrupole|Bend|SBend|RBend|Solenoid|Marker|RFCavity|"
    r"Sextupole|Octupole|Kicker|Multipole|Undulator|Collimator)\b"
)
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?$")
_COMMENT_PREFIXES = ("!", "#", ";", "//")

FIELDMAP_SUFFIXES: frozenset[str] = frozenset(
    {
        ".edz",
        ".edx",
        ".edy",
        ".bdx",
        ".bdy",
        ".bdz",
        ".bsx",
        ".bsy",
        ".bsz",
        ".esx",
        ".esy",
        ".esz",
        ".ouv",
        ".scc",
        ".vane",
        ".rfq",
        ".ele",
        ".mag",
        ".edp",
        ".bdp",
        ".edr",
        ".bdr",
    }
)
"""TraceWin ASCII field-map / RF-structure file suffixes (content is numeric, suffix decides)."""

_SUFFIX_PRIOR: dict[str, tuple[str, ...]] = {
    ".madx": ("madx",),
    ".seq": ("madx",),
    ".mad": ("madx",),
    ".flat": ("mad8",),
    ".mad8": ("mad8",),
    ".lat": ("mad8", "flame"),
    ".lte": ("elegant",),
    ".bmad": ("bmad",),
    ".dat": ("tracewin",),
    ".xml": ("pyorbit",),
}
_PYORBIT_RE = re.compile(r"<accElement\b")


def is_binary(head: bytes) -> bool:
    """True when *head* (the first bytes of a file) does not look like text.

    A NUL byte is decisive (TraceWin ``Density_*.dat`` dumps, ``.DS_Store``); otherwise
    more than 10 % control bytes outside the usual whitespace set counts as binary.
    UTF-16 files with a BOM are text.
    """
    if not head:
        return False
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return False
    if b"\x00" in head:
        return True
    control = sum(1 for b in head if b < 32 and b not in (9, 10, 12, 13))
    return control > 0.1 * len(head)


def _decode(head: bytes) -> str:
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return head.decode("utf-16", errors="replace")
    return head.decode("utf-8", errors="replace")


def _strip_comment(line: str) -> str:
    for prefix in _COMMENT_PREFIXES:
        if line.lstrip().startswith(prefix):
            return ""
    # MAD/Elegant/Bmad trailing comments; keep TraceWin ';' comments out of statement text
    for marker in ("!", "//"):
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return line


def _is_numeric_row(line: str) -> bool:
    tokens = line.replace(",", " ").split()
    return bool(tokens) and all(_NUMBER_RE.match(t) for t in tokens)


def sniff_text(text: str, path: str | os.PathLike[str] | None = None) -> str:
    """Classify *text* (the head of a file) into one of :data:`FORMATS`.

    *path* only supplies the suffix prior (``.lte`` -> elegant, ``.FLAT`` -> mad8,
    ``.lat`` -> mad8/flame, ...) used to break ties between dialects that share the
    ``NAME: TYPE, key=value`` grammar.  Content always wins over the suffix.
    """
    name = Path(path).name if path is not None else ""
    suffix = Path(name).suffix.lower()
    lower_name = name.lower()

    if suffix in FIELDMAP_SUFFIXES:
        return "fieldmap"
    if lower_name.endswith((".pals.yaml", ".pals.yml", ".pals.json")):
        return "pals"
    if _PYORBIT_RE.search(text):
        return "pyorbit"

    nonblank = [ln for ln in text.splitlines() if ln.strip()]
    if not nonblank:
        return "unknown"
    first = nonblank[0]

    if _TFS_RE.match(first):
        return "tfs"
    if any(r.match(first) for r in _TW_OUTPUT_RES):
        return "tracewin_output"
    if sum(1 for ln in nonblank if _IMPACTX_RE.match(ln)) >= 2:
        return "impactx"
    if _PALS_RE.search(text) and suffix in (".yaml", ".yml", ".json"):
        return "pals"

    code = [_strip_comment(ln) for ln in nonblank]
    code = [ln for ln in code if ln.strip()][:SNIFF_LINES]
    nonblank = nonblank[: 4 * SNIFF_LINES]  # raw lines only feed the RPN / IMPACT-Z row counts

    tw_cards = sum(1 for ln in code if _TW_RE.match(ln))
    mad_defs = sum(1 for ln in code if _MAD_DEF_RE.match(ln))
    semis = sum(1 for ln in code if ";" in ln)
    assigns = sum(1 for ln in code if _MAD_ASSIGN_RE.match(ln))
    line_defs = sum(1 for ln in code if _LINE_DEF_RE.search(ln))
    continuations = sum(1 for ln in code if _CONTINUATION_RE.search(ln))
    bare_use = sum(1 for ln in code if _BARE_USE_RE.match(ln))
    returns = sum(1 for ln in code if _RETURN_RE.match(ln))
    impactz_rows = sum(1 for ln in nonblank if _IMPACTZ_ROW_RE.match(ln))
    body = "\n".join(code)

    if tw_cards and tw_cards >= mad_defs:
        return "tracewin"
    if impactz_rows >= 3 and not mad_defs and not tw_cards:
        return "impactz"
    if impactz_rows >= 1 and lower_name in ("impactz.in", "impact.in", "impactz.in.txt") and not mad_defs:
        return "impactz"

    madx_keys = sum(1 for r in _MADX_KEYS if r.search(body))
    flame_keys = sum(1 for r in _FLAME_KEYS if r.search(body))
    bmad_keys = sum(1 for r in _BMAD_KEYS if r.search(body))
    elegant_types = sum(1 for ln in code if _ELEGANT_TYPE_RE.search(ln))
    elegant_rpn = sum(1 for ln in nonblank if _ELEGANT_RPN_RE.match(ln))
    mad_family = bool(
        mad_defs
        or line_defs
        or (assigns and semis)
        or madx_keys
        or flame_keys
        or bmad_keys
        or elegant_types
        or bare_use
    )
    if mad_family:
        prior = _SUFFIX_PRIOR.get(suffix, ())
        # MAD-X and FLAME terminate (nearly) every statement with ';'; MAD8, Elegant and
        # Bmad never do, so a stray ';' in a comment or string must not flip the family.
        semi_ratio = semis / max(1, len(code))
        with_semis = 10 if semi_ratio >= 0.25 else 0
        no_semis = 0 if with_semis else 10
        scores = {
            "madx": 10 * madx_keys + (10 if (with_semis and ":=" in body) else 0) + with_semis,
            "mad8": 10 * bool(continuations) + 10 * bool(bare_use or returns) + no_semis,
            "elegant": 10 * bool(elegant_types)
            + elegant_types
            + 10 * bool(elegant_rpn)
            + 10 * bool(continuations)
            + 10 * bool(bare_use or returns)
            + no_semis,
            "bmad": 10 * bmad_keys + 10 * bool(bare_use) + no_semis,
            "flame": 10 * flame_keys + with_semis,
        }
        for fmt in prior:
            if fmt in scores:
                scores[fmt] += 3
        return max(scores, key=lambda k: (scores[k], -list(scores).index(k)))

    numeric_rows = sum(1 for ln in code if _is_numeric_row(ln))
    if numeric_rows >= 2 and numeric_rows >= 0.6 * len(code):
        return "table"
    return "unknown"


def sniff_format(path: str | os.PathLike[str], head: bytes | None = None) -> str:
    """Detect the format of the file at *path* (reads :data:`SNIFF_BYTES` unless *head* given).

    Returns ``"binary"`` for non-text files so callers can skip them.
    """
    p = Path(path)
    if head is None:
        with open(p, "rb") as fh:
            head = fh.read(SNIFF_BYTES)
    if is_binary(head):
        return "binary"
    return sniff_text(_decode(head), p)


# --------------------------------------------------------------------------- helpers


def corpus_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the corpus root from *root*, else ``$LATTIX_CORPUS_DIR``."""
    if root is None:
        root = os.environ.get(ENV_VAR)
    if not root:
        raise ValueError(f"no corpus root: pass --root or set {ENV_VAR}")
    return Path(root).expanduser().resolve()


def sha256_of(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, streamed in *chunk*-byte blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


_SLUG_BAD = re.compile(r"[^a-z0-9._+-]+")


def slugify(rel: str | os.PathLike[str]) -> str:
    """Lower-case, path-preserving slug: ``Old design/B1 (v2).dat`` -> ``old-design/b1-v2.dat``."""
    parts = []
    for part in Path(rel).as_posix().split("/"):
        s = _SLUG_BAD.sub("-", part.lower())
        s = re.sub(r"-{2,}", "-", s)
        s = re.sub(r"-+(?=\.)|(?<=\.)-+", "", s).strip("-")
        if s:
            parts.append(s)
    return "/".join(parts)


def expand_braces(pattern: str) -> list[str]:
    """Shell-style ``{a,b}`` expansion (nested braces allowed): ``x.{lat,lte}`` -> two patterns."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end() :]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(head + alt + tail))
    return out


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    return any(part.startswith(".") for part in rel.parts)


def expand_source_paths(source: dict[str, Any]) -> list[Path]:
    """All files a source entry points at: sorted, de-duplicated, hidden files skipped.

    Each pattern in ``source["paths"]`` is brace-expanded, then treated as a glob
    (``**`` recurses), a single file or a directory (recursed).  Relative patterns are
    anchored at ``source["root"]``.
    """
    root = Path(source["root"]).expanduser()
    files: set[Path] = set()
    for raw in source.get("paths", []):
        for pattern in expand_braces(str(raw)):
            pat = Path(pattern)
            full = pat if pat.is_absolute() else root / pat
            for hit in sorted(glob.glob(str(full), recursive=True)):
                hp = Path(hit)
                if hp.is_dir():
                    for dirpath, dirnames, filenames in os.walk(hp):
                        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                        for fn in filenames:
                            if not fn.startswith("."):
                                files.add(Path(dirpath) / fn)
                elif hp.is_file() and not _is_hidden(hp, root):
                    files.add(hp)
    return sorted(files)


def load_sources(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read and validate a sources file; returns its ``sources`` list."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    sources = doc.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{path}: expected a top-level 'sources' list")
    seen: set[str] = set()
    for i, src in enumerate(sources):
        for key in ("name", "root", "paths", "origin", "license", "redistributable"):
            if key not in src:
                raise ValueError(f"{path}: source #{i} lacks required key '{key}'")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(src["name"])):
            raise ValueError(f"{path}: source name {src['name']!r} must match [a-z0-9-]+")
        if src["name"] in seen:
            raise ValueError(f"{path}: duplicate source name {src['name']!r}")
        seen.add(src["name"])
        if not isinstance(src["redistributable"], bool):
            raise ValueError(f"{path}: source {src['name']!r}: redistributable must be a bool")
        for fmt in src.get("formats", []) or []:
            if fmt not in FORMATS:
                raise ValueError(f"{path}: source {src['name']!r}: unknown format {fmt!r}")
        if src.get("format") is not None and src["format"] not in FORMATS:
            raise ValueError(f"{path}: source {src['name']!r}: unknown format {src['format']!r}")
    return sources


def _store_path(path: Path, root: Path) -> str:
    """Manifest path field: relative to *root* when inside it, else absolute."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_path(root: str | os.PathLike[str], entry: dict[str, Any]) -> Path:
    """Absolute path of a manifest *entry* (its ``path`` may be relative to *root*)."""
    p = Path(entry["path"])
    return p if p.is_absolute() else Path(root) / p


# --------------------------------------------------------------------------- build


@dataclass
class BuildResult:
    """What :func:`build_manifest` produced, plus what it left out and why."""

    manifest: dict[str, Any]
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """``(path, reason)`` for files skipped: too large, binary, filtered by ``formats``."""
    ambiguities: list[tuple[str, str]] = field(default_factory=list)
    """``(path, note)`` for entries whose sniff was weak (unknown/table) or overridden."""

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self.manifest["entries"]

    def counts_by_format(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e["format"]] = out.get(e["format"], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def build_manifest(
    root: str | os.PathLike[str],
    sources_path: str | os.PathLike[str],
    *,
    carry_from: list[dict[str, Any]] | None = None,
    max_size: int | None = None,
) -> BuildResult:
    """Walk every source, sniff and hash every kept file, return the manifest document.

    *carry_from* (normally the previous manifest's entries) supplies
    ``expected_warning_classes`` / ``anchors`` for ids that still exist.
    """
    rootp = corpus_root(root)
    limit = MAX_FILE_SIZE if max_size is None else max_size
    sources = load_sources(sources_path)
    previous = {e["id"]: e for e in (carry_from or [])}
    result = BuildResult(
        manifest={
            "manifest_version": MANIFEST_VERSION,
            "generated": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
            "root": rootp.as_posix(),
            "sources_file": Path(sources_path).resolve().as_posix(),
            "entries": [],
        }
    )
    ids: set[str] = set()
    for src in sources:
        src_root = Path(src["root"]).expanduser().resolve()
        keep = set(src.get("formats") or [])
        override = src.get("format")
        for path in expand_source_paths(src):
            spath = path.as_posix()
            size = path.stat().st_size
            if size > limit:
                result.skipped.append((spath, f"too large ({size} B > {limit} B)"))
                continue
            detected = sniff_format(path)
            if detected == "binary":
                result.skipped.append((spath, "binary"))
                continue
            fmt = override or detected
            if keep and fmt not in keep:
                result.skipped.append((spath, f"filtered: sniffed as {detected}"))
                continue
            if detected in ("unknown", "table") and override is None:
                result.ambiguities.append((spath, f"weak sniff: {detected}"))
            elif override is not None and override != detected:
                result.ambiguities.append((spath, f"override {override} (sniffed {detected})"))
            try:
                rel = path.resolve().relative_to(src_root)
            except ValueError:
                rel = Path(path.name)
            base_id = f"{src['name']}/{slugify(rel)}"
            entry_id, n = base_id, 1
            while entry_id in ids:
                n += 1
                entry_id = f"{base_id}~{n}"
            ids.add(entry_id)
            old = previous.get(entry_id, {})
            entry: dict[str, Any] = {
                "id": entry_id,
                "path": _store_path(path, rootp),
                "source": src["name"],
                "format": fmt,
                "sha256": sha256_of(path),
                "size": size,
                "origin": src["origin"],
                "license": src["license"],
                "redistributable": src["redistributable"],
            }
            if src.get("notes"):
                entry["notes"] = str(src["notes"]).strip()
            if override is not None and override != detected:
                entry["detected_format"] = detected
            entry["expected_warning_classes"] = list(old.get("expected_warning_classes", []))
            entry["anchors"] = list(old.get("anchors", []))
            result.entries.append(entry)
    result.entries.sort(key=lambda e: e["id"])
    return result


def write_manifest(
    root: str | os.PathLike[str], manifest: dict[str, Any], out: str | os.PathLike[str] | None = None
) -> Path:
    """Write *manifest* to *out* (default ``<root>/manifest.yaml``); returns the path."""
    rootp = corpus_root(root)
    target = Path(out) if out is not None else rootp / MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True, width=120)
    return target


# --------------------------------------------------------------------------- read / verify


def load_manifest(root: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Entries of ``<root>/manifest.yaml`` (root defaults to ``$LATTIX_CORPUS_DIR``)."""
    rootp = corpus_root(root)
    path = rootp / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no manifest at {path}; run 'python -m lattix.corpus build-manifest'")
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    entries = doc.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected an 'entries' list")
    return entries


def iter_decks(
    root: str | os.PathLike[str] | None = None,
    fmt: str | Iterable[str] | None = None,
    redistributable: bool | None = None,
    source: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield manifest entries (copies, with ``path`` made absolute), optionally filtered.

    *fmt* is one format id or an iterable of them; *redistributable* filters on the
    flag when not ``None``; *source* filters on the source name.
    """
    rootp = corpus_root(root)
    fmts = {fmt} if isinstance(fmt, str) else (set(fmt) if fmt is not None else None)
    for entry in load_manifest(rootp):
        if fmts is not None and entry.get("format") not in fmts:
            continue
        if redistributable is not None and bool(entry.get("redistributable")) != redistributable:
            continue
        if source is not None and entry.get("source") != source:
            continue
        e = dict(entry)
        e["path"] = resolve_path(rootp, entry).as_posix()
        yield e


@dataclass(frozen=True)
class Problem:
    """One verification failure."""

    id: str
    path: str
    kind: str
    """``missing``, ``size`` or ``sha256``."""
    detail: str = ""


def verify_manifest(root: str | os.PathLike[str] | None = None) -> list[Problem]:
    """Recompute size and sha256 of every entry; return the mismatches (empty when clean)."""
    rootp = corpus_root(root)
    problems: list[Problem] = []
    for entry in load_manifest(rootp):
        path = resolve_path(rootp, entry)
        if not path.is_file():
            problems.append(Problem(entry["id"], path.as_posix(), "missing"))
            continue
        size = path.stat().st_size
        if size != entry.get("size"):
            problems.append(Problem(entry["id"], path.as_posix(), "size", f"{entry.get('size')} -> {size}"))
            continue
        digest = sha256_of(path)
        if digest != entry.get("sha256"):
            problems.append(
                Problem(entry["id"], path.as_posix(), "sha256", f"{entry.get('sha256')} -> {digest}")
            )
    return problems


# --------------------------------------------------------------------------- CLI


def cmd_build_manifest(args: argparse.Namespace) -> int:
    root = corpus_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    carry: list[dict[str, Any]] | None = None
    if not args.fresh and (root / MANIFEST_NAME).is_file():
        carry = load_manifest(root)
    result = build_manifest(root, args.sources, carry_from=carry)
    target = write_manifest(root, result.manifest, args.out)
    print(f"wrote {target} ({len(result.entries)} entries)")
    for fmt, n in result.counts_by_format().items():
        print(f"  {fmt:16s} {n:6d}")
    if result.skipped:
        reasons: dict[str, int] = {}
        for _, why in result.skipped:
            key = why.split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"skipped {len(result.skipped)} files:")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}  {why}")
    if result.ambiguities:
        print(f"sniff notes ({len(result.ambiguities)}):")
        for path, note in result.ambiguities:
            print(f"  {note:40s} {path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = corpus_root(args.root)
    problems = verify_manifest(root)
    total = len(load_manifest(root))
    for p in problems:
        detail = f"  ({p.detail})" if p.detail else ""
        print(f"{p.kind:8s} {p.id}  {p.path}{detail}")
    if problems:
        print(f"FAILED: {len(problems)} of {total} entries missing or changed")
        return 1
    print(f"OK: {total} entries verified")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = corpus_root(args.root)
    n = 0
    for e in iter_decks(
        root,
        fmt=args.format,
        redistributable=True if args.redistributable else None,
        source=args.source,
    ):
        n += 1
        flag = "R" if e.get("redistributable") else "-"
        print(f"{e['id']:70s} {e['format']:16s} {flag} {e['size']:>10d}  {e['path']}")
    print(f"{n} entries")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m lattix.corpus",
        description="Build, verify and list the lattix corpus manifest.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def add_root(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", help=f"corpus root (default: ${ENV_VAR})")

    b = sub.add_parser("build-manifest", help="walk a sources file and write manifest.yaml")
    add_root(b)
    b.add_argument("--sources", default="corpus/sources.yaml", help="sources YAML file")
    b.add_argument("--out", help=f"manifest path (default: <root>/{MANIFEST_NAME})")
    b.add_argument(
        "--fresh", action="store_true", help="do not carry curated fields over from the old manifest"
    )
    b.set_defaults(func=cmd_build_manifest)

    v = sub.add_parser("verify", help="recompute every sha256; exit 1 on any mismatch")
    add_root(v)
    v.set_defaults(func=cmd_verify)

    ls = sub.add_parser("list", help="print manifest entries")
    add_root(ls)
    ls.add_argument("--format", choices=FORMATS, help="only this format")
    ls.add_argument("--redistributable", action="store_true", help="only redistributable entries")
    ls.add_argument("--source", help="only this source name")
    ls.set_defaults(func=cmd_list)
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
