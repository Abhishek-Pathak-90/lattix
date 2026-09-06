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
    "impactt": FormatSpec("impactt", ("impactt.in",), "lattix.formats.impactt", description="IMPACT-T ImpactT.in"),
    "scibmad": FormatSpec("scibmad", (".scibmad.jl", ".jl"), "lattix.formats.scibmad",
                          description="SciBmad / Beamlines.jl lattice (Julia)"),
    # reached through Bmad's own converters (lattix.formats.bmad_bridge): write-only unless noted
    "astra": FormatSpec("astra", (".astra",), "lattix.formats.bmad_bridge", reader_attr=None, writer_attr="AstraWriter",
                        description="Astra (via Bmad's bmad_to_astra)", options={"bridge": True, "doc": "bmad_bridge"}),
    "gpt": FormatSpec("gpt", (".gpt",), "lattix.formats.bmad_bridge", reader_attr=None, writer_attr="GptWriter",
                      description="GPT (via Bmad's bmad_to_gpt)", options={"bridge": True, "doc": "bmad_bridge"}),
    "csrtrack": FormatSpec("csrtrack", (".csrtrk.in", ".csrtrack"), "lattix.formats.bmad_bridge", reader_attr=None,
                           writer_attr="CsrtrackWriter", description="CSRtrack (via Bmad's bmad_to_csrtrack)",
                           options={"bridge": True, "doc": "bmad_bridge"}),
    "merlin": FormatSpec("merlin", (".merlin.tfs", ".tfs"), "lattix.formats.bmad_bridge", reader_attr=None,
                         writer_attr="MerlinWriter", description="Merlin++ TFS (via Bmad's bmad_to_merlin)",
                         options={"bridge": True, "doc": "bmad_bridge"}),
    "slicktrack": FormatSpec("slicktrack", (".slick",), "lattix.formats.bmad_bridge", reader_attr=None,
                             writer_attr="SlicktrackWriter", description="SLICKTRACK (via Bmad's bmad_to_slicktrack)",
                             options={"bridge": True, "doc": "bmad_bridge"}),
    "sad": FormatSpec("sad", (".sad",), "lattix.formats.bmad_bridge", reader_attr="SadReader", writer_attr="SadWriter",
                      description="SAD (via Tao 'write sad' / Bmad's sad_to_bmad.py)",
                      options={"bridge": True, "doc": "bmad_bridge"}),
    "sxf": FormatSpec("sxf", (".sxf",), "lattix.formats.bmad_bridge", reader_attr="SxfReader", writer_attr=None,
                      description="SXF (via Bmad's sxf_to_bmad.py)", options={"bridge": True, "doc": "bmad_bridge"}),
    "at": FormatSpec("at", (".at",), "lattix.formats.bmad_bridge", reader_attr="AtReader", writer_attr=None,
                     description="Accelerator Toolkit (via Bmad's accelerator_toolkit_to_bmad.py)",
                     options={"bridge": True, "doc": "bmad_bridge"}),
    "madng": FormatSpec("madng", (".madng",), "lattix.formats.madng", reader_attr=None,
                        description="MAD-NG Lua sequence (writer only, via xtrack)"),
    "cheetah": FormatSpec("cheetah", (".cheetah.json",), "lattix.formats.cheetah",
                          description="Cheetah LatticeJSON (cheetah.latticejson)"),
    "pyorbit": FormatSpec("pyorbit", (".pyorbit.xml", ".xml"), "lattix.formats.pyorbit",
                          description="PyORBIT3 linac XML (SNS_LinacLatticeFactory)"),
    "ocelot": FormatSpec("ocelot", (".ocelot.py", ".py"), "lattix.formats.ocelot",
                         description="Ocelot lattice module (MagneticLattice cell, python)"),
    "dynac": FormatSpec("dynac", (".dynac.in", ".dyn"), "lattix.formats.dynac",
                        description="DYNAC V6 deck (cm, kG, MV type codes)"),
    "synergia": FormatSpec("synergia", (".synergia.json",), "lattix.formats.synergia",
                           description="Synergia 3 lattice JSON (Lattice.as_json)"),
    "opal": FormatSpec("opal", (".opal.in", ".opal"), "lattix.formats.opal",
                       description="OPAL-T input deck (ELEMEDGE placement, 1-D field-map files)"),
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
        if isinstance(head, dict) and "elements" in head and "lattices" in head and \
                str(head.get("version", "")).startswith("cheetah"):
            return "cheetah"
        if isinstance(head, dict) and isinstance(head.get("value0"), dict) and "elements" in head["value0"] \
                and "reference_particle_value" in head["value0"]:
            return "synergia"
    return fmt


def read(path: str | Path, fmt: str | None = None, **options) -> tuple[Lattice, FidelityReport]:
    fmt = fmt or guess_format(path)
    rd = FORMATS[fmt].reader()
    if rd is None:
        raise ValueError(f"format {fmt!r} has no reader")
    lat, rep = rd.read(Path(path), **options)
    lat.restore_rf_focusing_marks()
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
    lattice, focus_entries = with_rf_focusing(lattice, fmt)
    try:
        rep = wr.write(lattice, Path(path), strict=strict, **options)
        for entry in focus_entries:
            rep.entries.append(entry)
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


def note_quad_higher_orders(el, rep, target: str) -> None:
    """A quadrupole that carries higher-order components (TraceWin ``G3..G6``) loses them in a
    target whose quadrupole holds only the gradient: say so in the ledger (never silently)."""
    mp = getattr(el, "multipole", None)
    if mp is None:
        return
    orders = sorted({n for n, v in mp.Bn.items() if n > 1 and v} | {n for n, v in mp.Bs.items() if n > 1 and v})
    if orders:
        rep.lossy("QUAD_HIGHER_ORDER_DROPPED",
                  f"{target} quadrupole holds only the gradient; multipole orders {orders} of "
                  f"{el.name!r} are dropped", element=el.name, kind=el.kind, orders=orders)


#: targets whose thin cavity has no transverse RF kick but which carry a first-order matrix
#: element, so TraceWin's thin-gap defocusing travels as an explicit thin lens
RF_FOCUSING_AS_MATRIX = frozenset({"madx", "elegant", "bmad", "xtrack", "pals", "impactx", "flame", "scibmad",
                                   "cheetah"})
#: targets that have neither (the kick is lost and recorded)
RF_FOCUSING_LOST = frozenset({"mad8", "impactz"})
_RF_FOCUS_SUFFIX = "_rfdefocus"


def with_rf_focusing(lattice: Lattice, fmt: str) -> tuple[Lattice, list]:
    """A copy of *lattice* in which every thin RF gap is followed by a ``Taylor`` thin lens
    carrying the TraceWin/HELIX RF defocusing (:func:`lattix.ir.rf.thin_gap_defocusing`), for
    targets in :data:`RF_FOCUSING_AS_MATRIX`; for :data:`RF_FOCUSING_LOST` targets the kick is
    only recorded.  A lattice that already carries the lenses (a re-read deck) is left alone,
    so write → read → write is a fixed point.  Returns the lattice and the ledger entries."""
    from lattix.fidelity import FidelityClass, FidelityEntry
    from lattix.ir.elements import Taylor
    from lattix.ir.lattice import LineItem
    from lattix.ir.rf import thin_gap_defocusing
    from lattix.ir.walk import propagate

    if fmt not in RF_FOCUSING_AS_MATRIX and fmt not in RF_FOCUSING_LOST:
        return lattice, []
    entries: list = []
    k_by_name: dict[str, float] = {}
    try:
        placed = propagate(lattice)
    except Exception:  # noqa: BLE001 - an unpropagatable lattice gets no lenses
        return lattice, []
    for i, p in enumerate(placed):
        e = p.element
        if e.kind != "RFCavity" or p.length != 0.0 or not e.rf.voltage_V or p.ref_out is None:
            continue
        nxt = placed[i + 1].element if i + 1 < len(placed) else None
        if nxt is not None and nxt.kind == "Taylor" and nxt.meta.get("rf_focusing_of") == e.name:
            continue                                  # already carried (a re-read deck)
        bg = p.ref_out.beta * p.ref_out.gamma
        k = thin_gap_defocusing(e.rf.voltage_V, e.rf.phase_rad, e.rf.frequency_Hz or p.ref_out.rf_frequency_Hz,
                                p.ref_out.species.mass_eV, bg)
        if k == 0.0:
            continue
        if fmt in RF_FOCUSING_LOST:
            entries.append(FidelityEntry(element=e.name, kind=e.kind, cls=FidelityClass.LOSSY,
                                         code="THIN_CAVITY_NO_RF_FOCUSING",
                                         message=f"the target's thin cavity has no transverse RF kick and no "
                                                 f"matrix element to carry it (k = {k:.6g} 1/m dropped)",
                                         details={"k_per_m": k}))
            continue
        k_by_name.setdefault(e.name, k)
    if not k_by_name:
        return lattice, entries
    new = lattice.model_copy(deep=True)
    for name, k in k_by_name.items():
        m = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        m[1][0] = k
        m[3][2] = k
        lens = Taylor(name=f"{name}{_RF_FOCUS_SUFFIX}", matrix=m, basis="common",
                      meta={"rf_focusing_of": name})
        new.elements[lens.name] = lens
        entries.append(FidelityEntry(element=name, kind="RFCavity", cls=FidelityClass.EQUIVALENT,
                                     code="THIN_GAP_RF_FOCUSING_AS_MATRIX",
                                     message=f"TraceWin thin-gap RF defocusing k = {k:.6g} 1/m written as the "
                                             f"thin lens '{lens.name}' after the cavity",
                                     details={"k_per_m": k}))
    for line in new.lines.values():
        items = []
        for it in line.items:
            items.append(it)
            if it.ref in k_by_name and not it.reverse:
                items.append(LineItem(ref=f"{it.ref}{_RF_FOCUS_SUFFIX}"))
            elif it.ref in k_by_name:
                items.insert(len(items) - 1, LineItem(ref=f"{it.ref}{_RF_FOCUS_SUFFIX}"))
        line.items = items
    return new, entries


def check_rules_coverage(writer: Writer) -> set[str]:
    """Kinds a writer's RULES table does not mention (must be empty; tested per writer)."""
    return set(ALL_KINDS) - set(writer.RULES)


def rule_table(**rules: Callable) -> dict[str, Callable]:
    return dict(rules)
