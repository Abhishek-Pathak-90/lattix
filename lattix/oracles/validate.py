"""One validation run shared by ``lattix validate`` and the browser UI: the same lattice as one deck per
format, every requested engine on the deck it reads, and the pairwise comparison of the cumulative maps
(:mod:`lattix.oracles.compare`)."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lattix.oracles.base import _REGISTRY, BeamSpec, OracleResult, _load_adapters, get_oracle
from lattix.oracles.compare import BLOCKS, PairComparison, compare_all


@dataclass
class EngineRun:
    engine: str
    fmt: str | None = None
    result: OracleResult | None = None
    skipped: str | None = None          # why the engine did not run (unavailable, no deck, stopped)
    error: str | None = None            # the engine ran and failed
    seconds: float = 0.0


@dataclass
class ValidationOutcome:
    runs: dict[str, EngineRun] = field(default_factory=dict)
    comparisons: list[PairComparison] = field(default_factory=list)
    worst: float = 0.0                  # max |ΔR̂cum| over the pairs (nan-free)

    @property
    def results(self) -> dict[str, OracleResult]:
        return {n: r.result for n, r in self.runs.items() if r.result is not None}


def engines_for(fmt: str) -> list[str]:
    """Registered engine adapters that read decks of ``fmt`` directly."""
    _load_adapters()
    return sorted(n for n, cls in _REGISTRY.items() if fmt in getattr(cls, "formats", ()))


def _pick_deck(oracle, decks: dict[str, Path], pinned: str | None) -> str | None:
    if pinned is not None:
        return pinned if pinned in decks and pinned in oracle.formats else None
    return next((f for f in oracle.formats if f in decks), None)


def run_validation(decks: dict[str, Path], engines: list[str | tuple[str, str | None]], beam: BeamSpec, *,
                   workdir: Path | None = None, log: Callable[[str], None] | None = None,
                   progress: Callable[[int, int, str], None] | None = None,
                   should_stop: Callable[[], bool] | None = None, lat=None) -> ValidationOutcome:
    """Run every engine on the deck it reads (``{format: path}``; an engine given as ``(name, format)``
    is pinned to that deck) and compare all pairs.  An unavailable engine, one without a deck or one that
    raises is recorded, never fatal; ``log`` receives the one-line summaries ``lattix validate`` prints,
    ``progress(done, total, current)`` the count, ``should_stop()`` is polled between engines."""
    out = ValidationOutcome()
    total = len(engines)
    for k, spec in enumerate(engines):
        name, pinned = (spec, None) if isinstance(spec, str) else (spec[0], spec[1])
        run = EngineRun(name)
        out.runs[name] = run
        if progress:
            progress(k, total, name)
        if should_stop and should_stop():
            run.skipped = "stopped"
            continue
        try:
            o = get_oracle(name)
        except KeyError as exc:
            run.skipped = str(exc)
            _say(log, f"{name:8s} skipped: {exc}")
            continue
        ok, why = o.available()
        if not ok:
            run.skipped = why
            _say(log, f"{name:8s} skipped: {why}")
            continue
        if lat is not None:
            from lattix.crossval import constant_p0_note

            note = constant_p0_note(lat, (name,))
            if note:
                run.skipped = note
                _say(log, f"{name:8s} skipped: {note}")
                continue
        fmt = _pick_deck(o, decks, pinned)
        if fmt is None:
            run.skipped = f"no deck in a format it reads ({o.formats})"
            _say(log, f"{name:8s} skipped: {run.skipped}")
            continue
        run.fmt = fmt
        wd = Path(workdir) / name if workdir else None
        t0 = time.perf_counter()
        try:
            r = o.run(decks[fmt], fmt=fmt, beam=beam, workdir=wd)
        except Exception as exc:  # noqa: BLE001 - one engine's failure must not hide the others
            run.error = f"{type(exc).__name__}: {exc}"[:6000]
            run.seconds = time.perf_counter() - t0
            _say(log, f"{name:8s} failed: {run.error}")
            continue
        run.result = r
        run.seconds = time.perf_counter() - t0
        _say(log, f"{name:8s} {fmt:8s} n={r.n:5d} L={r.total_length:.9g} m  "
                  f"W_end={r.ref_kinetic_eV_out[-1]:.6e} eV  warnings={len(r.warnings)}")
    if progress:
        progress(total, total, "")
    results = out.results
    if len(results) >= 2:
        out.comparisons = compare_all(results)
        vals = [pc.max_rcum_abs for pc in out.comparisons if math.isfinite(pc.max_rcum_abs)]
        out.worst = max(vals) if vals else 0.0
    return out


def _say(log: Callable[[str], None] | None, text: str) -> None:
    if log is not None:
        log(text)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def comparison_to_dict(pc: PairComparison, verdict=None) -> dict:
    """A JSON-ready :class:`PairComparison` (``per_boundary`` as rows, NaN as null, the printed row)."""
    d = {
        "a": pc.a, "b": pc.b, "n_shared": pc.n_shared, "length_a": pc.length_a, "length_b": pc.length_b,
        "max_rcum_abs": _num(pc.max_rcum_abs), "max_rcum_rel": _num(pc.max_rcum_rel),
        "final_abs": _num(pc.final_abs), "energy_rel": _num(pc.energy_rel),
        "survey_end_abs": _num(pc.survey_end_abs) if pc.survey_end_abs is not None else None,
        "blocks": {k: _num(v) for k, v in pc.blocks.items()},
        "notes": list(pc.notes), "row": pc.row(),
        "per_boundary": [{"s": float(s), "blocks": {k: _num(v) for k, v in blocks.items()}}
                         for s, blocks in pc.per_boundary],
    }
    if verdict is not None:
        d["verdict"] = {
            "ok": verdict.ok, "tier": verdict.tier, "metric": _num(verdict.metric),
            "metric_rel": _num(verdict.metric_rel), "energy_rel": _num(verdict.energy_rel),
            "blocks_used": sorted(verdict.blocks_used), "map_tol": verdict.map_tol,
            "energy_tol": verdict.energy_tol, "energy_checked": verdict.energy_checked, "floor": verdict.floor,
            "note": verdict.note, "notes": list(verdict.notes),
        }
    return d


def outcome_to_dict(o: ValidationOutcome, *, decks: dict[str, Path], beam: BeamSpec,
                    verdicts: dict[tuple[str, str], object] | None = None, full: bool = False) -> dict:
    """The JSON payload of a run: per-engine summaries (with the per-element arrays when ``full``),
    the comparisons and the worst map difference."""
    engines = {}
    for name, run in o.runs.items():
        e: dict = {"format": run.fmt, "skipped": run.skipped, "error": run.error, "seconds": run.seconds}
        r = run.result
        if r is not None:
            e.update({"n": r.n, "total_length": float(r.total_length),
                      "W_end": float(r.ref_kinetic_eV_out[-1]), "warnings": list(r.warnings)})
            if full:
                e.update({"names": list(r.names), "s_out": r.s_out.tolist(),
                          "R_cum_common": r.to_common().R_cum.tolist(), "ke_out": r.ref_kinetic_eV_out.tolist()})
        engines[name] = e
    comps = [comparison_to_dict(pc, (verdicts or {}).get((pc.a, pc.b))) for pc in o.comparisons]
    return {"decks": {f: str(p) for f, p in decks.items()},
            "beam": {"species": beam.species, "kinetic_energy_eV": beam.kinetic_energy_eV,
                     "frequency_Hz": beam.frequency_Hz, "betx": beam.betx, "alfx": beam.alfx, "bety": beam.bety,
                     "alfy": beam.alfy, "dx": beam.dx, "dpx": beam.dpx},
            "engines": engines, "comparisons": comps, "worst": _num(o.worst), "blocks": list(BLOCKS)}


def read_lattice(decks: dict[str, Path], beam: BeamSpec | None = None):
    """The IR of the first deck that reads (the source deck comes first), with the beam the engines were
    given where the reader takes it — what the battery's verdict looks at (element kinds, species)."""
    import inspect

    from lattix.formats.base import FORMATS, read

    for fmt, path in decks.items():
        spec = FORMATS.get(fmt)
        if spec is None or spec.reader() is None:
            continue
        try:
            params = inspect.signature(spec.reader().read).parameters
        except (TypeError, ValueError):
            params = {}
        opts = {}
        if beam is not None:
            for key, val in (("species", beam.species), ("kinetic_energy_eV", beam.kinetic_energy_eV),
                             ("frequency_Hz", beam.frequency_Hz)):
                if key in params and val is not None:
                    opts[key] = val
        try:
            lat, _rep = read(path, fmt, **opts)
        except Exception:  # noqa: BLE001 - a deck an engine could run but lattix cannot read: no verdict
            continue
        return lat
    return None


def verdicts_for(outcome: ValidationOutcome, lat, tier: str = "lossy", codes=()) -> dict:
    """The battery's verdict for every comparison (``lattix.crossval.engine_verdict``): which map blocks the
    pair can be held to (a HELIX bend map has no path row, a field-map cavity no shared longitudinal model …),
    the tier's tolerance and the caveats.  ``tier`` and ``codes`` are the translation's (report only when
    unknown)."""
    from lattix.crossval import engine_verdict

    if lat is None:
        return {}
    code_counts = dict(codes) if isinstance(codes, dict) else {str(c): 1 for c in codes}
    out = {}
    for pc in outcome.comparisons:
        ra = outcome.runs.get(pc.a).result if pc.a in outcome.runs else None
        rb = outcome.runs.get(pc.b).result if pc.b in outcome.runs else None
        if ra is None or rb is None:
            continue
        out[(pc.a, pc.b)] = engine_verdict(pc, pc.a, pc.b, lat, tier, code_counts, ra, rb)
    return out
