"""``lattix`` command line.

Phase 0 commands: ``oracles`` (availability), ``fingerprint`` (basis
conventions per engine), ``validate`` (run several engines on the same
lattice — one deck per format — and compare cumulative optics).  ``convert``,
``inspect`` and ``report`` arrive with the IR in Phase 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from lattix._version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from lattix.oracles.base import BeamSpec


def _beam_from_args(a) -> BeamSpec:
    from lattix.oracles.base import BeamSpec

    return BeamSpec(species=a.species, kinetic_energy_eV=a.ke, frequency_Hz=a.freq,
                    betx=a.betx, alfx=a.alfx, bety=a.bety, alfy=a.alfy, dx=a.dx, dpx=a.dpx)


def _add_beam_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--species", default="proton")
    p.add_argument("--ke", type=float, default=2.1e6, help="kinetic energy [eV]")
    p.add_argument("--freq", type=float, default=None, help="RF clock [Hz]")
    p.add_argument("--betx", type=float, default=10.0)
    p.add_argument("--alfx", type=float, default=0.0)
    p.add_argument("--bety", type=float, default=10.0)
    p.add_argument("--alfy", type=float, default=0.0)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dpx", type=float, default=0.0)


def cmd_oracles(a) -> int:
    from lattix.oracles import available_oracles

    for name, (ok, why) in available_oracles().items():
        print(f"{name:10s} {'OK ' if ok else '-- '} {why}")
    return 0


def cmd_fingerprint(a) -> int:
    from lattix.oracles import get_oracle
    from lattix.oracles.fingerprint import DECKS, fingerprint

    engines = a.engines.split(",") if a.engines else list(DECKS)
    wd = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="lattix_fp_"))
    out = {}
    for e in engines:
        ok, why = get_oracle(e).available()
        if not ok:
            print(f"{e:8s} skipped: {why}")
            continue
        fp = fingerprint(e, wd / e)
        out[e] = fp
        d, c = fp["drift"], fp["cavity"]
        print(f"{e:8s} drift R56 native={d['R56_native']:+.4e} common={d['R56_common']:+.4e} "
              f"expected={d['R56_expected']:+.4e} err={d['max_abs_err_common']:.1e} | "
              f"cavity R65 native={c['R65_native']:+.4e} common={c['R65_common']:+.4e} "
              f"gain={c['gain_eV']:+.4e} eV (expected {c['gain_expected_eV']:+.4e})")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


def cmd_validate(a) -> int:
    from lattix.oracles import get_oracle, guess_format
    from lattix.oracles.compare import compare_all

    decks: dict[str, Path] = {}
    for spec in a.deck:
        fmt, _, path = spec.partition("=")
        if not path:
            path, fmt = fmt, guess_format(Path(fmt))
        decks[fmt] = Path(path)
    engines = a.oracles.split(",")
    beam = _beam_from_args(a)
    results = {}
    for e in engines:
        o = get_oracle(e)
        ok, why = o.available()
        if not ok:
            print(f"{e:8s} skipped: {why}", file=sys.stderr)
            continue
        fmt = next((f for f in o.formats if f in decks), None)
        if fmt is None:
            print(f"{e:8s} skipped: no deck in a format it reads ({o.formats})", file=sys.stderr)
            continue
        wd = Path(a.workdir) / e if a.workdir else None
        results[e] = o.run(decks[fmt], fmt=fmt, beam=beam, workdir=wd)
        r = results[e]
        print(f"{e:8s} {fmt:8s} n={r.n:5d} L={r.total_length:.9g} m  "
              f"W_end={r.ref_kinetic_eV_out[-1]:.6e} eV  warnings={len(r.warnings)}")
    if len(results) < 2:
        print("need at least two engines to compare", file=sys.stderr)
        return 2
    worst = 0.0
    comparisons = compare_all(results)
    for pc in comparisons:
        print(pc.row() + ("  " + "; ".join(pc.notes) if pc.notes else ""))
        worst = max(worst, pc.max_rcum_abs)
    if a.json:
        payload = {e: {"names": r.names, "s_out": r.s_out.tolist(),
                       "R_cum_common": r.to_common().R_cum.tolist(),
                       "ke_out": r.ref_kinetic_eV_out.tolist()} for e, r in results.items()}
        Path(a.json).write_text(json.dumps(payload) + "\n")
    if a.html:
        from lattix.report_html import write_validate_html

        write_validate_html(a.html, comparisons, title=", ".join(str(p) for p in decks.values()))
    if a.tol is not None and worst > a.tol:
        print(f"FAIL: max cumulative-map difference {worst:.3e} > tol {a.tol:.1e}")
        return 1
    return 0


def cmd_convert(a) -> int:
    from lattix.formats import translate

    rep = translate(a.src, a.dst, src_fmt=a.from_fmt, dst_fmt=a.to, strict=a.strict,
                    read_options=_kv(a.read_option), write_options=_kv(a.write_option))
    rep.print_summary()
    if a.report:
        rep.to_json(a.report)
    return 0 if rep.ok else 1


def cmd_inspect(a) -> int:
    from collections import Counter

    from lattix.formats import read
    from lattix.ir.walk import propagate

    lat, rep = read(a.src, a.from_fmt, **_kv(a.read_option))
    warns: list[str] = []
    placed = propagate(lat, warnings=warns)
    kinds = Counter(p.element.kind for p in placed)
    r0, r1 = lat.reference, placed[-1].ref_out if placed else lat.reference
    print(f"{a.src}: {len(placed)} placed elements, {len(lat.elements)} definitions, "
          f"{len(lat.lines)} lines, L = {placed[-1].s_out if placed else 0:.6f} m")
    print(f"reference: {r0.species.name} {r0.kinetic_energy_eV * 1e-6:.6g} MeV -> "
          f"{r1.kinetic_energy_eV * 1e-6:.6g} MeV; RF clock {r0.rf_frequency_Hz}")
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {n}")
    if a.elements:
        for p in placed:
            print(f"{p.index:5d} {p.s_out:12.6f} {p.element.kind:14s} {p.name:24s} L={p.length:.6g}")
    rep.print_summary()
    for w in warns[:10]:
        print("walk:", w, file=sys.stderr)
    return 0


def cmd_report(a) -> int:
    from lattix.formats import read, write

    lat, rep_in = read(a.src, a.from_fmt, **_kv(a.read_option))
    wr_fmt = a.to
    tmp = Path(tempfile.mkdtemp(prefix="lattix_report_")) / ("out." + wr_fmt)
    rep = write(lat, tmp, wr_fmt, strict=False, **_kv(a.write_option))
    rep.entries = rep_in.entries + rep.entries
    rep.source_format = rep_in.source_format
    rep.source_file = rep_in.source_file
    print(rep.summary())
    if a.json:
        rep.to_json(a.json)
    return 0


def _kv(items: list[str] | None) -> dict:
    out: dict = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = float(v) if v.replace(".", "", 1).replace("e", "", 1).lstrip("-+").isdigit() else v
        except ValueError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lattix", description=__doc__)
    p.add_argument("--version", action="version", version=f"lattix {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("oracles", help="list engine adapters and their availability")
    s.set_defaults(func=cmd_oracles)

    s = sub.add_parser("fingerprint", help="measure each engine's basis conventions")
    s.add_argument("--engines", default=None, help="comma list (default: all)")
    s.add_argument("--workdir", default=None)
    s.add_argument("--json", default=None)
    s.set_defaults(func=cmd_fingerprint)

    s = sub.add_parser("validate", help="run engines on the same lattice and compare optics")
    s.add_argument("--deck", action="append", required=True,
                   help="FORMAT=PATH (repeatable; one deck per format, e.g. madx=fodo.madx bmad=fodo.bmad)")
    s.add_argument("--oracles", default="madx,xtrack,bmad,helix")
    s.add_argument("--tol", type=float, default=None, help="fail if max ΔR̂cum exceeds this")
    s.add_argument("--html", default=None, help="write an HTML report: per-boundary map differences vs s")
    s.add_argument("--workdir", default=None)
    s.add_argument("--json", default=None)
    _add_beam_args(s)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("convert", help="translate a deck between formats")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--from", dest="from_fmt", default=None)
    s.add_argument("--to", default=None)
    s.add_argument("--strict", action="store_true", help="fail on the first LOSSY/DROPPED element")
    s.add_argument("--report", default=None, help="write the fidelity report JSON here")
    s.add_argument("--read-option", action="append", help="KEY=VALUE for the reader (e.g. species=h-)")
    s.add_argument("--write-option", action="append", help="KEY=VALUE for the writer (e.g. energy_mode=local)")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("inspect", help="summarise a deck through the IR")
    s.add_argument("src")
    s.add_argument("--from", dest="from_fmt", default=None)
    s.add_argument("--elements", action="store_true")
    s.add_argument("--read-option", action="append")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("report", help="fidelity report of a conversion without keeping the output")
    s.add_argument("src")
    s.add_argument("--from", dest="from_fmt", default=None)
    s.add_argument("--to", required=True)
    s.add_argument("--json", default=None)
    s.add_argument("--read-option", action="append")
    s.add_argument("--write-option", action="append")
    s.set_defaults(func=cmd_report)

    a = p.parse_args(argv)
    from lattix.fidelity import TranslationError

    try:
        return a.func(a)
    except TranslationError as exc:
        print(f"lattix: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
