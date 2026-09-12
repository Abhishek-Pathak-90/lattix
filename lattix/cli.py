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
    from lattix.oracles import guess_format
    from lattix.oracles.validate import outcome_to_dict, read_lattice, run_validation, verdicts_for

    decks: dict[str, Path] = {}
    for spec in a.deck:
        fmt, _, path = spec.partition("=")
        if not path:
            path, fmt = fmt, guess_format(Path(fmt))
        decks[fmt] = Path(path)
    engines = a.oracles.split(",")
    beam = _beam_from_args(a)
    lat = read_lattice(decks, beam)
    outcome = run_validation(decks, engines, beam, workdir=Path(a.workdir) if a.workdir else None,
                             log=lambda line: print(line, file=sys.stderr if "skipped" in line else sys.stdout),
                             lat=lat)
    codes = {}
    for item in (a.codes or "").split(","):
        if item:
            name, _, n = item.partition("=")
            codes[name] = int(n) if n.isdigit() else 1
    verdicts = verdicts_for(outcome, lat, a.tier or "lossy", codes) if len(outcome.results) >= 2 else {}
    if a.json:
        # written before the exit-2 path too: a skipped/failed engine and its reason must reach the caller
        payload = outcome_to_dict(outcome, decks=decks, beam=beam, verdicts=verdicts, full=True)
        Path(a.json).write_text(json.dumps(payload) + "\n")
    if len(outcome.results) < 2:
        print("need at least two engines to compare", file=sys.stderr)
        return 2
    for pc in outcome.comparisons:
        print(pc.row() + ("  " + "; ".join(pc.notes) if pc.notes else ""))
        v = verdicts.get((pc.a, pc.b))
        if v is not None:
            tol = f"tolerance {v.map_tol:.1e}" if v.map_tol is not None else "report only"
            print(f"{'':21s}verdict: {'ok' if v.ok else 'NOT ok'} — tier {v.tier} ({tol}); blocks compared: "
                  f"{', '.join(sorted(v.blocks_used))}; max over them {v.metric_rel:.2e} (relative)"
                  + ("".join(f"\n{'':21s}  · {n}" for n in v.notes)))
    worst = outcome.worst
    if a.html:
        from lattix.report_html import write_validate_html

        write_validate_html(a.html, outcome.comparisons, title=", ".join(str(p) for p in decks.values()))
    if a.tol is not None and worst > a.tol:
        print(f"FAIL: max cumulative-map difference {worst:.3e} > tol {a.tol:.1e}")
        return 1
    return 0


def cmd_ui(a) -> int:
    from lattix.ui.server import Settings, serve, stable_token

    root = Path(a.root).expanduser().resolve() if a.root else Path.cwd()
    return serve(Settings(host=a.host, port=a.port, root=root, any_path=a.any_path, open_browser=not a.no_browser,
                          token=stable_token(rotate=a.new_token), plugins=not a.no_plugins),
                 check=a.check, deck=a.deck)


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


def cmd_survey(a) -> int:
    from lattix.formats import read
    from lattix.ir.frames import frame_survey, site_frame, survey_csv, survey_table

    lat, rep = read(a.src, a.from_fmt, **_kv(a.read_option))
    pose = (a.x0, a.y0, a.z0, a.theta0, a.phi0, a.psi0)
    frames = frame_survey(lat.flatten(), lat=lat, start=site_frame(*pose),
                          apply_shift=not a.no_shift, expand_children=a.children)
    rows = survey_table(frames, at=a.at)
    fmt = rep.source_format or a.from_fmt or lat.meta.get("source_format") or "?"
    comments = [f"lattix {__version__} survey of {a.src} ({fmt}): {len(rows)} rows",
                f"frames: {a.at}; misalignments {'ignored' if a.no_shift else 'applied to the body frame'}; "
                f"superposition children {'expanded' if a.children else 'not expanded'}",
                "start pose (MAD-X SURVEY x0 y0 z0 theta0 phi0 psi0; m, rad): " + " ".join(f"{v:.12g}" for v in pose),
                "units: m and rad; theta, phi, psi are MAD-X survey angles, theta continuous along the line"]
    if a.csv:
        Path(a.csv).write_text(survey_csv(rows, comments=comments), encoding="utf-8")
        print(f"{len(rows)} rows -> {a.csv}")
    elif a.json:
        doc = {"deck": str(a.src), "format": fmt, "at": a.at, "shift": not a.no_shift, "children": a.children,
               "start": dict(zip(("x0", "y0", "z0", "theta0", "phi0", "psi0"), pose, strict=True)),
               "columns": list(rows[0]) if rows else [], "rows": rows}
        Path(a.json).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"{len(rows)} rows -> {a.json}")
    else:
        for c in comments:
            print(f"# {c}")
        if rows:
            cols = list(rows[0])
            widths = [max(len(c), *(len(_cell(r[c])) for r in rows)) for c in cols]
            print("  ".join(c.rjust(w) for c, w in zip(cols, widths, strict=True)))
            for r in rows:
                print("  ".join(_cell(r[c]).rjust(w) for c, w in zip(cols, widths, strict=True)))
    if not rep.ok:
        print(rep.summary(), file=sys.stderr)
    return 0


def _cell(v) -> str:
    return f"{v:.9g}" if isinstance(v, float) else str(v)


def cmd_crossval(a) -> int:
    from lattix import crossval

    fmts = a.formats.split(",") if a.formats else None
    decks = [d for d in crossval.DECKS if not a.decks or any(x in d[0] for x in a.decks.split(","))]
    if not crossval.PUBLIC.is_dir() or not any((crossval.PUBLIC / d[0]).is_file() for d in decks):
        print(f"no public sample decks under {crossval.PUBLIC}: run from a checkout of the repository, or set "
              "LATTIX_PUBLIC_DECKS to its tests/data/public directory", file=sys.stderr)
        return 2
    results = crossval.run_matrix(decks, fmts, workdir=Path(a.workdir or tempfile.mkdtemp(prefix="lattix_crossval_")),
                                  engines=a.engines, only_src=a.src, only_dst=a.dst, derived=a.derived)
    print(crossval.to_markdown(results))
    print(crossval.summary(results), file=sys.stderr)
    if a.json:
        Path(a.json).write_text(crossval.to_json(results))
    if a.markdown:
        Path(a.markdown).write_text(crossval.to_markdown(results) + "\n\n" + crossval.summary(results) + "\n")
    bad = [r for r in results if r.error or r.ir_ok is False or r.fixed_ok is False or r.engine_ok is False]
    return 1 if bad else 0


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
    s.add_argument("--tier", choices=("exact", "equivalent", "lossy"), default=None,
                   help="the translation's fidelity tier the pairs are held to (default: report only)")
    s.add_argument("--codes", default=None,
                   help="comma list of the translation's ledger codes, CODE or CODE=count (FM_* selects the field-map "
                        "comparison rules; the counts feed the verdict notes)")
    _add_beam_args(s)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("ui", help="browser UI: translate a deck and compare the beam line before and after")
    s.add_argument("--host", default="127.0.0.1", help="loopback address to bind (127.0.0.1 or localhost)")
    s.add_argument("--port", type=int, default=0, help="TCP port (0 = any free port)")
    s.add_argument("--root", default=None, help="directory the file picker may browse (default: the current one)")
    s.add_argument("--any-path", action="store_true", help="allow absolute paths outside --root")
    s.add_argument("--no-browser", action="store_true", help="do not open the browser")
    s.add_argument("--deck", default=None, help="deck to load on start")
    s.add_argument("--check", action="store_true", help="start, self-test the API and exit")
    s.add_argument("--new-token", action="store_true",
                   help="issue a new access token (the link changes; the old one is kept in ~/.config/lattix/ui-token)")
    s.add_argument("--no-plugins", action="store_true", help="start without the installed workbench plugins")
    s.set_defaults(func=cmd_ui)
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

    s = sub.add_parser("survey", help="floor coordinates of every element (MAD-X survey frame): table, CSV or JSON")
    s.add_argument("src")
    s.add_argument("--from", dest="from_fmt", default=None)
    s.add_argument("--read-option", action="append", help="KEY=VALUE for the reader (e.g. species=h-)")
    s.add_argument("--at", default="exit", choices=["entrance", "centre", "exit", "body", "all"],
                   help="which frame of each element to tabulate (body = centre after the misalignment); "
                        "all gives every frame with the prefixes in_, c_, out_, body_")
    s.add_argument("--no-shift", action="store_true", help="ignore misalignments (the body frame equals the centre)")
    s.add_argument("--children", action="store_true", help="expand superposition clusters into their field maps")
    for name, what in (("x0", "X [m]"), ("y0", "Y [m]"), ("z0", "Z [m]"), ("theta0", "azimuth [rad]"),
                       ("phi0", "elevation [rad]"), ("psi0", "roll [rad]")):
        s.add_argument(f"--{name}", type=float, default=0.0, help=f"start pose {what}, as MAD-X SURVEY")
    s.add_argument("--csv", default=None, help="write the table as CSV here (12 significant digits)")
    s.add_argument("--json", default=None, help="write the table as JSON here")
    s.set_defaults(func=cmd_survey)

    s = sub.add_parser("crossval", help="cross-format battery: every format pair on every public deck")
    s.add_argument("--formats", default=None, help="comma list of formats (default: all)")
    s.add_argument("--decks", default=None, help="comma list of deck-name fragments (default: all)")
    s.add_argument("--src", default=None)
    s.add_argument("--dst", default=None)
    s.add_argument("--engines", action="store_true", help="also run the engine pair comparison")
    s.add_argument("--derived", action="store_true", help="also use decks written by lattix itself as sources")
    s.add_argument("--workdir", default=None)
    s.add_argument("--json", default=None)
    s.add_argument("--markdown", default=None)
    s.set_defaults(func=cmd_crossval)
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
