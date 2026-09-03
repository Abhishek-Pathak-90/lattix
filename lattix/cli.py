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
    for pc in compare_all(results):
        print(pc.row() + ("  " + "; ".join(pc.notes) if pc.notes else ""))
        worst = max(worst, pc.max_rcum_abs)
    if a.json:
        payload = {e: {"names": r.names, "s_out": r.s_out.tolist(),
                       "R_cum_common": r.to_common().R_cum.tolist(),
                       "ke_out": r.ref_kinetic_eV_out.tolist()} for e, r in results.items()}
        Path(a.json).write_text(json.dumps(payload) + "\n")
    if a.tol is not None and worst > a.tol:
        print(f"FAIL: max cumulative-map difference {worst:.3e} > tol {a.tol:.1e}")
        return 1
    return 0


def _not_yet(name):
    def run(a):
        print(f"{name}: arrives with the IR in Phase 1 (see PLAN.md)", file=sys.stderr)
        return 3

    return run


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
    s.add_argument("--workdir", default=None)
    s.add_argument("--json", default=None)
    _add_beam_args(s)
    s.set_defaults(func=cmd_validate)

    for name in ("convert", "inspect", "report"):
        s = sub.add_parser(name)
        s.add_argument("args", nargs="*")
        s.set_defaults(func=_not_yet(name))

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
