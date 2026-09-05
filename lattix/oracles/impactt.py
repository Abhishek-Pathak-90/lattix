"""IMPACT-T oracle: per-element 6×6 maps and reference energies from ``ImpactTexe``.

How the numbers are obtained (everything below MEASURED on this machine 2026-09-05 with
conda-forge ``impact-t`` 3.1.5 (``nompi``), osx-arm64, against the analytic drift/quadrupole/
solenoid maps and against cpymad — see ``docs/oracles.md``, "Phase 5.5"):

* IMPACT-T integrates in **time** (Boris/leapfrog, ``dt`` from the header); its phase-space
  dumps (``-2`` cards, ``fort.<mapstp>``) hold every particle *at one instant*: ``x [m], γβx,
  y [m], γβy, z [m], γβz, q/m, weight, id``.  The adapter reads a 13-particle probe
  (``flagdist = 16``, ``partcl.data``: the reference plus ``±h`` in each coordinate), drifts
  every particle through free space to the reference's plane (``x += x'·Δz``, ``t += Δz/v_z``)
  and forms the central-difference Jacobian in the common basis ``(x, px/p0, y, py/p0,
  z ahead-positive = −βc·Δt, δ = Δp/p0)``; ``R_elem[i] = J_i · J_{i−1}⁻¹``.
* A hard-edge element is only exact when its edges fall on integration steps, so the adapter
  gives every element its own time step: ``dt_k = L_k/(N_k·β_k·c)`` with ``N_k`` steps of
  ≈ ``dt`` (1 ps), switched by a ``-4`` card.  Triggers fire on the centroid position *after*
  a push: a ``-2``/``-4``/``-1`` card placed at ``z_b + dzz/2`` acts exactly at the boundary
  ``z_b`` (a ``-2`` at ``z_b − dzz/2`` dumps one step early).  With that, a 0.3 m quadrupole
  of 5 T/m at 2.1 MeV matches the analytic thick-quad map to 4e-8 and a drift to 5e-13.
* Dipoles: IMPACT-T re-bases its frame at the reference's exit, one integration step past the
  face (deterministic when the arc is ``N + 1e-4`` steps long), so every later ``zedge`` is
  shifted by that step; the post-bend dump is taken one step into the following drift and
  drifted back to the face.  IMPACT-T bends the *whole bunch* by the reference's angle: its
  dipole map has ``R11 = cos θ, R12 = ρ sin θ, R16 = ρ(1 − cos θ)`` but ``R21 = 0, R26 = 0``
  and no pole-face focusing — a model difference the battery treats as report-only
  (``meta["bends"]`` lists them).
* Cavities (type 104) accelerate the reference; the step count uses the mean β of the element
  from the deck's own profile (:func:`lattix.formats.impactt.rfprofile.gain_from_profile`).
* IMPACT-T caps ``-2`` dumps and ``-4`` changes at 100 per run: longer decks run in chunks,
  the next chunk starting from the raw dump of the previous one (same time origin shift
  applied to every ``theta0``).

Binary lookup: ``$IMPACTT_EXE``, then ``$LATTIX_ENV_BIN/ImpactTexe``, then
``~/anaconda3/envs/lattix/bin/ImpactTexe``, then ``ImpactTexe`` on PATH.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from lattix.formats.impactt.reader import Card, Header, card_span, parse_deck, read_rfdata, rfdata_name
from lattix.formats.impactt.rfprofile import C_LIGHT, gain_from_profile
from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register

_DEFAULT_ENV_BIN = Path.home() / "anaconda3" / "envs" / "lattix" / "bin"
#: units IMPACT-T writes itself (fort.18, 24-32, 40-43, 50, 60, 70 …) — kept clear of the dumps
_RESERVED_UNITS = frozenset({5, 6, 8, 11, 12, 13, 14, 15, 18, *range(24, 33), *range(40, 44), 50, 60, 70})
FIRST_DUMP_UNIT = 1001
DEFAULT_DT_S = 1e-12
#: probe amplitudes in the common basis: x [m], x', y [m], y', z [m], δ
PROBE_EPS = (1e-5, 1e-5, 1e-5, 1e-5, 3e-5, 3e-5)   # z and δ wider: position roundoff over ~5e4 steps
MAX_CONTROLS = 186           # IMPACT-T's Nbpmmax = 200 (NumConst.f90): every negative-type card of a run
CARD_MARGIN_M = 2.0          # physical cards kept beyond a chunk's end (never reached: the -99 stops first)
BEND_CLEARANCE_M = 0.01      # field-free distance kept between a dipole's faces and the controls around it: IMPACT-T
                             # triggers on the bunch centroid, which runaway probes (rings under its dipole model)
                             # drag behind the reference, and a trigger passed inside the dipole loop blocks every
                             # later one of its kind (the lists are walked in order)
#: run-control cards the adapter strips (it plants its own)
_STRIPPED = frozenset({-2, -3, -4, -9, -99})
#: value indices holding a position for the run-control types (0 = zedge)
_POSITION_VALUES: dict[int, tuple[int, ...]] = {-1: (0, 1), -11: (0, 1), -5: (0, 2), -6: (0, 2, 3), -8: (0, 2),
                                                -12: (0,), -13: (0, 1), -16: (0,), -17: (0,), -18: (0,)}
_RF_TYPES = frozenset({101, 102, 103, 104, 105, 110, 111, 112, 113})


def find_impactt() -> Path | None:
    """``$IMPACTT_EXE`` → ``$LATTIX_ENV_BIN`` → ``~/anaconda3/envs/lattix/bin`` → PATH."""
    v = os.environ.get("IMPACTT_EXE")
    cands: list[Path] = []
    if v:
        cands.append(Path(v).expanduser())
    v = os.environ.get("LATTIX_ENV_BIN")
    if v:
        cands.append(Path(v).expanduser() / "ImpactTexe")
    cands.append(_DEFAULT_ENV_BIN / "ImpactTexe")
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    w = shutil.which("ImpactTexe")
    return Path(w) if w else None


def _g17(x: float) -> str:
    return f"{float(x):.17g}"


@dataclass
class _Elem:
    """One reported element: a card with a length, a gap between cards, or a zero-length card."""

    name: str
    kind: str
    s_in: float                   # physical (arc) position
    length: float
    card: Card | None = None
    ke_in: float = 0.0
    ke_out: float = 0.0
    is_bend: bool = False
    n_steps: int = 0
    dt: float = 0.0
    dump: bool = True             # a plane is measured at the exit
    unit: int = 0
    plane_shift: float = 0.0      # drift from the dump plane to the boundary (+ forward, − back)
    delta: float = 0.0            # frame re-basing at a dipole exit (label = physical − Σ delta)
    pre: float = 0.0              # field-free clearance before the element (own steps): dipole probe lead, table ramp
    post: float = 0.0             # field-free clearance after it (own steps): the dipole's overshoot step, table ramp
    pad: float = 0.0              # how far the element's field table extends beyond its nominal ends
    skip: bool = False            # shorter than the overshoot of the dipole before it: no steps or controls of its own
    pre_ok: bool = True           # the element before this dipole/table can hold the whole clearance


@register
class ImpacttOracle:
    """IMPACT-T adapter (``ImpactTexe``)."""

    name = "impactt"
    formats = ("impactt",)
    timeout_s: ClassVar[float] = 1800.0

    def __init__(self, exe: Path | str | None = None):
        self._exe = Path(exe) if exe else None

    def exe(self) -> Path:
        p = self._exe or find_impactt()
        if p is None:
            raise RuntimeError("ImpactTexe not found: set IMPACTT_EXE or LATTIX_ENV_BIN, or install it "
                               f"(conda install -c conda-forge impact-t; {_DEFAULT_ENV_BIN}/ImpactTexe, PATH)")
        return p

    def available(self) -> tuple[bool, str]:
        try:
            p = self.exe()
        except RuntimeError as e:
            return False, str(e)
        return True, str(p)

    # ------------------------------------------------------------------ run
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None, probe: Probe | None = None,
            workdir: Path | None = None, dt_s: float = DEFAULT_DT_S, eps=PROBE_EPS,
            keep_workdir: bool = True) -> OracleResult:
        """Run IMPACT-T on *deck* (an ``ImpactT.in`` with its ``rfdataN`` files next to it)."""
        if fmt not in (None, "impactt"):
            raise ValueError(f"IMPACT-T reads ImpactT.in decks only, not {fmt!r}")
        exe = self.exe()
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        wd = Path(workdir).resolve() if workdir else Path(tempfile.mkdtemp(prefix="lattix_impactt_"))
        wd.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        header, cards, _ = parse_deck(deck.read_text(errors="replace"))
        cards = [c for c in cards if c.itype not in _STRIPPED]
        if not any(c.length > 0 or c.itype in (-1, -11) for c in cards):
            raise RuntimeError(f"{deck} has no beam-line elements")
        if beam is not None:
            header.kinetic_energy_eV = beam.kinetic_energy_eV
            header.mass_eV = beam.mass_eV
            header.charge = float(beam.charge)
            if beam.frequency_Hz:
                header.frequency_Hz = beam.frequency_Hz
            warnings.append(f"the BeamSpec overrode the deck's own reference particle ({beam.species}, "
                            f"{beam.kinetic_energy_eV:.6g} eV)")
        if header.current_A:
            warnings.append(f"the deck's {header.current_A:g} A beam current was set to 0: space charge makes the "
                            "transfer map amplitude-dependent")
        for f in sorted(list(deck.parent.glob("rfdata*")) + list(deck.parent.glob("*.T7"))):
            if f.is_file():
                shutil.copy2(f, wd / f.name)
        profiles = self._profiles(cards, deck.parent, warnings)
        elems = self._schedule(cards, header, profiles, dt_s, warnings, eps)
        planes = [e for e in elems if e.dump]
        units = self._units(len(planes))
        for e, u in zip(planes, units, strict=True):
            e.unit = u
        # chunks: the reference starts each one where the previous one dumped
        chunks = self._chunks(elems)
        mass, q = header.mass_eV, header.charge
        g0 = 1.0 + header.kinetic_energy_eV / mass
        bg0 = math.sqrt(g0 * g0 - 1.0)
        beta0 = bg0 / g0
        # lattix's time origin is the deck's nominal start (the walk's s = 0); the first controls may
        # sit before it (a padded table, a dipole's probe lead), so the bunch starts further back and
        # every driven phase is delayed by that time of flight
        z_nom = min(card_span(c)[0] for c in cards)
        stepped0 = [e for e in elems if e.length > 0]
        dz0 = (stepped0[0].dt * beta0 * C_LIGHT) if stepped0 else beta0 * C_LIGHT * dt_s
        first_ctrl = min([z_nom] + [e.s_in - e.pre for e in stepped0[:1]])
        z_start = min(z_nom, first_ctrl) - 2.0 * dz0
        delta_t = (z_nom - z_start) / (beta0 * C_LIGHT)
        extra = None if probe is None else np.asarray(probe.coords, dtype=float)
        pin = _probe_particles(eps, z_start, bg0, q / mass, extra)
        (wd / "partcl.data").write_text(pin)
        t_elapsed = z_nom / (beta0 * C_LIGHT)
        delta_cum = 0.0
        dumps: dict[int, np.ndarray] = {}
        missing: list[str] = []
        ci = 0
        while ci < len(chunks):
            a, b = chunks[ci]
            cwd = wd / f"chunk{ci}" if len(chunks) > 1 else wd
            cwd.mkdir(parents=True, exist_ok=True)
            if cwd != wd:
                for f in list(wd.glob("rfdata*")) + list(wd.glob("*.T7")):
                    shutil.copy2(f, cwd / f.name)
                shutil.copy2(wd / "partcl.data", cwd / "partcl.data")
            part = elems[a:b]
            n_part = 13 + (0 if extra is None else len(extra))
            t0 = t_elapsed - (delta_t if ci == 0 else 0.0)
            text, _ = self._instrumented_deck(header, cards, elems, part, t0, delta_cum, n_part,
                                                      extra_steps=int(math.ceil(delta_t / dt_s)))
            (cwd / "ImpactT.in").write_text(text)
            log = self._execute(exe, cwd)
            warnings += _log_warnings(log)
            got: list[_Elem] = []
            for e in part:
                if not e.dump:
                    continue
                try:
                    dumps[e.unit] = _read_dump(cwd / f"fort.{e.unit}", n_part)
                    got.append(e)
                except RuntimeError:
                    # IMPACT-T triggers on the bunch centroid, which a runaway probe (rings under its dipole
                    # model) can drag past a plane inside a dipole loop; a skipped trigger blocks every later
                    # one of its kind, so the chunk is cut at the last dump that was written and resumed there
                    missing.append(e.name)
                    e.dump = False
            if b < len(elems) and (not got or got[-1] is not part[-1]):
                if not got:
                    raise RuntimeError(f"IMPACT-T wrote none of the dumps of chunk {ci} ({cwd}); see run.log")
                b = elems.index(got[-1]) + 1
                part = elems[a:b]
                chunks = chunks[:ci] + [(a, b)] + [(b + x, b + y) for x, y in self._chunks(elems[b:])]
                warnings.append(f"chunk {ci} resumed at {got[-1].name}: its planned boundary dump was not written")
            t_elapsed += sum(e.n_steps * e.dt for e in part)
            delta_cum = sum(e.delta for e in elems[:b] if e.is_bend)
            if b < len(elems):
                raw = dumps[part[-1].unit]
                (wd / "partcl.data").write_text(_raw_particles(raw))
            ci += 1
        # maps: the first plane is the nominal start, reached from the launch point through free space
        cm0, _ = _to_common(np.array(_probe_rows(eps, z_start, bg0, None), dtype=float), z_nom)
        J_prev = _jacobian(cm0, eps)
        names, lengths, s_out, R, ke_in, ke_out, kinds = [], [], [], [], [], [], []
        block: list[_Elem] = []
        ke_run = header.kinetic_energy_eV
        for e in elems:
            if e.length <= 0 and not e.dump:
                # a zero-length card (kick, collimator, marker): identity map, no plane of its own
                names.append(e.name)
                kinds.append(e.kind)
                lengths.append(0.0)
                s_out.append(e.s_in - z_nom)                 # positions relative to the deck's nominal start
                R.append(np.eye(6))
                ke_in.append(ke_run)
                ke_out.append(ke_run)
                continue
            block.append(e)
            if not e.dump:
                continue
            pts = dumps[e.unit]
            zref = pts[0, 4]
            cm, gam = _to_common(pts[:13], zref + e.plane_shift)
            J = _jacobian(cm, eps)
            r = J @ np.linalg.inv(J_prev)
            J_prev = J
            ke_new = (gam - 1.0) * mass
            names.append("+".join(x.name for x in block))
            kinds.append("+".join(x.kind for x in block))
            lengths.append(sum(x.length for x in block))
            s_out.append(block[-1].s_in + block[-1].length - z_nom)
            R.append(r)
            ke_in.append(ke_run)
            ke_out.append(ke_new)
            ke_run = ke_new
            block = []
        probe_out = None
        planes = [e for e in elems if e.dump]
        if extra is not None and dumps and planes:
            last = dumps[planes[-1].unit]
            cm, _ = _to_common(last, last[0, 4] + planes[-1].plane_shift)
            probe_out = cm[13:]
        if missing:
            warnings.append(f"{len(missing)} dump(s) were not written by IMPACT-T (its triggers follow the bunch "
                            f"centroid, which a runaway probe dragged past the plane inside a dipole loop); the maps "
                            f"there span two elements: {', '.join(missing[:6])}")
        bends = [e.name for e in elems if e.is_bend]
        merged = [n for n in names if "+" in n]
        if merged:
            warnings.append(f"{len(merged)} block(s) reported as one element (no field-free plane between a dipole "
                            f"and its neighbour): {', '.join(merged[:6])}")
        meta = {
            "workdir": str(wd), "deck": str(wd / "ImpactT.in"), "impactt": str(exe), "dt_s": dt_s,
            "steps": int(sum(e.n_steps for e in elems)), "chunks": len(chunks), "dump_units": units,
            "probe_eps": list(eps), "bends": bends, "kinds": kinds, "missing_dumps": missing,
            "start_offset_m": z_nom - z_start,
            "native_basis": "(x [m], gamma*beta_x, y [m], gamma*beta_y, z [m], gamma*beta_z) at one time; "
                            "converted to the common basis by drifting to the reference plane",
            "twiss": "not available: IMPACT-T computes no Twiss/dispersion/floor output",
        }
        if bends:
            meta["bend_model"] = ("IMPACT-T bends the whole bunch by the reference angle (R21 = R26 = 0, no "
                                  "pole-face focusing): its dipole maps are not the hard-edge ones")
        n = len(names)
        return OracleResult(engine=self.name, basis=Basis.COMMON, names=names, length=np.array(lengths),
                            s_out=np.array(s_out), R_elem=np.array(R).reshape(n, 6, 6),
                            ref_kinetic_eV_in=np.array(ke_in), ref_kinetic_eV_out=np.array(ke_out),
                            mass_eV=mass, charge=int(round(q)),
                            rf_frequency_Hz=np.full(n, float(header.frequency_Hz or 0.0)),
                            probe_out=probe_out, warnings=warnings, meta=meta)

    # ------------------------------------------------------------------ scheduling
    @staticmethod
    def _profiles(cards: list[Card], src: Path, warnings: list[str]) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for c in cards:
            if c.itype == 104 and c.v(5) > 0:
                fid = int(round(c.v(5)))
                p = src / rfdata_name(fid)
                if p.is_file():
                    out[fid] = read_rfdata(p)
                else:
                    warnings.append(f"{p.name} missing: the type-104 card at line {c.line} gets no energy gain")
        return out

    def _schedule(self, cards: list[Card], h: Header, profiles: dict[int, list[float]], dt_s: float,
                  warnings: list[str], eps=PROBE_EPS) -> list[_Elem]:
        self._eps = eps
        """Elements in order with their reference energies and time steps."""
        mass = h.mass_eV
        spans = {id(c): ((c.zedge, c.length) if c.itype in _RF_TYPES else card_span(c)) for c in cards}
        thick = sorted((c for c in cards if spans[id(c)][1] > 0), key=lambda c: (spans[id(c)][0], c.line))
        thin = sorted((c for c in cards if spans[id(c)][1] <= 0), key=lambda c: (c.zedge, c.line))
        elems: list[_Elem] = []
        s = min(spans[id(c)][0] for c in cards)              # the nominal start (a padded table begins earlier)
        ke = h.kinetic_energy_eV
        t = s / (_beta(ke, mass) * C_LIGHT)                  # the deck's clock: the reference at z = 0 at t = 0
        n_gap = 0
        ti = 0
        for c in thick:
            z_in, length = spans[id(c)]
            while ti < len(thin) and thin[ti].zedge <= z_in + 1e-12:
                tc = thin[ti]
                elems.append(_Elem(f"{_kind(tc)}_{tc.line}", _kind(tc), tc.zedge, 0.0, tc, ke, ke, dump=False))
                ti += 1
            if z_in > s + 1e-9:
                n_gap += 1
                elems.append(_Elem(f"gap_{n_gap}", "drift", s, z_in - s, None, ke, ke))
                t += (z_in - s) / (_beta(ke, mass) * C_LIGHT)
                s = z_in
            elif z_in < s - 1e-9:
                warnings.append(f"card at line {c.line} overlaps the previous element by {s - z_in:.3g} m; "
                                "the adapter cannot place its plane")
            kind = _kind(c)
            dE = 0.0
            if c.itype == 104 and c.v(3) and int(round(c.v(5))) in profiles:
                dE, _, _ = gain_from_profile(profiles[int(round(c.v(5)))], length, c.v(2), c.v(4), c.v(3), t,
                                               ke, mass, float(h.charge))
            e = _Elem(f"{kind}_{c.line}", kind, z_in, length, c, ke, ke + dE, is_bend=(c.itype == 4))
            if "pad" in c.tag and c.itype == 3:
                try:
                    e.pad = float(c.tag["pad"])
                except ValueError:
                    pass
            elems.append(e)
            ke += dE
            s = z_in + length
            t += length / (0.5 * (_beta(e.ke_in, mass) + _beta(e.ke_out, mass)) * C_LIGHT)
        while ti < len(thin):
            tc = thin[ti]
            elems.append(_Elem(f"{_kind(tc)}_{tc.line}", _kind(tc), tc.zedge, 0.0, tc, ke, ke, dump=False))
            ti += 1
        # time steps and dump planes; dipoles first (an element before a dipole ends with the
        # dipole's own step, started early enough that the leading probes are still outside it)
        stepped = [e for e in elems if e.length > 0]
        # first the elements whose own step is fixed by their length: dipoles (arc = N + 1e-4 steps,
        # the exit detected one step past the face) and padded solenoid tables (N steps over the
        # nominal length, plus field-free clearances of their own step before and after)
        for e in stepped:
            beta = 0.5 * (_beta(e.ke_in, mass) + _beta(e.ke_out, mass))
            if e.is_bend:
                n = max(1, math.ceil(e.length / (beta * C_LIGHT * dt_s)))
                e.dt = e.length / ((n + 1e-4) * beta * C_LIGHT)
                dzb = e.dt * beta * C_LIGHT
                extra = math.ceil(BEND_CLEARANCE_M / dzb)
                e.n_steps = n + 1 + extra
                e.delta = (n + 1) * dzb - e.length
                e.post = (1 + extra) * dzb                      # the post-exit pushes keep this dt
            elif e.pad > 0:
                # a table interval (pad = 1.5 h for lattix's straddling edges) needs ≥ 10 steps: at
                # h/5 the sampled edge kick is off by 5e-5, at h/10 the map is within 1e-6 (MEASURED)
                n = max(1, math.ceil(e.length / (beta * C_LIGHT * dt_s)), math.ceil(15.0 * e.length / e.pad))
                e.dt = e.length / (n * beta * C_LIGHT)
                dz = e.dt * beta * C_LIGHT
                e.pre = e.post = math.ceil(e.pad / dz - 1e-9) * dz
                e.n_steps = n + 2 * int(round(e.pre / dz))
        carry = 0.0                                          # clearance still owed by a dipole or table before
        for i, e in enumerate(stepped):
            beta = 0.5 * (_beta(e.ke_in, mass) + _beta(e.ke_out, mass))
            nxt = stepped[i + 1] if i + 1 < len(stepped) else None
            if e.is_bend or e.pad > 0:
                carry = e.delta + e.post
                continue
            # what the neighbours take: their clearances run on their own steps
            lead, carry = carry, 0.0
            pre = 0.0
            if nxt is not None and nxt.is_bend:
                # the IMPACT-T bend loop (no triggers inside) starts when the *first* particle
                # reaches the dipole: switch to the dipole's step and dump this far before the face
                dzb = nxt.dt * _beta(nxt.ke_in, mass) * C_LIGHT
                g = 1.0 + e.ke_out / mass
                ahead = self._eps[4] + self._eps[5] * (e.s_in + e.length) / (g * g) + 2.0 * dzb
                pre = max((math.ceil(ahead / dzb) + 1) * dzb, math.ceil(BEND_CLEARANCE_M / dzb) * dzb)
                nxt.pre_ok = pre + 2.0 * dzb <= e.length - lead
                pre = min(pre, max(e.length - lead - 1e-9, 0.0))
                nxt.pre = pre
            elif nxt is not None and nxt.pad > 0:
                nxt.pre_ok = nxt.pre <= e.length - lead
                pre = min(nxt.pre, max(e.length - lead - 1e-9, 0.0))
                nxt.pre = pre
            avail = e.length - lead - pre
            if avail <= 1e-9:
                # shorter than the overshoot of the dipole before it (PSB: a 3 µm drift after a bend):
                # no steps or controls of its own, the clearance carries on to the next element
                e.n_steps, e.dt, e.skip = 0, dt_s, True
                carry = max(lead - e.length, 0.0)
                continue
            n = max(1, math.ceil(avail / (beta * C_LIGHT * dt_s)))
            e.dt = avail / (n * beta * C_LIGHT)
            e.n_steps = n
        # where can the exit plane be measured?  A clearance needs free space in the neighbour.
        live = [e for e in stepped if not e.skip]           # the deck builder sees the same neighbours
        for i, e in enumerate(live):
            nxt = live[i + 1] if i + 1 < len(live) else None
            if e.is_bend or e.pad > 0:
                need = e.delta + e.post
                e.dump = nxt is None or (nxt.kind == "drift" and nxt.length > 3.0 * need + 1e-12)
            elif nxt is not None and (nxt.is_bend or nxt.pad > 0):
                # a dump inside the dipole loop is skipped and blocks every later dump (IMPACT-T walks
                # its lists in order): none unless the drift holds the whole clearance
                e.dump = e.kind == "drift" and nxt.pre_ok
            else:
                e.dump = True
        for e in stepped:
            if e.skip:
                e.dump = False
        if stepped and not stepped[-1].dump:
            stepped[-1].dump = True
        return elems

    @staticmethod
    def _units(n: int) -> list[int]:
        units, u = [], FIRST_DUMP_UNIT
        while len(units) < n:
            if u not in _RESERVED_UNITS:
                units.append(u)
            u += 1
        return units

    @staticmethod
    def _cost(e: _Elem) -> int:
        """BPM-type cards an element needs in a run: its dump, its -4 step change, and the deck's own
        -1/-11/… card when it is one (IMPACT-T keeps all of those in one array of Nbpmmax)."""
        return int(e.dump) + int(e.length > 0) + int(e.card is not None and e.card.itype < 0)

    @classmethod
    def _chunks(cls, elems: list[_Elem]) -> list[tuple[int, int]]:
        """Index ranges each costing at most ``MAX_CONTROLS`` BPM-type cards, ending at a standard plane."""
        out: list[tuple[int, int]] = []
        start, cost = 0, 0
        last_ok = None
        for i, e in enumerate(elems):
            cost += cls._cost(e)
            if e.dump:
                nxt = elems[i + 1] if i + 1 < len(elems) else None
                standard = not e.is_bend and not (nxt is not None and nxt.is_bend)
                if standard:
                    last_ok = i + 1
            if cost >= MAX_CONTROLS and last_ok is not None and last_ok > start:
                out.append((start, last_ok))
                cost = sum(cls._cost(x) for x in elems[last_ok:i + 1])
                start, last_ok = last_ok, None
        out.append((start, len(elems)))
        return [(a, b) for a, b in out if b > a]

    def _instrumented_deck(self, h: Header, cards: list[Card], elems: list[_Elem], part: list[_Elem],
                           t0: float, delta_cum: float, n_particles: int,
                           extra_steps: int = 0) -> tuple[str, float]:
        """The deck for one chunk: the physical cards (labels shifted by the dipole re-basing),
        a ``-2`` dump and a ``-4`` step change per stepped element, and a ``-99`` at the end."""
        stepped_all = [e for e in elems if e.length > 0 and not e.skip]
        first = next((e for e in part if e.length > 0 and not e.skip), None)
        dt0 = first.dt if first is not None else h.dt_s
        n_steps = sum(e.n_steps for e in part) + 2000 + extra_steps
        end_s = part[-1].s_in + part[-1].length
        lines = ["! instrumented by lattix.oracles.impactt — do not edit",
                 "! flagdist 16 reads partcl.data (13-particle probe); -4 cards give every element a time step",
                 "1 1", f"{_g17(dt0)} {n_steps} 1", f"6 {n_particles} 1 {h.flagerr} 1 0 {_g17(h.zimage)}",
                 f"{h.nx} {h.ny} {h.nz} {h.flagbc} {_g17(h.xrad)} {_g17(h.yrad)} {_g17(h.perdlen)}",
                 "16 0 0 -1 1e-12", "0 0 0 1 1 0 0", "0 0 0 1 1 0 0", "0 0 0 1 1 0 0",
                 f"0 {_g17(h.kinetic_energy_eV)} {_g17(h.mass_eV)} {_g17(h.charge)} {_g17(h.frequency_Hz)} 0"]
        part_ids = {id(e) for e in part}
        start_s = part[0].s_in
        delta = delta_cum
        # every card of the deck from the chunk start on, with the label shift of the dipoles before it
        deltas_before: dict[int, float] = {}
        d = delta_cum
        for e in elems:
            deltas_before[id(e)] = d
            if e.is_bend and id(e) in part_ids:
                d += e.delta
        shifts = {id(e): deltas_before[id(e)] for e in elems}
        for e in elems:
            c = e.card
            if c is None or e.s_in + e.length < start_s - 1e-9 or e.s_in > end_s + CARD_MARGIN_M:
                continue                          # only what this chunk's bunch can reach (Ndriftmax, Nquadmax …)
            vals = list(c.values)
            dz = -shifts[id(e)]
            for k in _POSITION_VALUES.get(c.itype, (0,)):
                if k < len(vals):
                    vals[k] += dz
            if c.itype in _RF_TYPES and len(vals) >= 4 and t0:
                vals[3] = (vals[3] + 360.0 * c.v(3) * t0 + 180.0) % 360.0 - 180.0
            lines.append(f"{_g17(c.length)} {c.nseg} {c.mapstp} {c.itype} " + " ".join(_g17(v) for v in vals) + " /")
        # controls
        stepped = [e for e in part if e.length > 0 and not e.skip]
        dump_at: list[float] = []
        for e in stepped:
            j = stepped_all.index(e)
            nxt = stepped_all[j + 1] if j + 1 < len(stepped_all) else None
            label_end = e.s_in + e.length - shifts[id(e)] - (e.delta if e.is_bend else 0.0)
            # every control fires on the integration step nearest to its position (MEASURED: a
            # trigger at z_b ± 0.25 dzz acts at the landing step, at ± 0.75 dzz one step off), so
            # the cards sit exactly on the boundary; the -4 takes effect from the next push
            if e.is_bend or e.pad > 0:
                # the plane is ``post`` beyond the nominal exit (a dipole: the frame is re-based at the
                # exit iteration and the next iteration starts one step further), drifted back
                exit_label = e.s_in + e.length - shifts[id(e)]
                where = exit_label + e.post
                if nxt is not None:
                    lines.append(f"0 1 1 -4 {_g17(where)} 1 {_g17(where)} {_g17(nxt.dt)} /")
                if e.dump:
                    lines.append(f"0 1 {e.unit} -2 {_g17(where)} 1 {_g17(where)} /")
                    e.plane_shift = -(e.delta + e.post)
                    dump_at.append(where)
                continue
            if nxt is not None and (nxt.is_bend or nxt.pad > 0):
                # the next element's step starts ``pre`` before its nominal entrance (the -4 there);
                # a dipole's dump one dipole step later still sees a field-free bunch, a table's
                # dump right at the switch (its ramp starts within ``pre``); both drifted forward
                pre_label = label_end - nxt.pre
                dzn = nxt.dt * _beta(nxt.ke_in, h.mass_eV) * C_LIGHT
                lines.append(f"0 1 1 -4 {_g17(pre_label)} 1 {_g17(pre_label)} {_g17(nxt.dt)} /")
                if e.dump:
                    at = pre_label + dzn if nxt.is_bend else pre_label
                    lines.append(f"0 1 {e.unit} -2 {_g17(at)} 1 {_g17(at)} /")
                    e.plane_shift = label_end - at
                    dump_at.append(at)
                continue
            if e.dump:
                lines.append(f"0 1 {e.unit} -2 {_g17(label_end)} 1 {_g17(label_end)} /")
                e.plane_shift = 0.0
                dump_at.append(label_end)
            if nxt is not None and id(nxt) in part_ids:
                lines.append(f"0 1 1 -4 {_g17(label_end)} 1 {_g17(label_end)} {_g17(nxt.dt)} /")
        last = stepped[-1] if stepped else part[-1]
        end_label = last.s_in + last.length - shifts[id(last)] - (last.delta if last.is_bend else 0.0)
        dzz_last = (last.dt * _beta(last.ke_out, h.mass_eV) * C_LIGHT) if last.length > 0 else 0.0
        stop = max([end_label + last.post, *dump_at]) + 0.5 * dzz_last     # never before the last dump
        lines.append(f"0 1 1 -99 {_g17(end_label + 1.0)} 1 {_g17(stop)} /")
        for e in part:
            if e.is_bend:
                delta += e.delta
        return "\n".join(lines) + "\n", delta

    def _execute(self, exe: Path, wd: Path) -> str:
        try:
            r = subprocess.run([str(exe)], cwd=wd, capture_output=True, text=True, timeout=self.timeout_s,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ImpactTexe timed out after {self.timeout_s} s in {wd}") from e
        log = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
        (wd / "run.log").write_text(log)
        if r.returncode != 0:
            tail = "\n".join(log.strip().splitlines()[-30:])
            raise RuntimeError(f"ImpactTexe failed (exit status {r.returncode}) in {wd}:\n{tail}")
        return log


# ---------------------------------------------------------------------------
def _kind(c: Card) -> str:
    return {0: "drift", 1: "quad", 2: "constfoc", 3: "solenoid", 4: "bend", 5: "multipole", -1: "kick",
            -11: "collimator"}.get(c.itype, "cavity" if c.itype in _RF_TYPES else f"type{c.itype}")


def _beta(ke: float, mass: float) -> float:
    g = 1.0 + ke / mass
    return math.sqrt(max(1.0 - 1.0 / (g * g), 0.0))


def _probe_rows(eps, z0: float, bg: float, extra: np.ndarray | None) -> list[list[float]]:
    """The reference, the 12 symmetric probes and any extra particles in IMPACT-T's native columns."""
    rows = [[0.0, 0.0, 0.0, 0.0, z0, bg]]
    for i in range(6):
        for s in (+1.0, -1.0):
            r = [0.0, 0.0, 0.0, 0.0, z0, bg]
            if i == 0:
                r[0] = s * eps[0]
            elif i == 1:
                r[1] = s * eps[1] * bg
            elif i == 2:
                r[2] = s * eps[2]
            elif i == 3:
                r[3] = s * eps[3] * bg
            elif i == 4:
                r[4] = z0 + s * eps[4]
            else:
                r[5] = bg * (1.0 + s * eps[5])
            rows.append(r)
    if extra is not None:
        for c in extra:                                  # common-basis coordinates → native
            x, xp, y, yp, z, d = (float(v) for v in c)
            p = bg * (1.0 + d)
            pz = math.sqrt(max(p * p - (xp * bg) ** 2 - (yp * bg) ** 2, 0.0))
            rows.append([x, xp * bg, y, yp * bg, z0 + z, pz])
    return rows


def _probe_particles(eps, z0: float, bg: float, qmcc: float, extra: np.ndarray | None) -> str:
    rows = _probe_rows(eps, z0, bg, extra)
    out = [str(len(rows))]
    for i, r in enumerate(rows):
        out.append(" ".join(_g17(v) for v in r) + f" {_g17(qmcc)} 0 {i + 1}")
    return "\n".join(out) + "\n"


def _raw_particles(raw: np.ndarray) -> str:
    out = [str(len(raw))]
    for row in raw:
        out.append(" ".join(_g17(v) for v in row[:8]) + f" {int(round(row[8]))}")
    return "\n".join(out) + "\n"


def _read_dump(path: Path, n_expected: int) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"IMPACT-T wrote no phase-space dump {path}; the run probably stopped early (see run.log)")
    a = np.atleast_2d(np.loadtxt(path))
    if a.size == 0 or a.shape[1] < 9:
        raise RuntimeError(f"{path}: expected 9 columns per particle, got {a.shape}")
    out = np.full((max(n_expected, int(a[:, 8].max())), 9), np.nan)
    for row in a:
        i = int(round(row[8])) - 1
        if 0 <= i < len(out):
            out[i] = row
    return out


def _to_common(parts: np.ndarray, z_plane: float) -> tuple[np.ndarray, float]:
    """Fixed-time dump → common-basis coordinates on the plane ``z_plane`` (free-space drift),
    relative to the reference (row 0); also the reference's γ."""
    x, px, y, py, z, pz = (parts[:, i] for i in range(6))
    ptot = np.sqrt(px * px + py * py + pz * pz)
    gam = np.sqrt(1.0 + ptot * ptot)
    dz = z_plane - z
    xs = x + px / pz * dz
    ys = y + py / pz * dz
    vz = C_LIGHT * pz / gam
    t = dz / vz
    t = t - t[0]
    p0 = ptot[0]
    b0 = ptot[0] / gam[0]
    cm = np.column_stack([xs - xs[0], px / p0 - px[0] / p0, ys - ys[0], py / p0 - py[0] / p0, -b0 * C_LIGHT * t,
                          ptot / p0 - 1.0])
    return cm, float(gam[0])


def _jacobian(cm: np.ndarray, eps) -> np.ndarray:
    J = np.zeros((6, 6))
    for i in range(6):
        J[:, i] = (cm[1 + 2 * i] - cm[2 + 2 * i]) / (2.0 * eps[i])
    return J


def _log_warnings(log: str) -> list[str]:
    out: list[str] = []
    for ln in log.splitlines():
        s = ln.strip()
        if ("out of the range" in s or "not implemented" in s or "maximum" in s or "wrong" in s) and s not in out:
            out.append(s)
    return out
