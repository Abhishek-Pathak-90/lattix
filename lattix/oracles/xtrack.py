"""xtrack oracle (xtrack 0.103, CPU context).

Per-element 6×6 maps are *measured* by central finite differences through each
element in isolation: 12 particles (reference ± h in each of the six native
coordinates, one ``Particles`` object per element) are tracked with
``line.track(p, ele_start=i, ele_stop=i + 1)`` — verified on 0.103.5 to track
exactly element ``i`` (a 1 m drift gives R12 = 1.0, the 0.3 m quad that follows
gives the thick-quad matrix) — and column ``k`` of ``R`` is
``(out⁺ − out⁻) / (in⁺ − in⁻)``.

The denominator is the *read-back* perturbation, not the nominal ``2h``:
``xt.Particles`` stores ``delta`` through an energy round trip that at
γ ≈ 1.002 (2.1 MeV protons) loses ~1e-14 absolute, so ``delta = 1e-7`` reads
back as ``9.99999707e-8`` (and ``rvv`` is consistent with the stored value).
With nominal denominators that alone gives ``R66 = 0.99999988`` and a 1.2e-7
error on R56; with read-back denominators and ``h = 1e-5`` — balancing that
roundoff against the O(h²) truncation of the exact kinematic terms — the drift
and quadrupole maps are exact to ≲1e-10 (measured on 0.103.5: R56 5.6e-11
relative, quad elements 2.9e-11 absolute, det − 1 ≲ 1e-15).

Native basis ``(x, px, y, py, zeta, delta)``: ``zeta = s − β0 c t`` (ahead
positive), ``delta = Δp/p0``.  xtrack keeps ``p0c`` constant through RF
(energy changes go into ``delta``), so ``ref_kinetic_eV_in == ref_kinetic_eV_out``
everywhere; compare with p0-following engines after damping normalisation.

Deck loading: MAD-X through cpymad (so BEAM handling is *the same code* as the
cpymad oracle: explicit BeamSpec overrides, otherwise the deck's BEAM), then
``xt.Line.from_madx_sequence(deferred_expressions=False)``; the sequence's
``$start``/``$end`` markers are dropped like the cpymad oracle does.  Without
cpymad, xtrack's own MAD-X parser (``xt.load``) is used — element names then
differ (``||drift_1`` instead of ``drift_0``) and the deck's BEAM is not
carried over, so a BeamSpec is required.  ``.json`` decks use
``xt.Line.from_json``.

Twiss and survey come from ``line.twiss(betx=…, alfx=…, …)`` (open line
propagated from the BeamSpec optics) and ``line.survey()``.  Both tables have
one row per element at its *entrance* plus a final ``_end_point`` row, so the
exit of element ``i`` is row ``i + 1`` (checked against the downstream
``s``; an s-based lookup is the fallback).  If ``line.twiss`` fails, β/α and
dispersion are propagated through the measured cumulative maps instead and a
warning says so.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from lattix.oracles.base import (
    SPECIES,
    Basis,
    BeamSpec,
    OracleResult,
    Probe,
    guess_format,
    register,
)

#: finite-difference half-step for all six native coordinates (see module doc)
FD_STEP = 1e-5
_COORDS = ("x", "px", "y", "py", "zeta", "delta")


def _stack(p) -> np.ndarray:
    """(N, 6) native coordinates of a Particles object."""
    return np.stack([np.asarray(getattr(p, k), dtype=float) for k in _COORDS], axis=1)


def fd_seed(h: float = FD_STEP) -> dict[str, np.ndarray]:
    """Coordinates of the 12 finite-difference particles: particle ``2k`` is the
    reference ``+h`` in coordinate ``k``, particle ``2k + 1`` is ``−h``."""
    seed = {}
    for k, name in enumerate(_COORDS):
        v = np.zeros(12)
        v[2 * k] = h
        v[2 * k + 1] = -h
        seed[name] = v
    return seed


def fd_matrix(ins: np.ndarray, outs: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian from the 12-particle layout of :func:`fd_seed`,
    using the *actual* input perturbations as denominators."""
    R = np.empty((6, 6))
    for k in range(6):
        R[:, k] = (outs[2 * k] - outs[2 * k + 1]) / (ins[2 * k, k] - ins[2 * k + 1, k])
    return R


def _beam_from_particle_ref(pref) -> BeamSpec:
    if pref is None:
        raise ValueError("the line carries no reference particle (particle_ref is None) — "
                         "pass beam=BeamSpec(...)")
    mass_eV = float(np.atleast_1d(pref.mass0)[0])
    charge = int(round(float(np.atleast_1d(pref.q0)[0])))
    ke = float(np.atleast_1d(pref.kinetic_energy0)[0])
    species = None
    for nm, (mv, q) in SPECIES.items():
        if abs(mv - mass_eV) / mv < 1e-4 and q == charge:
            species = nm
            break
    spec = BeamSpec(species=species or "proton", kinetic_energy_eV=ke)
    if species is None:
        spec.__dict__["_mass_override"] = mass_eV
        spec.__dict__["_charge_override"] = charge
    return spec


def _exit_rows(table, n_all: int, s_down: np.ndarray, what: str) -> np.ndarray:
    """Row index of every element's *exit* in an xtrack twiss/survey table
    (rows = element entrances + ``_end_point``, so exit(i) = row i + 1)."""
    names = list(table.name)
    s_tab = np.asarray(table.s, dtype=float)
    if (len(names) == n_all + 1 and names[-1] == "_end_point"
            and np.allclose(s_tab[1:], s_down, atol=1e-9, rtol=0.0)):
        return np.arange(1, n_all + 1)
    rows = []
    for s in s_down:  # fallback: last row sitting at the exit position
        hits = np.nonzero(np.abs(s_tab - s) <= 1e-9)[0]
        if len(hits) == 0:
            raise ValueError(f"cannot align the xtrack {what} table: no row at s={s!r}")
        rows.append(int(hits[-1]))
    return np.asarray(rows, dtype=int)


def twiss_from_maps(R_cum: np.ndarray, beam: BeamSpec) -> tuple[dict, dict]:
    """Propagate the initial (β, α) and dispersion through cumulative maps in an
    (x, px, y, py, z, δ)-type basis — uncoupled, first order.  Used only when
    ``line.twiss`` fails."""
    n = R_cum.shape[0]
    twiss = {k: np.empty(n) for k in ("betx", "alfx", "bety", "alfy")}
    disp = {k: np.empty(n) for k in ("dx", "dpx", "dy", "dpy")}
    d0 = np.array([beam.dx, beam.dpx, beam.dy, beam.dpy])
    for i in range(n):
        R = R_cum[i]
        for plane, (b0, a0) in (("x", (beam.betx, beam.alfx)), ("y", (beam.bety, beam.alfy))):
            o = 0 if plane == "x" else 2
            m11, m12, m21, m22 = R[o, o], R[o, o + 1], R[o + 1, o], R[o + 1, o + 1]
            g0 = (1 + a0 * a0) / b0
            twiss["bet" + plane][i] = m11 * m11 * b0 - 2 * m11 * m12 * a0 + m12 * m12 * g0
            twiss["alf" + plane][i] = (-m11 * m21 * b0 + (m11 * m22 + m12 * m21) * a0
                                       - m12 * m22 * g0)
        d = R[:4, :4] @ d0 + R[:4, 5]
        disp["dx"][i], disp["dpx"][i], disp["dy"][i], disp["dpy"][i] = d
    return twiss, disp


@register
class XtrackOracle:
    name = "xtrack"
    formats = ("madx", "xtrack")

    def available(self) -> tuple[bool, str]:
        try:
            import xtrack as xt
        except Exception as e:  # noqa: BLE001
            return False, f"xtrack import failed: {e}"
        return True, f"xtrack {xt.__version__}"

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            sequence: str | None = None) -> OracleResult:
        import xtrack as xt

        deck = Path(deck).resolve()
        fmt = (fmt or guess_format(deck)).lower()
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_xtrack_"))
        wd.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        meta: dict = {
            "xtrack_version": xt.__version__, "workdir": str(wd), "fd_step": FD_STEP,
            "p0_model": "constant: xtrack keeps p0c fixed through RF "
                        "(energy changes go into delta)",
            "twiss_source": "line.twiss(betx=..., ...) rows i+1 (element exits)",
        }

        if fmt in ("xtrack", "json"):
            line = xt.Line.from_json(str(deck))
            beam_used = beam or _beam_from_particle_ref(line.particle_ref)
            meta["loader"] = "xt.Line.from_json"
        elif fmt == "madx":
            line, beam_used, meta["loader"], meta["sequence"] = self._load_madx(
                deck, wd, beam, sequence, warnings)
        else:
            raise ValueError(f"xtrack oracle cannot read format {fmt!r} (formats: {self.formats})")

        mass_eV = float(beam_used.__dict__.get("_mass_override") or beam_used.mass_eV)
        charge = int(beam_used.__dict__.get("_charge_override") or beam_used.charge)
        ke = float(beam_used.kinetic_energy_eV)
        line.particle_ref = xt.Particles(mass0=mass_eV, q0=charge, kinetic_energy0=ke)
        line.build_tracker()

        names_all = list(line.element_names)
        n_all = len(names_all)
        s_down = np.asarray(line.get_s_elements(mode="downstream"), dtype=float)
        s_up = np.asarray(line.get_s_elements(mode="upstream"), dtype=float)
        keep = [i for i, nm in enumerate(names_all)
                if not nm.lower().endswith(("$start", "$end"))]
        names = [names_all[i] for i in keep]
        n = len(names)
        s_out = s_down[keep]
        lengths = (s_down - s_up)[keep]

        # -- per-element maps by finite differences --------------------------
        seed = line.build_particles(**fd_seed(FD_STEP))
        ins = _stack(seed)
        R = np.full((n, 6, 6), np.nan)
        for j, i in enumerate(keep):
            p = seed.copy()
            line.track(p, ele_start=i, ele_stop=i + 1)
            p.sort(interleave_lost_particles=True)
            if np.any(np.asarray(p.state) <= 0):
                warnings.append(f"element {names_all[i]!r} (index {i}) lost finite-difference "
                                f"particles; its map is NaN")
                continue
            R[j] = fd_matrix(ins, _stack(p))

        # -- twiss / dispersion at exits -------------------------------------
        b = beam_used
        try:
            tw = line.twiss(betx=b.betx, alfx=b.alfx, bety=b.bety, alfy=b.alfy,
                            dx=b.dx, dpx=b.dpx, dy=b.dy, dpy=b.dpy)
            rows = _exit_rows(tw, n_all, s_down, "twiss")[keep]
            col = {k: np.asarray(tw[k], dtype=float)[rows]
                   for k in ("betx", "alfx", "bety", "alfy", "dx", "dpx", "dy", "dpy")}
            twiss = {k: col[k] for k in ("betx", "alfx", "bety", "alfy")}
            disp = {k: col[k] for k in ("dx", "dpx", "dy", "dpy")}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"line.twiss failed ({type(e).__name__}: {e}); Twiss propagated "
                            f"through the finite-difference cumulative maps instead")
            meta["twiss_source"] = "propagated through R_cum (line.twiss failed)"
            R_cum = np.empty_like(R)
            acc = np.eye(6)
            for j in range(n):
                acc = R[j] @ acc
                R_cum[j] = acc
            twiss, disp = twiss_from_maps(R_cum, b)

        # -- survey at exits ---------------------------------------------------
        sv = line.survey()
        rows = _exit_rows(sv, n_all, s_down, "survey")[keep]
        survey = np.stack([np.asarray(sv[k], dtype=float)[rows] for k in ("X", "Y", "Z", "theta")],
                          axis=1)

        # -- probe -------------------------------------------------------------
        probe_out = None
        if probe is not None:
            probe_out = self._track_probe(line, probe, ke, mass_eV)

        return OracleResult(
            engine=self.name, basis=Basis.XTRACK, names=names, length=lengths, s_out=s_out,
            R_elem=R, ref_kinetic_eV_in=np.full(n, ke), ref_kinetic_eV_out=np.full(n, ke),
            mass_eV=mass_eV, charge=charge, twiss=twiss, disp=disp, survey=survey,
            probe_out=probe_out, warnings=warnings, meta=meta,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_madx(deck: Path, wd: Path, beam: BeamSpec | None, sequence: str | None,
                   warnings: list[str]):
        import xtrack as xt

        try:
            from cpymad.madx import Madx
        except Exception as e:  # noqa: BLE001 -- no cpymad: xtrack's own MAD-X parser
            warnings.append(f"cpymad unavailable ({e}); deck parsed by xtrack's own MAD-X "
                            f"reader — element names differ from the cpymad oracle and the "
                            f"deck's BEAM is not carried over")
            env = xt.load(str(deck), format="madx")
            seqs = list(env.lines.keys())
            if not seqs:
                raise ValueError("deck defines no SEQUENCE/LINE that xtrack could build") from None
            seq = sequence or seqs[-1]
            line = env.lines[seq]
            beam_used = beam or _beam_from_particle_ref(line.particle_ref)
            return line, beam_used, "xt.load(format='madx')", seq

        from lattix.oracles.cpymad import MadxOracle

        m = Madx(stdout=False)
        m.chdir(str(wd))
        m.call(str(deck))
        seq = sequence or MadxOracle._pick_sequence(m)
        if beam is None:
            # a bare ``beam, ...;`` (no sequence=) is only attached to the sequence by USE;
            # MadxOracle._apply_beam reads sequence.beam, so USE first
            try:
                m.use(sequence=seq)
            except Exception as e:  # noqa: BLE001 -- MAD-X aborts: "USE - sequence without beam"
                raise ValueError(f"{deck.name}: sequence {seq!r} has no BEAM and no BeamSpec "
                                 f"was given") from e
        beam_used = MadxOracle._apply_beam(m, seq, beam)
        m.use(sequence=seq)
        line = xt.Line.from_madx_sequence(m.sequence[seq], deferred_expressions=False)
        m.quit()
        return line, beam_used, "cpymad + xt.Line.from_madx_sequence", seq

    @staticmethod
    def _track_probe(line, probe: Probe, ke: float, mass_eV: float) -> np.ndarray:
        from lattix.oracles.basis import transform_matrix

        Tinv = np.linalg.inv(transform_matrix(Basis.XTRACK, ke, mass_eV))
        native = np.asarray(probe.coords, dtype=float) @ Tinv.T
        p = line.build_particles(**{k: native[:, i].copy() for i, k in enumerate(_COORDS)})
        line.track(p)
        p.sort(interleave_lost_particles=True)
        out = _stack(p)
        out[np.asarray(p.state) <= 0] = np.nan
        return out
