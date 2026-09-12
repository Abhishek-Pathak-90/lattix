"""MAD-X oracle through cpymad (MAD-X 5.09.03 bundled).

Per-element maps come from ``twiss, sectormap`` (the ``sectortable``: one row
per expanded-sequence element, map from the previous element's exit to this
element's exit — the same table HELIX pins its MAD-X importer against in
``tests/io/test_madx_conventions.py``).  Survey from ``survey``.  Probe bunch
via ``track, onepass, onetable``.  MAD-X keeps p0 constant through RF, so
``ref_kinetic_eV_in == ref_kinetic_eV_out`` everywhere (compare with
p0-following engines after damping normalisation, PLAN §5.1).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from lattix.oracles.base import SPECIES, Basis, BeamSpec, OracleResult, Probe, register

_MADX_PARTICLE = {"proton": "proton", "electron": "electron", "positron": "positron",
                  "h-": "ion", "deuteron": "ion"}


def _strip_occurrence(name: str) -> str:
    """MAD-X table names carry an occurrence suffix (``qf:1``); drop it."""
    n = str(name).lower()
    return n.rsplit(":", 1)[0] if ":" in n and n.rsplit(":", 1)[1].isdigit() else n


def _align(s_ref: np.ndarray, s_tbl: np.ndarray, tol: float = 1e-9, *,
           names_ref: list[str] | None = None, names_tbl: list[str] | None = None) -> list[int]:
    """Index into a MAD-X table (twiss/survey) whose rows are at positions ``s_tbl`` for each
    requested exit position in ``s_ref``.  With names, the row of the same element at that
    position wins (a thick element and a zero-length frame card ending at one s have different
    survey rows); otherwise the last match, so zero-length elements sharing an s pick the
    downstream row."""
    tbl = [_strip_occurrence(n) for n in names_tbl] if names_tbl is not None else None
    out = []
    for k, s in enumerate(s_ref):
        hits = np.nonzero(np.abs(s_tbl - s) <= tol)[0]
        if len(hits) == 0:
            raise ValueError(f"no MAD-X table row at s={s!r}")
        pick = int(hits[-1])
        if tbl is not None and names_ref is not None:
            same = [int(h) for h in hits if tbl[h] == names_ref[k]]
            if same:
                pick = same[-1]
        out.append(pick)
    return out


_MADX_OUTPUT_NAMES = {"sectormap", "sectormap.tfs", "trackone", "track.obs0001.p0001"}


def _mirror_deck_dir(deck: Path, wd: Path) -> Path:
    """Symlink the deck's folder into *wd* so relative ``CALL``s resolve while every
    file MAD-X writes (sectormap, track tables) lands in the temp dir, never next to
    the user's deck.  Names MAD-X itself writes are never linked (a linked
    ``sectormap`` would be overwritten through the link)."""
    for child in deck.parent.iterdir():
        if child.name in _MADX_OUTPUT_NAMES or child.name.startswith("."):
            continue
        target = wd / child.name
        if not target.exists() and not target.is_symlink():
            try:
                target.symlink_to(child)
            except OSError:
                pass
    return wd / deck.name


@register
class MadxOracle:
    name = "madx"
    formats = ("madx",)

    def available(self) -> tuple[bool, str]:
        try:
            from cpymad.madx import Madx  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"cpymad import failed: {e}"
        return True, "cpymad"

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            sequence: str | None = None) -> OracleResult:
        from cpymad.madx import Madx

        deck = Path(deck).resolve()
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_madx_"))
        wd.mkdir(parents=True, exist_ok=True)
        m = Madx(stdout=False)
        local = _mirror_deck_dir(deck, wd)
        m.chdir(str(wd))
        m.call(str(local))
        seq = sequence or self._pick_sequence(m)
        beam_used = self._apply_beam(m, seq, beam)
        m.use(sequence=seq)
        mass_eV, charge = beam_used.mass_eV, beam_used.charge
        ke = beam_used.kinetic_energy_eV

        try:
            tw = m.twiss(sequence=seq, betx=beam_used.betx, alfx=beam_used.alfx,
                         bety=beam_used.bety, alfy=beam_used.alfy, dx=beam_used.dx,
                         dpx=beam_used.dpx, dy=beam_used.dy, dpy=beam_used.dpy, sectormap=True,
                         sectorfile=str(wd / "sectormap.tfs"))
        except Exception as exc:  # noqa: BLE001 - cpymad raises TwissFailed, a RuntimeError subclass
            raise RuntimeError(
                f"MAD-X twiss failed on the open line ({type(exc).__name__}); MAD-X keeps p0 constant and expands "
                "its maps about the start momentum — a strongly accelerating line cannot be tracked this way: "
                "validate it with a p0-following engine (Bmad, Elegant, TraceWin/HELIX)") from exc
        st = m.table.sectortable
        names_all = [str(n) for n in st.name]
        keep = [i for i, n in enumerate(names_all) if not n.lower().endswith(("$start", "$end"))]
        names = [_strip_occurrence(names_all[i]) for i in keep]
        n = len(keep)
        R = np.empty((n, 6, 6))
        for a in range(6):
            for b in range(6):
                col = np.asarray(st[f"r{a + 1}{b + 1}"], dtype=float)
                R[:, a, b] = col[keep]
        s_out = np.asarray(st.pos, dtype=float)[keep]
        lengths = np.diff(np.concatenate([[0.0], s_out]))

        twiss = {}
        tw_idx = _align(s_out, np.asarray(tw.s, dtype=float), names_ref=names, names_tbl=list(tw.name))
        for key in ("betx", "alfx", "bety", "alfy", "dx", "dpx", "dy", "dpy"):
            twiss[key] = np.asarray(tw[key], dtype=float)[tw_idx]
        disp = {k: twiss.pop(k) for k in ("dx", "dpx", "dy", "dpy")}

        sv = m.survey(sequence=seq)
        sv_idx = _align(s_out, np.asarray(sv.s, dtype=float), names_ref=names, names_tbl=list(sv.name))
        survey6 = np.stack([np.asarray(sv[k], dtype=float)[sv_idx]
                            for k in ("x", "y", "z", "theta", "phi", "psi")], axis=1)
        survey = survey6[:, :4].copy()

        probe_out = None
        if probe is not None:
            probe_out = self._track_probe(m, seq, probe, ke, mass_eV)

        m.quit()
        return OracleResult(
            engine=self.name, basis=Basis.MADX, names=names, length=lengths, s_out=s_out,
            R_elem=R, ref_kinetic_eV_in=np.full(n, ke), ref_kinetic_eV_out=np.full(n, ke),
            mass_eV=mass_eV, charge=charge, twiss=twiss, disp=disp, survey=survey,
            probe_out=probe_out,
            meta={"sequence": seq, "workdir": str(wd),
                  "survey6": survey6.tolist()},        # x, y, z, theta, phi, psi at every exit
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_sequence(m) -> str:
        seqs = list(m.sequence.keys())
        if not seqs:
            raise ValueError("deck defines no SEQUENCE (LINE-only decks need a sequence)")
        return seqs[-1]

    @staticmethod
    def _apply_beam(m, seq: str, beam: BeamSpec | None) -> BeamSpec:
        """Explicit BeamSpec overrides the deck's BEAM; otherwise read the deck's."""
        if beam is not None:
            mass_GeV = beam.mass_eV * 1e-9
            m.command.beam(particle=_MADX_PARTICLE[beam.species.lower()], mass=mass_GeV,
                           charge=beam.charge, energy=beam.total_energy_eV * 1e-9, sequence=seq)
            return beam
        b = m.sequence[seq].beam
        mass_eV = float(b.mass) * 1e9
        charge = int(round(float(b.charge)))
        # identify species by mass (fallback: generic)
        species = "proton"
        for nm, (mv, q) in SPECIES.items():
            if abs(mv - mass_eV) / mv < 1e-4 and q == charge:
                species = nm
                break
        ke = float(b.energy) * 1e9 - mass_eV
        spec = BeamSpec(species=species, kinetic_energy_eV=ke)
        if species == "proton" and abs(mass_eV - SPECIES["proton"][0]) / mass_eV > 1e-4:
            spec.__dict__["_mass_override"] = mass_eV  # exotic ion: keep MAD-X mass
        return spec

    @staticmethod
    def _track_probe(m, seq: str, probe: Probe, ke: float, mass_eV: float) -> np.ndarray:
        from lattix.oracles.basis import transform_matrix

        Tinv = np.linalg.inv(transform_matrix(Basis.MADX, ke, mass_eV))
        native = probe.coords @ Tinv.T
        m.command.track(onepass=True, onetable=True)
        for row in native:
            m.command.start(x=row[0], px=row[1], y=row[2], py=row[3], t=row[4], pt=row[5])
        m.command.run(turns=1)
        m.command.endtrack()
        t = m.table.trackone
        num = np.asarray(t.number, dtype=int)
        turn = np.asarray(t.turn, dtype=int)
        out = np.full((len(native), 6), np.nan)
        cols = [np.asarray(t[k], dtype=float) for k in ("x", "px", "y", "py", "t", "pt")]
        for i in range(len(num)):
            if turn[i] == 1:
                out[num[i] - 1] = [c[i] for c in cols]
        return out
