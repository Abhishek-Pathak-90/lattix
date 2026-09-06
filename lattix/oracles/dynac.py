"""DYNAC oracle: per-card first-order maps of a DYNAC deck from a tracked probe.

DYNAC (V6R16, freeware under its own EULA) prints no transfer matrices: the adapter runs one DYNAC job per
optics card, each with a fresh ``RDBEAM`` probe — an on-axis particle and ± offsets in each of the six
coordinates ``(x cm, x′ rad, y cm, y′ rad, φ rad, W MeV)`` at the energy reached so far — and a ``WRBEAM``
dump after the card, and fits the card's 6×6 map from the twelve offset particles.  Energies are the on-axis
particle's (``WRBEAM IREC = 1``: absolute energies; DYNAC's own "reference" bookkeeping after an ``RDBEAM``
gives a buncher half its gain and is not used).  The basis is DYNAC's ``(x cm, x′, y cm, y′, φ rad late-
positive w.r.t. the master frequency, ΔW MeV)`` (:data:`Basis.DYNAC`; the master frequency follows
``NEWF``).

The binary comes from ``LATTIX_DYNAC_EXE``, ``dynac`` on the PATH, or the local clone's build
(``particle_tracking_codes/tier3_peers/dynac/build/source/dynac``); it is never in CI.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lattix.formats.dynac.cards import OPTICS_CARDS, Card, fnum, parse_deck
from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register

_AMU_EV = 931.49410242e6
#: probe offsets: x [cm], x′ [rad], y [cm], y′ [rad], φ [rad], W [MeV]
PROBE_AMP = (0.01, 1e-3, 0.01, 1e-3, 0.01, 1e-3)
_LOCAL_BUILD = Path("/Users/abhishekpathak/Desktop/Projects/particle_tracking_codes/tier3_peers/dynac"
                    "/build/source/dynac")
#: cards that only define, print or plot the beam: dropped from the probe deck
_DROP = frozenset({"GEBEAM", "INPUT", "RDBEAM", "ETAC", "EMITGR", "ENVEL", "PROFGR", "ACCEPT", "T3D", "WRBEAM",
                   "EMIT", "EMITL", "EMIPRT", "ZONES", "DCBEAM", "REFCOG",
                   "SCDYNAC", "SCDYNEL", "SCPOS",           # linear optics: no space charge on the probe
                   "REJECT", "CHASE", "COMPRES"})           # and no windows: the probe must survive
_WIDE_CM = 1000.0        # the pole-tip radius every lens gets in the probe deck (its field scaled along)
_AUX = frozenset({"ALINER", "ZROT", "TWQA", "REJECT", "FIELD", "HARM", "RWFIELD"})


@register
class DynacOracle:
    name = "dynac"
    formats = ("dynac",)

    def exe(self) -> Path:
        env = os.environ.get("LATTIX_DYNAC_EXE")
        if env:
            p = Path(env)
            if p.is_file():
                return p
            raise RuntimeError(f"LATTIX_DYNAC_EXE={env!r} is not a file")
        which = shutil.which("dynac")
        if which:
            return Path(which)
        if _LOCAL_BUILD.exists():
            return _LOCAL_BUILD
        raise RuntimeError("no dynac binary: set LATTIX_DYNAC_EXE or put dynac on the PATH (build the clone with "
                           "cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 and gfortran)")

    def available(self) -> tuple[bool, str]:
        try:
            return True, str(self.exe())
        except RuntimeError as e:
            return False, str(e)

    # ------------------------------------------------------------------ run
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None, probe: Probe | None = None,
            workdir: Path | None = None, amplitudes=PROBE_AMP) -> OracleResult:
        """One DYNAC job per optics card: a fresh, axis-aligned probe (``RDBEAM``) enters the card at the
        energy the on-axis particle had after the card before, and a ``WRBEAM`` dump after it gives the
        card's map.  (A single run with dumps after every card fits the maps from *propagated* offsets,
        whose conditioning collapses in strongly focusing lines — 1e5 on a 5 m chicane — beyond what
        six-digit dumps can resolve.)  The persistent DYNAC states are carried along: the master
        frequency (``NEWF``), ``SECORD``, an active ``TWQA`` roll, the ``FIELD``/``HARM`` block a cavity
        card reads, ``NREF`` offsets of the reference; ``ALINER`` (an affine beam shift) has no first-order
        map and is skipped; a ``TOF`` card that ties the RF phases to the time of flight cannot be honoured
        job by job (recorded in the warnings)."""
        if fmt not in (None, "dynac"):
            raise ValueError(f"DYNAC reads its own decks only, not {fmt!r}")
        exe = self.exe()
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        text = deck.read_text(encoding="latin-1", errors="replace")
        title, cards = parse_deck(text)
        beam = beam or self._beam_of(text, cards, deck)
        f0 = self._master_frequency(cards, beam)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_dynac_"))
        wd.mkdir(parents=True, exist_ok=True)
        atm = max(1, int(round(beam.mass_eV / _AMU_EV)))
        uem_MeV = beam.mass_eV * 1e-6 / atm                    # DYNAC's rest mass is UEM × ATM
        names = _card_names(cards)
        master = f0
        w_probe = beam.kinetic_energy_eV * 1e-6                # MeV, the on-axis particle
        w_ref_offset = 0.0                                     # NREF: the reference's energy − the probe's
        secord = False
        twqa: list[str] | None = None
        field_file: str | None = None
        field_att = "1."
        field_index: dict[str, int] = {}
        harm: list[str] | None = None
        blocks_cache: dict[str, list] = {}
        out_names, kinds, lengths, s_out, R, w_in, w_out, freqs, warns = [], [], [], [], [], [], [], [], []
        s = 0.0
        job = 0
        for c in cards:
            if c.name == "STOP":
                break
            if c.name == "NEWF":
                master = float(c.lines[0].split()[0])
            elif c.name == "SECORD":
                secord = True
            elif c.name == "FIRORD":
                secord = False
            elif c.name == "TWQA":
                twqa = None if float(c.lines[0].split()[1]) == 0.0 else list(c.lines)
            elif c.name == "FIELD":
                field_file, field_att, harm = c.lines[0].strip(), c.lines[1].split()[0], None
            elif c.name == "HARM":
                harm, field_file = list(c.lines), None
            elif c.name == "TOF" and c.lines and float(c.lines[0].split()[0]) == 0.0:
                warns.append("TOF 0: the deck ties its RF phases to the time of flight, which a job-per-card probe "
                             "cannot carry (phases taken at their offsets)")
            if c.name not in OPTICS_CARDS or c.name in ("NEWF", "ALINER", "TILT"):
                continue
            name = names.get(id(c)) or f"{c.name.lower()}_{len(out_names) + 1}"
            fstate = {"file": field_file, "index": field_index, "harm": float(harm[0].split()[0]) * 1e-2 if harm
                      else None}
            length = self._length(c, deck.parent, dict(fstate, index=dict(field_index)))  # noqa: E501
            jd = wd / f"job_{job:04d}"
            jd.mkdir(exist_ok=True)
            job += 1
            rows = self._probe_rows(w_probe, amplitudes)
            with open(jd / "probe.dst", "w") as fh:
                fh.write(f"{len(rows)} 0 0\n")
                for r in rows:
                    fh.write(" ".join(f"{v:.12g}" for v in r) + "\n")
            lines = [f"{title.strip()[:60]} (lattix probe {name})"[:80], "RDBEAM", "probe.dst", "0",
                     f"{fnum(master * 1e-6)} 0.", f"{fnum(uem_MeV)} {atm}",
                     f"{fnum(w_probe + w_ref_offset)} {fnum(float(beam.charge))}", "REFCOG", "1"]
            if secord:
                lines.append("SECORD")
            if twqa and c.name in ("QUADRUPO", "QUADSXT", "SOQUAD", "QUAFK"):
                lines += ["TWQA", *twqa]
            if c.name in ("CAVNUM", "CAVMC"):
                if harm is not None:
                    lines += ["HARM", *harm]
                elif field_file:
                    if field_file not in blocks_cache:
                        from lattix.formats.dynac.reader import _read_field_file

                        src = deck.parent / field_file
                        blocks_cache[field_file] = _read_field_file(src) if src.is_file() else []
                    blocks = blocks_cache[field_file]
                    k = field_index.get(field_file, 0)
                    field_index[field_file] = k + 1
                    if blocks:
                        z, e, fq = blocks[min(k, len(blocks) - 1)]
                        with open(jd / "field.txt", "w") as fh:
                            pairs = "".join(f"{fnum(zi)} {fnum(ei)}\n" for zi, ei in zip(z, e, strict=True))
                            fh.write(f"{fnum(fq)}\n{pairs}0. 0.\n")
                        lines += ["FIELD", "field.txt", field_att]
                    else:
                        warns.append(f"{name}: FIELD file {field_file!r} not found; the cavity is tracked as a drift")
            elif c.name in ("RFQPTQ", "EGUN", "FSOLE"):
                self._copy_side_files(c, deck.parent, jd)
            lines.append(c.name)
            lines += _widened(c)
            lines += ["WRBEAM", "out.dst", "1 100", "STOP"]
            (jd / "probe.in").write_text("\n".join(lines) + "\n", encoding="latin-1", errors="replace")
            proc = subprocess.run([str(exe), "probe.in"], cwd=str(jd), capture_output=True, text=True, check=False)
            out = jd / "out.dst"
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"dynac failed on card {job - 1} ({c.name}, {name}; exit {proc.returncode}) in {jd}:"
                                   f"\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")
            cur, w_abs = self._load_dump(out, warns, name)
            if cur.shape[0] != len(rows):
                raise RuntimeError(f"{cur.shape[0]} of {len(rows)} probe particles survived card {job - 1} "
                                   f"({c.name}, {name}); see {jd}")
            prev = np.array(rows, dtype=float)
            prev[:, 5] -= w_probe
            da = prev[1:, :] - prev[0]
            db = cur[1:, :] - cur[0]
            Rk, *_ = np.linalg.lstsq(da, db, rcond=None)
            if c.name == "NREF":
                t = c.lines[0].split()
                dew, irewf = float(t[1]), int(float(t[3])) if len(t) > 3 else 1
                if irewf == 1:
                    w_ref_offset += dew
                elif irewf == 2:
                    w_ref_offset = dew - w_probe
                else:
                    w_ref_offset += dew * 1e-2 * (w_probe + w_ref_offset)
            out_names.append(name)
            kinds.append(c.name)
            lengths.append(length)
            s += length
            s_out.append(s)
            R.append(Rk.T)
            w_in.append(w_probe * 1e6)
            w_probe = w_abs * 1e-6
            w_out.append(w_probe * 1e6)
            freqs.append(master)
        n = len(out_names)
        return OracleResult(engine="dynac", basis=Basis.DYNAC, names=out_names, length=np.array(lengths, dtype=float),
                            s_out=np.array(s_out, dtype=float),
                            R_elem=np.array(R, dtype=float).reshape(n, 6, 6) if n else np.zeros((0, 6, 6)),
                            ref_kinetic_eV_in=np.array(w_in, dtype=float),
                            ref_kinetic_eV_out=np.array(w_out, dtype=float),
                            mass_eV=float(beam.mass_eV), charge=int(beam.charge),
                            rf_frequency_Hz=np.array(freqs, dtype=float), warnings=warns,
                            meta={"workdir": str(wd), "p0_model": "follows the RF elements (on-axis probe particle)",
                                  "exe": str(exe), "kinds": kinds, "jobs": job})

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _probe_rows(ke_MeV: float, amp) -> list[list[float]]:
        rows = [[0.0, 0.0, 0.0, 0.0, 0.0, ke_MeV]]
        for k in range(6):
            for sgn in (+1.0, -1.0):
                r = [0.0, 0.0, 0.0, 0.0, 0.0, ke_MeV]
                r[k] += sgn * amp[k]
                rows.append(r)
        return rows

    @staticmethod
    def _load_dump(path: Path, warns: list[str], name: str) -> tuple[np.ndarray, float]:
        lines = path.read_text().splitlines()
        rows = []
        for ln in lines[1:]:
            t = ln.split()
            if len(t) >= 6:
                rows.append([float(v) for v in t[:6]])
        arr = np.array(rows, dtype=float)
        if not np.all(np.isfinite(arr)):
            warns.append(f"{name}: non-finite probe coordinates")
        w_abs = float(arr[0, 5]) * 1e6                    # IREC = 1: absolute (kinetic) energy [MeV]
        arr[:, 5] = (arr[:, 5] - arr[0, 5])                # energies relative to the on-axis particle
        return arr, w_abs

    def _beam_of(self, text: str, cards: list[Card], deck: Path) -> BeamSpec:
        from lattix.ir.reference_tag import parse_reference_tag

        tag = parse_reference_tag(text)
        if tag is not None:
            return BeamSpec(species=tag.species.name, kinetic_energy_eV=tag.kinetic_energy_eV,
                            frequency_Hz=tag.rf_frequency_Hz)
        for c in cards:
            if c.name == "INPUT" and len(c.lines) >= 2:
                t0, t1 = c.lines[0].split(), c.lines[1].split()
                mass = float(t0[0]) * 1e6 * (float(t0[1]) if len(t0) > 1 else 1.0)
                q, ke = int(round(float(t0[2]))), float(t1[0]) * 1e6
                return _beam_from_mass(mass, q, ke)
            if c.name == "RDBEAM" and len(c.lines) >= 5:
                t3 = c.lines[3].split()
                mass = float(t3[0]) * 1e6 * (float(t3[1]) if len(t3) > 1 else 1.0)
                ke, q = float(c.lines[4].split()[0]) * 1e6, int(round(float(c.lines[4].split()[1])))
                return _beam_from_mass(mass, q, ke)
        raise ValueError(f"{deck.name}: no beam block and no lattix reference tag; pass a BeamSpec")

    @staticmethod
    def _master_frequency(cards: list[Card], beam: BeamSpec) -> float:
        for c in cards:
            if c.name == "GEBEAM" and len(c.lines) >= 2:
                return float(c.lines[1].split()[0])
            if c.name == "RDBEAM" and len(c.lines) >= 3:
                return float(c.lines[2].split()[0]) * 1e6
        return float(beam.frequency_Hz or 1e6)

    @staticmethod
    def _copy_side_files(c: Card, src: Path, wd: Path) -> None:
        if c.name in ("FIELD", "RFQPTQ", "EGUN", "FSOLE") and c.lines:
            name = c.lines[0].strip()
            p = src / name
            if p.is_file() and not (wd / name).exists():
                (wd / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, wd / name)

    @staticmethod
    def _length(c: Card, deck_dir: Path, field_state: dict) -> float:
        t = c.lines[0].split() if c.lines else []
        try:
            if c.name in ("DRIFT", "FDRIFT", "QUADRUPO", "QUAELEC"):
                return float(t[0]) * 1e-2
            if c.name == "SEXTUPO":
                return float(t[2]) * 1e-2
            if c.name in ("QUADSXT", "SOQUAD"):
                return float(t[3]) * 1e-2
            if c.name == "SOLENO":
                return float(t[1]) * 1e-2
            if c.name == "QUAFK":
                return float(t[2]) * 1e-2
            if c.name == "BMAGNET":
                t1 = c.lines[1].split()
                return abs(math.radians(float(t1[0])) * float(t1[1]) * 1e-2)
            if c.name == "EDFLEC":
                t1 = c.lines[1].split()
                return abs(float(t1[0]) * 1e-2 * math.radians(float(t1[1])))
            if c.name == "CAVSC":
                return float(t[3]) * 1e-2
            if c.name in ("CAVNUM", "CAVMC"):
                if field_state.get("harm"):
                    return float(field_state["harm"])
                name = field_state.get("file")
                if name:
                    from lattix.formats.dynac.reader import _read_field_file

                    blocks = _read_field_file(deck_dir / name)
                    idx = field_state["index"].get(name, 0)
                    field_state["index"][name] = idx + 1
                    if blocks:
                        z = blocks[min(idx, len(blocks) - 1)][0]
                        return float(z[-1] - z[0])
                return 0.0
        except (IndexError, ValueError):
            return 0.0
        return 0.0


def _widened(c: Card) -> list[str]:
    """The card's lines with a pole-tip radius of :data:`_WIDE_CM` and the field scaled to keep the gradient
    (DYNAC loses particles outside a lens's radius; the probe's 1 mrad rays reach centimetres)."""
    try:
        t = c.lines[0].split() if c.lines else []
        if c.name == "QUADRUPO":
            L, b, r = float(t[0]), float(t[1]), float(t[2])
            if r > 0:
                return [f"{fnum(L)} {fnum(b * _WIDE_CM / r)} {fnum(_WIDE_CM)}", *c.lines[1:]]
        elif c.name == "SEXTUPO":
            imks, arg, L, r = int(float(t[0])), float(t[1]), float(t[2]), float(t[3])
            if r > 0:
                arg = arg * (_WIDE_CM / r) ** 2 if imks != 0 else arg
                return [f"{imks} {fnum(arg)} {fnum(L)} {fnum(_WIDE_CM)}", *c.lines[1:]]
        elif c.name == "BUNCHER":
            v, ph, h, r = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            return [f"{fnum(v)} {fnum(ph)} {fnum(h)} {fnum(_WIDE_CM)}", *c.lines[1:]]
        elif c.name in ("QUADSXT", "SOQUAD"):
            iksq, a1, a2, L, r = int(float(t[0])), float(t[1]), float(t[2]), float(t[3]), float(t[4])
            if r > 0:
                if iksq != 0:
                    if c.name == "QUADSXT":
                        a1, a2 = a1 * (_WIDE_CM / r) ** 2, a2 * (_WIDE_CM / r)
                    else:
                        a2 = a2 * (_WIDE_CM / r)
                return [f"{iksq} {fnum(a1)} {fnum(a2)} {fnum(L)} {fnum(_WIDE_CM)}", *c.lines[1:]]
        elif c.name == "QUAFK":
            ityqu, k, L, r = int(float(t[0])), float(t[1]), float(t[2]), float(t[3])
            return [f"{ityqu} {fnum(k)} {fnum(L)} {fnum(_WIDE_CM)}", *c.lines[1:]]
        elif c.name == "QUAELEC":
            L, v, r = float(t[0]), float(t[1]), float(t[2])
            if r > 0:
                return [f"{fnum(L)} {fnum(v * (_WIDE_CM / r) ** 2)} {fnum(_WIDE_CM)}", *c.lines[1:]]
    except (IndexError, ValueError):
        pass
    return list(c.lines)


def _card_names(cards: list[Card]) -> dict[int, str]:
    """``{id(card): name}`` from the ``; lattix: name=…`` tags: within a tagged group the principal optics card
    (the first one that is neither padding nor a modifier) carries the element's name, the others ``name~card``."""
    import re

    out: dict[int, str] = {}
    groups: list[tuple[str, list[Card]]] = []
    for c in cards:
        tag = None
        for ln in c.comments:
            m = re.search(r'lattix:\s+name=("([^"]*)"|\S+)', ln)
            if m:
                tag = m.group(2) if m.group(2) is not None else m.group(1)
        if tag is not None:
            groups.append((tag, [c]))
        elif groups:
            groups[-1][1].append(c)
    for tag, members in groups:
        optics = [c for c in members if c.name in OPTICS_CARDS]
        principal = next((c for c in optics if c.name not in _AUX and c.name != "DRIFT"), None)
        if principal is None:
            principal = next((c for c in optics if c.name not in _AUX), None) or (optics[0] if optics else None)
        k = 0
        for c in optics:
            if c is principal:
                out[id(c)] = tag
            else:
                k += 1
                out[id(c)] = f"{tag}~{c.name.lower()}{k}"
    return out


def _beam_from_mass(mass_eV: float, q: int, ke_eV: float) -> BeamSpec:
    from lattix.oracles.base import SPECIES

    for name, (m, qq) in SPECIES.items():
        if abs(m - mass_eV) / m < 1e-4 and qq == q:
            return BeamSpec(species=name, kinetic_energy_eV=ke_eV)
    raise ValueError(f"rest mass {mass_eV * 1e-6:.6f} MeV, charge {q} is not one of the engines' species")
