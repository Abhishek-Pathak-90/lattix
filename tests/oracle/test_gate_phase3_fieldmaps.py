"""Phase-3 gate: field maps (PLAN §6 task 3.1/3.7, the A5 TraceWin leg).

``mebt+hwr.dat`` (483 cards, 12 real RF field maps and 8 solenoid maps) is translated to the
two p0-following targets that can carry an accelerating cavity — Bmad ``.bmad`` and elegant
``.lte`` — and the engines are asked what the *reference energy* does along the line:

* every field map's ``dE_ref`` from ``lattix.ir.fieldmap.integrate_map`` reproduces HELIX's
  own ``advance_ref`` (Exact tier, ≤ 1e-6 relative on every map);
* the reference energy after every cavity agrees with HELIX's within 0.5 % (Equivalent tier);
* the transverse 4×4 block up to the first map is still 1e-7 (the warm MEBT is untouched).

``pytest -s`` prints the per-cavity table (HELIX vs the IR summary vs the engine).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.compare import compare_pair
from lattix.testing import corpus_dir, helix_path

HELIX_EXAMPLES = helix_path('examples')
FIELDS = HELIX_EXAMPLES.parent / "Fields"
BEAM = BeamSpec("h-", 2.1e6, 162.5e6)
READ_OPTIONS = {"species": "h-", "kinetic_energy_eV": 2.1e6, "frequency_Hz": 162.5e6}
#: PLAN §5.2 Equivalent tier for a reference-energy profile through translated cavities
ENERGY_TOL = 5e-3


def _formats_ready(*names: str) -> None:
    for n in names:
        try:
            importlib.import_module(f"lattix.formats.{n}")
        except ModuleNotFoundError:
            pytest.skip(f"format {n!r} not implemented yet")


def _engine(name: str):
    o = get_oracle(name)
    ok, why = o.available()
    if not ok:
        pytest.skip(why)
    return o


def _deck() -> Path:
    p = HELIX_EXAMPLES / "pipii" / "mebt+hwr" / "mebt+hwr.dat"
    if p.is_file() and FIELDS.is_dir():
        return p
    root = corpus_dir()
    if root:
        from lattix.corpus import load_manifest

        for e in load_manifest(root):
            if e.get("id", "").endswith("pipii/mebt+hwr/mebt+hwr.dat"):
                return Path(e["path"])
    pytest.skip("mebt+hwr deck or the ANL/CEA field maps are not available here")


def _truncate(res, s_max: float):
    k = int(np.count_nonzero(res.s_out <= s_max + 1e-9))
    return res.__class__(
        engine=res.engine, basis=res.basis, names=res.names[:k], length=res.length[:k],
        s_out=res.s_out[:k], R_elem=res.R_elem[:k], ref_kinetic_eV_in=res.ref_kinetic_eV_in[:k],
        ref_kinetic_eV_out=res.ref_kinetic_eV_out[:k], mass_eV=res.mass_eV, charge=res.charge,
        rf_frequency_Hz=None if res.rf_frequency_Hz is None else res.rf_frequency_Hz[:k],
        survey=None if res.survey is None else res.survey[:k], warnings=list(res.warnings),
        meta=dict(res.meta))


def _energy_at(res, s: float) -> float:
    """Reference kinetic energy just after position *s* [eV]."""
    k = int(np.count_nonzero(res.s_out <= s + 1e-6)) - 1
    return float(res.ref_kinetic_eV_out[max(k, 0)])


def _maps(lat) -> list[tuple[str, float, float, str]]:
    """(name, s_out, dE_ref, kind) for every field map in deck order."""
    from lattix.ir.walk import propagate

    out = []
    for p in propagate(lat):
        el = p.element
        if el.kind != "FieldMap":
            continue
        s = (el.meta or {}).get("map_summary") or {}
        out.append((el.name, p.s_out, el.rf.dE_ref_eV or 0.0, s.get("kind", "?")))
    return out


@pytest.fixture(scope="module")
def ir():
    _formats_ready("tracewin")
    from lattix.formats.tracewin import read

    lat, rep = read(_deck(), **READ_OPTIONS)
    maps = _maps(lat)
    if not maps or all(m[3] == "?" for m in maps):
        pytest.skip("the deck's field maps were not integrated (map files absent)")
    return lat, rep, maps


# ---------------------------------------------------------------------------
@pytest.mark.oracle_helix
def test_every_map_gain_matches_helix(ir, capsys):
    """The IR's integrated dE_ref is HELIX's advance_ref to round-off, map by map."""
    from tests.formats.test_fieldmap_integration import _helix_gains

    _engine("helix")
    _lat, rep, maps = ir
    theirs = _helix_gains(_deck(), 2.1, 162.5)
    assert len(theirs) == len(maps)
    worst = 0.0
    rows = []
    for (name, s, dE, kind), want in zip(maps, theirs, strict=True):
        rel = abs(dE - want) / abs(want) if want else 0.0
        worst = max(worst, rel if abs(want) > 1e-3 else 0.0)
        rows.append(f"  {name:<16s} {kind:<9s} s={s:8.4f} m  HELIX={want:14.6e} eV  "
                    f"IR={dE:14.6e} eV  rel={rel:.2e}")
        assert dE == pytest.approx(want, rel=1e-6, abs=1e-5)
    with capsys.disabled():
        print(f"\nper-map reference gain, IR vs HELIX (worst rel {worst:.1e}):")
        print("\n".join(rows))
    assert "FM_INTEGRATED" in {e.code for e in rep.entries}
    assert worst <= 1e-6


@pytest.mark.oracle_helix
@pytest.mark.parametrize("fmt, suffix", [("bmad", ".bmad"), ("elegant", ".lte")])
def test_reference_energy_after_every_cavity_follows_helix(ir, fmt, suffix, tmp_path, capsys):
    """PLAN A5, Equivalent tier: a p0-following target reproduces the linac's energy profile."""
    _formats_ready("tracewin", fmt)
    from lattix.formats import translate

    helix, engine = _engine("helix"), _engine(fmt)
    _lat, _rep, maps = ir
    src = _deck()
    out = tmp_path / f"mebt_hwr{suffix}"
    rep = translate(src, out, read_options=READ_OPTIONS)
    codes = rep.codes()
    assert codes.get("FM_TO_CAVITY", 0) >= 12 and codes.get("FM_SOL_HARDEDGE", 0) >= 8
    assert "FM_TO_DRIFT" not in codes

    h = helix.run(src, fmt="tracewin", beam=BEAM)
    try:
        r = engine.run(out, fmt=fmt, beam=BEAM, workdir=tmp_path / fmt)
    except RuntimeError as exc:                    # engine build without the needed output
        pytest.skip(f"{fmt} oracle cannot report the reference energy here: {exc}")
    assert abs(r.total_length - h.total_length) < 1e-6

    rows, worst = [], 0.0
    e0 = BEAM.kinetic_energy_eV
    prev_h = prev_r = e0
    for name, s, dE, kind in maps:
        if kind != "rf":
            continue
        eh, er = _energy_at(h, s), _energy_at(r, s)
        rel = abs(er - eh) / eh
        worst = max(worst, rel)
        rows.append(f"  {name:<16s} s={s:8.4f} m  dE: HELIX={eh - prev_h:12.4e}  "
                    f"IR={dE:12.4e}  {fmt}={er - prev_r:12.4e} eV   "
                    f"E: HELIX={eh * 1e-6:9.6f}  {fmt}={er * 1e-6:9.6f} MeV  rel={rel:.2e}")
        prev_h, prev_r = eh, er
        assert rel < ENERGY_TOL, f"{fmt} reference energy after {name}: {er:.6e} vs HELIX {eh:.6e}"
    with capsys.disabled():
        print(f"\nreference energy after every cavity, {fmt} vs HELIX "
              f"(worst {worst * 100:.3f} %, tier limit {ENERGY_TOL * 100:.1f} %):")
        print("\n".join(rows))
    assert _energy_at(r, h.s_out[-1]) == pytest.approx(_energy_at(h, h.s_out[-1]),
                                                       rel=ENERGY_TOL)
    assert worst < ENERGY_TOL


@pytest.mark.oracle_helix
@pytest.mark.parametrize("fmt, suffix", [("bmad", ".bmad"), ("elegant", ".lte")])
def test_transverse_block_before_the_first_map_is_unchanged(ir, fmt, suffix, tmp_path):
    """Degrading the maps must not disturb the warm MEBT in front of them (1e-7, Exact tier)."""
    _formats_ready("tracewin", fmt)
    from lattix.formats import translate

    helix, engine = _engine("helix"), _engine(fmt)
    _lat, _rep, maps = ir
    src = _deck()
    out = tmp_path / f"mebt_hwr{suffix}"
    translate(src, out, read_options=READ_OPTIONS)
    h = helix.run(src, fmt="tracewin", beam=BEAM)
    try:
        r = engine.run(out, fmt=fmt, beam=BEAM, workdir=tmp_path / fmt)
    except RuntimeError as exc:
        pytest.skip(f"{fmt} oracle unavailable for this deck: {exc}")
    s1 = maps[0][1] - 1e-9 - 0.24                       # entrance of the first map
    pc = compare_pair(_truncate(h, s1), _truncate(r, s1))
    assert pc.n_shared > 5, pc.row()
    assert pc.blocks["T4x4"] < 1e-7, pc.row()
    assert pc.blocks["disp"] < 1e-7, pc.row()


@pytest.mark.oracle_madx
def test_madx_gets_the_same_cavities_and_hard_edges(ir, tmp_path):
    """MAD-X cannot follow p0 (PLAN §8), so its own ``twiss`` diverges once 8 MeV of gain is
    linearised about a 2.1 MeV reference — the gate here is the *deck*: every map is an
    rfcavity or a hard-edge magnet, never a drift, MAD-X itself accepts the sequence, and the
    line is exactly as long as HELIX's."""
    _formats_ready("tracewin", "madx")
    _engine("madx")
    from cpymad.madx import Madx

    from lattix.formats import translate

    _lat, _rep0, maps = ir
    out = tmp_path / "mebt_hwr.madx"
    rep = translate(_deck(), out, read_options=READ_OPTIONS,
                    write_options={"energy_mode": "constant"})
    codes = rep.codes()
    assert codes.get("FM_TO_CAVITY", 0) >= 12 and codes.get("FM_SOL_HARDEDGE", 0) >= 8
    assert "FM_TO_DRIFT" not in codes
    text = out.read_text()
    assert text.count(": rfcavity,") >= 12 and text.count(": solenoid,") >= 8
    # every accelerating map is reported as EQUIVALENT:CONST_P0, never silently
    assert codes.get("CONST_P0", 0) >= 12

    m = Madx(stdout=False)
    try:
        m.chdir(str(tmp_path))
        m.call(out.name)
        seq = next(iter(m.sequence))
        length = m.sequence[seq].length
        n_cav = sum(1 for e in m.sequence[seq].elements if e.base_type.name == "rfcavity")
        n_sol = sum(1 for e in m.sequence[seq].elements if e.base_type.name == "solenoid")
    finally:
        m.quit()
    assert n_cav >= 12 and n_sol >= 8
    assert length == pytest.approx(max(s for _n, s, _d, _k in maps) + 0.0, abs=2.0)
    helix = _engine("helix")
    h = helix.run(_deck(), fmt="tracewin", beam=BEAM)
    assert length == pytest.approx(h.total_length, abs=1e-6)
