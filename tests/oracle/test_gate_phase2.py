"""Phase-2 acceptance gates (PLAN §5.6): A2 (fodo -> Bmad, Elegant), A3 all legs, A4 (BTL MAD8).

They skip until the corresponding formats exist and when an engine is missing.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.compare import compare_pair
from lattix.testing import corpus_dir

PUBLIC = Path(__file__).parents[1] / "data" / "public"
HELIX_EXAMPLES = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/examples")
HELIX_ROOT = HELIX_EXAMPLES.parent


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


def _truncate(res, s_max: float):
    k = int(np.count_nonzero(res.s_out <= s_max + 1e-9))
    return res.__class__(
        engine=res.engine, basis=res.basis, names=res.names[:k], length=res.length[:k], s_out=res.s_out[:k],
        R_elem=res.R_elem[:k], ref_kinetic_eV_in=res.ref_kinetic_eV_in[:k],
        ref_kinetic_eV_out=res.ref_kinetic_eV_out[:k], mass_eV=res.mass_eV, charge=res.charge,
        rf_frequency_Hz=None if res.rf_frequency_Hz is None else res.rf_frequency_Hz[:k],
        survey=None if res.survey is None else res.survey[:k], warnings=list(res.warnings), meta=dict(res.meta))


def _first_map_s(res) -> float:
    for s, L, n in zip(res.s_out, res.length, res.names, strict=True):
        if "FIELD_MAP" in n.upper() or n.upper().startswith("FM"):
            return float(s - L) - 1e-9
    return float(res.s_out[-1])


def _deck_from_manifest(mid: str, fallback: Path | None = None) -> Path:
    if fallback is not None and fallback.is_file():
        return fallback
    root = corpus_dir()
    if root:
        from lattix.corpus import load_manifest

        for e in load_manifest(root):
            if e.get("id") == mid and Path(e["path"]).is_file():
                return Path(e["path"])
    pytest.skip(f"{mid} not available")


# ---------------------------------------------------------------------------
@pytest.mark.oracle_madx
@pytest.mark.oracle_bmad
@pytest.mark.oracle_elegant
def test_A2_fodo_to_bmad_and_elegant(tmp_path):
    """fodo.madx -> .bmad and .lte: cpymad vs Tao vs elegant agree on the 6x6 maps
    (drift-padded thin cavities declared) and reproduce the end Twiss values."""
    _formats_ready("madx", "bmad", "elegant")
    from lattix.formats import read, translate

    madx, bmad, elegant = _engine("madx"), _engine("bmad"), _engine("elegant")
    src = PUBLIC / "helix" / "fodo.madx"
    ke = read(src)[0].reference.kinetic_energy_eV
    beam = BeamSpec("proton", ke, 352.21e6)
    out_b, out_e = tmp_path / "fodo.bmad", tmp_path / "fodo.lte"
    assert translate(src, out_b).ok
    assert translate(src, out_e).ok
    a = madx.run(src, fmt="madx", beam=beam, workdir=tmp_path / "a")
    b = bmad.run(out_b, fmt="bmad", beam=beam, workdir=tmp_path / "b")
    e = elegant.run(out_e, fmt="elegant", beam=beam, workdir=tmp_path / "e")
    for other in (b, e):
        pc = compare_pair(a, other)
        assert pc.n_shared >= 4, pc.row()
        assert pc.max_rcum_abs < 1e-8, pc.row()
        assert abs(pc.length_a - pc.length_b) < 1e-9
    for res in (a, b, e):
        if res.twiss:
            assert res.twiss["betx"][-1] == pytest.approx(7.163515, rel=1e-5)
            assert res.twiss["bety"][-1] == pytest.approx(15.286662, rel=1e-5)


@pytest.mark.oracle_helix
@pytest.mark.oracle_bmad
@pytest.mark.oracle_elegant
def test_A3_mebt_to_bmad_and_elegant_follow_p0(tmp_path):
    """PIP-II MEBT -> Bmad / Elegant: p0-following engines; the warm section before the
    first buncher map agrees with HELIX to 1e-7, and the reference energy through the
    whole line (gaps only; maps degrade) agrees with the IR walk."""
    _formats_ready("tracewin", "bmad", "elegant")
    from lattix.formats import read, translate

    helix, bmad, elegant = _engine("helix"), _engine("bmad"), _engine("elegant")
    src = _deck_from_manifest("helix-examples/examples/pipii/mebt/mebt.dat", HELIX_EXAMPLES / "pipii/mebt/mebt.dat")
    ropts = {"species": "h-", "kinetic_energy_eV": 2.1e6}
    lat, _ = read(src, **ropts)
    beam = BeamSpec("h-", 2.1e6, 162.5e6)
    h = helix.run(src, fmt="tracewin", beam=beam)
    s1 = _first_map_s(h)
    for fmt, eng in (("bmad", bmad), ("elegant", elegant)):
        out = tmp_path / f"mebt.{fmt if fmt == 'bmad' else 'lte'}"
        rep = translate(src, out, read_options=ropts)
        assert all(e.code in {"FM_TO_DRIFT", "FM_AS_LCAVITY", "APERTURE_SHAPE", "APERTURE_DROPPED", "SPECIES_ASSUMED",
                              "RF_ABSOLUTE_PHASE", "FM_FILES_MISSING", "DRIFT_SHIFT_DROPPED", "EKICK_AS_MAGNETIC"}
                   for e in rep.problems()), [e.code for e in rep.problems()]
        r = eng.run(out, fmt=fmt, beam=beam, workdir=tmp_path / fmt)
        pc = compare_pair(_truncate(h, s1), _truncate(r, s1))
        assert pc.n_shared > 5, pc.row()
        assert pc.blocks["T4x4"] < 1e-7 and pc.blocks["disp"] < 1e-7, (fmt, pc.row())
        assert abs(r.total_length - h.total_length) < 1e-9


@pytest.mark.oracle_madx
@pytest.mark.oracle_helix
def test_A4_btl_mad8_to_tracewin_and_madx(tmp_path):
    """BTL2025v0703.lat (MAD8, 949 elements, 307.969918 m) -> .dat and .madx:
    HELIX on the .dat vs cpymad on the .madx (lockstep through two writers)."""
    _formats_ready("mad8", "tracewin", "madx")
    from lattix.formats import read, translate

    madx, helix = _engine("madx"), _engine("helix")
    src = _deck_from_manifest("helix-pipii-root/btl2025v0703.lat", HELIX_ROOT / "BTL2025v0703.lat")
    lat, rep0 = read(src)
    ke = lat.reference.kinetic_energy_eV
    beam = BeamSpec("h-", ke, 162.5e6)
    dat, mad = tmp_path / "btl.dat", tmp_path / "btl.madx"
    rep1 = translate(src, dat)
    rep2 = translate(src, mad, write_options={"energy_mode": "constant"})
    allowed = {"INSTRUMENT_AS_MARKER", "APERTURE_DROPPED", "DRIFT_SHIFT_DROPPED", "RIGIDITY_FROM_BRHO",
               "KICKER_BODY", "EXPRESSION_DROPPED", "UNPARSEABLE_IDENTIFIER",
               # the BTL deck's MAD8 negative drift (DBV3NT = -0.204 m) cannot exist in MAD-X:
               "OVERLAP_SHIFTED", "DRIFT_INSIDE_OVERLAP", "MARKER_MOVED_OUT_OF_OVERLAP", "NEGATIVE_DRIFT_DROPPED"}
    for rep in (rep1, rep2):
        bad = [e.code for e in rep.problems() if e.code not in allowed]
        assert not bad, (sorted(set(bad)), rep.summary())
    h = helix.run(dat, fmt="tracewin", beam=beam)
    m = madx.run(mad, fmt="madx", beam=beam, workdir=tmp_path / "m")
    assert h.total_length == pytest.approx(307.969918, abs=1e-6)
    assert m.total_length == pytest.approx(307.969918, abs=1e-6)
    pc = compare_pair(h, m)
    assert pc.n_shared > 200, pc.row()
    assert pc.blocks["T4x4"] < 1e-7 and pc.blocks["disp"] < 1e-7, pc.row()
    assert m.survey is not None
