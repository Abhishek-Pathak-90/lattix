"""Phase-1 acceptance gates (PLAN §5.6): A1, and the MAD-X legs of A3 and A5.

Each test translates a real deck through the IR and compares engines on BOTH
ends (never round-trip-only).  They skip until the TraceWin and MAD-X
readers/writers exist, and when an engine is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.compare import compare_pair
from lattix.testing import corpus_dir

PUBLIC = Path(__file__).parents[1] / "data" / "public"
HELIX_EXAMPLES = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/examples")


def _formats_ready(*names: str) -> None:
    import importlib

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
    """Rows of an OracleResult up to and including exit position s_max."""
    k = int(np.count_nonzero(res.s_out <= s_max + 1e-9))
    return res.__class__(
        engine=res.engine, basis=res.basis, names=res.names[:k], length=res.length[:k], s_out=res.s_out[:k],
        R_elem=res.R_elem[:k], ref_kinetic_eV_in=res.ref_kinetic_eV_in[:k],
        ref_kinetic_eV_out=res.ref_kinetic_eV_out[:k], mass_eV=res.mass_eV, charge=res.charge,
        rf_frequency_Hz=None if res.rf_frequency_Hz is None else res.rf_frequency_Hz[:k],
        survey=None if res.survey is None else res.survey[:k], warnings=list(res.warnings), meta=dict(res.meta))


def _first_map_s(res) -> float:
    """Entrance position of the first field map (rows strictly before it are comparable)."""
    for s, L, n in zip(res.s_out, res.length, res.names, strict=True):
        if "FIELD_MAP" in n.upper() or n.upper().startswith("FM"):
            return float(s - L) - 1e-9
    return float(res.s_out[-1])


EXPECTED_MEBT_CODES = {"FM_TO_DRIFT", "APERTURE_SHAPE", "APERTURE_DROPPED", "DRIFT_SHIFT_DROPPED",
                       "SPECIES_ASSUMED", "RF_ABSOLUTE_PHASE", "FM_FILES_MISSING"}


def _mebt_deck() -> Path:
    p = HELIX_EXAMPLES / "pipii" / "mebt" / "mebt.dat"
    if p.is_file():
        return p
    root = corpus_dir()
    if root:
        from lattix.corpus import load_manifest

        for e in load_manifest(root):
            if e.get("id") == "helix-examples/examples/pipii/mebt/mebt.dat":
                return Path(e["path"])
    pytest.skip("PIP-II MEBT deck not available")


# ---------------------------------------------------------------------------
@pytest.mark.oracle_madx
@pytest.mark.oracle_helix
def test_A1_fodo_madx_to_tracewin_to_madx(tmp_path):
    """fodo.madx -> .dat -> .madx: cpymad on both MAD-X decks and HELIX on the .dat
    agree in the transverse block, dispersion and survey; HELIX's known bend
    path-length gap is excluded from the assertion."""
    _formats_ready("madx", "tracewin")
    from lattix.formats import translate

    madx, helix = _engine("madx"), _engine("helix")
    from lattix.formats import read

    src = PUBLIC / "helix" / "fodo.madx"
    dat = tmp_path / "fodo.dat"
    back = tmp_path / "fodo_back.madx"
    ke = read(src)[0].reference.kinetic_energy_eV      # the deck's BEAM: 799.99991 MeV, not 800
    rep1 = translate(src, dat, write_options={"species": "proton"})
    rep2 = translate(dat, back, read_options={"species": "proton", "kinetic_energy_eV": ke,
                                              "frequency_Hz": 352.21e6})
    assert rep1.ok and rep2.ok, (rep1.summary(), rep2.summary())
    beam = BeamSpec("proton", ke, 352.21e6)
    a = madx.run(src, fmt="madx", beam=beam, workdir=tmp_path / "a")
    b = madx.run(back, fmt="madx", beam=beam, workdir=tmp_path / "b")
    h = helix.run(dat, fmt="tracewin", beam=beam)
    ab = compare_pair(a, b)
    assert ab.n_shared >= 4 and abs(ab.length_a - ab.length_b) < 1e-9
    assert ab.max_rcum_abs < 1e-8, ab.row()
    assert ab.survey_end_abs is not None and ab.survey_end_abs < 1e-9
    ah = compare_pair(a, h)
    assert ah.blocks["T4x4"] < 1e-6 and ah.blocks["disp"] < 1e-7, ah.row()


@pytest.mark.oracle_madx
@pytest.mark.oracle_helix
def test_A3_mebt_tracewin_to_madx(tmp_path):
    """PIP-II MEBT (427 elements, quads + gaps, no maps) -> MAD-X: HELIX on the
    source vs cpymad on the target, constant-p0 rescaled, transverse block to 1e-7
    (quads only see the energy through the local rigidity: energy_mode=local)."""
    _formats_ready("madx", "tracewin")
    from lattix.formats import translate

    madx, helix = _engine("madx"), _engine("helix")
    src = _mebt_deck()
    out = tmp_path / "mebt.madx"
    rep = translate(src, out, read_options={"species": "h-", "kinetic_energy_eV": 2.1e6},
                    write_options={"energy_mode": "constant"})
    unexpected = [e for e in rep.problems() if e.code not in EXPECTED_MEBT_CODES]
    assert not unexpected, [e.code for e in unexpected]
    assert rep.codes().get("FM_TO_DRIFT") == 4, "the MEBT has four buncher field maps"
    beam = BeamSpec("h-", 2.1e6, 162.5e6)
    h = helix.run(src, fmt="tracewin", beam=beam)
    m = madx.run(out, fmt="madx", beam=beam, workdir=tmp_path / "m")
    assert abs(h.total_length - m.total_length) < 1e-9
    # the four bunchers are drifts in MAD-X (Phase 3 brings FM_TO_CAVITY): compare the
    # warm section before the first field map, where the physics is identical
    s1 = _first_map_s(h)
    pc = compare_pair(_truncate(h, s1), _truncate(m, s1))
    assert pc.n_shared > 5, pc.row()
    assert pc.blocks["T4x4"] < 1e-7, pc.row()
    assert pc.blocks["disp"] < 1e-7, pc.row()


@pytest.mark.oracle_madx
@pytest.mark.oracle_helix
def test_A5_mebt_hwr_fieldmaps_to_madx_are_reported(tmp_path):
    """mebt+hwr.dat has FIELD_MAP cavities/solenoids.  Phase 1 degrades them to
    drifts (Phase 3 brings FM_TO_CAVITY): the report must list every map, strict
    mode must raise, and the transverse optics of the warm MEBT section (before
    the first map) must still agree with HELIX."""
    _formats_ready("madx", "tracewin")
    from lattix.fidelity import TranslationError
    from lattix.formats import translate

    src = HELIX_EXAMPLES / "pipii" / "mebt+hwr" / "mebt+hwr.dat"
    if not src.is_file():
        pytest.skip("mebt+hwr deck not available")
    madx, helix = _engine("madx"), _engine("helix")
    out = tmp_path / "mebt_hwr.madx"
    rep = translate(src, out, read_options={"species": "h-", "kinetic_energy_eV": 2.1e6},
                    write_options={"energy_mode": "constant"})
    fm = [e for e in rep.entries if e.kind == "FieldMap" and e.cls.value in ("LOSSY", "DROPPED")]
    assert fm, "field-map degradation must be reported"
    with pytest.raises(TranslationError):
        translate(src, tmp_path / "strict.madx", strict=True,
                  read_options={"species": "h-", "kinetic_energy_eV": 2.1e6})
    beam = BeamSpec("h-", 2.1e6, 162.5e6)
    h = helix.run(src, fmt="tracewin", beam=beam)
    m = madx.run(out, fmt="madx", beam=beam, workdir=tmp_path / "m")
    s1 = _first_map_s(h)
    pc = compare_pair(_truncate(h, s1), _truncate(m, s1))
    assert pc.n_shared > 5, pc.row()
    assert pc.blocks["T4x4"] < 1e-7 and pc.blocks["disp"] < 1e-7, pc.row()
