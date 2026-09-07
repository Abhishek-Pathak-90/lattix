"""MAD-X reader tests (PLAN §6 task 1.4).

Numbers are either hand-computed from the deck text or pinned against
``tests/corpus/golden.yaml`` (cpymad's own expansion of the two public decks:
11 / 20 expanded rows, 6.6 m / 12.0 m).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.madx import Reader
from lattix.ir.elements import (
    Bend,
    Collimator,
    Directive,
    Drift,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    Octupole,
    Quadrupole,
    RFCavity,
    Sextupole,
    Solenoid,
    Taylor,
)
from lattix.ir.rf import phase_from_madx_lag
from lattix.testing import require

require("madx")

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
FODO = DATA / "fodo.madx"
TRANSPORT = DATA / "transport.madx"

M_P = 938_272_088.16
M_H = M_P + 2 * 510_998.95


def _deck(tmp_path: Path, body: str, name: str = "d.madx") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


@pytest.fixture(scope="module")
def fodo():
    return Reader().read(FODO)


@pytest.fixture(scope="module")
def transport():
    return Reader().read(TRANSPORT)


# --------------------------------------------------------------------------- fodo.madx
def test_fodo_expansion_matches_cpymad(fodo):
    lat, rep = fodo
    assert lat.meta["madx_expanded_rows"] == 11        # golden.yaml madx_n_rows
    assert lat.meta["madx_sequence"] == "fodo"
    assert lat.meta["madx_title"] == "HELIX MAD-X demo - FODO cell"
    placed = lat.flatten()
    assert [p.name for p in placed] == [
        "qf", "drift_0", "b1", "drift_1", "qd", "drift_2", "b1", "drift_3", "m1"]
    assert [type(p.element) for p in placed] == [
        Quadrupole, Drift, Bend, Drift, Quadrupole, Drift, Bend, Drift, Marker]
    assert lat.total_length == pytest.approx(6.6, abs=1e-12)   # golden.yaml madx_length_m
    assert rep.counts["EXACT"] == 9
    assert rep.ok


def test_fodo_reference_particle(fodo):
    lat, _ = fodo
    ref = lat.reference
    assert ref.species.name == "proton"
    assert ref.kinetic_energy_eV == pytest.approx(1.738272e9 - M_P, abs=1e-3)
    assert ref.brho_signed == pytest.approx(4.881029892585647, rel=1e-14)   # cpymad beam.brho


def test_fodo_gradient_is_k1_times_brho(fodo):
    """I-3: the IR keeps the lab gradient; k1·Bρ_signed must reproduce it."""
    lat, _ = fodo
    brho = lat.reference.brho_signed
    assert lat.elements["qf"].gradient == pytest.approx(0.6 * brho, rel=1e-15)
    assert lat.elements["qd"].gradient == pytest.approx(-0.6 * brho, rel=1e-15)
    assert lat.elements["qf"].gradient == pytest.approx(2.928617935551388, rel=1e-14)
    assert lat.elements["qf"].length == 0.3


def test_fodo_bend_geometry(fodo):
    lat, _ = fodo
    b = lat.elements["b1"]
    assert b.length == 1.0
    assert b.bend.angle == pytest.approx(0.1)
    assert b.bend.e1 == pytest.approx(0.05)
    assert b.bend.e2 == pytest.approx(0.05)
    assert b.bend.rect is False
    assert b.bend.edge_int1 == 0.0
    assert b.bend.edge_int2 is None            # MAD-X fintx = -1 means "same as fint"
    assert b.rho == pytest.approx(10.0)


def test_fodo_variables_captured(fodo):
    lat, _ = fodo
    assert set(lat.variables) == {"lquad", "kfocus", "lbend", "abend", "ehalf"}
    assert lat.variables["kfocus"].value == pytest.approx(0.6)
    assert lat.variables["lquad"].value == pytest.approx(0.3)
    assert lat.variables["ehalf"].value == pytest.approx(0.05)
    # MAD-X evaluates `=` immediately, so these carry no deferred expression
    assert all(v.expression is None for v in lat.variables.values())


# ----------------------------------------------------------------- transport.madx
def test_transport_expansion(transport):
    lat, rep = transport
    assert lat.meta["madx_expanded_rows"] == 20
    placed = lat.flatten()
    assert len(placed) == 18
    assert lat.total_length == pytest.approx(12.0, abs=1e-12)
    kinds = [type(p.element) for p in placed]
    assert kinds.count(Drift) == 9
    assert Sextupole in kinds and RFCavity in kinds and Multipole in kinds
    assert isinstance(lat.elements["bpm"], Instrument)
    assert lat.elements["bpm"].family == "BPM"
    assert rep.ok


def test_transport_rbend_uses_the_arc_length(transport):
    """MAD-X reports the chord in ``l`` and the arc as the node length (rbarc=true)."""
    lat, _ = transport
    br = lat.elements["br"]
    angle = 0.08
    arc = 0.8 * (angle / 2) / math.sin(angle / 2)
    assert br.length == pytest.approx(arc, rel=1e-15)
    assert br.length == pytest.approx(0.800213373162275, rel=1e-14)
    assert br.bend.rect is True
    # RBEND pole faces are chord-referenced: the IR stores the sector reference
    assert br.bend.e1 == pytest.approx(angle / 2)
    assert br.bend.e2 == pytest.approx(angle / 2)


def test_transport_rfcavity_phase(transport):
    lat, _ = transport
    cav = lat.elements["cav"]
    assert cav.rf.voltage_V == pytest.approx(1.5e6)
    assert cav.rf.frequency_Hz == pytest.approx(325e6)
    assert cav.rf.phase_rad == pytest.approx(phase_from_madx_lag(0.0))
    assert cav.rf.phase_rad == pytest.approx(-math.pi / 2)      # lag 0 = zero crossing
    assert cav.rf.L_active_m == pytest.approx(0.4)


def test_transport_multipole_and_sextupole(transport):
    lat, _ = transport
    brho = lat.reference.brho_signed
    assert lat.elements["sf"].multipole.Bn[2] == pytest.approx(2.0 * brho)
    oc = lat.elements["oct"]
    assert isinstance(oc, Multipole)
    assert oc.length == 0.0
    assert oc.multipole.BnL == {3: pytest.approx(0.5 * brho)}


# ----------------------------------------------------------------------- species
def test_hminus_beam_flips_the_gradient_sign(tmp_path):
    body = """
beam, particle=ion, mass={mass}, charge={q}, energy={e};
qf: quadrupole, l=0.3, k1=0.6;
s: sequence, l=1.0, refer=centre;
  qf, at=0.5;
endsequence;
use, sequence=s;
"""
    p = _deck(tmp_path, body.format(mass=M_P / 1e9, q=1, e=(M_P + 8e8) / 1e9), "p.madx")
    h = _deck(tmp_path, body.format(mass=M_H / 1e9, q=-1, e=(M_H + 8e8) / 1e9), "h.madx")
    lat_p, _ = Reader().read(p)
    lat_h, _ = Reader().read(h)
    assert lat_h.reference.species.name == "h-"
    assert lat_h.reference.species.charge == -1
    assert lat_p.elements["qf"].gradient > 0 > lat_h.elements["qf"].gradient
    assert lat_h.elements["qf"].gradient == pytest.approx(0.6 * lat_h.reference.brho_signed)
    assert lat_h.reference.brho_signed < 0


def test_species_and_energy_overrides(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
qf: quadrupole, l=0.3, k1=0.6;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p, species="h-", kinetic_energy_eV=8e8)
    assert lat.reference.species.name == "h-"
    assert lat.reference.kinetic_energy_eV == 8e8
    assert lat.elements["qf"].gradient < 0


def test_unknown_species_becomes_a_custom_species(tmp_path):
    p = _deck(tmp_path, """
beam, particle=ion, mass=12.0, charge=6, energy=20.0;
qf: quadrupole, l=0.3, k1=0.6;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    assert lat.reference.species.charge == 6
    assert lat.reference.species.mass_eV == pytest.approx(12e9)
    assert any("custom Species" in w for w in lat.warnings)


# ------------------------------------------------------------------- conventions
def test_skew_quadrupole_folds_into_a_tilt(tmp_path):
    """Measured: (k1, k1s) == normal quad hypot(k1, k1s) rotated by -atan2(k1s, k1)/2."""
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q: quadrupole, l=0.3, k1=0.4, k1s=0.3;
s: sequence, l=1.0; q, at=0.5; endsequence;
use, sequence=s;
""")
    lat, rep = Reader().read(p)
    brho = lat.reference.brho_signed
    q = lat.elements["q"]
    assert q.multipole.Bn[1] == pytest.approx(math.hypot(0.4, 0.3) * brho)
    assert q.skew_rad == pytest.approx(-math.atan2(0.3, 0.4) / 2)
    assert "SKEW_QUAD_AS_TILT" in rep.codes()
    assert rep.ok                       # EQUIVALENT never blocks


def test_pure_skew_quadrupole_is_a_minus_45_degree_tilt(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q: quadrupole, l=0.3, k1s=0.5;
s: sequence, l=1.0; q, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    assert lat.elements["q"].skew_rad == pytest.approx(-math.pi / 4)


def test_element_expressions_and_variables(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
kqf := 0.31 + 0.29;
lq = 0.3;
qf: quadrupole, l=lq, k1:=kqf;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    assert lat.variables["kqf"].value == pytest.approx(0.6)
    assert lat.variables["kqf"].expression.deferred is True
    assert lat.variables["kqf"].expression.text == "0.31+0.29"
    qf = lat.elements["qf"]
    assert qf.expressions["multipole.Bn[1]"].text == "kqf"
    assert qf.expressions["multipole.Bn[1]"].deferred is True
    assert qf.native["madx"]["k1_expr"] == "kqf"          # normalized MAD-X quantity
    assert qf.multipole.Bn[1] == pytest.approx(0.6 * lat.reference.brho_signed)


def test_keep_expressions_false(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
kqf := 0.31 + 0.29;
qf: quadrupole, l=0.3, k1:=kqf;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p, keep_expressions=False)
    assert lat.elements["qf"].expressions == {}
    assert lat.variables["kqf"].expression is None


def test_bend_full_attribute_set(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
bb: sbend, l=1.0, angle=0.1, e1=0.02, e2=0.03, fint=0.4, fintx=0.3, hgap=0.03,
    k1=0.2, k2=0.1, tilt=0.2;
s: sequence, l=2.0; bb, at=1.0; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    brho = lat.reference.brho_signed
    b = lat.elements["bb"]
    assert (b.bend.angle, b.bend.e1, b.bend.e2) == pytest.approx((0.1, 0.02, 0.03))
    assert b.bend.edge_int1 == pytest.approx(0.4)
    assert b.bend.edge_int2 == pytest.approx(0.3)
    assert b.bend.hgap == pytest.approx(0.03)
    assert b.bend.tilt_ref == pytest.approx(0.2)
    assert b.multipole.Bn[1] == pytest.approx(0.2 * brho)
    assert b.multipole.Bn[2] == pytest.approx(0.1 * brho)


def test_solenoid_kicker_collimator_matrix(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
sol: solenoid, l=0.5, ks=0.3;
hk: hkicker, kick=0.001;
vk: vkicker, kick=-0.002;
kk: kicker, hkick=0.005, vkick=0.006;
rc: rcollimator, l=0.1, xsize=0.02, ysize=0.03;
ec: ecollimator, l=0.1, xsize=0.02, ysize=0.03;
mx: matrix, l=0.0, rm12=0.5, rm16=0.7, kick1=0.001;
oc: octupole, l=0.2, k3=4.0;
s: sequence, l=4.0, refer=entry;
  sol, at=0.0; hk, at=0.6; vk, at=0.7; kk, at=0.8; rc, at=1.0; ec, at=1.2;
  mx, at=1.5; oc, at=1.6;
endsequence;
use, sequence=s;
""")
    lat, rep = Reader().read(p)
    brho = lat.reference.brho_signed
    assert isinstance(lat.elements["sol"], Solenoid)
    assert lat.elements["sol"].solenoid.Bsol_T == pytest.approx(0.3 * brho)
    assert isinstance(lat.elements["hk"], Kicker)
    assert (lat.elements["hk"].hkick, lat.elements["hk"].vkick) == (0.001, 0.0)
    assert (lat.elements["vk"].hkick, lat.elements["vk"].vkick) == (0.0, -0.002)
    assert (lat.elements["kk"].hkick, lat.elements["kk"].vkick) == (0.005, 0.006)
    rc, ec = lat.elements["rc"], lat.elements["ec"]
    assert isinstance(rc, Collimator) and rc.aperture.shape == "RECTANGULAR"
    assert (rc.aperture.half_x, rc.aperture.half_y) == pytest.approx((0.02, 0.03))
    assert ec.aperture.shape == "ELLIPTICAL"
    mx = lat.elements["mx"]
    assert isinstance(mx, Taylor)
    assert mx.matrix[0][1] == 0.5 and mx.matrix[0][5] == 0.7 and mx.matrix[2][2] == 1.0
    assert mx.offset[0] == 0.001
    assert isinstance(lat.elements["oc"], Octupole)
    assert lat.elements["oc"].multipole.Bn[3] == pytest.approx(4.0 * brho)
    assert rep.ok


def test_element_aperture(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q1: quadrupole, l=0.2, k1=0.3, apertype=ellipse, aperture={0.01,0.02};
q2: quadrupole, l=0.2, k1=0.3, apertype=rectangle, aperture={0.03,0.04};
q3: quadrupole, l=0.2, k1=0.3, apertype=circle, aperture={0.05};
q4: quadrupole, l=0.2, k1=0.3;
s: sequence, l=4.0, refer=entry;
  q1, at=0.0; q2, at=0.5; q3, at=1.0; q4, at=1.5;
endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    assert lat.elements["q1"].aperture.shape == "ELLIPTICAL"
    assert (lat.elements["q1"].aperture.half_x, lat.elements["q1"].aperture.half_y) == \
        pytest.approx((0.01, 0.02))
    assert lat.elements["q2"].aperture.shape == "RECTANGULAR"
    assert lat.elements["q3"].aperture.half_x == pytest.approx(0.05)
    assert lat.elements["q4"].aperture is None


def test_unsupported_apertype_is_lossy(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q1: quadrupole, l=0.2, k1=0.3, apertype=racetrack, aperture={0.01,0.02,0.003,0.004};
s: sequence, l=1.0; q1, at=0.5; endsequence;
use, sequence=s;
""")
    lat, rep = Reader().read(p)
    assert "APERTYPE_UNSUPPORTED" in rep.codes()
    assert lat.elements["q1"].aperture.shape == "RECTANGULAR"
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


# ------------------------------------------------------------------------ errors
def test_ealign_becomes_a_body_shift(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q1: quadrupole, l=0.2, k1=0.3;
q2: quadrupole, l=0.2, k1=0.3;
s: sequence, l=2.0, refer=entry; q1, at=0.0; q2, at=1.0; endsequence;
use, sequence=s;
select, flag=error, clear;
select, flag=error, range=q1;
ealign, dx=0.001, dy=0.002, ds=0.0005, dphi=0.0001, dtheta=0.0002, dpsi=0.0003;
""")
    lat, _ = Reader().read(p)
    sh = lat.elements["q1"].shift
    assert (sh.x_offset, sh.y_offset, sh.z_offset) == pytest.approx((0.001, 0.002, 0.0005))
    assert (sh.x_rot, sh.y_rot, sh.tilt) == pytest.approx((0.0001, 0.0002, 0.0003))
    assert lat.elements["q2"].shift is None


def test_efcomp_is_recorded_and_lossy(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q1: quadrupole, l=0.2, k1=0.3;
s: sequence, l=1.0; q1, at=0.5; endsequence;
use, sequence=s;
select, flag=error, clear;
select, flag=error, range=q1;
efcomp, order=1, radius=0.01, dknr={0,0.01};
""")
    lat, rep = Reader().read(p)
    assert "EFCOMP_DROPPED" in rep.codes()
    assert lat.elements["q1"].native["madx"]["field_errors"]["dkn"][1] != 0.0
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


# ---------------------------------------------------------------------- dipedge
_DIPEDGE_DECK = """
beam, particle=proton, energy=1.738272;
de1: dipedge, h=0.1, e1=0.05, fint=0.5, hgap=0.02;
de2: dipedge, h=0.1, e1=0.04, fint=0.5, hgap=0.02;
bb:  sbend, l=1.0, angle=0.1{extra};
s: sequence, l=3.0, refer=entry;
  de1, at=1.0; bb, at=1.0; de2, at=2.0;
endsequence;
use, sequence=s;
"""


def test_dipedge_folds_into_the_adjacent_bend(tmp_path):
    lat, rep = Reader().read(_deck(tmp_path, _DIPEDGE_DECK.format(extra="")))
    b = lat.elements["bb"]
    assert b.bend.e1 == pytest.approx(0.05)
    assert b.bend.e2 == pytest.approx(0.04)
    assert b.bend.edge_int1 == pytest.approx(0.5)
    assert b.bend.hgap == pytest.approx(0.02)
    assert rep.codes()["DIPEDGE_FOLDED"] == 2
    assert not any(isinstance(e, Directive) for e in lat.elements.values())
    assert rep.ok


def test_dipedge_that_cannot_fold_is_lossy(tmp_path):
    """A bend that already carries fint/e1 would double-count the fringe."""
    p = _deck(tmp_path, _DIPEDGE_DECK.format(extra=", e1=0.01, e2=0.01, fint=0.4, hgap=0.03"))
    lat, rep = Reader().read(p)
    assert rep.codes()["DIPEDGE_UNFOLDED"] == 2
    assert isinstance(lat.elements["de1"], Directive)
    assert lat.elements["de1"].card == "dipedge"
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


# ------------------------------------------------------------ unsupported / misc
def test_unsupported_type_is_dropped_and_strict_raises(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
cc: crabcavity, l=0.0, volt=1.0;
s: sequence, l=1.0; cc, at=0.5; endsequence;
use, sequence=s;
""")
    lat, rep = Reader().read(p)
    assert isinstance(lat.elements["cc"], Marker)
    entry = next(e for e in rep.entries if e.code == "UNSUPPORTED_MADX_TYPE")
    assert entry.details["madx_type"] == "crabcavity"
    assert lat.elements["cc"].native["madx"]["base"] == "crabcavity"
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


def test_sequence_choice_and_ambiguity_warning(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q: quadrupole, l=0.2, k1=0.3;
sa: sequence, l=1.0; q, at=0.5; endsequence;
sb: sequence, l=2.0; q, at=0.5; endsequence;
""")
    lat, _ = Reader().read(p)                       # neither is USEd -> last, with a warning
    assert lat.meta["madx_sequence"] == "sb"
    assert any("candidate sequences" in w for w in lat.warnings)
    lat_a, _ = Reader().read(p, sequence="sa")
    assert lat_a.meta["madx_sequence"] == "sa"
    assert lat_a.total_length == pytest.approx(1.0)
    with pytest.raises(KeyError):
        Reader().read(p, sequence="nope")


def test_repeated_element_shares_one_definition(fodo):
    lat, _ = fodo
    placed = [p for p in lat.flatten() if p.name == "b1"]
    assert len(placed) == 2
    assert placed[0].element is placed[1].element
    assert "b1_2" not in lat.elements


def test_name_tag_is_read_back_into_provenance(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
q_f: quadrupole, l=0.2, k1=0.3;   ! lattix: name="Q/F#1" type="QUAD"
s: sequence, l=1.0; q_f, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    prov = lat.elements["q_f"].provenance
    assert prov.original_name == "Q/F#1"
    assert prov.original_type == "QUAD"
    assert prov.format == "madx"


def test_call_chain_is_resolved_by_madx(tmp_path):
    (tmp_path / "sub.madx").write_text("qf: quadrupole, l=0.3, k1=0.6;\n")
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
call, file="sub.madx";
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    assert lat.elements["qf"].gradient == pytest.approx(0.6 * lat.reference.brho_signed)


def test_missing_cpymad_raises_a_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cpymad.madx", None)
    with pytest.raises(RuntimeError, match=r"MAD-X reader needs cpymad"):
        Reader().read(FODO)


# ------------------------------- remaining downgrade codes, both regimes
_DOWNGRADES = {
    "BEND_K0_NE_ANGLE": "bb: sbend, l=1.0, angle=0.1, k0=0.5;",
    "BEND_ATTR_DROPPED": "bb: sbend, l=1.0, angle=0.1, h1=0.2, k3=0.4;",
    "SOLENOID_KSI_DROPPED": "bb: solenoid, l=1.0, ks=0.2, ksi=0.7;",
}


@pytest.mark.parametrize("code", sorted(_DOWNGRADES))
def test_remaining_downgrades_in_both_regimes(code, tmp_path):
    p = _deck(tmp_path, f"""
beam, particle=proton, energy=1.738272;
{_DOWNGRADES[code]}
s: sequence, l=2.0; bb, at=1.0; endsequence;
use, sequence=s;
""", f"{code.lower()}.madx")
    lat, rep = Reader().read(p)
    assert code in rep.codes(), rep.summary()
    assert not rep.ok
    assert lat.elements["bb"].native["madx"]          # the raw MAD-X value is kept
    with pytest.raises(TranslationError) as exc:
        Reader().read(p, strict=True)
    assert code in str(exc.value)


def test_frequency_option_sets_the_rf_clock(tmp_path):
    p = _deck(tmp_path, """
beam, particle=proton, energy=1.738272;
qf: quadrupole, l=0.3, k1=0.6;
s: sequence, l=1.0; qf, at=0.5; endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(p)
    assert lat.reference.rf_frequency_Hz is None
    lat, _ = Reader().read(p, frequency_Hz=352.21e6)
    assert lat.reference.rf_frequency_Hz == 352.21e6 and lat.reference.kinetic_energy_eV == pytest.approx(8e8, rel=1e-6)
