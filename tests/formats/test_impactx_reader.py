"""ImpactX reader tests (PLAN §6 task 3.2): the AMReX ``inputs`` grammar, element
mapping, dipedge clustering, the rigidity walk, downgrades in both regimes, and the
write ∘ read fixed point (invariant I-13).

The vendored decks under ``tests/data/public/impactx`` come from the ImpactX repository
(BSD-3-Clause-LBNL).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.impactx import Reader, Writer
from lattix.formats.impactx.reader import ImpactxParseError, parse_inputs
from lattix.ir.reference import ReferenceParticle, species

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "impactx"


def write_deck(tmp_path: Path, body: str, name: str = "x.impactx.in") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


BEAM = """\
beam.npart = 1000
beam.units = static
beam.kin_energy = 2.1
beam.charge = 1.0e-12
beam.particle = proton
beam.distribution = waterbag
beam.lambdaX = 1.0e-4
beam.lambdaY = beam.lambdaX
beam.lambdaT = 1.0e-4
beam.lambdaPx = 1.0e-4
beam.lambdaPy = beam.lambdaPx
beam.lambdaPt = 1.0e-4
beam.muxpx = -0.5
beam.muypy = -beam.muxpx
"""


# ----------------------------------------------------------------- grammar
def test_parse_inputs_handles_comments_and_continuations():
    values, order = parse_inputs(
        "# a comment\n"
        "lattice.elements = a b   \\\n"
        "                   c d\n"
        "a.type = drift   # trailing comment\n"
        "a.ds  =  0.5\n"
        "\n"
    )
    assert values["lattice.elements"] == ["a", "b", "c", "d"]
    assert values["a.type"] == ["drift"]
    assert values["a.ds"] == ["0.5"]
    assert order[0] == "lattice.elements"


def test_parse_inputs_rejects_garbage():
    with pytest.raises(ImpactxParseError):
        parse_inputs("this is not an assignment\n")


def test_value_references_resolve():
    values, _ = parse_inputs("a.x = 3\na.y = a.x\na.z = -a.x\n")
    from lattix.formats.impactx.reader import _float

    assert _float(values["a.y"], values, "a.y") == 3.0
    assert _float(values["a.z"], values, "a.z") == -3.0


# ------------------------------------------------------------ beam/reference
def test_beam_block_is_kept_for_round_trip(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = d\nd.type = drift\nd.ds = 1.0\n")
    lat, rep = Reader().read(p)
    assert lat.reference.species.name == "proton"
    assert lat.reference.kinetic_energy_eV == pytest.approx(2.1e6)
    beam = lat.meta["impactx_beam"]
    assert beam["npart"] == 1000
    assert beam["lambdaY"] == pytest.approx(1e-4)      # resolved reference
    assert beam["muypy"] == pytest.approx(0.5)         # -beam.muxpx
    assert rep.ok


def test_missing_beam_energy_is_lossy(tmp_path):
    p = write_deck(tmp_path, "lattice.elements = d\nd.type = drift\nd.ds = 1.0\n")
    lat, rep = Reader().read(p)
    assert "NO_BEAM_ENERGY" in rep.codes()
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)
    # an explicit reference silences it
    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=1e8)
    lat2, rep2 = Reader().read(p, reference=ref)
    assert lat2.reference.species.name == "h-"
    assert "NO_BEAM_ENERGY" not in rep2.codes()


def test_unknown_species_is_lossy(tmp_path):
    p = write_deck(tmp_path, "beam.kin_energy = 5\nbeam.particle = muon\n"
                             "lattice.elements = d\nd.type = drift\nd.ds = 1.0\n")
    _, rep = Reader().read(p)
    assert "UNKNOWN_SPECIES" in rep.codes()


def test_python_script_is_not_parsed(tmp_path):
    p = tmp_path / "deck.impactx.py"
    p.write_text("from impactx import ImpactX, elements\nsim = ImpactX()\n")
    with pytest.raises(TranslationError) as exc:
        Reader().read(p)
    assert exc.value.entry.code == "IMPACTX_PYTHON_NOT_READ"


# --------------------------------------------------------------- elements
def test_quad_and_solenoid_use_signed_rigidity(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = q s
q.type = quad
q.ds = 0.3
q.k = 2.5
s.type = solenoid
s.ds = 0.4
s.ks = 0.8
""")
    lat, rep = Reader().read(p)
    b = lat.reference.brho_signed
    q = lat.elements["q"]
    s = lat.elements["s"]
    assert q.multipole.Bn[1] == pytest.approx(2.5 * b)
    assert s.solenoid.Bsol_T == pytest.approx(0.8 * b)
    assert q.tracking["nslice"] == 1
    assert rep.ok


def test_quad_with_unit_1_is_a_lab_gradient(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = q
q.type = quad_chromatic
q.ds = 0.05
q.k = 11.4
q.units = 1
""")
    lat, rep = Reader().read(p)
    assert lat.elements["q"].multipole.Bn[1] == pytest.approx(11.4)
    assert "EXACT_QUAD_MODEL" in rep.codes()


def test_dipedge_clustering(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = e1 b e2
e1.type = dipedge
e1.psi = 0.05
e1.rc = 10.0
e1.g = 0.04
e1.K2 = 0.45
e1.location = entry
b.type = sbend
b.ds = 1.0
b.rc = 10.0
e2.type = dipedge
e2.psi = 0.06
e2.rc = 10.0
e2.g = 0.04
e2.K2 = 0.5
e2.location = exit
""")
    lat, rep = Reader().read(p)
    placed = lat.flatten()
    assert [pl.element.kind for pl in placed] == ["Bend"]
    b = placed[0].element
    assert b.bend.angle == pytest.approx(0.1)
    assert b.bend.e1 == pytest.approx(0.05)
    assert b.bend.e2 == pytest.approx(0.06)
    assert b.bend.edge_int1 == pytest.approx(0.45)
    assert b.bend.edge_int2 == pytest.approx(0.5)
    assert b.bend.hgap == pytest.approx(0.02)
    assert "DIPEDGE_FOLDED" in rep.codes()


def test_orphan_dipedge_is_lossy(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = e1 d
e1.type = dipedge
e1.psi = 0.05
e1.rc = 10.0
e1.g = 0.0
e1.K2 = 0.0
d.type = drift
d.ds = 1.0
""")
    lat, rep = Reader().read(p)
    assert "DIPEDGE_ORPHAN" in rep.codes()
    assert lat.elements["e1"].kind == "Marker"
    assert lat.elements["e1"].native["impactx"]["psi"] == pytest.approx(0.05)
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


def test_shortrf_phase_and_voltage(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = c
c.type = shortrf
c.V = 0.0010657889233488304
c.freq = 162.5e6
c.phase = -30.0
""")
    lat, _ = Reader().read(p)
    c = lat.elements["c"]
    assert c.rf.voltage_V == pytest.approx(1.0e6, rel=1e-8)   # ImpactX m_p differs at 1.4e-9
    assert c.rf.phase_rad == pytest.approx(math.radians(-30.0))
    assert c.rf.frequency_Hz == pytest.approx(162.5e6)


def test_shortrf_default_phase_is_minus_90(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = c\nc.type = shortrf\n"
                                    "c.V = 1e-3\nc.freq = 1e8\n")
    lat, _ = Reader().read(p)
    assert lat.elements["c"].rf.phase_rad == pytest.approx(-math.pi / 2)


def test_rigidity_walk_follows_shortrf(tmp_path):
    """A quad after an accelerating gap sees a larger rigidity, so the same k is a
    larger lab gradient (ImpactX follows the reference energy)."""
    p = write_deck(tmp_path, BEAM + """
lattice.elements = q1 c q2
q1.type = quad
q1.ds = 0.1
q1.k = 5.0
c.type = shortrf
c.V = 0.0010657889233488304
c.freq = 162.5e6
c.phase = 0.0
q2.type = quad
q2.ds = 0.1
q2.k = 5.0
""")
    lat, _ = Reader().read(p)
    g1 = lat.elements["q1"].multipole.Bn[1]
    g2 = lat.elements["q2"].multipole.Bn[1]
    assert g2 > g1
    ke_out = lat.reference.kinetic_energy_eV + 1.0e6
    ref2 = ReferenceParticle(species=lat.reference.species, kinetic_energy_eV=ke_out)
    assert g2 == pytest.approx(5.0 * ref2.brho_signed, rel=1e-9)


def test_multipole_kicker_aperture_monitor(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = m k a mon
m.type = multipole
m.multipole = 3
m.k_normal = 0.4
m.k_skew = 0.1
k.type = kicker
k.xkick = 2.0e-3
k.ykick = 3.0e-3
a.type = aperture
a.aperture_x = 0.02
a.aperture_y = 0.03
a.shape = elliptical
mon.type = beam_monitor
mon.backend = h5
""")
    lat, rep = Reader().read(p)
    b = lat.reference.brho_signed
    assert lat.elements["m"].multipole.BnL[2] == pytest.approx(0.4 * b)
    assert lat.elements["m"].multipole.BsL[2] == pytest.approx(0.1 * b)
    assert lat.elements["k"].hkick == pytest.approx(2e-3)
    ap = lat.elements["a"].aperture
    assert ap.shape == "ELLIPTICAL"
    assert ap.half_x == pytest.approx(0.02)
    assert lat.elements["mon"].kind == "Instrument"
    assert rep.ok


def test_kicker_in_tesla_metres(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = k\nk.type = kicker\n"
                                    "k.xkick = 0.01\nk.ykick = 0\nk.units = T-m\n")
    lat, _ = Reader().read(p)
    assert lat.elements["k"].hkick == pytest.approx(0.01 / lat.reference.brho_signed)


def test_linear_map_sign_conjugation(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = m\nm.type = linear_map\n"
                                    "m.ds = 0.5\nm.R12 = 0.5\nm.R16 = 0.25\nm.R56 = 1.5\n")
    lat, rep = Reader().read(p)
    t = lat.elements["m"]
    assert t.kind == "Taylor"
    assert t.matrix[0][1] == pytest.approx(0.5)      # transverse block unchanged
    assert t.matrix[0][5] == pytest.approx(-0.25)    # dispersion column flips
    assert t.matrix[4][5] == pytest.approx(1.5)      # R56 is sign invariant
    assert "TAYLOR_TIME_SIGN" in rep.codes()


def test_unsupported_type_is_lossy(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = n\nn.type = nonlinear_lens\n"
                                    "n.knll = 1e-3\nn.cnll = 0.01\n")
    lat, rep = Reader().read(p)
    assert "UNSUPPORTED_TYPE" in rep.codes()
    assert lat.elements["n"].native["impactx"]["knll"] == pytest.approx(1e-3)
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


def test_undefined_element_is_dropped(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = d ghost\nd.type = drift\nd.ds = 1\n")
    lat, rep = Reader().read(p)
    assert "UNDEFINED_ELEMENT" in rep.codes()
    assert list(lat.elements) == ["d"]


# ------------------------------------------------------------------ lines
def test_line_periods_and_reverse_are_expanded(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = sub d3
lattice.periods = 2
sub.type = line
sub.elements = d1 d2
sub.repeat = 2
d1.type = drift
d1.ds = 0.1
d2.type = drift
d2.ds = 0.2
d3.type = drift
d3.ds = 0.3
""")
    lat, rep = Reader().read(p)
    names = [pl.name for pl in lat.flatten()]
    assert names == ["d1", "d2", "d1", "d2", "d3"] * 2
    assert lat.total_length == pytest.approx(2 * (0.1 + 0.2) * 2 + 2 * 0.3)
    assert "LINE_FLATTENED" in rep.codes()


def test_reverse_flag(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = a b\nlattice.reverse = true\n"
                                    "a.type = drift\na.ds = 1\nb.type = drift\nb.ds = 2\n")
    lat, _ = Reader().read(p)
    assert [pl.name for pl in lat.flatten()] == ["b", "a"]


# ------------------------------------------------------- vendored ImpactX decks
def test_vendored_input_fodo():
    deck = DATA / "input_fodo.in"
    if not deck.exists():
        pytest.skip(f"{deck} not vendored")
    lat, rep = Reader().read(deck)
    assert lat.reference.species.name == "electron"
    assert lat.reference.kinetic_energy_eV == pytest.approx(2.0e9)
    kinds = [pl.element.kind for pl in lat.flatten()]
    assert kinds.count("Instrument") == 6
    assert kinds.count("Quadrupole") == 2
    assert lat.total_length == pytest.approx(3.0)
    assert lat.meta["impactx_nslice"] == 25
    assert all(e.tracking.get("nslice") == 25
               for e in lat.elements.values() if e.length > 0)
    assert rep.ok


# ------------------------------------------------------------- fixed point
@pytest.mark.parametrize("nslice", [1, 4])
def test_write_read_write_is_a_fixed_point(tmp_path, nslice):
    from tests.formats.test_impactx_writer import all_kinds_lattice, demo_lattice, linac_lattice

    for i, lat in enumerate((demo_lattice(), linac_lattice(), all_kinds_lattice())):
        a = tmp_path / f"a{i}.impactx.in"
        Writer().write(lat, a, flavor="inputs", nslice=nslice)
        lat_b, _ = Reader().read(a)
        b = tmp_path / f"b{i}.impactx.in"
        Writer().write(lat_b, b, flavor="inputs", nslice=nslice)
        lat_c, _ = Reader().read(b)
        c = tmp_path / f"c{i}.impactx.in"
        Writer().write(lat_c, c, flavor="inputs", nslice=nslice)
        assert b.read_text() == c.read_text(), f"not a fixed point for lattice {i}"
        assert lat_b.total_length == pytest.approx(lat_c.total_length)


@pytest.mark.oracle_madx
def test_fodo_madx_round_trip_preserves_optics_parameters(tmp_path):
    pytest.importorskip("cpymad")
    from lattix.formats.madx import Reader as MadxReader

    src, _ = MadxReader().read(Path(__file__).resolve().parents[1] / "data" / "public"
                               / "helix" / "fodo.madx")
    p = tmp_path / "fodo.impactx.in"
    Writer().write(src, p, flavor="inputs")
    back, rep = Reader().read(p)
    assert back.total_length == pytest.approx(src.total_length)
    a = {pl.name: pl.element for pl in src.flatten()}
    b = {pl.name: pl.element for pl in back.flatten()}
    assert a["qf"].multipole.Bn[1] == pytest.approx(b["qf"].multipole.Bn[1], rel=1e-12)
    assert a["b1"].bend.angle == pytest.approx(b["b1"].bend.angle, rel=1e-12)
    assert a["b1"].bend.e1 == pytest.approx(b["b1"].bend.e1, rel=1e-12)
    assert rep.ok


def test_thick_rfcavity_gain_is_unknown(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = c
c.type = rfcavity
c.ds = 0.24
c.escale = 1.0e-3
c.freq = 162.5e6
c.phase = 43.8
c.cos_coefficients = 2.0 0.0
c.sin_coefficients = 0.0 0.0
""")
    lat, rep = Reader().read(p)
    assert "RFCAVITY_GAIN_UNKNOWN" in rep.codes()
    assert "RFCAVITY_FOURIER_PROFILE" in rep.codes()
    c = lat.elements["c"]
    assert c.rf.gradient_V_per_m == pytest.approx(1.0e-3 * lat.reference.species.mass_eV)
    assert c.native["impactx"]["cos_coefficients"] == [2.0, 0.0]
    assert c.rf.phase_is_sync is False
    with pytest.raises(TranslationError):
        Reader().read(p, strict=True)


def test_thin_dipole_is_lossy(tmp_path):
    p = write_deck(tmp_path, BEAM + "lattice.elements = t\nt.type = thin_dipole\n"
                                    "t.theta = 2.0\nt.rc = 10.0\n")
    lat, rep = Reader().read(p)
    assert "THIN_DIPOLE_AS_MULTIPOLE" in rep.codes()
    b = lat.reference.brho_signed
    assert lat.elements["t"].multipole.BnL[0] == pytest.approx(math.radians(2.0) * b)


def test_dipedge_extra_field_integrals_are_lossy(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = e1 b
e1.type = dipedge
e1.psi = 0.05
e1.rc = 10.0
e1.g = 0.04
e1.K2 = 0.45
e1.K0 = 1.6449340668482264
e1.model = nonlinear
e1.location = entry
b.type = sbend
b.ds = 1.0
b.rc = 10.0
""")
    _, rep = Reader().read(p)
    assert "DIPEDGE_FIELD_INTEGRALS" in rep.codes()
    assert "DIPEDGE_NONLINEAR_MODEL" in rep.codes()


def test_buncher_and_soft_solenoid_and_sbend_exact(tmp_path):
    p = write_deck(tmp_path, BEAM + """
lattice.elements = bu ss sx
bu.type = buncher
bu.V = 1.0e-4
bu.k = 3.4
ss.type = solenoid_softedge
ss.ds = 0.3
ss.bscale = 0.5
ss.unit = 1
sx.type = sbend_exact
sx.ds = 1.0
sx.phi = 5.0
""")
    lat, rep = Reader().read(p)
    for code in ("BUNCHER_AS_CAVITY", "SOFT_SOLENOID_HARD_EDGE", "EXACT_SBEND_MODEL"):
        assert code in rep.codes(), sorted(rep.codes())
    assert lat.elements["bu"].rf.phase_rad == pytest.approx(-math.pi / 2)
    assert lat.elements["ss"].solenoid.Bsol_T == pytest.approx(0.5)
    assert lat.elements["sx"].bend.angle == pytest.approx(math.radians(5.0))


def test_every_reader_downgrade_code_is_exercised():
    import inspect
    import re as _re

    from lattix.formats.impactx import reader as reader_mod

    src = inspect.getsource(reader_mod)
    codes = set(_re.findall(r'rep\.(?:lossy|dropped)\(\s*"?\n?\s*"([A-Z_]+)"', src))
    codes |= set(_re.findall(r'rep\.(?:lossy|dropped)\(\s*"([A-Z_]+)"', src))
    tested = {"UNKNOWN_SPECIES", "NO_BEAM_ENERGY", "IMPACTX_PYTHON_NOT_READ", "DIPEDGE_ORPHAN",
              "UNSUPPORTED_TYPE", "UNDEFINED_ELEMENT", "RFCAVITY_GAIN_UNKNOWN",
              "THIN_DIPOLE_AS_MULTIPOLE", "DIPEDGE_FIELD_INTEGRALS"}
    assert not codes - tested, f"untested reader downgrades: {sorted(codes - tested)}"
