"""SciBmad (Beamlines.jl) reader: the Julia subset, tags, Bmad's own converter output, round trips."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.formats.scibmad import Reader
from lattix.formats.scibmad.reader import parse_kwargs, parse_map_function, parse_value, split_top_level
from lattix.ir.elements import Taylor
from lattix.ir.reference import ReferenceParticle, species
from lattix.testing import codes_path
from tests.formats.test_impactx_writer import all_kinds_lattice

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
BMAD_REFERENCE = codes_path("tier3_peers", "bmad-ecosystem", "regression_tests", "write_foreign_test",
                            "scibmad.correct")


# ------------------------------------------------------------------ tokenizer
def test_split_top_level_respects_brackets_and_strings():
    assert split_top_level('a, Species("x, y"), f(1, 2), [3, 4]') == ["a", 'Species("x, y")', "f(1, 2)", "[3, 4]"]


def test_parse_value_kinds():
    assert parse_value("1.5e-3").kind == "number" and parse_value("1.5e-3").value == 1.5e-3
    assert parse_value("2*pi").value == pytest.approx(2 * math.pi)
    assert parse_value("true").value is True
    assert parse_value("ApertureShape.Rectangular").kind == "enum"
    assert parse_value('Species("proton")').kind == "call"
    assert parse_value("lattix_map_q").kind == "ident"
    pos, kw = parse_kwargs("L = 0.5, Kn1 = -0.36; x_offset = 1e-3")
    assert pos == [] and kw["L"].value == 0.5 and kw["Kn1"].value == -0.36 and kw["x_offset"].value == 1e-3


def test_parse_map_function_reads_lattix_and_bmad_shapes():
    ours = ["function f(v, q, p=nothing)", "    v1 = 1*v[1] + 0.02*v[2]", "    v2 = 0.7*v[1] + 1*v[2] + 0.001",
            "    v3 = 1*v[3]", "    v4 = 1*v[4]", "    v5 = 1*v[5]", "    v6 = 1*v[6]",
            "    return (v1, v2, v3, v4, v5, v6), q", "end"]
    m, o, trunc = parse_map_function(ours)
    assert m[0][1] == 0.02 and m[1][0] == 0.7 and m[1][1] == 1.0 and o[1] == 0.001 and not trunc
    bmad = ["function map_match1(v, q)", "  v_out1= ", "      1.5650E+00*v[1] +", "     -7.3485E-02*v[2]",
            "  v_out2= ", "      7.3485E-02*v[1] +", "      6.8042E-01*v[2] +", "     -1.8689E-01*v[1]^2",
            "  q_out0 = 1.0", "  return (v_out1, v_out2), (q_out1)", "end"]
    m, o, trunc = parse_map_function(bmad)
    assert m[0][0] == pytest.approx(1.565) and m[0][1] == pytest.approx(-0.073485)
    assert m[1][0] == pytest.approx(0.073485) and m[1][1] == pytest.approx(0.68042)
    assert trunc                                   # v[1]^2 is second order: noted and dropped


# ------------------------------------------------------------------ small decks
def _read_text(tmp_path: Path, text: str, name: str = "deck.jl", **kw):
    p = tmp_path / name
    p.write_text(text)
    return Reader().read(p, **kw)


def test_minimal_beamline(tmp_path):
    lat, rep = _read_text(tmp_path, """
using Beamlines
@elements begin
  qf = Quadrupole(L = 0.5, Kn1 = 0.36)
  d = Drift(L = 1.2)
  qd = Quadrupole(L = 0.5, Kn1 = -0.36)
end
fodo = Beamline([qf, d, qd, d]; E_ref = 18e9, species_ref = Species("electron"))
""")
    assert rep.ok
    assert lat.use == "fodo" and [p.element.name for p in lat.flatten()] == ["qf", "d", "qd", "d"]
    assert lat.reference.species.name == "electron"
    assert lat.reference.total_energy_eV == pytest.approx(18e9)
    brho = lat.reference.brho_signed
    assert lat.elements["qf"].multipole.Bn[1] == pytest.approx(0.36 * brho)
    assert lat.total_length == pytest.approx(3.4)


def test_vectors_splats_reverse_and_repeat(tmp_path):
    lat, rep = _read_text(tmp_path, """
@elements begin
  a = Drift(L = 1)
  b = Quadrupole(L = 0.2, Kn1 = 1)
end
cell = [a, b]
ring = Beamline([cell..., reverse(cell)..., repeat(cell, 2)...]; pc_ref = 1e9, species_ref = Species("proton"))
""")
    names = [p.element.name for p in lat.flatten()]
    assert names == ["a", "b", "b", "a", "a", "b", "a", "b"]
    assert lat.lines["cell"].items[0].ref == "a"


def test_reference_from_pc_ref_p_over_q_ref_and_tag(tmp_path):
    pc = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6).pc_eV
    lat, _ = _read_text(tmp_path, f'@elements begin\n d = Drift(L = 1)\nend\nl = Beamline([d]; pc_ref = {pc:.12g},'
                                  ' species_ref = Species("proton"))\n')
    assert lat.reference.kinetic_energy_eV == pytest.approx(2.1e6, rel=1e-9)
    lat, _ = _read_text(tmp_path, '@elements begin\n d = Drift(L = 1)\nend\nl = Beamline([d]; p_over_q_ref = 0.2,'
                                  ' species_ref = Species("proton"))\n', name="b.jl")
    assert lat.reference.brho_signed == pytest.approx(0.2)
    lat, rep = _read_text(tmp_path, '# lattix: reference species="h-" mass_eV=939294074.05 charge=-1 '
                                    'kinetic_energy_eV=2100000 rf_frequency_Hz=162500000\n'
                                    '@elements begin\n d = Drift(L = 1)\nend\nl = Beamline([d])\n', name="c.jl")
    assert lat.reference.species.name == "h-" and lat.reference.rf_frequency_Hz == 162.5e6
    assert "REFERENCE_FROM_TAG" in rep.codes()


_MARKER_DECK = ('using Beamlines\n@elements begin\n'
                '  lat_begin = Marker(species_ref = Species("#1H-"), E_ref = 941391806.25)\n'
                '  q1 = Quadrupole(L = 0.3, Kn1 = -2.0)\nend\n'
                'lattice = Beamline([lat_begin, q1])\n')


def test_reference_carried_on_a_leading_marker(tmp_path):
    """The other carrier Beamlines.jl allows: a leading ``Marker`` holding the reference, which is the
    form HELIX's SciBmad examples and Bmad's own converter emit.  Without it the deck reads as the
    assumed proton at 1 GeV and every normalised strength is denormalised at the wrong rigidity."""
    lat, rep = _read_text(tmp_path, _MARKER_DECK)
    assert lat.reference.species.name == "h-"
    assert lat.reference.kinetic_energy_eV == pytest.approx(941391806.25 - lat.reference.species.mass_eV,
                                                            rel=1e-12)
    assert "SPECIES_ASSUMED" not in rep.codes() and "ENERGY_ASSUMED" not in rep.codes()
    # a negative Kn1 on a negative charge is a positive gradient: the signed rigidity did the work
    assert lat.elements["q1"].multipole.Bn[1] == pytest.approx(2.0 * abs(lat.reference.brho_signed), rel=1e-12)


def test_read_options_win_over_the_marker_field_by_field(tmp_path):
    """``species=``/``kinetic_energy_eV=`` override the carrier one field at a time: what the caller
    did not give still comes from the marker."""
    lat, _ = _read_text(tmp_path, _MARKER_DECK, name="sp.jl", species="proton")
    assert lat.reference.species.name == "proton"                       # option
    assert lat.reference.kinetic_energy_eV == pytest.approx(941391806.25 - species("proton").mass_eV,
                                                            rel=1e-12)  # marker
    lat, _ = _read_text(tmp_path, _MARKER_DECK, name="ke.jl", kinetic_energy_eV=5e6)
    assert lat.reference.species.name == "h-"                           # marker
    assert lat.reference.kinetic_energy_eV == pytest.approx(5e6)        # option


def test_beamline_keywords_win_over_the_marker(tmp_path):
    lat, _ = _read_text(tmp_path, _MARKER_DECK.replace("Beamline([lat_begin, q1])",
                                                       'Beamline([lat_begin, q1]; E_ref = 1738272000.0,'
                                                       ' species_ref = Species("proton"))'))
    assert lat.reference.species.name == "proton"
    assert lat.reference.kinetic_energy_eV == pytest.approx(799999911.84, rel=1e-12)


def test_bend_field_and_fringe_from_the_file(tmp_path):
    lat, rep = _read_text(tmp_path, """
@elements begin
  # lattix: name="B.1" hgap="0.03" rect="true"
  b_1 = SBend(L = 1, g_ref = 0.1, Kn0 = 0.1, e1 = 0.05, e2 = 0.05, edge1_int = 0.0135, edge2_int = 0.015,
              tilt_ref = 0.2)
  b2 = SBend(L = 1, g_ref = 0.1, Kn0 = 0.11)
end
l = Beamline([b_1, b2]; pc_ref = 1e9, species_ref = Species("proton"))
""")
    b = lat.elements["b_1"]                         # the Julia name; the original lives in the provenance
    assert b.kind == "Bend" and b.bend.angle == pytest.approx(0.1) and b.bend.rect
    assert b.bend.hgap == 0.03 and b.bend.edge_int1 == pytest.approx(0.45) and b.bend.edge_int2 == pytest.approx(0.5)
    assert b.bend.tilt_ref == 0.2 and b.provenance.original_name == "B.1"
    assert "BEND_K0_NE_G" in rep.codes() and lat.elements["b2"].native["scibmad"]["k0"] == pytest.approx(0.11)


def test_cavity_phase_conventions(tmp_path):
    lat, rep = _read_text(tmp_path, """
@elements begin
  # lattix: name="c1" n_cell="3" L_active="0.2"
  c1 = RFCavity(L = 0.3, voltage = -300000, phi0 = -0.5235987755982988, rf_frequency = 162500000, traveling_wave = true)
  c2 = RFCavity(L = 0.3, voltage = 300000, phi0 = 0, rf_frequency = 162500000, zero_phase = PhaseRef.AboveTransition)
end
l = Beamline([c1, c2]; pc_ref = 62807919.6, species_ref = Species("proton"))
""")
    c1 = lat.elements["c1"]
    assert c1.rf.voltage_V == pytest.approx(3e5) and c1.rf.phase_rad == pytest.approx(-math.pi / 6)
    assert c1.rf.cavity_type == "TRAVELING_WAVE" and c1.rf.n_cell == 3
    assert lat.elements["c2"].rf.phase_rad == pytest.approx(math.pi / 2)     # zero crossing -> crest convention
    assert "ZERO_PHASE_SHIFTED" in rep.codes()


def test_unknown_kind_is_a_dropped_marker_not_a_crash(tmp_path):
    lat, rep = _read_text(tmp_path, '@elements begin\n w = Wiggler(L = 2, B_max = 1)\nend\n'
                                    'l = Beamline([w]; pc_ref = 1e9, species_ref = Species("proton"))\n')
    assert lat.elements["w"].kind == "Drift" and lat.elements["w"].length == 2.0
    assert "UNSUPPORTED_SCIBMAD_KIND" in rep.codes()


# ------------------------------------------------------------------ round trips
def _same(a, b, rel=1e-11):
    return a == pytest.approx(b, rel=rel, abs=1e-14)


def test_all_kinds_round_trip(tmp_path):
    lat = all_kinds_lattice()
    out = tmp_path / "all.jl"
    write(lat, out, "scibmad")
    back, rep = read(out, "scibmad")
    a = {p.element.name: p.element for p in lat.flatten()}
    b = {p.element.name: p.element for p in back.flatten()}
    assert set(a) <= set(b) | {n for n in a if a[n].kind in ("Directive", "Superposition")}
    assert back.total_length == pytest.approx(lat.total_length, abs=1e-12)
    for name, e in a.items():
        if name not in b:
            continue
        e2 = b[name]
        if e.kind in ("FieldMap", "NCells", "RFQCell", "Foil", "Directive", "Superposition"):
            continue
        assert e2.kind == e.kind, (name, e.kind, e2.kind)
        assert _same(e2.length, e.length)
        mp, mp2 = getattr(e, "multipole", None), getattr(e2, "multipole", None)
        if mp is not None and mp2 is not None:
            for table in ("Bn", "Bs", "BnL", "BsL"):
                for k, v in getattr(mp, table).items():
                    if v:
                        assert _same(getattr(mp2, table).get(k, 0.0), v), (name, table, k)
        if e.kind == "Bend":
            assert _same(e2.bend.angle, e.bend.angle) and _same(e2.bend.e1, e.bend.e1)
        if e.kind == "RFCavity":
            assert _same(e2.rf.voltage_V, e.rf.voltage_V) and _same(e2.rf.phase_rad, e.rf.phase_rad)
        if e.kind == "Kicker":
            assert _same(e2.hkick, e.hkick) and _same(e2.vkick, e.vkick)
        if e.kind == "Solenoid":
            assert _same(e2.solenoid.Bsol_T, e.solenoid.Bsol_T)
        if e.kind == "Taylor":
            assert isinstance(e2, Taylor)
            for i in range(6):
                for j in range(6):
                    assert _same(e2.matrix[i][j], e.matrix[i][j])
        if e.aperture is not None and e.aperture.x_limits is not None:
            assert e2.aperture is not None and e2.aperture.shape == e.aperture.shape
            assert e2.aperture.x_limits == pytest.approx(e.aperture.x_limits)
        if e.shift is not None:
            assert e2.shift is not None and _same(e2.shift.x_offset, e.shift.x_offset)
    # write -> read -> write is a fixed point
    out2 = tmp_path / "all2.jl"
    write(back, out2, "scibmad")

    def physics(text: str) -> list[str]:
        # the header names the source format; directives are comments, not elements, so they do
        # not come back
        return [ln for ln in text.splitlines()
                if ln.strip() and not ln.startswith(("# lattix 0", "# lattix directive:"))]

    assert physics(out2.read_text()) == physics(out.read_text())


def test_fodo_round_trip_is_exact(tmp_path):
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.jl"
    write(lat, out, "scibmad")
    back, rep = read(out, "scibmad")
    assert rep.ok
    for n in ("qf", "qd"):
        assert back.elements[n].multipole.Bn[1] == pytest.approx(lat.elements[n].multipole.Bn[1], rel=1e-12)
    assert back.elements["b1"].bend.angle == pytest.approx(0.1)
    assert back.reference.kinetic_energy_eV == pytest.approx(lat.reference.kinetic_energy_eV, rel=1e-12)


@pytest.mark.skipif(not BMAD_REFERENCE.exists(), reason="Bmad's write_foreign_test/scibmad.correct not available")
def test_bmads_own_converter_output_parses():
    """The reference file Bmad's ``bmad_to_scibmad`` writes for its all-elements test lattice."""
    try:
        BMAD_REFERENCE.read_text()
    except PermissionError:
        pytest.skip("Bmad clone not readable here")
    lat, rep = Reader().read(BMAD_REFERENCE)
    names = [p.element.name for p in lat.flatten()]
    assert len(names) == 52 and names[0] == "ab_multipole1" and names[-1] == "end_b0"
    assert lat.reference.species.name == "positron"
    assert lat.reference.pc_eV == pytest.approx(8.5958e5, rel=1e-4)
    kinds = {p.element.name: p.element.kind for p in lat.flatten()}
    assert kinds["quadrupole1"] == "Quadrupole" and kinds["sbend5"] == "Bend" and kinds["solenoid1"] == "Solenoid"
    assert kinds["match1"] == "Taylor" and kinds["taylor1"] == "Taylor" and kinds["patch1"] == "Patch"
    assert kinds["rcollimator1"] == "Drift" and lat.elements["rcollimator1"].aperture.shape == "RECTANGULAR"
    assert "TAYLOR_ORDER_TRUNCATED" in rep.codes()          # taylor1 has second-order terms
    assert lat.elements["sbend5"].bend.angle == pytest.approx(0.006)
