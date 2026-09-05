"""xtrack JSON deck ``Reader``/``Writer``: goldens, sniffing, round trips, idempotence.

Golden snapshots live in ``tests/golden/xtrack``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_xtrack_writer.py``.

Normalisation (:func:`_norm`).  xtrack JSON carries no timestamps, and the payload is
byte-identical between xtrack 0.103.5 and 0.112.0 apart from the ``xtrack_version``
stamp — so the goldens keep everything lattix controls (``element_names``, ``elements``,
``metadata["lattix"]``) plus the physics of ``particle_ref``, and drop the engine's own
bookkeeping (``config``, ``_extra_config``, ``_var_manager``, ``_var_management_data``,
``env_particles``, ``mode``, ``xtrack_version``) together with the lattix version string.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

xt = pytest.importorskip("xtrack")

from lattix.fidelity import TranslationError  # noqa: E402
from lattix.formats.base import guess_format  # noqa: E402
from lattix.formats.xtrack import (  # noqa: E402
    METADATA_KEY,
    Reader,
    Writer,
    from_line,
    load_document,
    sniff_json,
    to_line,
)
from lattix.ir.elements import (  # noqa: E402
    RFP,
    ApertureP,
    Bend,
    BendP,
    Collimator,
    Drift,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    NCells,
    Quadrupole,
    RFCavity,
    Solenoid,
    SolenoidP,
)
from lattix.ir.lattice import Lattice  # noqa: E402
from lattix.ir.reference import ReferenceParticle, species  # noqa: E402
from tests.formats.test_xtrack_convert import all_kinds_lattice, proton_ref  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "xtrack"

#: engine bookkeeping that is not part of the lattice
_DROP = ("config", "_extra_config", "_var_manager", "_var_management_data",
         "env_particles", "mode", "xtrack_version", "__class__")
#: the physics of the reference particle (the rest of particle_ref is per-particle state)
_KEEP_PREF = ("mass0", "q0", "p0c", "beta0", "gamma0")


def _norm(text: str) -> str:
    d = json.loads(text)
    for k in _DROP:
        d.pop(k, None)
    pref = d.get("particle_ref")
    if isinstance(pref, dict):
        d["particle_ref"] = {k: _scalar(pref[k]) for k in _KEEP_PREF if k in pref}
    meta = (d.get("metadata") or {}).get(METADATA_KEY)
    if isinstance(meta, dict) and "version" in meta:
        meta["version"] = "<version>"
    return json.dumps(d, indent=1, sort_keys=True) + "\n"


def _scalar(v):
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def assert_golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = _norm(path.read_text())
    if os.environ.get("LATTIX_UPDATE_GOLDEN"):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert golden.exists(), f"missing golden {golden}; rerun with LATTIX_UPDATE_GOLDEN=1"
    assert text == golden.read_text()


def demo_lattice() -> Lattice:
    """FODO with two bends, a solenoid, a thin cavity, a kicker and a collimator."""
    ref = proton_ref()
    b = ref.brho_signed
    els = [
        Quadrupole(name="QF", length=0.3, multipole=MagneticMultipoleP(Bn={1: 0.6 * b})),
        Drift(name="D1", length=0.5),
        Bend(name="B1", length=1.0,
             bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.5, hgap=0.02)),
        Drift(name="D2", length=0.5),
        Solenoid(name="SOL", length=0.4, solenoid=SolenoidP(Bsol_T=0.3 * b)),
        Drift(name="D3", length=0.2),
        RFCavity(name="CAV", length=0.0,
                 rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6, frequency_Hz=325e6)),
        Drift(name="D4", length=0.2),
        Kicker(name="COR", length=0.0, hkick=1e-3, vkick=-2e-3),
        Drift(name="D5", length=0.3),
        Collimator(name="COLL", length=0.1, aperture=ApertureP.rect(0.02, 0.03)),
        Drift(name="D6", length=0.5),
        Bend(name="B2", length=1.0, bend=BendP(angle=-0.1, e1=-0.05, e2=-0.05, rect=True)),
        Drift(name="D7", length=0.5),
        Quadrupole(name="QD", length=0.3, multipole=MagneticMultipoleP(Bn={1: -0.6 * b})),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("demo", els, ref)


def linac_lattice() -> Lattice:
    """Three thin 300 kV gaps at -30 deg with a quad triplet, proton at 2.1 MeV."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6,
                            rf_frequency_Hz=162.5e6)
    b = ref.brho_signed
    els = []
    for i in range(3):
        sign = 1.0 if i % 2 == 0 else -1.0
        els += [
            Quadrupole(name=f"q{i + 1}", length=0.1,
                       multipole=MagneticMultipoleP(Bn={1: sign * 5.0 * b})),
            Drift(name=f"da{i + 1}", length=0.2),
            RFCavity(name=f"cav{i + 1}", length=0.0,
                     rf=RFP(voltage_V=3e5, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
            Drift(name=f"db{i + 1}", length=0.2),
        ]
    return Lattice.from_sequence("linac", els, ref)


# ----------------------------------------------------------------- goldens
def test_golden_demo(tmp_path):
    out = tmp_path / "demo.json"
    rep = Writer().write(demo_lattice(), out)
    assert rep.counts.get("DROPPED", 0) == 0
    assert_golden("demo.json", out)


def test_golden_all_kinds(tmp_path):
    out = tmp_path / "all_kinds.json"
    Writer().write(all_kinds_lattice(), out)
    assert_golden("all_kinds.json", out)


def test_golden_linac_local_and_constant(tmp_path):
    lat = linac_lattice()
    a = tmp_path / "linac_local.json"
    b = tmp_path / "linac_constant.json"
    Writer().write(lat, a, energy_mode="local")
    Writer().write(lat, b, energy_mode="constant")
    assert_golden("linac_local.json", a)
    assert_golden("linac_constant.json", b)
    assert a.read_text() != b.read_text()


@pytest.mark.oracle_madx
def test_golden_fodo_from_madx(tmp_path):
    pytest.importorskip("cpymad")
    from lattix.formats.madx import Reader as MadxReader

    lat, rep_in = MadxReader().read(DATA / "fodo.madx")
    out = tmp_path / "fodo_from_madx.json"
    rep = Writer().write(lat, out, strict=True)
    assert rep_in.ok and rep.ok
    assert_golden("fodo_from_madx.json", out)


# ------------------------------------------------------------------ sniffing
#: the registry row this package needs in ``lattix/formats/base.py``.  ``.json`` is a
#: prefix-free suffix only if it is tried AFTER ``.lattix.json`` and ``.pals.json``,
#: so the entry must be appended to FORMATS, never inserted before those two.
XTRACK_SPEC_KWARGS = dict(name="xtrack", suffixes=(".json",), module="lattix.formats.xtrack",
                          description="xtrack Line/Environment JSON")


def test_guess_format_is_ambiguous_for_plain_json():
    """The registry entry uses ``.json``; ``.lattix.json``/``.pals.json`` must win, which
    is an ordering property of FORMATS.  Content sniffing settles the rest."""
    assert guess_format("x.lattix.json") == "lattix"
    assert guess_format("x.pals.json") == "pals"


def test_the_proposed_registry_entry_dispatches_correctly(tmp_path, monkeypatch):
    """The FormatSpec line reported for ``lattix/formats/base.py`` really works: appended
    last it claims ``.json`` without stealing ``.lattix.json`` / ``.pals.json``, and
    ``read``/``write``/``translate`` reach this package's Reader/Writer."""
    from lattix.formats import base

    formats = dict(base.FORMATS)
    formats["xtrack"] = base.FormatSpec(**XTRACK_SPEC_KWARGS)
    monkeypatch.setattr(base, "FORMATS", formats)

    assert base.guess_format("deck.json") == "xtrack"
    assert base.guess_format("deck.lattix.json") == "lattix"
    assert base.guess_format("deck.pals.json") == "pals"
    assert isinstance(formats["xtrack"].reader(), Reader)
    assert isinstance(formats["xtrack"].writer(), Writer)
    assert base.check_rules_coverage(formats["xtrack"].writer()) == set()

    out = tmp_path / "deck.json"
    rep = base.write(demo_lattice(), out)
    assert rep.target_format == "xtrack"
    lat, rep_in = base.read(out)
    assert rep_in.source_format == "xtrack"
    assert lat.total_length == pytest.approx(demo_lattice().total_length, abs=1e-12)

    rep = base.translate(out, tmp_path / "again.json")
    assert rep.source_format == "xtrack" and rep.target_format == "xtrack"


def test_sniff_json_accepts_xtrack_and_rejects_lattix(tmp_path):
    lat = demo_lattice()
    xt_json = tmp_path / "line.json"
    Writer().write(lat, xt_json)
    assert sniff_json(xt_json)

    from lattix.formats.lattix_json import Writer as LattixWriter

    ir_json = tmp_path / "ir.lattix.json"
    LattixWriter().write(lat, ir_json)
    assert not sniff_json(ir_json)

    plain = tmp_path / "other.json"
    plain.write_text('{"hello": 1}')
    assert not sniff_json(plain)
    assert not sniff_json(tmp_path / "missing.json")


def test_sniff_environment_json(tmp_path):
    env = xt.Environment()
    env.new("d1", "Drift", length=1.0)
    env.new_line(name="myline", components=["d1"])
    p = tmp_path / "env.json"
    env.to_json(str(p))
    assert sniff_json(p)


# ------------------------------------------------------------------ reading
def test_reader_round_trip_preserves_the_physics(tmp_path):
    lat = demo_lattice()
    out = tmp_path / "demo.json"
    Writer().write(lat, out)
    back, rep = Reader().read(out)
    assert rep.counts.get("DROPPED", 0) == 0
    a = [p.element for p in lat.flatten()]
    b = [p.element for p in back.flatten()]
    assert [e.kind for e in a] == [e.kind for e in b]
    for x, y in zip(a, b, strict=True):
        assert x.length == pytest.approx(y.length, abs=1e-12)
    assert back.reference.species.name == lat.reference.species.name
    assert back.reference.kinetic_energy_eV == pytest.approx(lat.reference.kinetic_energy_eV)


def test_reader_accepts_an_environment_json(tmp_path):
    env = xt.Environment()
    env.new("q1", "Quadrupole", length=0.3, k1=0.5)
    env.new("d1", "Drift", length=1.0)
    env.new_line(name="l1", components=["q1", "d1"])
    p = tmp_path / "env.json"
    env.to_json(str(p))
    lat, rep = Reader().read(p, kinetic_energy_eV=8e8, species="proton")
    assert [e.element.kind for e in lat.flatten()] == ["Quadrupole", "Drift"]
    assert lat.elements["q1"].multipole.Bn[1] == pytest.approx(0.5 * proton_ref().brho_signed)


def test_reader_picks_a_named_line_from_an_environment(tmp_path):
    env = xt.Environment()
    env.new("d1", "Drift", length=1.0)
    env.new("d2", "Drift", length=2.0)
    env.new_line(name="a", components=["d1"])
    env.new_line(name="b", components=["d2"])
    p = tmp_path / "env.json"
    env.to_json(str(p))
    lat, _ = Reader().read(p, line="a", kinetic_energy_eV=8e8, species="proton")
    assert lat.total_length == pytest.approx(1.0)
    with pytest.raises(KeyError):
        Reader().read(p, line="nope", kinetic_energy_eV=8e8, species="proton")
    _, rep = Reader().read(p, kinetic_energy_eV=8e8, species="proton")
    assert "MULTI_LINE_ENVIRONMENT" in rep.codes()


def test_reader_rejects_a_non_xtrack_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"elements": {}}')
    with pytest.raises(ValueError, match="element_names"):
        load_document(p)
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="not an xtrack JSON"):
        load_document(p)


def test_reader_species_override(tmp_path):
    out = tmp_path / "demo.json"
    Writer().write(demo_lattice(), out)
    lat, _ = Reader().read(out, species="h-", kinetic_energy_eV=2.1e6)
    assert lat.reference.species.name == "h-"
    assert lat.reference.kinetic_energy_eV == pytest.approx(2.1e6)


def test_reader_records_the_source_file(tmp_path):
    out = tmp_path / "demo.json"
    Writer().write(demo_lattice(), out)
    lat, _ = Reader().read(out)
    prov = lat.elements["QF"].provenance
    assert prov is not None and prov.format == "xtrack" and prov.file == str(out)


# -------------------------------------------------------------- idempotence
def _physics(lat: Lattice):
    rows = []
    for p in lat.flatten():
        e = p.element
        m = getattr(e, "multipole", None)
        rows.append((
            e.kind, round(e.length, 15),
            tuple(sorted((m.Bn | {}).items())) if m else (),
            tuple(sorted((m.Bs | {}).items())) if m else (),
            tuple(sorted((m.BnL | {}).items())) if m else (),
            tuple(sorted((m.BsL | {}).items())) if m else (),
            tuple(sorted((m.tilt | {}).items())) if m else (),
            (e.bend.angle, e.bend.e1, e.bend.e2, e.bend.edge_int1, e.bend.hgap,
             e.bend.tilt_ref) if e.kind == "Bend" else (),
            (e.rf.voltage_V, e.rf.phase_rad, e.rf.frequency_Hz)
            if getattr(e, "rf", None) is not None else (),
            (e.hkick, e.vkick) if e.kind == "Kicker" else (),
            e.solenoid.Bsol_T if e.kind == "Solenoid" else 0.0,
        ))
    return rows


def _assert_same_physics(a: Lattice, b: Lattice, tol: float = 1e-12) -> None:
    ra, rb = _physics(a), _physics(b)
    assert [r[0] for r in ra] == [r[0] for r in rb]
    for x, y in zip(ra, rb, strict=True):
        assert x[0] == y[0]
        for fa, fb in zip(x[1:], y[1:], strict=True):
            if isinstance(fa, tuple):
                assert len(fa) == len(fb)
                for u, v in zip(fa, fb, strict=True):
                    if isinstance(u, tuple):
                        assert u[0] == v[0]
                        assert u[1] == pytest.approx(v[1], abs=tol, rel=tol)
                    else:
                        assert u == pytest.approx(v, abs=tol, rel=tol)
            else:
                assert fa == pytest.approx(fb, abs=tol, rel=tol)


@pytest.mark.parametrize("build", [demo_lattice, linac_lattice], ids=["demo", "linac"])
def test_from_line_of_to_line_preserves_the_physics(build):
    """Every kind in these two lattices is exactly representable, so the round trip is a
    fixed point of kinds, lengths, strengths and angles to 1e-12."""
    lat = build()
    _assert_same_physics(lat, from_line(to_line(lat)))


#: what each of the 22 kinds comes back as after a round trip through xtrack, in the
#: order ``all_kinds_lattice`` lays them out (the degradations RULES documents).
_ALL_KINDS_AFTER_ROUND_TRIP = [
    "Drift", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Bend", "Solenoid",
    "RFCavity",
    "RFCavity",          # FieldMap    -> EQUIVALENT FM_AS_CAVITY
    "Drift",             # NCells      -> LOSSY NCELLS_TO_DRIFT
    "Drift",             # RFQCell     -> LOSSY RFQ_TO_DRIFT
    "Kicker",
    "Collimator",
    "Marker",
    "Instrument",        # Instrument  -> EQUIVALENT MONITOR_AS_DRIFT, restored from the metadata
    "Foil",              # Foil        -> LOSSY FOIL_TO_MARKER in xtrack, restored from the metadata
    "Taylor",
    "Patch",
    "ReferenceChange",
    "Freq",              # Freq        -> a marker in xtrack, restored from the metadata
    "Directive",         # Directive   -> DROPPED FOREIGN_DIRECTIVE, restored from the metadata
    "Drift",             # Superposition child
]


def test_all_kinds_round_trip_degrades_exactly_as_documented():
    lat = all_kinds_lattice()
    back = from_line(to_line(lat))
    assert [p.element.kind for p in back.flatten()] == _ALL_KINDS_AFTER_ROUND_TRIP
    assert back.total_length == pytest.approx(lat.total_length, abs=1e-12)
    # the exactly-representable ones keep their physics
    names = ["dr", "qp", "sx", "oc", "mp", "bd", "sl", "cv", "kk"]
    a = {p.element.name: p.element for p in lat.flatten()}
    b = {p.element.name: p.element for p in back.flatten()}
    for nm in names:
        _assert_same_physics(Lattice.from_sequence("a", [a[nm]], lat.reference),
                             Lattice.from_sequence("b", [b[nm]], lat.reference))


def test_round_trip_is_a_fixed_point_of_the_json(tmp_path):
    """write → read → write → read → write reproduces the deck byte for byte
    (invariant I-13; the first write has no per-element provenance to carry yet)."""
    lat = demo_lattice()
    # same file name in three directories: the Reader names the lattice after the stem
    paths = []
    for step in ("a", "b", "c"):
        d = tmp_path / step
        d.mkdir()
        paths.append(d / "demo.json")
    Writer().write(lat, paths[0])
    lat_b, _ = Reader().read(paths[0])
    Writer().write(lat_b, paths[1])
    lat_c, _ = Reader().read(paths[1])
    Writer().write(lat_c, paths[2])
    assert _norm(paths[1].read_text()) == _norm(paths[2].read_text())


def test_to_json_from_json_round_trip(tmp_path):
    """The deck lattix writes is a deck xtrack itself reads back unchanged."""
    lat = demo_lattice()
    out = tmp_path / "demo.json"
    Writer().write(lat, out)
    line = xt.Line.from_json(str(out))
    mine = to_line(lat)
    assert line.element_names == mine.element_names
    assert line.metadata[METADATA_KEY]["lattice"] == "demo"
    for nm in line.element_names:
        assert type(line.element_dict[nm]).__name__ == type(mine.element_dict[nm]).__name__


def test_strict_write_raises_on_a_lossy_kind(tmp_path):
    lat = Lattice.from_sequence("s", [NCells(name="nc", length=0.5)], proton_ref())
    with pytest.raises(TranslationError):
        Writer().write(lat, tmp_path / "x.json", strict=True)


def test_writer_rules_is_the_shared_table():
    from lattix.formats.xtrack.convert import RULES as CONVERT_RULES

    assert Writer.RULES is CONVERT_RULES


def test_kicker_and_multipole_survive_the_json(tmp_path):
    ref = proton_ref()
    b = ref.brho_signed
    lat = Lattice.from_sequence("k", [
        Kicker(name="cor", length=0.2, hkick=1e-3, vkick=-2e-3),
        Multipole(name="mp", multipole=MagneticMultipoleP(BnL={2: 0.3 * b})),
    ], ref)
    out = tmp_path / "k.json"
    Writer().write(lat, out)
    back, _ = Reader().read(out)
    cor = back.elements["cor"]
    assert cor.kind == "Kicker"
    assert cor.hkick == pytest.approx(1e-3) and cor.vkick == pytest.approx(-2e-3)
    assert cor.length == pytest.approx(0.2)
    assert back.elements["mp"].multipole.BnL[2] == pytest.approx(0.3 * b)
