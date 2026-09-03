"""PALS interoperability: three independent implementations check what lattix writes and reads.

PALS is the only format in this repo with a foreign **writer** (Bmad/Tao ``write pals``) and a
foreign **reader** (``pals-schema`` and ImpactX's ``KnownElementsList.load_file``) already on
this machine, so a PALS conversion can be pinned without a round trip through ourselves
(PLAN §3, design consequence (a)).

* ``pals-schema`` (import name ``pals``, the pals-python reference implementation) validates
  every document the writer produces — env ``lattix``; skipped elsewhere.
* **ImpactX** ``elements.KnownElementsList.load_file`` reads our document back into ImpactX
  elements; the element count, names, lengths and quadrupole strengths must match the IR.
* **Bmad/Tao** writes a PALS file of its own from ``tests/data/public/helix/fodo.bmad``;
  we read *that* with our reader and compare it with the same lattice read from
  ``fodo.madx`` through :mod:`lattix.formats.madx`.

Measured 2026-09-03 with pals-schema 0.3.0, ImpactX 26.08 and Bmad 20260828 — the
divergences the tests below pin are recorded in ``docs/oracles.md``.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lattix.formats.pals import Reader, Writer
from lattix.ir import (
    RFP,
    Bend,
    BendP,
    Drift,
    Lattice,
    MagneticMultipoleP,
    Marker,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    species,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
BMAD_PYTHON = Path("/Users/abhishekpathak/anaconda3/envs/bmad/bin/python")
BMAD_TAO = Path("/Users/abhishekpathak/anaconda3/envs/bmad/bin/tao")

try:                                    # pals-schema, import name `pals`
    import pals as pals_schema
except Exception:                       # noqa: BLE001 - any import failure means "not available"
    pals_schema = None

try:
    import impactx as _impactx
except Exception:                       # noqa: BLE001
    _impactx = None

needs_pals_schema = pytest.mark.skipif(
    pals_schema is None,
    reason="pals-schema not importable in this environment (pip install pals-schema)")
needs_impactx = pytest.mark.skipif(
    _impactx is None, reason="impactx not importable in this environment")
needs_bmad = pytest.mark.skipif(
    not BMAD_TAO.exists(), reason=f"Tao not found at {BMAD_TAO}")


def proton_ref(ke: float = 8e8) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke)


def fodo_ir() -> Lattice:
    """A drift/quad FODO built in the IR — the subset every PALS consumer supports."""
    els = [
        Drift(name="d1", length=0.25),
        Quadrupole(name="q1", length=1.0, multipole=MagneticMultipoleP(Bn={1: 1.0})),
        Drift(name="d2", length=0.5),
        Quadrupole(name="q2", length=1.0, multipole=MagneticMultipoleP(Bn={1: -1.0})),
        Drift(name="d3", length=0.25),
    ]
    return Lattice.from_sequence("fodo", els, proton_ref())


def mixed_ir() -> Lattice:
    els = [
        Drift(name="d", length=0.5),
        Bend(name="b", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05)),
        RFCavity(name="c", length=1.0, rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=650e6)),
        Marker(name="m"),
    ]
    return Lattice.from_sequence("mixed", els, proton_ref())


def _sources():
    yield "fodo_ir", fodo_ir()
    yield "mixed_ir", mixed_ir()
    for name in ("fodo", "iota", "bend_angle_radius", "rf_voltage", "rf_gradient",
                 "drift_quad_bend"):
        yield name, Reader().read(DATA / "pals" / f"{name}.pals.yaml")[0]
    yield "impactx_fodo", Reader().read(DATA / "impactx" / "fodo.pals.yaml")[0]


SOURCES = list(_sources())


# ── (a) pals-schema validates every document we write ─────────────────────────────────────
@needs_pals_schema
@pytest.mark.parametrize(("name", "lat"), SOURCES, ids=[n for n, _ in SOURCES])
@pytest.mark.parametrize("flavor", ["standard", "flat"])
def test_pals_schema_loads_every_written_document(tmp_path, name, lat, flavor):
    out = tmp_path / f"{name}_{flavor}.pals.yaml"
    Writer().write(lat, out, flavor=flavor)
    root = pals_schema.load(str(out))          # raises pydantic.ValidationError on any error
    assert root.facility, f"{out.name} loaded but has an empty facility"
    # the last facility entry is our `use:` statement, which pals-schema models as a name
    assert type(root.facility[-1]).__name__ == "PlaceholderName"


@needs_pals_schema
def test_pals_schema_reads_back_the_numbers_we_wrote(tmp_path):
    lat = fodo_ir()
    out = tmp_path / "fodo.pals.yaml"
    Writer().write(lat, out, flavor="flat", beginning=False)
    root = pals_schema.load(str(out))
    line = root.facility[0].line
    ir = lat.flatten()
    assert len(line) == len(ir)
    for got, want in zip(line, ir, strict=True):
        assert got.name == want.element.name
        assert got.length == pytest.approx(want.length)
        if want.element.kind == "Quadrupole":
            assert got.MagneticMultipoleP.Bn1 == pytest.approx(want.element.multipole.Bn[1])


@needs_pals_schema
def test_pals_schema_030_lags_the_standard_text(tmp_path):
    """Documented divergence (2026-09-03): the reference implementation is an older draft.

    It has ``SBend``/``RBend`` but no ``Bend`` kind, so a standard ``kind: Bend`` silently
    degrades to a ``PlaceholderName`` instead of raising.  lattix writes the *standard*
    spelling and its reader accepts both, so this test pins the gap rather than hiding it.
    """
    kinds = set(dir(pals_schema))
    assert {"SBend", "RBend"} <= kinds and "Bend" not in kinds
    assert "ReferenceChange" not in kinds
    bend_fields = set(pals_schema.BendParameters.model_fields)
    assert "angle_ref" not in bend_fields and "rho_ref" in bend_fields
    assert "edge_int1" in bend_fields and "edge1_int" not in bend_fields
    rf_fields = set(pals_schema.RFParameters.model_fields)
    assert "n_cell" in rf_fields and "num_cells" not in rf_fields
    assert not {"zero_phase", "L_active", "dE_ref"} & rf_fields

    out = tmp_path / "b.pals.yaml"
    Writer().write(mixed_ir(), out, flavor="flat")
    root = pals_schema.load(str(out))
    degraded = [type(e).__name__ for e in root.facility[0].line]
    assert "PlaceholderName" in degraded          # the Bend, silently
    # ... while everything pals-schema 0.3.0 does know keeps its kind
    assert {"Drift", "RFCavity", "Marker", "BeginningEle"} <= set(degraded)


# ── (b) ImpactX reads our document ────────────────────────────────────────────────────────
@pytest.mark.oracle_impactx
@needs_impactx
@pytest.mark.parametrize("name", ["fodo_ir", "impactx_fodo", "fodo"])
def test_impactx_loads_our_pals_file(tmp_path, name):
    """ImpactX 26.08 ``pals_to_impactx.read_lattice`` supports Drift and Quadrupole only and
    raises on a ``BeginningEle``, so the oracle uses ``flavor="flat", beginning=False``."""
    import warnings

    from impactx import elements

    lat = dict(SOURCES)[name]
    out = tmp_path / f"{name}.pals.yaml"
    rep = Writer().write(lat, out, flavor="flat", beginning=False)
    assert "PALS_NO_BEGINNING_ELE" in rep.codes()

    kel = elements.KnownElementsList()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # ImpactX's "preview parser" warning
        kel.load_file(str(out), nslice=1)

    placed = lat.flatten()
    assert len(kel) == len(placed)
    for got, want in zip(kel, placed, strict=True):
        assert got.ds == pytest.approx(want.length), want.element.name
        # the flat flavor gives each occurrence its own name (`drift1`, `drift1_2`, ...)
        assert re.fullmatch(rf"{re.escape(want.element.name)}(_\d+)?", got.name), got.name
        if want.element.kind == "Quadrupole":
            # ImpactX takes PALS `Bn1` as ChrQuad(k=..., unit=1), i.e. a field in T/m
            assert got.k == pytest.approx(want.element.multipole.Bn[1])
    assert sum(e.ds for e in kel) == pytest.approx(sum(p.length for p in placed))


@pytest.mark.oracle_impactx
@needs_impactx
def test_impactx_rejects_kinds_it_does_not_model(tmp_path):
    """A recorded limitation, not a lattix bug: ImpactX's PALS reader is Drift + Quadrupole."""
    import warnings

    from impactx import elements

    out = tmp_path / "mixed.pals.yaml"
    Writer().write(mixed_ir(), out, flavor="flat", beginning=False)
    kel = elements.KnownElementsList()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError) as exc:
            kel.load_file(str(out), nslice=1)
    # Either message is the same limitation: the Bend has no ImpactX counterpart, and because
    # pals-schema 0.3.0 has no `Bend` kind it reaches ImpactX as an unresolvable placeholder.
    assert ("No support for elements of kind" in str(exc.value)
            or "Cannot resolve PALS element reference" in str(exc.value)), exc.value


# ── (c) Bmad/Tao writes PALS, we read it ──────────────────────────────────────────────────
_TAO_SCRIPT = """
import sys
from pytao import Tao
tao = Tao('-lat %s -noplot -no_stopping')
print(tao.cmd('write pals %s'))
"""


def _tao_write_pals(tmp_path: Path) -> Path:
    src = tmp_path / "fodo.bmad"
    shutil.copy(DATA / "helix" / "fodo.bmad", src)
    out = tmp_path / "bmad.pals.yaml"
    proc = subprocess.run(
        [str(BMAD_PYTHON), "-c", _TAO_SCRIPT % (src.name, out.name)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False)
    if not out.exists():
        pytest.fail(f"Tao did not write {out.name}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return out


@pytest.mark.oracle_bmad
@needs_bmad
def test_bmad_pals_output_matches_the_madx_lattice(tmp_path):
    """Bmad ``write pals`` on ``fodo.bmad`` vs ``fodo.madx`` through lattix's MAD-X reader.

    Both decks describe the same 6.6 m FODO+bend cell; the Bmad deck was generated from the
    MAD-X one by Bmad's own ``madx_to_bmad.py`` (see ``tests/data/public/README.md``).
    """
    pytest.importorskip("cpymad", reason="the MAD-X leg of this comparison needs cpymad")
    from lattix.formats.madx import Reader as MadxReader

    pals_file = _tao_write_pals(tmp_path)
    bm, rep = Reader().read(pals_file)
    assert rep.ok, rep.summary()
    mx, _ = MadxReader().read(DATA / "helix" / "fodo.madx")

    def strip(lat):
        """Drop the zero-length drifts Bmad emits to pad a `refer=centre` sequence."""
        return [p for p in lat.flatten() if not (p.element.kind == "Drift" and p.length == 0.0)]

    b, m = strip(bm), strip(mx)
    if b and b[-1].element.kind == "Marker" and len(b) == len(m) + 1:
        b = b[:-1]                       # Bmad's own `end_b0` branch-end marker
    assert [p.element.kind for p in b] == [p.element.kind for p in m]
    assert [round(p.length, 9) for p in b] == [round(p.length, 9) for p in m]
    assert sum(p.length for p in b) == pytest.approx(6.6)

    # reference particle: Bmad writes pc_ref; its proton mass differs from CODATA-2018 by
    # 1.35e-9 relative (docs/oracles.md), so compare at 1e-6.
    assert bm.reference.species.name == mx.reference.species.name == "proton"
    assert bm.reference.kinetic_energy_eV == pytest.approx(mx.reference.kinetic_energy_eV,
                                                           rel=1e-6)

    # quadrupole gradients: Bmad writes the normalized Kn1, lattix converts with its own Brho
    for name in ("qf", "qd"):
        assert bm.elements[name].multipole.Bn[1] == \
            pytest.approx(mx.elements[name].multipole.Bn[1], rel=1e-6)

    # bend: Bmad writes g_ref (+ length), lattix's MAD-X reader gets `angle` — same number
    assert bm.elements["b1"].bend.angle == pytest.approx(mx.elements["b1"].bend.angle, rel=1e-12)
    assert bm.elements["b1"].bend.e1 == pytest.approx(mx.elements["b1"].bend.e1, rel=1e-12)
    assert bm.elements["b1"].bend.e2 == pytest.approx(mx.elements["b1"].bend.e2, rel=1e-12)


@pytest.mark.oracle_bmad
@needs_bmad
def test_which_pals_fields_bmad_emits(tmp_path):
    """Pin exactly what Bmad 20260828 puts in a PALS file (the first independent writer).

    Findings, all consistent with the standard text at commit ``a2b1083``:

    * ``kind: Bend`` (not ``pals-schema`` 0.3.0's ``SBend``/``RBend``);
    * a bend is given by ``length`` + ``BendP.g_ref`` — the curvature/length pair — and it
      *also* writes the actual bending field as ``MagneticMultipoleP.Kn0`` even though
      ``BendP.Kn0_from_g_ref`` defaults to true;
    * magnets carry the **normalized** ``Kn1``/``Kn0``, never the lab field ``Bn1``;
    * the reference particle is ``ReferenceP {pc_ref, species_ref}`` — momentum, not
      ``E_tot_ref`` — on a ``BeginningEle`` first in the line, with ``TwissP`` beside it;
    * a zero-length drift is written with **no** ``length`` key at all;
    * constants come back as a ``- constants:`` facility entry;
    * Bmad-specific data goes in a ``Bmad:``/``Bmad_``-prefixed extension declared in
      ``extension_labels`` (``extensions.md``), e.g. ``Bmad_key: SBend``;
    * ``Lattice.branches`` uses the ``{name: {periodic: true}}`` form and there is **no**
      ``use:`` statement (the default "last Lattice" applies);
    * numbers are Fortran ``E`` notation with a signed exponent (``1.0E+000``), which YAML 1.1
      resolves as floats — unlike the standard's own examples, whose ``1.0e9`` (unsigned
      exponent) YAML resolves as a *string*.
    """
    doc = yaml.safe_load(_tao_write_pals(tmp_path).read_text())
    root = doc["PALS"]
    assert "use" not in root
    entries = {k: v for e in root["facility"] if isinstance(e, dict) and len(e) == 1
               for k, v in e.items()}
    nodes = {k: v for k, v in entries.items() if isinstance(v, dict)}
    assert root["extension_labels"]["names"]["Bmad"]
    assert isinstance(entries["constants"], list)

    assert set(entries["b1"]) == {"kind", "length", "BendP", "MagneticMultipoleP", "Bmad"}
    assert entries["b1"]["kind"] == "Bend"
    assert set(entries["b1"]["BendP"]) == {"g_ref", "e1", "e2"}
    assert entries["b1"]["MagneticMultipoleP"] == {"Kn0": pytest.approx(0.1)}
    assert entries["b1"]["Bmad"]["Bmad_key"] == "SBend"

    assert set(entries["qf"]["MagneticMultipoleP"]) == {"Kn1"}
    assert "length" not in entries["drift0"]              # zero-length drift

    begin = next(v for v in nodes.values() if v.get("kind") == "BeginningEle")
    assert set(begin["ReferenceP"]) == {"pc_ref", "species_ref"}
    assert "TwissP" in begin

    lattice = next(v for v in nodes.values() if v.get("kind") == "Lattice")
    assert lattice["branches"] == [{"fodo": {"periodic": True}}]


@pytest.mark.oracle_bmad
@needs_bmad
def test_bmad_pals_output_survives_a_lattix_round_trip(tmp_path):
    """Read Bmad's file, write it back as PALS, read again: same elements and numbers."""
    src = _tao_write_pals(tmp_path)
    a, rep_a = Reader().read(src)
    out = tmp_path / "again.pals.yaml"
    Writer().write(a, out, strict=True)
    b, rep_b = Reader().read(out)
    assert rep_a.ok and rep_b.ok
    assert [(p.element.kind, round(p.length, 12)) for p in a.flatten()] == \
           [(p.element.kind, round(p.length, 12)) for p in b.flatten()]
    assert b.reference.pc_eV == pytest.approx(a.reference.pc_eV, rel=1e-12)
    assert b.elements["qf"].multipole.Bn[1] == pytest.approx(a.elements["qf"].multipole.Bn[1])


def test_interop_environment_is_reported():
    """Never silently green: say which oracle legs ran (visible with ``pytest -s``)."""
    print(f"\npals-schema: {'yes' if pals_schema else 'no'}   "
          f"impactx: {'yes' if _impactx else 'no'}   "
          f"tao: {'yes' if BMAD_TAO.exists() else 'no'}   ({sys.executable})")
