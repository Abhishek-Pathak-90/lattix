"""Unit tests for ``lattix.corpus`` (manifest build / verify / list) on synthetic data.

No private corpus is needed: every test builds its own tiny corpus under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from lattix import corpus
from lattix.corpus import (
    build_manifest,
    expand_braces,
    is_binary,
    iter_decks,
    load_manifest,
    load_sources,
    main,
    sha256_of,
    slugify,
    sniff_format,
    sniff_text,
    verify_manifest,
    write_manifest,
)

# --------------------------------------------------------------------------- samples

TRACEWIN = """;------------------------------------
; TraceWin Version: 2.18.2.1
;------------------------------------
FIELD_MAP_PATH Fields

; Section #1
LATTICE 4 0
FREQ 162.5
DRIFT 177.422 15 0 0 0
QUAD 200 5.66763 100 0 0 0 0 0 0
BA1011:BEND 6.5637426 21384.55 0.000 25.400 0 ; labelled card
FIELD_MAP 7700 350 -12.5 30 1 1 0 0 HWRDonut
end
"""
MADX = """! MAD-X input
TITLE, "FODO";
beam, particle=proton, energy=1.0;
QF: QUADRUPOLE, L=0.5, K1:=0.8;
QD: QUADRUPOLE, L=0.5, K1:=-0.8;
D: DRIFT, L=1.0;
FODO: SEQUENCE, REFER=centre, L=4.0;
  QF, at=0.25;
  D, at=1.0;
  QD, at=2.25;
ENDSEQUENCE;
use, sequence=FODO;
twiss;
"""
MAD8 = """! DATE AND TIME:    16/07/25  13.37.12
! FILE:             BTL2025V0703.FLAT
H: MONITOR
OLQ: DRIFT, L=0.15
QLF1: QUADRUPOLE, L=0.2, K1=-2.18472718
DRFCAVU: DRIFT, L=0.5*(T1B[L]+0.601E-3+OS1[L]+O2A[L]-D2RFCAV[L]-2.0*DIRFCAV1[&
L])
BTL: LINE=(H, OLQ, QLF1, OLQ, &
  DRFCAVU)
USE, BTL
RETURN
"""
ELEGANT = """! elegant lattice
OLQ: DRIF, L=0.15
QLF1: QUAD, L=0.2, K1=-2.18
B1: CSBEND, L=1.0, ANGLE=0.1, N_KICKS=20
W1: WATCH, FILENAME="%s.w1"
BTL: LINE=(OLQ, QLF1, B1, W1)
"""
ELEGANT_FULLNAMES = """! ELEGANT Lattice for HWR Cryomodule
D1: DRIFT, L=0.177422, APERTURE=0.007500
S1: SOLENOID, L=0.3, KS=1.2
C1: RFCA, L=0.2, VOLT=1.0e6, FREQ=162.5e6, PHASE=90
HWR: LINE=(D1, S1, C1, D1)
"""
MAD_FULLNAMES_NOSEMI = """! only MAD-style full type names, no ';' and no '&': dialect is ambiguous
D1: DRIFT, L=0.177422
S1: SOLENOID, L=0.3, KS=1.2
Q1: QUADRUPOLE, L=0.2, K1=1.0
HWR: LINE=(D1, S1, Q1, D1)
"""
BMAD = """! bmad lattice
parameter[geometry] = open
parameter[p0c] = 1e9
beginning[beta_a] = 10
q1: quadrupole, l = 0.5, k1 = 0.8
d1: drift, l = 1
c1: lcavity, l = 1, voltage = 1e6, rf_frequency = 1.3e9
fodo: line = (q1, d1, c1)
use, fodo
"""
FLAME = """# Beam envelope simulation.
sim_type = "MomentMatrix";
MpoleLevel = "2";
IonEk = 500e3;
IonEs = 931494320.0;
drift_1: drift, L = 0.1;
ls1: LINE = (drift_1);
USE: ls1;
"""
FLAME_MINIMAL = """# used by test of parse()
foo : bar;
baz : LINE = (foo, foo);
"""
PYORBIT_MADX = """!---  DRIFT SPACES DEFINITION
DR: drift, L = 1.191;
RADDEG=1/57.2958;
ALPHA  := 15.0 * RADDEG;
RB: SBEND,L = 1.0,ANGLE = ALPHA;
"""
IMPACTX = """###############################################################################
# Particle Beam(s)
###############################################################################
beam.npart = 10000
beam.units = static
beam.kin_energy = 2.0e3
lattice.elements = drift1 quad1 drift2
drift1.type = drift
"""
IMPACTZ = """!Input file for the IMPACT-Z beam dynamics code:
1 1
6 200 2 0 1
32 32 64 1 0.02 0.02 0.1
7.886300e-02 110 20 101 2503370.0 3.240000e+08 222 8.520000e+02 6.500000e-03 /
8.034760e-02 110 20 101 2503370.0 3.240000e+08 222 8.680000e+02 6.500000e-03 /
8.183570e-02 110 20 101 2503370.0 3.240000e+08 222 8.840000e+02 6.500000e-03 /
"""
PALS = """fodo_cell:
  kind: BeamLine
  line:
  - drift1:
      kind: Drift
      length: 0.25
"""
TFS = """@ NAME             %7s "DCTABLE"
@ TYPE             %04s "USER"
* NAME   I
$ %s     %le
 "DCA"   1.0
"""
TW_MATRIX = """ELE# 1 : 0.25 m
+1.000000e+000 +2.500000e-001 +0.000000e+000 +0.000000e+000 +0.000000e+000 +0.000000e+000
+0.000000e+000 +1.000000e+000 +0.000000e+000 +0.000000e+000 +0.000000e+000 +0.000000e+000
"""
TW_ENVELOPE = "position\tgam-1\tcentroid position(x,x',y,y')\n+0.0e+000\t+2.2e-003\t+0.0e+000\t+0.0e+000\n"
TABLE = """x y z
0.0 1.0 2.0
0.1 1.1 2.1
0.2 1.2 2.2
"""
PROSE = "Dear colleague,\nplease find the attached decks.\nRegards\n"


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        (TRACEWIN, "latf.dat", "tracewin"),
        (TRACEWIN, "HWR_Lat.lat", "tracewin"),  # PIP-II TraceWin dialect despite .lat
        (MADX, "fodo.madx", "madx"),
        (MADX, "fodo.seq", "madx"),
        (MADX, "anything.txt", "madx"),  # content wins over an unknown suffix
        (MAD8, "BTL2025V0703.FLAT", "mad8"),
        (MAD8, "BTL2025v0703.lat", "mad8"),
        (MAD8, "mad8.correct", "mad8"),
        (ELEGANT, "elegant_lattice.lte", "elegant"),
        (ELEGANT, "lte.correct", "elegant"),  # elegant-only types decide, not the suffix
        (ELEGANT_FULLNAMES, "TraceWin_elegant_lattice.lte", "elegant"),
        (ELEGANT_FULLNAMES, "hwr.FLAT", "elegant"),  # RFCA is elegant-only: content beats suffix
        (MAD_FULLNAMES_NOSEMI, "hwr.lte", "elegant"),  # ambiguous grammar: suffix breaks the tie
        (MAD_FULLNAMES_NOSEMI, "hwr.FLAT", "mad8"),
        (MAD_FULLNAMES_NOSEMI, "hwr.txt", "mad8"),  # no prior at all: MAD8 is the default
        (BMAD, "lat.bmad", "bmad"),
        (BMAD, "lat.txt", "bmad"),
        (FLAME, "LS1.lat", "flame"),
        (FLAME_MINIMAL, "parse1.lat", "flame"),  # no FLAME globals: .lat prior over madx
        (PYORBIT_MADX, "fodo.lat", "madx"),  # ':=' with ';' is MAD-X, even in a .lat
        (IMPACTX, "input_fodo.in", "impactx"),
        (IMPACTZ, "ImpactZ.in", "impactz"),
        (PALS, "fodo.pals.yaml", "pals"),
        (PALS, "fodo.yaml", "pals"),
        (TFS, "DC.dat", "tfs"),
        (TW_MATRIX, "Transfer_matrix1.dat", "tracewin_output"),
        (TW_ENVELOPE, "ENV+SC.txt", "tracewin_output"),
        ("1 2\n3 4\n", "beta092.edz", "fieldmap"),
        ("1 2\n3 4\n", "SOL1-PXIE.scc", "fieldmap"),
        (TABLE, "scan.txt", "table"),
        ("from ocelot import *\nd = Drift(l=1.0, eid='d')\ncell = (d,)\nlattice = MagneticLattice(cell)\n",
         "lattice.py", "ocelot"),
        ("import numpy as np\nprint(np.pi)\n", "script.py", "unknown"),   # a python file is not a lattice
        (PROSE, "notes.txt", "unknown"),
        ("", "empty.dat", "unknown"),
    ],
)
def test_sniff_text(text: str, name: str, expected: str) -> None:
    assert sniff_text(text, name) == expected


def test_tracewin_card_lookahead_rejects_mad_definitions() -> None:
    # An element *named* like a TraceWin card must not count as a card.
    assert sniff_text("DRIFT: DRIFT, L=1;\nQUAD: QUADRUPOLE, L=1, K1=1;\n", "x.madx") == "madx"
    assert sniff_text("BEND: SBEND, L=1, ANGLE=0.1\nQUAD: QUADRUPOLE, L=1\n", "x.FLAT") == "mad8"


def test_sniff_format_reads_file_and_flags_binary(tmp_path: Path) -> None:
    deck = tmp_path / "deck.dat"
    deck.write_text(TRACEWIN)
    assert sniff_format(deck) == "tracewin"
    blob = tmp_path / "Density_Env.dat"
    blob.write_bytes(b"\x00\x01\x02" * 100)
    assert sniff_format(blob) == "binary"
    utf16 = tmp_path / "deck16.dat"
    utf16.write_bytes("﻿".encode("utf-16") + TRACEWIN.encode("utf-16-le"))
    assert sniff_format(utf16) == "tracewin"


def test_is_binary() -> None:
    assert not is_binary(b"")
    assert not is_binary(b"DRIFT 100 30\n")
    assert is_binary(b"abc\x00def")
    assert is_binary(bytes(range(1, 32)) * 4 + b"x")
    assert not is_binary(b"\xff\xfeD\x00R\x00")


def test_sha256_of(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    data = bytes(range(256)) * 5000  # > 1 MiB, exercises chunking
    f.write_bytes(data)
    assert sha256_of(f) == hashlib.sha256(data).hexdigest()


def test_slugify() -> None:
    assert slugify("Old design/B1 (v2).dat") == "old-design/b1-v2.dat"
    assert slugify("pipii/mebt+hwr/mebt+hwr.dat") == "pipii/mebt+hwr/mebt+hwr.dat"
    assert slugify("lboptimization - Copy/calculations/latf.dat") == (
        "lboptimization-copy/calculations/latf.dat"
    )
    assert slugify("Virtual Accelerator/x.lte") == "virtual-accelerator/x.lte"


def test_expand_braces() -> None:
    assert expand_braces("a.dat") == ["a.dat"]
    assert expand_braces("dir/{a.FLAT,b.lte}") == ["dir/a.FLAT", "dir/b.lte"]
    assert expand_braces("**/*.{madx,seq,dat}") == ["**/*.madx", "**/*.seq", "**/*.dat"]
    assert expand_braces("{x,y}/{1,2}") == ["x/1", "x/2", "y/1", "y/2"]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture()
def corpus_tree(tmp_path: Path) -> dict[str, Path]:
    """A private-looking data tree outside the corpus root, plus one deck inside it."""
    data = tmp_path / "data"
    (data / "PIP" / "SC Linac" / "optimized_datfiles").mkdir(parents=True)
    (data / "PIP" / "SC Linac" / "optimized_datfiles" / "latf.dat").write_text(TRACEWIN)
    (data / "PIP" / "SC Linac" / "latf.dat").write_text(TRACEWIN + "DRIFT 1 1\n")
    (data / "PIP" / "SC Linac" / "Transfer_matrix1.dat").write_text(TW_MATRIX)
    (data / "PIP" / "SC Linac" / "Density_Env.dat").write_bytes(b"\x00\x01" * 64)
    (data / "PIP" / ".hidden.dat").write_text(TRACEWIN)
    (data / "MAD").mkdir()
    (data / "MAD" / "BTL.FLAT").write_text(MAD8)
    (data / "MAD" / "elegant_lattice.lte").write_text(ELEGANT)
    (data / "MAD" / "fodo.madx").write_text(MADX)
    (data / "Fields").mkdir()
    (data / "Fields" / "beta092.edz").write_text("1 2\n3 4\n")
    (data / "Fields" / "notes.txt").write_text(PROSE)
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "local").mkdir()
    (root / "local" / "fodo_cell.dat").write_text(TRACEWIN)
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sources": [
                    {
                        "name": "pipii-tracewin",
                        "root": str(data / "PIP"),
                        "paths": ["**/*.dat"],
                        "formats": ["tracewin"],
                        "origin": "PIP-II design decks (Fermilab)",
                        "license": "private",
                        "redistributable": False,
                        "notes": "test",
                    },
                    {
                        "name": "anchors",
                        "root": str(data),
                        "paths": ["MAD/{BTL.FLAT,elegant_lattice.lte}", "MAD/*.madx"],
                        "origin": "anchors",
                        "license": "private",
                        "redistributable": False,
                    },
                    {
                        "name": "fieldmaps",
                        "root": str(data),
                        "paths": ["Fields"],
                        "format": "fieldmap",
                        "origin": "ANL/CEA",
                        "license": "private",
                        "redistributable": False,
                    },
                    {
                        "name": "local",
                        "root": str(root),
                        "paths": ["local/*.dat"],
                        "origin": "HELIX (Abhishek Pathak)",
                        "license": "GPL-3.0 / owner",
                        "redistributable": True,
                    },
                ],
            }
        )
    )
    return {"data": data, "root": root, "sources": sources}


# --------------------------------------------------------------------------- build


def test_build_manifest_entries(corpus_tree: dict[str, Path]) -> None:
    root, sources, data = corpus_tree["root"], corpus_tree["sources"], corpus_tree["data"]
    result = build_manifest(root, sources)
    ids = [e["id"] for e in result.entries]
    assert ids == sorted(ids)
    assert ids == [
        "anchors/mad/btl.flat",
        "anchors/mad/elegant_lattice.lte",
        "anchors/mad/fodo.madx",
        "fieldmaps/fields/beta092.edz",
        "fieldmaps/fields/notes.txt",
        "local/local/fodo_cell.dat",
        "pipii-tracewin/sc-linac/latf.dat",
        "pipii-tracewin/sc-linac/optimized_datfiles/latf.dat",
    ]
    by_id = {e["id"]: e for e in result.entries}
    assert by_id["anchors/mad/btl.flat"]["format"] == "mad8"
    assert by_id["anchors/mad/elegant_lattice.lte"]["format"] == "elegant"
    assert by_id["anchors/mad/fodo.madx"]["format"] == "madx"
    assert by_id["pipii-tracewin/sc-linac/latf.dat"]["format"] == "tracewin"
    # format override is recorded together with what the sniffer actually saw
    notes = by_id["fieldmaps/fields/notes.txt"]
    assert notes["format"] == "fieldmap" and notes["detected_format"] == "unknown"
    assert "detected_format" not in by_id["fieldmaps/fields/beta092.edz"]
    # provenance and curated fields
    deck = by_id["pipii-tracewin/sc-linac/optimized_datfiles/latf.dat"]
    assert deck["origin"] == "PIP-II design decks (Fermilab)"
    assert deck["license"] == "private"
    assert deck["redistributable"] is False
    assert deck["notes"] == "test"
    assert deck["expected_warning_classes"] == [] and deck["anchors"] == []
    assert deck["size"] == len(TRACEWIN.encode())
    assert deck["sha256"] == hashlib.sha256(TRACEWIN.encode()).hexdigest()
    # paths: absolute outside the root, relative inside it
    assert Path(deck["path"]).is_absolute()
    assert by_id["local/local/fodo_cell.dat"]["path"] == "local/fodo_cell.dat"
    assert by_id["local/local/fodo_cell.dat"]["redistributable"] is True
    # skipped: binary, filtered output, hidden file never seen
    skipped = dict(result.skipped)
    assert skipped[(data / "PIP" / "SC Linac" / "Density_Env.dat").as_posix()] == "binary"
    assert skipped[(data / "PIP" / "SC Linac" / "Transfer_matrix1.dat").as_posix()].startswith(
        "filtered: sniffed as tracewin_output"
    )
    assert not any(".hidden" in p for p, _ in result.skipped)
    assert result.counts_by_format() == {
        "tracewin": 3,
        "fieldmap": 2,
        "elegant": 1,
        "mad8": 1,
        "madx": 1,
    }
    assert [n for _, n in result.ambiguities] == ["override fieldmap (sniffed unknown)"]


def test_build_manifest_size_limit(corpus_tree: dict[str, Path]) -> None:
    result = build_manifest(corpus_tree["root"], corpus_tree["sources"], max_size=200)
    assert not any(e["id"] == "anchors/mad/btl.flat" for e in result.entries)
    assert any("too large" in why for _, why in result.skipped)


def test_build_manifest_carries_curated_fields(corpus_tree: dict[str, Path]) -> None:
    root, sources = corpus_tree["root"], corpus_tree["sources"]
    first = build_manifest(root, sources)
    for e in first.entries:
        if e["id"] == "anchors/mad/btl.flat":
            e["anchors"] = ["btl-flat-vs-lte"]
            e["expected_warning_classes"] = ["W_UNSUPPORTED_ELEMENT"]
    write_manifest(root, first.manifest)
    again = build_manifest(root, sources, carry_from=load_manifest(root))
    e = next(x for x in again.entries if x["id"] == "anchors/mad/btl.flat")
    assert e["anchors"] == ["btl-flat-vs-lte"]
    assert e["expected_warning_classes"] == ["W_UNSUPPORTED_ELEMENT"]
    fresh = build_manifest(root, sources)
    e = next(x for x in fresh.entries if x["id"] == "anchors/mad/btl.flat")
    assert e["anchors"] == []


def test_write_and_load_manifest_roundtrip(corpus_tree: dict[str, Path]) -> None:
    root = corpus_tree["root"]
    result = build_manifest(root, corpus_tree["sources"])
    target = write_manifest(root, result.manifest)
    assert target == root / corpus.MANIFEST_NAME
    doc = yaml.safe_load(target.read_text())
    assert doc["manifest_version"] == corpus.MANIFEST_VERSION
    assert doc["root"] == root.resolve().as_posix()
    assert load_manifest(root) == result.entries
    # key order is stable so rebuild diffs stay readable
    assert list(result.entries[0]) == [
        "id",
        "path",
        "source",
        "format",
        "sha256",
        "size",
        "origin",
        "license",
        "redistributable",
        "expected_warning_classes",
        "anchors",
    ]


def test_load_sources_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("sources: [{name: X, root: /, paths: [], origin: o, license: l, redistributable: false}]")
    with pytest.raises(ValueError, match="must match"):
        load_sources(bad)
    bad.write_text("sources: [{name: x, root: /, paths: [], origin: o, license: l, redistributable: 'no'}]")
    with pytest.raises(ValueError, match="bool"):
        load_sources(bad)
    bad.write_text("sources: [{name: x, root: /, paths: [], origin: o, license: l}]")
    with pytest.raises(ValueError, match="redistributable"):
        load_sources(bad)
    bad.write_text(
        "sources: [{name: x, root: /, paths: [], origin: o, license: l, redistributable: true,"
        " formats: [nope]}]"
    )
    with pytest.raises(ValueError, match="unknown format"):
        load_sources(bad)
    bad.write_text("sources: {}")
    with pytest.raises(ValueError, match="sources"):
        load_sources(bad)


# --------------------------------------------------------------------------- verify / list


def test_verify_detects_edits_and_deletions(corpus_tree: dict[str, Path]) -> None:
    root, data = corpus_tree["root"], corpus_tree["data"]
    write_manifest(root, build_manifest(root, corpus_tree["sources"]).manifest)
    assert verify_manifest(root) == []

    flat = data / "MAD" / "BTL.FLAT"
    flat.write_text(MAD8.replace("K1=-2.18472718", "K1=-2.18472719"))  # same size
    problems = verify_manifest(root)
    assert [(p.id, p.kind) for p in problems] == [("anchors/mad/btl.flat", "sha256")]

    (data / "MAD" / "fodo.madx").write_text(MADX + "! edited\n")
    (root / "local" / "fodo_cell.dat").unlink()
    kinds = {p.id: p.kind for p in verify_manifest(root)}
    assert kinds == {
        "anchors/mad/btl.flat": "sha256",
        "anchors/mad/fodo.madx": "size",
        "local/local/fodo_cell.dat": "missing",
    }


def test_iter_decks_filters(corpus_tree: dict[str, Path]) -> None:
    root = corpus_tree["root"]
    write_manifest(root, build_manifest(root, corpus_tree["sources"]).manifest)
    all_entries = list(iter_decks(root))
    assert len(all_entries) == 8
    assert all(Path(e["path"]).is_absolute() and Path(e["path"]).is_file() for e in all_entries)
    assert {e["format"] for e in iter_decks(root, fmt="tracewin")} == {"tracewin"}
    assert len(list(iter_decks(root, fmt=("madx", "mad8")))) == 2
    assert [e["id"] for e in iter_decks(root, redistributable=True)] == ["local/local/fodo_cell.dat"]
    assert len(list(iter_decks(root, redistributable=False))) == 7
    assert [e["id"] for e in iter_decks(root, source="anchors", fmt="elegant")] == [
        "anchors/mad/elegant_lattice.lte"
    ]


def test_root_from_environment(corpus_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root = corpus_tree["root"]
    monkeypatch.setenv(corpus.ENV_VAR, str(root))
    assert corpus.corpus_root() == root.resolve()
    monkeypatch.delenv(corpus.ENV_VAR)
    with pytest.raises(ValueError, match=corpus.ENV_VAR):
        corpus.corpus_root()
    with pytest.raises(FileNotFoundError, match="build-manifest"):
        load_manifest(root)


def test_cli_build_verify_list(corpus_tree: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    root, sources, data = corpus_tree["root"], corpus_tree["sources"], corpus_tree["data"]
    assert main(["build-manifest", "--root", str(root), "--sources", str(sources)]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and "(8 entries)" in out
    assert "tracewin" in out and "binary" in out and "filtered" in out

    assert main(["verify", "--root", str(root)]) == 0
    assert "OK: 8 entries verified" in capsys.readouterr().out

    assert main(["list", "--root", str(root), "--format", "tracewin"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("3 entries")
    assert "pipii-tracewin/sc-linac/latf.dat" in out and "btl.flat" not in out

    assert main(["list", "--root", str(root), "--redistributable"]) == 0
    out = capsys.readouterr().out
    assert "local/local/fodo_cell.dat" in out and out.strip().endswith("1 entries")

    (data / "MAD" / "BTL.FLAT").write_text("! gone\n")
    assert main(["verify", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "size" in out and "anchors/mad/btl.flat" in out and "FAILED: 1 of 8" in out


def test_cli_errors_are_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(corpus.ENV_VAR, raising=False)
    assert main(["verify"]) == 2
    assert corpus.ENV_VAR in capsys.readouterr().err
    assert main(["verify", "--root", str(tmp_path)]) == 2
    assert "no manifest" in capsys.readouterr().err
