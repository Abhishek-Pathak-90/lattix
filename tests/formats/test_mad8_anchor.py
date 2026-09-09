"""LOCKSTEP ANCHOR — the PIP-II BTL through two independent lattix front ends.

``BTL2025v0703.lat`` (MAD8 flat, the optics group's master file) and
``examples/pipii/btl/btl_2025v0703.dat`` (its TraceWin export, produced years apart by a
separate conversion script) describe the same machine.  The two readers share nothing but
the IR element classes, so a systematic conversion error on either side cannot cancel:
this test aligns the physical elements by position and compares them one by one.

This is lattix's counterpart of HELIX ``tests/io/test_mad8_parser.py::test_btl_anchor_lockstep``,
which is marked ``xfail(strict=True)`` there.  HELIX's mismatch was a *representation*
problem: its ``Dipole`` has no reference tilt, so a vertical bend had to become ρ > 0 plus an
``hv`` flag with the sign moved into the angle, and its ``Edge`` uses ``β = sign(θ)·e`` —
which disagrees with what the export script wrote for the negative-angle vertical bends.
The lattix IR stores MAD-X/PALS quantities (signed ``angle``, sector-referenced ``e1``/``e2``,
``tilt_ref``), so both paths land on the same numbers and the anchor *passes*.

Measured on 2026-09-03 (873 aligned physical elements):

===================================  ==========
quantity                             max |Δ|
===================================  ==========
element length                       4.0e-10 m
quadrupole gradient (relative)       1.4e-10
bend angle                           7.4e-12 rad
bend e1 / e2                         7.8e-12 rad
total length                         2.5e-8 m
===================================  ==========

The single systematic difference is the *sign* of ``tilt_ref`` on the four vertical bends
(BVDU, BVDD, ORB1, ORB2): the MAD8 file says ``TILT = -pi/2``, the TraceWin export encodes
only ``hv = 1`` (i.e. ``+pi/2``).  HELIX documented the same thing — the export script drops
the sign, a global vertical mirror that leaves the optics unchanged.  The assertion below
pins it at exactly four elements so a *new* tilt disagreement would fail.
"""
from __future__ import annotations

import pytest

from lattix.formats.mad8 import Reader as Mad8Reader
from lattix.formats.mad8.reader import _significant_cards
from lattix.formats.tracewin import Reader as TraceWinReader
from lattix.testing import corpus_dir, helix_path

pytestmark = [pytest.mark.corpus, pytest.mark.oracle_helix]

#: ids in ``$LATTIX_CORPUS_DIR/manifest.yaml``; the fallbacks are this machine's paths.
MAD8_ID = "helix-pipii-root/btl2025v0703.lat"
TRACEWIN_ID = "helix-examples/examples/pipii/btl/btl_2025v0703.dat"
_HELIX = str(helix_path())
FALLBACK = {MAD8_ID: f"{_HELIX}/BTL2025v0703.lat",
            TRACEWIN_ID: f"{_HELIX}/examples/pipii/btl/btl_2025v0703.dat"}

TOL_LENGTH_M = 1e-9
TOL_GRADIENT_REL = 1e-9
TOL_ANGLE_RAD = 1e-9
#: BVDU, BVDD, ORB1, ORB2 — the export script writes ``hv=1`` and loses ``TILT = -pi/2``.
EXPECTED_TILT_SIGN_FLIPS = 4


def _deck(deck_id: str) -> str:
    from pathlib import Path

    root = corpus_dir()
    if root is not None and (root / "manifest.yaml").is_file():
        from lattix.corpus import load_manifest, resolve_path

        for entry in load_manifest(root):
            if entry["id"] == deck_id:
                path = resolve_path(root, entry)
                if path.is_file():
                    return str(path)
    fallback = Path(FALLBACK[deck_id])
    if fallback.is_file():
        return str(fallback)
    pytest.skip(f"{deck_id} is not available (set LATTIX_CORPUS_DIR)")
    raise AssertionError                                        # pragma: no cover


def _stream(lat):
    """The physical elements in order, as ``(tag, length, strengths…, name)`` rows.

    Markers, monitors and directives carry no transport and are dropped; a zero-strength
    kicker and a zero-angle placeholder bend are drifts of their own length in both decks
    (the export script wrote them that way), so they are tagged ``D``.
    """
    rows = []
    for p in lat.flatten():
        e = p.element
        kind = e.kind
        if kind in ("Marker", "Instrument", "Directive", "Freq", "Collimator", "Patch"):
            continue
        if kind == "Drift":
            if e.length == 0.0:
                continue                    # bookkeeping drift, present in both decks
            rows.append(("D", e.length, 0.0, 0.0, 0.0, 0.0, e.name, p.s_in))
        elif kind == "Kicker":
            assert not (e.hkick or e.vkick), f"{e.name}: the BTL decks carry no live kicks"
            rows.append(("D", e.length, 0.0, 0.0, 0.0, 0.0, e.name, p.s_in))
        elif kind == "Quadrupole":
            rows.append(("Q", e.length, e.multipole.Bn.get(1, 0.0), 0.0, 0.0,
                         e.multipole.tilt.get(1, 0.0), e.name, p.s_in))
        elif kind == "Bend":
            if e.bend.angle == 0.0:
                rows.append(("D", e.length, 0.0, 0.0, 0.0, 0.0, e.name, p.s_in))
            else:
                rows.append(("B", e.length, e.bend.angle, e.bend.e1, e.bend.e2,
                             e.bend.tilt_ref, e.name, p.s_in))
        else:                               # pragma: no cover - neither deck has others
            rows.append((kind, e.length, 0.0, 0.0, 0.0, 0.0, e.name, p.s_in))
    return rows


@pytest.fixture(scope="module")
def decks():
    mad8, _ = Mad8Reader().read(_deck(MAD8_ID))
    tracewin, rep = TraceWinReader().read(
        _deck(TRACEWIN_ID), species="h-",
        kinetic_energy_eV=mad8.reference.kinetic_energy_eV)
    assert rep.ok, rep.summary()
    return mad8, tracewin


def test_total_length_matches_the_helix_anchor(decks):
    mad8, tracewin = decks
    assert mad8.total_length == pytest.approx(307.969918, abs=1e-5)     # 307 969.918 mm
    assert mad8.total_length == pytest.approx(tracewin.total_length, abs=1e-6)


def test_reference_particle(decks):
    mad8, _ = decks
    assert mad8.reference.species.name == "h-"
    assert mad8.reference.brho_abs == pytest.approx(4.881, rel=1e-12)
    assert mad8.reference.kinetic_energy_eV / 1e6 == pytest.approx(799.52, abs=0.2)


def test_physical_elements_align_one_to_one(decks):
    mad8, tracewin = decks
    a, b = _stream(mad8), _stream(tracewin)
    assert len(a) == len(b) == 873
    mismatched = [(i, a[i][6], a[i][0], b[i][6], b[i][0])
                  for i in range(len(a)) if a[i][0] != b[i][0]]
    assert mismatched == []


def test_per_element_agreement(decks):
    """Lengths, quadrupole gradients and bend geometry, element by element."""
    a, b = _stream(decks[0]), _stream(decks[1])
    worst = {"length": 0.0, "s_in": 0.0, "quad_gradient_rel": 0.0,
             "bend_angle": 0.0, "bend_e1": 0.0, "bend_e2": 0.0, "bend_tilt_abs": 0.0}
    tilt_sign_flips = []
    edge_sign_flips = []
    for x, y in zip(a, b, strict=True):
        worst["length"] = max(worst["length"], abs(x[1] - y[1]))
        worst["s_in"] = max(worst["s_in"], abs(x[7] - y[7]))
        if x[0] == "Q":
            scale = max(abs(x[2]), abs(y[2]), 1e-30)
            worst["quad_gradient_rel"] = max(worst["quad_gradient_rel"], abs(x[2] - y[2]) / scale)
        elif x[0] == "B":
            worst["bend_angle"] = max(worst["bend_angle"], abs(x[2] - y[2]))
            # MEASURED 2026-09-03: the two negative-angle vertical bends read with opposite pole-face
            # signs from the two formats.  The bend-face work of 2026-09-04 to 2026-09-06 (758a8bb,
            # 86b06ca) settled the negative-bend convention against the engines, MAD-X's identity
            # (angle<0, e1, e2) = (angle>0, tilt pi, -e1, -e2), after which both readers agree and no
            # sign flip is expected.  Any element listed here is a regression in one of the readers.
            d1, d2 = abs(x[3] - y[3]), abs(x[4] - y[4])
            if abs(x[3] + y[3]) < d1 and abs(x[4] + y[4]) < d2 and abs(x[3]) > 1e-12:
                edge_sign_flips.append(x[6])
                d1, d2 = abs(x[3] + y[3]), abs(x[4] + y[4])
            worst["bend_e1"] = max(worst["bend_e1"], d1)
            worst["bend_e2"] = max(worst["bend_e2"], d2)
            worst["bend_tilt_abs"] = max(worst["bend_tilt_abs"], abs(abs(x[5]) - abs(y[5])))
            if x[5] != y[5]:
                tilt_sign_flips.append(x[6])
    print("\nBTL MAD8 vs TraceWin, worst per-element difference:")
    for key, value in worst.items():
        print(f"  {key:20s} {value:.3e}")
    print(f"  tilt sign flips      {sorted(tilt_sign_flips)}")

    assert worst["length"] < TOL_LENGTH_M
    assert worst["s_in"] < 1e-7                      # accumulated over 308 m
    assert worst["quad_gradient_rel"] < TOL_GRADIENT_REL
    assert worst["bend_angle"] < TOL_ANGLE_RAD
    assert worst["bend_e1"] < TOL_ANGLE_RAD
    assert worst["bend_e2"] < TOL_ANGLE_RAD
    assert worst["bend_tilt_abs"] < TOL_ANGLE_RAD
    # the export script's known vertical mirror, and nothing else
    assert sorted(tilt_sign_flips) == ["BVDD", "BVDU", "ORB1", "ORB2"]
    # the readers agree on every pole face, the two negative vertical bends included (see above)
    assert edge_sign_flips == []
    assert len(tilt_sign_flips) == EXPECTED_TILT_SIGN_FLIPS


def _brackets(lat) -> list[tuple[int, int]]:
    """``(n declared, significant elements inside)`` for every LATTICE bracket."""
    out: list[tuple[int, int]] = []
    open_n: int | None = None
    inside = 0
    for p in lat.flatten():
        e = p.element
        if e.kind == "Directive" and e.role == "period_start":
            open_n, inside = int(float(e.args[0])), 0
        elif e.kind == "Directive" and e.role == "period_end":
            assert open_n is not None
            out.append((open_n, inside))
            open_n = None
        elif open_n is not None:
            inside += _significant_cards(e)
    return out


def test_periodicity_brackets_match_the_tracewin_export(decks):
    """The LINE hierarchy the MAD8 file carries reproduces the ten hand-checked
    ``LATTICE n 0`` brackets of the TraceWin deck, count for count."""
    mad8, tracewin = decks
    a, b = _brackets(mad8), _brackets(tracewin)
    assert len(a) == 10
    assert a == b
    assert [n for n, _ in a] == [41, 37, 27, 32, 34, 41, 43, 37, 41, 37]
    for n, inside in a:
        assert inside % n == 0, (n, inside)          # a whole number of periods
    assert [inside // n for n, inside in a] == [2, 1, 1, 1, 1, 1, 1, 6, 1, 2]


def test_writing_the_mad8_lattice_as_tracewin_reproduces_the_lattice_cards(decks, tmp_path):
    """End to end: MAD8 → IR → TraceWin must emit the export's own ``LATTICE`` cards, and
    each bracket must still hold a whole number of periods once TraceWin has expanded the
    bends into EDGE+BEND+EDGE.

    The ``n`` this reader declares is a count of *TraceWin cards*, so it is coupled to what
    :mod:`lattix.formats.tracewin.writer` emits per IR element (today: one card for a thick
    kicker, three for a bend with pole faces).  This test is where that coupling breaks
    loudly if either side changes.
    """
    from lattix.formats.tracewin import Reader as TraceWinReader2
    from lattix.formats.tracewin import Writer as TraceWinWriter

    mad8, _ = decks
    out = tmp_path / "btl_from_mad8.dat"
    rep = TraceWinWriter().write(mad8, out)
    cards = [ln.split(";")[0].split()
             for ln in out.read_text(encoding="latin-1").splitlines()
             if ln.split(";")[0].strip().upper().startswith("LATTICE")]
    assert [c[1] for c in cards if c[0].upper() == "LATTICE"] == \
        ["41", "37", "27", "32", "34", "41", "43", "37", "41", "37"]
    assert sum(1 for c in cards if c[0].upper() == "LATTICE_END") == 10

    written, _ = TraceWinReader2().read(
        out, species="h-", kinetic_energy_eV=mad8.reference.kinetic_energy_eV)
    assert _brackets(written) == _brackets(mad8)

    # the TraceWin writer's own downgrades on this deck, for the record
    assert {"INSTRUMENT_AS_MARKER", "ZERO_ANGLE_BEND_AS_DRIFT"} <= set(rep.codes())
    # zero-kick thick correctors are written as their body drift, so nothing is lost
    kicker_length = sum(p.length for p in mad8.flatten() if p.element.kind == "Kicker")
    assert kicker_length == pytest.approx(3.114)          # 51 x 60 mm + 1 x 54 mm
    assert written.total_length == pytest.approx(mad8.total_length, abs=1e-6)
