"""Lockstep anchor (PLAN §5.4): the PIP-II BTL exported by two tools — MAD8 flat file
and Elegant `.lte` — must describe the same machine in the IR."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from lattix.ir import Bend, Quadrupole, propagate
from lattix.testing import corpus_dir

pytestmark = pytest.mark.corpus

FLAT_ID = ("pipii-anchors/studies_and_related_material/beam_dynamics_studies/btl/"
           "btl_lattice_with_spacecharge/mad_lattice/btl2022v0922_newcol.flat")
LTE_ID = ("pipii-anchors/studies_and_related_material/beam_dynamics_studies/btl/"
          "btl_lattice_with_spacecharge/mad_lattice/elegant_lattice.lte")


def _deck(mid: str) -> Path:
    root = corpus_dir()
    if root is None:
        pytest.skip("LATTIX_CORPUS_DIR not set")
    from lattix.corpus import load_manifest

    for e in load_manifest(root):
        if e.get("id") == mid and Path(e["path"]).is_file():
            return Path(e["path"])
    pytest.skip(f"{mid} not in manifest")


def test_btl_mad8_and_elegant_exports_agree():
    for n in ("mad8", "elegant"):
        try:
            importlib.import_module(f"lattix.formats.{n}")
        except ModuleNotFoundError:
            pytest.skip(f"format {n!r} not implemented yet")
    from lattix.formats import read

    flat, lte = _deck(FLAT_ID), _deck(LTE_ID)
    a, ra = read(flat)
    ke = a.reference.kinetic_energy_eV
    b, rb = read(lte, species="h-", kinetic_energy_eV=ke)
    pa = [p for p in propagate(a) if isinstance(p.element, (Quadrupole, Bend))]
    pb = [p for p in propagate(b) if isinstance(p.element, (Quadrupole, Bend))]
    # The PIP-II .lte exports are value-less skeletons (`NAME: TYPE, , L` — elegant itself
    # rejects them), so only the STRUCTURE can be anchored: same magnets in the same order.
    assert len(pa) == len(pb), (len(pa), len(pb))
    assert [p.element.kind for p in pa] == [p.element.kind for p in pb]
    assert [p.name.upper() for p in pa] == [p.name.upper() for p in pb]
    assert any(e.code == "BARE_ATTRIBUTE_TOKEN" for e in rb.entries), "the .lte skeleton must be reported"
    print(f"BTL flat vs lte: {len(pa)} magnets in identical order (values absent in the .lte export)")
