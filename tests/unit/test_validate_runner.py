"""run_validation: the loop behind `lattix validate` and the UI's engine job."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lattix.cli import main
from lattix.oracles.base import SPECIES, Basis, BeamSpec, OracleResult, register
from lattix.oracles.basis import drift_common
from lattix.oracles.validate import comparison_to_dict, engines_for, outcome_to_dict, run_validation

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
MP = SPECIES["proton"][0]


def _res(engine, n=3, ke=2.1e6):
    d = drift_common(0.5, ke, MP)
    return OracleResult(engine=engine, basis=Basis.COMMON, names=[f"d{i}" for i in range(n)],
                        length=np.full(n, 0.5), s_out=np.cumsum(np.full(n, 0.5)), R_elem=np.array([d] * n),
                        ref_kinetic_eV_in=np.full(n, ke), ref_kinetic_eV_out=np.full(n, ke), mass_eV=MP, charge=1)


class _A:
    name, formats = "vfake_a", ("tracewin",)

    def available(self):
        return True, "fake"

    def run(self, deck, *, fmt=None, beam=None, probe=None, workdir=None):
        return _res(self.name)


class _B(_A):
    name, formats = "vfake_b", ("elegant", "tracewin")


class _Off(_A):
    name, formats = "vfake_off", ("tracewin",)

    def available(self):
        return False, "not installed"


class _Boom(_A):
    name, formats = "vfake_boom", ("tracewin",)

    def run(self, deck, *, fmt=None, beam=None, probe=None, workdir=None):
        raise RuntimeError("engine exploded")


for cls in (_A, _B, _Off, _Boom):
    register(cls)


def test_run_validation_runs_pairs_records_skips_and_errors(tmp_path):
    decks = {"tracewin": DATA / "helix" / "fodo_cell.dat", "elegant": tmp_path / "x.lte"}
    log, prog = [], []
    o = run_validation(decks, ["vfake_a", ("vfake_b", "elegant"), "vfake_off", "vfake_boom", "nosuch"], BeamSpec(),
                       log=log.append, progress=lambda d, t, c: prog.append((d, t, c)))
    assert o.runs["vfake_a"].fmt == "tracewin" and o.runs["vfake_b"].fmt == "elegant"
    assert o.runs["vfake_off"].skipped == "not installed" and "unknown oracle" in o.runs["nosuch"].skipped
    assert "engine exploded" in o.runs["vfake_boom"].error
    assert len(o.comparisons) == 1 and o.comparisons[0].n_shared == 3 and o.worst == 0.0
    assert any(line.startswith("vfake_a  tracewin") for line in log) and prog[-1] == (5, 5, "")
    d = outcome_to_dict(o, decks=decks, beam=BeamSpec())
    assert d["engines"]["vfake_a"]["n"] == 3 and d["comparisons"][0]["per_boundary"][0]["s"] == 0.5
    assert json.dumps(d)                                   # JSON-ready
    assert comparison_to_dict(o.comparisons[0])["row"].startswith(" vfake_a vs vfake_b")


def test_run_validation_pinned_format_without_deck_and_stop():
    o = run_validation({"tracewin": DATA / "helix" / "fodo_cell.dat"}, [("vfake_b", "elegant"), "vfake_a"],
                       BeamSpec(), should_stop=lambda: True)
    assert o.runs["vfake_b"].skipped == "stopped" and o.runs["vfake_a"].skipped == "stopped"
    o = run_validation({"tracewin": DATA / "helix" / "fodo_cell.dat"}, [("vfake_b", "elegant")], BeamSpec())
    assert "no deck" in o.runs["vfake_b"].skipped


def test_engines_for_formats():
    assert "vfake_a" in engines_for("tracewin") and "vfake_b" in engines_for("elegant")
    assert "madx" in engines_for("madx") and "helix" in engines_for("tracewin")


def test_cmd_validate_prints_rows_and_json(tmp_path, capsys):
    out = tmp_path / "v.json"
    rc = main(["validate", "--deck", f"tracewin={DATA / 'helix' / 'fodo_cell.dat'}", "--oracles",
               "vfake_a,vfake_b,vfake_off", "--json", str(out)])
    assert rc == 0
    cap = capsys.readouterr()
    assert "vfake_a vs vfake_b" in cap.out and "vfake_off skipped" in cap.err.replace("  ", " ")
    payload = json.loads(out.read_text())
    assert payload["comparisons"][0]["n_shared"] == 3 and payload["engines"]["vfake_a"]["names"] == ["d0", "d1", "d2"]
    assert main(["validate", "--deck", f"tracewin={DATA / 'helix' / 'fodo_cell.dat'}", "--oracles", "vfake_a"]) == 2
    with pytest.raises(SystemExit):
        main(["validate"])


def test_cmd_validate_verdict_with_tier_and_codes(tmp_path, capsys):
    out = tmp_path / "v.json"
    rc = main(["validate", "--deck", f"tracewin={DATA / 'helix' / 'fodo_cell.dat'}", "--oracles", "vfake_a,vfake_b",
               "--json", str(out), "--tier", "equivalent", "--codes", "APERTURE_DROPPED"])
    assert rc == 0
    cap = capsys.readouterr()
    assert "verdict: ok" in cap.out and "tier equivalent" in cap.out and "blocks compared:" in cap.out
    v = json.loads(out.read_text())["comparisons"][0]["verdict"]
    assert v["tier"] == "equivalent" and v["ok"] is True and "T4x4" in v["blocks_used"]
    assert v["map_tol"] is not None and v["metric_rel"] is not None


def test_constant_p0_engine_is_skipped_with_its_reason(tmp_path, monkeypatch):
    from lattix import crossval
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=352.2e6)
    hot = Lattice.from_sequence("hot", [Drift(name="d", length=0.1),
                                        RFCavity(name="c", length=0.0, rf=RFP(voltage_V=5e8, phase_rad=0.0,
                                                                              frequency_Hz=352.2e6)),
                                        Drift(name="e", length=0.1)], ref)
    monkeypatch.setitem(crossval.FOLLOWS_P0, "vfake_a", False)
    beam = BeamSpec(species="proton", kinetic_energy_eV=2.1e6)
    deck = DATA / "helix" / "fodo_cell.dat"
    o = run_validation({"tracewin": deck}, ["vfake_a", "vfake_b"], beam, workdir=tmp_path, lat=hot)
    assert o.runs["vfake_a"].skipped and "keeps p0 constant" in o.runs["vfake_a"].skipped
    assert o.runs["vfake_b"].result is not None and not o.comparisons


def test_cmd_validate_writes_json_even_with_one_engine(tmp_path):
    out = tmp_path / "v.json"
    rc = main(["validate", "--deck", f"tracewin={DATA / 'helix' / 'fodo_cell.dat'}", "--oracles", "vfake_a,vfake_off",
               "--json", str(out)])
    assert rc == 2 and out.exists()
    payload = json.loads(out.read_text())
    assert payload["engines"]["vfake_off"]["skipped"] and payload["comparisons"] == []
