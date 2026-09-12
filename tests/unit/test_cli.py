from __future__ import annotations

import pytest

from lattix.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "lattix" in capsys.readouterr().out


def test_oracles_lists_adapters(capsys):
    assert main(["oracles"]) == 0
    out = capsys.readouterr().out
    assert "madx" in out and "helix" in out and "tracewin" in out


def test_convert_roundtrip_lattix_json(tmp_path, capsys):
    from lattix.formats import write
    from lattix.ir import Drift, Lattice, ReferenceParticle, species

    lat = Lattice.from_sequence("l", [Drift(name="d", length=1.5)],
                                ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6))
    src = tmp_path / "a.lattix.json"
    write(lat, src)
    assert main(["convert", str(src), str(tmp_path / "b.lattix.json")]) == 0
    assert main(["inspect", str(tmp_path / "b.lattix.json"), "--elements"]) == 0
    out = capsys.readouterr()
    assert "Drift" in out.out and "L = 1.500000 m" in out.out
    assert main(["report", str(src), "--to", "lattix"]) == 0


def test_survey_table_csv_and_json(tmp_path, capsys):
    import csv
    import json
    import math

    from lattix.formats import write
    from lattix.ir import Bend, BendP, BodyShiftP, Drift, Lattice, Quadrupole, ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    lat = Lattice.from_sequence("l", [Drift(name="d", length=2.0),
                                      Bend(name="b", length=1.0, bend=BendP(angle=math.pi / 2)),
                                      Quadrupole(name="q", length=0.4, shift=BodyShiftP(y_offset=1e-3))], ref)
    src = tmp_path / "bend.lattix.json"
    write(lat, src)
    assert main(["survey", str(src)]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if not ln.startswith("#")]
    assert lines[0].split()[:4] == ["i", "name", "kind", "parent"] and "theta" in lines[0]
    cols, row = lines[0].split(), lines[2].split()          # the empty parent column collapses in the row
    theta = float(row[len(row) - (len(cols) - cols.index("theta"))])
    assert row[1] == "b" and theta == pytest.approx(-math.pi / 2, abs=1e-8)     # the table prints 9 digits
    # CSV of every frame, with the misalignment in the body columns
    out_csv = tmp_path / "s.csv"
    assert main(["survey", str(src), "--at", "all", "--csv", str(out_csv)]) == 0
    text = out_csv.read_text()
    rows = list(csv.DictReader(ln for ln in text.splitlines() if not ln.startswith("#")))
    assert [r["name"] for r in rows] == ["d", "b", "q"]
    assert float(rows[2]["body_X"]) != float(rows[2]["c_X"]) or float(rows[2]["body_Y"]) != float(rows[2]["c_Y"])
    assert text.startswith("# lattix ") and "start pose" in text
    # ignoring the shift, and a start pose, and JSON
    out_json = tmp_path / "s.json"
    assert main(["survey", str(src), "--at", "body", "--no-shift", "--x0", "1", "--theta0", "0.5",
                 "--json", str(out_json)]) == 0
    doc = json.loads(out_json.read_text())
    assert doc["start"] == {"x0": 1.0, "y0": 0.0, "z0": 0.0, "theta0": 0.5, "phi0": 0.0, "psi0": 0.0}
    assert doc["shift"] is False and doc["rows"][0]["X"] == pytest.approx(1.0 + 1.0 * math.sin(0.5))
    assert doc["rows"][0]["theta"] == pytest.approx(0.5) and doc["columns"][0] == "i"
    with pytest.raises(SystemExit):
        main(["survey", str(src), "--at", "middle"])
