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
