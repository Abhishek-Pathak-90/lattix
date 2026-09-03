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


@pytest.mark.parametrize("cmd", ["convert", "inspect", "report"])
def test_phase1_commands_not_yet(cmd, capsys):
    assert main([cmd, "x"]) == 3
    assert "Phase 1" in capsys.readouterr().err
