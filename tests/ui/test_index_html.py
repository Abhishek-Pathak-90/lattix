"""The page ships as package data, works offline and its pure geometry passes under node (when present)."""
from __future__ import annotations

import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest

PAGE = resources.files("lattix.ui").joinpath("static/index.html")


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_is_self_contained_and_small():
    text = _text()
    assert len(text.encode("utf-8")) <= 400_000
    assert "<script src" not in text and 'rel="stylesheet"' not in text and "@import" not in text
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", text))
    assert urls <= {"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"}
    assert "<title>lattix</title>" in text and 'type="module"' in text
    assert "/* PURE-BEGIN */" in text and "/* PURE-END */" in text
    assert text.count("--k-Quadrupole:") >= 3 and "prefers-color-scheme: dark" in text


def _pure_block() -> str:
    text = _text()
    return text.split("/* PURE-BEGIN */", 1)[1].split("/* PURE-END */", 1)[0]


NODE_TEST = (Path(__file__).with_name("pure_check.js")).read_text()



@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_pure_block_under_node(tmp_path: Path):
    (tmp_path / "pure.js").write_text(_pure_block() + "\n" + NODE_TEST)
    r = subprocess.run(["node", str(tmp_path / "pure.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "pure ok" in r.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_module_script_parses_under_node(tmp_path: Path):
    m = re.search(r'<script type="module">\n(.*?)\n</script>', _text(), re.S)
    assert m
    (tmp_path / "app.mjs").write_text(m.group(1))
    r = subprocess.run(["node", "--check", str(tmp_path / "app.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_kind_colours_cover_every_kind():
    from lattix.ir.elements import ALL_KINDS

    text = _text()
    for kind in ALL_KINDS:
        assert f"--k-{kind}:" in text, kind
