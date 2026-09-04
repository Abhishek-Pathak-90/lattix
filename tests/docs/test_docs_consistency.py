"""The documentation stays in step with the source (PLAN §6 task 4.3).

Modelled on HELIX ``tests/core/test_docs_consistency.py``: the pages are checked against the
code they describe, so a new fidelity code, element kind, phase conversion, format or version
bump fails here until the docs follow.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FORMAT_PAGE_SECTIONS = ("## Reading", "## Writing", "## Units and conventions", "## Known limits", "## Oracle")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_fidelity_catalogue_block_is_current():
    from lattix.fidelity_catalog import scan, to_markdown

    text = _read(DOCS / "fidelity.md")
    m = re.search(r"<!-- BEGIN fidelity-catalog -->\n(.*?)<!-- END fidelity-catalog -->", text, re.S)
    assert m, "docs/fidelity.md lacks the generated fidelity-catalog block"
    expected = to_markdown(scan()).strip()
    assert m.group(1).strip() == expected, (
        "docs/fidelity.md is stale: regenerate the block with "
        "`python -m lattix.fidelity_catalog --markdown`")


def test_conventions_name_every_element_kind():
    from lattix.ir.elements import ALL_KINDS

    text = _read(DOCS / "conventions.md")
    missing = sorted(k for k in ALL_KINDS if f"`{k}`" not in text)
    assert not missing, f"docs/conventions.md does not mention kinds {missing}"


def test_conventions_name_every_phase_conversion():
    import lattix.ir.rf as rf

    names = [n for n, v in vars(rf).items()
             if callable(v) and not n.startswith("_") and getattr(v, "__module__", "") == rf.__name__]
    assert names
    text = _read(DOCS / "conventions.md")
    missing = [n for n in names if not re.search(rf"`{n}\b", text)]
    assert not missing, f"docs/conventions.md does not mention rf conversions {missing}"


def test_every_format_has_a_page_and_is_indexed():
    from lattix.formats.base import FORMATS

    index = _read(DOCS / "index.md")
    for key in [*FORMATS, "helix"]:
        page = DOCS / "formats" / f"{key}.md"
        assert page.exists(), f"missing {page.relative_to(ROOT)}"
        assert f"formats/{key}.md" in index, f"docs/index.md does not link formats/{key}.md"
        text = _read(page)
        for section in FORMAT_PAGE_SECTIONS:
            assert section in text, f"{page.name} lacks the section {section!r}"


def test_format_pages_only_cite_real_fidelity_codes():
    """A backticked ALL_CAPS token that is *claimed* as a fidelity code must exist in the catalogue."""
    from lattix.fidelity_catalog import scan

    codes = set(scan())
    bad = []
    for page in sorted((DOCS / "formats").glob("*.md")):
        for cls, code in re.findall(r"\b(EXACT|EQUIVALENT|LOSSY|DROPPED)\s+`([A-Z][A-Z0-9_]+)`", _read(page)):
            if code not in codes:
                bad.append(f"{page.name}: {cls} `{code}`")
    assert not bad, f"unknown fidelity codes cited: {bad}"


def test_version_is_consistent():
    import lattix

    project = tomllib.loads(_read(ROOT / "pyproject.toml"))["project"]["version"]
    cff = re.search(r"^version:\s*(\S+)", _read(ROOT / "CITATION.cff"), re.M).group(1)
    assert project == lattix.__version__ == cff


@pytest.mark.parametrize("doc", ["README.md", "docs/index.md", "docs/tutorial.md", "docs/fidelity.md",
                                 "docs/conventions.md", "docs/oracles.md", "docs/corpus.md"])
def test_relative_links_resolve(doc):
    path = ROOT / doc
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", _read(path)):
        target = target.split("#", 1)[0]
        if not target or re.match(r"[a-z]+://", target):
            continue
        if not (path.parent / target).exists():
            broken.append(target)
    assert not broken, f"{doc}: broken links {broken}"


def test_tutorial_uses_real_subcommands(capsys):
    from lattix.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    listed = set(re.search(r"\{([a-z,]+)\}", capsys.readouterr().out).group(1).split(","))
    used = set(re.findall(r"^\$ lattix (\w+)", _read(DOCS / "tutorial.md"), re.M))
    assert used, "tutorial shows no `$ lattix <command>` lines"
    assert used <= listed, f"tutorial uses unknown subcommands {used - listed}"
