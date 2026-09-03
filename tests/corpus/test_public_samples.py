"""Every vendored public sample listed in ``tests/data/public/README.md`` must exist, sniff to
its declared format, carry its license copy, and the whole tree must stay small."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lattix.corpus import FORMATS, sniff_format

PUBLIC = Path(__file__).resolve().parents[1] / "data" / "public"
README = PUBLIC / "README.md"
MAX_TOTAL_BYTES = 5 * 1024 * 1024

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([a-z_]+)\s*\|")
_LICENSE_REF = re.compile(r"`LICENSES/([A-Za-z0-9_.-]+)`")


def _readme_rows() -> list[tuple[str, str]]:
    rows = [m.groups() for m in map(_ROW.match, README.read_text(encoding="utf-8").splitlines()) if m]
    assert rows, "README has no sample rows"
    return [(path, fmt) for path, fmt in rows]


ROWS = _readme_rows()


@pytest.mark.parametrize(("rel", "fmt"), ROWS, ids=[r for r, _ in ROWS])
def test_sample_exists_and_sniffs_to_declared_format(rel: str, fmt: str) -> None:
    path = PUBLIC / rel
    assert path.is_file(), f"{rel} listed in README but missing"
    assert fmt in FORMATS, f"{rel}: unknown format {fmt!r} in README"
    assert sniff_format(path) == fmt


def test_every_vendored_file_is_listed() -> None:
    listed = {PUBLIC / rel for rel, _ in ROWS}
    on_disk = {
        p for p in PUBLIC.rglob("*") if p.is_file() and p.name != "README.md" and "LICENSES" not in p.parts
    }
    assert on_disk == listed, (
        f"unlisted: {sorted(str(p.relative_to(PUBLIC)) for p in on_disk - listed)}; "
        f"missing: {sorted(str(p.relative_to(PUBLIC)) for p in listed - on_disk)}"
    )


def test_license_copies_present() -> None:
    refs = set(_LICENSE_REF.findall(README.read_text(encoding="utf-8")))
    assert refs, "README references no LICENSES/ files"
    for name in refs:
        assert (PUBLIC / "LICENSES" / name).is_file(), f"LICENSES/{name} missing"

    # every sample directory has a license reference in its README section
    def key(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    for top in sorted({Path(rel).parts[0] for rel, _ in ROWS}):
        assert any(key(top) in key(n) for n in refs), f"no LICENSES/ entry for {top}/"


def test_total_size_under_cap() -> None:
    total = sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file())
    assert total < MAX_TOTAL_BYTES, f"public samples are {total} bytes"
