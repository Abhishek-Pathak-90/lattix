"""Set the release version everywhere it lives, in one step.

    python tools/bump_version.py 0.2.0rc1      # a rehearsal: TestPyPI when tagged v0.2.0rc1
    python tools/bump_version.py 0.2.0         # the release: PyPI when tagged v0.2.0

Touches lattix/_version.py, CITATION.cff (version and date), CHANGELOG.md (the Unreleased section
becomes the base version's section, dated today), the README's image links (pinned to the tag so a
PyPI page keeps the assets of its release) and its quoted banner line, then redraws the badge strip.
The version must be canonical PEP 440; the docs test enforces the same rule.
"""
from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    v = argv[1]
    parsed = Version(v)
    if str(parsed) != v:
        print(f"{v!r} is not canonical PEP 440; use {str(parsed)!r}")
        return 2
    base = parsed.base_version
    today = dt.date.today().isoformat()
    tag = f"v{v}"

    (ROOT / "lattix" / "_version.py").write_text(f'__version__ = "{v}"\n')

    cff = (ROOT / "CITATION.cff").read_text()
    cff = re.sub(r"^version:\s*\S+", f"version: {v}", cff, count=1, flags=re.M)
    cff = re.sub(r"^date-released:\s*\S+", f"date-released: {today}", cff, count=1, flags=re.M)
    (ROOT / "CITATION.cff").write_text(cff)

    log = (ROOT / "CHANGELOG.md").read_text()
    if f"## [{base}]" in log:
        log = re.sub(rf"^## \[{re.escape(base)}\].*$", f"## [{base}] - {today}", log, count=1, flags=re.M)
    else:
        log = log.replace("## [Unreleased]\n", f"## [Unreleased]\n\n## [{base}] - {today}\n", 1)
    (ROOT / "CHANGELOG.md").write_text(log)

    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"(raw\.githubusercontent\.com/Accel-Toolkit/lattix/)[^/]+(/docs/assets/)", rf"\g<1>{tag}\g<2>", readme)
    readme = re.sub(r"! lattix \d[\w.+!-]* from IR", f"! lattix {v} from IR", readme)
    (ROOT / "README.md").write_text(readme)

    subprocess.run([sys.executable, str(ROOT / "tools" / "readme_badges.py")], check=True)
    print(f"version {v} (changelog section {base}, dated {today}, README assets pinned to {tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
