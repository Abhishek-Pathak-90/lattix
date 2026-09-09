"""Draw the README's label strip from the code itself, so the numbers on it cannot go stale.

    python tools/readme_badges.py            # writes docs/assets/readme-badges.png

The counts come from the format registry, the oracle registry and the fidelity catalogue; the
licence and the Python floor come from pyproject.toml. No external badge service is involved.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import tomllib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "readme-badges.png"
SS = 2                                                     # drawn at twice the size, for crisp text

INK = (196, 205, 218)
VALUE_INK = (16, 18, 22)
PLATE = (41, 47, 58)
BLUE = (83, 163, 242)
CYAN = (34, 211, 238)
GREEN = (74, 222, 128)
AMBER = (251, 191, 36)
INDIGO = (99, 133, 214)


def _font(size: int, bold: bool):
    """The project's display face where it exists, else a portable fallback."""
    px = size * SS
    for path, index in (("/System/Library/Fonts/Avenir Next.ttc", 2 if bold else 5),):
        try:
            return ImageFont.truetype(path, px, index=index)
        except OSError:
            pass
    try:
        import matplotlib
        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        return ImageFont.truetype(str(base / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), px)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def facts() -> list[tuple[str, str, tuple[int, int, int]]]:
    sys.path.insert(0, str(ROOT))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    import lattix.oracles as oracles_pkg
    from lattix.formats.base import FORMATS
    for m in pkgutil.iter_modules(oracles_pkg.__path__):
        try:
            importlib.import_module(f"lattix.oracles.{m.name}")
        except Exception:  # noqa: BLE001  (an engine's optional dependency may be missing here)
            pass
    from lattix.fidelity_catalog import scan
    from lattix.oracles.base import _REGISTRY
    codes = len([c for c in scan() if c != "OK"])
    licence = project["license"] if isinstance(project["license"], str) else project["license"]["text"]
    licence = {"BSD-3-Clause": "BSD 3-Clause"}.get(licence, licence)
    python = project["requires-python"].replace(">=", "") + "+"
    return [("licence", licence, BLUE), ("python", python, INDIGO), ("formats", str(len(FORMATS)), CYAN),
            ("engines", str(len(_REGISTRY)), GREEN), ("ledger codes", str(codes), AMBER)]


def draw(badges) -> Image.Image:
    fl, fv = _font(15, False), _font(15, True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    width = lambda s, f: probe.textbbox((0, 0), s, font=f)[2] / SS      # noqa: E731
    H, PADX, GAP, R = 30, 13, 10, 6
    pieces = [(lab, val, col, width(lab, fl) + 2 * PADX, width(val, fv) + 2 * PADX) for lab, val, col in badges]
    total = sum(wl + wv for _, _, _, wl, wv in pieces) + GAP * (len(pieces) - 1)
    im = Image.new("RGBA", (int((total + 4) * SS), int((H + 4) * SS)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 2.0
    for lab, val, col, wl, wv in pieces:
        d.rounded_rectangle([x * SS, 2 * SS, (x + wl + wv) * SS, (2 + H) * SS], radius=R * SS, fill=PLATE + (255,))
        d.rounded_rectangle([(x + wl - R) * SS, 2 * SS, (x + wl + wv) * SS, (2 + H) * SS], radius=R * SS,
                            fill=col + (255,))
        d.rectangle([(x + wl) * SS, 2 * SS, (x + wl + R) * SS, (2 + H) * SS], fill=col + (255,))
        d.text(((x + wl / 2) * SS, (2 + H / 2) * SS), lab, font=fl, fill=INK + (255,), anchor="mm")
        d.text(((x + wl + wv / 2) * SS, (2 + H / 2) * SS), val, font=fv, fill=VALUE_INK + (255,), anchor="mm")
        x += wl + wv + GAP
    return im


if __name__ == "__main__":
    badges = facts()
    draw(badges).save(OUT, optimize=True)
    print(OUT.relative_to(ROOT), " ".join(f"{lab}={val}" for lab, val, _ in badges))
