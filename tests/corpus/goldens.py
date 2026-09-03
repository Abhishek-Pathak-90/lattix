"""The golden dozen: pinned structural/optics numbers for anchor decks (PLAN §5.4).

``python -m tests.corpus.goldens --write`` (re)measures every entry with the
engines available and rewrites ``golden.yaml``; ``test_golden_dozen.py``
re-measures and compares.  Public entries resolve under ``tests/data/public``;
private ones through the corpus manifest id (``LATTIX_CORPUS_DIR``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
GOLDEN = HERE / "golden.yaml"
PUBLIC = HERE.parent / "data" / "public"

# (key, public path | None, manifest id | None, format, extra kwargs for HELIX parsers)
ENTRIES: list[tuple[str, str | None, str | None, str]] = [
    ("fodo_madx", "helix/fodo.madx", None, "madx"),
    ("transport_madx", "helix/transport.madx", None, "madx"),
    ("fodo_cell_dat", "helix/fodo_cell.dat", None, "tracewin"),
    ("bend_line_dat", "helix/bend_line.dat", None, "tracewin"),
    ("mebt_dat", None, "helix-examples/examples/pipii/mebt/mebt.dat", "tracewin"),
    ("mebt_hwr_dat", None, "helix-examples/examples/pipii/mebt+hwr/mebt+hwr.dat", "tracewin"),
    ("btl_dat", None, "helix-examples/examples/pipii/btl/btl.dat", "tracewin"),
    ("btl2025_lat", None, "helix-pipii-root/btl2025v0703.lat", "mad8"),
    ("bal2025_flat", None, "helix-pipii-root/bal2025v0213.flat", "mad8"),
    ("btl2022_flat", None, "pipii-anchors/studies_and_related_material/beam_dynamics_studies/btl/"
                            "btl_lattice_with_spacecharge/mad_lattice/btl2022v0922_newcol.flat", "mad8"),
    ("btl2022_lte", None, "pipii-anchors/studies_and_related_material/beam_dynamics_studies/btl/"
                          "btl_lattice_with_spacecharge/mad_lattice/elegant_lattice.lte", "elegant"),
    ("hwr_cm_lte", None, "pipii-anchors/studies_and_related_material/virtual-accelerator/prototypes/"
                         "v1_14_august_2024/tracewin_elegant_lattice.lte", "elegant"),
    ("fnalscl_dat", None, "helix-examples/examples/piplattice/fnalscl.dat", "tracewin"),
    ("lebt_pxie_dat", None, "helix-examples/examples/lebt_pxie/lebt_pxie.dat", "tracewin"),
    # booster/new/booster_pip2_20250722.seq is not self-contained (fmag/dmag classes come from
    # its driver run_beam_dynamics.madx) — revisit when the MAD-X reader handles CALL chains.
]


def resolve(public: str | None, manifest_id: str | None) -> Path | None:
    if public:
        p = PUBLIC / public
        return p if p.is_file() else None
    from lattix.corpus import load_manifest
    from lattix.testing import corpus_dir

    root = corpus_dir()
    if root is None or not (root / "manifest.yaml").is_file():
        return None
    for e in load_manifest(root):
        if e.get("id") == manifest_id:
            p = Path(e["path"])
            return p if p.is_file() else None
    return None


def measure_helix(path: Path, fmt: str) -> dict:
    from lattix.oracles.helix import HelixOracle, _import_helix

    _import_helix()
    lat, meta = HelixOracle._parse(path, fmt, None)
    return {
        "helix_n_elements": len(lat.elements),
        "helix_total_length_m": round(float(lat.total_length) * 1e-3, 9),
        "helix_n_warnings": len(meta.get("warnings", [])) if isinstance(meta, dict) else 0,
    }


def measure_madx(path: Path) -> dict:
    from cpymad.madx import Madx

    m = Madx(stdout=False)
    import tempfile

    m.chdir(tempfile.mkdtemp(prefix="lattix_golden_"))
    m.call(str(path))
    seqs = list(m.sequence.keys())
    seq = seqs[-1]
    m.use(sequence=seq)
    tw = m.twiss(sequence=seq, betx=10, bety=10)
    out = {"madx_sequence": seq, "madx_n_rows": len(tw.name),
           "madx_length_m": round(float(tw.s[-1]), 9),
           "madx_betx_end": round(float(tw.betx[-1]), 6),
           "madx_bety_end": round(float(tw.bety[-1]), 6)}
    m.quit()
    return out


def measure(key: str, path: Path, fmt: str) -> dict:
    out: dict = {"format": fmt, "file": path.name}
    try:
        out.update(measure_helix(path, fmt))
    except Exception as e:  # noqa: BLE001
        out["helix_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    if fmt == "madx":
        try:
            out.update(measure_madx(path))
        except Exception as e:  # noqa: BLE001
            out["madx_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def measure_all(only: set[str] | None = None) -> dict:
    res = {}
    for key, public, mid, fmt in ENTRIES:
        if only and key not in only:
            continue
        p = resolve(public, mid)
        if p is None:
            continue
        res[key] = measure(key, p, fmt)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    res = measure_all()
    if a.write:
        old = yaml.safe_load(GOLDEN.read_text()) if GOLDEN.exists() else {}
        old.update(res)
        GOLDEN.write_text(yaml.safe_dump(old, sort_keys=True))
        print(f"wrote {len(res)} entries to {GOLDEN}")
    for k, v in res.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
