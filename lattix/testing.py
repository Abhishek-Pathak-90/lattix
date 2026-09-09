"""Test guards shared by lattix's own suite and by downstream users.

One place decides whether an engine or a private data set is available, so
every skip carries the same reason string (mirrors HELIX ``tests/dataguard.py``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_DATA_REASON = (
    "undistributed reference data absent: {} — set LATTIX_CORPUS_DIR / HELIX_ROOT / "
    "TRACEWIN_EXE on a machine that has it"
)
_ENGINE_REASON = "engine '{}' not available in this environment ({})"


def corpus_dir() -> Path | None:
    v = os.environ.get("LATTIX_CORPUS_DIR")
    return Path(v).expanduser() if v else None


def helix_root() -> Path | None:
    v = os.environ.get("HELIX_ROOT")
    return Path(v).expanduser() if v else None


def tracewin_exe() -> Path | None:
    v = os.environ.get("TRACEWIN_EXE")
    return Path(v).expanduser() if v else None


_UNSET = "/lattix-not-configured"          # a path that exists on no machine: the usual skips fire


def helix_path(*parts: str) -> Path:
    """A path inside the HELIX checkout named by ``HELIX_ROOT``; a nonexistent path when it is unset."""
    root = helix_root()
    return (root if root is not None else Path(_UNSET, "HELIX_ROOT")).joinpath(*parts)


def codes_path(*parts: str) -> Path:
    """A path inside the local mirror of upstream code repositories (``LATTIX_CODES_DIR``: the clones
    the readers and oracles were measured against); a nonexistent path when it is unset."""
    v = os.environ.get("LATTIX_CODES_DIR")
    return (Path(v).expanduser() if v else Path(_UNSET, "LATTIX_CODES_DIR")).joinpath(*parts)


def conda_env_bin(env: str, name: str) -> Path:
    """``<conda root>/envs/<env>/bin/<name>`` in the first conda root that has it; nonexistent otherwise."""
    from lattix.oracles.envs import conda_roots

    for root in conda_roots():
        p = root / "envs" / env / "bin" / name
        if p.exists():
            return p
    return Path(_UNSET, "conda", env, "bin", name)


def require_data(*paths: str | Path) -> None:
    """Module-level skip when any of *paths* is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(_DATA_REASON.format(", ".join(missing)), allow_module_level=True)


def needs_data(*paths: str | Path):
    """Decorator form of :func:`require_data`."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=_DATA_REASON.format(", ".join(missing)))


def require(engine: str) -> None:
    """Module-level skip when the named oracle engine is unavailable."""
    from lattix.oracles import get_oracle

    o = get_oracle(engine)
    ok, why = o.available()
    if not ok:
        pytest.skip(_ENGINE_REASON.format(engine, why), allow_module_level=True)


def needs(engine: str):
    """Decorator form of :func:`require`."""
    from lattix.oracles import get_oracle

    ok, why = get_oracle(engine).available()
    return pytest.mark.skipif(not ok, reason=_ENGINE_REASON.format(engine, why))
