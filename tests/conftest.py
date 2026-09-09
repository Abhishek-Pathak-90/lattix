"""Shared pytest configuration: engine/data guards come from lattix.testing."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from lattix.testing import corpus_dir, helix_root, tracewin_exe  # noqa: F401  (re-exported helpers)

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XSUITE_ALLOW_KERNEL_COMPILATION", "1")  # xtrack >= 0.112 JIT kernels


def pytest_sessionstart(session):
    """With LATTIX_EXPECT_INSTALLED set, the suite must exercise an installed wheel, not this checkout.

    The release workflow installs the built wheel and runs these tests from the checkout with
    ``--import-mode=append`` (the checkout goes to the end of sys.path, so ``tests.*`` imports work
    while ``lattix`` resolves to site-packages) and ``PYTHONSAFEPATH=1``.  If ``lattix`` still came
    from the checkout, every result would be about the wrong code, so that is an error, not a skip."""
    if os.environ.get("LATTIX_EXPECT_INSTALLED"):
        import lattix

        where = Path(lattix.__file__).resolve()
        if ROOT in where.parents:
            raise pytest.UsageError(f"lattix was imported from the checkout ({where}), not from the installed "
                                    "wheel; run with --import-mode=append and PYTHONSAFEPATH=1")


@pytest.fixture(scope="session")
def data_public():
    from pathlib import Path

    return Path(__file__).parent / "data" / "public"


# --- environment limits: a missing optional dependency, or private data the sandbox will not open,
#     turns the test into a skip instead of a failure (setup and call phases alike) ---
def _skip_if_environmental(exc: BaseException) -> None:
    from lattix.errors import MissingDependencyError

    if isinstance(exc, MissingDependencyError):
        pytest.skip(f"optional dependency missing: {exc}")
    if isinstance(exc, PermissionError) and exc.filename:
        target = Path(str(exc.filename))
        if target.is_absolute() and ROOT not in target.parents:
            pytest.skip(f"sandbox denies reading {target}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except (PermissionError, RuntimeError) as exc:
        _skip_if_environmental(exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except (PermissionError, RuntimeError) as exc:
        _skip_if_environmental(exc)
        raise


# --- hypothesis profiles (PLAN §5.3): ci = 200 examples, nightly = 5000 ---
try:
    from hypothesis import HealthCheck
    from hypothesis import settings as _hy_settings

    _hy_settings.register_profile("ci", max_examples=200, deadline=None,
                                  suppress_health_check=[HealthCheck.too_slow])
    _hy_settings.register_profile("nightly", max_examples=5000, deadline=None,
                                  suppress_health_check=[HealthCheck.too_slow])
    _hy_settings.load_profile("ci")
except ImportError:  # pragma: no cover
    pass
