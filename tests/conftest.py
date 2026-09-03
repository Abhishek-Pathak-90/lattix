"""Shared pytest configuration: engine/data guards come from lattix.testing."""
from __future__ import annotations

import pytest

from lattix.testing import corpus_dir, helix_root, tracewin_exe  # noqa: F401  (re-exported helpers)


@pytest.fixture(scope="session")
def data_public():
    from pathlib import Path

    return Path(__file__).parent / "data" / "public"
