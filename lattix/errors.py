"""Exceptions shared by the readers, writers and oracle adapters."""
from __future__ import annotations


class MissingDependencyError(RuntimeError):
    """An optional dependency a reader, writer or adapter needs is not importable.

    Raised instead of a bare ``RuntimeError`` so that callers (and the test suite, which
    turns it into a skip) can tell "install cpymad" apart from a real failure.
    """
