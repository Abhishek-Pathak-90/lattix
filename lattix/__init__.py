"""lattix — translate accelerator lattice files between codes.

Top-level API (imported lazily so that ``import lattix`` stays cheap)::

    import lattix
    lat, report_in = lattix.read("linac.dat")             # format from the suffix
    report = lattix.write(lat, "linac.madx", strict=False) # FidelityReport
    lattix.translate("linac.dat", "linac.lte", strict=True)

``lattix.FORMATS`` is the registry (key → FormatSpec); ``lattix.guess_format``
resolves a path to a key.  The oracle harness lives in ``lattix.oracles``.
"""
from lattix._version import __version__

__all__ = ["__version__", "read", "write", "translate", "guess_format", "FORMATS"]

_LAZY = {"read", "write", "translate", "guess_format", "FORMATS"}


def __getattr__(name: str):
    if name in _LAZY:
        from lattix.formats import base

        return getattr(base, name)
    raise AttributeError(f"module 'lattix' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY)
