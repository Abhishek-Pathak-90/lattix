"""lattix — translate accelerator lattice files between codes.

Phase 0 exposes only the oracle harness (`lattix.oracles`) and the test
guards (`lattix.testing`).  The IR, readers and writers land in Phase 1+
(see PLAN.md).
"""
from lattix._version import __version__

__all__ = ["__version__"]
