"""MAD-NG lattice output, delegated to xtrack's own ``mad_writer.to_madng_sequence`` (Phase 5.1).

Writer only: the IR becomes an :class:`xtrack.Line` (every ledger row of that conversion applies,
plus ``EQUIVALENT:VIA_XTRACK`` per element) and xtrack renders the Lua ``sequence``; knobs come
out as MAD-NG deferred expressions (``k1 =\\ (kq)``).  A reader needs a Lua parser or MAD-NG
itself and is deferred.
"""
from lattix.formats.madng.writer import Writer, write

__all__ = ["Writer", "write"]
