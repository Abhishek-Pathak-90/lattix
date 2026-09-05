"""The battery pairs only formats that can be read back (a writer-only format has no fixed point)."""
from lattix.crossval import pairs, readable_formats
from lattix.formats.base import FORMATS


def test_writer_only_formats_are_not_battery_pairs():
    assert FORMATS["madng"].reader_attr is None
    assert "madng" not in readable_formats()
    assert all("madng" not in pair for pair in pairs())
    assert all(a != b for a, b in pairs())
    n = len(readable_formats())
    assert len(pairs()) == n * (n - 1)
    assert set(readable_formats()) == {k for k, s in FORMATS.items()
                                       if s.reader_attr and s.writer_attr and not s.options.get("bridge")}
