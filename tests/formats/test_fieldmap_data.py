"""Field-map file readers: layouts, loop orders, SI units and the on-axis profiles.

Every layout is exercised against a file written by the test itself, so the numbers are
hand-checkable; the real (undistributed) PIP-II maps are read in
``test_fieldmap_integration.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix.formats.tracewin.fieldmap_files import (
    Channel,
    checksums,
    component_files,
    decode_geom,
    expected_files,
    sha256,
)
from lattix.ir.fieldmap import (
    RF_E,
    STAT_B,
    FieldMapData,
    clear_cache,
    read_component,
)

FIELDS = Path("/Users/abhishekpathak/Desktop/Projects/HELIX_unzipped/HELIX_v3/Fields")


# ---------------------------------------------------------------- fixtures
def write_1d(path: Path, values, zmax_m: float, norm: float = 1.0) -> Path:
    """Canonical TraceWin 1-D: ``Nz Zmax`` / ``Norm`` / ``Nz+1`` values."""
    lines = [f"{len(values) - 1} {zmax_m:.6f}", f"{norm:g}"] + [f"{v:.9g}" for v in values]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_2d_cyl(path: Path, grid, zmax_m: float, rmax_m: float, norm: float = 1.0) -> Path:
    """``Nz Zmax`` / ``Nr Rmax`` / ``Norm`` / for k in z: for i in r (r fastest)."""
    grid = np.asarray(grid, dtype=float)          # (Nz+1, Nr+1)
    nz, nr = grid.shape[0] - 1, grid.shape[1] - 1
    body = " ".join(f"{v:.9g}" for v in grid.reshape(-1))
    path.write_text(f"{nz} {zmax_m:.6f}\n{nr} {rmax_m:.6f}\n{norm:g}\n{body}\n")
    return path


def write_3d_cart(path: Path, grid, zmax_m: float, xlim, ylim, norm: float = 1.0) -> Path:
    """``Nz Zmax`` / ``Nx Xmin Xmax`` / ``Ny Ymin Ymax`` / ``Norm`` / k(z), j(y), i(x fastest)."""
    grid = np.asarray(grid, dtype=float)          # (Nx+1, Ny+1, Nz+1)
    nx, ny, nz = (n - 1 for n in grid.shape)
    flat = grid.transpose(2, 1, 0).reshape(-1)    # back to k, j, i order
    head = (f"{nz} {zmax_m:.6f}\n{nx} {xlim[0]:.6f} {xlim[1]:.6f}\n"
            f"{ny} {ylim[0]:.6f} {ylim[1]:.6f}\n{norm:g}\n")
    path.write_text(head + " ".join(f"{v:.9g}" for v in flat) + "\n")
    return path


def write_2d_cart(path: Path, grid, xlim, ylim, norm: float = 1.0) -> Path:
    """``Nx Xmin Xmax`` / ``Ny Ymin Ymax`` / ``Norm`` / for j in y: for i in x (x fastest)."""
    grid = np.asarray(grid, dtype=float)          # (Nx+1, Ny+1)
    nx, ny = grid.shape[0] - 1, grid.shape[1] - 1
    flat = grid.T.reshape(-1)
    head = f"{nx} {xlim[0]:.6f} {xlim[1]:.6f}\n{ny} {ylim[0]:.6f} {ylim[1]:.6f}\n{norm:g}\n"
    path.write_text(head + " ".join(f"{v:.9g}" for v in flat) + "\n")
    return path


# ---------------------------------------------------------------- geom decoding
def test_geom_decoding_and_expected_files():
    code = decode_geom(7700)
    assert (code.stat_E, code.stat_B, code.rf_E, code.rf_B, code.aper) == (0, 0, 7, 7, 0)
    assert not code.second_order
    assert decode_geom(-100).second_order
    assert expected_files(7700, "/m/QWR") == [
        "/m/QWR.edx", "/m/QWR.edy", "/m/QWR.edz", "/m/QWR.bdx", "/m/QWR.bdy", "/m/QWR.bdz"]
    assert expected_files(10, "/m/S") == ["/m/S.bsz"]
    assert component_files(Channel.STAT_B, 9) == [".bsz"]      # 1-D quad gradient G(z)
    with pytest.raises(ValueError):
        component_files(Channel.RF_E, 9)
    with pytest.raises(NotImplementedError):
        component_files(Channel.RF_E, 8)


# ---------------------------------------------------------------- 1-D
def test_1d_canonical_layout_is_read_in_si(tmp_path):
    # TraceWin writes metres and MV/m; lattix stores m and V/m
    write_1d(tmp_path / "c.edz", [0.0, 1.0, 2.0, 1.0, 0.0], zmax_m=0.4, norm=2.0)
    raw = read_component(tmp_path / "c.edz", 1)
    assert raw.norm == 2.0
    assert raw.z == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert raw.values == pytest.approx([0.0, 1e6, 2e6, 1e6, 0.0])


def test_1d_magnetic_file_keeps_tesla(tmp_path):
    write_1d(tmp_path / "s.bsz", [0.0, 0.5, 1.0, 0.5, 0.0], zmax_m=0.2)
    raw = read_component(tmp_path / "s.bsz", 1)
    assert raw.values == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])   # T, not scaled by 1e6


def test_1d_legacy_linac_gen_layout_still_reads(tmp_path):
    """HELIX's fixtures use ``N_pts`` / ``zmin zmax`` in cm and raw values."""
    (tmp_path / "l.edz").write_text("3\n0 20\n1\n2\n3\n")
    raw = read_component(tmp_path / "l.edz", 1)
    assert raw.norm == 1.0
    assert raw.z == pytest.approx([0.0, 0.1, 0.2])            # 20 cm → 0.2 m
    assert raw.values == pytest.approx([1e6, 2e6, 3e6])


# ---------------------------------------------------------------- 2-D / 3-D
def test_2d_cylindrical_grid_and_on_axis_slice(tmp_path):
    grid = np.array([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]])     # (Nz+1, Nr+1), r fastest
    write_2d_cyl(tmp_path / "m.edz", grid, zmax_m=0.2, rmax_m=0.01)
    write_1d(tmp_path / "m.edr", [0.0, 0.0, 0.0], zmax_m=0.2)  # unused by the axis profile
    raw = read_component(tmp_path / "m.edz", 4)
    assert raw.values.shape == (3, 2) and raw.r == pytest.approx([0.0, 0.01])
    assert raw.values[:, 0] == pytest.approx([1e6, 2e6, 3e6])


def test_3d_cartesian_loop_order_and_axis_extraction(tmp_path):
    nx = ny = 2                                   # 3 samples each: -d, 0, +d
    nz = 3
    prof = np.array([0.0, 1.0, 4.0, 2.0])
    grid = np.zeros((nx + 1, ny + 1, nz + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            grid[i, j, :] = prof * (1.0 + 0.5 * i + 0.25 * j)
    write_3d_cart(tmp_path / "c.edz", grid, 0.3, (-0.02, 0.02), (-0.02, 0.02))
    raw = read_component(tmp_path / "c.edz", 7)
    assert raw.values.shape == (3, 3, 4)
    assert raw.x == pytest.approx([-0.02, 0.0, 0.02]) and raw.z == pytest.approx([0, 0.1, 0.2, 0.3])
    # on-axis (x = y = 0) is the (1, 1) column: prof * (1 + 0.5 + 0.25)
    assert raw.values[1, 1, :] == pytest.approx(prof * 1.75 * 1e6)


def test_2d_cartesian_layout(tmp_path):
    grid = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])      # (Nx+1, Ny+1)
    write_2d_cart(tmp_path / "q.bsx", grid, (-0.01, 0.01), (-0.02, 0.02))
    raw = read_component(tmp_path / "q.bsx", 6)
    assert raw.values.shape == (3, 2)
    assert raw.values == pytest.approx(grid)
    assert raw.z is None                                        # invariant along z


# ---------------------------------------------------------------- assembly
def test_load_assembles_channels_and_on_axis_profiles(tmp_path):
    write_1d(tmp_path / "b.bsz", [0.0, 1.0, 0.0], zmax_m=0.2)
    data = FieldMapData.load(10, tmp_path, "b")
    assert list(data.channels) == [STAT_B]
    ch = data.channel(STAT_B)
    assert ch.digit == 1 and ch.is_static and not ch.is_electric
    z, bz = data.bz_profile(kb=3.0)
    assert z == pytest.approx([0.0, 0.1, 0.2]) and bz == pytest.approx([0.0, 3.0, 0.0])
    assert data.ez_profile() is None
    assert not data.has_electric


def test_load_reports_every_missing_component(tmp_path):
    write_1d(tmp_path / "c.edz", [0.0, 1.0], zmax_m=0.1)
    with pytest.raises(FileNotFoundError) as exc:
        FieldMapData.load(700, tmp_path, "c")               # 3-D E needs .edx/.edy/.edz
    assert "c.edx" in str(exc.value) and "c.edy" in str(exc.value)


def test_electric_profile_sums_channels_with_k_over_norm(tmp_path):
    write_1d(tmp_path / "e.esz", [0.0, 2.0, 0.0], zmax_m=0.2, norm=4.0)
    data = FieldMapData.load(1, tmp_path, "e")               # stat_E digit 1
    z, ez = data.ez_profile(ke=6.0)
    assert ez == pytest.approx([0.0, 2e6 * 6.0 / 4.0, 0.0])
    assert data.has_electric and data.channel("STAT_E").norm == 4.0


def test_component_cache_is_keyed_on_mtime(tmp_path):
    p = write_1d(tmp_path / "c.edz", [0.0, 1.0, 0.0], zmax_m=0.2)
    first = read_component(p, 1)
    assert read_component(p, 1) is first                      # same (path, mtime, size)
    import os
    import time

    time.sleep(0.01)
    write_1d(p, [0.0, 3.0, 0.0], zmax_m=0.2)
    os.utime(p, (time.time() + 1, time.time() + 1))
    again = read_component(p, 1)
    assert again is not first and again.values[1] == pytest.approx(3e6)
    clear_cache()


# ---------------------------------------------------------------- checksums (I-12)
def test_checksums_are_sha256_per_component_file(tmp_path):
    import hashlib

    a = write_1d(tmp_path / "a.edz", [0.0, 1.0], zmax_m=0.1)
    b = write_1d(tmp_path / "b.bsz", [0.0, 2.0], zmax_m=0.1)
    got = checksums([a, b, tmp_path / "absent.edz"])
    assert set(got) == {str(a), str(b)}
    assert got[str(a)] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert sha256(a) == got[str(a)] and sha256(a) != sha256(b)


@pytest.mark.skipif(not (FIELDS / "HWR-SOL-ANLMAP.bsz").is_file(), reason="ANL/CEA maps absent")
def test_real_1d_solenoid_map_reads_in_si():
    data = FieldMapData.load(10, FIELDS, "HWR-SOL-ANLMAP")
    z, bz = data.bz_profile(kb=1.0)
    assert len(z) == 61 and z[-1] == pytest.approx(0.3)
    assert bz.max() == pytest.approx(1.0) and math.isclose(bz[0], 0.001347006, rel_tol=1e-9)


@pytest.mark.skipif(not (FIELDS / "QWR-2012-02.edz").is_file(), reason="ANL/CEA maps absent")
def test_real_3d_cavity_map_has_both_rf_channels():
    data = FieldMapData.load(7700, FIELDS, "QWR-2012-02")
    assert sorted(data.channels) == ["RF_B", RF_E] and [c.channel for c in data.ordered] == [RF_E, "RF_B"]
    ch = data.channel(RF_E)
    assert ch.digit == 7 and ch.Fx is not None and ch.Fy is not None and ch.Fz is not None
    z, ez = data.ez_profile(ke=1.0)
    assert len(z) == 201 and z[-1] == pytest.approx(0.24)
    assert abs(ez).max() > 1e6                                   # V/m, not MV/m
    assert len(data.files) == 6
