"""Engine-free tests of the floor frames: the survey wrapper against the loop it replaced, the
MAD-X angles, body shifts, Superposition children, the start pose and the survey table."""
from __future__ import annotations

import csv
import io
import math

import numpy as np
import pytest

from lattix.crossval import DECKS, PUBLIC
from lattix.formats import read
from lattix.ir import (
    Bend,
    BendP,
    BodyShiftP,
    Drift,
    FieldMap,
    Lattice,
    Line,
    LineItem,
    Patch,
    Quadrupole,
    ReferenceParticle,
    Superposition,
    frame_survey,
    site_frame,
    species,
    survey,
    survey_csv,
    survey_table,
)
from lattix.ir.frames import Frame, frame_from_angles, madx_angles, quaternion, rot_s, rot_x, rot_y, unwrap

PROTON = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)


def _legacy_survey(placed) -> np.ndarray:
    """The loop ``survey()`` was before frames.py existed, kept verbatim as the regression guard."""
    V = np.zeros(3)
    W = np.eye(3)
    out = np.empty((len(placed), 4))
    theta = 0.0
    for i, p in enumerate(placed):
        e = p.element
        L = e.length
        if isinstance(e, Bend) and e.bend.angle != 0.0:
            a = -e.bend.angle if p.reversed else e.bend.angle
            rho = L / a
            psi = e.bend.tilt_ref
            ct, st = math.cos(psi), math.sin(psi)
            T = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
            R_local = np.array([rho * (math.cos(a) - 1.0), 0.0, rho * math.sin(a)])
            ca, sa = math.cos(a), math.sin(a)
            S = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
            V = V + W @ (T @ R_local)
            W = W @ T @ S @ T.T
            if psi == 0.0:
                theta -= a
        elif isinstance(e, Patch):
            V = V + W @ np.array([e.x_offset, e.y_offset, e.z_offset])
            cy, sy = math.cos(e.y_rot), math.sin(e.y_rot)
            cx, sx = math.cos(e.x_rot), math.sin(e.x_rot)
            ct, st = math.cos(e.tilt), math.sin(e.tilt)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Rs = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
            W = W @ Ry @ Rx @ Rs
            theta += e.y_rot
        else:
            V = V + W[:, 2] * L
        out[i] = (V[0], V[1], V[2], theta)
    return out


def _line(*elements) -> Lattice:
    return Lattice.from_sequence("l", list(elements), PROTON)


def _bend(name: str, angle: float, length: float = 1.0, tilt_ref: float = 0.0) -> Bend:
    return Bend(name=name, length=length, bend=BendP(angle=angle, tilt_ref=tilt_ref))


# -- the wrapper reproduces the old numbers -----------------------------------------------------
def test_survey_matches_the_legacy_loop_on_synthetic_lines():
    lines = [
        _line(Drift(name="d", length=2.0), _bend("b", math.pi / 2), Drift(name="d2", length=1.0)),
        _line(_bend("b1", 0.3), _bend("b2", -0.7, 0.5), _bend("v", 0.2, 0.8, tilt_ref=math.pi / 2),
              Drift(name="d", length=0.4), _bend("w", -0.4, 0.6, tilt_ref=-math.pi / 2)),
        _line(Patch(name="p", x_offset=0.01, y_offset=-0.02, z_offset=0.3, y_rot=-0.1),
              Drift(name="d", length=1.0), _bend("b", 0.5), Patch(name="q", y_rot=0.2)),
    ]
    b = _bend("b", 0.4)
    ring = Lattice(reference=PROTON, elements={"d": Drift(name="d", length=1.0), "b": b},
                   lines={"cell": Line(name="cell", items=[LineItem(ref="d"), LineItem(ref="b")]),
                          "ring": Line(name="ring", items=[LineItem(ref="cell", repeat=2),
                                                           LineItem(ref="cell", reverse=True)])},
                   use="ring")
    lines.append(ring)
    for lat in lines:
        fl = lat.flatten()
        np.testing.assert_allclose(survey(fl), _legacy_survey(fl), atol=1e-12)


def test_theta_is_the_azimuth_once_the_line_is_pitched_or_rolled():
    """The old loop summed bend angles into theta, which is only the azimuth while the bend plane
    is horizontal.  After a pitched and rolled patch the positions still agree, and theta is
    the azimuth of the exit direction, not the sum."""
    lat = _line(Patch(name="p", x_rot=0.05, tilt=0.4), Drift(name="d", length=1.0), _bend("b", 0.5))
    fl = lat.flatten()
    sv, legacy = survey(fl), _legacy_survey(fl)
    np.testing.assert_allclose(sv[:, :3], legacy[:, :3], atol=1e-12)
    s_axis = frame_survey(fl)[-1].exit.s_axis
    assert sv[-1, 3] == pytest.approx(math.atan2(s_axis[0], s_axis[2]))
    assert abs(sv[-1, 3] - legacy[-1, 3]) > 1e-3


@pytest.mark.parametrize("rel,fmt,opts", DECKS, ids=[d[0] for d in DECKS])
def test_survey_matches_the_legacy_loop_on_public_decks(rel, fmt, opts):
    if fmt == "madx":
        pytest.importorskip("cpymad")
    lat, _ = read(PUBLIC / rel, fmt, **opts)
    fl = lat.flatten()
    np.testing.assert_allclose(survey(fl), _legacy_survey(fl), atol=1e-9)


# -- angles and quaternions ---------------------------------------------------------------------
def test_madx_angles_round_trip():
    rng = np.random.default_rng(7)
    for _ in range(50):
        theta, phi, psi = rng.uniform(-3.1, 3.1), rng.uniform(-1.5, 1.5), rng.uniform(-3.1, 3.1)
        W = frame_from_angles(theta, phi, psi)
        np.testing.assert_allclose(W @ W.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(madx_angles(W), (theta, phi, psi), atol=1e-12)
    # the identity is the start of every line; the pure rotations read as themselves
    assert madx_angles(np.eye(3)) == (0.0, 0.0, 0.0)
    assert madx_angles(rot_y(0.3))[0] == pytest.approx(0.3)
    assert madx_angles(rot_x(0.3))[1] == pytest.approx(-0.3)     # a positive pitch tips the beam down
    assert madx_angles(rot_s(0.3))[2] == pytest.approx(0.3)


def test_quaternion_rotates_like_the_matrix():
    rng = np.random.default_rng(11)
    for theta, phi, psi in rng.uniform(-3.0, 3.0, size=(30, 3)):
        W = frame_from_angles(theta, phi / 2, psi)
        x, y, z, w = quaternion(W)
        assert x * x + y * y + z * z + w * w == pytest.approx(1.0)
        v = rng.normal(size=3)
        q = np.array([x, y, z])
        rotated = v + 2.0 * w * np.cross(q, v) + 2.0 * np.cross(q, np.cross(q, v))
        np.testing.assert_allclose(rotated, W @ v, atol=1e-12)


def test_unwrap_keeps_the_nearest_turn():
    assert unwrap(-3.1, 3.1) == pytest.approx(-3.1 + 2 * math.pi)
    assert unwrap(0.2, -2 * math.pi) == pytest.approx(0.2 - 2 * math.pi)
    assert unwrap(1.0, 1.5) == 1.0


# -- frames of the element kinds ----------------------------------------------------------------
def test_frames_of_a_horizontal_bend():
    rho = 1.0 / (math.pi / 2)
    lat = _line(Drift(name="d", length=2.0), _bend("b", math.pi / 2), Drift(name="d2", length=1.0))
    fr = frame_survey(lat.flatten())
    d, b, d2 = fr
    np.testing.assert_allclose(b.entrance.V, [0, 0, 2.0])
    np.testing.assert_allclose(b.centre.V, [rho * (math.cos(math.pi / 4) - 1), 0, 2.0 + rho * math.sin(math.pi / 4)])
    np.testing.assert_allclose(b.exit.V, [-rho, 0, 2.0 + rho], atol=1e-12)
    np.testing.assert_allclose(b.exit.s_axis, [-1, 0, 0], atol=1e-12)
    assert (b.theta_in, b.theta_c, b.theta_out) == pytest.approx((0.0, -math.pi / 4, -math.pi / 2))
    assert b.angles("centre")[1:] == pytest.approx((0.0, 0.0))
    assert b.angle == math.pi / 2 and b.tilt_ref == 0.0 and not b.shifted and not b.reversed
    assert b.body is b.centre
    np.testing.assert_allclose(d2.exit.V, [-rho - 1.0, 0, 2.0 + rho], atol=1e-12)
    assert d.centre.V[2] == 1.0 and d.length == 2.0


def test_frames_of_a_vertical_bend():
    a, L = 0.3, 0.9
    rho = L / a
    fr = frame_survey(_line(_bend("v", a, L, tilt_ref=math.pi / 2)).flatten())[0]
    np.testing.assert_allclose(fr.exit.V, [0.0, rho * (math.cos(a) - 1.0), rho * math.sin(a)], atol=1e-12)
    theta, phi, psi = fr.angles("exit")
    assert theta == pytest.approx(0.0) and phi == pytest.approx(-a) and psi == pytest.approx(0.0, abs=1e-12)
    assert fr.angles("centre")[1] == pytest.approx(-a / 2)
    # the opposite tilt bends the other way
    up = frame_survey(_line(_bend("v", a, L, tilt_ref=-math.pi / 2)).flatten())[0]
    assert up.exit.V[1] == pytest.approx(-fr.exit.V[1]) and up.angles()[1] == pytest.approx(a)


def test_reversed_placement_negates_the_bend():
    b = _bend("b", 0.4)
    lat = Lattice(reference=PROTON, elements={"b": b},
                  lines={"l": Line(name="l", items=[LineItem(ref="b", reverse=True)])}, use="l")
    fr = frame_survey(lat.flatten())[0]
    assert fr.reversed and fr.angle == pytest.approx(-0.4) and fr.theta_out == pytest.approx(0.4)


def test_patch_frames():
    yaw = frame_survey(_line(Patch(name="p", y_rot=0.25)).flatten())[0]
    assert yaw.angles() == pytest.approx((0.25, 0.0, 0.0))
    assert yaw.centre is yaw.entrance and yaw.length == 0.0
    pitch = frame_survey(_line(Patch(name="p", x_rot=0.25)).flatten())[0]
    assert pitch.angles()[1] == pytest.approx(-0.25)
    roll = frame_survey(_line(Patch(name="p", tilt=0.25)).flatten())[0]
    assert roll.angles()[2] == pytest.approx(0.25)
    moved = frame_survey(_line(_bend("b", math.pi / 2), Patch(name="p", x_offset=0.1, z_offset=0.2)).flatten())[1]
    np.testing.assert_allclose(moved.exit.V - moved.entrance.V, moved.entrance.W @ [0.1, 0.0, 0.2])


def test_body_shift_is_about_the_centre_in_the_centre_frame():
    q = Quadrupole(name="q", length=0.4, shift=BodyShiftP(x_offset=1e-3, y_offset=-2e-3, z_offset=5e-4,
                                                          x_rot=1e-3, y_rot=2e-3, tilt=3e-3))
    fr = frame_survey(_line(_bend("b", math.pi / 2), q).flatten())[1]
    assert fr.shifted
    np.testing.assert_allclose(fr.body.V, fr.centre.V + fr.centre.W @ [1e-3, -2e-3, 5e-4])
    np.testing.assert_allclose(fr.body.W, fr.centre.W @ rot_y(2e-3) @ rot_x(1e-3) @ rot_s(3e-3))
    # the reference orbit is untouched, and the shift can be switched off
    plain = frame_survey(_line(_bend("b", math.pi / 2), q).flatten(), apply_shift=False)[1]
    assert not plain.shifted and plain.body is plain.centre
    np.testing.assert_allclose(plain.exit.V, fr.exit.V)
    # a zero shift is not a shift
    assert not frame_survey(_line(Quadrupole(name="q", length=0.4, shift=BodyShiftP())).flatten())[0].shifted


def test_superposition_children_follow_the_container():
    lat = _line(Drift(name="d", length=0.5), Superposition(name="cl", length=1.0, children=[(0.2, "m1"), (0.5, "m2")]))
    lat.add_element(FieldMap(name="m1", length=0.3))
    lat.add_element(FieldMap(name="m2", length=0.4, shift=BodyShiftP(y_offset=1e-3)))
    plain = frame_survey(lat.flatten())
    assert [f.name for f in plain] == ["d", "cl"]
    fr = frame_survey(lat.flatten(), lat=lat, expand_children=True)
    assert [f.name for f in fr] == ["d", "cl", "m1", "m2"]
    m1, m2 = fr[2], fr[3]
    assert m1.parent == 1 and m1.child == "m1" and m1.index == 1 and m1.kind == "FieldMap"
    assert (m1.s_in, m1.s_out) == pytest.approx((0.7, 1.0))
    np.testing.assert_allclose(m1.entrance.V, [0, 0, 0.7])
    np.testing.assert_allclose(m1.centre.V, [0, 0, 0.85])
    assert m2.shifted and m2.body.V[1] == pytest.approx(1e-3)
    # a child the lattice does not define is skipped rather than fatal
    lat.elements.pop("m2")
    assert [f.name for f in frame_survey(lat.flatten(), lat=lat, expand_children=True)] == ["d", "cl", "m1"]


def test_a_ring_closes_and_theta_counts_the_turn():
    cell = [Drift(name="d", length=1.0), _bend("b", math.pi / 2)]
    lat = _line(*(e.model_copy(update={"name": f"{e.name}{i}"}) for i in range(4) for e in cell))
    fr = frame_survey(lat.flatten())
    np.testing.assert_allclose(fr[-1].exit.V, [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(fr[-1].exit.W, np.eye(3), atol=1e-12)
    assert fr[-1].theta_out == pytest.approx(-2 * math.pi)
    assert fr[-1].angles()[0] == pytest.approx(-2 * math.pi)


def test_start_pose_is_applied():
    start = site_frame(1.0, 2.0, 3.0, theta0=0.5, phi0=0.1, psi0=0.2)
    lat = _line(Drift(name="d", length=2.0), _bend("b", 0.3))
    fr = frame_survey(lat.flatten(), start=start)
    np.testing.assert_allclose(fr[0].entrance.V, [1.0, 2.0, 3.0])
    assert fr[0].angles("entrance") == pytest.approx((0.5, 0.1, 0.2))
    np.testing.assert_allclose(fr[0].exit.V, start.V + 2.0 * start.W[:, 2])
    # the same line from the origin, transported by the start pose, lands on the same points
    plain = frame_survey(lat.flatten())
    for a, b in zip(plain, fr, strict=True):
        np.testing.assert_allclose(start.V + start.W @ a.exit.V, b.exit.V, atol=1e-12)
        np.testing.assert_allclose(start.W @ a.exit.W, b.exit.W, atol=1e-12)


def test_frame_helpers():
    f = Frame.origin().moved([1.0, 0.0, 0.0], rot_y(0.5))
    np.testing.assert_allclose(f.local([1.0, 0.0, 0.0]), [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(f.advanced(2.0).V, [1.0 + 2 * math.sin(0.5), 0.0, 2 * math.cos(0.5)])
    with pytest.raises(ValueError):
        frame_survey(_line(Drift(name="d", length=1.0)).flatten())[0].frame("middle")


# -- tables --------------------------------------------------------------------------------------
def test_survey_table_and_csv():
    lat = _line(Drift(name="d", length=2.0), _bend("b", math.pi / 2),
                Quadrupole(name="q", length=0.4, shift=BodyShiftP(x_offset=1e-3)))
    fr = frame_survey(lat.flatten())
    rows = survey_table(fr)
    assert list(rows[0]) == ["i", "name", "kind", "parent", "s_in", "s_out", "L", "X", "Y", "Z", "theta", "phi", "psi",
                             "angle", "tilt_ref"]
    assert rows[1]["theta"] == pytest.approx(-math.pi / 2) and rows[1]["angle"] == pytest.approx(math.pi / 2)
    assert rows[0]["parent"] == "" and rows[2]["kind"] == "Quadrupole"
    every = survey_table(fr, at="all")
    assert "in_X" in every[0] and "c_theta" in every[0] and "out_psi" in every[0] and "body_X" in every[2]
    # after the 90° bend the local x axis lies along global Z: the offset shows up there
    assert every[2]["body_Z"] != every[2]["c_Z"] and every[1]["body_X"] == every[1]["c_X"]
    assert every[1]["c_theta"] == pytest.approx(-math.pi / 4)
    with pytest.raises(ValueError):
        survey_table(fr, at="middle")
    text = survey_csv(rows, comments=["lattix test", "deck l"])
    lines = text.splitlines()
    assert lines[0] == "# lattix test" and lines[1] == "# deck l"
    parsed = list(csv.DictReader(io.StringIO("\n".join(lines[2:]))))
    assert [r["name"] for r in parsed] == ["d", "b", "q"]
    assert float(parsed[1]["X"]) == pytest.approx(-1.0 / (math.pi / 2), rel=1e-11)
    assert parsed[1]["theta"] == f"{-math.pi / 2:.12g}" and parsed[0]["i"] == "0"
    assert survey_csv([]) == ""
