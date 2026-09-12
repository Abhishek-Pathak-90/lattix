"""Floor frames of the expanded line: the position and the 3×3 orientation of every element at
its entrance, its centre and its exit, the body frame after its misalignment, and the MAD-X
survey angles.

Global frame: the reference orbit starts at the origin along +Z with X horizontal and Y up
(right-handed).  A positive horizontal bend turns the direction toward −X, so its azimuth
``theta`` decreases — MAD-X ``survey``'s frame and angles.  :func:`lattix.ir.lattice.survey` is
the ``(X, Y, Z, theta)`` view of the exit frames returned here."""
from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from lattix.ir.elements import Bend, Element, Multipole, Octupole, Patch, Quadrupole, Sextupole, Superposition
from lattix.ir.lattice import Lattice, Placed

TWO_PI = 2.0 * math.pi


# -- rotations ----------------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    """Rotation about the local x axis (a pitch): the ``x_rot`` of a Patch or of a body shift."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    """Rotation about the local y axis (a yaw): ``y_rot``."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_s(a: float) -> np.ndarray:
    """Rotation about the local s axis (a roll): ``tilt``."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def madx_angles(W: np.ndarray) -> tuple[float, float, float]:
    """MAD-X survey angles of an orientation: ``W = Theta(theta) @ Phi(phi) @ Psi(psi)`` with
    ``theta`` the azimuth about the global Y axis, ``phi`` the elevation and ``psi`` the roll.
    ``theta`` and ``psi`` come back in (−π, π]; :func:`unwrap` keeps them continuous along a line.
    At an elevation of exactly ±π/2 the azimuth and the roll are not separable and both read 0."""
    theta = math.atan2(W[0, 2], W[2, 2])
    phi = math.asin(max(-1.0, min(1.0, W[1, 2])))
    psi = math.atan2(W[1, 0], W[1, 1])
    return theta, phi, psi


def frame_from_angles(theta: float, phi: float, psi: float) -> np.ndarray:
    """The orientation with the given MAD-X survey angles (the inverse of :func:`madx_angles`)."""
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    cs, ss = math.cos(psi), math.sin(psi)
    big_theta = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    big_phi = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
    big_psi = np.array([[cs, -ss, 0.0], [ss, cs, 0.0], [0.0, 0.0, 1.0]])
    return big_theta @ big_phi @ big_psi


def unwrap(angle: float, previous: float) -> float:
    """``angle + 2πk`` closest to ``previous`` (MAD-X ``proxim``): a ring ends at −2π, not 0."""
    return angle + TWO_PI * round((previous - angle) / TWO_PI)


def quaternion(W: np.ndarray) -> tuple[float, float, float, float]:
    """``(x, y, z, w)`` of the rotation whose matrix is ``W`` — the order three.js and glTF use."""
    m = np.asarray(W, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s


# -- frames --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """A position ``V`` [m] and an orientation ``W`` in the global frame; ``W``'s columns are the
    local x, y and s axes expressed in global coordinates."""
    V: np.ndarray
    W: np.ndarray

    @classmethod
    def origin(cls) -> Frame:
        return cls(np.zeros(3), np.eye(3))

    @property
    def s_axis(self) -> np.ndarray:
        return self.W[:, 2]

    def angles(self) -> tuple[float, float, float]:
        return madx_angles(self.W)

    def quaternion(self) -> tuple[float, float, float, float]:
        return quaternion(self.W)

    def moved(self, dV_local, R_local: np.ndarray) -> Frame:
        """The frame displaced by ``dV_local`` (in this frame's axes) and rotated by ``R_local``."""
        return Frame(self.V + self.W @ np.asarray(dV_local, dtype=float), self.W @ R_local)

    def advanced(self, L: float) -> Frame:
        """The frame ``L`` further along the local s axis."""
        return Frame(self.V + self.W[:, 2] * L, self.W)

    def local(self, point) -> np.ndarray:
        """Coordinates of a global point in this frame."""
        return self.W.T @ (np.asarray(point, dtype=float) - self.V)


def site_frame(x0: float = 0.0, y0: float = 0.0, z0: float = 0.0,
               theta0: float = 0.0, phi0: float = 0.0, psi0: float = 0.0) -> Frame:
    """The start pose of MAD-X ``SURVEY``: position ``(x0, y0, z0)`` and angles ``theta0, phi0, psi0``."""
    return Frame(np.array([x0, y0, z0], dtype=float), frame_from_angles(theta0, phi0, psi0))


_PREFIX = {"entrance": "in_", "centre": "c_", "exit": "out_", "body": "body_"}


@dataclass(frozen=True)
class PlacedFrames:
    """The frames of one placed element.  ``body`` is the centre frame rolled by the element's
    own field roll (a skew quadrupole is a rolled quadrupole) and moved by ``Element.shift``;
    it is ``centre`` itself when neither applies.  The reference orbit never moves because of a
    shift.  ``theta_*`` are the azimuths kept continuous along the line."""
    index: int
    name: str
    kind: str
    s_in: float
    s_out: float
    entrance: Frame
    centre: Frame
    exit: Frame
    body: Frame
    theta_in: float
    theta_c: float
    theta_out: float
    angle: float = 0.0          # signed bend angle after reversal; 0 for anything but a bend
    tilt_ref: float = 0.0
    roll: float = 0.0           # the field roll folded into the body frame [rad]
    shifted: bool = False
    reversed: bool = False
    parent: int | None = None   # a Superposition child carries its container's index here
    child: str | None = None

    FRAMES = ("entrance", "centre", "exit", "body")

    @property
    def length(self) -> float:
        return self.s_out - self.s_in

    def frame(self, at: str = "exit") -> Frame:
        if at not in self.FRAMES:
            raise ValueError(f"at must be one of {self.FRAMES}, not {at!r}")
        return getattr(self, at)

    def angles(self, at: str = "exit") -> tuple[float, float, float]:
        """``(theta, phi, psi)`` at one of the frames, theta continuous along the line."""
        theta, phi, psi = self.frame(at).angles()
        ref = {"entrance": self.theta_in, "centre": self.theta_c, "exit": self.theta_out,
               "body": self.theta_c}[at]
        return unwrap(theta, ref), phi, psi


def _arc(L: float, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Local displacement and rotation across a horizontal bend of arc length ``L`` and angle ``a``."""
    rho = L / a
    ca, sa = math.cos(a), math.sin(a)
    dV = np.array([rho * (ca - 1.0), 0.0, rho * sa])
    R = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])      # rotation about y by −a
    return dV, R


def field_roll(e: Element) -> float:
    """The roll of an element's own field about s: a skew quadrupole is a rolled quadrupole, so
    its body frame is rolled by ``multipole.tilt[1]`` (sextupole ``[2]``, octupole ``[3]``; a thin
    multipole when all its orders share one tilt).  Bmad's ``tilt`` is exactly this, and its floor
    positions include it (measured: ``tests/oracles/test_frame_survey.py``)."""
    order = {Quadrupole: 1, Sextupole: 2, Octupole: 3}.get(type(e))
    if order is not None:
        return float(e.multipole.tilt.get(order, 0.0))
    if isinstance(e, Multipole):
        tilts = {t for n, t in e.multipole.tilt.items() if t}
        return float(tilts.pop()) if len(tilts) == 1 else 0.0
    return 0.0


def _body(centre: Frame, e: Element, apply_shift: bool) -> tuple[Frame, float, bool]:
    """The body frame: the centre rolled by the field roll, then moved by the shift (offsets in
    the centre frame, then yaw, pitch, roll — Bmad's body coordinates, measured)."""
    roll = field_roll(e)
    sh = e.shift if apply_shift and e.shift is not None and not e.shift.is_zero() else None
    if sh is None:
        return (centre.moved((0.0, 0.0, 0.0), rot_s(roll)) if roll else centre), roll, False
    return centre.moved((sh.x_offset, sh.y_offset, sh.z_offset),
                        rot_y(sh.y_rot) @ rot_x(sh.x_rot) @ rot_s(sh.tilt + roll)), roll, True


def frame_survey(placed: list[Placed], *, lat: Lattice | None = None, start: Frame | None = None,
                 apply_shift: bool = True, expand_children: bool = False) -> list[PlacedFrames]:
    """The frames of every placed element, in order.

    A bend moves the frame along its arc (``tilt_ref`` rotates the bend plane; a reversed
    placement negates the angle) and its centre is the arc midpoint; a Patch jumps by its offsets
    in the local frame and rotates by yaw, pitch, roll (``Ry(y_rot)·Rx(x_rot)·Rs(tilt)``), its
    centre being its entrance; everything else advances straight.  With ``expand_children`` the
    children of a Superposition (looked up in ``lat.elements``) follow their container, placed
    along its straight axis at ``s_in + z_offset``."""
    f = start if start is not None else Frame.origin()
    out: list[PlacedFrames] = []
    theta_prev = f.angles()[0]
    for p in placed:
        e = p.element
        L = e.length
        f_in = f
        theta_in = unwrap(f_in.angles()[0], theta_prev)
        angle = tilt = 0.0
        if isinstance(e, Bend) and e.bend.angle != 0.0:
            angle = -e.bend.angle if p.reversed else e.bend.angle
            tilt = e.bend.tilt_ref
            T = rot_s(tilt)
            dV_half, R_half = _arc(L / 2.0, angle / 2.0)
            dV, R = _arc(L, angle)
            f_c = f_in.moved(T @ dV_half, T @ R_half @ T.T)
            f = f_in.moved(T @ dV, T @ R @ T.T)
        elif isinstance(e, Patch):
            f = f_in.moved((e.x_offset, e.y_offset, e.z_offset),
                           rot_y(e.y_rot) @ rot_x(e.x_rot) @ rot_s(e.tilt))
            f_c = f_in
        else:
            f_c = f_in.advanced(L / 2.0)
            f = f_in.advanced(L)
        theta_c = unwrap(f_c.angles()[0], theta_in)
        theta_out = unwrap(f.angles()[0], theta_c)
        body, roll, shifted = _body(f_c, e, apply_shift)
        out.append(PlacedFrames(p.index, e.name, e.kind, p.s_in, p.s_out, f_in, f_c, f, body,
                                theta_in, theta_c, theta_out, angle, tilt, roll, shifted, p.reversed))
        theta_prev = theta_out
        if expand_children and isinstance(e, Superposition) and lat is not None:
            for z0, name in e.children:
                c = lat.elements.get(name)
                if c is None:
                    continue
                c_in = f_in.advanced(z0)
                c_c = c_in.advanced(c.length / 2.0)
                c_body, c_roll, c_shifted = _body(c_c, c, apply_shift)
                out.append(PlacedFrames(p.index, name, c.kind, p.s_in + z0, p.s_in + z0 + c.length,
                                        c_in, c_c, c_in.advanced(c.length), c_body,
                                        theta_in, theta_in, theta_in, 0.0, 0.0, c_roll, c_shifted, p.reversed,
                                        parent=p.index, child=name))
    return out


# -- tables --------------------------------------------------------------------------------------
def survey_table(frames: list[PlacedFrames], *, at: str = "exit") -> list[dict]:
    """Flat rows: ``i, name, kind, parent, s_in, s_out, L`` then ``X, Y, Z, theta, phi, psi`` of
    the requested frame (``entrance``, ``centre``, ``exit`` or ``body``), or of all four with the
    prefixes ``in_``, ``c_``, ``out_``, ``body_`` when ``at="all"``; then ``angle, tilt_ref``."""
    if at == "all":
        which = list(PlacedFrames.FRAMES)
    elif at in PlacedFrames.FRAMES:
        which = [at]
    else:
        raise ValueError(f"at must be 'all' or one of {PlacedFrames.FRAMES}, not {at!r}")
    rows: list[dict] = []
    for f in frames:
        row: dict = {"i": f.index, "name": f.name, "kind": f.kind,
                     "parent": "" if f.parent is None else f.parent,
                     "s_in": float(f.s_in), "s_out": float(f.s_out), "L": float(f.length)}
        for w in which:
            fr = f.frame(w)
            theta, phi, psi = f.angles(w)
            prefix = _PREFIX[w] if at == "all" else ""
            row[prefix + "X"], row[prefix + "Y"], row[prefix + "Z"] = (float(v) for v in fr.V)
            row[prefix + "theta"], row[prefix + "phi"], row[prefix + "psi"] = theta, phi, psi
        row["angle"], row["tilt_ref"] = float(f.angle), float(f.tilt_ref)
        rows.append(row)
    return rows


def _cell(v) -> str:
    if isinstance(v, bool | int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.12g}"
    return "" if v is None else str(v)


def survey_csv(rows: list[dict], *, comments: Iterable[str] = ()) -> str:
    """The rows of :func:`survey_table` as CSV: ``#`` comment lines, a header row, then one row
    per element with floats at 12 significant digits (metres and radians)."""
    buf = io.StringIO()
    for c in comments:
        buf.write(f"# {c}\n")
    if rows:
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([_cell(v) for v in r.values()])
    return buf.getvalue()
