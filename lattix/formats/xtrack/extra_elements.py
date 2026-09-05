"""The rest of xtrack's element zoo (Phase 5.1): classes the core converter in
:mod:`lattix.formats.xtrack.convert` does not map itself.

Three groups, every one with its own ledger code (PLAN §4.4: nothing unmapped is silent):

* **mapped** — ``Magnet`` (a Bend when it curves, else the thick multipole magnet it is),
  ``SimpleThinBend``/``SimpleThinQuadrupole`` (thin multipoles), ``Wedge`` (a thin dipole kick),
  the ``Limit*`` apertures (a rectangular bounding box where the shape has no IR form),
  ``RFMultipole`` (its cavity part), ``SecondOrderTaylorMap`` (its linear part),
  ``VariableSolenoid`` (a hard-edge solenoid with the same ∫ks), ``ReferenceEnergyChange``,
  ``Rotation``/``Translation`` (patches), ``TimeDelay`` (a reference-time change) and the monitor
  family (``Instrument``);
* **folded** — ``Misalignment`` pairs become the enclosed element's ``BodyShiftP``, standalone
  ``DipoleEdge``/``MagnetEdge``/``DipoleFringe`` become the adjacent bend's pole face (the MAD-X
  ``dipedge`` rule) or, with no bend beside them, a thin lens (:func:`fold_edge`);
* **dropped with a name** — crab cavities, AC dipoles, electron lenses, wires, exciters,
  nonlinear lenses, electron coolers, ``LineSegmentMap``, ``SplineBoris``, ``TempRF``,
  ``MultipoleEdge``, ``LongitudinalLimitRect`` and the ``Random*`` generators (not lattice
  elements): a marker with the element's ``to_dict`` as a native passthrough.

Measured conventions reused from the core: ``knl[0] = −hkick``, ``ksl[0] = +vkick``, thin
``Multipole`` strengths integrate ``Bn<n>L``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Element,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    Sextupole,
    Solenoid,
    SolenoidP,
    Taylor,
)


def _dropped_sites(rep: FidelityReport) -> None:
    """One literal ledger call per xtrack class the IR cannot hold at all, so that
    ``lattix.fidelity_catalog.scan`` catalogues the codes; :data:`DROPPED_CLASSES` is built from it
    (the element name of each row is the xtrack class)."""
    rep.dropped("CRAB_CAVITY_DROPPED", "a crab cavity (transverse RF deflection) has no IR kind",
                element="CrabCavity", kind="Marker")
    rep.dropped("AC_DIPOLE_DROPPED", "an AC dipole (turn-by-turn excitation) has no IR kind",
                element="ACDipole", kind="Marker")
    rep.dropped("ELENS_DROPPED", "an electron lens has no IR kind", element="Elens", kind="Marker")
    rep.dropped("WIRE_DROPPED", "a wire compensator has no IR kind", element="Wire", kind="Marker")
    rep.dropped("EXCITER_DROPPED", "a turn-by-turn exciter has no IR kind", element="Exciter", kind="Marker")
    rep.dropped("NONLINEAR_LENS_DROPPED", "an IOTA-type nonlinear lens has no IR kind",
                element="NonLinearLens", kind="Marker")
    rep.dropped("ELECTRON_COOLER_DROPPED", "an electron cooler has no IR kind",
                element="ElectronCooler", kind="Marker")
    rep.dropped("LINE_SEGMENT_MAP_DROPPED",
                "a Twiss-parametrised segment map (tunes, chromaticities, detuning) has no IR kind",
                element="LineSegmentMap", kind="Marker")
    rep.dropped("SPLINE_BORIS_DROPPED", "a spline-field Boris integrator element has no IR kind",
                element="SplineBoris", kind="Marker")
    rep.dropped("TEMP_RF_DROPPED", "xtrack's experimental thick RF element has no IR kind",
                element="TempRF", kind="Marker")
    rep.dropped("MULTIPOLE_EDGE_DROPPED", "a nonlinear multipole fringe has no IR kind",
                element="MultipoleEdge", kind="Marker")
    rep.dropped("LONGITUDINAL_APERTURE_DROPPED", "a longitudinal aperture has no IR kind",
                element="LongitudinalLimitRect", kind="Marker")
    for name in ("RandomUniform", "RandomUniformAccurate", "RandomExponential", "RandomNormal",
                 "RandomRutherford"):
        rep.dropped("NON_LATTICE_ELEMENT", "a random-number generator is not a lattice element",
                    element=name, kind="Marker")


def _dropped_table() -> dict[str, tuple[str, str]]:
    rep = FidelityReport()
    _dropped_sites(rep)
    return {e.element: (e.code, e.message) for e in rep.entries}


#: xtrack class -> (ledger code, message) for elements the IR cannot hold at all
DROPPED_CLASSES: dict[str, tuple[str, str]] = _dropped_table()

#: monitor classes -> IR Instrument family
MONITOR_CLASSES: dict[str, str] = {
    "BeamPositionMonitor": "BPM", "BeamProfileMonitor": "PROFILE", "BeamSizeMonitor": "SIZE",
    "BeamStatsMonitor": "MONITOR", "ParticlesMonitor": "MONITOR", "LastTurnsMonitor": "MONITOR",
    "MultiElementMonitor": "MONITOR",
}

#: every xtrack ``BeamElement`` class the reader knows, by how it is treated (tested against
#: ``xt.__dict__`` so a new xtrack release shows up as a failing test, not as a silent marker)
HANDLED_HERE = frozenset(DROPPED_CLASSES) | frozenset(MONITOR_CLASSES) | frozenset({
    "Magnet", "SimpleThinBend", "SimpleThinQuadrupole", "Wedge", "LimitPolygon", "LimitRectEllipse",
    "LimitRacetrack", "RFMultipole", "SecondOrderTaylorMap", "VariableSolenoid", "ReferenceEnergyChange",
    "Rotation", "Translation", "TimeDelay", "Misalignment", "DipoleEdge", "MagnetEdge", "DipoleFringe",
})


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _arr(x) -> list[float]:
    return [float(v) for v in np.atleast_1d(x)] if x is not None else []


# ---------------------------------------------------------------------------
# element conversions

def convert_extra(cname: str, raw: Any, ir_name: str, rep: FidelityReport,
                  norm: dict, warn: list[str]) -> Element | None:
    """The IR element for one of the classes above, or ``None`` when *cname* is not ours.
    ``norm`` receives the normalized strengths to be scaled by Bρ later (as in the core)."""
    if cname in DROPPED_CLASSES:
        code, msg = DROPPED_CLASSES[cname]
        el = Marker(name=ir_name)
        el.native["xtrack"] = {"class": cname, "element": _payload(raw)}
        rep.dropped(code, f"{msg}; written as a marker with a verbatim native passthrough",
                    element=ir_name, kind="Marker", xtrack_class=cname)
        return el

    if cname in MONITOR_CLASSES:
        el = Instrument(name=ir_name, family=MONITOR_CLASSES[cname], params={"xtrack_class": cname})
        rep.equivalent("MONITOR_AS_INSTRUMENT",
                       f"xt.{cname} is read as an Instrument (its acquisition settings are not physics)",
                       element=ir_name, kind="Instrument", xtrack_class=cname)
        return el

    if cname == "Magnet":
        return _magnet(raw, ir_name, rep, norm)

    if cname == "SimpleThinBend":
        knl = _arr(raw.knl)
        el = Multipole(name=ir_name)
        norm["knl"] = {i: v for i, v in enumerate(knl) if v}
        hxl = _f(getattr(raw, "hxl", 0.0))
        if hxl:
            el.native.setdefault("xtrack", {})["hxl"] = hxl
            rep.lossy("MULTIPOLE_HXL_KEPT_NATIVE", "xt.SimpleThinBend.hxl is a geometric bend the IR Multipole "
                      "cannot hold; kept as a native passthrough", element=ir_name, kind="Multipole", hxl=hxl)
        if _f(getattr(raw, "length", 0.0)):
            el.native.setdefault("xtrack", {})["lrad"] = _f(raw.length)
        return el

    if cname == "SimpleThinQuadrupole":
        el = Multipole(name=ir_name)
        norm["knl"] = {i: v for i, v in enumerate(_arr(raw.knl)) if v}
        return el

    if cname == "Wedge":
        el = Multipole(name=ir_name)
        norm["knl"] = {0: _f(raw.k)} if _f(raw.k) else {}
        el.native["xtrack"] = {"class": "Wedge", "angle": _f(raw.angle), "k": _f(raw.k), "k1": _f(raw.k1),
                               "quad_wedge_then_dip_wedge": bool(getattr(raw, "quad_wedge_then_dip_wedge", 0))}
        rep.lossy("WEDGE_AS_THIN_DIPOLE",
                  "xt.Wedge (a dipole-field wedge) is read as a thin dipole kick of its integrated strength; "
                  "the wedge angle and its quadrupole term are kept as a native passthrough",
                  element=ir_name, kind="Multipole", angle=_f(raw.angle), k1=_f(raw.k1))
        return el

    if cname == "LimitPolygon":
        xs, ys = _arr(raw.x_vertices), _arr(raw.y_vertices)
        el = Collimator(name=ir_name)
        if xs and ys:
            el.aperture = ApertureP(shape="RECTANGULAR", x_limits=(min(xs), max(xs)), y_limits=(min(ys), max(ys)))
        el.native["xtrack"] = {"class": "LimitPolygon", "x_vertices": xs, "y_vertices": ys}
        rep.lossy("APERTURE_SHAPE", "a polygon aperture is read as its rectangular bounding box (the "
                  "vertices are kept as a native passthrough)", element=ir_name, kind="Collimator",
                  vertices=len(xs))
        return el

    if cname == "LimitRectEllipse":
        el = Collimator(name=ir_name)
        mx, my = _f(raw.max_x), _f(raw.max_y)
        a, b = math.sqrt(max(_f(raw.a_squ), 0.0)), math.sqrt(max(_f(raw.b_squ), 0.0))
        el.aperture = ApertureP(shape="RECTANGULAR", x_limits=(-mx, mx), y_limits=(-my, my))
        el.native["xtrack"] = {"class": "LimitRectEllipse", "max_x": mx, "max_y": my, "a": a, "b": b}
        rep.lossy("APERTURE_SHAPE", "a rectangle-ellipse intersection is read as the rectangle (the ellipse "
                  "half-axes are kept as a native passthrough)", element=ir_name, kind="Collimator", a=a, b=b)
        return el

    if cname == "LimitRacetrack":
        el = Collimator(name=ir_name)
        el.aperture = ApertureP(shape="RECTANGULAR", x_limits=(_f(raw.min_x), _f(raw.max_x)),
                                y_limits=(_f(raw.min_y), _f(raw.max_y)))
        el.native["xtrack"] = {"class": "LimitRacetrack", "a": _f(raw.a), "b": _f(raw.b)}
        rep.lossy("APERTURE_SHAPE", "a racetrack aperture is read as its rectangle (the corner radii are "
                  "kept as a native passthrough)", element=ir_name, kind="Collimator", a=_f(raw.a), b=_f(raw.b))
        return el

    if cname == "RFMultipole":
        from lattix.formats.xtrack.convert import phase_from_xtrack

        rf = RFP(frequency_Hz=_f(raw.frequency) or None, voltage_V=_f(raw.voltage),
                 phase_rad=phase_from_xtrack(_f(raw.lag), _f(getattr(raw, "phase", 0.0))))
        el = RFCavity(name=ir_name, length=0.0, rf=rf)
        knl, ksl = _arr(raw.knl), _arr(raw.ksl)
        if any(knl) or any(ksl):
            el.native["xtrack"] = {"class": "RFMultipole", "knl": knl, "ksl": ksl, "pn": _arr(raw.pn),
                                   "ps": _arr(raw.ps)}
            rep.lossy("RF_MULTIPOLE_TERMS_DROPPED",
                      "xt.RFMultipole is read as its cavity; the time-dependent multipole terms are kept as "
                      "a native passthrough only", element=ir_name, kind="RFCavity")
        else:
            rep.equivalent("RF_MULTIPOLE_AS_CAVITY", "an xt.RFMultipole without multipole terms is a cavity",
                           element=ir_name, kind="RFCavity")
        return el

    if cname == "SecondOrderTaylorMap":
        R = np.asarray(raw.R, dtype=float).reshape(6, 6)
        k = np.asarray(getattr(raw, "k", np.zeros(6)), dtype=float).reshape(6)
        T = np.asarray(getattr(raw, "T", np.zeros((6, 6, 6))), dtype=float)
        el = Taylor(name=ir_name, length=_f(getattr(raw, "length", 0.0)), matrix=R.tolist(), offset=k.tolist(),
                    basis="xtrack")
        if np.any(T):
            el.native["xtrack"] = {"class": "SecondOrderTaylorMap", "element": _payload(raw)}
            rep.lossy("TAYLOR_ORDER_TRUNCATED", "only the linear part of the second-order map was read; T is "
                      "kept as a native passthrough", element=ir_name, kind="Taylor")
        else:
            rep.equivalent("TAYLOR_BASIS_XTRACK", "the map is stored in xtrack's (x, px, y, py, zeta, delta) basis",
                           element=ir_name, kind="Taylor")
        return el

    if cname == "VariableSolenoid":
        prof = _arr(raw.ks_profile)
        length = _f(raw.length)
        ks0, ks1 = (prof + [0.0, 0.0])[:2]
        mean = 0.5 * (ks0 + ks1)
        el = Solenoid(name=ir_name, length=length, solenoid=SolenoidP())
        norm["ksol"] = mean
        el.native["xtrack"] = {"class": "VariableSolenoid", "ks_profile": prof}
        rep.equivalent("VARIABLE_SOLENOID_MEAN",
                       "xt.VariableSolenoid (linear ks profile) is read as a hard-edge solenoid with the same "
                       "integrated ks; the profile is kept as a native passthrough",
                       element=ir_name, kind="Solenoid", ks_entry=ks0, ks_exit=ks1)
        return el

    if cname == "ReferenceEnergyChange":
        el = ReferenceChange(name=ir_name)
        el.native.setdefault("xtrack", {})["p0c"] = _f(raw.p0c)
        rep.equivalent("REFCHANGE_AS_P0C", "xt.ReferenceEnergyChange sets the reference momentum; its kinetic "
                       "energy is resolved with the species", element=ir_name, kind="ReferenceChange",
                       p0c=_f(raw.p0c))
        return el

    if cname == "Rotation":
        el = Patch(name=ir_name, tilt=_f(raw.rot_s_rad), x_rot=_f(raw.rot_x_rad), y_rot=_f(raw.rot_y_rad))
        rep.equivalent("ROTATION_AS_PATCH", "xt.Rotation (a frame rotation) is read as a Patch",
                       element=ir_name, kind="Patch")
        return el

    if cname == "Translation":
        el = Patch(name=ir_name, x_offset=_f(raw.shift_x), y_offset=_f(raw.shift_y), z_offset=_f(raw.shift_s))
        rep.equivalent("TRANSLATION_AS_PATCH", "xt.Translation (a frame shift) is read as a Patch",
                       element=ir_name, kind="Patch")
        return el

    if cname == "TimeDelay":
        el = ReferenceChange(name=ir_name)
        el.native.setdefault("xtrack", {})["dzeta"] = _f(raw.shift_zeta)
        rep.equivalent("ZETASHIFT_AS_REFCHANGE", "xt.TimeDelay is read as a reference-time change; the time "
                       "offset needs the local beta, kept as a native passthrough",
                       element=ir_name, kind="ReferenceChange", dzeta=_f(raw.shift_zeta))
        return el

    return None


def _magnet(raw: Any, ir_name: str, rep: FidelityReport, norm: dict) -> Element:
    """``xt.Magnet``: the unified thick magnet.  Curvature makes it a Bend; otherwise it is the
    thick multipole magnet of its lowest non-zero order (higher orders ride along on the IR
    element's multipole group, which every writer either keeps or notes)."""
    from lattix.formats.xtrack.convert import _bend_k0

    length = _f(raw.length)
    h = _f(raw.h)
    angle = _f(getattr(raw, "angle", 0.0)) or h * length
    k = {i: _f(getattr(raw, f"k{i}", 0.0)) for i in range(4)}
    ks = {i: _f(getattr(raw, f"k{i}s", 0.0)) for i in range(4)}
    knl, ksl = _arr(raw.knl), _arr(raw.ksl)
    if h or angle:
        el = Bend(name=ir_name, length=length,
                  bend=BendP(angle=angle, e1=_f(raw.edge_entry_angle), e2=_f(raw.edge_exit_angle),
                             edge_int1=_f(raw.edge_entry_fint), edge_int2=_f(raw.edge_exit_fint),
                             hgap=_f(raw.edge_entry_hgap), tilt_ref=_f(getattr(raw, "rot_s_rad", 0.0))))
        k0 = _bend_k0(raw)
        hh = angle / length if length else h
        if not bool(getattr(raw, "k0_from_h", False)) and abs(k0 - hh) > 1e-15 * max(1.0, abs(hh)):
            el.native.setdefault("xtrack", {})["k0"] = k0
            rep.lossy("BEND_K0_NE_H", "the dipole field (k0) differs from the reference curvature (h); the IR "
                      "Bend holds only the geometry, k0 kept as a native passthrough",
                      element=ir_name, kind="Bend", k0=k0, h=hh)
        norm["kn"] = {i: v for i, v in k.items() if i >= 1 and v}
        norm["ks"] = {i: v for i, v in ks.items() if i >= 1 and v}
        rep.equivalent("MAGNET_AS_BEND", "xt.Magnet with curvature is read as a Bend (edge models and integrator "
                       "choices are xtrack's own)", element=ir_name, kind="Bend")
    else:
        lowest = next((i for i in (1, 2, 3) if k[i] or ks[i]), None)
        cls = {1: Quadrupole, 2: Sextupole, 3: Octupole}.get(lowest, Quadrupole)
        el = cls(name=ir_name, length=length)
        norm["kn"] = {i: v for i, v in k.items() if v}
        norm["ks"] = {i: v for i, v in ks.items() if v}
        if k[0] or ks[0]:
            rep.equivalent("MAGNET_DIPOLE_TERM", "the straight xt.Magnet's k0/k0s (a corrector field) rides "
                           "along as the element's order-0 multipole", element=ir_name, kind=el.kind)
        rep.equivalent("MAGNET_AS_MULTIPOLE_MAGNET",
                       f"a straight xt.Magnet is read as a {el.kind} carrying every order it has",
                       element=ir_name, kind=el.kind)
    if any(knl) or any(ksl):
        norm["knl"] = {i: v for i, v in enumerate(knl) if v}
        norm["ksl"] = {i: v for i, v in enumerate(ksl) if v}
        rep.equivalent("BEND_THIN_MULTIPOLES_FOLDED", "the magnet's thin knl/ksl kicks were folded into the "
                       "IR element's integrated multipole content", element=ir_name, kind=el.kind)
    for key in ("model", "integrator", "edge_entry_model", "edge_exit_model"):
        v = getattr(raw, key, None)
        if v not in (None, "", "adaptive", "linear"):
            el.native.setdefault("xtrack", {})[key] = str(v)
    return el


# ---------------------------------------------------------------------------
# folding

def misalignment_shift(raw: Any) -> BodyShiftP:
    """``xt.Misalignment`` → the IR body shift (dx, dy, ds, pitch phi, yaw theta, roll psi)."""
    return BodyShiftP(x_offset=_f(raw.dx), y_offset=_f(raw.dy), z_offset=_f(raw.ds),
                      x_rot=_f(raw.phi), y_rot=_f(raw.theta), tilt=_f(raw.psi))


def misalignment_as_patch(raw: Any, ir_name: str, rep: FidelityReport) -> Patch:
    """A ``Misalignment`` that does not enclose an element (no matching exit, or an exit without
    an entry) is a frame patch."""
    sign = -1.0 if bool(getattr(raw, "is_exit", False)) else 1.0
    el = Patch(name=ir_name, x_offset=sign * _f(raw.dx), y_offset=sign * _f(raw.dy), z_offset=sign * _f(raw.ds),
               x_rot=sign * _f(raw.phi), y_rot=sign * _f(raw.theta), tilt=sign * _f(raw.psi))
    el.native["xtrack"] = {"class": "Misalignment", "anchor": _f(raw.anchor), "is_exit": bool(raw.is_exit)}
    rep.equivalent("MISALIGNMENT_AS_PATCH", "an xt.Misalignment with no enclosed element is read as a frame patch",
                   element=ir_name, kind="Patch")
    return el


def fold_edge(raw: Any, cname: str, bend: Element | None, side: str, ir_name: str,
              rep: FidelityReport) -> Element | None:
    """A standalone ``DipoleEdge``/``MagnetEdge``/``DipoleFringe`` next to *bend* is folded into
    its pole face (returns ``None``); with no bend beside it, a ``DipoleEdge`` becomes the thin
    lens it is (``r21``/``r43``) and the others are dropped by name."""
    if cname == "DipoleEdge":
        e = _f(raw.e1)
        fint, hgap = _f(raw.fint), _f(raw.hgap)
        r21, r43 = _f(raw.r21), _f(raw.r43)
    elif cname == "MagnetEdge":
        e = _f(getattr(raw, "face_angle", 0.0))
        fint, hgap = _f(getattr(raw, "fringe_integral", 0.0)), _f(getattr(raw, "half_gap", 0.0))
        r21 = r43 = 0.0
    else:                                                         # DipoleFringe
        e = 0.0
        fint, hgap = _f(raw.fint), _f(raw.hgap)
        r21 = r43 = 0.0
    if isinstance(bend, Bend):
        b = bend.bend
        if side == "entry":
            if e:
                b.e1 = e
            if fint:
                b.edge_int1 = fint
        else:
            if e:
                b.e2 = e
            if fint:
                b.edge_int2 = fint
        if hgap:
            b.hgap = hgap
        rep.equivalent("DIPEDGE_FOLDED", f"an xt.{cname} beside its bend became the bend's "
                       f"{'e1' if side == 'entry' else 'e2'}/fint/hgap", element=bend.name, kind="Bend",
                       edge=ir_name)
        return None
    if cname == "DipoleEdge" and (r21 or r43):
        el = Taylor(name=ir_name, basis="xtrack")
        el.matrix[1][0] = r21
        el.matrix[3][2] = r43
        el.native["xtrack"] = {"class": cname, "element": _payload(raw)}    # re-emitted verbatim to xtrack
        rep.equivalent("DIPEDGE_AS_MATRIX", "an xt.DipoleEdge with no bend beside it is read as the thin lens "
                       "it applies (r21, r43)", element=ir_name, kind="Taylor", r21=r21, r43=r43)
        return el
    el = Marker(name=ir_name)
    el.native["xtrack"] = {"class": cname, "element": _payload(raw)}
    rep.dropped("EDGE_DROPPED", f"an xt.{cname} with no bend beside it has no IR form; written as a marker "
                "with a native passthrough", element=ir_name, kind="Marker", xtrack_class=cname)
    return el


def _payload(raw: Any) -> Any:
    import warnings

    from lattix.formats.xtrack.convert import _plain

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return _plain(raw.to_dict())
    except Exception:                                             # noqa: BLE001 - best effort
        return {"class": type(raw).__name__}


__all__ = ["DROPPED_CLASSES", "HANDLED_HERE", "MONITOR_CLASSES", "convert_extra", "fold_edge",
           "misalignment_as_patch", "misalignment_shift", "Kicker"]
