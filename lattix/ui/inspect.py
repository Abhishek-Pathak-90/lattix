"""Per-element numbers for the inspector (derived at the entrance reference, as the writers compute them)
and the location of an element's statement in a written deck."""
from __future__ import annotations

import math
import re
import urllib.parse

import numpy as np

from lattix.ir.lattice import Lattice, Placed
from lattix.ir.normalize import bl_from_kick, field_index_from_k1, k1_from_gradient, kn_from_bn, ks_from_field
from lattix.ir.reference import ReferenceParticle
from lattix.ir.walk import energy_gain_eV

C_LIGHT = 299_792_458.0

#: units of the derived numbers (the page shows them; docs/ui.md lists them)
UNITS: dict[str, str] = {
    "k1": "1/m²", "k1s": "1/m²", "K1L": "1/m", "GL": "T", "focal_length_m": "m", "skew_angle_rad": "rad",
    "k2": "1/m³", "K2L": "1/m²", "k3": "1/m⁴", "K3L": "1/m³", "knl": "1/mⁿ", "ksl": "1/mⁿ",
    "hkick": "rad", "vkick": "rad", "rho_m": "m", "B0_T": "T", "field_index": "", "chord_m": "m",
    "angle_deg": "deg", "e1_deg": "deg", "e2_deg": "deg", "ks": "1/m", "int_B_Tm": "T·m", "larmor_rad": "rad",
    "V_eff_V": "V", "gain_eV": "eV", "gain_explicit": "", "phase_deg": "deg", "lambda_m": "m", "beta_lambda_m": "m",
    "harmonic": "", "rf_defocusing_1_per_m": "1/m", "int_Bdl_x_Tm": "T·m", "int_Bdl_y_Tm": "T·m",
    "det_x": "", "det_y": "", "symplectic_err": "", "thickness_mg_cm2": "mg/cm²", "phase_advance_deg": "deg",
}


def reference_state(ref: ReferenceParticle | None) -> dict | None:
    if ref is None:
        return None
    sp = ref.species
    f = ref.rf_frequency_Hz
    lam = C_LIGHT / f if f else None
    return {"species": sp.name, "mass_eV": sp.mass_eV, "charge": sp.charge, "ke_eV": ref.kinetic_energy_eV,
            "gamma": ref.gamma, "beta": ref.beta, "betagamma": ref.beta * ref.gamma, "pc_eV": ref.pc_eV,
            "total_energy_eV": ref.total_energy_eV, "brho_abs_Tm": ref.brho_abs, "brho_signed_Tm": ref.brho_signed,
            "time_s": ref.time_s, "rf_frequency_Hz": f, "wavelength_m": lam,
            "beta_lambda_m": ref.beta * lam if lam else None}


def _mp(mp, order: int, ref: ReferenceParticle, d: dict, key: str, skey: str, lkey: str) -> None:
    bn, bs = float(mp.Bn.get(order, 0.0)), float(mp.Bs.get(order, 0.0))
    d[key] = kn_from_bn(order, bn, ref) if ref.brho_signed else None
    if bs:
        d[skey] = kn_from_bn(order, bs, ref) if ref.brho_signed else None


def derived_numbers(p: Placed, lat: Lattice) -> dict:
    """What a physicist reads off an element beyond its stored fields: normalized strengths, bend geometry,
    RF gains and lengths, kicker field integrals, map invariants — at the entrance reference."""
    e = p.element
    ref = p.ref_in or lat.reference
    d: dict = {}
    L = float(p.length)
    k = e.kind
    if k == "Quadrupole":
        G = float(e.multipole.Bn.get(1, 0.0))
        k1 = k1_from_gradient(G, ref) if ref.brho_signed else 0.0
        d["k1"] = k1
        if e.multipole.Bs.get(1):
            d["k1s"] = k1_from_gradient(float(e.multipole.Bs[1]), ref)
        d["K1L"] = k1 * L
        d["GL"] = G * L
        d["focal_length_m"] = 1.0 / (k1 * L) if k1 * L else None
        if e.multipole.tilt.get(1):
            d["skew_angle_rad"] = float(e.multipole.tilt[1])
    elif k == "Sextupole":
        _mp(e.multipole, 2, ref, d, "k2", "k2s", "K2L")
        d["K2L"] = (d["k2"] or 0.0) * L
    elif k == "Octupole":
        _mp(e.multipole, 3, ref, d, "k3", "k3s", "K3L")
        d["K3L"] = (d["k3"] or 0.0) * L
    elif k == "Multipole":
        b = ref.brho_signed
        if b:
            d["knl"] = {str(n): float(v) / b for n, v in e.multipole.BnL.items() if v}
            d["ksl"] = {str(n): float(v) / b for n, v in e.multipole.BsL.items() if v}
            if e.multipole.BnL.get(0):
                d["hkick"] = -float(e.multipole.BnL[0]) / b
            if e.multipole.BsL.get(0):
                d["vkick"] = float(e.multipole.BsL[0]) / b
    elif k == "Bend":
        b = e.bend
        a = float(b.angle)
        d["angle_deg"] = math.degrees(a)
        d["e1_deg"], d["e2_deg"] = math.degrees(float(b.e1)), math.degrees(float(b.e2))
        if a and L:
            rho = L / a
            d["rho_m"] = rho
            d["B0_T"] = ref.brho_signed * a / L
            d["chord_m"] = 2.0 * rho * math.sin(a / 2.0)
            G = float(e.multipole.Bn.get(1, 0.0))
            if G and ref.brho_signed:
                k1 = k1_from_gradient(G, ref)
                d["k1"] = k1
                d["field_index"] = field_index_from_k1(k1, rho)
        d["angle_h"] = a * math.cos(float(b.tilt_ref))
        d["angle_v"] = a * math.sin(float(b.tilt_ref))
    elif k == "Solenoid":
        B = float(e.solenoid.Bsol_T)
        ks = ks_from_field(B, ref) if ref.brho_signed else 0.0
        d["ks"] = ks
        d["int_B_Tm"] = B * L
        d["larmor_rad"] = 0.5 * ks * L
    elif k in ("RFCavity", "FieldMap", "NCells", "RFQCell", "Superposition"):
        rf = getattr(e, "rf", None)
        if rf is not None:
            V = float(rf.voltage_V or 0.0)
            if not V and rf.gradient_V_per_m is not None:
                V = float(rf.gradient_V_per_m) * float(rf.L_active_m if rf.L_active_m is not None else L)
            d["V_eff_V"] = V
            d["gain_eV"] = energy_gain_eV(e, ref)
            d["gain_explicit"] = rf.dE_ref_eV is not None
            d["phase_deg"] = math.degrees(float(rf.phase_rad))
            f = rf.frequency_Hz or ref.rf_frequency_Hz
            if f:
                lam = C_LIGHT / float(f)
                d["lambda_m"], d["beta_lambda_m"] = lam, ref.beta * lam
                if ref.rf_frequency_Hz:
                    d["harmonic"] = float(f) / ref.rf_frequency_Hz
                if k == "RFCavity" and L == 0.0 and V:
                    from lattix.ir.rf import thin_gap_defocusing

                    out = ref.advanced(dE_eV=d["gain_eV"])
                    d["rf_defocusing_1_per_m"] = thin_gap_defocusing(V, float(rf.phase_rad), float(f),
                                                                     ref.species.mass_eV, out.beta * out.gamma)
        s = (e.meta or {}).get("map_summary")
        if s:
            d["map_summary"] = dict(s)
            if s.get("phase_advance_deg") is not None:
                d["phase_advance_deg"] = s["phase_advance_deg"]
        if k == "FieldMap":
            try:
                from lattix.ir.fieldmap import replacement_for

                r = replacement_for(e)
                d["replacement"] = {"code": r.code, "cls": r.cls, "message": r.message,
                                    "parts": [{"name": x.name, "kind": x.kind, "length": float(x.length)}
                                              for x in r.parts]}
            except Exception:  # noqa: BLE001 - a summary-less map has nothing to replace it with
                pass
    elif k == "Kicker":
        if not e.electric and ref.brho_signed:
            d["int_Bdl_x_Tm"] = bl_from_kick(float(e.hkick), ref)
            d["int_Bdl_y_Tm"] = bl_from_kick(float(e.vkick), ref)
    elif k == "Taylor":
        m = np.asarray(e.matrix, dtype=float)
        if m.shape == (6, 6):
            d["det_x"] = float(np.linalg.det(m[:2, :2]))
            d["det_y"] = float(np.linalg.det(m[2:4, 2:4]))
            j = np.zeros((6, 6))
            for i in range(3):
                j[2 * i, 2 * i + 1], j[2 * i + 1, 2 * i] = 1.0, -1.0
            d["symplectic_err"] = float(np.max(np.abs(m.T @ j @ m - j)))
    elif k == "Foil":
        d["thickness_mg_cm2"] = float(e.thickness_kg_per_m2) * 100.0
    f = ref.rf_frequency_Hz
    if f and L and "phase_advance_deg" not in d:
        d["phase_advance_deg"] = 360.0 * L * f / (ref.beta * C_LIGHT)
    return d


# ---------------------------------------------------------------------------------------------- statements
_TAG_NAME = re.compile(r"lattix:[^\n]*?\bname=(?:\"([^\"]*)\"|'([^']*)'|(\S+))")
_COMMENT_START = ("!", "//", "#", ";", "%")
_INLINE_TAG_FORMATS = {"madx", "elegant", "opal", "ocelot", "bmad", "scibmad"}
_JSON_FORMATS = {"cheetah", "synergia", "xtrack", "lattix"}


def _tag_name(line: str) -> str | None:
    m = _TAG_NAME.search(line)
    if not m:
        return None
    v = m.group(1) if m.group(1) is not None else (m.group(2) if m.group(2) is not None else m.group(3))
    return urllib.parse.unquote(v)


def _is_comment(line: str) -> bool:
    t = line.lstrip()
    return not t or t.startswith(_COMMENT_START)


def _sanitized(name: str, fmt: str) -> set[str]:
    out = {name}
    try:
        if fmt in ("madx", "madng"):
            from lattix.formats.madx.naming import sanitize
        elif fmt == "elegant":
            from lattix.formats.elegant.naming import sanitize
        elif fmt == "bmad":
            from lattix.formats.bmad.naming import sanitize
        elif fmt == "mad8":
            from lattix.formats.mad8.writer import sanitize
        elif fmt == "flame":
            from lattix.formats.flame.writer import sanitize
        elif fmt == "scibmad":
            from lattix.formats.scibmad.writer import sanitize
        elif fmt == "impactx":
            from lattix.formats.impactx.writer import sanitize
        elif fmt == "pals":
            from lattix.formats.pals.writer import sanitize
        else:
            sanitize = None
        if sanitize is not None:
            out.add(sanitize(name))
    except Exception:  # noqa: BLE001 - a sanitizer that rejects the name adds nothing
        pass
    out.add(re.sub(r"[^0-9A-Za-z_]", "_", name))
    return {x for x in out if x}


def locate_statements(lines: list[str], fmt: str, name: str, original_name: str | None = None,
                      prov_line: int | None = None, *, cap: int = 8) -> list[dict]:
    """Where the element ``name`` sits in a written deck: its ``lattix: name=`` tag line (and the statement
    it annotates), the line its reader reported, the statement starting with its identifier; then plain
    references.  ``[{"line": 1-based, "text", "role": tag|definition|reference}]``, best effort."""
    hits: list[dict] = []
    seen: set[int] = set()

    def add(no: int, role: str) -> None:
        if 0 <= no < len(lines) and no not in seen and len(hits) < cap:
            seen.add(no)
            hits.append({"line": no + 1, "text": lines[no].rstrip("\n"), "role": role})

    for no, line in enumerate(lines):
        if "lattix:" in line and _tag_name(line) == name:
            add(no, "tag")
            if fmt in _INLINE_TAG_FORMATS and not _is_comment(line):
                continue
            k = no + 1
            while k < len(lines) and _is_comment(lines[k]):
                k += 1
            add(k, "definition")
    if prov_line:
        add(prov_line - 1, "definition")
    idents = _sanitized(name, fmt) | ({original_name} if original_name else set())
    flags = re.IGNORECASE if fmt in ("madx", "mad8", "elegant", "bmad", "opal", "madng") else 0
    for ident in sorted(idents, key=len, reverse=True):
        esc = re.escape(ident)
        if fmt in _JSON_FORMATS:
            pat = re.compile(r'"name"\s*:\s*"' + esc + r'"', flags)
        elif fmt == "pyorbit":
            pat = re.compile(r'name="' + esc + r'"', flags)
        elif fmt in ("scibmad", "ocelot"):
            pat = re.compile(r"^\s*" + esc + r"\s*=", flags)
        elif fmt == "tracewin":
            pat = re.compile(r"^\s*" + esc + r"\s*:", flags)
        else:
            pat = re.compile(r"^\s*" + esc + r"\s*:", flags)
        for no, line in enumerate(lines):
            if len(hits) >= cap:
                break
            if pat.search(line):
                add(no, "definition")
        word = re.compile(r"(?<![\w.])" + esc + r"(?![\w.])", flags)
        for no, line in enumerate(lines):
            if len(hits) >= cap:
                break
            if no not in seen and word.search(line) and not _is_comment(line):
                add(no, "reference")
    return hits
