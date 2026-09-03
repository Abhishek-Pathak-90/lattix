"""Positional argument schemas for TraceWin ``.dat`` cards.

Ported from HELIX ``linac_gen/io/tracewin_syntax.py`` (SCHEMA + ``parse_positionals``) with two
extra columns per field:

* ``unit`` — the unit of the token *in the deck* (``mm``, ``deg``, ``T/m``, ``V``, ``MHz`` …; the
  IR is SI + eV, the reader/writer convert at the boundary);
* ``writer_default`` — the value the writer treats as "default" when eliding trailing tokens
  (``_SAME`` means "same as ``default``"; ``EDGE`` K1/K2 elide at TraceWin's own 0.45 / 2.80).

Entries whose value is ``None`` are recognised keywords that are not parsed positionally
(``TITLE``, ``END``, ``ADJUST_BEAM_*`` with a variable flag tail …).  Card-name classification
sets used by the reader (diagnostics, hardware markers, command roles) live at the bottom.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_SAME = object()


@dataclass(frozen=True)
class Field:
    name: str
    cast: Callable[[str], Any]
    required: bool = False
    default: Any = None
    unit: str | None = None
    writer_default: Any = _SAME

    @property
    def elide_value(self) -> Any:
        """Value at which the writer may drop this (trailing) token."""
        return self.default if self.writer_default is _SAME else self.writer_default


def _int(tok: str) -> int:
    """TraceWin integer operands are often written as floats (``00``, ``1.0``)."""
    return int(float(tok))


def parse_positionals(fields: list[Field], tokens: list[str]) -> dict:
    """Turn ``tokens`` into ``{name: value}`` per ``fields``.

    Raises ``ValueError`` when a required field is missing or a cast fails.  Extra trailing
    tokens are ignored here (the reader records them as ``EXTRA_TOKENS_IGNORED``).
    """
    required_min = sum(1 for f in fields if f.required)
    if len(tokens) < required_min:
        names = ", ".join(f.name for f in fields if f.required)
        raise ValueError(
            f"card requires at least {required_min} positional args ({names}); got {len(tokens)}"
        )
    out: dict = {}
    for i, field in enumerate(fields):
        if i < len(tokens):
            try:
                out[field.name] = field.cast(tokens[i])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"failed to cast {field.name}={tokens[i]!r}: {exc}") from exc
        else:
            out[field.name] = field.default
    return out


# --------------------------------------------------------------------------
# Per-card schemas (TraceWin ordering, authoritative)
# --------------------------------------------------------------------------
SCHEMA: dict[str, list[Field] | None] = {
    "DRIFT": [
        Field("length", float, required=True, unit="mm"),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("aperture_y", float, default=None, unit="mm"),  # None / 0 => circular
        Field("x_shift", float, default=0.0, unit="mm"),
        Field("y_shift", float, default=0.0, unit="mm"),
    ],
    "QUAD": [
        Field("length", float, required=True, unit="mm"),
        Field("gradient", float, required=True, unit="T/m"),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("skew_angle", float, default=0.0, unit="deg"),
        Field("g3", float, default=0.0, unit="T/m^2"),
        Field("g4", float, default=0.0, unit="T/m^3"),
        Field("g5", float, default=0.0, unit="T/m^4"),
        Field("g6", float, default=0.0, unit="T/m^5"),
        Field("gfr", float, default=0.0, unit="mm"),  # good-field radius
    ],
    "SOLENOID": [
        Field("length", float, required=True, unit="mm"),
        Field("field", float, required=True, unit="T"),
        Field("aperture", float, default=0.0, unit="mm"),
    ],
    "GAP": [
        Field("e0tl", float, required=True, unit="V"),  # effective gap voltage (TTF folded)
        Field("phase", float, required=True, unit="deg"),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("p_flag", _int, default=0),  # 0 relative, 1 absolute, 2/3 variants
    ],
    "FIELD_MAP": [
        Field("geom", _int, required=True),  # 5-digit geometry code (may be < 0)
        Field("length", float, required=True, unit="mm"),
        Field("phase", float, default=0.0, unit="deg"),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("kb", float, default=1.0),  # magnetic scale
        Field("ke", float, default=1.0),  # electric scale
        Field("ki", float, default=0.0),  # space-charge compensation
        Field("ka", float, default=1.0),  # aperture scale
        Field("filename", str, required=True),
        Field("p_flag", _int, default=0),
    ],
    "FIELD_MAP_PATH": [
        Field("path", str, required=True),
    ],
    "BEND": [
        Field("angle", float, required=True, unit="deg"),
        Field("rho", float, required=True, unit="mm"),
        Field("field_index", float, default=0.0),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("hv", _int, default=0),  # 0 horizontal, 1 vertical
    ],
    "EDGE": [
        Field("pole_rotation", float, required=True, unit="deg"),
        Field("rho", float, required=True, unit="mm"),
        Field("gap", float, default=0.0, unit="mm"),  # full magnetic gap
        Field("k1", float, default=0.45),  # fringe factor (TraceWin default)
        Field("k2", float, default=2.80),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("hv", _int, default=0),
    ],
    "APERTURE": [
        Field("dx", float, required=True, unit="mm"),
        Field("dy", float, default=0.0, unit="mm"),
        Field(
            "ap_type", _int, default=0
        ),  # 0 rect, 1 circle/ellipse, 2 pepperpot, 3 fraction, 4/5 finger, 6 ring
    ],
    "THIN_STEERING": [  # alias STEERER
        Field("bl_x", float, required=True, unit="T.m"),  # ∫Bx dl  (V when elec=1)
        Field("bl_y", float, required=True, unit="T.m"),
        Field("aperture", float, default=0.0, unit="mm"),
        Field("elec", _int, default=0),  # 0 magnetic, 1 electric
    ],
    "FREQ": [
        Field("frequency", float, required=True, unit="MHz"),
    ],
    "LATTICE": [
        Field("n_elements", _int, default=0),
        Field("n_periods", _int, default=0),
    ],
    "SUPERPOSE_MAP": [
        Field("z0", float, required=True, unit="mm"),
        Field("x0", float, default=0.0, unit="mm"),
        Field("y0", float, default=0.0, unit="mm"),
        Field("theta_z", float, default=0.0, unit="deg"),
        Field("theta_x", float, default=0.0, unit="deg"),
        Field("theta_y", float, default=0.0, unit="deg"),
    ],
    "SHIFT_IN_FIELD_MAP": [
        Field("dz", float, required=True, unit="mm"),
    ],
    "SPACE_CHARGE_COMP": [
        Field("factor", float, required=True),
    ],
    "PARTRAN_STEP": [
        Field("steps_per_metre", float, required=True),
        Field("sc_steps_per_metre", float, default=0.0),
    ],
    "DIAG_POSITION": [  # N X Y [dm]; |X|,|Y| >= 1e50 = unconstrained
        Field("diag", _int, default=None),
        Field("x_target", float, default=None, unit="mm"),
        Field("y_target", float, default=None, unit="mm"),
        Field("accuracy", float, default=1.0, unit="mm"),
    ],
    "NCELLS": [  # + optional TTF tail βs Ts kT's k2T''s Ti … To …
        Field("mode", _int, required=True),  # 0 = 2π, 1 = π, 2 = π&2π
        Field("n_cells", _int, required=True),
        Field("beta_g", float, required=True),
        Field("eot", float, required=True, unit="V/m"),
        Field("theta_s", float, required=True, unit="deg"),
        Field("aperture", float, required=True, unit="mm"),
        Field("p_flag", _int, default=0),
        Field("k_eot_i", float, default=0.0),
        Field("k_eot_o", float, default=0.0),
        Field("dz_i", float, default=0.0, unit="mm"),
        Field("dz_o", float, default=0.0, unit="mm"),
    ],
    "RFQ_CELL": [
        Field("voltage", float, required=True, unit="V"),
        Field("r0", float, required=True, unit="mm"),
        Field("a10", float, required=True),
        Field("modulation", float, required=True),
        Field("length", float, required=True, unit="mm"),
        Field("phi_s", float, required=True, unit="deg"),
        Field("cell_type", _int, required=True),
        Field("tc", float, default=0.0, unit="mm"),
        Field("dp", float, default=0.0, unit="deg"),
    ],
    # ------------------------------------------------------------------
    # SET / ADJUST family (TraceWin matching language).  Parsed only to fill
    # ``Directive.args`` typed views; the cards are re-emitted verbatim.
    # ------------------------------------------------------------------
    "SET_SYNC_PHASE": [],
    "SET_BEAM_PHASE_ERROR": [
        Field("dphi_deg", float, default=0.0, unit="deg"),
        Field("random_flag", _int, default=0),
    ],
    "SET_BEAM_E0_P0": [
        Field("k", _int, default=0),
        Field("dE_MeV", float, default=0.0, unit="MeV"),
        Field("dphi_deg", float, default=0.0, unit="deg"),
        Field("ke", _int, default=0),
        Field("kp", _int, default=0),
    ],
    "SET_BEAM_ENERGY": [Field("k", _int, default=0), Field("energy_MeV", float, required=True, unit="MeV")],
    "SET_GAUSSIAN_CUT_OFF": [Field("sigma", float, default=4.0)],
    "SET_TWISS": [
        Field("family", str, default=""),
        Field("alpha_x", float, default=0.0),
        Field("beta_x", float, default=0.0, unit="mm/mrad"),
        Field("alpha_y", float, default=0.0),
        Field("beta_y", float, default=0.0, unit="mm/mrad"),
        Field("alpha_z", float, default=0.0),
        Field("beta_z", float, default=0.0),
        Field("kax", _int, default=0),
        Field("kbx", _int, default=0),
        Field("kay", _int, default=0),
        Field("kby", _int, default=0),
        Field("kaz", _int, default=0),
        Field("kbz", _int, default=0),
    ],
    "SET_POSITION": [
        Field("k", float, default=0.0),
        Field("x_mm", float, default=0.0, unit="mm"),
        Field("xp_mrad", float, default=0.0, unit="mrad"),
        Field("y_mm", float, default=0.0, unit="mm"),
        Field("yp_mrad", float, default=0.0, unit="mrad"),
    ],
    "SET_ACHROMAT": [
        Field("k", _int, default=0),
        Field("f1", _int, default=0),
        Field("f2", _int, default=0),
        Field("plane", _int, default=0),
    ],
    "SET_SIZE": [
        Field("k", float, default=0.0),
        Field("x_mm", float, default=0.0, unit="mm"),
        Field("y_mm", float, default=0.0, unit="mm"),
        Field("phi_or_z", float, default=0.0),
        Field("k2", _int, default=0),
    ],
    "SET_SIZE_MAX": [
        Field("k", float, default=0.0),
        Field("n_elems", _int, default=1),
        Field("x_mm", float, default=0.0, unit="mm"),
        Field("y_mm", float, default=0.0, unit="mm"),
        Field("phi_or_z", float, default=0.0),
        Field("k2", _int, default=0),
    ],
    "SET_SIZE_MIN": [
        Field("k", float, default=0.0),
        Field("n_elems", _int, default=1),
        Field("x_mm", float, default=0.0, unit="mm"),
        Field("y_mm", float, default=0.0, unit="mm"),
        Field("phi_or_z", float, default=0.0),
        Field("k2", _int, default=0),
    ],
    "SET_BEAM_PHASE_ADV": [
        Field("k", float, default=0.0),
        Field("n_elems", _int, default=1),
        Field("mu_x_deg", float, default=0.0, unit="deg"),
        Field("mu_y_deg", float, default=0.0, unit="deg"),
        Field("mu_z_deg", float, default=0.0, unit="deg"),
    ],
    "SET_SEPARATION": [
        Field("k", float, default=0.0),
        Field("sx", float, default=0.0),
        Field("sy", float, default=0.0),
    ],
    "SET_ADV": [Field("kxot", float, default=0.0), Field("kyot", float, default=0.0)],
    "MIN_EMIT_GROWTH": [Field("plane", str, required=True), Field("weight", float, default=1.0)],
    "MIN_EMIT_4D_GROWTH": [
        Field("weight", float, default=1.0),
        Field("tol_4d", float, default=1.0),
        Field("tol_z", float, default=1.0),
    ],
    "SET_KE_OUT_MIN": [
        Field("energy_mev", float, required=True, unit="MeV"),
        Field("weight", float, default=1.0),
    ],
    "MIN_TRANSMISSION": [Field("threshold_pct", float, default=99.0), Field("weight", float, default=1.0)],
    "ADJUST": [
        Field("target", str, required=True),
        Field("param_idx", _int, required=True),
        Field("link_group", _int, default=0),
        Field("vmin", float, default=0.0),
        Field("vmax", float, default=0.0),
        Field("start_step", float, default=0.0),
        Field("kn", _int, default=0),
    ],
    "ADJUST_STEERER": [
        Field("diag_n", _int, required=True),
        Field("vmax", float, default=0.0),
        Field("first_step", float, default=0.0),
    ],
    "ADJUST_STEERER_BX": [
        Field("diag_n", _int, required=True),
        Field("vmax", float, default=0.0),
        Field("first_step", float, default=0.0),
    ],
    "ADJUST_STEERER_BY": [
        Field("diag_n", _int, required=True),
        Field("vmax", float, default=0.0),
        Field("first_step", float, default=0.0),
    ],
    # Control / variable-tail cards: recognised, not positionally typed.
    "STEERER": None,  # alias of THIN_STEERING
    "ADJUST_BEAM_TWISS": None,
    "ADJUST_BEAM_CENTROID": None,
    "ADJUST_BEAM_EMIT": None,
    "ADJUST_BEAM_CURRENT": None,
    "LATTICE_END": None,
    "TITLE": None,
    "END": None,
    "MARKER": None,
    "REPEAT_ELE": None,
}

#: TraceWin's default RF frequency assumed when an RF card precedes any FREQ (HELIX
#: ``tracewin_parser._DEFAULT_FREQ_MHZ``).
DEFAULT_FREQ_MHZ = 352.21

#: Diagnostic cards mapped to ``Instrument`` (HELIX: Marker/BPM no-ops).  The first four carry
#: TraceWin matching data; the rest are the PIP-II deck hardware vocabulary (``BPM :`` …).
DIAGNOSTIC_CARDS: frozenset[str] = frozenset({"DIAG_SIZE", "DIAG_EMIT", "DIAG_PHASE", "DIAG_POSITION"})
HARDWARE_MARKER_CARDS: frozenset[str] = frozenset(
    {
        "BPM",
        "XCOR",
        "YCOR",
        "ACCT",
        "COL",
        "RPU",
        "FFC",
        "DPI",
        "DCCT",
        "RWCM",
        "ASCN",
        "FASTGV",
        "LASERPROFILE",
        "MEBTABSORBER",
        "CHOPPER",
    }
)

#: Command cards kept inline as ``Directive`` with a role (writers of other formats drop them).
COMMAND_ROLES: dict[str, str] = {
    "SET_SYNC_PHASE": "sync_phase",
    "LATTICE": "period_start",
    "LATTICE_END": "period_end",
    "TITLE": "title",
    "PARTRAN_STEP": "tracking",
    "SPACE_CHARGE_COMP": "tracking",
    "CHANGE_FREQ": "tracking",
    "PLOT_DST": "tracking",
    "READ_DST": "beam",
    "BEAM_ROT": "beam",
    "SET_BEAM_PHASE_ERROR": "beam",
    "SET_GAUSSIAN_CUT_OFF": "beam",
    "BUNCHED_BEAM": "beam",
    "SET_BEAM_CURRENT": "beam",
    "SUPERPOSE_MAP": "superpose",
    "SHIFT_IN_FIELD_MAP": "superpose",
    "SUPERPOSE_MAP_OUT": "superpose",
    "RFQ_GEOM": "rfq",
    "RFQ_GAP_RMS_FFS": "rfq",
    "RFQ_ELE": "rfq",
    "REPEAT_ELE": "other",
    "SET_TWISS": "matching",
    "SET_POSITION": "matching",
    "SET_ACHROMAT": "matching",
    "SET_SIZE": "matching",
    "SET_SIZE_MAX": "matching",
    "SET_SIZE_MIN": "matching",
    "SET_BEAM_PHASE_ADV": "matching",
    "SET_SEPARATION": "matching",
    "SET_ADV": "matching",
    "MIN_EMIT_GROWTH": "matching",
    "MIN_EMIT_GROW": "matching",
    "MIN_EMIT_4D_GROWTH": "matching",
    "MIN_TRANSMISSION": "matching",
    "SET_KE_OUT_MIN": "matching",
    "ADJUST": "matching",
    "ADJUST_STEERER": "matching",
    "ADJUST_STEERER_BX": "matching",
    "ADJUST_STEERER_BY": "matching",
    "ADJUST_BEAM_TWISS": "matching",
    "ADJUST_BEAM_CENTROID": "matching",
    "ADJUST_BEAM_EMIT": "matching",
    "ADJUST_BEAM_CURRENT": "matching",
    "MATCH_FAM_FIELD": "matching",
    "MATCH_FAM_GRAD": "matching",
    "MATCH_FAM_PHASE": "matching",
    "MIN_FIELD_VARIATION": "matching",
    "START_ACHROMAT": "matching",
    "DIAG_ACHROMAT": "matching",
    "DIAG_ENERGY": "matching",
    "DIAG_TWISS": "matching",
    "DIAG_WAIST": "matching",
    "DIAG_DSIZE": "matching",
    "DIAG_DPHASE": "matching",
    "DIAG_DENERGY": "matching",
    "DIAG_DIVERGENCE": "matching",
    "DIAG_SIZE_MAX": "matching",
    "DIAG_SIZE_MIN": "matching",
    "DIAG_SIZE_P": "matching",
    "DIAG_SIZE_FWHM": "matching",
    "DIAG_EMIT_99": "matching",
    "DIAG_HALO": "matching",
    "DIAG_SET_MATRIX": "matching",
    "DIAG_CURRENT": "matching",
    "DIAG_DCURRENT": "matching",
    "DIAG_LUMINOSITY": "matching",
    "DIAG_SEPARATION": "matching",
    "DIAG_PHASE_ADV": "matching",
    "DIAG_CENTROID": "matching",
    "DIAG_DCENTROID": "matching",
}

#: Recognised TraceWin elements the IR does not model (kept verbatim, LOSSY: they may carry length).
UNSUPPORTED_ELEMENT_CARDS: dict[str, str] = {
    "THIN_LENS": "thin lens with independent focal lengths",
    "THIN_MATRIX": "explicit thin matrix",
    "DTL_CEL": "DTL cell (thick)",
    "CAVSIN": "sinusoidal cavity (thick)",
    "FUNNEL_GAP": "funnel gap",
    "ELECTRODE": "electrostatic electrode",
    "ELECTROSTA_QUAD": "electrostatic quadrupole",
    "MULTIPOLE": "TraceWin multipole",
    "GAP_ERROR": "gap error",
    "BUNCHER": "buncher",
}
