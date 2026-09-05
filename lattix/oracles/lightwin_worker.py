"""LightWin worker: run ``Envelope3D`` on a TraceWin deck and dump per-element maps as JSON.

Started by :mod:`lattix.oracles.lightwin` as ``<python> -I lightwin_worker.py deck.dat --out r.json …``
inside the LightWin environment (python ≥ 3.12).  Nothing here imports lattix.

LightWin's solver mesh has one point per integration step; an element's map is
``cumulated[out] · cumulated[in]⁻¹`` between the mesh indices its ``element_to_index`` gives,
in LightWin's native basis ``(x [m], x', y [m], y', z [m], dp/p)`` (the docstring of
``envelope_3d/transfer_matrices_p.py``: "units are taken exactly as in TraceWin").  The reference
energy follows the cavities (``w_kin`` per mesh point).  Elements LightWin has no 3-D model for
(GAP, EDGE, THIN_STEERING, apertures, diagnostics …) are propagated as drifts of their length by
LightWin itself; they are reported in ``substituted`` so the caller can downgrade the comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _beam_toml(dat: Path, results: Path, a: argparse.Namespace, tool_table: str) -> str:
    sigma = ",".join("[" + ",".join("1e-6" if i == j else "0.0" for j in range(6)) + "]" for i in range(6))
    return (f'[files]\ndat_file = "{dat.as_posix()}"\nproject_folder = "{results.as_posix()}"\n\n'
            f"{tool_table}\n"
            f"[beam]\ne_mev = {float(a.ke_ev / 1e6)!r}\ne_rest_mev = {float(a.mass_ev / 1e6)!r}\n"
            f"f_bunch_mhz = {float(a.freq_hz / 1e6)!r}\ni_milli_a = 0.0\nq_adim = {float(a.charge)!r}\n"
            f"sigma = [{sigma}]\n")


def _flip(tok: str) -> str:
    """The negated number as text (``-1.8`` → ``1.8``, ``2.4`` → ``-2.4``)."""
    v = float(tok)
    return f"{-v:.15g}" if v else tok


def _sanitized_copy(deck: Path, work: Path, captured: list[str], *, negative_charge: bool = False) -> Path:
    """A copy of the deck LightWin accepts: ``ERROR_*`` statistical-error commands are commented
    out (LightWin refuses them; they do not touch the reference optics) and a relative
    ``FIELD_MAP_PATH`` is made absolute so the copy still finds the maps.

    ``negative_charge``: LightWin's synchronous phase is charge-blind (an H⁻ run with
    ``q_adim = −1`` *decelerates* at φs = −30°, measured 2026-09-05), so a negative particle is
    emulated as a positive one of the same mass in the mirrored fields: QUAD gradients,
    THIN_STEERING fields and static field-map amplitudes (``kb``) are negated, and a relative
    (non-synchronous) RF phase is shifted by 180° — TraceWin's own rule (``lattix.ir.rf``)."""
    lines = deck.read_text(encoding="latin-1", errors="replace").splitlines()
    out, n_err, n_sol, n_flip = [], 0, 0, 0
    sync_armed = False
    for line in lines:
        body, _, comment = line.partition(";")
        words = body.split()
        key = words[0].upper() if words else ""
        label = ""
        if len(words) > 1 and words[0].endswith(":"):
            label, key, words = words[0] + " ", words[1].upper(), words[1:]
        tail = (" ;" + comment) if comment else ""
        if key.startswith("ERROR_"):
            out.append("; lattix: " + line)
            n_err += 1
            continue
        if negative_charge and key == "QUAD" and len(words) >= 3:
            words[2] = _flip(words[2])
            out.append(label + " ".join(words) + tail)
            n_flip += 1
            continue
        if negative_charge and key == "THIN_STEERING" and len(words) >= 3:
            words[1], words[2] = _flip(words[1]), _flip(words[2])
            out.append(label + " ".join(words) + tail)
            n_flip += 1
            continue
        if negative_charge and key == "FIELD_MAP" and len(words) >= 9:
            geom = int(float(words[1]))
            has_e = (geom // 100) % 10 != 0
            if has_e and not sync_armed:
                words[3] = f"{(float(words[3]) + 180.0) % 360.0:.15g}"      # relative RF phase: +180°
            words[5] = _flip(words[5])                                      # kb (static magnetic map)
            out.append(label + " ".join(words) + tail)
            n_flip += 1
            sync_armed = False
            continue
        if key == "SET_SYNC_PHASE":
            sync_armed = True
        elif key in ("FIELD_MAP", "GAP", "NCELLS", "DTL_CEL"):
            sync_armed = False
        if key == "SOLENOID" and len(words) >= 3:
            # LightWin 0.16.5: SolenoidEnvelope3DParameters raises NotImplementedError; a drift of
            # the same length keeps s and the energy profile, the transverse block is lost (audited)
            args = words[1:] if not words[0].endswith(":") else words[2:]
            label = words[0] + " " if words[0].endswith(":") else ""
            out.append(f"{label}DRIFT {args[0]} {args[2] if len(args) > 2 else 30} 0 0 0 ; lattix: was {line.strip()}")
            n_sol += 1
            continue
        if key == "FIELD_MAP_PATH" and len(words) > 1:
            target = Path(words[1])
            if not target.is_absolute():
                target = (deck.parent / target).resolve()
            out.append(f"FIELD_MAP_PATH {target}")
            continue
        out.append(line)
    if n_err:
        captured.append(f"{n_err} ERROR_* command(s) commented out (LightWin does not implement them)")
    if n_sol:
        captured.append(f"SOLENOID_AS_DRIFT: {n_sol} SOLENOID card(s) written as drifts "
                        "(Envelope3D has no solenoid model: NotImplementedError)")
    if negative_charge:
        captured.append(f"NEGATIVE_CHARGE_EMULATED: a positive particle of the same mass in mirrored fields "
                        f"({n_flip} card(s) adjusted); LightWin's synchronous phase is charge-blind")
    has_path = any(ln.split()[:1] == ["FIELD_MAP_PATH"] for ln in out)
    if not has_path:
        out.insert(0, f"FIELD_MAP_PATH {deck.parent}")
    local = work / deck.name
    local.write_text("\n".join(out) + "\n", encoding="latin-1")
    return local


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("deck")
    p.add_argument("--out", required=True)
    p.add_argument("--ke-ev", type=float, required=True)
    p.add_argument("--mass-ev", type=float, required=True)
    p.add_argument("--charge", type=float, default=1.0)
    p.add_argument("--freq-hz", type=float, required=True, help="bunch frequency (the deck's FREQ sets the RF)")
    p.add_argument("--n-steps", type=int, default=40, help="Envelope3D n_steps_per_cell")
    p.add_argument("--phase-policy", default="as_in_original_dat",
                   choices=("as_in_original_dat", "phi_0_rel", "phi_0_abs", "phi_s"))
    p.add_argument("--parse-only", action="store_true", help="element table only, no propagation")
    a = p.parse_args(argv)

    import numpy as np

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                captured.append(record.getMessage().replace("\n", " ")[:300])

    logging.getLogger().addHandler(_Capture())
    logging.getLogger().setLevel(logging.WARNING)

    import lightwin
    import lightwin.config.config_manager as config_manager
    from lightwin.core.accelerator.factory import AcceleratorFactory
    from lightwin.ui.workflow_setup import set_up_solvers

    work = Path.cwd()
    negative = a.charge < 0 and not a.parse_only      # the parse table shows the deck as written
    deck = _sanitized_copy(Path(a.deck).resolve(), work, captured, negative_charge=negative)
    if negative:
        a.charge = -a.charge                       # LightWin runs the mirrored, positive-charge deck
    results = work / "results"
    tool = ('[envelope3d]\ntool = "Envelope3D"\n'
            f"n_steps_per_cell = {a.n_steps}\nreference_phase_policy = \"{a.phase_policy}\"\n")
    toml = work / "lightwin.toml"
    toml.write_text(_beam_toml(deck, results, a, tool))
    config = config_manager.process_config(toml, {"files": "files", "beam_calculator": "envelope3d",
                                                  "beam": "beam"})
    solver = set_up_solvers(reset_factory=True, **config)[0]
    accelerator = AcceleratorFactory(beam_calculators=solver, **config).create_reference()
    elts = list(accelerator.elts)
    so = None
    if a.parse_only:
        try:                                    # solver parameters (element models) without propagation
            solver.init_solver_parameters(accelerator)
        except Exception as exc:                # noqa: BLE001
            captured.append(f"init_solver_parameters failed: {type(exc).__name__}: {exc}"[:300])
    else:
        so = solver.compute(accelerator)        # sets beam_calc_param (the per-element models) too

    table = []
    for e in elts:
        param = e.beam_calc_param.get(solver.id) if hasattr(e, "beam_calc_param") else None
        row = {"name": str(e.name), "kind": type(e).__name__, "length_m": float(getattr(e, "length_m", 0.0) or 0.0),
               "model": type(param).__name__ if param is not None else None,
               "line": str(getattr(e, "line", "") or "")[:200]}
        for attr in ("grad",):
            if hasattr(e, attr):
                row[attr] = float(getattr(e, attr))
        cs = getattr(e, "cavity_settings", None)
        if cs is not None:
            # only attributes that do not trigger LightWin's lazy phase conversions (those need
            # the propagation and raise MissingAttributeError before it)
            for attr in ("k_e", "reference", "phi_ref"):
                try:
                    v = getattr(cs, attr, None)
                except Exception:                                   # noqa: BLE001
                    continue
                if v is not None and not callable(v):
                    row[attr] = float(v) if isinstance(v, (int, float)) else str(v)
        table.append(row)
    # deck lines that produced neither an element nor a command: LightWin skips keywords it does
    # not implement (GAP, NCELLS, DTL_CEL, …) without a trace, which silently removes physics
    dropped = []
    try:
        files = getattr(accelerator.elts, "files", {}) or {}
        for ins in files.get("elts_n_cmds") or []:
            if type(ins).__name__ != "Dummy":         # LightWin's placeholder for keywords it does not know
                continue
            idx = getattr(ins, "idx", None)
            i = int(idx["dat_idx"]) if isinstance(idx, dict) and "dat_idx" in idx else int(idx or -1)
            line = getattr(ins, "line", None)
            key = str(getattr(line, "instruction", "") or "").upper()
            if not key:
                words = str(getattr(line, "original_line", line) or "").split()
                key = words[0].upper() if words else "?"
            dropped.append({"dat_idx": i, "keyword": key})
    except Exception as exc:                                    # noqa: BLE001
        captured.append(f"could not audit dropped lines: {type(exc).__name__}: {exc}"[:300])
    drift_like = {"Drift", "Diagnostic"}
    substituted = [r["name"] for r in table
                   if r["model"] == "DriftEnvelope3DParameters" and r["kind"] not in drift_like]
    data = {"lightwin_version": getattr(lightwin, "__version__", "?"), "python": sys.version.split()[0],
            "tool": "Envelope3D", "phase_policy": a.phase_policy, "n_steps_per_cell": a.n_steps,
            "elements": table, "substituted": substituted, "dropped": dropped, "warnings": captured}
    if so is not None:
        tm = so.transfer_matrix
        cum = np.asarray(tm.cumulated, dtype=float)
        z = np.asarray(so.z_abs, dtype=float)
        w = np.asarray(so.get("w_kin", to_numpy=True), dtype=float)
        names, kinds, length, s_out, R, w_in, w_out, params = [], [], [], [], [], [], [], []
        for e, row in zip(elts, table, strict=True):
            i_in = int(so.element_to_index(e, pos="in"))
            i_out = int(so.element_to_index(e, pos="out"))
            r = cum[i_out] @ np.linalg.inv(cum[i_in])
            names.append(row["name"])
            kinds.append(row["kind"])
            length.append(float(z[i_out] - z[i_in]))
            s_out.append(float(z[i_out]))
            R.extend(float(x) for x in r.reshape(-1))
            w_in.append(float(w[i_in]) * 1e6)
            w_out.append(float(w[i_out]) * 1e6)
            extra = {}
            if row["kind"].startswith("FieldMap"):
                cs = getattr(e, "cavity_settings", None)
                for key in ("v_cav_mv", "phi_s", "phi_0_rel", "phi_0_abs"):
                    try:
                        val = getattr(cs, key, None) if cs is not None else None
                        extra[key] = float(val) if isinstance(val, (int, float)) else None
                    except Exception as exc:                    # noqa: BLE001 - optional diagnostics
                        extra[key] = None
                        captured.append(f"{row['name']}: {key} unavailable ({type(exc).__name__})")
            params.append(extra)
        data.update({"names": names, "kinds": kinds, "length": length, "s_out": s_out, "R": R,
                     "w_in_ev": w_in, "w_out_ev": w_out, "params": params, "n_mesh": int(cum.shape[0]),
                     "w_end_ev": float(w[-1]) * 1e6})
    Path(a.out).write_text(json.dumps(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
