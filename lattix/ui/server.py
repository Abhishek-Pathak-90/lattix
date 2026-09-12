"""The ``lattix ui`` server: a loopback-only standard-library HTTP server serving the page and a JSON API
(read a deck, translate it, compare before and after, run the engines in a job, download the results).
Every request must carry the session token; sessions live in a temporary directory removed on exit."""
from __future__ import annotations

import atexit
import io
import ipaddress
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import webbrowser
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport, TranslationError
from lattix.formats.base import FORMATS, guess_format, read, write
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.walk import propagate
from lattix.ui.jobs import Job, JobManager
from lattix.ui.model import (
    canonical_read_options,
    catalog_view,
    coerce_options,
    format_catalog,
    jsonable,
    lattice_view,
    ledger_view,
    sample_decks,
)

_LOCAL_ONLY = frozenset({"tracewin", "dynac", "synergia", "helix"})
_DECK_NAMES = {"impactz": "ImpactZ.in", "impactt": "ImpactT.in"}


class ApiError(Exception):
    def __init__(self, status: int, type_: str, message: str, **extra) -> None:
        super().__init__(message)
        self.status, self.type, self.extra = status, type_, extra


# ------------------------------------------------------------------------------------------------ state
def token_file() -> Path:
    """Where the access token persists between runs (``$LATTIX_CONFIG_DIR``, else ``$XDG_CONFIG_HOME/lattix``,
    else ``~/.config/lattix``)."""
    base = os.environ.get("LATTIX_CONFIG_DIR")
    if base:
        return Path(base) / "ui-token"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "lattix" / "ui-token"


def stable_token(path: Path | None = None, *, rotate: bool = False) -> str:
    """The access token of this user's UI: kept in a private file (mode 0600) so the link ``lattix ui``
    prints stays valid across restarts; ``rotate`` writes a fresh one.  Falls back to a random token when
    the file cannot be used."""
    path = path or token_file()
    try:
        if not rotate and path.is_file():
            tok = path.read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{16,}", tok):
                return tok
        tok = secrets.token_urlsafe(24)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(tok + "\n")
        os.chmod(path, 0o600)
        return tok
    except OSError:
        return secrets.token_urlsafe(24)


_NEED_LINK_HTML = """<!doctype html><meta charset="utf-8"><title>lattix ui</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:40em;margin:4em auto;padding:0 1em;color:#1e293b}
code{background:#eef2f7;padding:1px 4px;border-radius:3px}</style>
<h2>lattix ui: this page needs its access link</h2>
<p>Open the link that <code>lattix ui</code> printed in the terminal — it carries the access token
(<code>http://127.0.0.1:&lt;port&gt;/?token=…</code>).  A plain address without the token, or a link
from a server started with <code>--new-token</code>, is refused.</p>
<p>The link is the same each time you start <code>lattix ui</code> as this user; bookmark it once.
If the terminal is gone, rebuild it: the token is the content of <code>~/.config/lattix/ui-token</code>
(<code>cat ~/.config/lattix/ui-token</code>), so the address is
<code>http://127.0.0.1:PORT/?token=</code> followed by that value.</p>
"""


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 0
    root: Path = field(default_factory=Path.cwd)
    any_path: bool = False
    open_browser: bool = True
    max_body: int = 64 * 2**20
    session_limit: int = 16
    job_timeout_s: float = 1800.0
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    quiet: bool = True


@dataclass
class Loaded:
    path: Path
    fmt: str
    options: dict
    lattice: Lattice
    report: FidelityReport
    placed: list[Placed]
    walk_warnings: list[str]
    view: dict
    text: str


@dataclass
class Translation:
    id: str
    fmt: str
    options: dict
    strict: bool
    dir: Path
    deck: Path
    side_files: list[Path]
    report: FidelityReport
    text: str
    target: Loaded | None
    target_error: dict | None
    comparison: dict | None
    tier: str
    payload: dict


@dataclass
class Session:
    id: str
    dir: Path
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    source: Loaded | None = None
    translations: dict[str, Translation] = field(default_factory=dict)
    n_translations: int = 0


class SessionStore:
    def __init__(self, root: Path, limit: int) -> None:
        self.root, self.limit = root, limit
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        with self._lock:
            sid = secrets.token_urlsafe(6)
            s = Session(sid, self.root / sid)
            s.dir.mkdir(parents=True, exist_ok=True)
            self._sessions[sid] = s
            while len(self._sessions) > self.limit:
                oldest = min(self._sessions.values(), key=lambda x: x.last_used)
                self._drop(oldest.id)
            return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.last_used = time.time()
            return s

    def _drop(self, sid: str) -> None:
        s = self._sessions.pop(sid, None)
        if s is not None:
            shutil.rmtree(s.dir, ignore_errors=True)

    def drop(self, sid: str) -> bool:
        with self._lock:
            if sid not in self._sessions:
                return False
            self._drop(sid)
            return True

    def close(self) -> None:
        with self._lock:
            for sid in list(self._sessions):
                self._drop(sid)


class App:
    """Everything the handler needs: settings, sessions, jobs, the cached catalogues, the engine probe."""

    def __init__(self, settings: Settings, *, oracle_probe: Callable[[], dict] | None = None,
                 validation_runner: Callable[[Job, dict], dict] | None = None) -> None:
        self.settings = settings
        self.tmp_root = Path(tempfile.mkdtemp(prefix="lattix_ui_"))
        self.sessions = SessionStore(self.tmp_root, settings.session_limit)
        self.jobs = JobManager(max_workers=1, default_timeout_s=settings.job_timeout_s)
        self._oracle_probe = oracle_probe
        self.validation_runner = validation_runner or _subprocess_validation
        self.oracles: dict = {"ready": False, "probing": False, "engines": {}}
        self._oracle_lock = threading.Lock()
        self.started = time.time()
        atexit.register(self.close)

    # -- paths
    def resolve_path(self, user_path: str) -> Path:
        p = Path(user_path).expanduser()
        p = (self.settings.root / p).resolve() if not p.is_absolute() else p.resolve()
        root = self.settings.root.resolve()
        if not self.settings.any_path and root not in (p, *p.parents):
            raise ApiError(403, "forbidden", f"{user_path!r} is outside the UI root {root} (start with --any-path "
                                             "or --root to allow it)")
        return p

    # -- engines
    def start_oracle_probe(self, *, refresh: bool = False) -> None:
        with self._oracle_lock:
            if self.oracles["probing"] or (self.oracles["ready"] and not refresh):
                return
            self.oracles["probing"] = True
        threading.Thread(target=self._probe, args=(refresh,), daemon=True, name="lattix-oracle-probe").start()

    def _probe(self, refresh: bool) -> None:
        engines: dict = {}
        try:
            if self._oracle_probe is not None:
                found = self._oracle_probe()
            else:
                from lattix.oracles import available_oracles, get_oracle

                if refresh:
                    from lattix.oracles.base import _REGISTRY

                    for cls in _REGISTRY.values():
                        reset = getattr(cls, "reset_cache", None)
                        if callable(reset):
                            reset()
                found = {}
                for name, (ok, why) in available_oracles().items():
                    try:
                        fmts = list(get_oracle(name).formats)
                    except Exception:  # noqa: BLE001
                        fmts = []
                    found[name] = (ok, why, fmts)
            for name, info in found.items():
                ok, why, fmts = (info + ([],))[:3] if len(info) == 2 else info
                engines[name] = {"available": bool(ok), "why": str(why), "formats": list(fmts),
                                 "local_only": name in _LOCAL_ONLY}
        except Exception as exc:  # noqa: BLE001
            engines = {"error": {"available": False, "why": f"{type(exc).__name__}: {exc}", "formats": [],
                                 "local_only": False}}
        with self._oracle_lock:
            self.oracles = {"ready": True, "probing": False, "engines": engines}

    def close(self) -> None:
        try:
            self.jobs.shutdown()
            self.sessions.close()
        finally:
            shutil.rmtree(self.tmp_root, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ actions
def _walk(lat: Lattice) -> tuple[list[Placed], list[str]]:
    warnings: list[str] = []
    return propagate(lat, warnings=warnings), warnings


def load_source(app: App, session: Session, body: dict) -> Loaded:
    src = body.get("source") or {}
    fmt = body.get("format") or None
    options = dict(body.get("options") or {})
    if "sample" in src:
        from lattix.crossval import PUBLIC

        rel = str(src["sample"])
        path = (PUBLIC / rel).resolve()
        if not path.is_file() or PUBLIC.resolve() not in path.parents:
            raise ApiError(404, "not_found", f"no sample deck {rel!r}")
        for s in sample_decks():
            if s["path"] == rel:
                fmt = fmt or s["format"]
                options = {**s["options"], **options}
    elif "content" in src:
        name = re.sub(r"[^\w.\-]", "_", str(src.get("filename") or "deck.dat")) or "deck.dat"
        d = session.dir / "source"
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        path.write_text(str(src["content"]), encoding="latin-1", errors="replace")
    elif "path" in src:
        raw = str(src["path"])
        p = Path(raw)
        if not p.is_absolute() and (session.dir / "source" / p).is_file():
            path = (session.dir / "source" / p).resolve()             # an upload of this session
        else:
            path = app.resolve_path(raw)
        if not path.is_file():
            raise ApiError(404, "not_found", f"no such file: {raw}")
    else:
        raise ApiError(400, "bad_request", "source must give a sample, a path or content")
    if fmt is None:
        try:
            fmt = guess_format(path)
        except ValueError as exc:
            raise ApiError(400, "bad_request", f"{exc}; pass format") from None
    if fmt not in FORMATS or not FORMATS[fmt].reader_attr:
        raise ApiError(400, "bad_request", f"no reader for format {fmt!r}")
    warnings: list[str] = []
    if "base_dir" in options:
        options["base_dir"] = str(app.resolve_path(str(options["base_dir"])))
    opts = canonical_read_options(fmt, options, warnings)
    lat, rep = read(path, fmt, **opts)
    placed, walk_warnings = _walk(lat)
    try:
        text = path.read_text(encoding="latin-1", errors="replace")
    except OSError:
        text = ""
    view = lattice_view(lat, rep, fmt=fmt, path=path, placed=placed, walk_warnings=warnings + walk_warnings)
    view["detected_format"] = fmt
    return Loaded(path, fmt, opts, lat, rep, placed, warnings + walk_warnings, view, text)


def translate(app: App, session: Session, body: dict) -> Translation:
    from lattix.crossval import _read_options, tier_of
    from lattix.ui.compare import compare_translation

    if session.source is None:
        raise ApiError(409, "conflict", "read a source deck first")
    fmt = str(body.get("format") or "")
    if fmt not in FORMATS or not FORMATS[fmt].writer_attr:
        raise ApiError(400, "bad_request", f"no writer for format {fmt!r}")
    strict = bool(body.get("strict", False))
    reread = bool(body.get("reread", True))
    options = coerce_options(fmt, "write", body.get("options") or {})
    with session.lock:
        session.n_translations += 1
        tid = f"t{session.n_translations}"
        tdir = session.dir / tid
        tdir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.\-]", "_", session.source.path.name.split(".")[0]) or "lattice"
        suffix = FORMATS[fmt].suffixes[0]
        deck = tdir / _DECK_NAMES.get(fmt, f"{stem}{suffix if suffix.startswith('.') else '.' + suffix}")
        src = session.source
        try:
            rep = write(src.lattice, deck, fmt, strict=strict, **options)
        except TranslationError as exc:
            shutil.rmtree(tdir, ignore_errors=True)
            session.n_translations -= 1
            raise ApiError(422, "translation", str(exc), entry=exc.entry.model_dump(mode="json")) from None
        except Exception:
            shutil.rmtree(tdir, ignore_errors=True)
            session.n_translations -= 1
            raise
        text = deck.read_text(encoding="latin-1", errors="replace") if deck.is_file() else ""
        side = sorted(p for p in tdir.iterdir() if p.is_file() and p != deck)
        target: Loaded | None = None
        target_error: dict | None = None
        comparison: dict | None = None
        warnings: list[str] = []
        if reread and FORMATS[fmt].reader_attr:
            try:
                lat2, rep2 = read(deck, fmt, **_read_options(fmt, src.lattice.reference.species))
                placed2, w2 = _walk(lat2)
                view2 = lattice_view(lat2, rep2, fmt=fmt, path=deck, placed=placed2, walk_warnings=w2,
                                     deck_text=text)
                target = Loaded(deck, fmt, {}, lat2, rep2, placed2, w2, view2, text)
                comparison = compare_translation(src.lattice, rep, lat2, rep2, src_fmt=src.fmt, dst_fmt=fmt,
                                                 placed=src.placed, placed2=placed2)
            except Exception as exc:  # noqa: BLE001 - the deck is written; reading it back is a bonus
                target_error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                warnings.append(f"the written deck could not be read back: {type(exc).__name__}: "
                                f"{str(exc)[:200]}")
        payload = jsonable({
            "translation": tid, "format": fmt, "tier": tier_of(rep), "strict": strict, "options": options,
            "deck": {"name": deck.name, "text": text, "lines": text.count("\n") + (1 if text else 0),
                     "bytes": len(text.encode("latin-1", errors="replace"))},
            "side_files": [{"name": p.name, "bytes": p.stat().st_size} for p in side],
            "fidelity": ledger_view(rep), "target": target.view if target else None,
            "target_error": target_error, "comparison": comparison, "warnings": warnings,
        })
        tr = Translation(tid, fmt, options, strict, tdir, deck, side, rep, text, target, target_error,
                         comparison, tier_of(rep), payload)
        session.translations[tid] = tr
        return tr


def _subprocess_validation(job: Job, spec: dict) -> dict:
    """Run ``lattix validate --json`` in a child interpreter (in-process engines never share the server)."""
    out = Path(spec["workdir"]) / "validate.json"
    cmd = [sys.executable, "-m", "lattix.cli", "validate", "--json", str(out), "--workdir", str(spec["workdir"]),
           "--oracles", ",".join(spec["engines"])]
    for fmt, path in spec["decks"].items():
        cmd += ["--deck", f"{fmt}={path}"]
    if spec.get("tier"):
        cmd += ["--tier", str(spec["tier"])]
    if spec.get("codes"):
        cmd += ["--codes", ",".join(spec["codes"])]
    beam = spec.get("beam") or {}
    for key, flag in (("species", "--species"), ("kinetic_energy_eV", "--ke"), ("frequency_Hz", "--freq"),
                      ("betx", "--betx"), ("alfx", "--alfx"), ("bety", "--bety"), ("alfy", "--alfy"),
                      ("dx", "--dx"), ("dpx", "--dpx")):
        if beam.get(key) is not None:
            cmd += [flag, str(beam[key])]
    job.say("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=spec["workdir"])
    job.proc = proc
    total = len(spec["engines"])
    done = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        job.say(line)
        head = line.split()
        if head and head[0] in spec["engines"] and (len(head) > 1 and head[1] in ("skipped:", "failed:")
                                                    or "n=" in line):
            done += 1
            job.set_progress(min(done, total), total, head[0])
    rc = proc.wait()
    if job.stopped():
        raise RuntimeError("cancelled")
    if not out.is_file():
        raise RuntimeError(f"lattix validate exited with {rc} and wrote no result (see the log)")
    result = json.loads(out.read_text())
    result["exit_code"] = rc
    return result


def _code_counts(report) -> list[str]:
    """``CODE=count`` for every non-EXACT ledger code (the verdict names the optics-affecting ones)."""
    counts: dict[str, int] = {}
    for e in report.entries:
        if e.cls.value != "EXACT":
            counts[e.code] = counts.get(e.code, 0) + 1
    return [f"{c}={n}" for c, n in sorted(counts.items())]


def start_validation(app: App, session: Session, body: dict) -> Job:
    from lattix.crossval import beam_from_lattice, pick_engine

    if session.source is None:
        raise ApiError(409, "conflict", "read a source deck first")
    tid = str(body.get("translation") or "")
    tr = session.translations.get(tid)
    src = session.source
    decks: dict[str, str] = {src.fmt: str(src.path)}
    if tr is not None:
        decks[tr.fmt] = str(tr.deck)
    for fmt, p in (body.get("extra_decks") or {}).items():
        if fmt in FORMATS:
            decks[str(fmt)] = str(app.resolve_path(str(p)))
    engines: list[str] = []
    for e in body.get("engines") or []:
        name = e if isinstance(e, str) else str(e.get("name", ""))
        if name and name not in engines:
            engines.append(name)
    if not engines:
        for fmt in decks:
            try:
                e = pick_engine(fmt)
            except Exception:  # noqa: BLE001
                e = None
            if e and e not in engines:
                engines.append(e)
    if len(engines) < 2:
        raise ApiError(400, "bad_request", "pick at least two engines (one per deck format, or two on the source)")
    beam = dict(body.get("beam") or {})
    if not all(beam.get(k) for k in ("species", "kinetic_energy_eV")):
        # anything the form left out comes from the source lattice — never from an engine's own default
        try:
            b = beam_from_lattice(src.lattice)
        except RuntimeError as exc:
            raise ApiError(400, "bad_request", f"{exc}; pick a named species in the beam form") from None
        fallback = (("species", b.species), ("kinetic_energy_eV", b.kinetic_energy_eV),
                    ("frequency_Hz", b.frequency_Hz))
        for k, v in fallback:
            if not beam.get(k) and v is not None:
                beam[k] = v
    wd = (tr.dir if tr is not None else session.dir) / "validate"
    wd.mkdir(parents=True, exist_ok=True)
    spec = {"decks": decks, "engines": engines, "beam": beam, "workdir": str(wd),
            "tier": tr.tier if tr is not None else "exact",          # two engines on one deck: exact
            "codes": _code_counts(tr.report) if tr is not None else []}

    def run(job: Job) -> dict:
        job.set_progress(0, len(engines), engines[0])
        return app.validation_runner(job, spec)

    return app.jobs.submit("validate", run, meta={"session": session.id, "translation": tid, "engines": engines,
                                                    "decks": decks})


# ------------------------------------------------------------------------------------------------ HTTP
_HOSTS = ("127.0.0.1", "localhost", "[::1]")
_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self' data:")


def _page() -> bytes:
    return resources.files("lattix.ui").joinpath("static/index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = f"lattix/{__version__}"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # noqa: D102 - quiet by default
        if not self.app.settings.quiet:
            super().log_message(fmt, *args)

    # -- helpers
    def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(jsonable(obj), allow_nan=False).encode("utf-8"))

    def _error(self, status: int, type_: str, message: str, **extra) -> None:
        self._json(status, {"error": {"type": type_, "message": message, **extra}})

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > self.app.settings.max_body:
            # drain (bounded) so the client can finish sending and read the 413, then drop the connection
            left = min(n, 4 * self.app.settings.max_body)
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                left -= len(chunk)
            self.close_connection = True
            raise ApiError(413, "too_large", f"body of {n} bytes exceeds the limit of {self.app.settings.max_body}")
        return self.rfile.read(n) if n else b""

    def _body(self) -> bytes:
        raw = getattr(self, "_raw", None)
        return raw if raw is not None else self._read_body()

    def _json_body(self) -> dict:
        raw = self._body()
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(400, "bad_request", f"invalid JSON body: {exc}") from None
        if not isinstance(obj, dict):
            raise ApiError(400, "bad_request", "the body must be a JSON object")
        return obj

    def _authorised(self, query: dict, path: str = "") -> bool:
        host = (self.headers.get("Host") or "").strip()
        port = self.server.server_address[1]  # type: ignore[attr-defined]
        if host not in {f"{h}:{port}" for h in _HOSTS} | set(_HOSTS):
            self._error(403, "forbidden", "unexpected Host header")
            return False
        token = self.headers.get("X-Lattix-Token") or (query.get("token") or [None])[0]
        if not token or not secrets.compare_digest(token, self.app.settings.token):
            if self.command == "GET" and path in ("/", "/index.html"):
                # a person typed the bare address or kept an old link: say so in words, not JSON
                page = _NEED_LINK_HTML.replace("PORT", str(port))
                self._send(403, page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._error(403, "forbidden", "missing or wrong token (open the URL lattix ui printed)")
            return False
        origin = self.headers.get("Origin")
        if self.command in ("POST", "PUT", "DELETE") and origin:
            if origin.rstrip("/") not in {f"http://{h}:{port}" for h in _HOSTS}:
                self._error(403, "forbidden", "cross-origin request refused")
                return False
        return True

    # -- dispatch
    def _dispatch(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        path = parsed.path
        self._raw = None
        if not self._authorised(query, path):
            self.close_connection = True          # the body (if any) was not read
            return
        try:
            # the body is consumed here, once, whether or not the route wants it: on a keep-alive connection
            # an unread body would be parsed as the start of the browser's next request
            self._raw = self._read_body()
            for method, pattern, fn in ROUTES:
                if method != self.command:
                    continue
                m = pattern.fullmatch(path)
                if m:
                    fn(self, query, *m.groups())
                    return
            self._error(404, "not_found", f"no route {self.command} {path}")
        except ApiError as exc:
            self._error(exc.status, exc.type, str(exc), **exc.extra)
        except TranslationError as exc:
            self._error(422, "translation", str(exc), entry=exc.entry.model_dump(mode="json"))
        except FileNotFoundError as exc:
            self._error(404, "not_found", str(exc))
        except PermissionError as exc:
            self._error(403, "forbidden", str(exc))
        except (ValueError, KeyError, TypeError) as exc:
            self._error(400, "bad_request", f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "MissingDependencyError":
                self._error(424, "missing_dependency", str(exc))
                return
            traceback.print_exc()
            self._error(500, "internal", f"{type(exc).__name__}: {exc}")

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _dispatch

    # -- routes
    def r_page(self, query) -> None:
        self._send(200, _page(), "text/html; charset=utf-8", {"Content-Security-Policy": _CSP})

    def r_ping(self, query) -> None:
        self._json(200, {"ok": True, "version": __version__, "root": str(self.app.settings.root),
                         "any_path": self.app.settings.any_path, "uptime_s": time.time() - self.app.started})

    def r_formats(self, query) -> None:
        self._json(200, format_catalog())

    def r_samples(self, query) -> None:
        self._json(200, {"decks": sample_decks()})

    def r_catalog(self, query) -> None:
        self._json(200, catalog_view())

    def r_oracles(self, query) -> None:
        refresh = (query.get("refresh") or ["0"])[0] in ("1", "true")
        self.app.start_oracle_probe(refresh=refresh)
        self._json(200, self.app.oracles)

    def r_browse(self, query) -> None:
        raw = (query.get("path") or [""])[0] or "."
        p = self.app.resolve_path(raw)
        if not p.is_dir():
            raise ApiError(404, "not_found", f"not a directory: {raw}")
        entries = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if child.name.startswith("."):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            fmt = None
            if child.is_file():
                try:
                    fmt = guess_format(child.name)
                except ValueError:
                    fmt = None
            entries.append({"name": child.name, "dir": child.is_dir(), "bytes": st.st_size, "mtime": st.st_mtime,
                            "format": fmt})
            if len(entries) >= 2000:
                break
        root = self.app.settings.root.resolve()
        parent = str(p.parent) if p != root or self.app.settings.any_path else None
        self._json(200, {"path": str(p), "parent": parent, "root": str(root), "entries": entries})

    def r_session_create(self, query) -> None:
        s = self.app.sessions.create()
        self._json(201, {"session": s.id})

    def _session(self, sid: str) -> Session:
        s = self.app.sessions.get(sid)
        if s is None:
            raise ApiError(404, "not_found", f"no session {sid!r}")
        return s

    def r_upload(self, query, sid: str) -> None:
        s = self._session(sid)
        rel = (query.get("path") or [""])[0]
        parts = [x for x in Path(rel).parts if x not in ("", ".", "..") and not x.startswith("/")]
        if not parts:
            raise ApiError(400, "bad_request", "path= must name the file (a relative path)")
        dest = (s.dir / "source").joinpath(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._body())
        self._json(201, {"path": "/".join(parts), "bytes": dest.stat().st_size})

    def r_read(self, query) -> None:
        body = self._json_body()
        sid = body.get("session")
        s = self._session(str(sid)) if sid else self.app.sessions.create()
        with s.lock:
            s.source = load_source(self.app, s, body)
            s.translations.clear()
            s.n_translations = 0
        self._json(201, {"session": s.id, "detected_format": s.source.fmt, "source": s.source.view,
                         "read_options": s.source.options})

    def r_session(self, query, sid: str) -> None:
        s = self._session(sid)
        self._json(200, {"session": s.id, "source": s.source.view if s.source else None,
                         "translations": {tid: tr.payload for tid, tr in s.translations.items()}})

    def r_session_delete(self, query, sid: str) -> None:
        self._json(200, {"dropped": self.app.sessions.drop(sid)})

    def r_source_text(self, query, sid: str) -> None:
        s = self._session(sid)
        if s.source is None:
            raise ApiError(409, "conflict", "no source deck in this session")
        self._json(200, {"name": s.source.path.name, "text": s.source.text})

    def r_survey(self, query, sid: str) -> None:
        """The survey table of the session's source, or of a re-read translation's target with
        ``?translation=``: ``format=csv`` as an attachment, otherwise JSON.  ``at`` picks the frame
        (``entrance``, ``centre``, ``exit``, ``body`` or ``all``), ``shift=0`` ignores misalignments,
        ``children=0`` keeps superposition clusters whole, and ``x0 … psi0`` set the start pose."""
        from lattix.ir.frames import frame_survey, site_frame, survey_csv, survey_table

        s = self._session(sid)
        tid = (query.get("translation") or [None])[0]
        if tid:
            _, tr = self._translation(sid, tid)
            loaded = tr.target
            if loaded is None:
                raise ApiError(409, "conflict", f"translation {tid!r} was not re-read; no target lattice to survey")
        else:
            loaded = s.source
            if loaded is None:
                raise ApiError(409, "conflict", "no source deck in this session")

        def flag(name: str, default: str = "1") -> bool:
            return (query.get(name) or [default])[0].lower() not in ("0", "false", "no")

        names = ("x0", "y0", "z0", "theta0", "phi0", "psi0")
        try:
            pose = tuple(float((query.get(k) or ["0"])[0]) for k in names)
        except ValueError as exc:
            raise ApiError(400, "bad_request", f"start pose must be numbers: {exc}") from None
        at = (query.get("at") or ["exit"])[0]
        shift, children = flag("shift"), flag("children")
        frames = frame_survey(loaded.placed, lat=loaded.lattice, start=site_frame(*pose),
                              apply_shift=shift, expand_children=children)
        try:
            rows = survey_table(frames, at=at)
        except ValueError as exc:
            raise ApiError(400, "bad_request", str(exc)) from None
        fmt = (query.get("format") or ["json"])[0]
        if fmt == "csv":
            comments = [f"lattix {__version__} survey of {loaded.path.name} ({loaded.fmt}): {len(rows)} rows",
                        f"frames: {at}; misalignments {'applied to the body frame' if shift else 'ignored'}; "
                        f"superposition children {'expanded' if children else 'not expanded'}",
                        "start pose (MAD-X SURVEY x0 y0 z0 theta0 phi0 psi0; m, rad): "
                        + " ".join(f"{v:.12g}" for v in pose),
                        "units: m and rad; theta, phi, psi are MAD-X survey angles, theta continuous along the line"]
            self._file(survey_csv(rows, comments=comments).encode("utf-8"),
                       f"{loaded.path.stem}.survey.csv", "text/csv; charset=utf-8")
        elif fmt == "json":
            self._json(200, {"deck": loaded.path.name, "format": loaded.fmt, "at": at, "shift": shift,
                             "children": children, "start": dict(zip(names, pose, strict=True)),
                             "columns": list(rows[0]) if rows else [], "rows": rows})
        else:
            raise ApiError(400, "bad_request", f"unknown survey format {fmt!r}; use csv or json")

    def r_translate(self, query) -> None:
        body = self._json_body()
        s = self._session(str(body.get("session") or ""))
        tr = translate(self.app, s, body)
        self._json(200, tr.payload)

    def _translation(self, sid: str, tid: str) -> tuple[Session, Translation]:
        s = self._session(sid)
        tr = s.translations.get(tid)
        if tr is None:
            raise ApiError(404, "not_found", f"no translation {tid!r}")
        return s, tr

    def r_translation(self, query, sid: str, tid: str) -> None:
        self._json(200, self._translation(sid, tid)[1].payload)

    def r_download(self, query, sid: str, tid: str | None = None) -> None:
        what = (query.get("what") or ["deck"])[0]
        s = self._session(sid)
        if what == "source":
            if s.source is None:
                raise ApiError(409, "conflict", "no source deck")
            self._file(s.source.path.read_bytes(), s.source.path.name)
            return
        if tid is None:
            raise ApiError(400, "bad_request", "a translation is needed for that download")
        _, tr = self._translation(sid, tid)
        if what == "deck":
            self._file(tr.deck.read_bytes(), tr.deck.name)
        elif what == "fidelity":
            self._file(tr.report.to_json().encode("utf-8"), f"{tr.deck.stem}.fidelity.json", "application/json")
        elif what == "side":
            name = (query.get("name") or [""])[0]
            p = next((x for x in tr.side_files if x.name == name), None)
            if p is None:
                raise ApiError(404, "not_found", f"no side file {name!r}")
            self._file(p.read_bytes(), p.name)
        elif what == "zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in [tr.deck, *tr.side_files]:
                    z.write(p, p.name)
                z.writestr("fidelity.json", tr.report.to_json())
            self._file(buf.getvalue(), f"{tr.deck.stem}.{tr.fmt}.zip", "application/zip")
        else:
            raise ApiError(400, "bad_request", f"unknown download {what!r}")

    def _file(self, data: bytes, name: str, ctype: str = "application/octet-stream") -> None:
        self._send(200, data, ctype, {"Content-Disposition": f'attachment; filename="{name}"'})

    def r_validate(self, query) -> None:
        body = self._json_body()
        s = self._session(str(body.get("session") or ""))
        job = start_validation(self.app, s, body)
        self._json(202, {"job": job.id})

    def r_job(self, query, jid: str) -> None:
        job = self.app.jobs.get(jid)
        if job is None:
            raise ApiError(404, "not_found", f"no job {jid!r}")
        self._json(200, job.to_dict())

    def r_job_cancel(self, query, jid: str) -> None:
        if self.app.jobs.get(jid) is None:
            raise ApiError(404, "not_found", f"no job {jid!r}")
        self._json(200, {"cancelled": self.app.jobs.cancel(jid)})


ROUTES: list[tuple[str, re.Pattern, Callable]] = [
    ("GET", re.compile(r"/|/index\.html"), Handler.r_page),
    ("HEAD", re.compile(r"/|/index\.html"), Handler.r_page),
    ("GET", re.compile(r"/api/ping"), Handler.r_ping),
    ("GET", re.compile(r"/api/formats"), Handler.r_formats),
    ("GET", re.compile(r"/api/samples"), Handler.r_samples),
    ("GET", re.compile(r"/api/catalog"), Handler.r_catalog),
    ("GET", re.compile(r"/api/oracles"), Handler.r_oracles),
    ("GET", re.compile(r"/api/browse"), Handler.r_browse),
    ("POST", re.compile(r"/api/session"), Handler.r_session_create),
    ("PUT", re.compile(r"/api/session/([\w\-]+)/upload"), Handler.r_upload),
    ("POST", re.compile(r"/api/read"), Handler.r_read),
    ("GET", re.compile(r"/api/session/([\w\-]+)"), Handler.r_session),
    ("DELETE", re.compile(r"/api/session/([\w\-]+)"), Handler.r_session_delete),
    ("GET", re.compile(r"/api/session/([\w\-]+)/source_text"), Handler.r_source_text),
    ("GET", re.compile(r"/api/session/([\w\-]+)/survey"), Handler.r_survey),
    ("GET", re.compile(r"/api/session/([\w\-]+)/download"), Handler.r_download),
    ("POST", re.compile(r"/api/translate"), Handler.r_translate),
    ("GET", re.compile(r"/api/session/([\w\-]+)/translation/([\w\-]+)"), Handler.r_translation),
    ("GET", re.compile(r"/api/session/([\w\-]+)/translation/([\w\-]+)/download"), Handler.r_download),
    ("POST", re.compile(r"/api/validate"), Handler.r_validate),
    ("GET", re.compile(r"/api/jobs/([\w\-]+)"), Handler.r_job),
    ("POST", re.compile(r"/api/jobs/([\w\-]+)/cancel"), Handler.r_job_cancel),
]


# ------------------------------------------------------------------------------------------------ lifecycle
def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.app = app  # type: ignore[attr-defined]
    return srv


def start_in_thread(app: App, host: str = "127.0.0.1",
                    port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    srv = make_server(app, host, port)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="lattix-ui")
    t.start()
    return srv, t, srv.server_address[1]


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def serve(settings: Settings, *, check: bool = False, deck: str | None = None) -> int:
    if not _is_loopback(settings.host):
        print(f"lattix ui: refusing to bind {settings.host!r}: the UI serves the local browser only "
              "(127.0.0.1 or localhost)", file=sys.stderr)
        return 2
    app = App(settings)
    srv, thread, port = start_in_thread(app, settings.host, settings.port)
    url = f"http://{settings.host}:{port}/?token={settings.token}"
    try:
        sid = None
        if deck:
            s = app.sessions.create()
            s.source = load_source(app, s, {"source": {"path": deck}})
            sid = s.id
            url += f"&session={sid}"
        if check:
            import urllib.request

            req = urllib.request.Request(f"http://{settings.host}:{port}/api/ping",
                                         headers={"X-Lattix-Token": settings.token})
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = json.loads(resp.read().decode("utf-8")).get("ok") is True
            print(f"lattix ui: {url}", flush=True)
            print("lattix ui: self-test " + ("passed" if ok else "FAILED"), flush=True)
            return 0 if ok else 1
        print(f"lattix ui: {url}", flush=True)
        print("lattix ui: the link stays the same on the next start (--new-token changes it); Ctrl-C to stop",
              file=sys.stderr, flush=True)
        app.start_oracle_probe()
        if settings.open_browser:
            webbrowser.open(url)
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            print("\nlattix ui: stopping", file=sys.stderr)
        return 0
    finally:
        srv.shutdown()
        srv.server_close()
        app.close()
