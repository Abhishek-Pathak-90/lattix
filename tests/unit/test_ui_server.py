"""The UI server: security, every route, downloads, translation payloads, validation jobs (fake engines)."""
from __future__ import annotations

import http.client
import io
import json
import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from lattix.oracles.base import SPECIES, Basis, BeamSpec, OracleResult
from lattix.oracles.basis import drift_common
from lattix.ui.server import App, Settings, start_in_thread

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
MP = SPECIES["proton"][0]


def _fake_runner(job, spec):
    """An in-thread stand-in for the `lattix validate` subprocess: two fake engines, one comparison."""
    from lattix.oracles.validate import outcome_to_dict, run_validation

    class _A:
        name, formats = "srv_fake_a", ("tracewin", "elegant")

        def available(self):
            return True, "fake"

        def run(self, deck, *, fmt=None, beam=None, probe=None, workdir=None):
            d = drift_common(0.5, 2.1e6, MP)
            return OracleResult(engine=self.name, basis=Basis.COMMON, names=["d0", "d1"], length=np.array([0.5, 0.5]),
                                s_out=np.array([0.5, 1.0]), R_elem=np.array([d, d]),
                                ref_kinetic_eV_in=np.array([2.1e6, 2.1e6]), ref_kinetic_eV_out=np.array([2.1e6, 2.1e6]),
                                mass_eV=MP, charge=1)

    class _B(_A):
        name = "srv_fake_b"

    from lattix.oracles.base import register
    register(_A)
    register(_B)
    job.say("fake engines")
    o = run_validation({f: Path(p) for f, p in spec["decks"].items()}, ["srv_fake_a", "srv_fake_b"],
                       BeamSpec(), log=job.say, progress=job.set_progress, should_stop=job.stopped)
    return outcome_to_dict(o, decks={f: Path(p) for f, p in spec["decks"].items()}, beam=BeamSpec())


@pytest.fixture(scope="module")
def server():
    settings = Settings(root=DATA, open_browser=False, max_body=4096 * 64, quiet=True)
    app = App(settings, oracle_probe=lambda: {"srv_fake_a": (True, "fake", ["tracewin"])},
              validation_runner=_fake_runner)
    srv, thread, port = start_in_thread(app, "127.0.0.1", 0)
    yield {"app": app, "port": port, "token": settings.token, "settings": settings}
    srv.shutdown()
    srv.server_close()
    app.close()


def api(server, method, path, body=None, *, token=True, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", server["port"], timeout=60)
    hdrs = dict(headers or {})
    if token:
        hdrs["X-Lattix-Token"] = server["token"]
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    payload = resp.read()
    ctype = resp.getheader("Content-Type") or ""
    out = json.loads(payload.decode()) if ctype.startswith("application/json") else payload
    conn.close()
    return resp.status, out, resp


def test_security(server):
    assert api(server, "GET", "/api/ping", token=False)[0] == 403
    status, body, _ = api(server, "GET", "/api/ping", token=False, headers={"X-Lattix-Token": "wrong"})
    assert status == 403 and body["error"]["type"] == "forbidden"
    assert api(server, "GET", "/api/ping", headers={"Host": "evil.example:1"})[0] == 403
    assert api(server, "POST", "/api/session", body={}, headers={"Origin": "http://evil.example"})[0] == 403
    big = b"x" * (server["settings"].max_body + 1)
    status, body, _ = api(server, "POST", "/api/read", raw=big, headers={"Content-Type": "application/json"})
    assert status == 413
    status, body, _ = api(server, "GET", "/api/nonsense")
    assert status == 404 and body["error"]["type"] == "not_found"
    status, body, _ = api(server, "GET", f"/api/ping?token={server['token']}", token=False)
    assert status == 200 and body["ok"] is True and body["version"]


def test_page_is_offline(server):
    status, body, resp = api(server, "GET", "/")
    text = body.decode()
    assert status == 200 and resp.getheader("Content-Type").startswith("text/html")
    assert "<script" in text and "<script src" not in text and 'rel="stylesheet"' not in text
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", text))
    assert urls <= {"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"}
    assert "Content-Security-Policy" in dict(resp.getheaders())


def test_catalogues(server):
    status, f, _ = api(server, "GET", "/api/formats")
    assert status == 200 and f["formats"]["madx"]["write_options"] and "opal" in f["writable"]
    status, s, _ = api(server, "GET", "/api/samples")
    assert any(d["path"] == "helix/fodo_cell.dat" for d in s["decks"])
    status, b, _ = api(server, "GET", "/api/browse?path=helix")
    assert status == 200 and any(e["name"] == "fodo_cell.dat" and e["format"] == "tracewin" for e in b["entries"])
    assert api(server, "GET", "/api/browse?path=../..")[0] == 403
    status, c, _ = api(server, "GET", "/api/catalog")
    assert len(c) > 500 and "FM_TO_CAVITY" in c
    status, o, _ = api(server, "GET", "/api/oracles")
    for _ in range(50):
        if o["ready"]:
            break
        time.sleep(0.05)
        status, o, _ = api(server, "GET", "/api/oracles")
    assert o["ready"] and o["engines"]["srv_fake_a"]["available"]


def test_read_translate_download_and_session(server):
    status, r, _ = api(server, "POST", "/api/read", body={"source": {"sample": "helix/mebt_line.dat"}})
    assert status == 201 and r["detected_format"] == "tracewin" and r["source"]["header"]["n_placed"] == 29
    sid = r["session"]
    status, t, _ = api(server, "POST", "/api/translate", body={"session": sid, "format": "elegant"})
    assert status == 200 and t["translation"] == "t1" and t["deck"]["lines"] > 10
    assert t["target"]["header"]["format"] == "elegant" and t["comparison"]["alignment"]["counts"]["name"] >= 20
    assert t["fidelity"]["counts"] and t["comparison"]["ir"]["ok"] is True
    status, t2, _ = api(server, "POST", "/api/translate", body={"session": sid, "format": "impactt"})
    assert status == 200 and any(f["name"].startswith("rfdata") for f in t2["side_files"])
    status, z, resp = api(server, "GET", f"/api/session/{sid}/translation/t2/download?what=zip")
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert "ImpactT.in" in names and "fidelity.json" in names and any(n.startswith("rfdata") for n in names)
    status, d, resp = api(server, "GET", f"/api/session/{sid}/translation/t1/download?what=deck")
    assert d.decode("latin-1") == t["deck"]["text"] and "attachment" in resp.getheader("Content-Disposition")
    status, fid, _ = api(server, "GET", f"/api/session/{sid}/translation/t1/download?what=fidelity")
    assert fid["entries"]                                    # served as application/json
    status, e, _ = api(server, "POST", "/api/translate", body={"session": sid, "format": "madx", "strict": True})
    assert status == 422 and e["error"]["type"] == "translation" and e["error"]["entry"]["code"]
    status, t3, _ = api(server, "POST", "/api/translate", body={"session": sid, "format": "elegant", "reread": False})
    assert status == 200 and t3["target"] is None and t3["comparison"] is None
    status, s, _ = api(server, "GET", f"/api/session/{sid}")
    assert s["source"]["header"]["n_placed"] == 29 and set(s["translations"]) >= {"t1", "t2"}
    status, txt, _ = api(server, "GET", f"/api/session/{sid}/source_text")
    assert "QUAD" in txt["text"].upper()
    status, again, _ = api(server, "GET", f"/api/session/{sid}/translation/t1")
    assert again["deck"]["text"] == t["deck"]["text"]
    assert api(server, "DELETE", f"/api/session/{sid}")[1]["dropped"] is True
    assert api(server, "GET", f"/api/session/{sid}")[0] == 404


def test_read_by_path_content_upload_and_errors(server):
    status, r, _ = api(server, "POST", "/api/read", body={"source": {"path": "helix/fodo_cell.dat"}})
    assert status == 201 and r["source"]["header"]["counts"]["Quadrupole"] == 8
    text = (DATA / "helix" / "fodo_cell.dat").read_text()
    status, r2, _ = api(server, "POST", "/api/read", body={"source": {"content": text, "filename": "x.dat"},
                                                            "options": {"species": "h-", "line": "ignored"}})
    assert status == 201 and r2["source"]["header"]["reference"]["species"]["name"] == "h-"
    assert any("no line" in w for w in r2["source"]["header"]["warnings"])
    status, s, _ = api(server, "POST", "/api/session", body={})
    sid = s["session"]
    status, u, _ = api(server, "PUT", f"/api/session/{sid}/upload?path=maps/deck.dat", raw=text.encode("latin-1"))
    assert status == 201 and u["path"] == "maps/deck.dat"
    status, r3, _ = api(server, "POST", "/api/read", body={"session": sid, "source": {"path": "maps/deck.dat"}})
    assert status == 201 and r3["session"] == sid
    assert api(server, "POST", "/api/read", body={"source": {"path": "nope.dat"}})[0] == 404
    body = {"source": {"sample": "helix/fodo_cell.dat"}, "format": "zzz"}
    assert api(server, "POST", "/api/read", body=body)[0] == 400
    assert api(server, "POST", "/api/read", body={"source": {}})[0] == 400
    assert api(server, "POST", "/api/read", raw=b"{not json", headers={"Content-Type": "application/json"})[0] == 400


def test_validation_job(server):
    status, r, _ = api(server, "POST", "/api/read", body={"source": {"sample": "helix/fodo_cell.dat"}})
    sid = r["session"]
    api(server, "POST", "/api/translate", body={"session": sid, "format": "elegant"})
    status, j, _ = api(server, "POST", "/api/validate", body={"session": sid, "translation": "t1",
                                                              "engines": ["srv_fake_a", "srv_fake_b"]})
    assert status == 202
    jid = j["job"]
    for _ in range(200):
        status, job, _ = api(server, "GET", f"/api/jobs/{jid}")
        if job["state"] in ("done", "failed", "cancelled", "timeout"):
            break
        time.sleep(0.05)
    assert job["state"] == "done", job
    res = job["result"]
    assert res["comparisons"] and res["comparisons"][0]["per_boundary"] and "row" in res["comparisons"][0]
    assert res["engines"]["srv_fake_a"]["n"] == 2 and job["meta"]["engines"] == ["srv_fake_a", "srv_fake_b"]
    assert api(server, "GET", "/api/jobs/nope")[0] == 404
    status, c, _ = api(server, "POST", f"/api/jobs/{jid}/cancel", body={})
    assert status == 200 and c["cancelled"] is False
    assert api(server, "POST", "/api/validate", body={"session": sid, "engines": ["only_one"]})[0] == 400


def test_serve_check_and_refused_host(capsys):
    from lattix.cli import main

    assert main(["ui", "--no-browser", "--port", "0", "--check", "--root", str(DATA)]) == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:" in out and "token=" in out and "self-test passed" in out
    assert main(["ui", "--host", "0.0.0.0", "--check", "--no-browser"]) == 2


def test_threads_do_not_leak(server):
    before = threading.active_count()
    for _ in range(3):
        api(server, "GET", "/api/ping")
    assert threading.active_count() <= before + 3
