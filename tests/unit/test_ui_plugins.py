"""Workbench plugins: discovery through entry points, routes under /plugins/<name>/ behind the same
token, the tab listing the page mounts, error isolation, and the switch that turns them off."""
from __future__ import annotations

import http.client
import json
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from lattix.ui import plugins as plugins_mod
from lattix.ui.plugins import UiPlugin, UiTab, compile_plugin, load_plugins, plugins_view
from lattix.ui.server import App, Settings, start_in_thread
from tests.unit._fake_plugin import register

DATA = Path(__file__).resolve().parents[1] / "data" / "public"


@pytest.fixture(scope="module")
def server():
    settings = Settings(root=DATA, open_browser=False, quiet=True)
    app = App(settings, oracle_probe=lambda: {}, plugins=[compile_plugin(register())])
    srv, thread, port = start_in_thread(app, "127.0.0.1", 0)
    yield {"app": app, "port": port, "token": settings.token}
    srv.shutdown()
    srv.server_close()
    app.close()


def api(server, method: str, path: str, body=None, *, token: str | None = "ok"):
    conn = http.client.HTTPConnection("127.0.0.1", server["port"], timeout=30)
    headers = {}
    if token is not None:
        headers["X-Lattix-Token"] = server["token"] if token == "ok" else token
    raw = None
    if body is not None:
        raw, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    conn.request(method, path, body=raw, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    ctype = resp.getheader("Content-Type") or ""
    out = json.loads(data) if ctype.startswith("application/json") else data
    conn.close()
    return resp.status, out, resp


def test_plugin_is_listed_and_its_routes_are_token_gated(server):
    status, ping, _ = api(server, "GET", "/api/ping")
    assert status == 200 and ping["plugins"] == ["fake"]
    status, listing, _ = api(server, "GET", "/api/plugins")
    assert status == 200 and listing["errors"] == []
    assert listing["plugins"] == [{"name": "fake", "version": "0.0.1",
                                   "tabs": [{"id": "fake", "label": "Fake", "page": "/plugins/fake/", "order": 5}]}]
    status, page, resp = api(server, "GET", "/plugins/fake/")
    assert status == 200 and b"hello from the fake plugin" in page
    assert resp.getheader("Content-Security-Policy") == "default-src 'none'; frame-ancestors 'self'"
    assert api(server, "GET", "/plugins/fake/api/hello")[1] == {"hello": "world", "elements": None, "hits": None}
    assert api(server, "GET", "/plugins/fake/api/hello", token=None)[0] == 403
    assert api(server, "GET", "/plugins/fake/api/hello", token="wrong")[0] == 403
    status, page, resp = api(server, "GET", "/plugins/fake/", token=None)      # the friendly page, like /
    assert status == 403 and b"needs its access link" in page
    status, css, resp = api(server, "GET", "/plugins/fake/static/app.css")
    assert status == 200 and css == b"body{}" and resp.getheader("Cache-Control") == "private, max-age=60"
    # a public route: the browser fetches scripts without headers, so no token is needed there and only there
    assert api(server, "GET", "/plugins/fake/static/app.css", token=None)[0] == 200
    assert api(server, "GET", "/plugins/fake/static/app.css", token="wrong")[0] == 200
    assert api(server, "GET", "/plugins/fake/api/hello", token=None)[0] == 403
    assert api(server, "POST", "/plugins/fake/static/app.css", token=None)[0] == 403
    assert api(server, "GET", "/plugins/fake/nothing")[0] == 404
    assert api(server, "GET", "/plugins/other/api/hello")[0] == 404
    status, err, _ = api(server, "GET", "/plugins/fake/api/boom")
    assert status == 418 and err["error"]["type"] == "teapot"


def test_plugin_route_sees_the_session(server):
    status, r, _ = api(server, "POST", "/api/read", body={"source": {"sample": "helix/fodo_cell.dat"}})
    sid = r["session"]
    status, h, _ = api(server, "GET", f"/plugins/fake/api/hello?session={sid}")
    assert status == 200 and h["elements"] == r["source"]["header"]["n_placed"] and h["hits"] == 1
    assert api(server, "GET", f"/plugins/fake/api/hello?session={sid}")[1]["hits"] == 2
    assert server["app"].sessions.get(sid).plugin_state == {"fake": {"hits": 2}}
    assert api(server, "GET", "/plugins/fake/api/hello?session=nope")[0] == 404


def test_page_csp_allows_same_origin_frames(server):
    status, page, resp = api(server, "GET", "/")
    assert status == 200 and "frame-src 'self'" in resp.getheader("Content-Security-Policy")


def test_compile_plugin_validates():
    with pytest.raises(TypeError):
        compile_plugin(object())
    with pytest.raises(ValueError):
        compile_plugin(UiPlugin(name="Bad-Name"))
    with pytest.raises(ValueError):
        compile_plugin(UiPlugin(name="ok", tabs=[UiTab("Tab", "x")]))
    with pytest.raises(ValueError):
        compile_plugin(UiPlugin(name="ok", tabs=[UiTab("tab", "x", page="index.html")]))
    with pytest.raises(TypeError):
        compile_plugin(UiPlugin(name="ok", routes=[("GET", "/x", "not callable")]))
    with pytest.raises(ValueError):
        compile_plugin(UiPlugin(name="ok", routes=[("POST", "/x", lambda h, q: None, True)]))
    with pytest.raises(TypeError):
        compile_plugin(UiPlugin(name="ok", routes=[("GET", "/x")]))
    pub = compile_plugin(UiPlugin(name="ok", routes=[("GET", r"/s/(.+)", lambda h, q, r: None, True)]))
    assert pub.public and pub.public[0].fullmatch("/plugins/ok/s/a.js")
    loaded = compile_plugin(UiPlugin(name="ok", routes=[("get", r"/x/(\d+)", lambda h, q, n: None)]))
    assert loaded.routes[0][0] == "GET" and loaded.routes[0][1].fullmatch("/plugins/ok/x/12")
    assert not loaded.routes[0][1].fullmatch("/plugins/okx/12") and loaded.plugin.prefix == "/plugins/ok"
    assert plugins_view([loaded], ["oops"]) == {"plugins": [{"name": "ok", "version": "", "tabs": []}],
                                                "errors": ["oops"]}


def test_load_plugins_from_entry_points_isolates_failures(monkeypatch):
    eps = [EntryPoint("fake", "tests.unit._fake_plugin:register", "lattix.ui.plugins"),
           EntryPoint("broken", "tests.unit._broken_plugin:register", "lattix.ui.plugins"),
           EntryPoint("missing", "tests.unit._no_such_module:register", "lattix.ui.plugins"),
           EntryPoint("twice", "tests.unit._fake_plugin:register", "lattix.ui.plugins")]
    monkeypatch.setattr(plugins_mod, "entry_points", lambda group: [e for e in eps if e.group == group])
    errors: list[str] = []
    loaded = load_plugins(errors=errors)
    assert [p.plugin.name for p in loaded] == ["fake"]
    assert len(errors) == 3
    assert any("broken on purpose" in e and "'broken'" in e for e in errors)
    assert any("'missing'" in e for e in errors)
    assert any("already loaded" in e and "'twice'" in e for e in errors)
    # the App reports the same errors on /api/plugins, and --no-plugins loads none
    app = App(Settings(root=DATA, open_browser=False), oracle_probe=lambda: {})
    try:
        assert [p.plugin.name for p in app.plugins] == ["fake"] and len(app.plugin_errors) == 3
    finally:
        app.close()
    app = App(Settings(root=DATA, open_browser=False, plugins=False), oracle_probe=lambda: {})
    try:
        assert app.plugins == [] and app.plugin_errors == []
    finally:
        app.close()
