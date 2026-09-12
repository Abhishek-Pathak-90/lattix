"""A workbench plugin for the tests: a page, a JSON route that reads the session, a raising route."""
from __future__ import annotations

from lattix.ui.plugins import UiPlugin, UiTab
from lattix.ui.server import ApiError

PAGE = b"<!doctype html><title>fake plugin</title><p>hello from the fake plugin"
CSP = "default-src 'none'; frame-ancestors 'self'"


def r_page(handler, query) -> None:
    handler._send(200, PAGE, "text/html; charset=utf-8", {"Content-Security-Policy": CSP})


def r_hello(handler, query) -> None:
    sid = (query.get("session") or [""])[0]
    n = None
    if sid:
        s = handler._session(sid)
        s.plugin_state.setdefault("fake", {})["hits"] = s.plugin_state.get("fake", {}).get("hits", 0) + 1
        n = len(s.source.placed) if s.source else 0
    handler._json(200, {"hello": "world", "elements": n, "hits": s.plugin_state["fake"]["hits"] if sid else None})


def r_static(handler, query, name: str) -> None:
    handler._send(200, b"body{}", "text/css", cache="private, max-age=60")


def r_boom(handler, query) -> None:
    raise ApiError(418, "teapot", "the fake plugin refuses")


def register() -> UiPlugin:
    return UiPlugin(name="fake", version="0.0.1",
                    routes=[("GET", r"/", r_page), ("get", r"/api/hello", r_hello),
                            ("GET", r"/static/([\w.]+)", r_static, True), ("GET", r"/api/boom", r_boom)],
                    tabs=[UiTab("fake", "Fake", "/", order=5)])
