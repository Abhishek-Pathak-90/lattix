"""Workbench plugins: an installed package registers extra routes under ``/plugins/<name>/`` and
tabs that the page mounts as same-origin iframes, served by the same loopback server with the
same token and session.

Discovery is the ``lattix.ui.plugins`` entry-point group.  An entry point names a callable that
returns a :class:`UiPlugin` (or the object itself).  Route handlers have the shape of the core
routes, ``fn(handler, query, *groups)``, and use the same helpers: ``handler._session(sid)``,
``handler._json(status, obj)``, ``handler._send(status, body, ctype, extra, cache=)``,
``handler._json_body()``, ``handler.app.resolve_path(user_path)``; a ``Session`` offers
``plugin_state`` (key it by the plugin name) and its ``source`` / ``translations``.  The page
side is described in ``docs/ui.md`` ("Plugins")."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "lattix.ui.plugins"
_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class UiTab:
    """A tab of the workbench: ``page`` is a path under the plugin's prefix that one of its routes
    serves as HTML; the page is loaded in an iframe when the tab is first opened."""
    id: str
    label: str
    page: str = "/"
    order: int = 100


@dataclass(frozen=True)
class UiPlugin:
    name: str                           # the URL segment /plugins/<name>/ — use the import package name
    version: str = ""
    routes: list[tuple[str, str, Callable]] = field(default_factory=list)   # (method, regex under the prefix, fn)
    tabs: list[UiTab] = field(default_factory=list)

    @property
    def prefix(self) -> str:
        return f"/plugins/{self.name}"


@dataclass
class LoadedPlugin:
    plugin: UiPlugin
    routes: list[tuple[str, re.Pattern, Callable]]      # compiled with the prefix, method upper-cased


def compile_plugin(plugin: UiPlugin) -> LoadedPlugin:
    """Validate a plugin and compile its routes under ``/plugins/<name>``."""
    if not isinstance(plugin, UiPlugin):
        raise TypeError(f"a plugin must be a UiPlugin, not {type(plugin).__name__}")
    if not _NAME.fullmatch(plugin.name):
        raise ValueError(f"plugin name {plugin.name!r} must match [a-z][a-z0-9_]*")
    for t in plugin.tabs:
        if not _NAME.fullmatch(t.id):
            raise ValueError(f"tab id {t.id!r} of plugin {plugin.name!r} must match [a-z][a-z0-9_]*")
        if not t.page.startswith("/"):
            raise ValueError(f"tab page {t.page!r} of plugin {plugin.name!r} must start with /")
    routes = []
    for method, pattern, fn in plugin.routes:
        if not callable(fn):
            raise TypeError(f"route {method} {pattern!r} of plugin {plugin.name!r} has no callable")
        routes.append((str(method).upper(), re.compile(re.escape(plugin.prefix) + pattern), fn))
    return LoadedPlugin(plugin, routes)


def load_plugins(*, group: str = ENTRY_POINT_GROUP, errors: list[str] | None = None) -> list[LoadedPlugin]:
    """Every installed plugin, sorted by name.  A broken plugin is reported in ``errors`` and
    skipped; it never takes the workbench down."""
    out: list[LoadedPlugin] = []
    seen: set[str] = set()
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
            plugin = obj if isinstance(obj, UiPlugin) else obj()
            loaded = compile_plugin(plugin)
            if loaded.plugin.name in seen:
                raise ValueError(f"a plugin named {loaded.plugin.name!r} is already loaded")
            seen.add(loaded.plugin.name)
            out.append(loaded)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not stop the others
            if errors is not None:
                errors.append(f"plugin {ep.name!r} ({ep.value}): {type(exc).__name__}: {exc}")
    return sorted(out, key=lambda p: p.plugin.name)


def plugins_view(loaded: list[LoadedPlugin], errors: list[str]) -> dict:
    """What ``GET /api/plugins`` returns: the tabs with their page URLs, and the load errors."""
    return {
        "plugins": [{"name": p.plugin.name, "version": p.plugin.version,
                     "tabs": [{"id": t.id, "label": t.label, "page": p.plugin.prefix + t.page, "order": t.order}
                              for t in p.plugin.tabs]}
                    for p in loaded],
        "errors": list(errors),
    }
