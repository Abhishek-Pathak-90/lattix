"""Static catalogue of every fidelity code the package can emit (PLAN §4.4).

The ledger API lives in :mod:`lattix.fidelity`; the codes themselves are scattered over the
readers and writers.  This module recovers them by parsing ``lattix/**/*.py`` — no import, no
execution — so ``docs/fidelity.md`` can be generated from the source and a docs-consistency
test can fail when the two drift apart (``tests/docs/test_docs_consistency.py``).

Call shapes that carry a code:

=================================================  =========================================
shape                                              what is read
=================================================  =========================================
``rep.exact(element, kind, code, message)``        ``EXACT`` + ``code`` (default ``"OK"``)
``rep.equivalent("CODE", "message", …)``           ``EQUIVALENT`` + first two positionals
``rep.lossy("CODE", …)`` / ``rep.dropped(…)``      ``LOSSY`` / ``DROPPED``
``rep.add(FidelityClass.X, "CODE", "message")``    the named class
``Rule(target, cls, code, message)``               the writers' ``RULES`` capability tables,
                                                   replayed as ``rep.add(rule.cls, rule.code…)``
per-module wrappers                                any function that forwards its own
                                                   parameters into one of the above
                                                   (``tracewin/reader.py::_entry``,
                                                   ``tracewin/writer.py::comment_drop``)
=================================================  =========================================

String arguments are resolved through literals, ``"A" if c else "B"`` conditionals and simple
local ``name = "LITERAL"`` assignments in the enclosing function, which covers every
indirection used today (e.g. ``madx/writer.py`` choosing ``CONST_P0_LOCAL_RIGIDITY`` vs
``CONST_P0_START_RIGIDITY``).  A ``rep.lossy(rule.code, …)`` replay of a ``RULES`` row resolves
through the ``Rule(...)`` constructor instead.

Usage::

    python -m lattix.fidelity_catalog                # text listing
    python -m lattix.fidelity_catalog --markdown     # the tables in docs/fidelity.md
    python -m lattix.fidelity_catalog --json         # machine-readable
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

CLASSES = ("EXACT", "EQUIVALENT", "LOSSY", "DROPPED")

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class _Sig:
    """Where a recording call keeps its fidelity class, code and message."""

    cls: str | None = None            # fixed class …
    cls_arg: int | None = None        # … or the positional index carrying it,
    cls_kw: str = "cls"
    cls_default: str | None = None    # … with this default when the caller omits it
    code_arg: int = 0
    code_kw: str = "code"
    code_default: str | None = None
    msg_arg: int | None = 1
    msg_kw: str = "message"


#: The :class:`lattix.fidelity.FidelityReport` API itself.
_BUILTINS: dict[str, _Sig] = {
    "exact": _Sig(cls="EXACT", code_arg=2, code_default="OK", msg_arg=3),
    "equivalent": _Sig(cls="EQUIVALENT"),
    "lossy": _Sig(cls="LOSSY"),
    "dropped": _Sig(cls="DROPPED"),
    "add": _Sig(cls_arg=0, code_arg=1, msg_arg=2),
    # the writers' capability tables: Rule(target, cls="EXACT", code="OK", message="")
    "Rule": _Sig(cls_arg=1, cls_default="EXACT", code_arg=2, code_default="OK", msg_arg=3),
    # a ledger entry built directly, by keyword: FidelityEntry(element=…, kind=…, cls=…, code=…, message=…)
    "FidelityEntry": _Sig(cls_arg=None, code_arg=None, msg_arg=None),
}


# ---------------------------------------------------------------------------
def _strings(node: ast.AST | None, scope: dict[str, list[str]]) -> list[str]:
    """Every string literal ``node`` can evaluate to (empty when it is not resolvable)."""
    if node is None:
        return []
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    if isinstance(node, ast.IfExp):
        return _strings(node.body, scope) + _strings(node.orelse, scope)
    if isinstance(node, ast.JoinedStr):          # f-string: keep the literal part
        parts = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return ["".join(parts)] if parts else []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [a + b for a in _strings(node.left, scope) for b in _strings(node.right, scope)]
    if isinstance(node, ast.Name):
        return list(scope.get(node.id, ()))
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "FidelityClass":      # FidelityClass.LOSSY → "LOSSY"
            return [node.attr]
    return []


def _first(values: list[str], default: str = "") -> str:
    return values[0] if values else default


def _node_at(call: ast.Call, index: int | None, keyword: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    if index is not None and index < len(call.args):
        return call.args[index]
    return None


def _arg(call: ast.Call, index: int | None, keyword: str, scope: dict[str, list[str]]) -> list[str]:
    return _strings(_node_at(call, index, keyword), scope)


def _local_strings(fn: ast.AST) -> dict[str, list[str]]:
    """``name -> possible string values`` for plain assignments inside one function body."""
    scope: dict[str, list[str]] = defaultdict(list)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            values = _strings(node.value, {})
            for target in node.targets if values else ():
                if isinstance(target, ast.Name):
                    scope[target.id].extend(values)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            scope[node.target.id].extend(_strings(node.value, {}))
    return dict(scope)


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


# ---------------------------------------------------------------------------
def _parameters(fn: ast.FunctionDef) -> tuple[list[str], dict[str, str]]:
    """Positional parameter names as a *caller* sees them (``self`` removed) + string defaults."""
    args = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    defaults: dict[str, str] = {}
    for name, node in zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults, strict=True):
        defaults[name] = _first(_strings(node, {}))
    for name, node in zip([a.arg for a in fn.args.kwonlyargs], fn.args.kw_defaults, strict=True):
        defaults[name] = _first(_strings(node, {}))
    if args and args[0] in ("self", "cls"):
        args = args[1:]
    return args, defaults


def _wrappers(tree: ast.AST, known: dict[str, _Sig]) -> dict[str, _Sig]:
    """Functions that forward their own parameters into a recording call.

    ``_entry(self, cls, code, msg, **details)`` calling ``self.report.add(cls, code, msg, …)``
    becomes a signature with ``cls_arg=0, code_arg=1, msg_arg=2`` under the wrapper's own
    parameter names, so its call sites are read like a native ``add``.
    """
    found: dict[str, _Sig] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        params, defaults = _parameters(fn)
        index = {name: i for i, name in enumerate(params)}
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            sig = known.get(_call_name(call))
            if sig is None:
                continue
            code_node = _node_at(call, sig.code_arg, sig.code_kw)
            if not (isinstance(code_node, ast.Name) and code_node.id in index):
                continue
            code = code_node.id
            msg_node = _node_at(call, sig.msg_arg, sig.msg_kw)
            msg = msg_node.id if isinstance(msg_node, ast.Name) and msg_node.id in index else None
            cls_node = _node_at(call, sig.cls_arg, sig.cls_kw) if sig.cls is None else None
            if isinstance(cls_node, ast.Name) and cls_node.id in index:
                new = replace(sig, cls=None, cls_arg=index[cls_node.id], cls_kw=cls_node.id,
                              cls_default=defaults.get(cls_node.id) or sig.cls_default)
            elif sig.cls is not None:
                new = replace(sig, cls=sig.cls, cls_arg=None)
            else:
                continue
            found[fn.name] = replace(new, code_arg=index[code], code_kw=code, code_default=None,
                                     msg_arg=index[msg] if msg else None, msg_kw=msg or "message")
            break
    return found


class _Visitor(ast.NodeVisitor):
    def __init__(self, signatures: dict[str, _Sig]) -> None:
        self.signatures = signatures
        self.hits: list[tuple[str, str, str, int]] = []   # (cls, code, message, line)
        self._scopes: list[dict[str, list[str]]] = [{}]

    def _enter_function(self, node) -> None:
        merged = dict(self._scopes[-1])
        merged.update(_local_strings(node))
        self._scopes.append(merged)
        self.generic_visit(node)
        self._scopes.pop()

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    def visit_Call(self, node: ast.Call) -> None:
        sig = self.signatures.get(_call_name(node))
        if sig is not None:
            scope = self._scopes[-1]
            classes = [sig.cls] if sig.cls else (
                _arg(node, sig.cls_arg, sig.cls_kw, scope) or ([sig.cls_default] if sig.cls_default else []))
            codes = _arg(node, sig.code_arg, sig.code_kw, scope) or (
                [sig.code_default] if sig.code_default else [])
            message = _first(_arg(node, sig.msg_arg, sig.msg_kw, scope))
            for cls in classes:
                for code in codes:
                    if cls in CLASSES and code:
                        self.hits.append((cls, code, message, node.lineno))
        self.generic_visit(node)


# ---------------------------------------------------------------------------
def _package_of(relpath: str) -> str:
    """``formats/madx/writer.py`` → ``formats/madx``; ``fidelity.py`` → ``core``."""
    parts = Path(relpath).parts[:-1]
    return "/".join(parts) if parts else "core"


def scan(root: Path | str | None = None) -> dict[str, dict]:
    """``{code: {"class": …, "where": ["formats/madx/writer.py:150", …], "message": …}}``.

    ``class`` is the strongest class the code is ever recorded with
    (DROPPED > LOSSY > EQUIVALENT > EXACT); ``message`` is the first non-empty message literal
    found for it; ``where`` lists every source site, and ``packages`` every package, in the
    order the files are scanned.
    """
    root = Path(root) if root is not None else PACKAGE_ROOT
    rank = {c: i for i, c in enumerate(CLASSES)}
    out: dict[str, dict] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:                                     # pragma: no cover - source is ours
            continue
        signatures = dict(_BUILTINS)
        signatures.update(_wrappers(tree, signatures))
        visitor = _Visitor(signatures)
        visitor.visit(tree)
        for cls, code, message, line in visitor.hits:
            entry = out.setdefault(code, {"class": cls, "where": [], "message": "", "packages": []})
            if rank[cls] > rank[entry["class"]]:
                entry["class"] = cls
            entry["where"].append(f"{rel}:{line}")
            if message and not entry["message"]:
                entry["message"] = message
            pkg = _package_of(rel)
            if pkg not in entry["packages"]:
                entry["packages"].append(pkg)
    return dict(sorted(out.items()))


def by_package(catalog: dict[str, dict] | None = None) -> dict[str, list[str]]:
    """``{package: [code, …]}`` grouped by the package of each code's first source site."""
    catalog = scan() if catalog is None else catalog
    groups: dict[str, list[str]] = defaultdict(list)
    for code, entry in catalog.items():
        groups[entry["packages"][0]].append(code)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def counts(catalog: dict[str, dict] | None = None) -> dict[str, int]:
    catalog = scan() if catalog is None else catalog
    out = dict.fromkeys(CLASSES, 0)
    for entry in catalog.values():
        out[entry["class"]] += 1
    return out


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def to_markdown(catalog: dict[str, dict] | None = None) -> str:
    """The catalogue tables of ``docs/fidelity.md``, grouped by package."""
    catalog = scan() if catalog is None else catalog
    n = counts(catalog)
    lines = ["<!-- generated by `python -m lattix.fidelity_catalog --markdown`; "
             f"{len(catalog)} codes: " + ", ".join(f"{k} {n[k]}" for k in CLASSES) + " -->", ""]
    for package, codes in by_package(catalog).items():
        lines += [f"### `lattix/{package}`", "", "| code | class | meaning | source |", "|---|---|---|---|"]
        for code in codes:
            entry = catalog[code]
            where = entry["where"][0].rsplit("/", 1)[-1].split(":")[0]   # file only: line numbers churn
            extra = f" +{len(entry['where']) - 1}" if len(entry["where"]) > 1 else ""
            lines.append(f"| `{code}` | {entry['class']} | {_cell(entry['message']) or '—'} "
                         f"| `{where}`{extra} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lattix.fidelity_catalog",
                                description="catalogue the fidelity codes in the lattix source")
    p.add_argument("--markdown", action="store_true", help="emit the docs/fidelity.md tables")
    p.add_argument("--json", action="store_true", help="emit the catalogue as JSON")
    p.add_argument("--root", default=None, help="package directory to scan (default: lattix/)")
    a = p.parse_args(argv)

    catalog = scan(a.root)
    if a.markdown:
        print(to_markdown(catalog))
    elif a.json:
        print(json.dumps(catalog, indent=1))
    else:
        n = counts(catalog)
        print(f"{len(catalog)} fidelity codes: " + " ".join(f"{k}={n[k]}" for k in CLASSES))
        for package, codes in by_package(catalog).items():
            print(f"\n{package}")
            for code in codes:
                print(f"  {catalog[code]['class']:<10} {code:<34} {catalog[code]['where'][0]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
