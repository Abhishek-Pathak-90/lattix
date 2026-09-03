"""Safe expression evaluation for lattice files.

Three dialects share one whitelist (ported from HELIX ``madx_parser._eval_node``):

* infix (MAD-X, MAD8, Bmad): ``+ - * / ^`` (``**``), unary sign, ``sqrt abs sin cos tan
  asin acos atan atan2 exp log log10 floor ceil``, constants ``pi twopi e clight emass
  pmass true false``, names resolved through *variables* or a *resolver* callback
  (MAD8 ``NAME[ATTR]`` references, lazily);
* RPN (Elegant): postfix with ``+ - * / ^ sqrt sin cos tan asin acos atan exp ln log
  chs abs`` and ``pi``;
* :class:`Expression` records the source text and whether it was deferred (``:=``) so
  writers for formats with variables can re-emit the symbol instead of the number.
"""
from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict

CONSTANTS: dict[str, float] = {
    "pi": math.pi, "twopi": 2 * math.pi, "e": math.e, "clight": 299_792_458.0,
    "emass": 0.51099895e-3, "pmass": 0.93827208816, "true": 1.0, "false": 0.0,
    "degrad": 180.0 / math.pi, "raddeg": math.pi / 180.0,
    # PALS constant names (fundamentals.md)
    "c_light": 299_792_458.0, "e_charge": 1.602_176_634e-19, "h_planck": 6.626_070_15e-34,
    "hbar": 1.054_571_817e-34, "k_boltzmann": 1.380_649e-23, "mu_0": 1.256_637_062_12e-6,
    "epsilon_0": 8.854_187_812_8e-12, "r_electron": 2.817_940_326_2e-15, "r_proton": 1.534_698_2e-18,
    "fine_structure": 7.297_352_569_3e-3, "n_avogadro": 6.022_140_76e23,
}
FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "abs": abs, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "exp": math.exp, "log": math.log, "log10": math.log10, "floor": math.floor,
    "ceil": math.ceil, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    # PALS function names
    "cot": lambda x: 1.0 / math.tan(x), "sinc": lambda x: 1.0 if x == 0 else math.sin(x) / x,
    "factorial": lambda n: float(math.factorial(int(round(n)))), "nint": lambda x: float(round(x)),
    "sign": lambda x: float((x > 0) - (x < 0)), "ceiling": math.ceil, "modulo": math.fmod,
    "max": max, "min": min,
}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

Resolver = Callable[[str], float]


class ExpressionError(ValueError):
    pass


class Expression(BaseModel):
    """Source form of a numeric parameter (kept next to its evaluated float)."""

    model_config = ConfigDict(extra="forbid")
    text: str
    deferred: bool = False
    dialect: str = "infix"


def evaluate(text: str, variables: Mapping[str, float] | None = None,
             resolver: Resolver | None = None) -> float:
    """Evaluate an infix expression with the whitelist above.  ``^`` means power."""
    src = text.strip().replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse {text!r}: {e.msg}") from None
    return _eval_node(tree.body, variables or {}, resolver)


def _eval_node(node: ast.AST, variables: Mapping[str, float], resolver: Resolver | None) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in variables:
            return float(variables[key])
        if node.id in variables:
            return float(variables[node.id])
        if resolver is not None:
            # a deck's own parameter (MAD8 `E :=`, `PI :=`) beats the built-in constant
            try:
                return float(resolver(node.id))
            except ExpressionError:
                if key not in CONSTANTS:
                    raise
        if key in CONSTANTS:
            return CONSTANTS[key]
        raise ExpressionError(f"unknown identifier {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, variables, resolver),
                                      _eval_node(node.right, variables, resolver))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval_node(node.operand, variables, resolver))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = FUNCTIONS.get(node.func.id.lower())
        if fn is None:
            raise ExpressionError(f"unknown function {node.func.id!r}")
        return float(fn(*[_eval_node(a, variables, resolver) for a in node.args]))
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and resolver is not None:
        # MAD8 NAME[ATTR] element-attribute reference
        attr = node.slice.id if isinstance(node.slice, ast.Name) else None
        if attr is not None:
            return float(resolver(f"{node.value.id}[{attr}]"))
    raise ExpressionError(f"unsupported syntax in expression: {ast.dump(node)[:60]}")


_RPN_BIN = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
            "^": operator.pow, "pow": operator.pow, "atan2": math.atan2, "max2": max, "min2": min}
_RPN_UN = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
           "asin": math.asin, "acos": math.acos, "atan": math.atan, "exp": math.exp,
           "ln": math.log, "log": math.log10, "chs": operator.neg, "abs": abs, "sqr": lambda x: x * x,
           "dsin": lambda x: math.sin(math.radians(x)), "dcos": lambda x: math.cos(math.radians(x)),
           "dtan": lambda x: math.tan(math.radians(x)), "int": float}


def evaluate_rpn(text: str, variables: Mapping[str, float] | None = None) -> float:
    """Evaluate an Elegant-style RPN expression (``"2 3 * pi /"``)."""
    variables = variables or {}
    stack: list[float] = []
    for tok in text.replace(",", " ").split():
        low = tok.lower()
        if low in _RPN_BIN:
            if len(stack) < 2:
                raise ExpressionError(f"RPN stack underflow at {tok!r} in {text!r}")
            b, a = stack.pop(), stack.pop()
            stack.append(float(_RPN_BIN[low](a, b)))
        elif low in _RPN_UN:
            if not stack:
                raise ExpressionError(f"RPN stack underflow at {tok!r} in {text!r}")
            stack.append(float(_RPN_UN[low](stack.pop())))
        elif low in CONSTANTS:
            stack.append(CONSTANTS[low])
        elif tok in variables or low in variables:
            stack.append(float(variables.get(tok, variables.get(low))))
        else:
            try:
                stack.append(float(tok))
            except ValueError:
                raise ExpressionError(f"unknown RPN token {tok!r} in {text!r}") from None
    if len(stack) != 1:
        raise ExpressionError(f"RPN expression {text!r} leaves {len(stack)} values on the stack")
    return stack[0]


class LazyResolver:
    """Resolve named parameters lazily with memoisation and cycle detection
    (port of HELIX ``mad8_parser._Mad8File.resolve``)."""

    def __init__(self, definitions: Mapping[str, str], dialect: str = "infix",
                 attr_resolver: Resolver | None = None):
        self.definitions = {k.lower(): v for k, v in definitions.items()}
        self.dialect = dialect
        self.attr_resolver = attr_resolver
        self._cache: dict[str, float] = {}
        self._resolving: set[str] = set()

    def __call__(self, name: str) -> float:
        key = name.lower()
        if "[" in key and self.attr_resolver is not None:
            return float(self.attr_resolver(name))
        if key in self._cache:
            return self._cache[key]
        if key not in self.definitions:
            if key in CONSTANTS:
                return CONSTANTS[key]
            raise ExpressionError(f"unknown identifier {name!r}")
        if key in self._resolving:
            raise ExpressionError(f"circular definition of {name!r}")
        self._resolving.add(key)
        try:
            text = self.definitions[key]
            v = evaluate_rpn(text, {}) if self.dialect == "rpn" else evaluate(text, {}, self)
        finally:
            self._resolving.discard(key)
        self._cache[key] = v
        return v
