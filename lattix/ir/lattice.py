"""Lattice container: element definitions, nested lines, flatten() and survey()."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lattix.ir.elements import Element, element_from_dict, element_to_dict
from lattix.ir.expr import Expression
from lattix.ir.reference import ReferenceParticle


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str                 # element or line name
    repeat: int = 1
    reverse: bool = False


class Line(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    items: list[LineItem] = Field(default_factory=list)


class Variable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    expression: Expression | None = None


class Placed(BaseModel):
    """One occurrence of an element in the expanded line."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    element: Element
    index: int
    s_in: float
    s_out: float
    reversed: bool = False
    path: tuple[str, ...] = ()
    ref_in: ReferenceParticle | None = None
    ref_out: ReferenceParticle | None = None

    @property
    def name(self) -> str:
        return self.element.name

    @property
    def length(self) -> float:
        return self.element.length


class Lattice(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str = "lattice"
    elements: dict[str, Element] = Field(default_factory=dict)
    lines: dict[str, Line] = Field(default_factory=dict)
    use: str | None = None
    reference: ReferenceParticle
    variables: dict[str, Variable] = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_sequence(cls, name: str, elements: Iterable[Element], reference: ReferenceParticle,
                      **kw) -> Lattice:
        """A flat lattice: one line referencing each element once, in order.
        Duplicate names get ``_2``, ``_3`` … suffixes."""
        lat = cls(name=name, reference=reference, **kw)
        line = Line(name=name)
        for e in elements:
            nm = lat.add_element(e)
            line.items.append(LineItem(ref=nm))
        lat.lines[name] = line
        lat.use = name
        return lat

    def add_element(self, e: Element) -> str:
        """Register an element definition; returns the (possibly uniquified) name."""
        nm = e.name
        if nm in self.elements:
            if self.elements[nm] is e:
                return nm
            k = 2
            while f"{nm}_{k}" in self.elements:
                k += 1
            nm = f"{nm}_{k}"
            e.name = nm
        self.elements[nm] = e
        return nm

    # -- expansion ----------------------------------------------------------
    def restore_rf_focusing_marks(self) -> None:
        """Re-attach ``meta['rf_focusing_of']`` to thin lenses written by
        :func:`lattix.formats.base.with_rf_focusing` (name ``<cavity>_rfdefocus``) after a read."""
        lower = {n.lower(): n for n in self.elements}          # Elegant/Bmad decks change the case
        for name, e in self.elements.items():
            if e.kind == "Taylor" and name.lower().endswith("_rfdefocus") and "rf_focusing_of" not in e.meta:
                cav = lower.get(name[: -len("_rfdefocus")].lower())
                if cav is not None and self.elements[cav].kind == "RFCavity":
                    e.meta["rf_focusing_of"] = cav

    def flatten(self, use: str | None = None) -> list[Placed]:
        """Expand the root line (repeat/reverse honoured) into placed elements with s."""
        root = use or self.use
        if root is None:
            raise ValueError("lattice has no root line (use)")
        seq: list[tuple[Element, bool, tuple[str, ...]]] = []
        self._expand(root, False, (), seq, depth=0)
        out: list[Placed] = []
        s = 0.0
        for i, (e, rev, path) in enumerate(seq):
            out.append(Placed(element=e, index=i, s_in=s, s_out=s + e.length, reversed=rev, path=path))
            s += e.length
        return out

    def _expand(self, name: str, reverse: bool, path: tuple[str, ...], out: list, depth: int) -> None:
        if depth > 64:
            raise ValueError(f"line nesting deeper than 64 at {name!r} (cycle?)")
        # an element and a line may share a name (from_sequence("s", [Drift("s")])): a line
        # referring to its own name means the element, never itself
        if name in self.lines and not (path and path[-1] == name and name in self.elements):
            items = self.lines[name].items
            seq = list(reversed(items)) if reverse else items
            for it in seq:
                for _ in range(it.repeat):
                    self._expand(it.ref, reverse ^ it.reverse, path + (name,), out, depth + 1)
        elif name in self.elements:
            out.append((self.elements[name], reverse, path))
        else:
            raise KeyError(f"line {path} references unknown element/line {name!r}")

    @property
    def total_length(self) -> float:
        return sum(p.length for p in self.flatten())

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        d = self.model_dump(mode="json", exclude={"elements"})
        d["elements"] = {k: element_to_dict(v) for k, v in self.elements.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Lattice:
        d = dict(d)
        els = {k: element_from_dict(v) for k, v in d.pop("elements", {}).items()}
        lat = cls.model_validate(d)
        lat.elements = els
        return lat


# ---------------------------------------------------------------------------
def survey(placed: list[Placed]) -> np.ndarray:
    """Floor coordinates at element exits, MAD-X convention: returns (N, 4) rows of
    (X, Y, Z, theta).  The reference orbit starts at the origin pointing along +Z;
    a positive horizontal bend angle rotates the direction toward −X (theta decreases),
    a vertical bend (tilt_ref = ±π/2) toward ±Y.  Straight elements advance along the
    local s axis; patches apply their offsets in the local frame.  theta is the MAD-X
    azimuth of the exit frame, kept continuous along the line; the full frames come from
    :func:`lattix.ir.frames.frame_survey`."""
    from lattix.ir.frames import frame_survey  # frames builds on Placed: imported lazily

    out = np.empty((len(placed), 4))
    for f in frame_survey(placed, apply_shift=False):
        out[f.index] = (f.exit.V[0], f.exit.V[1], f.exit.V[2], f.theta_out)
    return out
