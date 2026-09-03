"""Never-silent fidelity ledger (PLAN §4.4)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FidelityClass(StrEnum):
    EXACT = "EXACT"
    EQUIVALENT = "EQUIVALENT"
    LOSSY = "LOSSY"
    DROPPED = "DROPPED"


class FidelityEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element: str | None
    kind: str | None
    cls: FidelityClass
    code: str
    message: str = ""
    details: dict = Field(default_factory=dict)
    line: int | None = None


class TranslationError(RuntimeError):
    def __init__(self, entry: FidelityEntry):
        super().__init__(f"{entry.cls}:{entry.code} on {entry.kind} {entry.element!r}: {entry.message}")
        self.entry = entry


class FidelityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_format: str | None = None
    target_format: str | None = None
    source_file: str | None = None
    target_file: str | None = None
    entries: list[FidelityEntry] = Field(default_factory=list)
    allowlist: set[str] = Field(default_factory=set)

    # -- recording ----------------------------------------------------------
    def add(self, cls: FidelityClass | str, code: str, message: str = "", *, element: str | None = None,
            kind: str | None = None, line: int | None = None, **details) -> FidelityEntry:
        e = FidelityEntry(element=element, kind=kind, cls=FidelityClass(cls), code=code,
                          message=message, details=details, line=line)
        self.entries.append(e)
        return e

    def exact(self, element, kind, code="OK", message=""):
        return self.add(FidelityClass.EXACT, code, message, element=element, kind=kind)

    def equivalent(self, code, message="", *, element=None, kind=None, **d):
        return self.add(FidelityClass.EQUIVALENT, code, message, element=element, kind=kind, **d)

    def lossy(self, code, message="", *, element=None, kind=None, line=None, **d):
        return self.add(FidelityClass.LOSSY, code, message, element=element, kind=kind, line=line, **d)

    def dropped(self, code, message="", *, element=None, kind=None, line=None, **d):
        return self.add(FidelityClass.DROPPED, code, message, element=element, kind=kind, line=line, **d)

    def extend(self, other: FidelityReport) -> None:
        self.entries.extend(other.entries)

    # -- queries --------------------------------------------------------------
    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(e.cls.value for e in self.entries))

    def problems(self) -> list[FidelityEntry]:
        return [e for e in self.entries if e.cls in (FidelityClass.LOSSY, FidelityClass.DROPPED)
                and e.code not in self.allowlist]

    @property
    def ok(self) -> bool:
        return not self.problems()

    def raise_if(self, strict: bool) -> None:
        if strict:
            p = self.problems()
            if p:
                raise TranslationError(p[0])

    def codes(self) -> dict[str, int]:
        return dict(Counter(e.code for e in self.entries if e.cls is not FidelityClass.EXACT))

    # -- output ---------------------------------------------------------------
    def summary(self) -> str:
        c = self.counts
        parts = [f"{k}={c.get(k, 0)}" for k in ("EXACT", "EQUIVALENT", "LOSSY", "DROPPED")]
        head = f"fidelity {self.source_format or '?'}→{self.target_format or '?'}: " + " ".join(parts)
        lines = [head]
        for code, n in sorted(self.codes().items(), key=lambda kv: -kv[1]):
            ex = next(e for e in self.entries if e.code == code)
            lines.append(f"  {ex.cls:<10} {code:<28} ×{n:<4} {ex.message}")
        return "\n".join(lines)

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.model_dump(mode="json"), indent=1)
        if path is not None:
            Path(path).write_text(s + "\n")
        return s

    def print_summary(self, file=sys.stderr) -> None:
        print(self.summary(), file=file)
