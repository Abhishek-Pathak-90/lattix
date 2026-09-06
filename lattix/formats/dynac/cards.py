"""DYNAC deck cards: the type codes (``dynac.F`` V6R16), their parameter-line layout and a tokenizer shared
by the reader, the writer and the oracle.

A deck is a mandatory title line followed by type codes, each on a line of its own and followed by its
parameter lines (free format); lines starting with ``;`` are comments (lattix keeps its ``; lattix:`` tags
there).  Most cards have a fixed number of parameter lines; ``GEBEAM``, ``ETAC``, ``RANDALI``, ``HARM``
and ``SCDYNAC`` depend on their own entries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: every type code DYNAC V6R16 knows (dynac.F:160-172)
TYPE_CODES = (
    "GEBEAM", "INPUT", "RDBEAM", "ETAC", "DRIFT", "QUADRUPO", "SEXTUPO", "QUADSXT", "SOLENO", "SOQUAD",
    "BMAGNET", "CAVMC", "CAVSC", "FIELD", "HARM", "BUNCHER", "RFQCL", "NEWF", "NREF", "SCDYNAC",
    "SCDYNEL", "SCPOS", "TILT", "TILZ", "CHANGREF", "TOF", "REJECT", "ZROT", "ALINER", "ACCEPT",
    "EMIT", "EMITGR", "COMMENT", "WRBEAM", "ENVEL", "CHASE", "RWFIELD", "RANDALI", "TWQA", "EMIPRT",
    "MMODE", "RFQPTQ", "STRIPPER", "STEER", "ZONES", "PROFGR", "SECORD", "RASYN", "FDRIFT", "FSOLE",
    "EGUN", "COMPRES", "STOP", "REFCOG", "FPART", "QUAELEC", "QUAFK", "CAVNUM", "EDFLEC", "EMITL",
    "RFKICK", "FIRORD", "DCBEAM", "T3D",
)
#: the list-directed reads of every card (user guide V6R16 chapter 6): an integer is a read of that many
#: numbers (Fortran continues on the next line when a line runs out), "name" a file name / title line,
#: "rest" the remaining tokens of the current line; the cards not listed take every line up to the next
#: type code (SCDYNAC, RFQCL, FPART, RFKICK)
CARD_READS: dict[str, list] = {
    "INPUT": [3, 2], "RDBEAM": ["name", 1, 2, 2, 2], "DRIFT": [1], "QUADRUPO": [3], "SEXTUPO": [4], "QUADSXT": [5],
    "SOLENO": [3], "SOQUAD": [5], "BMAGNET": [1, 5, 5, 5], "CAVMC": [1, 5], "CAVSC": [16], "FIELD": ["name", 1],
    "BUNCHER": [4], "NEWF": [1], "NREF": [4], "SCDYNEL": [1], "SCPOS": [1], "TILT": [1, 5], "TILZ": [1],
    "CHANGREF": [3], "TOF": [2], "REJECT": [6], "ZROT": [1], "ALINER": [4], "ACCEPT": ["name", 2, 8, "name", 2, 8],
    "EMIT": [], "EMITGR": ["name", 2, 8], "COMMENT": ["name"], "WRBEAM": ["name", "rest"], "ENVEL": ["name", 1, 2, 4],
    "CHASE": [3], "RWFIELD": [], "TWQA": [2], "EMIPRT": [1], "MMODE": [3], "RFQPTQ": ["name", 1, 4], "STRIPPER": [4],
    "STEER": [2], "ZONES": [2, "rest"], "PROFGR": ["name", 2, 4], "SECORD": [], "RASYN": [], "FDRIFT": [3],
    "FSOLE": ["name", 2], "EGUN": ["name", 2], "COMPRES": [1], "STOP": [], "REFCOG": [1], "QUAELEC": [3],
    "QUAFK": [4], "CAVNUM": [1, 5], "EDFLEC": [1, 4], "EMITL": ["name"], "FIRORD": [], "DCBEAM": [1], "T3D": [],
}
#: cards that change the beam line's optics / the coordinates (the oracle dumps after each of them);
#: everything else is beam definition, output or a modifier of the elements that follow
OPTICS_CARDS = frozenset({
    "DRIFT", "QUADRUPO", "SEXTUPO", "QUADSXT", "SOLENO", "SOQUAD", "BMAGNET", "CAVMC", "CAVSC", "BUNCHER",
    "RFQCL", "NREF", "CHANGREF", "ZROT", "ALINER", "STRIPPER", "STEER", "FDRIFT", "FSOLE", "EGUN", "QUAELEC",
    "QUAFK", "CAVNUM", "EDFLEC", "RFQPTQ", "TILT", "NEWF",
})
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")


@dataclass
class Card:
    name: str
    lines: list[str] = field(default_factory=list)          # one string per list-directed read (tokens joined)
    comments: list[str] = field(default_factory=list)       # ';' lines that preceded the card
    lineno: int = 0                                          # 1-based line of the type code

    @property
    def tokens(self) -> list[list[str]]:
        return [ln.split() for ln in self.lines]

    def floats(self, i: int = 0) -> list[float]:
        return [float(t.replace("D", "e").replace("d", "e")) for t in self.lines[i].split() if _NUM.match(t)]

    def text(self) -> str:
        return "\n".join([*self.comments, self.name, *self.lines])


def is_card_line(line: str) -> str | None:
    """The type code when *line* is one (the code alone, case-insensitive), else ``None``."""
    s = line.strip()
    if not s or s.startswith(";"):
        return None
    tok = s.split()
    if len(tok) == 1 and tok[0].upper() in TYPE_CODES:
        return tok[0].upper()
    return None


class _Stream:
    """Tokens of the parameter lines that follow a type code, consumed read by read."""

    def __init__(self, raw: list[str], i: int) -> None:
        self.raw, self.i = raw, i
        self.buf: list[str] = []

    def _fill(self) -> bool:
        while self.i < len(self.raw):
            s = self.raw[self.i].strip()
            if not s or s.startswith(";"):
                self.i += 1
                continue
            if is_card_line(self.raw[self.i]) is not None:
                return False
            self.buf = s.split()
            self.i += 1
            return True
        return False

    def line(self) -> str | None:
        """A whole physical line (a file name or a title)."""
        if self.buf:
            out, self.buf = " ".join(self.buf), []
            return out
        if not self._fill():
            return None
        out, self.buf = " ".join(self.buf), []
        return out

    def rest(self) -> str:
        out, self.buf = " ".join(self.buf), []
        return out

    def numbers(self, n: int) -> str | None:
        out: list[str] = []
        while len(out) < n:
            if not self.buf and not self._fill():
                return " ".join(out) if out else None
            take = min(n - len(out), len(self.buf))
            out += self.buf[:take]
            self.buf = self.buf[take:]
        return " ".join(out)


def _reads_of(name: str, first: str | None, stream: _Stream) -> list:
    """The read list of a card whose layout depends on its first entries."""
    if name == "GEBEAM":
        law_itwiss = stream.numbers(2) or "0 0"
        itwiss = int(float(law_itwiss.split()[1])) if len(law_itwiss.split()) > 1 else 0
        return [law_itwiss, 2, 6] + ([3, 3, 3] if itwiss == 1 else [6])
    if name == "ETAC":
        n_line = stream.numbers(1) or "0"
        n = int(float(n_line))
        return [n_line] + ([3] * n if n > 0 else ["name"])
    if name == "RANDALI":
        flag = stream.numbers(1) or "0"
        return [flag] + ([4] if int(float(flag)) == 1 else [])
    if name == "HARM":
        head = stream.numbers(4) or ""
        nh = stream.numbers(1) or "0"
        return [head, nh, int(float(nh))]
    return []


def parse_deck(text: str) -> tuple[str, list[Card]]:
    """``(title, cards)`` of a DYNAC deck.  Every card's parameter entries are gathered read by read the way
    DYNAC's list-directed input does (a read short of numbers continues on the next line); cards with an
    unknown layout take every line up to the next type code."""
    raw = text.splitlines()
    # DYNAC's first line is a mandatory title; a converter's output (tw2dyn) may start with a type code
    # straight away — then there is no title and the line is a card
    title = raw[0].rstrip() if raw and is_card_line(raw[0]) is None else ""
    cards: list[Card] = []
    pending: list[str] = []
    i = 1 if title or not raw else 0
    while i < len(raw):
        line = raw[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith(";"):
            pending.append(s)
            i += 1
            continue
        name = is_card_line(line)
        if name is None:                       # stray text: kept as a comment
            pending.append("; " + s)
            i += 1
            continue
        card = Card(name=name, comments=pending, lineno=i + 1)
        pending = []
        i += 1
        stream = _Stream(raw, i)
        if name in CARD_READS:
            reads = CARD_READS[name]
        elif name in ("GEBEAM", "ETAC", "RANDALI", "HARM"):
            reads = _reads_of(name, None, stream)
        else:
            reads = None
        if reads is None:                      # unknown layout: whole lines up to the next type code
            while True:
                ln = stream.line()
                if ln is None:
                    break
                card.lines.append(ln)
        else:
            for r in reads:
                if isinstance(r, str) and r not in ("name", "rest"):
                    card.lines.append(r)       # an entry already read by _reads_of
                    continue
                if r == "name":
                    v = stream.line()
                elif r == "rest":
                    v = stream.rest()
                else:
                    v = stream.numbers(int(r))
                if v is None:
                    break
                card.lines.append(v)
        i = stream.i
        cards.append(card)
    if pending:
        cards.append(Card(name="STOP", comments=pending, lineno=len(raw)))
    return title, cards


def fnum(v: float, digits: int = 12) -> str:
    """A free-format number DYNAC reads back exactly enough (12 significant digits)."""
    if v == 0:
        return "0."
    s = f"{float(v):.{digits}g}"
    if "e" in s:
        mant, exp = s.split("e")
        if "." not in mant:
            mant += "."
        return f"{mant}e{int(exp)}"
    return s if "." in s else s + "."
