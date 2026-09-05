"""The ``lattix: reference …`` comment tag that carries the reference particle through formats
that have no beam definition of their own (TraceWin ``.dat``, Elegant ``.lte``).

Grammar (one line, any comment leader)::

    <leader> lattix: reference species="<name>" mass_eV=<m> charge=<q> kinetic_energy_eV=<T> [rf_frequency_Hz=<f>]

Explicit reader options always win over the tag; a tag-derived reference is recorded as
EQUIVALENT ``REFERENCE_FROM_TAG``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lattix.ir.reference import Species

_TAG = re.compile(r"^\s*[;!#]\s*lattix:\s*reference\s+(?P<body>.*)$")
_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')


@dataclass(frozen=True)
class ReferenceTag:
    species: Species
    kinetic_energy_eV: float
    rf_frequency_Hz: float | None = None


def format_reference_tag(ref, leader: str) -> str:
    f = ref.rf_frequency_Hz
    return (f'{leader} lattix: reference species="{ref.species.name}" mass_eV={ref.species.mass_eV:.12g} '
            f'charge={ref.species.charge} kinetic_energy_eV={ref.kinetic_energy_eV:.12g}'
            + (f" rf_frequency_Hz={f:.12g}" if f else ""))


def parse_reference_tag(text: str) -> ReferenceTag | None:
    for line in text.splitlines():
        m = _TAG.match(line)
        if not m:
            continue
        kv = {k: (q if v.startswith('"') else v) for k, v, q in _KV.findall(m.group("body"))}
        try:
            sp = Species(name=kv.get("species", "proton"), mass_eV=float(kv["mass_eV"]), charge=int(kv["charge"]))
            f = float(kv["rf_frequency_Hz"]) if "rf_frequency_Hz" in kv else None
            return ReferenceTag(sp, float(kv["kinetic_energy_eV"]), f)
        except (KeyError, ValueError):
            return None
    return None
