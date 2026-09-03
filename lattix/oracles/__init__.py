"""Engine adapters ("oracles") that run real accelerator codes on deck files.

    from lattix.oracles import get_oracle, available_oracles
    res = get_oracle("madx").run(Path("fodo.madx"), beam=BeamSpec(...))
"""
from lattix.oracles.base import (
    SPECIES,
    Basis,
    BeamSpec,
    Oracle,
    OracleResult,
    Probe,
    available_oracles,
    get_oracle,
    guess_format,
    register,
)

__all__ = [
    "SPECIES", "Basis", "BeamSpec", "Oracle", "OracleResult", "Probe",
    "available_oracles", "get_oracle", "guess_format", "register",
]
