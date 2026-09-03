# ruff: noqa: F401
"""Intermediate representation."""
from lattix.ir.elements import (
    ALL_KINDS,
    ELEMENT_KINDS,
    RF_KINDS,
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    NCells,
    Octupole,
    Patch,
    Provenance,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    SolenoidP,
    Superposition,
    Taylor,
)
from lattix.ir.expr import Expression, ExpressionError, LazyResolver, evaluate, evaluate_rpn
from lattix.ir.lattice import Lattice, Line, LineItem, Placed, Variable, survey
from lattix.ir.reference import SPECIES, ReferenceParticle, Species, species
from lattix.ir.walk import propagate

__all__ = [n for n in dir() if not n.startswith("_")]
