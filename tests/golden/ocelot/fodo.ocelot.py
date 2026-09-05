# Ocelot lattice written by lattix 0.1.0
# lattix: reference species="proton" mass_eV=938272088.16 charge=1 kinetic_energy_eV=799999911.84
# lattix: lattice name=fodo use=fodo
import numpy as np
from ocelot import *

tws0 = Twiss()
tws0.E = 1.738272  # total energy [GeV] of the reference

qf = Quadrupole(l=0.3, k1=0.6, eid='qf')  # lattix: name=qf kind=Quadrupole
drift_0 = Drift(l=0.7, eid='drift_0')  # lattix: name=drift_0 kind=Drift
b1 = SBend(l=1.0, angle=0.1, e1=0.05, e2=0.05, eid='b1')  # lattix: name=b1 kind=Bend
drift_1 = Drift(l=1.0, eid='drift_1')  # lattix: name=drift_1 kind=Drift
qd = Quadrupole(l=0.3, k1=-0.6, eid='qd')  # lattix: name=qd kind=Quadrupole
drift_2 = Drift(l=1.0, eid='drift_2')  # lattix: name=drift_2 kind=Drift
drift_3 = Drift(l=1.3, eid='drift_3')  # lattix: name=drift_3 kind=Drift
m1 = Marker(eid='m1')  # lattix: name=m1 kind=Marker

cell = (
    qf, drift_0, b1, drift_1, qd, drift_2, b1, drift_3, m1,
)
lattice = MagneticLattice(cell)
