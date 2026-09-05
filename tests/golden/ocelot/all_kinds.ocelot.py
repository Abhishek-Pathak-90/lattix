# Ocelot lattice written by lattix 0.1.0
# lattix: reference species="electron" mass_eV=510998.95 charge=-1 kinetic_energy_eV=2100000 rf_frequency_Hz=162500000
# lattix: lattice name=all_kinds use=all_kinds
import numpy as np
from ocelot import *

tws0 = Twiss()
tws0.E = 0.00261099895  # total energy [GeV] of the reference

d1 = Drift(l=0.5, eid='d1')  # lattix: name=d1 kind=Drift
q1 = Quadrupole(l=0.2, k1=-140.499894556575, eid='q1')  # lattix: name=q1 kind=Quadrupole
q1.dx = 0.001
s1 = Sextupole(l=0.1, k2=-351.249736391438, eid='s1')  # lattix: name=s1 kind=Sextupole
o1 = Octupole(l=0.1, k3=-468.33298185525, eid='o1')  # lattix: name=o1 kind=Octupole
m1 = Multipole(kn=[-1.17083245463813, -2.34166490927625], eid='m1')  # lattix: name=m1 kind=Multipole
b1 = SBend(l=1.0, angle=0.1, e1=0.05, e2=0.05, gap=0.06, fint=0.45, eid='b1')  # lattix: name=b1 kind=Bend
sol1 = Solenoid(l=0.3, k=-29.2708113659531, eid='sol1')  # lattix: name=sol1 kind=Solenoid
d2 = Drift(l=0.296243815243, eid='d2')  # lattix: name=d2 kind=Drift
c1 = Cavity(l=0.007512369515, v=8e-05, phi=85.0, freq=162500000.0, eid='c1')  # lattix: name=c1 kind=RFCavity L=0 pad=0.003756184757
d3 = Drift(l=0.296243815242, eid='d3')  # lattix: name=d3 kind=Drift
c2 = Cavity(l=0.2, v=0.001, phi=30.0, freq=162500000.0, eid='c2')  # lattix: name=c2 kind=RFCavity tw=1
k1 = Hcor(l=0.1, angle=0.001, eid='k1')  # lattix: name=k1 kind=Kicker
k1_v = Vcor(l=0.0, angle=-0.002, eid='k1_v')  # lattix: name=k1 kind=Kicker role=vkick
col1 = Aperture(xmax=0.01, ymax=0.02, type='rect', eid='col1')  # lattix: name=col1 kind=Collimator
col1_body = Drift(l=0.05, eid='col1_body')  # lattix: name=col1 kind=Collimator role=body
mk1 = Marker(eid='mk1')  # lattix: name=mk1 kind=Marker
bpm1 = Monitor(l=0.0, eid='bpm1')  # lattix: name=bpm1 kind=Instrument family=BPM
scr1 = Monitor(l=0.0, eid='scr1')  # lattix: name=scr1 kind=Instrument family=SCREEN
f1 = Marker(eid='f1')  # lattix: name=f1 kind=Foil material=C thick=0.001
t1 = Matrix(l=0.1, r11=1.0, r12=0.1, r22=1.0, r33=1.0, r44=1.0, r55=1.0, r66=1.0, eid='t1')  # lattix: name=t1 kind=Taylor basis=ocelot
p1 = Marker(eid='p1')  # lattix: name=p1 kind=Patch x_offset=0.001
rc1 = Matrix(l=0.0, delta_e=1e-06, r11=1.0, r22=1.0, r33=1.0, r44=1.0, r55=1.0, r66=1.0, eid='rc1')  # lattix: name=rc1 kind=ReferenceChange dE=1000.0
fq1 = Marker(eid='fq1')  # lattix: name=fq1 kind=Freq f=162500000.0
dir1 = Marker(eid='dir1')  # lattix: name=dir1 kind=Directive format=tracewin card=LATTICE args="4 0" role=period_start
d4_aper_in = Aperture(xmax=0.015, ymax=0.015, type='ellipt', eid='d4_aper_in')  # lattix: name=d4 kind=Drift role=aperture_in
d4 = Drift(l=0.2, eid='d4')  # lattix: name=d4 kind=Drift
d4_aper_out = Aperture(xmax=0.015, ymax=0.015, type='ellipt', eid='d4_aper_out')  # lattix: name=d4 kind=Drift role=aperture_out

cell = (
    d1, q1, s1, o1, m1, b1, sol1, d2, c1, d3, c2, k1, k1_v, col1, col1_body, mk1, bpm1, scr1, f1,
    t1, p1, rc1, fq1, dir1, d4_aper_in, d4, d4_aper_out,
)
lattice = MagneticLattice(cell)
